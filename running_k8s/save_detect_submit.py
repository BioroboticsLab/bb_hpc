#!/usr/bin/env python3
import argparse, json, os, subprocess
from pathlib import Path
from datetime import datetime

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_save_detect
import re

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dates", nargs="+", required=True,
                   help="YYYYMMDD (one or many). Example: 20250901 20250902")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists & Job spec, but do not kubectl apply.")
    return p.parse_args()

def write_filelist(dir_host: Path, idx: int, job_args_list):
    """
    Write JSONL where datetimes are ISO strings and paths are already HPC-visible.
    """
    dir_host.mkdir(parents=True, exist_ok=True)
    f = dir_host / f"savedetect_{idx:05d}.txt"
    with open(f, "w") as out:
        for item in job_args_list:
            safe_item = dict(item)
            # ensure datetime -> ISO8601
            for k in ("from_dt", "to_dt"):
                v = safe_item.get(k)
                if isinstance(v, datetime):
                    safe_item[k] = v.isoformat()
            out.write(json.dumps(safe_item) + "\n")
    return f

def make_indexed_job(job_name: str, completions: int, parallelism: int,
                     filelist_dir_pod: str, runner_path: str):
    k = settings.k8s
    env_list = [{"name": k_, "value": str(v_)} for k_, v_ in k.get("env", {}).items()]
    # Prefer a CPU-only resources block for save-detect if provided
    resources = k.get("resources_save_detect", k["resources"])

    bash = f"""\
set -euo pipefail

# Activate conda env
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)"
fi
conda activate beesbook || true

# Pick filelist for this completion index
idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{filelist_dir_pod}/savedetect_${{idx5}}.txt"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

# Run N workers in parallel (CPU-only path)
WORKERS="${{WORKERS_PER_POD:-${{WORKERS:-2}}}}"
echo "WORKERS_PER_POD=${{WORKERS}}"

tmpd=$(mktemp -d)
split -n "l/${{WORKERS}}" -d -a 2 "$fl" "${{tmpd}}/shard_"

pids=()
for shard in "${{tmpd}}"/shard_*; do
  echo "Starting worker on shard: $shard"
  python -u {runner_path} "$shard" &
  pids+=($!)
done

rc=0
for p in "${{pids[@]}}"; do
  if ! wait "$p"; then
    rc=1
  fi
done
exit $rc
"""

    pod_spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": k["image_pull_secret"]}],
        "containers": [{
            "name": "save-detect",
            "image": k["image"],
            "command": ["bash", "-lc", bash],
            "resources": resources,
            "env": env_list,
            "volumeMounts": k["volume_mounts"],
            "tty": True,
            "stdin": True,
        }],
        "volumes": k["volumes"],
    }
    if k.get("affinity"):
        pod_spec["affinity"] = k["affinity"]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": k["namespace"],
            "labels": {"app": "save-detect", "bbhpc": "save-detect-indexed"},
        },
        "spec": {
            "backoffLimit": int(k["job"].get("backoff_limit", 1)),
            "completions": int(completions),
            "parallelism": int(min(max(1, parallelism), max(1, completions))),
            "completionMode": "Indexed",
            "template": {
                "metadata": {
                    "labels": {
                        "app": "save-detect",
                        "bbhpc": "save-detect-indexed",
                        "job-name": job_name
                    }
                },
                "spec": pod_spec
            }
        }
    }

def _sanitize_job_name(name: str) -> str:
    """
    Convert arbitrary strings to a DNS-1123 compliant name for K8s metadata.name.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "job"

def _map_job_paths_to_hpc(item: dict) -> dict:
    """
    Convert job dict paths to the pod-visible HPC roots.
    Read caches from LOCAL, but write HPC paths for the pod.
    """
    out = dict(item)
    # Force repo_path/save_path to HPC-visible roots
    out["repo_path"] = str(Path(settings.pipeline_root_hpc))
    out["save_path"] = str(Path(settings.resultdir_hpc) / "data_alldetections")
    return out

def main():
    args = parse_args()

    # Use LOCAL roots to read fileinfo caches,
    # but emit HPC-visible paths into the job files.
    resultdir_local   = Path(settings.resultdir_local)
    pipeline_root_local = Path(settings.pipeline_root_local)

    # Where to write filelists/spec (host) and where the Pod will read them (HPC path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filelist_dir_host = Path(settings.jobdir_local) / "k8s" / "save_detect" / stamp
    filelist_dir_pod  = Path(settings.jobdir_hpc)   / "k8s" / "save_detect" / stamp

    # Generate work using LOCAL dirs (for cache files)
    s = settings.save_detect_settings
    chunk_size = int(s.get("chunk_size", 50))
    maxjobs    = s.get("maxjobs", None)

    batches = list(generate_jobs_save_detect(
        RESULTDIR=str(resultdir_local),
        PIPELINE_ROOT=str(pipeline_root_local),
        chunk_size=chunk_size,
        maxjobs=maxjobs,
        datestring=args.dates,
    ))
    if not batches:
        print("No save-detect work to submit.")
        return

    # Map each job's paths to HPC view before writing the JSONL
    filelists = []
    for idx, batch in enumerate(batches):
        jl = batch.get("job_args_list", [])
        if not jl:
            continue
        mapped = [_map_job_paths_to_hpc(x) for x in jl]
        f = write_filelist(filelist_dir_host, idx, mapped)
        filelists.append(f)

    jobname = s.get("jobname", "save_detect")
    full_job_name = _sanitize_job_name(f"{jobname}-{stamp}")
    par = int(settings.k8s["job"].get("parallelism", 1))

    # Runner inside the pod
    runner_path = settings.k8s.get(
        "save_detect_runner_path",
        "/abyss/home/jacob-davidson/cascbstorage/jacob/bb_hpc/running_k8s/run_save_detect.py",
    )

    job_dict = make_indexed_job(
        job_name        = full_job_name,
        completions     = len(filelists),
        parallelism     = par,
        filelist_dir_pod= str(filelist_dir_pod),
        runner_path     = runner_path,
    )

    spec_path = filelist_dir_host / f"{full_job_name}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(job_dict, f, indent=2)

    print(f"Wrote {spec_path}  (completions={len(filelists)}, parallelism={par})")

    if args.dry_run:
        print("Dry run: not applying Job.")
        return

    subprocess.run(["kubectl", "apply", "-f", str(spec_path),
                    "-n", settings.k8s["namespace"]], check=True)
    print(f"Submitted Job {full_job_name} in ns {settings.k8s['namespace']}")
    print(f"Filelists dir (host): {filelist_dir_host}")
    print(f"Filelists dir (pod) : {filelist_dir_pod}")

if __name__ == "__main__":
    main()
