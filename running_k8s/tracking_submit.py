#!/usr/bin/env python3
import argparse, json, os, time, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_tracking


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit tracking shards to Kubernetes (Indexed Job)")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists & Job spec, but do not kubectl apply.")
    return p.parse_args()

def _map_path_local_to_hpc(p: str, pairs: list[tuple[Path, Path]]) -> str:
    """Translate an absolute LOCAL path to corresponding HPC path using first matching root."""
    P = Path(p)
    for lhs, rhs in pairs:
        try:
            rel = P.relative_to(lhs)
            return str(rhs / rel)
        except ValueError:
            continue
    return str(P)

def remap_job_paths_for_pod(job: dict, pairs: list[tuple[Path, Path]]):
    """Map repo_path/save_path/temp_path from LOCAL roots to HPC roots for inside the Pod."""
    j = dict(job)
    for key in ("repo_path", "save_path", "temp_path"):
        if key in j and j[key]:
            j[key] = _map_path_local_to_hpc(j[key], pairs)
    # Ensure datetimes are JSON serializable (isoformat) will be handled on write
    return j

def write_filelist(dir_host: Path, idx: int, job_args_list):
    dir_host.mkdir(parents=True, exist_ok=True)
    f = dir_host / f"tracking_{idx:05d}.txt"
    with open(f, "w") as out:
        for item in job_args_list:
            safe = dict(item)
            for k in ("from_dt", "to_dt"):
                v = safe.get(k)
                if hasattr(v, "isoformat"):
                    safe[k] = v.isoformat()
            out.write(json.dumps(safe) + "\n")
    return f


def make_indexed_job(job_name: str, completions: int, parallelism: int,
                     filelist_dir_pod: str, runner_path: str,
                     env_list: list[dict], resources: dict):
    """
    Build a single Job with Indexed completions.
    Each Pod computes: FILELIST=.../tracking_${JOB_COMPLETION_INDEX}.txt
    and then runs N shard workers in parallel on the same GPU.
    """
    k = settings.k8s

    bash = f"""\
set -euo pipefail

# Activate conda env
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)" || true
fi
conda activate beesbook || true

# Ensure conda libs are visible
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"

# Pick filelist for this completion index
idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{filelist_dir_pod}/tracking_${{idx5}}.txt"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

# Decide workers: GPU mode uses WORKERS_PER_GPU, CPU mode uses WORKERS_PER_POD
if [ "${{GPU_ENABLED:-1}}" = "1" ]; then
  WORKERS="${{WORKERS_PER_GPU:-2}}"
  echo "GPU mode: WORKERS_PER_GPU=${{WORKERS}}"
else
  WORKERS="${{WORKERS_PER_POD:-2}}"
  echo "CPU mode: WORKERS_PER_POD=${{WORKERS}}"
fi

tmpd="$(mktemp -d)"
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
            "name": "tracking",
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

    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": k["namespace"],
            "labels": {"app": "tracking", "bbhpc": "tracking-indexed"},
        },
        "spec": {
            "backoffLimit": int(k["job"].get("backoff_limit", 1)),
            "completions": int(completions),
            "parallelism": int(min(max(1, parallelism), max(1, completions))),
            "completionMode": "Indexed",
            "template": {
                "metadata": {
                    "labels": {"app": "tracking", "bbhpc": "tracking-indexed", "job-name": job_name}
                },
                "spec": pod_spec
            }
        }
    }
    return job


def main():
    args = parse_args()

    # ---------- derive paths ----------
    # Use LOCAL roots to enumerate and map to HPC paths for the Pod.
    resultdir_local = Path(settings.resultdir_local)
    pipeline_local  = Path(settings.pipeline_root_local)
    resultdir_hpc   = Path(settings.resultdir_hpc)
    pipeline_hpc    = Path(settings.pipeline_root_hpc)

    # Where to write filelists on the submission machine (LOCAL) and
    # where the Pod will read the same directory (HPC path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filelist_dir_host = Path(settings.jobdir_local) / "k8s" / "tracking" / stamp
    filelist_dir_pod  = Path(settings.jobdir_hpc)   / "k8s" / "tracking" / stamp

    # ---------- generator knobs ----------
    s = settings.track_settings
    chunk_size     = int(s.get("chunk_size", 1))
    maxjobs        = s.get("maxjobs", None)
    interval_hours = int(s.get("interval_hours", 1))
    temp_dir       = s.get("temp_path", "/tmp/bb_tracking_tmp")

    # Determine GPU/CPU mode and resources
    gpu_enabled = bool(s.get("gpu", True))

    # Base env from Kubernetes settings
    base_env = [{"name": k_, "value": str(v_)} for k_, v_ in settings.k8s.get("env", {}).items()]

    # Ensure worker env knobs are present
    if not any(e["name"] == "WORKERS_PER_GPU" for e in base_env):
        wpg = str(settings.k8s.get("job", {}).get("workers_per_gpu", 2))
        base_env.append({"name": "WORKERS_PER_GPU", "value": wpg})
    if not any(e["name"] == "WORKERS_PER_POD" for e in base_env):
        wpp = str(settings.k8s.get("job", {}).get("workers_per_pod", 2))
        base_env.append({"name": "WORKERS_PER_POD", "value": wpp})

    # Flag to tell the entrypoint which worker knob to use
    base_env.append({"name": "GPU_ENABLED", "value": "1" if gpu_enabled else "0"})

    # In CPU-only mode, explicitly hide GPUs to frameworks
    if not gpu_enabled:
        base_env.append({"name": "CUDA_VISIBLE_DEVICES", "value": ""})
        # Optional (harmless if NVIDIA runtime isn't used):
        base_env.append({"name": "NVIDIA_VISIBLE_DEVICES", "value": "none"})
    
    # Pick resources: allow per-job overrides, otherwise fall back to global k8s resources
    if gpu_enabled:
        resources = settings.k8s.get("resources_tracking_gpu", settings.k8s.get("resources", {}))
    else:
        resources = settings.k8s.get("resources_tracking_cpu", {"requests": {"cpu": "2", "memory": "8Gi"},
                                                               "limits":   {"cpu": "2", "memory": "8Gi"}})

    # Build local→HPC mapping pairs for repo/save/temp/jobdir
    mapping_pairs = [
        (pipeline_local, pipeline_hpc),
        (resultdir_local, resultdir_hpc),
        (Path(settings.jobdir_local), Path(settings.jobdir_hpc)),
    ]

    # ---------- build batches ----------
    batches = list(generate_jobs_tracking(
        RESULTDIR      = str(resultdir_local),
        PIPELINE_ROOT  = str(pipeline_local),
        TEMP_DIR       = str(temp_dir),
        datestring     = args.dates,
        chunk_size     = chunk_size,
        maxjobs        = maxjobs,
        interval_hours = interval_hours,
    ))
    if not batches:
        print("No tracking work to submit.")
        return

    # ---------- write JSONL shards (HPC-mapped paths) ----------
    shards = []
    for idx, batch in enumerate(batches):
        jl = batch.get("job_args_list", [])
        if jl:
            jl_mapped = [remap_job_paths_for_pod(it, mapping_pairs) for it in jl]
            shards.append(write_filelist(filelist_dir_host, idx, jl_mapped))

    jobname       = s.get("jobname", "tracking")
    full_job_name = f"{jobname}-{stamp}"
    par           = int(settings.k8s["job"].get("parallelism", 1))
    runner_path = settings.k8s.get("tracking_runner_path", "/workspace/run_tracking.py")

    job_dict = make_indexed_job(
        job_name         = full_job_name,
        completions      = len(shards),
        parallelism      = par,
        filelist_dir_pod = str(filelist_dir_pod),
        runner_path      = runner_path,
        env_list         = base_env,
        resources        = resources,
    )

    spec_path = filelist_dir_host / f"{full_job_name}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(job_dict, f, indent=2)

    print(f"Wrote {spec_path}  (completions={len(shards)}, parallelism={par})")

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
