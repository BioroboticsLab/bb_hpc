#!/usr/bin/env python3
import argparse, json, os, time, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_detect


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser()
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists & Job spec, but do not kubectl apply.")
    p.add_argument("--use-fileinfo", action="store_true",
                   help="Use fileinfo to skip already processed videos and speed up job generation.")
    p.add_argument("--check-read-bbb", action="store_true",
                   help="Read .bbb files to validate them (slower but skips zero/invalid outputs).")
    return p.parse_args()


def map_local_to_hpc(paths, local_root: Path, hpc_root: Path):
    """Translate absolute local paths to the equivalent HPC paths (for inside the Pod)."""
    out = []
    for p in paths:
        p = Path(p)
        try:
            rel = p.relative_to(local_root)
            out.append(str(hpc_root / rel))
        except ValueError:
            # Not under local_root; leave as-is (rare)
            out.append(str(p))
    return out


def write_filelist(dir_host: Path, idx: int, videos_hpc):
    dir_host.mkdir(parents=True, exist_ok=True)
    f = dir_host / f"videos_{idx:05d}.txt"
    with open(f, "w") as out:
        out.write("\n".join(videos_hpc))
    return f


def make_indexed_job(job_name: str, completions: int, parallelism: int,
                     filelist_dir_pod: str, runner_path: str):
    """
    Build a single Job with Indexed completions.
    Each Pod computes: FILELIST=.../videos_${JOB_COMPLETION_INDEX}.txt
    and then runs N shard workers in parallel on the same GPU.
    """
    k = settings.k8s

    # Env passed via Kubernetes (includes your PYTHONPATH, TF_* flags, etc.)
    env_list = [{"name": k_, "value": str(v_)} for k_, v_ in k.get("env", {}).items()]

    # If you want a default workers-per-GPU at the job level, inject it if not already provided
    if not any(e["name"] == "WORKERS_PER_GPU" for e in env_list):
        wpg = str(k.get("job", {}).get("workers_per_gpu", 2))
        env_list.append({"name": "WORKERS_PER_GPU", "value": wpg})

    # Bash payload:
    # - use f-string for Python interpolation
    # - double the braces around any Bash ${...} usage so Python doesn't try to format them
    bash = f"""\
set -euo pipefail

# Activate conda env
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)" || true
fi
conda activate beesbook || true

# Ensure conda libs (e.g., libstdc++, cuDNN) are visible
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"

# Pick filelist for this completion index
idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{filelist_dir_pod}/videos_${{idx5}}.txt"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

# ---- Run N workers in parallel on this one GPU ----
WORKERS="${{WORKERS_PER_GPU:-2}}"
echo "WORKERS_PER_GPU=${{WORKERS}}"

tmpd="$(mktemp -d)"
# Split the filelist into N shards (line-balanced)
split -n "l/${{WORKERS}}" -d -a 2 "$fl" "${{tmpd}}/shard_"

pids=()
for shard in "${{tmpd}}"/shard_*; do
  echo "Starting worker on shard: $shard"
  python -u {runner_path} "$shard" &
  pids+=($!)
done

# Wait & propagate failure if any
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
            "name": "detect",
            "image": k["image"],
            "command": ["bash", "-lc", bash],
            "resources": k["resources"],
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
            "labels": {
                "app": "detect",
                "bbhpc": "detect-indexed",
            },
        },
        "spec": {
            "backoffLimit": int(k["job"].get("backoff_limit", 1)),
            "completions": int(completions),
            "parallelism": int(min(max(1, parallelism), max(1, completions))),
            "completionMode": "Indexed",
            "template": {
                "metadata": {
                    "labels": {
                        "app": "detect",
                        "bbhpc": "detect-indexed",
                        "job-name": job_name,
                    }
                },
                "spec": pod_spec
            }
        }
    }
    return job


def main():
    args = parse_args()

    # ---------- derive paths ----------
    # Use LOCAL roots to enumerate work & check BBB outputs;
    # convert to HPC paths for filelists so the Pod sees the right paths.
    video_local = Path(settings.videodir_local)
    video_hpc   = Path(settings.videodir_hpc)
    repo_local  = Path(settings.pipeline_root_local)

    # Resolve result_local similarly to Docker detect_submit to allow skipping already processed videos
    # This speeds up job generation by avoiding re-processing videos with existing results.
    result_local = None
    if hasattr(settings, "resultdir_local") and settings.resultdir_local:
        result_local = Path(settings.resultdir_local)
    else:
        # Try to infer from pipeline_root_local if possible
        inferred = repo_local.parent / "results"
        if inferred.exists():
            result_local = inferred

    # Where to write filelists on the submission machine (LOCAL) and
    # where the Pod will read the same directory (HPC path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filelist_dir_host = Path(settings.jobdir_local) / "k8s" / "detect" / stamp
    filelist_dir_pod  = Path(settings.jobdir_hpc)   / "k8s" / "detect" / stamp

    # ---------- build chunks ----------
    s = settings.detect_settings
    chunks = list(generate_jobs_detect(
        video_root_dir   = str(video_local),
        repo_output_path = str(repo_local),
        slurmdir         = None,
        chunk_size       = s.get("chunk_size", 4),
        maxjobs          = s.get("maxjobs", None),
        datestring       = args.dates,
        verbose          = False,
        RESULTDIR        = result_local,
        use_fileinfo     = bool(args.use_fileinfo),
        check_read_bbb   = bool(args.check_read_bbb),
    ))
    if not chunks:
        print("No work to submit.")
        return

    # ---------- write filelists (HPC-style paths) ----------
    for idx, ch in enumerate(chunks):
        vids_hpc = map_local_to_hpc(ch["video_paths"], video_local, video_hpc)
        write_filelist(filelist_dir_host, idx, vids_hpc)

    # ---------- single Indexed Job ----------
    jobname       = s.get("jobname", "detect")
    full_job_name = f"{jobname}-{stamp}"
    par           = int(settings.k8s["job"].get("parallelism", 1))
    job_dict = make_indexed_job(
        job_name        = full_job_name,
        completions     = len(chunks),
        parallelism     = par,
        filelist_dir_pod= str(filelist_dir_pod),
        runner_path     = settings.k8s.get("runner_path", "/workspace/run_videos.py"),
    )

    # write one spec file (JSON is fine)
    spec_path = filelist_dir_host / f"{full_job_name}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(job_dict, f, indent=2)
    print(f"Wrote {spec_path}  (completions={len(chunks)}, parallelism={par})")

    if args.dry_run:
        print("Dry run: not applying Job.")
        return

    # ---------- apply ----------
    subprocess.run(["kubectl", "apply", "-f", str(spec_path),
                    "-n", settings.k8s["namespace"]], check=True)
    print(f"Submitted Job {full_job_name} in ns {settings.k8s['namespace']}")
    print(f"Filelists dir (host): {filelist_dir_host}")
    print(f"Filelists dir (pod) : {filelist_dir_pod}")


if __name__ == "__main__":
    main()
