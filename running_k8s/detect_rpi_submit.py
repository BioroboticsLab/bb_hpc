#!/usr/bin/env python3
import argparse, json, os, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_rpi_detect


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit RPi detect shards to Kubernetes")
    p.add_argument(
        "--dates",
        nargs="+",
        default=[yesterday, today],
        help="YYYYMMDD strings (UTC). Default: yesterday & today.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Write filelists & Job spec, but do not kubectl apply.",
    )
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
                     filelist_dir_pod: str, runner_path: str, use_clahe: bool):
    """
    Build a single Job with Indexed completions.
    Each Pod computes: FILELIST=.../videos_${JOB_COMPLETION_INDEX}.txt
    and then runs N shard workers in parallel (same pattern as detect submit).
    """
    k = settings.k8s

    # Env passed via Kubernetes (includes your PYTHONPATH, TF_* flags, etc.)
    env_list = [{"name": k_, "value": str(v_)} for k_, v_ in k.get("env", {}).items()]

    # Add CLAHE switch for the runner (runner reads RPI_CLAHE)
    env_list.append({"name": "RPI_CLAHE", "value": "1" if use_clahe else "0"})

    # Inject WORKERS_PER_GPU if not already provided (even if CPU-only in cluster, this controls intra-pod fanout)
    if not any(e["name"] == "WORKERS_PER_GPU" for e in env_list):
        wpg = str(k.get("job", {}).get("workers_per_gpu", 2))
        env_list.append({"name": "WORKERS_PER_GPU", "value": wpg})

    # Bash payload:
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

# Ensure conda libs are visible
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"

# Pick filelist for this completion index
idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{filelist_dir_pod}/videos_${{idx5}}.txt"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

# ---- Run N workers in parallel on this Pod ----
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

    job = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": k["namespace"],
            "labels": {
                "app": "rpi-detect",
                "bbhpc": "rpi-detect-indexed",
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
                        "app": "rpi-detect",
                        "bbhpc": "rpi-detect-indexed",
                        "job-name": job_name,
                    }
                },
                "spec": {
                    "restartPolicy": "Never",
                    "imagePullSecrets": [{"name": k["image_pull_secret"]}],
                    "containers": [{
                        "name": "rpi-detect",
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
            }
        }
    }
    return job


def main():
    args = parse_args()

    # ---------- derive paths ----------
    # Use LOCAL roots to enumerate work; convert to HPC paths for filelists so the Pod sees the right paths.
    video_local = Path(settings.pi_videodir_local)
    video_hpc   = Path(settings.pi_videodir_hpc)

    # Where to write filelists on the submission machine (LOCAL) and
    # where the Pod will read the same directory (HPC path)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filelist_dir_host = Path(settings.jobdir_local) / "k8s" / "rpi" / stamp
    filelist_dir_pod  = Path(settings.jobdir_hpc)   / "k8s" / "rpi" / stamp

    # ---------- build chunks ----------
    s = settings.rpi_detect_settings
    chunk_sz = int(s.get("chunk_size", 150))
    use_clahe = bool(s.get("use_clahe", True))

    chunks = list(generate_jobs_rpi_detect(
        video_root_dir = str(video_local),
        dates          = args.dates,
        chunk_size     = chunk_sz,
        clahe          = use_clahe,
    ))
    if not chunks:
        print("No work to submit.")
        return

    # ---------- write filelists (HPC-style paths) ----------
    for idx, ch in enumerate(chunks):
        vids_hpc = map_local_to_hpc(ch["video_paths"], video_local, video_hpc)
        write_filelist(filelist_dir_host, idx, vids_hpc)

    # ---------- single Indexed Job ----------
    jobname       = s.get("jobname", "rpi-detect")
    full_job_name = f"{jobname}-{stamp}"
    par           = int(settings.k8s["job"].get("parallelism", 1))
    runner_path   = settings.k8s.get("runner_path_rpi", "/workspace/run_rpi_videos.py")

    job_dict = make_indexed_job(
        job_name         = full_job_name,
        completions      = len(chunks),
        parallelism      = par,
        filelist_dir_pod = str(filelist_dir_pod),
        runner_path      = runner_path,
        use_clahe        = use_clahe,
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
    subprocess.run(
        ["kubectl", "apply", "-f", str(spec_path), "-n", settings.k8s["namespace"]],
        check=True
    )
    print(f"Submitted Job {full_job_name} in ns {settings.k8s['namespace']}")
    print(f"Filelists dir (host): {filelist_dir_host}")
    print(f"Filelists dir (pod) : {filelist_dir_pod}")


if __name__ == "__main__":
    main()
