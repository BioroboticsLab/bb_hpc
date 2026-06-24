#!/usr/bin/env python3
"""Submit cell-seg frame-extraction shards as a Kubernetes Indexed Job (GPU).

Mirrors running_slurm/frame_extract_submit.py: builds (date, camera) work units
(skipping already-done ones via the per-filename check), writes one JSONL
filelist per task, and applies an Indexed Job where each completion index runs
run_frame_extract.py on its filelist. One worker per pod (the engine does its
own per-camera threading and ffmpeg uses the GPU's NVDEC).

Enumeration reads the filesystem on the submit machine (via *_local roots) but
the work units carry pod-visible *_hpc paths.
"""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_frame_extract
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dates", nargs="+", required=True,
                   help="YYYYMMDD (one or many). Example: 20250603 20250604")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists & Job spec, but do not kubectl apply.")
    return p.parse_args()


def _sanitize_job_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "job"


def _remap_unit_to_hpc(unit: dict) -> dict:
    """Swap the local enumeration roots for pod-visible HPC roots."""
    out = dict(unit)
    out["video_root"] = str(Path(settings.videodir_hpc))
    out["frames_root"] = str(Path(settings.frames_dir_hpc))
    return out


def write_filelist(dir_host: Path, idx: int, work_units) -> Path:
    dir_host.mkdir(parents=True, exist_ok=True)
    f = dir_host / f"frame_extract_{idx:05d}.txt"
    with open(f, "w") as out:
        for item in work_units:
            out.write(json.dumps(item) + "\n")
    return f


def make_indexed_job(job_name, completions, parallelism, filelist_dir_pod, runner_path):
    k = settings.k8s
    env_list = [{"name": kk, "value": str(vv)} for kk, vv in k.get("env", {}).items()]
    resources = k.get("resources", {})  # GPU block (nvidia.com/gpu)

    bash = f"""\
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)"
fi
conda activate beesbook || true

idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{filelist_dir_pod}/frame_extract_${{idx5}}.txt"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

# One worker per pod: the engine threads per-camera internally and uses the GPU.
python -u {runner_path} "$fl"
"""

    pod_spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": k["image_pull_secret"]}],
        "containers": [{
            "name": "frame-extract",
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
            "labels": {"app": "frame-extract", "bbhpc": "frame-extract-indexed"},
        },
        "spec": {
            "backoffLimit": int(k["job"].get("backoff_limit", 1)),
            "completions": int(completions),
            "parallelism": int(min(max(1, parallelism), max(1, completions))),
            "completionMode": "Indexed",
            "template": {
                "metadata": {"labels": {"app": "frame-extract", "job-name": job_name}},
                "spec": pod_spec,
            },
        },
    }


def main():
    args = parse_args()

    s = settings.frame_extract_settings
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filelist_dir_host = Path(settings.jobdir_local) / "k8s" / "frame_extract" / stamp
    filelist_dir_pod = Path(settings.jobdir_hpc) / "k8s" / "frame_extract" / stamp

    # Enumerate using local roots (submit machine view of the shared storage).
    chunks = list(generate_jobs_frame_extract(
        video_root_dir  = str(Path(settings.videodir_local)),
        frames_root_dir = str(Path(settings.frames_dir_local)),
        datestring      = args.dates,
        interval_in_sec = int(s.get("interval_in_sec", 60)),
        fps             = int(s.get("fps", 3)),
        file_format     = s.get("file_format", "png"),
        decoder         = s.get("decoder", "hevc_cuvid"),
        max_workers     = int(s.get("max_workers", 2)),
        chunk_size      = int(s.get("chunk_size", 4)),
        maxjobs         = s.get("maxjobs", None),
    ))
    if not chunks:
        print("No frame-extract work to submit.")
        return

    filelists = []
    for idx, batch in enumerate(chunks):
        units = [_remap_unit_to_hpc(u) for u in batch.get("work_units", [])]
        if not units:
            continue
        filelists.append(write_filelist(filelist_dir_host, idx, units))

    jobname = s.get("jobname", "frame_extract")
    full_job_name = _sanitize_job_name(f"{jobname}-{stamp}")
    par = int(settings.k8s["job"].get("parallelism", 1))
    runner_path = settings.k8s.get(
        "frame_extract_runner_path",
        str(Path(settings.bb_hpc_dir_hpc) / "running_k8s" / "run_frame_extract.py"),
    )

    job_dict = make_indexed_job(full_job_name, len(filelists), par, str(filelist_dir_pod), runner_path)

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


if __name__ == "__main__":
    main()
