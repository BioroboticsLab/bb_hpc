#!/usr/bin/env python3
"""Submit cell-seg background-generation shards as a Kubernetes Indexed Job (GPU).

Mirrors running_slurm/background_submit.py. Run AFTER frame extraction for the
same dates. One worker per pod (the segmentation model loads once per pod).
"""
import argparse
import json
import subprocess
from pathlib import Path
from datetime import datetime

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_background
import re


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dates", nargs="+", default=None,
                   help="YYYYMMDD (one or many), date mode. Example: 20250603 20250604")
    p.add_argument("--source-dir", default=None,
                   help="Explicit frames dir containing cam-N/ (no date level). Overrides --dates.")
    p.add_argument("--label", default=None,
                   help="Output sub-key under --out-dir for --source-dir mode (default: source dir name).")
    p.add_argument("--out-dir", default=None,
                   help="Output base for --source-dir mode (default: settings.backgrounds_dir_hpc).")
    p.add_argument("--cams", nargs="+", default=None,
                   help="Optional camera filter, e.g. --cams cam-0 cam-1.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists & Job spec, but do not kubectl apply.")
    args = p.parse_args()
    if not args.source_dir and not args.dates:
        p.error("one of --dates or --source-dir is required")
    return args


def _sanitize_job_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "job"


def _remap_unit_to_hpc(unit: dict) -> dict:
    out = dict(unit)
    out["frames_root"] = str(Path(settings.frames_dir_hpc))
    out["backgrounds_root"] = str(Path(settings.backgrounds_dir_hpc))
    # In --source-dir mode the unit carries explicit source_path/output_path that
    # the user passed as cluster-visible absolute paths; leave them as-is.
    return out


def write_filelist(dir_host: Path, idx: int, work_units) -> Path:
    dir_host.mkdir(parents=True, exist_ok=True)
    f = dir_host / f"background_{idx:05d}.txt"
    with open(f, "w") as out:
        for item in work_units:
            out.write(json.dumps(item) + "\n")
    return f


def make_indexed_job(job_name, completions, parallelism, filelist_dir_pod, runner_path):
    k = settings.k8s
    env_list = [{"name": kk, "value": str(vv)} for kk, vv in k.get("env", {}).items()]
    resources = k.get("resources", {})  # GPU block (nvidia.com/gpu)
    comb_env = k.get("comb_conda_env", "combbg")

    bash = f"""\
set -euo pipefail

if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate {comb_env} || true

idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{filelist_dir_pod}/background_${{idx5}}.txt"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

# One worker per pod (the segmentation model loads once per process).
python -u {runner_path} "$fl"
"""

    pod_spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": k["image_pull_secret"]}],
        "containers": [{
            "name": "background",
            "image": k.get("image_comb", k["image"]),
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
            "labels": {"app": "background", "bbhpc": "background-indexed"},
        },
        "spec": {
            "backoffLimit": int(k["job"].get("backoff_limit", 1)),
            "completions": int(completions),
            "parallelism": int(min(max(1, parallelism), max(1, completions))),
            "completionMode": "Indexed",
            "template": {
                "metadata": {"labels": {"app": "background", "job-name": job_name}},
                "spec": pod_spec,
            },
        },
    }


def main():
    args = parse_args()

    s = settings.background_settings
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    filelist_dir_host = Path(settings.jobdir_local) / "k8s" / "background" / stamp
    filelist_dir_pod = Path(settings.jobdir_hpc) / "k8s" / "background" / stamp

    chunks = list(generate_jobs_background(
        frames_root_dir      = str(Path(settings.frames_dir_local)),
        backgrounds_root_dir = str(Path(settings.backgrounds_dir_local)),
        datestring           = args.dates,
        frame_interval_sec   = s.get("frame_interval_sec", None),
        background_window    = s.get("background_window", None),
        window_size          = int(s.get("window_size", 10)),
        num_median_images    = int(s.get("num_median_images", 200)),
        max_cycles           = s.get("max_cycles", None),
        jump_size            = int(s.get("jump_size", 1)),
        apply_clahe          = s.get("apply_clahe", "post"),
        mask_dilation        = int(s.get("mask_dilation", 15)),
        median_computation   = s.get("median_computation", "cupy"),
        device               = s.get("device", "cuda"),
        memmap_dir           = s.get("memmap_dir", None),
        chunk_size           = int(s.get("chunk_size", 2)),
        maxjobs              = s.get("maxjobs", None),
        source_dir           = args.source_dir,
        label                = args.label,
        out_dir              = args.out_dir,
        cams                 = args.cams,
    ))
    if not chunks:
        print("No background work to submit.")
        return

    filelists = []
    for idx, batch in enumerate(chunks):
        units = [_remap_unit_to_hpc(u) for u in batch.get("work_units", [])]
        if not units:
            continue
        filelists.append(write_filelist(filelist_dir_host, idx, units))

    jobname = s.get("jobname", "background")
    full_job_name = _sanitize_job_name(f"{jobname}-{stamp}")
    par = int(settings.k8s["job"].get("parallelism", 1))
    runner_path = settings.k8s.get(
        "background_runner_path",
        str(Path(settings.bb_hpc_dir_hpc) / "running_k8s" / "run_background.py"),
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
