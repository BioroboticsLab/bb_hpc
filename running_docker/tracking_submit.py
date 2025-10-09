#!/usr/bin/env python3
import argparse, json, os, shlex, socket, subprocess
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_tracking


def parse_args():
    p = argparse.ArgumentParser(description="Submit tracking shards to local Docker")
    p.add_argument("--dates", nargs="+", required=True,
                   help="YYYYMMDD strings (UTC). Example: 20160719 20160720")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists but don’t start containers.")
    p.add_argument("--gpus", default=settings.docker.get("gpus", "auto"),
                   help='"auto", "all", or a comma list like "0,1" to limit which GPUs to use.')
    p.add_argument("--containers-per-gpu", type=int,
                   default=int(settings.docker.get("containers_per_gpu", 1)),
                   help="How many containers to run per GPU (default 1).")
    return p.parse_args()

def list_gpu_ids(request: str):
    if request in ("all", "auto"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            ids = [s.strip() for s in out.strip().splitlines() if s.strip().isdigit()]
            return ids
        except Exception:
            # fallback to a single GPU 0 if nvidia-smi fails
            return ["0"]
    return [s.strip() for s in request.split(",") if s.strip()]

def _json_default(o):
    if isinstance(o, datetime):
        return o.isoformat()
    return str(o)


def host_to_container_path(path: Path, bind_pairs) -> str:
    p = Path(path)
    for host_p, cont_p in bind_pairs:
        host_p = Path(host_p); cont_p = Path(cont_p)
        try:
            rel = p.relative_to(host_p)
            return str(cont_p / rel)
        except ValueError:
            continue
    return str(p)


def remap_job_paths_for_container(job: dict, binds):
    def map_one(p):
        return host_to_container_path(Path(p), binds)
    j = dict(job)
    for key in ("repo_path", "save_path", "temp_path"):
        if key in j and j[key]:
            j[key] = map_one(j[key])
    return j


def write_filelist(dir_host: Path, idx: int, jobs_list):
    dir_host.mkdir(parents=True, exist_ok=True)
    fpath = dir_host / f"tracking_{idx:05d}.jsonl"
    with open(fpath, "w") as fh:
        for item in jobs_list:
            fh.write(json.dumps(item, default=_json_default) + "\n")
    return fpath


def _resolve_gpus_arg(cfg_value: str | None) -> str | None:
    """
    Turn settings.docker['gpus'] into a valid value for '--gpus'.
    Examples:
      'all'          -> 'all'
      'auto'         -> 'all'        (sane default)
      'device=0,1'   -> 'device=0,1'
      '0,1'          -> 'device=0,1'
      '1'            -> 'device=1'
      None/''        -> None         (no --gpus)
    """
    if not cfg_value:
        return None
    v = str(cfg_value).strip().lower()
    if v in ("auto", "all"):
        return "all"
    if v.startswith("device="):
        return v
    # bare list like "0" or "0,1"
    if all(ch.isdigit() or ch == "," for ch in v):
        return f"device={v}"
    return "all"


def docker_run_cmd(image, binds, env, runner_path, filelist_container, gpus_arg: str | None, runtime: str | None):
    parts = ["docker", "run", "--rm"]
    # Allow forcing a specific container runtime (e.g., nvidia)
    if runtime:
        parts += ["--runtime", str(runtime)]
    if gpus_arg:
        parts += ["--gpus", gpus_arg]
    for k, v in env.items():
        parts += ["-e", f"{k}={v}"]
    for host_p, cont_p in binds:
        parts += ["-v", f"{host_p}:{cont_p}:rw"]

    # bash wrapper (conda activate; ensure PYTHONPATH/LD paths)
    cmd = f"""set -euo pipefail
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate beesbook || true
fi
export PYTHONPATH="{env.get('PYTHONPATH','/abyss/home/bb_hpc')}:${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"
python -u {shlex.quote(runner_path)} {shlex.quote(filelist_container)}
"""
    parts += [image, "bash", "-lc", cmd]
    return parts


def main():
    args = parse_args()

    # Generator knobs (from settings)
    s_track        = settings.track_settings
    chunk_size     = int(s_track.get("chunk_size", 1))
    maxjobs        = s_track.get("maxjobs", None)
    interval_hours = int(s_track.get("interval_hours", 1))
    temp_dir       = s_track.get("temp_path", "/tmp/bb_tracking_tmp")
    gpu_enabled    = bool(s_track.get("gpu", False))

    # Filelists locations
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host  = socket.gethostname()
    filelist_dir_host = Path(settings.jobdir_local) / "docker" / "tracking" / host / stamp
    filelist_dir_ctr  = Path(settings.jobdir_hpc)   / "docker" / "tracking" / host / stamp

    # Docker knobs (define BEFORE we need 'binds')
    dkr       = settings.docker
    runtime   = dkr.get("runtime")
    image     = dkr["image"]
    env       = dict(dkr.get("env", {}))
    binds     = dkr.get("binds", [])
    runner    = dkr.get(
        "runner_path_tracking",
        settings.k8s.get("tracking_runner_path",
                         "/abyss/home/bb_hpc/running_k8s/run_tracking.py"),
    )

    # GPU flag (optional)
    gpus_arg = _resolve_gpus_arg(dkr.get("gpus", "auto")) if gpu_enabled else None

    # Build batches
    batches = list(generate_jobs_tracking(
        RESULTDIR      = str(settings.resultdir_local),
        PIPELINE_ROOT  = str(settings.pipeline_root_local),
        TEMP_DIR       = str(temp_dir),
        datestring     = args.dates,
        chunk_size     = chunk_size,
        maxjobs        = maxjobs,
        interval_hours = interval_hours,
    ))
    if not batches:
        print("Nothing to do.")
        return

    # Write shards (after remapping paths to container)
    shards: list[Path] = []
    for idx, batch in enumerate(batches):
        jl = batch.get("job_args_list", [])
        if jl:
            jl_mapped = [remap_job_paths_for_container(it, binds) for it in jl]
            shards.append(write_filelist(filelist_dir_host, idx, jl_mapped))

    if args.dry_run:
        print(f"Wrote {len(shards)} filelists under {filelist_dir_host}. (dry-run)")
        return

    # Concurrency: plan containers per GPU and pin each worker to a GPU id
    if not gpu_enabled:
        # no GPU requested: still use parallel workers, with the number set by containers_per_gpu
        workers = min(len(shards), int(args.containers_per_gpu))
        gpus_for_workers = [None] * workers
    else:
        gpu_ids = list_gpu_ids(args.gpus)
        if not gpu_ids:
            print("No GPUs found. Aborting.")
            return
        gpus_for_workers = []
        for gid in gpu_ids:
            for _ in range(int(args.containers_per_gpu)):
                gpus_for_workers.append(gid)
        # don’t spawn more workers than shards
        if len(gpus_for_workers) > len(shards):
            gpus_for_workers = gpus_for_workers[:len(shards)]

    def worker(gpu_id: str | None):
        while True:
            try:
                f = shards.pop(0)
            except IndexError:
                return
            fl_ctr = host_to_container_path(f, binds)
            # If GPU is enabled, pin this worker to its device; else run without --gpus
            per_worker_gpus_arg = None
            if gpu_enabled:
                per_worker_gpus_arg = f"device={gpu_id}" if gpu_id is not None else None
            cmd = docker_run_cmd(image, binds, env, runner, fl_ctr, per_worker_gpus_arg, runtime)
            print(f"Starting: {shlex.join(cmd)}", flush=True)
            rc = subprocess.call(cmd)
            print(f"Finished {f.name} -> rc={rc}", flush=True)

    with ThreadPoolExecutor(max_workers=max(1, len(gpus_for_workers))) as ex:
        futs = [ex.submit(worker, gid) for gid in (gpus_for_workers or [None])]
        for _ in as_completed(futs):
            pass

    print(f"Completed {len(shards)+0} container runs.")


if __name__ == "__main__":
    main()