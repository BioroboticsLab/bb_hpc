#!/usr/bin/env python3
import argparse, os, shlex, socket, subprocess, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_rpi_detect

def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="Submit RPi detect shards to local Docker")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists, but do not start containers.")
    p.add_argument("--gpus", default=settings.docker.get("gpus", "auto"),
                   help='"auto", "all", or a comma list like "0,1".')
    p.add_argument("--containers-per-gpu", type=int,
                   default=int(settings.docker.get("containers_per_gpu", 1)),
                   help="How many containers to run per GPU.")
    p.add_argument("--clahe", dest="clahe", action=argparse.BooleanOptionalAction,
                   default=bool(settings.rpi_detect_settings.get("use_clahe", True)),
                   help="Enable/disable CLAHE; controls output suffix -c/-nc.")
    p.add_argument("--chunk-size", type=int, default=int(settings.rpi_detect_settings.get("chunk_size", 150)),
                   help="Videos per shard.")
    return p.parse_args()

def list_gpu_ids(request: str):
    if request in ("all", "auto"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            return [s.strip() for s in out.splitlines() if s.strip().isdigit()]
        except Exception:
            return ["0"]
    return [s.strip() for s in request.split(",") if s.strip()]

def host_to_hpc(paths, host_root: Path, hpc_root: Path):
    out = []
    for p in paths:
        p = Path(p)
        try:
            rel = p.relative_to(host_root)
            out.append(str(hpc_root / rel))
        except ValueError:
            out.append(str(p))
    return out

def write_filelist(root_host: Path, idx: int, videos_hpc):
    root_host.mkdir(parents=True, exist_ok=True)
    f = root_host / f"videos_{idx:05d}.txt"
    with open(f, "w") as fh:
        fh.write("\n".join(videos_hpc))
    return f

def host_to_container_path(path: Path, host_bind: Path, container_bind: Path) -> str:
    rel = path.relative_to(host_bind)
    return str(container_bind / rel)

def docker_run_cmd(image, gpu_id, binds, env, runner_path, filelist_container, runtime=None):
    # Use a tiny bash prelude so we can activate conda if present & set PYTHONPATH etc.
    bash = f"""set -euo pipefail
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate beesbook || true
fi
export PYTHONPATH="{settings.docker.get('env', {}).get('PYTHONPATH', '')}:${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"
python -u {runner_path} "{filelist_container}"
"""
    parts = ["docker", "run", "--rm"]
    if runtime:
        parts += ["--runtime", runtime]
    parts += ["--gpus", f"device={gpu_id}"]
    for k, v in env.items():
        parts += ["-e", f"{k}={v}"]
    for host_p, cont_p in binds:
        parts += ["-v", f"{host_p}:{cont_p}:rw"]
    parts += [image, "bash", "-lc", bash]
    return parts

def worker_loop(gpu_id, queue, bind_pairs, image, base_env, runner_path, jobdir_host, polo_config, runtime=None):
    bind_map = [(Path(h), Path(c)) for (h, c) in bind_pairs]
    chosen = None
    for h, c in bind_map:
        try:
            jobdir_host.relative_to(h)
            chosen = (h, c); break
        except ValueError:
            pass
    if chosen is None:
        raise RuntimeError(f"jobdir_host {jobdir_host} is not under any docker.binds host path")
    host_bind, cont_bind = chosen

    while True:
        try:
            filelist_host, model_type = queue.pop(0)
        except IndexError:
            return

        # Build per-container env: inject model type and POLO config if needed
        env = dict(base_env)
        env["RPI_MODEL_TYPE"] = model_type
        if model_type == "polo" and polo_config:
            env.update(_polo_env_vars(polo_config))

        filelist_container = host_to_container_path(filelist_host, host_bind, cont_bind)
        cmd = docker_run_cmd(image, gpu_id, bind_pairs, env, runner_path, filelist_container, runtime=runtime)
        print(f"[GPU {gpu_id}] starting ({model_type}): {shlex.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        print(f"[GPU {gpu_id}] finished {filelist_host.name} ({model_type}) -> rc={rc}", flush=True)

def _polo_env_vars(polo_cfg):
    """Build env-var dict for POLO config to inject into containers."""
    return {
        "POLO_MODEL_PATH": polo_cfg["polo_model_path"],
        "POLO_ATTRIBUTES_PATH": polo_cfg["attributes_path"],
        "POLO_CONFIDENCE_THRESHOLD": str(polo_cfg.get("confidence_threshold", 0.5)),
        "POLO_IMGSZ": str(polo_cfg.get("imgsz", 640)),
        "POLO_NMS_RADIUS": str(polo_cfg.get("nms_radius", 30)),
    }


def main():
    args = parse_args()
    s = settings.rpi_detect_settings

    video_local = Path(settings.pi_videodir_local)
    video_hpc   = Path(settings.pi_videodir_hpc)

    # Per-camera model rules (backward compatible: default to empty)
    cam_model_rules = getattr(settings, "cam_model_rules", {}) or {}
    polo_config     = getattr(settings, "polo_config", {}) or {}

    # unique per-host timestamped jobdir (avoids collisions across machines)
    host = socket.gethostname()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobdir_host = Path(settings.jobdir_local) / "docker" / s.get("jobname", "rpi_detect") / host / stamp
    jobdir_hpc  = Path(settings.jobdir_hpc)   / "docker" / s.get("jobname", "rpi_detect") / host / stamp

    # Shard videos (LOCAL view), partitioned by model type
    chunks = list(generate_jobs_rpi_detect(
        video_root_dir   = str(video_local),
        dates            = args.dates,
        chunk_size       = int(args.chunk_size),
        clahe            = bool(args.clahe),
        cam_model_rules  = cam_model_rules,
    ))
    if not chunks:
        print("Nothing to do.")
        return
    print(f"Prepared {len(chunks)} shards (chunk_size={args.chunk_size}, clahe={args.clahe})")

    # Translate to HPC paths for containers and write shard filelists
    # Each entry is (filelist_path, model_type) so we can inject per-container env vars
    filelists = []
    for idx, ch in enumerate(chunks):
        model_type = ch.get("model_type", "default")
        vids_hpc = host_to_hpc(ch["video_paths"], video_local, video_hpc)
        f = write_filelist(jobdir_host, idx, vids_hpc)
        filelists.append((f, model_type))

    if args.dry_run:
        for f, mt in filelists:
            print(f"  {f.name}  model_type={mt}")
        print(f"Wrote {len(filelists)} filelists under {jobdir_host} (dry-run).")
        return

    # Plan GPUs
    gpu_ids = list_gpu_ids(args.gpus)
    if not gpu_ids:
        print("No GPUs detected; aborting.")
        return
    gpu_plan = []
    for gid in gpu_ids:
        for _ in range(int(args.containers_per_gpu)):
            gpu_plan.append(gid)

    # Docker/runtime bits
    dkr        = settings.docker
    image      = dkr["image"]
    runner     = dkr.get("runner_path_rpi", "running_k8s/run_rpi_videos.py")
    base_env   = dict(dkr.get("env", {}))  # copy
    # Pass CLAHE down to the container explicitly (runner reads RPI_CLAHE)
    base_env["RPI_CLAHE"] = "1" if args.clahe else "0"
    binds      = dkr.get("binds", [])
    runtime    = dkr.get("runtime")
    if runtime:
        print(f"Using Docker runtime: {runtime}")

    # Fan out workers (queue now holds (filelist, model_type) tuples)
    queue = list(filelists)
    with ThreadPoolExecutor(max_workers=len(gpu_plan)) as ex:
        futs = []
        for gid in gpu_plan:
            futs.append(ex.submit(
                worker_loop, gid, queue, binds, image, base_env, runner,
                jobdir_host, polo_config, runtime,
            ))
        for _ in as_completed(futs):
            pass

    print(f"Completed {len(filelists)} container runs.")

if __name__ == "__main__":
    main()