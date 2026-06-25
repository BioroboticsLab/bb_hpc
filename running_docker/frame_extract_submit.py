#!/usr/bin/env python3
"""Submit cell-seg frame-extraction shards to local Docker (GPU).

Mirrors running_k8s/frame_extract_submit.py but launches local containers. Work
is sharded by (date, camera); each work unit is one JSONL line and the pod-side
runner running_k8s/run_frame_extract.py reads a shard and calls
job_for_frame_extract_chunk. Uses the comb-background image (settings.docker
["image_comb"]) and activates the combbg conda env.

  python -m bb_hpc.running_docker.frame_extract_submit --dates 20250603 20250604

NOTE: the default decoder is NVIDIA NVDEC (hevc_cuvid), which needs a CUDA build
of ffmpeg in the image. For software decode set frame_extract_settings["decoder"]
= None (the extractor then falls back to ffmpeg on PATH).
"""
import argparse, json, shlex, subprocess
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_frame_extract


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit cell-seg frame-extraction shards to local Docker")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Write JSONL shards only; do not start containers.")
    p.add_argument("--gpus", default=settings.docker.get("gpus", "auto"),
                   help='"auto", "all", or a comma list like "0,1" to limit which GPUs to use.')
    p.add_argument("--containers-per-gpu", type=int, default=1,
                   help="How many containers to run per GPU (default 1).")
    return p.parse_args()


def list_gpu_ids(request: str):
    if request in ("all", "auto"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            return [s.strip() for s in out.strip().splitlines() if s.strip().isdigit()]
        except Exception:
            return ["0"]
    return [s.strip() for s in request.split(",") if s.strip()]


def host_to_container_path(path, bind_pairs) -> str:
    p = Path(path)
    for host_p, cont_p in bind_pairs:
        try:
            rel = p.relative_to(Path(host_p))
            return str(Path(cont_p) / rel)
        except ValueError:
            continue
    return str(p)


def to_container_unit(u, bind_pairs) -> dict:
    """Translate the path-bearing keys of a work unit to container paths."""
    out = dict(u)
    for k in ("video_root", "frames_root"):
        v = out.get(k)
        if v:
            out[k] = host_to_container_path(v, bind_pairs)
    return out


def write_shard(dir_host: Path, idx: int, work_units) -> Path:
    dir_host.mkdir(parents=True, exist_ok=True)
    f = dir_host / f"frame_extract_{idx:05d}.jsonl"
    with open(f, "w") as out:
        for item in work_units:
            out.write(json.dumps(item) + "\n")
    return f


def gpu_docker_flags(gpu_id):
    """How to expose a GPU to the container, per settings.docker['gpu_mode']:
      - "gpus"   : --gpus device=<id>  (+ --runtime if set)  [Docker native / CDI]
      - "nvidia" : --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=<id>  [legacy toolkit]
      - "cdi"    : --device nvidia.com/gpu=<id>  (needs /etc/cdi spec)
    """
    dkr = settings.docker
    mode = dkr.get("gpu_mode", "gpus")
    runtime = dkr.get("runtime")
    if mode == "nvidia":
        flags = ["--runtime", str(runtime or "nvidia"),
                 "-e", f"NVIDIA_VISIBLE_DEVICES={gpu_id}",
                 "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
    elif mode == "cdi":
        flags = ["--device", f"nvidia.com/gpu={gpu_id}"]
        if runtime:
            flags += ["--runtime", str(runtime)]
    else:  # "gpus"
        flags = ["--gpus", f"device={gpu_id}"]
        if runtime:
            flags += ["--runtime", str(runtime)]
    return flags


def docker_run_cmd(image, gpu_id, binds, env, runner_path, shard_container, conda_env):
    payload = f"""set -euo pipefail
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
fi
conda activate {conda_env} || true
export PYTHONPATH="{env.get('PYTHONPATH', '')}:${{PYTHONPATH:-}}"

python -u {shlex.quote(runner_path)} {shlex.quote(shard_container)}
"""
    parts = ["docker", "run", "--rm"] + gpu_docker_flags(gpu_id)
    for k, v in env.items():
        parts += ["-e", f"{k}={v}"]
    for host_p, cont_p in binds:
        parts += ["-v", f"{host_p}:{cont_p}:rw"]
    parts += [image, "bash", "-lc", payload]
    return parts


def worker_loop(gpu_id, queue, binds, image, env, runner_path, conda_env, verbose=True):
    while True:
        try:
            shard_host = queue.pop(0)
        except IndexError:
            return
        shard_container = host_to_container_path(shard_host, binds)
        cmd = docker_run_cmd(image, gpu_id, binds, env, runner_path, shard_container, conda_env)
        if verbose:
            print(f"[GPU {gpu_id}] starting: {shlex.join(cmd)}", flush=True)
        rc = subprocess.call(cmd)
        if verbose:
            print(f"[GPU {gpu_id}] finished {shard_host.name} -> rc={rc}", flush=True)


def main():
    args = parse_args()
    s = settings.frame_extract_settings
    dkr = settings.docker

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag = settings.submitter_tag
    shard_dir_host = Path(settings.jobdir_local) / "docker" / "frame_extract" / tag / stamp

    # Enumerate using the submit machine's *_local view of the shared storage.
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
        verbose         = bool(args.dry_run),
    ))
    if not chunks:
        print("No frame-extract work to submit.")
        return

    binds = dkr.get("binds", [])
    env = dkr.get("env", {})

    shards = []
    for idx, batch in enumerate(chunks):
        units = [to_container_unit(u, binds) for u in batch.get("work_units", [])]
        if not units:
            continue
        shards.append(write_shard(shard_dir_host, idx, units))

    print(f"Wrote {len(shards)} shard(s) under {shard_dir_host}.")
    if args.dry_run:
        print("(dry-run: no containers started)")
        return

    gpu_ids = list_gpu_ids(args.gpus)
    if not gpu_ids:
        print("No GPUs found. Aborting.")
        return

    image = dkr.get("image_comb", dkr["image"])
    runner_path = dkr.get(
        "runner_path_frame_extract",
        settings.k8s.get("frame_extract_runner_path",
                         str(Path(settings.bb_hpc_dir_hpc) / "running_k8s" / "run_frame_extract.py")),
    )
    conda_env = dkr.get("comb_conda_env", "combbg")

    gpu_plan = []
    for gid in gpu_ids:
        for _ in range(max(1, int(args.containers_per_gpu))):
            gpu_plan.append(gid)

    queue = list(shards)
    with ThreadPoolExecutor(max_workers=len(gpu_plan)) as ex:
        futs = [ex.submit(worker_loop, gid, queue, binds, image, env, runner_path, conda_env)
                for gid in gpu_plan]
        for _ in as_completed(futs):
            pass

    print(f"Completed {len(shards)} container run(s).")


if __name__ == "__main__":
    main()
