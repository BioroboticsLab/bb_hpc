#!/usr/bin/env python3
import argparse, json, os, shlex, subprocess, time
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_detect


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
    p = argparse.ArgumentParser()
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true", help="Write filelists but don’t start containers.")
    p.add_argument("--gpus", default=settings.docker.get("gpus", "auto"),
                   help='"auto", "all", or a comma list like "0,1" to limit which GPUs to use.')
    p.add_argument("--containers-per-gpu", type=int,
                   default=int(settings.docker.get("containers_per_gpu", 1)),
                   help="How many containers to run per GPU (default 1).")
    p.add_argument("--use-fileinfo", action="store_true",
                   help="Use RESULTDIR/bbb_fileinfo/bbb_info_*.parquet to skip videos whose .bbb already exists.")
    return p.parse_args()


def list_gpu_ids(request: str):
    if request == "all" or request == "auto":
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"], text=True
            )
            ids = [s.strip() for s in out.strip().splitlines() if s.strip().isdigit()]
            return ids
        except Exception:
            # fallback to a single GPU 0 if nvidia-smi fails
            return ["0"]
    # explicit comma list
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
    # Convert a host path to the path inside container for a given bind mapping.
    rel = path.relative_to(host_bind)
    return str(container_bind / rel)


def docker_run_cmd(image, gpu_id, binds, env, runner_path, filelist_container):
    payload = f"""set -euo pipefail
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate beesbook || true
fi
export PYTHONPATH="{env.get('PYTHONPATH','/abyss/home/bb_hpc')}:${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"

python -u {runner_path} "{filelist_container}"
"""
    parts = ["docker", "run", "--rm", "--gpus", f"device={gpu_id}"]
    # Allow forcing a specific container runtime (e.g., nvidia)
    runtime = settings.docker.get("runtime")
    if runtime:
        parts += ["--runtime", str(runtime)]
    for k, v in env.items():
        parts += ["-e", f"{k}={v}"]
    for host_p, cont_p in binds:
        parts += ["-v", f"{host_p}:{cont_p}:rw"]
    parts += [image, "bash", "-lc", payload]
    return parts


def worker_loop(gpu_id: str, queue: list[Path], bind_pairs, image, env, runner_path,
                jobdir_host: Path, jobdir_hpc: Path, verbose=True):
    """Each worker pulls filelists off the queue and runs a container on its GPU."""
    # Build a mapping for translating host filelist → container path
    bind_map = [(Path(h), Path(c)) for (h, c) in bind_pairs]
    # Find the bind that contains jobdir_host (so filelists are visible)
    chosen = None
    for h, c in bind_map:
        try:
            jobdir_host.relative_to(h)
            chosen = (h, c)
            break
        except ValueError:
            continue
    if chosen is None:
        raise RuntimeError(f"jobdir_host {jobdir_host} not under any docker.binds host path")

    host_bind, cont_bind = chosen

    while True:
        try:
            filelist_host = queue.pop(0)
        except IndexError:
            return  # queue empty

        filelist_container = host_to_container_path(filelist_host, host_bind, cont_bind)
        cmd = docker_run_cmd(image, gpu_id, bind_pairs, env, runner_path, filelist_container)
        if verbose:
            print(f"[GPU {gpu_id}] starting: {shlex.join(cmd)}")
        proc = subprocess.Popen(cmd)
        rc = proc.wait()
        if verbose:
            print(f"[GPU {gpu_id}] finished {filelist_host.name} -> rc={rc}")


def main():
    args = parse_args()
    s = settings.detect_settings

    # RESULTDIR (where bbb_fileinfo lives) — prefer explicit setting, else fall back near jobdir_local
    result_local = Path(getattr(settings, "resultdir_local", Path(settings.jobdir_local).parent))

    # Local → HPC roots
    video_local = Path(settings.videodir_local)
    video_hpc   = Path(settings.videodir_hpc)
    # Mirror the k8s layout but under "docker"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    tag   = settings.submitter_tag
    jobdir_host = Path(settings.jobdir_local) / "docker" / "detect" / tag / stamp
    jobdir_hpc  = Path(settings.jobdir_hpc)   / "docker" / "detect" / tag / stamp

    # Build chunks from LOCAL view
    chunks = list(generate_jobs_detect(
        video_root_dir    = str(video_local),
        repo_output_path  = str(Path(settings.pipeline_root_local)),
        RESULTDIR         = str(result_local),
        slurmdir          = None,
        chunk_size        = s.get("chunk_size", 4),
        maxjobs           = s.get("maxjobs", None),
        datestring        = args.dates,
        verbose           = False,
        use_fileinfo      = bool(args.use_fileinfo),
    ))
    if not chunks:
        print("Nothing to do.")
        return

    # Translate each chunk’s videos to HPC-style paths, write filelists under jobdir_host
    filelists = []
    for idx, ch in enumerate(chunks):
        vids_hpc = host_to_hpc(ch["video_paths"], video_local, video_hpc)
        f = write_filelist(jobdir_host, idx, vids_hpc)
        filelists.append(f)

    if args.dry_run:
        print(f"Wrote {len(filelists)} filelists under {jobdir_host}. (dry-run: no containers started)")
        return

    # Figure out GPUs and concurrency
    gpu_ids = list_gpu_ids(args.gpus)
    if not gpu_ids:
        print("No GPUs found. Aborting.")
        return
    workers = max(1, int(args.containers_per_gpu)) * len(gpu_ids)

    # Settings for docker
    dkr = settings.docker
    image = dkr["image"]
    runner_path = dkr["runner_path"]
    env = dkr.get("env", {})
    binds = dkr.get("binds", [])

    # Simple MPMC: one queue shared by workers, each pinned to a GPU id.
    # If containers_per_gpu > 1, we duplicate the GPU ids accordingly.
    gpu_plan = []
    for gid in gpu_ids:
        for _ in range(int(args.containers_per_gpu)):
            gpu_plan.append(gid)

    # Build a work queue (list copy since we’ll pop)
    queue = list(filelists)

    # Spin workers
    with ThreadPoolExecutor(max_workers=len(gpu_plan)) as ex:
        futs = []
        for gid in gpu_plan:
            futs.append(ex.submit(worker_loop, gid, queue, binds, image, env, runner_path,
                                  jobdir_host, jobdir_hpc))
        for _ in as_completed(futs):
            pass

    print(f"Completed {len(filelists)} container runs.")


if __name__ == "__main__":
    main()