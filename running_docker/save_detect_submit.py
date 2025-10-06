#!/usr/bin/env python3

## run with (example with expansion for date range)
# python -m bb_hpc.running_docker.save_detect_submit --dates $(for d in $(seq 0 62); do date -d "2016-07-19 +$d day" +%Y%m%d; done)

import argparse, json, os, shlex, socket, subprocess, time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

from bb_hpc import settings
from bb_hpc.src.generate import generate_jobs_save_detect

def parse_args():
    p = argparse.ArgumentParser(description="Submit save-detect shards to local Docker")
    p.add_argument("--dates", nargs="+", required=True,
                   help="YYYYMMDD strings (UTC). Example: 20250901 20250902")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists but don’t start containers.")
    # new: CPU-only concurrency (containers in parallel)
    default_workers = int(getattr(settings, "docker", {}).get("containers", 2))
    p.add_argument("--workers", type=int, default=default_workers,
                   help=f"How many containers to run in parallel (default: {default_workers})")
    return p.parse_args()

def _json_default(o):
    # Serialize datetimes to ISO 8601 strings
    if isinstance(o, datetime):
        return o.isoformat()
    # Fallback (shouldn’t be needed)
    return str(o)

def write_filelist(dir_host: Path, idx: int, jobs_list):
    """
    Write one JSONL shard file. Each line is a job dict with datetimes encoded as ISO strings.
    """
    dir_host.mkdir(parents=True, exist_ok=True)
    fpath = dir_host / f"savedetect_{idx:05d}.jsonl"
    with open(fpath, "w") as fh:
        for item in jobs_list:
            fh.write(json.dumps(item, default=_json_default) + "\n")
    return fpath

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

def docker_run_cmd(image, binds, env, runner_path, filelist_container):
    parts = ["docker", "run", "--rm"]
    for k, v in env.items():
        parts += ["-e", f"{k}={v}"]
    for host_p, cont_p in binds:
        parts += ["-v", f"{host_p}:{cont_p}:rw"]
    # run via bash -lc so conda activation works if needed
    cmd = f"""set -euo pipefail
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate beesbook || true
fi
export PYTHONPATH="{env.get('PYTHONPATH','/abyss/home/beesbook/jacob/bb_hpc')}:${{PYTHONPATH:-}}"
export LD_LIBRARY_PATH="/opt/conda/envs/beesbook/lib:${{LD_LIBRARY_PATH:-}}"

python -u {shlex.quote(runner_path)} {shlex.quote(filelist_container)}
"""
    parts += [image, "bash", "-lc", cmd]
    return parts

def main():
    args = parse_args()

    s = settings.save_detect_settings
    chunk_size = int(s.get("chunk_size", 50))
    maxjobs    = s.get("maxjobs", None)

    # Where to place filelists
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    host = socket.gethostname()
    filelist_dir_host = Path(settings.jobdir_local) / "docker" / "save_detect" / host / stamp

    # Build shards from generator
    resultdir_local = str(settings.resultdir_local)
    pipeline_root   = str(settings.pipeline_root_local)
    batches = list(generate_jobs_save_detect(
        RESULTDIR=resultdir_local,
        PIPELINE_ROOT=pipeline_root,
        chunk_size=chunk_size,
        maxjobs=maxjobs,
        datestring=args.dates,
    ))
    if not batches:
        print("Nothing to do.")
        return

    # Docker knobs
    dkr   = settings.docker
    image = dkr["image"]
    env   = dkr.get("env", {})
    binds = dkr.get("binds", [])
    # Prefer specific save-detect runner path, else general runner_path, else our default:
    runner_path = dkr.get(
        "runner_path_save_detect",
        settings.k8s.get("save_detect_runner_path",
                         "/abyss/home/jacob-davidson/cascbstorage/jacob/bb_hpc/running_k8s/run_save_detect.py"),
    )

    def _to_container_job(d):
        out = dict(d)
        # translate host -> container
        from pathlib import Path
        out["repo_path"] = host_to_container_path(Path(d["repo_path"]), binds)
        out["save_path"] = host_to_container_path(Path(d["save_path"]), binds)
        # serialize datetimes to ISO
        from datetime import datetime
        for k in ("from_dt", "to_dt"):
            v = out.get(k)
            if isinstance(v, datetime):
                out[k] = v.isoformat()
        return out

    shard_paths = []
    for idx, batch in enumerate(batches):
        jl = batch.get("job_args_list", [])
        if not jl:
            continue
        mapped = [_to_container_job(it) for it in jl]
        shard_paths.append(write_filelist(filelist_dir_host, idx, mapped))

    if args.dry_run:
        print(f"Wrote {len(shard_paths)} filelists under {filelist_dir_host}. (dry-run)")
        return



    # Launch
    queue = list(shard_paths)

    def worker():
        while True:
            try:
                f = queue.pop(0)
            except IndexError:
                return
            fl_ctr = host_to_container_path(f, binds)
            cmd = docker_run_cmd(image, binds, env, runner_path, fl_ctr)
            print(f"Starting: {shlex.join(cmd)}", flush=True)
            rc = subprocess.call(cmd)
            print(f"Finished {f.name} -> rc={rc}", flush=True)

    # cap workers to number of shards
    workers = max(1, min(args.workers, len(shard_paths)))
    print(f"Running {len(shard_paths)} shards with {workers} parallel container(s).", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(worker) for _ in range(workers)]
        for _ in as_completed(futures):
            pass

    print(f"Completed {len(shard_paths)} container runs.")

if __name__ == "__main__":
    main()