#!/usr/bin/env python3
from datetime import datetime, timezone, timedelta
import argparse, time

import bb_hpc.settings as settings
from slurmhelper import SLURMJob
from bb_hpc.src.generate import generate_jobs_rpi_detect
from bb_hpc.src.jobfunctions import job_for_process_rpi_videos  # <- adjust if your name differs


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")
    p = argparse.ArgumentParser(description="Submit RPi detect jobs to Slurm (CPU-only).")
    p.add_argument(
        "--dates", nargs="+", default=[yesterday, today],
        help="YYYYMMDD strings (UTC). Default: yesterday & today."
    )
    return p.parse_args()


def _normalize_mem(mem):
    """Convert '6GB' -> '6G'. Slurm also accepts integer MiB."""
    if not mem:
        return None
    mem = str(mem).strip().upper()
    return mem[:-2] + "G" if mem.endswith("GB") else mem


def main():
    args = parse_args()

    # ---------- settings ----------
    s = settings.rpi_detect_settings
    jobname = s.get("jobname", "rpi_detect")

    # Prefer HPC dirs for Slurm
    jobdir        = getattr(settings, "jobdir_hpc", getattr(settings, "jobdir", None))
    pi_videodir   = getattr(settings, "pi_videodir_hpc", getattr(settings, "pi_videodir", None))

    if not all([jobdir, pi_videodir]):
        raise RuntimeError("Missing required directory settings (jobdir/pi_videodir). Check settings.py.")

    chunk_size      = int(s.get("chunk_size", 150))
    use_clahe       = bool(s.get("use_clahe", True))
    jobtime_minutes = int(s.get("jobtime_minutes", 60))

    # ---------- build per-shard jobs (list-of-videos) ----------
    # generate_jobs_rpi_detect(...) yields shards with "video_paths"
    jobs_kwargs = []
    for shard in generate_jobs_rpi_detect(
        video_root_dir=str(pi_videodir),
        dates=args.dates,
        chunk_size=chunk_size,
        clahe=use_clahe,
    ):
        vids = shard.get("video_paths", [])
        if vids:
            # job_for_process_rpi_videos(videos) expects the videos as a list
            jobs_kwargs.append({"video_paths": vids})

    if not jobs_kwargs:
        print("Nothing to submit.")
        return

    # ---------- Slurm job object ----------
    job = SLURMJob(jobname, jobdir)
    job.map(job_for_process_rpi_videos, jobs_kwargs)

    # Base Slurm config + RPi overrides
    sl = dict(settings.slurm)
    sl.update(s.get("slurm", {}))

    job.qos                = sl.get("qos")
    job.partition          = sl.get("partition")
    job.custom_preamble    = sl.get("custom_preamble") or ""
    job.max_memory         = _normalize_mem(sl.get("max_memory", "6G"))
    job.n_cpus             = int(sl.get("n_cpus", 1))
    job.max_job_array_size = sl.get("max_job_array_size", 500)
    job.time_limit         = timedelta(minutes=jobtime_minutes)

    # CPU-only (ensure no GPU request in the sbatch file)
    job.n_gpus = 0

    # Exports: inherit any global env exports and add CLAHE flag
    base_exports = sl.get("exports", "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1")
    extra = "RPI_CLAHE={}".format("1" if use_clahe else "0")
    job.exports = f"{base_exports},{extra}" if base_exports else extra

    # ---------- write & submit ----------
    job.createjobs()
    job.write_batch_file()
    job.run_jobs()
    print(f"[{time.ctime()}] Submitted {len(jobs_kwargs)} RPi detect jobs for dates: {' '.join(args.dates)}")


if __name__ == "__main__":
    main()