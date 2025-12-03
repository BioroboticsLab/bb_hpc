#!/usr/bin/env python3
from datetime import datetime, timezone, timedelta
import argparse, time

import bb_hpc.settings as settings
from slurmhelper import SLURMJob
from bb_hpc.src.generate import generate_jobs_rpi_detect
from bb_hpc.src.jobfunctions import job_for_process_rpi_videos 
from bb_hpc.src.slurm_utils import resolve_slurm_config, apply_slurm_to_job


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

def main():
    args = parse_args()

    # ---------- settings ----------
    s = settings.rpi_detect_settings
    base_slurm = settings.slurm
    # Merge with "specific overrides general"
    slurm_cfg = resolve_slurm_config(base_slurm, s)   

    # Simplified: prefer unified HPC-style paths; fallback to legacy names
    chunk_size      = int(s.get("chunk_size", 150))
    use_clahe       = bool(s.get("use_clahe", True))
    print('Use clahe:', use_clahe)

    # ---------- build per-shard jobs (list-of-videos) ----------
    # generate_jobs_rpi_detect(...) yields shards with "video_paths"
    jobs_kwargs = []
    for shard in generate_jobs_rpi_detect(
        video_root_dir=str(settings.pi_videodir_hpc),
        dates=args.dates,
        chunk_size=chunk_size,
        clahe=use_clahe,
    ):
        vids = shard.get("video_paths", [])
        if vids:
            # job_for_process_rpi_videos(videos) expects the videos as a list
            jobs_kwargs.append({"video_paths": vids, "clahe": use_clahe})

    if not jobs_kwargs:
        print("Nothing to submit.")
        return

    # ---------- Slurm job object ----------
    job = SLURMJob(s.get("jobname", "rpi_detect"), settings.jobdir_hpc)
    job.map(job_for_process_rpi_videos, jobs_kwargs)

    apply_slurm_to_job(job, slurm_cfg) # Apply the merged config

    # Add CLAHE flag to exports while keeping what apply_slurm_to_job already set
    extra = "RPI_CLAHE={}".format("1" if use_clahe else "0")
    job.exports = f"{job.exports},{extra}" if getattr(job, "exports", "") else extra

    # Harden: ensure n_gpus stays 0 for this CPU-only pipeline
    job.n_gpus = 0

    # ---------- write & submit ----------
    job.clear_input_files = lambda: None # this is needed so that createjobs() does not delete existing input files
    job.createjobs()
    job.write_batch_file()
    job.run_jobs()
    print(f"[{time.ctime()}] Submitted {len(jobs_kwargs)} RPi detect jobs for dates: {' '.join(args.dates)}")


if __name__ == "__main__":
    main()
