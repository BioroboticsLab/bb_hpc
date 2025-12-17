#!/usr/bin/env python3
from slurmhelper import SLURMJob
from datetime import datetime, timezone, timedelta
import argparse, time

import bb_hpc.settings as settings
from bb_hpc.src.generate import generate_jobs_detect
from bb_hpc.src.jobfunctions import job_for_process_videos
from bb_hpc.src.slurm_utils import resolve_slurm_config, apply_slurm_to_job


# usage: python detect_submit.py --dates 20250710 20250709 ...
def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser()
    p.add_argument(
        "--dates", nargs="+",
        default=[yesterday, today],
        help="YYYYMMDD strings to scan (default: yesterday & today, UTC)"
    )
    p.add_argument(
        "--use-fileinfo",
        action="store_true",
        help="Use fileinfo to skip already processed videos and speed up job generation."
    )
    p.add_argument(
        "--check-read-bbb",
        action="store_true",
        help="Read .bbb files to validate them (slower but skips zero/invalid outputs)."
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- settings (lowercase) ---
    s = settings.detect_settings
    jobname = s.get("jobname", "detect")

    # HPC directories (required)
    jobdir        = settings.jobdir_hpc
    videodir      = settings.videodir_hpc
    pipeline_root = settings.pipeline_root_hpc
    resultdir     = settings.resultdir_hpc

    chunk_size = s.get("chunk_size", 4)
    maxjobs = s.get("maxjobs", None)

    # Create job
    job = SLURMJob(jobname, jobdir)

    # Optional de-dupe against already-queued work:
    # slurmdir = job.get_input_dir(local_path=True)
    slurmdir = None  # keep your current behavior (no queue-scan)

    # Map function over generated work items
    job.map(
        job_for_process_videos,
        generate_jobs_detect(
            video_root_dir=str(videodir),
            repo_output_path=str(pipeline_root),
            slurmdir=slurmdir,
            chunk_size=chunk_size,
            maxjobs=maxjobs,
            datestring=args.dates,
            verbose=False,
            RESULTDIR=resultdir,
            use_fileinfo=bool(args.use_fileinfo),
            check_read_bbb=bool(args.check_read_bbb),
        )
    )

    # --- Slurm knobs (specific overrides global) ---
    slurm_cfg = resolve_slurm_config(settings.slurm, settings.detect_settings)
    apply_slurm_to_job(job, slurm_cfg)

    # submit
    job.clear_input_files = lambda: None # this is needed so that createjobs() does not delete existing input files
    job.createjobs()
    job.write_batch_file()
    job.run_jobs()
    print(f"[{time.ctime()}] Checked & submitted any new detect jobs.")


if __name__ == "__main__":
    main()
