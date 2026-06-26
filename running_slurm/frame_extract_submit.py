#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from slurmhelper import SLURMJob
from bb_hpc.src.slurm_utils import resolve_slurm_config, apply_slurm_to_job, run_jobs_and_log
from bb_hpc.src.generate import generate_jobs_frame_extract
from bb_hpc.src.jobfunctions import job_for_frame_extract_chunk


# usage: python -m bb_hpc.running_slurm.frame_extract_submit --dates 20250603 20250604
def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit cell-seg frame-extraction shards to SLURM")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the work list + sbatch script only; do not submit.")
    return p.parse_args()


def main():
    args = parse_args()

    s = settings.frame_extract_settings
    base_slurm = getattr(settings, "slurm", {})
    # "specific overrides general" -- frame_extract_settings.slurm wins (e.g. gres=gpu:1)
    slurm_cfg = resolve_slurm_config(base_slurm, s)

    # Build (date, cam) work-unit chunks. Already-done units are skipped via the
    # per-filename check inside generate_jobs_frame_extract.
    chunks = list(generate_jobs_frame_extract(
        video_root_dir  = str(Path(settings.videodir_hpc)),
        frames_root_dir = str(Path(settings.frames_dir_hpc)),
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
        print("No work to submit.")
        return

    # Each chunk is {"work_units": [...]} -> job_for_frame_extract_chunk(work_units=[...])
    job = SLURMJob(s.get("jobname", "frame_extract"), settings.jobdir_hpc)
    job.map(job_for_frame_extract_chunk, chunks)

    apply_slurm_to_job(job, slurm_cfg)  # GPU (gres) comes from frame_extract_settings.slurm

    job.clear_input_files = lambda: None  # keep existing queued input files

    if args.dry_run:
        # Do NOT createjobs() on a dry-run -- it writes stale .dill inputs that the
        # next real run would submit as phantom tasks (clear_input_files is disabled).
        print(f"Dry run: {len(chunks)} task(s) staged; not submitting.")
        return

    job.createjobs()
    job.write_batch_file()

    run_jobs_and_log(job, settings.jobdir_hpc, s.get("jobname", "frame_extract"), args.dates)
    print("[frame_extract_submit] Submitted.")


if __name__ == "__main__":
    main()
