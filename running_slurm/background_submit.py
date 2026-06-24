#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from slurmhelper import SLURMJob
from bb_hpc.src.slurm_utils import resolve_slurm_config, apply_slurm_to_job, run_jobs_and_log
from bb_hpc.src.generate import generate_jobs_background
from bb_hpc.src.jobfunctions import job_for_background_chunk


# usage: python -m bb_hpc.running_slurm.background_submit --dates 20250603 20250604
# Run AFTER frame extraction for the same dates (it only schedules cameras whose
# extracted frames already exist).
def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit cell-seg background-generation shards to SLURM")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Build the work list + sbatch script only; do not submit.")
    return p.parse_args()


def main():
    args = parse_args()

    s = settings.background_settings
    base_slurm = getattr(settings, "slurm", {})
    slurm_cfg = resolve_slurm_config(base_slurm, s)

    # Build (date, cam) work-unit chunks. Already-done units (for THIS config)
    # are skipped via the per-filename check inside generate_jobs_background.
    chunks = list(generate_jobs_background(
        frames_root_dir      = str(Path(settings.frames_dir_hpc)),
        backgrounds_root_dir = str(Path(settings.backgrounds_dir_hpc)),
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
        verbose              = bool(args.dry_run),
    ))
    if not chunks:
        print("No work to submit.")
        return

    # Each chunk is {"work_units": [...]} -> job_for_background_chunk(work_units=[...])
    job = SLURMJob(s.get("jobname", "background"), settings.jobdir_hpc)
    job.map(job_for_background_chunk, chunks)

    apply_slurm_to_job(job, slurm_cfg)  # GPU (gres) comes from background_settings.slurm

    job.clear_input_files = lambda: None
    job.createjobs()
    job.write_batch_file()

    if args.dry_run:
        print(f"Dry run: {len(chunks)} task(s) staged; not submitting.")
        return

    run_jobs_and_log(job, settings.jobdir_hpc, s.get("jobname", "background"), args.dates)
    print("[background_submit] Submitted.")


if __name__ == "__main__":
    main()
