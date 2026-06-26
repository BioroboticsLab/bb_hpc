#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from slurmhelper import SLURMJob

from bb_hpc.src.generate import generate_jobs_tracking
from bb_hpc.src.jobfunctions import job_for_tracking_chunk
from bb_hpc.src.slurm_utils import resolve_slurm_config, apply_slurm_to_job, run_jobs_and_log


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit tracking shards to SLURM")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Create input files and sbatch script only.")
    return p.parse_args()


def main():
    args = parse_args()

    s_trk = settings.track_settings
    base_slurm = settings.slurm
    # Merge with "specific overrides general"
    slurm_cfg = resolve_slurm_config(base_slurm, s_trk)    

    # Build tracking jobs using the same knobs as your Docker submitter
    batches = list(generate_jobs_tracking(
        RESULTDIR      = str(settings.resultdir_hpc),
        PIPELINE_ROOT  = str(settings.pipeline_root_hpc),
        TEMP_DIR       = str(s_trk.get("temp_path", "/tmp/bb_tracking_tmp")),
        datestring     = args.dates,
        chunk_size     = int(s_trk.get("chunk_size", 1)),
        maxjobs        = s_trk.get("maxjobs", None),
        interval_hours = int(s_trk.get("interval_hours", 1)),
    ))
    if not batches:
        print("Nothing to do.")
        return

    # Keep each batch intact: one SLURM task processes a whole chunk of up to
    # chunk_size hourly windows via job_for_tracking_chunk(job_args_list=[...]).
    # (Do NOT flatten here — flattening produces one task per hour and makes
    # chunk_size a no-op.)
    job_args = [dict(b) for b in batches if b.get("job_args_list")]

    if not job_args:
        print("No shard args produced by generate_jobs_tracking.")
        return

    total_windows = sum(len(b["job_args_list"]) for b in job_args)
    print(f"[tracking_submit] {len(job_args)} SLURM tasks, "
          f"{total_windows} hourly windows total "
          f"(~{total_windows / max(len(job_args), 1):.1f} windows/task).")

    # Configure SLURM job
    job = SLURMJob(
        name=s_trk.get("jobname", "tracking"),
        job_root=settings.jobdir_hpc,
    )
    job.set_job_fun(job_for_tracking_chunk)
    job.set_job_arguments(job_args)
    apply_slurm_to_job(job, slurm_cfg) # Apply the merged config

    job.clear_input_files = lambda: None # this is needed so that createjobs() does not delete existing input files

    if args.dry_run:
        # Do NOT createjobs() on a dry-run -- it writes stale .dill inputs that the
        # next real run would submit as phantom tasks (clear_input_files is disabled).
        print("Dry run: not submitting.")
        return

    # Generate input dill + batch file
    job.createjobs()
    job.write_batch_file()

    # Submit
    run_jobs_and_log(job, settings.jobdir_hpc, s_trk.get("jobname", "tracking"), args.dates)
    print("[tracking_submit] Submitted.")


if __name__ == "__main__":
    from datetime import timedelta
    main()