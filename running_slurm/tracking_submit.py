#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from slurmhelper import SLURMJob

from bb_hpc.src.generate import generate_jobs_tracking
from bb_hpc.src.jobfunctions import job_for_tracking


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
    s_sl  = settings.slurm

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

    # Flatten to kwargs list for SLURMJob; matches job_for_tracking(**kwargs)
    job_args = []
    for batch in batches:
        for it in batch.get("job_args_list", []):
            job_args.append(dict(it))

    if not job_args:
        print("No shard args produced by generate_jobs_tracking.")
        return

    # Configure SLURM job
    job = SLURMJob(
        name=s_trk.get("jobname", "tracking"),
        job_root=settings.jobdir_hpc,
    )

    job.set_job_fun(job_for_tracking)
    job.set_job_arguments(job_args)

    # Resources (CPU by default; set n_gpus in settings.slurm if you want GPU)
    job.partition            = s_sl.get("partition", "dev")
    job.qos                  = s_sl.get("qos", "standard")
    job.time_limit           = timedelta(minutes=int(s_sl.get("jobtime_minutes", 180)))
    job.max_memory           = s_sl.get("max_memory", "8GB")
    job.n_nodes              = int(s_sl.get("nodes", 1))
    job.n_cpus               = int(s_sl.get("cpus_per_task", 1))
    job.n_tasks              = s_sl.get("ntasks_per_node", None)
    job.n_gpus               = int(s_sl.get("n_gpus", 0))  # keep 0 unless you really need GPU tracking
    job.nice                 = s_sl.get("nice", None)
    job.concurrent_job_limit = s_sl.get("concurrent_job_limit", None)
    job.max_job_array_size   = s_sl.get("max_job_array_size", "auto")
    job.exports              = s_sl.get("exports", "")
    job.custom_preamble      = s_sl.get("custom_preamble", "")

    # Generate input dill + batch file
    job.clear_input_files = lambda: None # this is needed so that createjobs() does not delete existing input files
    job.createjobs()
    job.write_batch_file()

    if args.dry_run:
        print("Dry run: not submitting.")
        return

    # Submit
    job.run_jobs()
    print("[tracking_submit] Submitted.")


if __name__ == "__main__":
    from datetime import timedelta
    main()