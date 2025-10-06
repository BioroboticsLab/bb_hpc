#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from slurmhelper import SLURMJob

# NOTE: If your function names differ, adjust these two imports.
from bb_hpc.src.generate import generate_jobs_save_detect  # <-- confirm this exists
from bb_hpc.src.jobfunctions import job_for_save_detect_chunk


def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(description="Submit save-detect shards to SLURM")
    p.add_argument("--dates", nargs="+", default=[yesterday, today],
                   help="YYYYMMDD strings (UTC). Default: yesterday & today.")
    p.add_argument("--dry-run", action="store_true",
                   help="Create input files and sbatch script only.")
    return p.parse_args()


def main():
    args = parse_args()

    # Settings buckets
    s_sd  = settings.save_detect_settings
    s_sl  = settings.slurm

    # Where SLURMJob writes (HPC paths, since slurm nodes see those)
    job_root = str(Path(settings.jobdir_hpc) / "save_detect")

    # Build shards (HPC view where we list videos/repo), same style as detect
    chunks = list(generate_jobs_save_detect(
        RESULTDIR      = str(Path(settings.resultdir_hpc)),
        PIPELINE_ROOT  = str(Path(settings.pipeline_root_hpc)),
        datestring     = args.dates,
        chunk_size     = int(s_sd.get("chunk_size", 50)),
        maxjobs        = s_sd.get("maxjobs", None),
    ))
    if not chunks:
        print("No work to submit.")
        return

    # Each chunk is already a dict with one key: {"job_args_list": [...]}
    # That matches the job function signature: job_for_save_detect_chunk(job_args_list=[...])
    job_args = chunks

    # Configure SLURM job
    job = SLURMJob(s_sd.get("jobname", "save_detect"), settings.jobdir_hpc)
    job.set_job_fun(job_for_save_detect_chunk)
    job.set_job_arguments(job_args)

    # Resources via settings.slurm (CPU-only by default)
    job.partition            = s_sl.get("partition", "dev")
    job.qos                  = s_sl.get("qos", "standard")
    job.time_limit           = timedelta(minutes=int(s_sl.get("jobtime_minutes", 60)))
    job.max_memory           = s_sl.get("max_memory", "6GB")
    job.n_nodes              = int(s_sl.get("nodes", 1))
    job.n_cpus               = int(s_sl.get("cpus_per_task", 1))
    job.n_tasks              = s_sl.get("ntasks_per_node", None)
    job.n_gpus               = int(s_sl.get("n_gpus", 0))  # 0 = CPU only
    job.nice                 = s_sl.get("nice", None)
    job.concurrent_job_limit = s_sl.get("concurrent_job_limit", None)
    job.max_job_array_size   = s_sl.get("max_job_array_size", "auto")
    job.exports              = s_sl.get("exports", "")  # e.g. env tuning
    job.custom_preamble      = s_sl.get("custom_preamble", "")

    # Create inputs + batch file
    job.createjobs()
    job.write_batch_file()

    if args.dry_run:
        print("Dry run: not submitting. Use --run via slurmhelper if desired.")
        return

    # Submit arrays (respects max_job_array_size / concurrent_job_limit)
    job.run_jobs()
    print("[save_detect_submit] Submitted.")


if __name__ == "__main__":
    main()