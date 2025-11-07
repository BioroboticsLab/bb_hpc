#!/usr/bin/env python3
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta

from bb_hpc import settings
from slurmhelper import SLURMJob
from bb_hpc.src.slurm_utils import resolve_slurm_config, apply_slurm_to_job
from bb_hpc.src.generate import generate_jobs_save_detect  
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
    base_slurm  = getattr(settings, "slurm", {})
    # Merge with "specific overrides general"
    slurm_cfg = resolve_slurm_config(base_slurm, s_sd)       

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
    job.map(job_for_save_detect_chunk, job_args)
 
    apply_slurm_to_job(job, slurm_cfg) # Apply the merged config
    # Ensure CPU-only
    job.n_gpus = 0

    # Create inputs + batch file
    job.clear_input_files = lambda: None  # prevent createjobs() from deleting existing input files
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