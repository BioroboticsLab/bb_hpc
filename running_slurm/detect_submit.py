#!/usr/bin/env python3
from slurmhelper import SLURMJob
from datetime import datetime, timezone, timedelta
import argparse, time
import re

import bb_hpc.settings as settings
from bb_hpc.src.generate import generate_jobs_detect
from bb_hpc.src.jobfunctions import job_for_process_videos


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
    return p.parse_args()


def _parse_gres_to_ngpus(gres_value):
    """Accepts 'gpu:1', 'gpu:2', 1, 2, or None; returns int count."""
    if gres_value is None:
        return 0
    if isinstance(gres_value, int):
        return gres_value
    s = str(gres_value).strip()
    m = re.match(r"gpu:(\d+)", s, flags=re.IGNORECASE)
    return int(m.group(1)) if m else 0

def _normalize_mem(mem):
    """Convert '6GB' -> '6G'. Slurm also accepts integer MiB."""
    if not mem:
        return None
    mem = str(mem).strip().upper()
    # Accept already-good forms like '6G' or '6144'
    if mem.endswith("GB"):
        return mem[:-2] + "G"
    return mem


def main():
    args = parse_args()

    # --- settings (lowercase) ---
    s = settings.detect_settings
    jobname = s.get("jobname", "detect")

    # Prefer HPC directories (this script submits to Slurm on the cluster).
    jobdir = getattr(settings, "jobdir_hpc", getattr(settings, "jobdir", None))
    videodir = getattr(settings, "videodir_hpc", getattr(settings, "videodir", None))
    pipeline_root = getattr(settings, "pipeline_root_hpc", getattr(settings, "pipeline_root", None))

    if not all([jobdir, videodir, pipeline_root]):
        raise RuntimeError("Missing required directory settings (jobdir/videodir/pipeline_root). Check settings.py.")

    chunk_size = s.get("chunk_size", 4)
    maxjobs = s.get("maxjobs", None)
    jobtime_minutes = s.get("jobtime_minutes", 60)

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
        )
    )

    # --- Slurm knobs (from settings + per-job overrides) ---
    sl = dict(settings.slurm)
    sl.update(settings.detect_settings.get("slurm", {}))

    job.qos                = sl.get("qos")
    job.partition          = sl.get("partition")
    job.custom_preamble    = sl.get("custom_preamble") or ""
    job.max_memory         = _normalize_mem(sl.get("max_memory", "6G"))
    job.n_cpus             = int(sl.get("n_cpus", 1))
    job.max_job_array_size = sl.get("max_job_array_size", 500)
    job.exports            = sl.get("exports", "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1")
    job.time_limit         = timedelta(minutes=jobtime_minutes)

    # GPUs (to make #SBATCH --gres appear)
    ngpus = _parse_gres_to_ngpus(sl.get("gres"))
    job.n_gpus = ngpus  # write_batch_file() will emit '#SBATCH --gres=gpu:<n>'

    # submit
    job.clear_input_files = lambda: None # this is needed so that createjobs() does not delete existing input files
    job.createjobs()
    job.write_batch_file()
    job.run_jobs()
    print(f"[{time.ctime()}] Checked & submitted any new detect jobs.")


if __name__ == "__main__":
    main()