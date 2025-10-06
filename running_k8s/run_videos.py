#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# --- path shim for finding the bb_hpc package ---
_here   = Path(__file__).resolve()
_repo   = _here.parents[1]        # .../bb_hpc
_parent = _repo.parent            # .../jacob   <-- this must be on sys.path
p = str(_parent)
if p not in sys.path:
    sys.path.insert(0, p)

from bb_hpc import settings  

# Respect env you defined (don’t override if already set)
for k, v in settings.k8s.get("env", {}).items():
    os.environ.setdefault(k, str(v))

from bb_hpc.src.jobfunctions import job_for_process_videos

def main():
    if len(sys.argv) < 2:
        print("Usage: run_videos.py FILELIST.txt")
        sys.exit(1)

    filelist = Path(sys.argv[1])
    with open(filelist) as f:
        videos = [line.strip() for line in f if line.strip()]

    # Paths
    repo_output = getattr(settings, "pipeline_root_hpc", "/tmp/pipeline_repo")

    # Detect-specific knobs (in settings.detect_settings)
    ds = getattr(settings, "detect_settings", {})
    timestamp_format = ds.get("timestamp_format", "basler")
    video_file_type  = ds.get("video_file_type", "basler")
    num_threads      = int(ds.get("num_threads", 1))
    copy_local       = bool(ds.get("copy_local", True))
    cache_dir        = ds.get("local_cache_dir", "/tmp/bb_localcache")

    job_for_process_videos(
        video_paths=videos,
        repo_output_path=repo_output,
        timestamp_format=timestamp_format,
        num_threads=num_threads,
        text_root_path=None,
        video_file_type=video_file_type,
        copy_local=copy_local,
        local_cache_dir=cache_dir,
    )

if __name__ == "__main__":
    main()