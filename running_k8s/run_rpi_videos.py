#!/usr/bin/env python3
import os, sys
from pathlib import Path


# --- path shim for finding the bb_hpc package ---
_here   = Path(__file__).resolve()
_repo   = _here.parents[1]        # .../bb_hpc
_parent = _repo.parent            # .../jacob   <-- this must be on sys.path
p = str(_parent)
if p not in sys.path:
    sys.path.insert(0, p)
# -----------------------------------------------

from bb_hpc.src.jobfunctions import job_for_process_rpi_videos


# Try to read CLAHE via env (overrides settings), else fall back to settings
USE_CLAHE_ENV = os.environ.get("RPI_CLAHE", "").strip().lower()
if USE_CLAHE_ENV in ("1", "true", "yes"):
    USE_CLAHE = True
elif USE_CLAHE_ENV in ("0", "false", "no"):
    USE_CLAHE = False
else:
    try:
        from bb_hpc import settings
        USE_CLAHE = bool(settings.rpi_detect_settings.get("use_clahe", True))
    except Exception:
        USE_CLAHE = True


#################################################################
##### Runner: reuse the job function per-video for metrics    ####
#################################################################
def main():
    if len(sys.argv) < 2:
        print("Usage: run_rpi_videos.py FILELIST.txt [--no-clahe]")
        sys.exit(1)

    filelist = sys.argv[1]
    use_clahe = True
    if len(sys.argv) > 2 and sys.argv[2] == "--no-clahe":
        use_clahe = False

    if not os.path.exists(filelist):
        print(f"Filelist not found: {filelist}", file=sys.stderr)
        sys.exit(2)

    with open(filelist) as f:
        videos = [line.strip() for line in f if line.strip()]

    job_for_process_rpi_videos(videos)

if __name__ == "__main__":
    main()