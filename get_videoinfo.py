#!/usr/bin/env python3
"""
Build a catalog of ALL raw video files (no date filtering).

Scans:
  <VideoDirLocal>/<YYYYMMDD>/cam-*/**/*.mp4

Saves Parquet to:
  <ResultDirLocal>/bbb_fileinfo/video_info_all.parquet

Columns:
  - file_name
  - full_path
  - starttime (UTC, from parse_video_fname)
  - endtime   (UTC, from parse_video_fname)
  - cam (parsed; extra column for convenience)
"""

import os
import sys
import argparse
from datetime import datetime, timezone
from typing import Iterable, Tuple, Optional

try:
    import pandas as pd
except Exception as e:
    print(f"ERROR: pandas is required: {e}", file=sys.stderr)
    sys.exit(1)

# bb_hpc settings + filename parser
try:
    from bb_hpc import settings
    from bb_binary.parsing import parse_video_fname  # cam, start_dt, end_dt
except Exception as e:
    print(f"ERROR: could not import required modules: {e}", file=sys.stderr)
    sys.exit(1)


# -------------------------------
# Helpers
# -------------------------------
def _get_local_paths() -> Tuple[str, str]:
    """
    Return (video_root_local, resultdir_local) from bb_hpc.settings.
    Supports either 'VideoDirLocal' (CamelCase) or 'videodir_local' (snake_case),
    and 'ResultDirLocal' or 'resultdir_local'.
    """
    video_root = getattr(settings, "videodir_local", "")
    resultdir  = getattr(settings, "resultdir_local", "")
    if not video_root:
        print("ERROR: videodir_local is not set in bb_hpc.settings.", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(video_root):
        print(f"ERROR: videodir_local path does not exist: {video_root}", file=sys.stderr)
        sys.exit(2)
    if not resultdir:
        print("ERROR: resultdir_local is not set in bb_hpc.settings.", file=sys.stderr)
        sys.exit(2)
    return video_root, resultdir


def _iter_all_day_dirs(video_root: str) -> Iterable[str]:
    """
    Yield absolute paths to top-level YYYYMMDD directories under video_root.
    (No date filtering; we scan them all.)
    """
    try:
        with os.scandir(video_root) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False) and len(e.name) == 8 and e.name.isdigit():
                    yield e.path
    except PermissionError:
        # best-effort
        for name in os.listdir(video_root):
            p = os.path.join(video_root, name)
            if os.path.isdir(p) and len(name) == 8 and name.isdigit():
                yield p


def _iter_videos_in_day(day_dir: str, exts: Tuple[str, ...]) -> Iterable[str]:
    """
    Walk a single day directory and yield files with extensions in 'exts' (case-insensitive).
    """
    exts_lower = tuple(e.lower() for e in exts)
    for root, _, files in os.walk(day_dir):
        for f in files:
            if f.lower().endswith(exts_lower):
                yield os.path.join(root, f)



# -------------------------------
# Core
# -------------------------------
def build_video_info_df(video_root: str, exts: Tuple[str, ...]) -> "pd.DataFrame":
    rows = []
    n_days = 0
    n_files = 0

    for day_dir in _iter_all_day_dirs(video_root):
        n_days += 1
        for path in _iter_videos_in_day(day_dir, exts):
            n_files += 1
            fn = os.path.basename(path)

            # Parse via bb_binary utility
            cam = None
            start = None
            end = None
            try:
                cam, start, end = parse_video_fname(fn)  # returns (cam, start_dt, end_dt)
            except Exception:
                # Leave as None if unparsable
                pass

            rows.append({
                "file_name": fn,
                "full_path": path,
                "starttime": start,
                "endtime": end,
                "cam": cam,
            })

    print(f"[scan] days={n_days}, files_seen={n_files}, files_kept={len(rows)}")
    cols = ["file_name", "full_path", "starttime", "endtime", "cam"]
    return pd.DataFrame(rows, columns=cols)


# -------------------------------
# CLI
# -------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Create catalog of ALL raw videos under VideoDirLocal.")
    p.add_argument(
        "--exts",
        nargs="+",
        default=[".mp4"],
        help="File extensions to include (case-insensitive). Default: .mp4",
    )
    p.add_argument(
        "--outfile",
        default=None,
        help="Optional explicit parquet path. Default: <RESULTDIR_LOCAL>/bbb_fileinfo/video_info_all.parquet",
    )
    return p.parse_args()


def main():
    args = parse_args()
    video_root, resultdir = _get_local_paths()

    cache_dir = os.path.join(resultdir, "bbb_fileinfo")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[config] VideoDirLocal = {video_root}")
    print(f"[config] ResultDirLocal= {resultdir}")
    print(f"[config] cache_dir     = {cache_dir}")
    print(f"[config] exts          = {args.exts}")

    df = build_video_info_df(video_root, tuple(args.exts))
    print(f"[videoinfo] collected {len(df)} files")

    out_path = args.outfile or os.path.join(cache_dir, "video_info_all.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[videoinfo] wrote {len(df)} rows -> {out_path}")
    print("✅ Done.")


if __name__ == "__main__":
    main()