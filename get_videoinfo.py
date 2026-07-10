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

By default this is INCREMENTAL: per-day catalogs are cached under
<ResultDirLocal>/bbb_fileinfo/daily/video_YYYYMMDD.parquet and a day is
rescanned only when it has files newer than its cache (plus the N newest days,
which are always rescanned). Use --no-use-cache for a full rescan.
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
    from bb_hpc.src.fileinfo import (
        VIDEO_INFO_COLUMNS,
        list_video_day,
        list_video_files_incremental,
        _is_day_dirname,
    )
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
                if e.is_dir(follow_symlinks=False) and _is_day_dirname(e.name):
                    yield e.path
    except PermissionError:
        # best-effort
        for name in os.listdir(video_root):
            p = os.path.join(video_root, name)
            if os.path.isdir(p) and _is_day_dirname(name):
                yield p


# -------------------------------
# Core
# -------------------------------
def build_video_info_df(video_root: str, exts: Tuple[str, ...]) -> "pd.DataFrame":
    """Full (non-incremental) rescan of every day directory under video_root."""
    frames = []
    n_days = 0

    for day_dir in _iter_all_day_dirs(video_root):
        n_days += 1
        frames.append(list_video_day(day_dir, exts))

    df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=VIDEO_INFO_COLUMNS)
    print(f"[scan] days={n_days}, files_kept={len(df)}")
    return df


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
    p.add_argument(
        "--use-cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse per-day caches, rescanning only changed days (default: true). "
             "Use --no-use-cache to force a full rescan of every day.",
    )
    p.add_argument(
        "--force-recent-days",
        type=int,
        default=2,
        help="Always rescan the N newest day directories, even if their cache looks fresh. "
             "Only meaningful with --use-cache.",
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
    print(f"[config] use_cache     = {args.use_cache}")
    if args.use_cache:
        print(f"[config] force_recent  = {args.force_recent_days} day(s)")

    if args.use_cache:
        df = list_video_files_incremental(
            video_root,
            cache_dir,
            exts=tuple(args.exts),
            force_recent_days=args.force_recent_days,
        )
    else:
        df = build_video_info_df(video_root, tuple(args.exts))
    print(f"[videoinfo] collected {len(df)} files")

    out_path = args.outfile or os.path.join(cache_dir, "video_info_all.parquet")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"[videoinfo] wrote {len(df)} rows -> {out_path}")
    print("✅ Done.")


if __name__ == "__main__":
    main()