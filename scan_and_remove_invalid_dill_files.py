#!/usr/bin/env python3
"""
Scan tracking .dill files for one or more dates and remove files that are
0 bytes or whose dill stream is structurally broken (silent-failure zombies
left behind when tracking exited mid-write).

A "valid" .dill is a non-empty file whose pickled batches can be read all
the way to a clean terminal EOF. Files that are 0 bytes, truncated, or
contain a partial pickled record are deleted.

After running this script, re-run get_fileinfo.py to refresh
save_tracking_outinfo.parquet so the deleted hours are picked up on the
next tracking run.

Examples:
  python scan_and_remove_invalid_dill_files.py --dates 20251001 --dry-run
  python scan_and_remove_invalid_dill_files.py --dates 20251001 --verbose
  python scan_and_remove_invalid_dill_files.py --dates 20251001 20251002 --num-workers 4
"""

import argparse
import glob
import os
import sys
from multiprocessing import Pool

from bb_hpc import settings
from bb_hpc.src.fileinfo import is_dill_file_valid_deep


def _pick_resultdir(prefer: str | None = None) -> str:
    rd_local = getattr(settings, "resultdir_local", "")
    rd_hpc = getattr(settings, "resultdir_hpc", "")

    def _exists(p: str) -> bool:
        return bool(p) and os.path.exists(p)

    if prefer == "local" and _exists(rd_local):
        return rd_local
    if prefer == "hpc" and _exists(rd_hpc):
        return rd_hpc

    if _exists(rd_local):
        return rd_local
    if _exists(rd_hpc):
        return rd_hpc
    return rd_local or rd_hpc


def _normalize_date(d: str) -> str:
    s = d.strip().replace("/", "").replace("-", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Invalid date {d!r}; expected YYYYMMDD or YYYY-MM-DD.")
    return s


def _find_dill_files(track_dir: str, date_yyyymmdd: str) -> list[str]:
    """
    Tracking outputs live in a flat directory <RESULTDIR>/data_tracked/ with
    filenames like Cam_<id>_<YYYY-MM-DD>T<HH_MM_SS>Z--...dill. Filter to the
    given date. Excludes any in-flight .dill.tmp files.
    """
    date_iso = f"{date_yyyymmdd[:4]}-{date_yyyymmdd[4:6]}-{date_yyyymmdd[6:8]}"
    pattern = os.path.join(track_dir, f"Cam_*_{date_iso}T*.dill")
    return sorted(p for p in glob.glob(pattern) if not p.endswith(".dill.tmp"))


def _validate_file(args: tuple[str, float]) -> tuple[str, bool]:
    path, max_end_gap_seconds = args
    return path, is_dill_file_valid_deep(path, max_end_gap_seconds=max_end_gap_seconds)


def _scan_day(
    track_dir: str,
    date_yyyymmdd: str,
    dry_run: bool,
    verbose: bool,
    num_workers: int = 1,
    max_end_gap_seconds: float = 300.0,
) -> tuple[int, int, int, list[str]]:
    files = _find_dill_files(track_dir, date_yyyymmdd)
    invalid: list[str] = []

    if num_workers > 1 and len(files) > 1:
        work_items = [(path, max_end_gap_seconds) for path in files]
        with Pool(processes=num_workers) as pool:
            for path, ok in pool.imap_unordered(_validate_file, work_items, chunksize=4):
                if not ok:
                    invalid.append(path)
                    if verbose:
                        print(f"[invalid] {path}")
    else:
        for path in files:
            ok = is_dill_file_valid_deep(path, max_end_gap_seconds=max_end_gap_seconds)
            if not ok:
                invalid.append(path)
                if verbose:
                    print(f"[invalid] {path}")

    removed = 0
    if not dry_run:
        for path in invalid:
            try:
                os.remove(path)
                removed += 1
            except Exception as e:
                print(f"[warn] failed to remove {path}: {e}")

    return len(files), len(invalid), removed, invalid


def parse_args():
    p = argparse.ArgumentParser(description="Scan tracking .dill files and remove invalid ones.")
    p.add_argument("--dates", nargs="+", required=True, help="Dates to scan (YYYYMMDD or YYYY-MM-DD).")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings paths to use (auto prefers *_local if they exist).")
    p.add_argument("--resultdir", default="", help="Override resultdir from settings.")
    p.add_argument("--dry-run", action="store_true", help="Scan only; do not delete files.")
    p.add_argument("--verbose", action="store_true", help="Print each invalid path.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="Number of parallel workers for validation (default: 4).")
    p.add_argument("--max-end-gap-seconds", type=float, default=300.0,
                   help="Flag a .dill as invalid if its last detection is more than this "
                        "many seconds before the file's nominal end time. Catches "
                        "cleanly-written-but-short files left by tracking runs that "
                        "stopped early (default: 300).")
    return p.parse_args()


def main():
    args = parse_args()

    resultdir = args.resultdir or _pick_resultdir(None if args.paths == "auto" else args.paths)
    if not resultdir or not os.path.exists(resultdir):
        print("ERROR: resultdir not set or does not exist.", file=sys.stderr)
        sys.exit(2)

    track_dir = os.path.join(resultdir, "data_tracked")
    if not os.path.exists(track_dir):
        print(f"ERROR: tracking directory does not exist: {track_dir}", file=sys.stderr)
        sys.exit(2)

    dates = []
    for d in args.dates:
        try:
            dates.append(_normalize_date(d))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

    print(f"[scan] {track_dir}")

    total_files = 0
    total_invalid = 0
    total_removed = 0

    for d in dates:
        n_files, n_invalid, n_removed, invalid_paths = _scan_day(
            track_dir, d, args.dry_run, args.verbose, args.num_workers,
            args.max_end_gap_seconds,
        )
        total_files += n_files
        total_invalid += n_invalid
        total_removed += n_removed
        if args.dry_run:
            print(f"[{d}] files={n_files} invalid={n_invalid} (dry-run)")
            if invalid_paths:
                print("[invalid paths]")
                for p in invalid_paths:
                    print(p)
        else:
            print(f"[{d}] files={n_files} invalid={n_invalid} removed={n_removed}")

    if args.dry_run:
        print(f"[total] files={total_files} invalid={total_invalid} (dry-run)")
    else:
        print(f"[total] files={total_files} invalid={total_invalid} removed={total_removed}")
        if total_removed > 0:
            print("Re-run get_fileinfo.py so save_tracking_outinfo.parquet reflects the deletions.")


if __name__ == "__main__":
    main()
