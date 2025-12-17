#!/usr/bin/env python3
"""
Scan .bbb files for one or more dates and remove files that cannot be read.

Example:
  python scan_and_remove_invalid_bbb_files.py --dates 20160819 20160820
  python scan_and_remove_invalid_bbb_files.py --dates 2016-08-19 --dry-run
"""

import argparse
import os
import shlex
import subprocess
import sys
from typing import Iterable

from bb_hpc import settings
from bb_hpc.src.fileinfo import is_bbb_file_valid_basicmatch


def _pick_pipeline_root(prefer: str | None = None) -> str:
    pr_local = getattr(settings, "pipeline_root_local", "")
    pr_hpc = getattr(settings, "pipeline_root_hpc", "")

    def _exists(p: str) -> bool:
        return bool(p) and os.path.exists(p)

    if prefer == "local" and _exists(pr_local):
        return pr_local
    if prefer == "hpc" and _exists(pr_hpc):
        return pr_hpc

    if _exists(pr_local):
        return pr_local
    if _exists(pr_hpc):
        return pr_hpc
    return pr_local or pr_hpc


def _normalize_date(d: str) -> str:
    s = d.strip().replace("/", "").replace("-", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Invalid date {d!r}; expected YYYYMMDD or YYYY-MM-DD.")
    return s


def _find_bbb_files(day_dir: str) -> list[str]:
    try:
        cmd = f"find {shlex.quote(day_dir)} -type f -name '*.bbb' -print"
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        return [line.decode("utf-8", "replace").strip() for line in out.splitlines() if line.strip()]
    except Exception:
        bbb_files: list[str] = []
        for root, _dirs, files in os.walk(day_dir):
            for name in files:
                if name.endswith(".bbb"):
                    bbb_files.append(os.path.join(root, name))
        return bbb_files


def _scan_day(day_dir: str, dry_run: bool, verbose: bool) -> tuple[int, int, int]:
    files = _find_bbb_files(day_dir)
    invalid: list[str] = []
    for path in files:
        if not is_bbb_file_valid_basicmatch(path, check_read_file=True):
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

    return len(files), len(invalid), removed


def parse_args():
    p = argparse.ArgumentParser(description="Scan .bbb files for dates and remove unreadable ones.")
    p.add_argument("--dates", nargs="+", required=True, help="Dates to scan (YYYYMMDD or YYYY-MM-DD).")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings paths to use (auto prefers *_local if they exist).")
    p.add_argument("--pipeline-root", default="", help="Override pipeline_root from settings.")
    p.add_argument("--dry-run", action="store_true", help="Scan only; do not delete files.")
    p.add_argument("--verbose", action="store_true", help="Print each invalid path.")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline_root = args.pipeline_root or _pick_pipeline_root(None if args.paths == "auto" else args.paths)
    if not pipeline_root or not os.path.exists(pipeline_root):
        print("ERROR: pipeline_root not set or does not exist.", file=sys.stderr)
        sys.exit(2)

    dates = []
    for d in args.dates:
        try:
            dates.append(_normalize_date(d))
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(2)

    total_files = 0
    total_invalid = 0
    total_removed = 0

    for d in dates:
        day_dir = os.path.join(pipeline_root, d[:4], d[4:6], d[6:8])
        if not os.path.exists(day_dir):
            print(f"[skip] missing day directory: {day_dir}")
            continue
        print(f"[scan] {day_dir}")
        n_files, n_invalid, n_removed = _scan_day(day_dir, args.dry_run, args.verbose)
        total_files += n_files
        total_invalid += n_invalid
        total_removed += n_removed
        if args.dry_run:
            print(f"[day] files={n_files} invalid={n_invalid} (dry-run)")
        else:
            print(f"[day] files={n_files} invalid={n_invalid} removed={n_removed}")

    if args.dry_run:
        print(f"[total] files={total_files} invalid={total_invalid} (dry-run)")
    else:
        print(f"[total] files={total_files} invalid={total_invalid} removed={total_removed}")


if __name__ == "__main__":
    main()
