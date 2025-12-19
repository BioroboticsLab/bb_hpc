#!/usr/bin/env python3
"""
Scan a pipeline repository for .bbb files whose expected cross-boundary
entries are missing (because symlinks are not supported) and fill them in
by copying the primary file.

- Dates: optional list (YYYYMMDD or YYYY-MM-DD). If omitted, scans all days.
- Dry run: use --dry-run to only print what would be copied.
- Uses pipeline_root from bb_hpc.settings unless overridden.
"""

import argparse
import os
import shutil
import sys
from datetime import timedelta
from pathlib import Path

from bb_hpc import settings
from bb_binary.parsing import parse_video_fname
from bb_binary.repository import Repository


def _normalize_date(d: str) -> str:
    s = d.strip().replace("-", "").replace("/", "")
    if len(s) != 8 or not s.isdigit():
        raise ValueError(f"Invalid date {d!r}; expected YYYYMMDD or YYYY-MM-DD.")
    return s


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


def expected_dirs(repo: Repository, begin, end) -> set[str]:
    """Return all repo directories this file should live in (20-min buckets)."""
    dirs = set()
    dt = begin
    step = timedelta(minutes=repo.minute_step)
    # safety cap: in case of bad metadata, stop after 200 steps
    for _ in range(200):
        dirs.add(repo._path_for_dt(dt, abs=True))
        if repo._path_for_dt(dt, abs=True) == repo._path_for_dt(end, abs=True):
            break
        dt = dt + step
    return dirs


def process_day(day_dir: Path, repo: Repository, dry_run: bool) -> tuple[int, int]:
    """Returns (missing_created, files_seen)."""
    files_seen = 0
    missing_created = 0
    for root, _dirs, files in os.walk(day_dir):
        for name in files:
            if not name.endswith(".bbb"):
                continue
            files_seen += 1
            src = Path(root) / name
            try:
                cam, begin, end = parse_video_fname(name, format="bbb")
            except Exception as e:
                print(f"[skip] failed to parse {src}: {e}")
                continue

            for target_dir in expected_dirs(repo, begin, end):
                target_path = Path(target_dir) / name
                if target_path.samefile(src) if target_path.exists() else False:
                    continue
                if target_path.exists():
                    continue
                print(f"[missing] would add {target_path} (source {src})")
                if not dry_run:
                    target_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, target_path)
                    missing_created += 1
    return missing_created, files_seen


def parse_args():
    p = argparse.ArgumentParser(description="Fill in missing .bbb cross-boundary copies when symlinks are unsupported.")
    p.add_argument("--dates", nargs="*", help="Dates to scan (YYYYMMDD or YYYY-MM-DD). If omitted, scan all days.")
    p.add_argument("--pipeline-root", default="", help="Override pipeline_root from settings.")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings paths to use (auto prefers *_local if they exist).")
    p.add_argument("--dry-run", action="store_true", help="Print planned copies without writing.")
    return p.parse_args()


def main():
    args = parse_args()
    pipeline_root = args.pipeline_root or _pick_pipeline_root(None if args.paths == "auto" else args.paths)
    if not pipeline_root or not os.path.exists(pipeline_root):
        print("ERROR: pipeline_root not set or does not exist.", file=sys.stderr)
        sys.exit(2)

    # Build date list
    dates = []
    if args.dates:
        for d in args.dates:
            dates.append(_normalize_date(d))
    else:
        # scan all YYYY/MM/DD under pipeline_root
        for year in sorted(p for p in os.listdir(pipeline_root) if len(p) == 4 and p.isdigit()):
            ydir = Path(pipeline_root) / year
            if not ydir.is_dir():
                continue
            for month in sorted(p for p in os.listdir(ydir) if len(p) == 2 and p.isdigit()):
                mdir = ydir / month
                if not mdir.is_dir():
                    continue
                for day in sorted(p for p in os.listdir(mdir) if len(p) == 2 and p.isdigit()):
                    dates.append(f"{year}{month}{day}")

    repo = Repository(pipeline_root)

    total_missing = 0
    total_files = 0
    for d in dates:
        day_dir = Path(pipeline_root) / d[:4] / d[4:6] / d[6:8]
        if not day_dir.exists():
            print(f"[skip] missing day directory: {day_dir}")
            continue
        print(f"[scan] {day_dir}")
        missing, seen = process_day(day_dir, repo, args.dry_run)
        total_missing += missing
        total_files += seen
        suffix = "(dry-run)" if args.dry_run else ""
        print(f"[day] files={seen} added={missing} {suffix}".strip())

    suffix = "(dry-run)" if args.dry_run else ""
    print(f"[total] files={total_files} added={total_missing} {suffix}".strip())


if __name__ == "__main__":
    main()
