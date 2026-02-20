#!/usr/bin/env python3
"""
Scan .bbb files for negative timestamp jumps (intra-container and inter-container)
and remove offending files.

Detects three kinds of problems:
  1. Intra-container: frames within a single .bbb have backward timestamps (NTP glitch).
  2. Inter-container overlap: a container's first timestamp < the previous container's
     last timestamp for the same camera.
  3. Duplicates: two .bbb files with identical (cam_id, fromTimestamp, toTimestamp),
     typically real copies instead of symlinks in legacy repos.

Example:
  python scan_and_fix_negative_timestamps_bbb_files.py --dates 20160801 20160803
  python scan_and_fix_negative_timestamps_bbb_files.py --dates 20160801 --dry-run --verbose
  python scan_and_fix_negative_timestamps_bbb_files.py --dates 20160801 --num-workers 8
"""

import argparse
import math
import os
import shlex
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

from bb_hpc import settings


# ---------------------------------------------------------------------------
# Helpers (same patterns as scan_and_remove_invalid_bbb_files.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Timestamp checking
# ---------------------------------------------------------------------------

def _check_one_file(path: str) -> dict | None:
    """Load a .bbb file and extract timestamp metadata.

    Returns a dict with keys: path, cam_id, from_ts, to_ts, first_frame_ts,
    last_frame_ts, n_frames, has_intra_negative.
    Returns None if the file cannot be read.
    """
    import bb_binary
    try:
        fc = bb_binary.load_frame_container(path)
    except Exception:
        return None  # unreadable files are handled by the other scan script

    timestamps = [f.timestamp for f in fc.frames]
    if not timestamps:
        return None

    has_negative = False
    for i in range(1, len(timestamps)):
        if timestamps[i] <= timestamps[i - 1]:
            has_negative = True
            break

    return dict(
        path=path,
        cam_id=int(fc.camId),
        from_ts=float(fc.fromTimestamp),
        to_ts=float(fc.toTimestamp),
        first_frame_ts=timestamps[0],
        last_frame_ts=timestamps[-1],
        n_frames=len(timestamps),
        has_intra_negative=has_negative,
    )


def _find_dangling_symlinks(removed_path: str) -> list[str]:
    """Find symlinks in adjacent 20-minute bucket directories that point to the
    removed file (by matching the filename)."""
    removed = Path(removed_path)
    fname = removed.name
    bucket_dir = removed.parent  # .../HH/MM

    dangling = []
    try:
        hour_dir = bucket_dir.parent  # .../HH
        for sibling in hour_dir.iterdir():
            if not sibling.is_dir():
                continue
            candidate = sibling / fname
            if candidate != removed and candidate.is_symlink():
                # Check if the symlink target matches the removed file
                try:
                    target = candidate.resolve()
                    if target == removed.resolve() or not target.exists():
                        dangling.append(str(candidate))
                except Exception:
                    dangling.append(str(candidate))
    except Exception:
        pass
    return dangling


def _scan_day(
    day_dir: str,
    dry_run: bool,
    verbose: bool,
    num_workers: int = 1,
) -> tuple[int, int, int, list[tuple[str, str]]]:
    """Scan a single day directory for timestamp problems.

    Returns (n_files, n_flagged, n_removed, flagged_list)
    where flagged_list is [(path, reason), ...].
    """
    files = _find_bbb_files(day_dir)
    if not files:
        return 0, 0, 0, []

    # Step 1: read all .bbb files (parallel)
    if num_workers > 1 and len(files) > 1:
        with Pool(processes=num_workers) as pool:
            results = list(pool.imap_unordered(_check_one_file, files, chunksize=10))
    else:
        results = [_check_one_file(f) for f in files]

    # Filter out unreadable files
    infos = [r for r in results if r is not None]

    # Step 2: group by camera, sort by from_ts
    by_cam: dict[int, list[dict]] = defaultdict(list)
    for info in infos:
        by_cam[info["cam_id"]].append(info)
    for cam_id in by_cam:
        by_cam[cam_id].sort(key=lambda x: x["from_ts"])

    # Step 3: detect problems
    flagged: list[tuple[str, str]] = []  # (path, reason)
    flagged_paths: set[str] = set()

    for cam_id, cam_infos in sorted(by_cam.items()):
        seen_keys: dict[tuple, str] = {}  # (from_ts, to_ts) -> first path
        last_end_ts: float | None = None

        for info in cam_infos:
            path = info["path"]
            reasons = []

            # Check 1: intra-container negative timestamps
            if info["has_intra_negative"]:
                reasons.append("intra-container negative timestamp jump")

            # Check 2: duplicate detection
            key = (info["from_ts"], info["to_ts"])
            if key in seen_keys:
                reasons.append(f"duplicate of {os.path.basename(seen_keys[key])}")
            else:
                seen_keys[key] = path

            # Check 3: inter-container overlap
            if last_end_ts is not None and info["first_frame_ts"] < last_end_ts:
                gap = info["first_frame_ts"] - last_end_ts
                reasons.append(f"inter-container overlap ({gap:.1f}s)")

            if reasons:
                flagged.append((path, "; ".join(reasons)))
                flagged_paths.add(path)
                if verbose:
                    print(f"  [flagged] cam={cam_id} {os.path.basename(path)}: {'; '.join(reasons)}")
            else:
                # Only advance last_end_ts for non-flagged files
                last_end_ts = info["last_frame_ts"]

    # Step 4: remove flagged files (unless dry-run)
    removed = 0
    if not dry_run:
        for path, reason in flagged:
            try:
                os.remove(path)
                removed += 1
                # Also clean up dangling symlinks
                for sym in _find_dangling_symlinks(path):
                    try:
                        os.remove(sym)
                        if verbose:
                            print(f"  [removed symlink] {sym}")
                    except Exception as e:
                        print(f"  [warn] failed to remove symlink {sym}: {e}")
            except Exception as e:
                print(f"  [warn] failed to remove {path}: {e}")

    return len(files), len(flagged), removed, flagged


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Scan .bbb files for negative timestamp jumps and remove offending files."
    )
    p.add_argument("--dates", nargs="+", required=True,
                   help="Dates to scan (YYYYMMDD or YYYY-MM-DD).")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings paths to use (auto prefers *_local if they exist).")
    p.add_argument("--pipeline-root", default="",
                   help="Override pipeline_root from settings.")
    p.add_argument("--dry-run", action="store_true",
                   help="Scan only; do not delete files.")
    p.add_argument("--verbose", action="store_true",
                   help="Print each flagged path and its reason.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of parallel workers for reading .bbb files (default: 8).")
    return p.parse_args()


def main():
    args = parse_args()

    pipeline_root = args.pipeline_root or _pick_pipeline_root(
        None if args.paths == "auto" else args.paths
    )
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
    total_flagged = 0
    total_removed = 0

    for d in dates:
        day_dir = os.path.join(pipeline_root, d[:4], d[4:6], d[6:8])
        if not os.path.exists(day_dir):
            print(f"[skip] missing day directory: {day_dir}")
            continue
        print(f"[scan] {day_dir}")
        n_files, n_flagged, n_removed, flagged_list = _scan_day(
            day_dir, args.dry_run, args.verbose, args.num_workers
        )
        total_files += n_files
        total_flagged += n_flagged
        total_removed += n_removed
        if args.dry_run:
            print(f"[day] files={n_files} flagged={n_flagged} (dry-run)")
            if flagged_list and not args.verbose:
                for path, reason in flagged_list:
                    print(f"  {path}: {reason}")
        else:
            print(f"[day] files={n_files} flagged={n_flagged} removed={n_removed}")

    if args.dry_run:
        print(f"\n[total] files={total_files} flagged={total_flagged} (dry-run)")
    else:
        print(f"\n[total] files={total_files} flagged={total_flagged} removed={total_removed}")


if __name__ == "__main__":
    main()
