#!/usr/bin/env python3
"""
Scan .bbb files for one or more dates and remove or quarantine unreadable ones.

Three depths of check, each a superset of the one above it:
  (default)         framing of the first FrameContainer only
  --deep-check-bbb  every FrameContainer to a clean EOF (catches truncation)
  --touch-frames    also every frame's detection fields (catches the lazy
                    field-level corruption the other two call clean)

Example:
  python scan_and_remove_invalid_bbb_files.py --dates 20160819 20160820
  python scan_and_remove_invalid_bbb_files.py --dates 2016-08-19 --dry-run
  python scan_and_remove_invalid_bbb_files.py --dates 20160819 --deep-check-bbb
  python scan_and_remove_invalid_bbb_files.py --dates 20160819 --touch-frames --hours 22 23
  python scan_and_remove_invalid_bbb_files.py --dates 20160819 --touch-frames \
      --quarantine-dir /data/bbb_quarantine
"""

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

from bb_hpc import settings
from bb_hpc.src.fileinfo import (
    BbbCheckInconclusive,
    is_bbb_file_valid_basicmatch,
    is_bbb_file_valid_deep,
    is_bbb_file_valid_frames,
)


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


def _hour_dirs(day_dir: str, hours: list[str] | None) -> list[str]:
    """
    Which directories to scan under a day: the whole day, or just some HH buckets.

    Worth narrowing because --touch-frames costs a subprocess plus a full frame
    traversal per file, and a day holds a few thousand files. Note .bbb files are
    bucketed by END time, so a file starting at 22:59 lives under 23/ -- when
    chasing a specific hour, include the neighbouring bucket.
    """
    if not hours:
        return [day_dir]
    out = []
    for h in hours:
        d = os.path.join(day_dir, f"{int(h):02d}")
        if os.path.isdir(d):
            out.append(d)
        else:
            print(f"[skip] no such hour dir: {d}")
    return out


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


def _validate_file(args: tuple[str, str]) -> tuple[str, bool, str]:
    """
    Worker for parallel validation. Returns (path, is_valid, note).

    A non-empty note means the file could not be checked -- an unrecognised schema,
    not corruption. Those are reported and left alone: this tool deletes and moves
    files, so "I did not understand it" must never be collapsed into "it is bad".
    """
    path, mode = args
    if mode == "frames":
        try:
            return path, is_bbb_file_valid_frames(path), ""
        except BbbCheckInconclusive as e:
            return path, True, str(e)
    if mode == "deep":
        return path, is_bbb_file_valid_deep(path), ""
    return path, is_bbb_file_valid_basicmatch(path, check_read_file=True), ""


def _scan_day(
    day_dir: str,
    dry_run: bool,
    verbose: bool,
    deep_check_bbb: bool,
    num_workers: int = 1,
    hours: list[str] | None = None,
    quarantine_dir: str | None = None,
    touch_frames: bool = False,
) -> tuple[int, int, int, list[str]]:
    mode = "frames" if touch_frames else ("deep" if deep_check_bbb else "basic")
    files: list[str] = []
    for d in _hour_dirs(day_dir, hours):
        files.extend(_find_bbb_files(d))
    invalid: list[str] = []
    unchecked: list[str] = []

    if num_workers > 1 and len(files) > 1:
        # Parallel validation.
        #
        # ProcessPoolExecutor, not multiprocessing.Pool: the validators read capnp
        # data, and a malformed .bbb aborts the reading process outright. Pool does
        # not notice a dead worker -- imap_unordered just waits for a result that is
        # never coming, so the scan hangs silently on exactly the corrupt files it
        # exists to find. ProcessPoolExecutor raises BrokenProcessPool instead.
        #
        # The validators isolate their own capnp reads now, so this should not
        # trigger; it is the backstop for anything that still kills a worker.
        work_items = [(path, mode) for path in files]
        try:
            with ProcessPoolExecutor(max_workers=num_workers) as ex:
                for path, ok, note in ex.map(_validate_file, work_items, chunksize=10):
                    if note:
                        unchecked.append(path)
                        print(f"[unchecked] {note}")
                    elif not ok:
                        invalid.append(path)
                        if verbose:
                            print(f"[invalid] {path}")
        except BrokenProcessPool as e:
            raise RuntimeError(
                f"a validation worker died while scanning {day_dir} ({e}). "
                f"The scan is incomplete, so nothing was removed for this day. "
                f"Re-run with --num-workers 1 to find the file that killed it."
            ) from e
    else:
        # Sequential validation
        for path in files:
            _, ok, note = _validate_file((path, mode))
            if note:
                unchecked.append(path)
                print(f"[unchecked] {note}")
            elif not ok:
                invalid.append(path)
                if verbose:
                    print(f"[invalid] {path}")

    if unchecked:
        print(f"[unchecked] {len(unchecked)} file(s) could not be checked and were left "
              f"alone; they are NOT counted as invalid.")

    removed = 0
    if not dry_run:
        for path in invalid:
            try:
                if quarantine_dir:
                    # Keep the full YYYYMMDD/hour/div layout so the original location is
                    # recoverable. Use the whole date, not the day basename: 07/18 and
                    # 09/18 both end in "18" and would overwrite each other.
                    parts = os.path.normpath(day_dir).split(os.sep)[-3:]
                    rel = os.path.relpath(path, day_dir)
                    dest = os.path.join(quarantine_dir, "".join(parts), rel)
                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.move(path, dest)
                    print(f"[quarantined] {path} -> {dest}")
                else:
                    os.remove(path)
                removed += 1
            except Exception as e:
                verb = "quarantine" if quarantine_dir else "remove"
                print(f"[warn] failed to {verb} {path}: {e}")

    return len(files), len(invalid), removed, invalid


def parse_args():
    p = argparse.ArgumentParser(description="Scan .bbb files for dates and remove unreadable ones.")
    p.add_argument("--dates", nargs="+", required=True, help="Dates to scan (YYYYMMDD or YYYY-MM-DD).")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings paths to use (auto prefers *_local if they exist).")
    p.add_argument("--pipeline-root", default="", help="Override pipeline_root from settings.")
    p.add_argument("--dry-run", action="store_true", help="Scan only; do not delete files.")
    p.add_argument("--verbose", action="store_true", help="Print each invalid path.")
    p.add_argument("--touch-frames", action="store_true",
                   help="Strongest check: read every FrameContainer AND touch every frame's "
                        "detection fields. Catches lazy field-level corruption that the other "
                        "checks call clean (capnp only validates a pointer when the field is "
                        "accessed). Supersedes --deep-check-bbb. Costs a subprocess and a full "
                        "frame traversal per file, so pair it with --hours.")
    p.add_argument("--hours", nargs="+", default=None, metavar="HH",
                   help="Only scan these hour buckets under each date, e.g. --hours 22 23. "
                        "Files are bucketed by END time, so a file starting at 22:59 lives "
                        "under 23/ -- include the neighbouring bucket when chasing one hour.")
    p.add_argument("--quarantine-dir", default=None, metavar="PATH",
                   help="Move invalid files here instead of deleting them, preserving the "
                        "date/hour/div layout. MUST be outside pipeline_root: a non-numeric "
                        "directory at the repo root breaks every bb_binary read.")
    p.add_argument("--deep-check-bbb", action="store_true",
                   help="Read through all frames for each .bbb (slower, catches premature EOF).")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Number of parallel workers for validation (default: 8, use 1 for sequential).")
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

    quarantine_dir = args.quarantine_dir
    if quarantine_dir:
        # bb_binary's Repository._get_directory takes min()/max() over the repo root's
        # subdirectories and int()-parses the winner, so one non-numeric directory in
        # there breaks every read of the whole repo. Refuse before moving anything.
        qd = os.path.abspath(quarantine_dir)
        root = os.path.abspath(pipeline_root)
        if qd == root or qd.startswith(root + os.sep):
            print(f"ERROR: --quarantine-dir must be OUTSIDE pipeline_root.\n"
                  f"  quarantine: {qd}\n  pipeline_root: {root}\n"
                  f"A non-numeric directory inside the repo root breaks every bb_binary read.",
                  file=sys.stderr)
            sys.exit(2)
        if not args.dry_run:
            os.makedirs(qd, exist_ok=True)
        quarantine_dir = qd

    if args.touch_frames and args.deep_check_bbb:
        print("[note] --touch-frames supersedes --deep-check-bbb; using --touch-frames.")

    total_files = 0
    total_invalid = 0
    total_removed = 0
    failed_days: list[str] = []

    for d in dates:
        day_dir = os.path.join(pipeline_root, d[:4], d[4:6], d[6:8])
        if not os.path.exists(day_dir):
            print(f"[skip] missing day directory: {day_dir}")
            continue
        print(f"[scan] {day_dir}")
        try:
            n_files, n_invalid, n_removed, invalid_paths = _scan_day(
                day_dir, args.dry_run, args.verbose, args.deep_check_bbb, args.num_workers,
                hours=args.hours, quarantine_dir=quarantine_dir,
                touch_frames=args.touch_frames,
            )
        except RuntimeError as e:
            # A partial invalid-list must not reach the removal step, and the day
            # must not be reported as clean, so skip it and fail at the end.
            print(f"[error] {e}", file=sys.stderr)
            failed_days.append(d)
            continue
        total_files += n_files
        total_invalid += n_invalid
        total_removed += n_removed
        if args.dry_run:
            print(f"[day] files={n_files} invalid={n_invalid} (dry-run)")
            if invalid_paths:
                print("[invalid paths]")
                for p in invalid_paths:
                    print(p)
        else:
            verb = "quarantined" if quarantine_dir else "removed"
            print(f"[day] files={n_files} invalid={n_invalid} {verb}={n_removed}")

    if args.dry_run:
        print(f"[total] files={total_files} invalid={total_invalid} (dry-run)")
    else:
        verb = "quarantined" if quarantine_dir else "removed"
        print(f"[total] files={total_files} invalid={total_invalid} {verb}={total_removed}")

    if failed_days:
        print(f"[error] {len(failed_days)} day(s) did not finish scanning: "
              f"{', '.join(failed_days)}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
