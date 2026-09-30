#!/usr/bin/env python3
"""
Find save_detect outputs (data_alldetections/*.parquet) that must be redone, and
print the commands to redo them.

progress_report and the submitters decide 'done' by mtime alone, so an empty or
cut-short parquet written by the old save_detect error handling looks finished and
is never scheduled again. This opens each 'done' output's parquet footer and compares
it with the .bbb catalog. Statuses are described in bb_hpc/src/save_detect_check.py.

Read-only, like progress_report: it never moves, deletes or submits anything. It
writes units.csv, flagged_files.txt, redo_dates.txt and commands.sh to --out-dir.

Examples
--------
    # every date in the catalogs, resultdir from bb_hpc.settings
    python -m bb_hpc.check_save_detect

    # another site / an explicit results dir
    python -m bb_hpc.check_save_detect --resultdir /mnt/trove/beesbook2026

    # a date range
    python -m bb_hpc.check_save_detect --since 20260620 --until 20260720
"""
import argparse
import os
import sys

try:
    import pandas as pd
    from bb_hpc.src.progress import BACKENDS, filter_dates, resolve_paths
    from bb_hpc.src.save_detect_check import (
        FLAG_STATUSES,
        bbb_dates,
        by_day,
        check_save_detect,
        fix_commands,
        write_outputs,
    )
except Exception as e:  # pragma: no cover
    print(f"ERROR: could not import bb_hpc modules: {e}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Find save_detect outputs that must be redone, and print the commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--resultdir", default=None,
                   help="Results root holding bbb_fileinfo/ and data_alldetections/. "
                        "Default: resultdir from bb_hpc.settings.")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings path set to use when --resultdir is not given.")
    sel = p.add_argument_group("date selection (default: every date in the .bbb catalog)")
    sel.add_argument("--dates", nargs="+", default=None, help="Explicit YYYYMMDD dates.")
    sel.add_argument("--since", default=None, help="Earliest YYYYMMDD (inclusive).")
    sel.add_argument("--until", default=None, help="Latest YYYYMMDD (inclusive).")
    p.add_argument("--tail-tol-min", type=float, default=3.0,
                   help="Check an output against its .bbb frames when its last detection is more than "
                        "this many minutes before the end of the .bbb coverage of its window.")
    p.add_argument("--no-verify-bbb", action="store_true",
                   help="Do not read .bbb frames after the last detection; flag every early-ending "
                        "output as truncated (fast, but sparse or dark hours become false positives).")
    p.add_argument("--interval-hours", type=int, default=1,
                   help="Window size (must match the submitters).")
    p.add_argument("--backend", choices=list(BACKENDS), default="k8s",
                   help="Which running_<backend>.save_detect_submit the printed commands target.")
    p.add_argument("--workers", type=int, default=16, help="Threads for reading parquet footers.")
    p.add_argument("--out-dir", default=None,
                   help="Where to write the report files. Default: <resultdir>/bbb_fileinfo/save_detect_check/")
    p.add_argument("--no-save", action="store_true", help="Print only; write nothing.")
    return p.parse_args()


def main():
    args = parse_args()
    settings_resultdir = resolve_paths(None if args.paths == "auto" else args.paths)["resultdir"]
    resultdir = args.resultdir or settings_resultdir
    if not resultdir:
        print("ERROR: pass --resultdir, or set resultdir in bb_hpc.settings.", file=sys.stderr)
        sys.exit(2)
    resultdir = os.path.abspath(resultdir)

    if args.dates:
        dates = sorted(set(args.dates))
    else:
        dates = filter_dates(bbb_dates(resultdir), since=args.since, until=args.until)
    if not dates:
        print(f"ERROR: no dates found in {resultdir}/bbb_fileinfo; run `python -m bb_hpc.get_fileinfo` "
              "first, or pass --dates.", file=sys.stderr)
        sys.exit(2)

    print(f"[check_save_detect] {resultdir}: {len(dates)} dates {dates[0]}..{dates[-1]}", flush=True)
    units = check_save_detect(resultdir, dates, tail_tol_min=args.tail_tol_min,
                              interval_hours=args.interval_hours, max_workers=args.workers,
                              verify_bbb=not args.no_verify_bbb)
    if units.empty:
        print("No save_detect units: the .bbb catalog has nothing on these dates.")
        return

    print("\nUnits by status (one unit = one camera-hour that has .bbb data):")
    print(units["status"].value_counts().to_string())
    if (units["status"] == "ok_empty_tail").any():
        print("(ok_empty_tail: detections end early, but the .bbb frames after them are empty too, "
              "e.g. camera dark or covered. Complete; nothing to redo.)")

    t = by_day(units)
    if not t.empty:
        print("\nDays with something to redo:")
        print(t.to_string())

    flagged = units[units["status"].isin(FLAG_STATUSES)]
    if len(flagged):
        print(f"\n{len(flagged)} output(s) are counted as done by the submitter but must be redone "
              "(move them aside first):")
        cols = ["cam_id", "from_dt", "status", "n_rows", "last_detection", "bbb_coverage_end",
                "tail_gap_min", "bbb_dets_after_last"]
        print(flagged[cols].to_string(index=False, max_rows=40))

    out_dir = args.out_dir or os.path.join(resultdir, "bbb_fileinfo", "save_detect_check")
    paths_flag = "" if args.paths == "auto" else f" --paths {args.paths}"
    commands = fix_commands(units, resultdir, out_dir, backend=args.backend, paths_flag=paths_flag)
    if not commands:
        print("\nNothing to redo.")
    else:
        if settings_resultdir and os.path.abspath(settings_resultdir) != resultdir:
            commands.insert(1, f"# NOTE: get_fileinfo and save_detect_submit use bb_hpc.settings "
                               f"(resultdir={settings_resultdir}), not {resultdir}")
        print("\nTo fix:\n")
        print("\n".join(commands))

    if not args.no_save:
        write_outputs(units, out_dir, commands)
        print(f"\n[saved] {out_dir}/ (units.csv, flagged_files.txt, redo_dates.txt, commands.sh)")


if __name__ == "__main__":
    main()
