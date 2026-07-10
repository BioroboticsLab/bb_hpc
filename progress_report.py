#!/usr/bin/env python3
"""
Report what the BeesBook pipeline has processed, and print the commands to fill the gaps.

Reads the fileinfo catalogs under <resultdir>/bbb_fileinfo/, classifies every work
unit of every stage (detect, save_detect, tracking, frame_extract, background, rpi)
as done / pending / skipped, prints a one-screen report, and saves an atomic
snapshot the companion notebook can load without recomputing anything.

Completion is decided by the SAME predicates the submitters use (bb_hpc.src.generate),
so a unit reported pending here is exactly a unit `<stage>_submit` would schedule.

Examples
--------
    # fast: read whatever the fileinfo cron produced
    python -m bb_hpc.progress_report

    # refresh the (incremental) video catalog first
    python -m bb_hpc.progress_report --refresh videos

    # rebuild every catalog, then report
    python -m bb_hpc.progress_report --refresh all

    # just the last week, only the comb stages, as markdown
    python -m bb_hpc.progress_report --last-days 7 --stages frame_extract background --markdown

This script PRINTS resubmit commands (and writes them to commands.sh). It never submits.
"""

import argparse
import os
import sys

try:
    import pandas as pd
except Exception as e:  # pragma: no cover
    print(f"ERROR: pandas is required: {e}", file=sys.stderr)
    sys.exit(1)

try:
    from bb_hpc import settings
    from bb_hpc.src.progress import (
        BACKENDS,
        STAGES,
        build_report,
        catalog_dates,
        default_out_dir,
        filter_dates,
        resolve_paths,
    )
except Exception as e:  # pragma: no cover
    print(f"ERROR: could not import bb_hpc modules: {e}", file=sys.stderr)
    sys.exit(1)


def parse_args():
    p = argparse.ArgumentParser(
        description="Report pipeline progress and emit resubmit commands.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sel = p.add_argument_group("date selection (default: every date in the catalogs)")
    sel.add_argument("--dates", nargs="+", default=None, help="Explicit YYYYMMDD dates.")
    sel.add_argument("--since", default=None, help="Earliest YYYYMMDD (inclusive).")
    sel.add_argument("--until", default=None, help="Latest YYYYMMDD (inclusive).")
    sel.add_argument("--last-days", type=int, default=None,
                     help="Only the N most recent dates present in the catalogs.")

    p.add_argument("--stages", nargs="+", choices=list(STAGES), default=list(STAGES),
                   help="Which stages to report on.")
    p.add_argument("--backend", choices=list(BACKENDS), default="k8s",
                   help="Which running_<backend>.*_submit module the printed commands target.")
    p.add_argument("--refresh", choices=["none", "videos", "fileinfo", "all"], default="none",
                   help="Rebuild catalogs before reporting. 'videos' is incremental and cheap.")
    p.add_argument("--paths", choices=["auto", "local", "hpc"], default="auto",
                   help="Which settings path set to use (auto prefers *_local when it exists).")
    p.add_argument("--interval-hours", type=int, default=1,
                   help="Window size for save_detect/tracking units (must match the submitters).")
    p.add_argument("--rpi-clahe", action=argparse.BooleanOptionalAction, default=None,
                   help="Measure RPi done-ness against CLAHE detections. "
                        "Default: settings.rpi_detect_settings['use_clahe'].")

    out = p.add_argument_group("output")
    out.add_argument("--out-dir", default=None,
                     help="Snapshot directory. Default: <resultdir>/bbb_fileinfo/progress/")
    out.add_argument("--no-save", action="store_true", help="Print only; write nothing.")
    out.add_argument("--markdown", action="store_true", help="Render the console report as markdown.")
    out.add_argument("--csv", default=None, help="Also write the per-day tables to this CSV path.")
    out.add_argument("--max-days", type=int, default=14,
                     help="Per stage, how many pending days to list before truncating.")
    out.add_argument("--exit-nonzero-if-pending", action="store_true",
                     help="Exit 1 when any unit is pending, for cron/CI alerting. "
                          "Off by default: mid-season, pending work is the normal state.")
    return p.parse_args()


def _refresh(what: str, paths: dict) -> None:
    """Rebuild catalogs in-process (same code paths as get_videoinfo / get_fileinfo)."""
    cache_dir = os.path.join(paths["resultdir"], "bbb_fileinfo")
    os.makedirs(cache_dir, exist_ok=True)

    if what in ("videos", "all"):
        from bb_hpc.src.fileinfo import list_video_files_incremental
        print(f"[refresh] video catalog (incremental) from {paths['videodir']}")
        df = list_video_files_incremental(paths["videodir"], cache_dir)
        out_path = os.path.join(cache_dir, "video_info_all.parquet")
        df.to_parquet(out_path, index=False)
        print(f"[refresh] wrote {len(df):,} rows -> {out_path}")

    if what in ("fileinfo", "all"):
        from bb_hpc.get_fileinfo import (
            build_bbb_info_parquet,
            build_outinfo_parquets,
            build_rpi_info_parquet,
        )
        print(f"[refresh] bbb catalog from {paths['pipeline_root']}")
        build_bbb_info_parquet(paths["pipeline_root"], cache_dir, recalc=False)
        print(f"[refresh] output catalogs from {paths['resultdir']}")
        build_outinfo_parquets(paths["resultdir"], cache_dir)
        pi_root = paths.get("pi_videodir")
        if pi_root and os.path.exists(pi_root):
            print(f"[refresh] rpi catalog from {pi_root}")
            build_rpi_info_parquet(pi_root, cache_dir)
        else:
            print("[refresh] pi_videodir missing; skipping rpi catalog", file=sys.stderr)


def _select_dates(args, resultdir: str) -> list[str]:
    if args.dates:
        return sorted(set(args.dates))
    dates = catalog_dates(resultdir)
    return filter_dates(dates, since=args.since, until=args.until, last_days=args.last_days)


def main():
    args = parse_args()

    paths = resolve_paths(None if args.paths == "auto" else args.paths)
    if not paths["resultdir"]:
        print("ERROR: resultdir is not set in bb_hpc.settings.", file=sys.stderr)
        sys.exit(2)

    if args.refresh != "none":
        _refresh(args.refresh, paths)

    dates = _select_dates(args, paths["resultdir"])
    if not dates:
        print("ERROR: no dates found. Run `python -m bb_hpc.get_videoinfo` and "
              "`python -m bb_hpc.get_fileinfo` first, or pass --dates.", file=sys.stderr)
        sys.exit(2)

    rpi_clahe = args.rpi_clahe
    if rpi_clahe is None:
        rpi_clahe = bool(getattr(settings, "rpi_detect_settings", {}).get("use_clahe", True))

    rep = build_report(
        **paths,
        dates=dates,
        stages=args.stages,
        backend=args.backend,
        frame_extract_settings=getattr(settings, "frame_extract_settings", {}),
        background_settings=getattr(settings, "background_settings", {}),
        rpi_clahe=rpi_clahe,
        interval_hours=args.interval_hours,
    )

    print(rep.render(markdown=args.markdown, max_days=args.max_days))

    if args.csv:
        frames = []
        for name, sp in rep.stages.items():
            if not sp.available or sp.by_day.empty:
                continue
            frames.append(sp.by_day.assign(stage=name))
        if frames:
            df = pd.concat(frames, ignore_index=True)
            df = df[["stage"] + [c for c in df.columns if c != "stage"]]
            df.to_csv(args.csv, index=False)
            print(f"\n[csv] wrote {len(df)} rows -> {args.csv}")

    if not args.no_save:
        out_dir = args.out_dir or default_out_dir(paths["resultdir"])
        latest = rep.save(out_dir)
        print(f"\n[snapshot] {latest}")
        print(f"[history]  {os.path.join(out_dir, 'history.jsonl')}")
        print("[notebook] from bb_hpc.src.progress import load_latest; rep = load_latest()")

    if args.exit_nonzero_if_pending:
        pending = sum(sp.totals()["pending"] for sp in rep.stages.values() if sp.available)
        sys.exit(1 if pending else 0)


if __name__ == "__main__":
    main()
