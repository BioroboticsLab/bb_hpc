#!/usr/bin/env python3
"""
Generate/refresh fileinfo caches for BeesBook:

  1) <RESULTDIR>/bbb_fileinfo/bbb_info_YYYYMMDD.parquet
     - catalog of all .bbb files in the repository (time windows + mtime)

  2) <RESULTDIR>/bbb_fileinfo/save_detect_outinfo.parquet
     - catalog of existing daily save-detect outputs (.parquet)

  3) <RESULTDIR>/bbb_fileinfo/save_tracking_outinfo.parquet
     - catalog of existing tracking outputs (.dill)

  4) <RESULTDIR>/bbb_fileinfo/rpi_info_YYYYMMDD.parquet
     - catalog of raw RPi videos and whether CLAHE/no-CLAHE detections exist

This script is platform-agnostic:
- Local, Slurm, K8s, or Docker — as long as `bb_hpc.settings` is importable
  and the directories are mounted/accessible, it will work.

CLI options let you pick which caches to (re)build and which path set to use.
"""

import os
import sys
import argparse
from datetime import datetime, timezone


# --- import settings + helpers from your package ---
try:
    from bb_hpc import settings
    from bb_hpc.src.fileinfo import (
        list_bbb_files,
        list_bbb_files_incremental,
        list_rpi_files_incremental,
        build_outinfo,
    )
except Exception as e:
    print(f"ERROR: could not import bb_hpc modules: {e}", file=sys.stderr)
    sys.exit(1)


def _pick_paths(prefer: str | None = None) -> tuple[str, str]:
    """
    Decide which pair of (pipeline_root, resultdir) to use:
      - if prefer == 'local': use *_local
      - if prefer == 'hpc'  : use *_hpc
      - else: prefer *_local if it exists, otherwise *_hpc
    """
    # Read all options from settings (with safe defaults)
    pr_local = getattr(settings, "pipeline_root_local", "")
    pr_hpc   = getattr(settings, "pipeline_root_hpc", "")
    rd_local = getattr(settings, "resultdir_local", "")
    rd_hpc   = getattr(settings, "resultdir_hpc", "")

    def _exists(p: str) -> bool:
        return bool(p) and os.path.exists(p)

    if prefer == "local" and _exists(pr_local):
        return pr_local, rd_local
    if prefer == "hpc" and _exists(pr_hpc):
        return pr_hpc, rd_hpc

    # Auto: prefer local if path exists
    if _exists(pr_local):
        return pr_local, rd_local
    if _exists(pr_hpc):
        return pr_hpc, rd_hpc

    # Last resort: return whatever strings we have (may not exist yet)
    return pr_local or pr_hpc, rd_local or rd_hpc


def _pick_pi_root(prefer: str | None = None) -> str:
    """
    Decide which pi_videodir to use (local vs hpc), preferring existing paths.
    """
    pi_local = getattr(settings, "pi_videodir_local", "")
    pi_hpc   = getattr(settings, "pi_videodir_hpc", "")

    def _exists(p: str) -> bool:
        return bool(p) and os.path.exists(p)

    if prefer == "local" and _exists(pi_local):
        return pi_local
    if prefer == "hpc" and _exists(pi_hpc):
        return pi_hpc

    if _exists(pi_local):
        return pi_local
    if _exists(pi_hpc):
        return pi_hpc
    return pi_local or pi_hpc


def build_bbb_info_parquet(
    pipeline_root: str,
    cache_dir: str,
    recalc: bool,
    check_read_bbb: bool = False,
    deep_check_bbb: bool = False,
) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    if recalc:
        df = list_bbb_files(
            pipeline_root,
            check_read_bbb=check_read_bbb,
            deep_check_bbb=deep_check_bbb,
        )
    else:
        df = list_bbb_files_incremental(
            pipeline_root,
            cache_dir,
            check_read_bbb=check_read_bbb,
            deep_check_bbb=deep_check_bbb,
        )
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = os.path.join(cache_dir, f"bbb_info_{today}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[bbb_info] wrote {len(df)} rows -> {out_path}")
    return out_path


def build_outinfo_parquets(resultdir: str, cache_dir: str) -> tuple[str, str]:
    os.makedirs(cache_dir, exist_ok=True)

    detect_save_dir = os.path.join(resultdir, "data_alldetections")
    track_save_dir  = os.path.join(resultdir, "data_tracked")

    os.makedirs(detect_save_dir, exist_ok=True)
    os.makedirs(track_save_dir, exist_ok=True)

    detect_cache = build_outinfo(
        output_dir=detect_save_dir,
        CACHE_DIR=cache_dir,
        extension="parquet",
        outname="save_detect_outinfo",
    )
    track_cache = build_outinfo(
        output_dir=track_save_dir,
        CACHE_DIR=cache_dir,
        extension="dill",
        outname="save_tracking_outinfo",
    )
    return detect_cache, track_cache


def build_rpi_info_parquet(pi_root: str, cache_dir: str, video_glob_pattern: str = "*.h264") -> str:
    os.makedirs(cache_dir, exist_ok=True)
    df = list_rpi_files_incremental(pi_root, cache_dir, video_glob_pattern=video_glob_pattern)
    today = datetime.now(timezone.utc).strftime("%Y%m%d")
    out_path = os.path.join(cache_dir, f"rpi_info_{today}.parquet")
    df.to_parquet(out_path, index=False)
    print(f"[rpi_info] wrote {len(df)} rows -> {out_path}")
    return out_path


def parse_args():
    p = argparse.ArgumentParser(description="Create/refresh BeesBook fileinfo caches.")
    p.add_argument(
        "--paths",
        choices=["auto", "local", "hpc"],
        default="auto",
        help="Which settings paths to use (auto prefers *_local if they exist).",
    )
    p.add_argument(
        "--what",
        choices=["all", "bbb", "outputs", "rpi"],
        default="all",
        help="Which caches to (re)build: all, only bbb catalog, only outputs catalogs, or only RPi catalog.",
    )
    # ✅ use-cache replaces recalc, inverted logic
    p.add_argument(
        "--use-cache",
        dest="use_cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use existing cached BBB file info (default: true). Use --no-use-cache to force full recalculation.",
    )
    p.add_argument(
        "--check-read-bbb",
        action="store_true",
        help="Attempt to read each .bbb to validate it (adds is_valid column; slower).",
    )
    p.add_argument(
        "--deep-check-bbb",
        action="store_true",
        help="Read through all frames for each .bbb (slower, catches premature EOF).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    pipeline_root, resultdir = _pick_paths(None if args.paths == "auto" else args.paths)
    if not pipeline_root or not resultdir:
        print("ERROR: pipeline_root/resultdir not set in settings.", file=sys.stderr)
        sys.exit(2)

    pi_root = _pick_pi_root(None if args.paths == "auto" else args.paths)

    cache_dir = os.path.join(resultdir, "bbb_fileinfo")
    os.makedirs(cache_dir, exist_ok=True)

    print(f"[config] pipeline_root = {pipeline_root}")
    print(f"[config] resultdir     = {resultdir}")
    print(f"[config] cache_dir     = {cache_dir}")
    print(f"[config] mode          = {args.what}")
    print(f"[config] use_cache     = {args.use_cache}")
    print(f"[config] check_read   = {args.check_read_bbb}")
    print(f"[config] deep_check  = {args.deep_check_bbb}")
    if args.what in ("all", "rpi"):
        print(f"[config] pi_root      = {pi_root}")

    if args.what in ("all", "bbb"):
        # invert logic: recalc = not use_cache
        build_bbb_info_parquet(
            pipeline_root,
            cache_dir,
            recalc=not args.use_cache,
            check_read_bbb=args.check_read_bbb or args.deep_check_bbb,
            deep_check_bbb=args.deep_check_bbb,
        )

    if args.what in ("all", "outputs"):
        build_outinfo_parquets(resultdir, cache_dir)

    if args.what in ("all", "rpi"):
        if not pi_root or not os.path.exists(pi_root):
            print("ERROR: pi_videodir not set or does not exist; skipping RPi catalog.", file=sys.stderr)
        else:
            build_rpi_info_parquet(pi_root, cache_dir)

    print("✅ Done.")


if __name__ == "__main__":
    main()
