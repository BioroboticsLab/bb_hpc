#!/usr/bin/env python3
"""
test_pipeline_single_video.py

Run bb_pipeline detection on a single video, calling the same job functions
used in production (bb_hpc.src.jobfunctions). This exercises the real pipeline
code path, so a pass here is meaningful for production readiness.

Modes:
  --mode hd    : standard heatmap localizer + decoder -> bb_binary repo in --out
  --mode polo  : POLO torchscript localizer + decoder -> parquet file in --out

POLO note:
  job_for_process_rpi_videos writes its parquet next to the input video. To
  keep test outputs isolated, this script copies the video into --out first,
  then runs detection on the copy.

Usage:
  python test_pipeline_single_video.py --video PATH --mode {hd,polo} --out DIR
"""

import argparse
import glob
import json
import os
import shutil
import sys
import time
from pathlib import Path


def _add_bb_hpc_to_path():
    """Ensure the parent of bb_hpc is on sys.path so `import bb_hpc.*` works.

    Matches the shim used by bb_hpc/running_k8s/run_videos.py.
    """
    here = Path(__file__).resolve()
    bb_hpc_dir = here.parents[1]       # .../bb_hpc
    parent = bb_hpc_dir.parent          # .../jacob
    p = str(parent)
    if p not in sys.path:
        sys.path.insert(0, p)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--video", required=True, help="Path to input video")
    p.add_argument("--mode", choices=["hd", "polo"], required=True)
    p.add_argument("--out", required=True, help="Output directory (created if missing)")

    clahe = p.add_mutually_exclusive_group()
    clahe.add_argument("--clahe", dest="clahe", action="store_true", default=True,
                       help="Apply CLAHE (POLO mode only). Default: on.")
    clahe.add_argument("--no-clahe", dest="clahe", action="store_false")

    # HD-only
    p.add_argument("--num-threads", type=int, default=1, help="HD only")
    p.add_argument("--timestamp-format", default="basler", help="HD only")
    p.add_argument("--video-file-type", default="basler", help="HD only")

    # POLO-only overrides (default to bb_hpc.settings.polo_config values)
    p.add_argument("--polo-model-path", default=None,
                   help="Override settings.polo_config['polo_model_path']")
    p.add_argument("--polo-attributes-path", default=None,
                   help="Override settings.polo_config['attributes_path']")

    return p.parse_args()


def run_hd(args):
    from bb_hpc.src.jobfunctions import job_for_process_videos
    job_for_process_videos(
        video_paths=[args.video],
        repo_output_path=args.out,
        timestamp_format=args.timestamp_format,
        num_threads=args.num_threads,
        text_root_path=None,
        video_file_type=args.video_file_type,
        copy_local=False,
    )


def run_polo(args):
    from bb_hpc.src.jobfunctions import job_for_process_rpi_videos
    from bb_hpc import settings

    polo_config = dict(settings.polo_config)
    if args.polo_model_path:
        polo_config["polo_model_path"] = args.polo_model_path
    if args.polo_attributes_path:
        polo_config["attributes_path"] = args.polo_attributes_path

    # Stage the video into --out so the parquet output lands there too.
    video_copy = os.path.join(args.out, os.path.basename(args.video))
    if not os.path.exists(video_copy):
        shutil.copy2(args.video, video_copy)

    job_for_process_rpi_videos(
        video_paths=[video_copy],
        clahe=args.clahe,
        model_type="polo",
        polo_config=polo_config,
    )


def main():
    _add_bb_hpc_to_path()
    args = parse_args()

    if not os.path.isfile(args.video):
        print(f"ERROR: video not found: {args.video}", file=sys.stderr)
        sys.exit(2)
    os.makedirs(args.out, exist_ok=True)

    result = {
        "mode": args.mode,
        "video": args.video,
        "out": args.out,
    }
    t0 = time.time()
    try:
        if args.mode == "hd":
            run_hd(args)
            n_out = len(glob.glob(os.path.join(args.out, "**", "*.bbb"), recursive=True))
            result["bbb_files"] = n_out
        else:
            run_polo(args)
            n_out = len(glob.glob(os.path.join(args.out, "*.parquet")))
            result["parquet_files"] = n_out
        # job_for_process_videos catches per-video exceptions as [WARN]s, so an
        # empty output directory is the real failure signal.
        if n_out == 0:
            result["rc"] = 1
            result["error"] = "no output files produced (see [WARN] above)"
        else:
            result["rc"] = 0
    except Exception as e:
        import traceback
        traceback.print_exc()
        result["rc"] = 1
        result["error"] = f"{type(e).__name__}: {e}"

    result["elapsed_sec"] = round(time.time() - t0, 2)
    print(f"TEST_RESULT: {json.dumps(result)}")
    sys.exit(result["rc"])


if __name__ == "__main__":
    main()
