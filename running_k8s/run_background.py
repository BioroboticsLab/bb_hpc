#!/usr/bin/env python3
"""Pod-side runner for the cell-seg background-generation stage.

Reads one shard file (JSONL of (date, camera) work-unit dicts) and runs the
canonical job function. Used by running_k8s/background_submit.py.
"""
import sys
import json
from pathlib import Path

# --- path shim for finding the bb_hpc package ---
_here = Path(__file__).resolve()
_repo = _here.parents[1]        # .../bb_hpc
_parent = _repo.parent          # parent dir that must be on sys.path
p = str(_parent)
if p not in sys.path:
    sys.path.insert(0, p)
# -----------------------------------------------


def main():
    if len(sys.argv) != 2:
        print("Usage: run_background.py SHARD.jsonl", file=sys.stderr)
        sys.exit(2)

    shard_path = Path(sys.argv[1])
    if not shard_path.exists():
        print(f"Shard file not found: {shard_path}", file=sys.stderr)
        sys.exit(2)

    work_units = []
    with shard_path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                work_units.append(json.loads(line))

    from bb_hpc.src.jobfunctions import job_for_background_chunk
    rc = job_for_background_chunk(work_units)
    if rc is not True:
        print(f"[background] job_for_background_chunk returned {rc!r}")


if __name__ == "__main__":
    main()
