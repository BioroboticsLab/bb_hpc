#!/usr/bin/env python3
import sys, json
from pathlib import Path
from datetime import datetime, timezone

# --- path shim for finding the bb_hpc package ---
import sys
from pathlib import Path
_here   = Path(__file__).resolve()
_repo   = _here.parents[1]        # .../bb_hpc
_parent = _repo.parent            # .../jacob   <-- this must be on sys.path
p = str(_parent)
if p not in sys.path:
    sys.path.insert(0, p)
# -----------------------------------------------

def _parse_job_line(s: str) -> dict:
    d = json.loads(s)
    # convert ISO timestamps back to datetime
    for k in ("from_dt", "to_dt"):
        v = d.get(k)
        if isinstance(v, str):
            dt = datetime.fromisoformat(v)
            # normalize to UTC if tz-naive
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            d[k] = dt
    return d

def main():
    if len(sys.argv) != 2:
        print("Usage: run_save_detect.py SHARD.jsonl", file=sys.stderr)
        sys.exit(2)

    shard_path = Path(sys.argv[1])
    if not shard_path.exists():
        print(f"Shard file not found: {shard_path}", file=sys.stderr)
        sys.exit(2)

    # Load JSONL -> list[dict]
    jobs = []
    with shard_path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            jobs.append(_parse_job_line(line))

    # Call the canonical implementation
    from bb_hpc.src.jobfunctions import job_for_save_detect_chunk
    rc = job_for_save_detect_chunk(jobs)
    if rc is not True:
        print(f"[save-detect] job_for_save_detect_chunk returned {rc!r}")

if __name__ == "__main__":
    main()