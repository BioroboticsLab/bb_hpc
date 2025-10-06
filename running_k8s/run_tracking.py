#!/usr/bin/env python3
import os, sys, json
from pathlib import Path
from datetime import datetime, timezone

# --- path shim for finding the bb_hpc package ---
_here   = Path(__file__).resolve()
_repo   = _here.parents[1]        # .../bb_hpc
_parent = _repo.parent            # .../jacob   <-- this must be on sys.path
p = str(_parent)
if p not in sys.path:
    sys.path.insert(0, p)
# -----------------------------------------------

from bb_hpc.src.jobfunctions import job_for_tracking


def _parse_dt(s: str) -> datetime:
    """Robust ISO8601 parser that accepts 'Z' as UTC."""
    if isinstance(s, datetime):
        return s
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    return datetime.fromisoformat(s)

def main():
    if len(sys.argv) != 2:
        print("Usage: run_tracking.py SHARD.jsonl", file=sys.stderr)
        sys.exit(2)

    shard_path = Path(sys.argv[1])
    if not shard_path.exists():
        print(f"File not found: {shard_path}", file=sys.stderr)
        sys.exit(2)

    rc = 0
    with shard_path.open("r") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            job = json.loads(line)

            # Normalize/validate fields expected by job_for_tracking
            try:
                job["from_dt"] = _parse_dt(job["from_dt"])
                job["to_dt"]   = _parse_dt(job["to_dt"])
                job["cam_id"]  = int(job["cam_id"])
                temp_path = job.get("temp_path")
                if temp_path:
                    os.makedirs(temp_path, exist_ok=True)

                print(f"[tracking] cam={job['cam_id']} "
                      f"{job['from_dt'].isoformat()} → {job['to_dt'].isoformat()}")
                job_for_tracking(**job)
            except Exception as e:
                print(f"[tracking] ERROR: {e}\n  job={job}", file=sys.stderr)
                rc = 1

    sys.exit(rc)


if __name__ == "__main__":
    main()