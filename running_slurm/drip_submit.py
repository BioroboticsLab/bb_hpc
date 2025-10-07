#!/usr/bin/env python3
"""
bb_hpc.running_slurm.drip_submit
--------------------------------
Generic drip-feed submitter for SLURM-backed bb_hpc jobs.

Supported job types:
  - detect
  - save_detect
  - tracking
  - detect_rpi

It submits one day at a time, waits, handles QOS submit limit backoff,
and persists state so you can stop/restart safely.

Running example with options:
python -m bb_hpc.running_slurm.drip_submit \
  --job tracking \
  --start 20250619 \
  --end 20250819 \
  --wait-hours 1 \
  --backoff-hours 3 
"""

from __future__ import annotations
import sys, os, json, time, subprocess, shlex
from pathlib import Path
from datetime import datetime, timedelta, timezone
import argparse

# Map job types to their submit modules
JOB_MODULES = {
    "detect":      "bb_hpc.running_slurm.detect_submit",
    "save_detect": "bb_hpc.running_slurm.save_detect_submit",
    "tracking":    "bb_hpc.running_slurm.tracking_submit",
    "detect_rpi":  "bb_hpc.running_slurm.detect_rpi_submit",
}

DEFAULT_WAIT_HOURS    = 4
DEFAULT_BACKOFF_HOURS = 2

# Patterns indicating submit-limit / policy throttling (cluster-specific but common)
SUBMIT_LIMIT_PATTERNS = (
    "QOSMaxSubmitJobPerUserLimit",
    "QOSMaxJobsPerUserLimit",
    "Job violates accounting/QOS policy",
)

def parse_args():
    now = datetime.now(timezone.utc)
    today = now.strftime("%Y%m%d")
    yesterday = (now - timedelta(days=1)).strftime("%Y%m%d")

    p = argparse.ArgumentParser(
        description="Drip-feed SLURM submissions for bb_hpc jobs (one day at a time)."
    )
    p.add_argument(
        "--job", required=True, choices=sorted(JOB_MODULES.keys()),
        help="Which bb_hpc SLURM submit module to call."
    )
    p.add_argument(
        "--start", default=yesterday, help="Start date (YYYYMMDD), inclusive. Default: yesterday (UTC)."
    )
    p.add_argument(
        "--end", default=today, help="End date (YYYYMMDD), inclusive. Default: today (UTC)."
    )
    p.add_argument(
        "--wait-hours", type=float, default=DEFAULT_WAIT_HOURS,
        help=f"Hours to wait between submissions (default: {DEFAULT_WAIT_HOURS})."
    )
    p.add_argument(
        "--backoff-hours", type=float, default=DEFAULT_BACKOFF_HOURS,
        help=f"Extra backoff if submit limit is hit (default: {DEFAULT_BACKOFF_HOURS})."
    )
    p.add_argument(
        "--state-dir", default=None,
        help="Directory to store drip state/logs. Default: <settings.jobdir_hpc>/slurm_drip/<job>"
    )
    p.add_argument(
        "--extra-args", default="", help="Extra args passed to the submit module (quoted string)."
    )
    p.add_argument(
        "--print-cmd", action="store_true",
        help="Print the exact python -m command invoked each time."
    )
    return p.parse_args()

def daterange(start_str: str, end_str: str):
    d0 = datetime.strptime(start_str, "%Y%m%d").date()
    d1 = datetime.strptime(end_str,   "%Y%m%d").date()
    if d1 < d0:
        raise ValueError("END date is before START date")
    d = d0
    while d <= d1:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)

def load_settings_paths():
    from bb_hpc import settings
    return settings

def resolve_state_dir(job: str, override: str | None):
    if override:
        return Path(override)
    s = load_settings_paths()
    return Path(s.jobdir_hpc) / "slurm_drip" / job

def log(msg: str, logfile: Path):
    stamp = time.strftime("[%Y-%m-%d %H:%M:%S]")
    logfile.parent.mkdir(parents=True, exist_ok=True)
    with logfile.open("a") as f:
        f.write(f"{stamp} {msg}\n")
    print(f"{stamp} {msg}", flush=True)

def load_done(done_path: Path) -> set[str]:
    if done_path.exists():
        try:
            return set(json.loads(done_path.read_text()))
        except Exception:
            backup = done_path.with_suffix(".corrupt.json")
            done_path.rename(backup)
            print(f"WARNING: state file corrupt; backed up to {backup}")
    return set()

def save_done(done_path: Path, done_set: set[str]):
    done_path.write_text(json.dumps(sorted(done_set)))

def hit_submit_limit(stdout: str, stderr: str) -> bool:
    combined = (stdout or "") + "\n" + (stderr or "")
    return any(pat in combined for pat in SUBMIT_LIMIT_PATTERNS)

def should_mark_done(stdout: str, stderr: str, rc: int) -> bool:
    if hit_submit_limit(stdout, stderr):
        return False
    return rc == 0

def run_one_day(module: str, day: str, extra_args_str: str, print_cmd: bool) -> tuple[int, str, str]:
    """
    Invoke: python -m <module> --dates YYYYMMDD [EXTRA_ARGS...]
    Return (rc, stdout, stderr).
    """
    extra = shlex.split(extra_args_str) if extra_args_str else []
    cmd = [sys.executable, "-m", module, "--dates", day, *extra]
    if print_cmd:
        print("CMD:", " ".join(shlex.quote(x) for x in cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr

def main():
    args = parse_args()

    module    = JOB_MODULES[args.job]
    state_dir = resolve_state_dir(args.job, args.state_dir)
    done_path = state_dir / "submitted_days.json"
    log_path  = state_dir / "drip_submit.log"

    # Prime state
    done = load_done(done_path)
    all_days = list(daterange(args.start, args.end))
    remaining = [d for d in all_days if d not in done]

    if not remaining:
        log(f"[{args.job}] All days already submitted. Nothing to do.", log_path)
        return

    log(f"[{args.job}] Starting drip submitter: {len(remaining)} days remaining; "
        f"wait={args.wait_hours}h backoff={args.backoff_hours}h", log_path)

    wait_seconds       = int(args.wait_hours * 3600)
    backoff_additional = int(args.backoff_hours * 3600)

    for day in remaining:
        log(f"[{args.job}] Submitting day {day} ...", log_path)
        rc, out, err = run_one_day(module, day, args.extra_args, args.print_cmd)

        if hit_submit_limit(out, err):
            log(f"[{args.job}] QOS submit limit hit; retrying after backoff ({args.backoff_hours}h).", log_path)
            time.sleep(wait_seconds + backoff_additional)
            rc2, out2, err2 = run_one_day(module, day, args.extra_args, args.print_cmd)
            log(f"[{args.job}] Retry return code: {rc2}", log_path)
            if should_mark_done(out2, err2, rc2):
                done.add(day)
                save_done(done_path, done)
                log(f"[{args.job}] Marked {day} as submitted (after retry).", log_path)
                time.sleep(wait_seconds)
            else:
                log(f"[{args.job}] Still submit-limited or error after retry; keeping day pending.", log_path)
                time.sleep(wait_seconds)
        else:
            if should_mark_done(out, err, rc):
                done.add(day)
                save_done(done_path, done)
                log(f"[{args.job}] Marked {day} as submitted.", log_path)
            else:
                log(f"[{args.job}] Non-zero exit; leaving day pending.", log_path)
            time.sleep(wait_seconds)

    log(f"[{args.job}] Finished drip loop over configured range.", log_path)

if __name__ == "__main__":
    main()