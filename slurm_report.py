#!/usr/bin/env python3
"""
Report SLURM job resource usage and recommend right-sized --time / --mem / chunk_size.

Runs on the cluster login node (needs `sacct` on PATH). Stdlib-only, so it works
outside the pipeline conda env. A drop-in replacement for `seff-array` when that
admin tool is not installed.

Examples
--------
  # Everything named "rpi" submitted in the last 3 days, with an email-ready table
  python -m bb_hpc.slurm_report --name rpi --markdown

  # A specific finished array (parent id expands to all _N tasks)
  python -m bb_hpc.slurm_report --jobs 25742737 --markdown

  # detect, sized so each task finishes inside 3h, write per-task rows to CSV
  python -m bb_hpc.slurm_report --name detect --target-walltime-min 180 --csv detect.csv

The tool reads chunk_size per job-name from bb_hpc.settings when importable (for the
per-video math); pass --chunk-size to override or when settings can't be imported.
"""

import os
import re
import sys
import csv
import json
import math
import argparse
import subprocess
from collections import defaultdict

# Optional: auto-discover chunk_size per job-name. Degrade gracefully on the login
# node where the pipeline env (and thus bb_hpc.settings' heavier deps) may be absent.
try:
    from bb_hpc import settings as _settings
except Exception:
    _settings = None


# sacct fields, in order. No -X so we also get .batch/.extern/.N step rows, which are
# the only rows that carry MaxRSS / TotalCPU.
FIELDS = [
    "JobID", "JobName", "State", "Partition", "QOS",
    "Elapsed", "ElapsedRaw", "Timelimit", "TimelimitRaw",
    "ReqMem", "MaxRSS", "AveRSS", "NCPUS", "AllocCPUS",
    "TotalCPU", "CPUTimeRAW", "Submit", "Start", "End", "ExitCode", "NNodes",
]

# States that have meaningful, finished resource numbers (used for the distributions).
TERMINAL_STATES = {"COMPLETED", "TIMEOUT", "FAILED", "OUT_OF_MEMORY", "NODE_FAIL", "CANCELLED"}


# --------------------------------------------------------------------------------------
# parsing helpers
# --------------------------------------------------------------------------------------
def hms_to_seconds(s):
    """Parse '[DD-]HH:MM:SS[.mmm]' (sacct Elapsed/TotalCPU) -> float seconds."""
    if not s or s in ("INVALID", "UNLIMITED", "Partition_Limit"):
        return None
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
    parts = s.split(":")
    parts = [float(p) for p in parts]
    while len(parts) < 3:
        parts.insert(0, 0.0)
    h, m, sec = parts[-3], parts[-2], parts[-1]
    return days * 86400 + h * 3600 + m * 60 + sec


_MEM_RE = re.compile(r"^([\d.]+)\s*([KMGTP]?)([nc]?)$", re.IGNORECASE)
_UNIT = {"": 1.0, "K": 1.0, "M": 1024.0, "G": 1024.0 ** 2, "T": 1024.0 ** 3, "P": 1024.0 ** 4}


def mem_to_kib(value, ncpus=1, nnodes=1):
    """
    Parse a sacct memory string -> KiB (float).

    Handles MaxRSS like '1234567K' and ReqMem in either the newer total form ('2G',
    '512M', '2097152K') or the older per-node / per-cpu form ('2Gn', '1024Mc').
    A bare number (no unit) is treated as MiB, matching old Slurm ReqMem.
    """
    if not value:
        return None
    m = _MEM_RE.match(value.strip())
    if not m:
        return None
    num = float(m.group(1))
    unit = m.group(2).upper()
    scope = m.group(3).lower()
    # bare number with no unit -> MiB (legacy ReqMem); explicit 'K' stays KiB
    kib = num * (_UNIT["M"] if unit == "" else _UNIT[unit])
    if scope == "c":
        kib *= max(int(ncpus or 1), 1)
    elif scope == "n":
        kib *= max(int(nnodes or 1), 1)
    return kib


def to_int(s, default=None):
    try:
        return int(s)
    except (TypeError, ValueError):
        return default


def pct(values, q):
    """Nearest-rank percentile (q in 0..1) over a list of numbers. None if empty."""
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if q <= 0:
        return vals[0]
    rank = max(1, math.ceil(q * len(vals)))
    return vals[min(rank, len(vals)) - 1]


def median(values):
    return pct(values, 0.5)


def fmt_dur(seconds):
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def fmt_kib(kib):
    if kib is None:
        return "-"
    g = kib / (1024.0 ** 2)
    if g >= 1:
        return f"{g:.2f}G"
    return f"{kib / 1024.0:.0f}M"


def fmt_pct(x):
    return "-" if x is None else f"{100 * x:.1f}%"


def mem_string_from_mib(mib):
    """Round MiB up to a tidy 256-MiB boundary (min 1024) -> Slurm --mem string."""
    mib = max(1024, int(math.ceil(mib / 256.0) * 256))
    return f"{mib // 1024}G" if mib % 1024 == 0 else f"{mib}M"


def ceil_5min(seconds):
    return max(10, int(math.ceil(seconds / 300.0) * 5))


# --------------------------------------------------------------------------------------
# sacct
# --------------------------------------------------------------------------------------
def run_sacct(args):
    cmd = ["sacct", "-P", "-n", "--units=K", "--format=" + ",".join(FIELDS)]
    if args.jobs:
        cmd += ["-j", ",".join(args.jobs)]
    else:
        cmd += ["-u", args.user, "-S", args.since]
        if args.until:
            cmd += ["-E", args.until]
        if args.name:
            cmd += ["--name", ",".join(args.name)]
    if args.state:
        cmd += ["--state", args.state]
    if args.verbose:
        print("[cmd]", " ".join(cmd), file=sys.stderr)
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError:
        print("ERROR: `sacct` not found. Run this on the curta login node.", file=sys.stderr)
        sys.exit(2)
    if proc.returncode != 0:
        print(f"ERROR: sacct exited {proc.returncode}:\n{proc.stderr}", file=sys.stderr)
        sys.exit(2)
    return [ln for ln in proc.stdout.splitlines() if ln.strip()]


def parse_tasks(lines):
    """
    Build a dict of task records keyed by task id (e.g. '25742737_135').

    Task (allocation) rows carry State/ReqMem/Timelimit/NCPUS/Elapsed; step rows
    ('.batch'/'.extern'/'.N') carry MaxRSS/TotalCPU. We merge the steps onto the parent.
    """
    tasks = {}
    steps = defaultdict(list)
    for ln in lines:
        f = ln.split("|")
        if len(f) != len(FIELDS):
            continue
        rec = dict(zip(FIELDS, f))
        jid = rec["JobID"]
        if "[" in jid:
            continue  # pending array master like 25742737_[140-893]
        if "." in jid:
            steps[jid.split(".")[0]].append(rec)
        else:
            tasks[jid] = rec

    out = {}
    for jid, t in tasks.items():
        ncpus = to_int(t.get("NCPUS")) or to_int(t.get("AllocCPUS")) or 1
        nnodes = to_int(t.get("NNodes")) or 1

        # MaxRSS: max over non-.extern steps (fall back to extern, then task row).
        rss_vals = []
        cpu_secs = []
        for st in steps.get(jid, []):
            is_extern = st["JobID"].endswith(".extern")
            r = mem_to_kib(st.get("MaxRSS"))
            if r is not None and not is_extern:
                rss_vals.append(r)
            c = hms_to_seconds(st.get("TotalCPU"))
            if c is not None and st["JobID"].endswith(".batch"):
                cpu_secs.append(c)
        if not rss_vals:  # only extern had a number, or task row carries it
            for st in steps.get(jid, []):
                r = mem_to_kib(st.get("MaxRSS"))
                if r is not None:
                    rss_vals.append(r)
            tr = mem_to_kib(t.get("MaxRSS"))
            if tr is not None:
                rss_vals.append(tr)
        if not cpu_secs:
            tc = hms_to_seconds(t.get("TotalCPU"))
            if tc is not None:
                cpu_secs.append(tc)

        elapsed = to_int(t.get("ElapsedRaw"))
        if elapsed is None:
            elapsed = hms_to_seconds(t.get("Elapsed"))
        tl_min = to_int(t.get("TimelimitRaw"))
        timelimit_s = tl_min * 60 if tl_min is not None else hms_to_seconds(t.get("Timelimit"))
        cputime_raw = to_int(t.get("CPUTimeRAW"))

        out[jid] = {
            "jobid": jid,
            "name": t.get("JobName", ""),
            "state": (t.get("State") or "").split()[0],  # 'CANCELLED by 420084' -> 'CANCELLED'
            "elapsed_s": elapsed,
            "timelimit_s": timelimit_s,
            "reqmem_kib": mem_to_kib(t.get("ReqMem"), ncpus, nnodes),
            "maxrss_kib": max(rss_vals) if rss_vals else None,
            "ncpus": ncpus,
            "totalcpu_s": max(cpu_secs) if cpu_secs else None,
            "cputime_raw_s": cputime_raw,
            "exitcode": t.get("ExitCode", ""),
        }
    return out


# --------------------------------------------------------------------------------------
# analysis
# --------------------------------------------------------------------------------------
def build_chunk_map(args):
    chunk_map, default_chunk = {}, None
    if _settings is not None:
        for attr in ("detect_settings", "rpi_detect_settings",
                     "save_detect_settings", "track_settings"):
            d = getattr(_settings, attr, None)
            if isinstance(d, dict) and d.get("jobname") and d.get("chunk_size"):
                chunk_map[d["jobname"]] = int(d["chunk_size"])
    for entry in (args.chunk_size or []):
        if "=" in entry:
            name, val = entry.split("=", 1)
            chunk_map[name.strip()] = int(val)
        else:
            default_chunk = int(entry)
    return chunk_map, default_chunk


def summarize(name, recs, chunk_size, args):
    state_counts = defaultdict(int)
    for r in recs:
        state_counts[r["state"]] += 1

    # distributions use terminal tasks with real timing only
    term = [r for r in recs if r["state"] in TERMINAL_STATES and r["elapsed_s"]]
    completed = [r for r in term if r["state"] == "COMPLETED"]

    elapsed = [r["elapsed_s"] for r in term]
    rss = [r["maxrss_kib"] for r in term if r["maxrss_kib"]]
    mem_eff = [r["maxrss_kib"] / r["reqmem_kib"]
               for r in term if r["maxrss_kib"] and r["reqmem_kib"]]
    time_eff = [r["elapsed_s"] / r["timelimit_s"]
                for r in term if r["elapsed_s"] and r["timelimit_s"]]
    cpu_eff = []
    for r in term:
        denom = r["cputime_raw_s"] or ((r["elapsed_s"] or 0) * (r["ncpus"] or 1))
        if r["totalcpu_s"] and denom:
            cpu_eff.append(r["totalcpu_s"] / denom)

    # per-video stats (prefer COMPLETED so TIMEOUT censoring doesn't bias the chunk math)
    base = completed or term
    per_video = [r["elapsed_s"] / chunk_size for r in base if r["elapsed_s"]] if chunk_size else []

    # short FAILED tasks are usually scheduler/startup rejections, not resource problems
    sched_rejects = sum(1 for r in recs
                        if r["state"] == "FAILED" and (r["elapsed_s"] or 0) < 60)

    reqmem = next((r["reqmem_kib"] for r in term if r["reqmem_kib"]), None)
    timelimit = next((r["timelimit_s"] for r in term if r["timelimit_s"]), None)

    return {
        "name": name, "chunk_size": chunk_size,
        "n_total": len(recs), "n_terminal": len(term), "n_completed": len(completed),
        "state_counts": dict(state_counts), "sched_rejects": sched_rejects,
        "reqmem_kib": reqmem, "timelimit_s": timelimit,
        "elapsed": {"min": pct(elapsed, 0), "med": median(elapsed),
                    "p95": pct(elapsed, 0.95), "max": pct(elapsed, 1.0)},
        "per_video": {"med": median(per_video), "p95": pct(per_video, 0.95)},
        "rss": {"med": median(rss), "p95": pct(rss, 0.95), "max": pct(rss, 1.0)},
        "mem_eff": {"med": median(mem_eff), "p95": pct(mem_eff, 0.95)},
        "time_eff": {"med": median(time_eff), "p95": pct(time_eff, 0.95)},
        "cpu_eff_med": median(cpu_eff),
        "recs": _recommend(elapsed, rss, per_video, chunk_size, args),
    }


def _recommend(elapsed, rss, per_video, chunk_size, args):
    out = {}
    p95_rss = pct(rss, 0.95)
    if p95_rss is not None:
        rec_mib = p95_rss * args.mem_margin / 1024.0
        out["mem"] = mem_string_from_mib(rec_mib)
        out["mem_basis"] = f"p95 MaxRSS {fmt_kib(p95_rss)} * {args.mem_margin}"
    p95_el = pct(elapsed, 0.95)
    if p95_el is not None:
        out["time_min"] = ceil_5min(p95_el * args.time_margin)
        out["time_basis"] = f"p95 elapsed {fmt_dur(p95_el)} * {args.time_margin}"
    p95_pv = pct(per_video, 0.95)
    if p95_pv and chunk_size:
        safe_pv = p95_pv * args.time_margin
        target_s = args.target_walltime_min * 60
        rec_chunk = max(1, int(math.floor(target_s / safe_pv)))
        out["chunk_size"] = rec_chunk
        out["chunk_jobtime_min"] = ceil_5min(rec_chunk * safe_pv)
        out["chunk_basis"] = (f"target {args.target_walltime_min}min / "
                              f"(p95 per-video {fmt_dur(p95_pv)} * {args.time_margin})")
    return out


# --------------------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------------------
def render_text(summaries, args):
    lines = []
    for s in summaries:
        sc = s["state_counts"]
        sc_str = " ".join(f"{k}={v}" for k, v in sorted(sc.items()))
        lines.append(f"=== name={s['name']}  tasks={s['n_total']}  "
                     f"(terminal={s['n_terminal']}, completed={s['n_completed']}) ===")
        lines.append(f"  States: {sc_str}")
        if s["sched_rejects"]:
            lines.append(f"  NOTE: {s['sched_rejects']} FAILED in <60s "
                         f"-> likely scheduler/startup rejections (e.g. QOS limit), "
                         f"not a resource problem; check the .err logs.")
        e = s["elapsed"]
        lines.append(f"  Elapsed:    min={fmt_dur(e['min'])}  med={fmt_dur(e['med'])}  "
                     f"p95={fmt_dur(e['p95'])}  max={fmt_dur(e['max'])}")
        if s["chunk_size"]:
            pv = s["per_video"]
            lines.append(f"  Per-video:  chunk_size={s['chunk_size']}  "
                         f"med={fmt_dur(pv['med'])}  p95={fmt_dur(pv['p95'])}")
        r = s["rss"]
        lines.append(f"  MaxRSS:     med={fmt_kib(r['med'])}  p95={fmt_kib(r['p95'])}  "
                     f"max={fmt_kib(r['max'])}   (ReqMem={fmt_kib(s['reqmem_kib'])})")
        lines.append(f"  Mem-eff:    med={fmt_pct(s['mem_eff']['med'])}  "
                     f"p95={fmt_pct(s['mem_eff']['p95'])}   (MaxRSS/ReqMem)")
        lines.append(f"  Time-eff:   med={fmt_pct(s['time_eff']['med'])}  "
                     f"p95={fmt_pct(s['time_eff']['p95'])}   (Elapsed/Timelimit) "
                     f"[limit={fmt_dur(s['timelimit_s'])}]")
        lines.append(f"  CPU-eff:    med={fmt_pct(s['cpu_eff_med'])}")

        rec = s["recs"]
        lines.append(f"  --- RECOMMENDATIONS (time x{args.time_margin}, mem x{args.mem_margin}) ---")
        if "mem" in rec:
            lines.append(f"    max_memory      : {rec['mem']:>8}   [{rec['mem_basis']}]")
        if "time_min" in rec:
            lines.append(f"    jobtime_minutes : {rec['time_min']:>8}   [{rec['time_basis']}]")
        if "chunk_size" in rec:
            lines.append(f"    chunk_size      : {rec['chunk_size']:>8}   "
                         f"(=> jobtime ~{rec['chunk_jobtime_min']}min)  [{rec['chunk_basis']}]")
        if s["n_completed"] < max(3, 0.1 * s["n_terminal"]):
            lines.append("    WARN: few COMPLETED tasks -> time/per-video figures are "
                         "censored lower bounds; re-measure after one right-sized run.")
        lines.append("")
    lines.append("Concurrency: read the per-user QOS cap with")
    lines.append("    sacctmgr show qos <qos> format=Name,MaxJobsPU,MaxSubmitPU,MaxTRESPU -P")
    lines.append("then set concurrent_job_limit = floor(0.8 * MaxJobsPU) in settings.py.")
    return "\n".join(lines)


def render_markdown(summaries):
    rows = ["| Job | Tasks | C/T/F | Elapsed p95 | Timelimit | Time-eff p95 | "
            "MaxRSS p95 | ReqMem | Mem-eff p95 | Rec mem | Rec time | Rec chunk |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in summaries:
        sc = s["state_counts"]
        ctf = f"{sc.get('COMPLETED', 0)}/{sc.get('TIMEOUT', 0)}/{sc.get('FAILED', 0)}"
        rec = s["recs"]
        rows.append(
            f"| {s['name']} | {s['n_total']} | {ctf} | "
            f"{fmt_dur(s['elapsed']['p95'])} | {fmt_dur(s['timelimit_s'])} | "
            f"{fmt_pct(s['time_eff']['p95'])} | {fmt_kib(s['rss']['p95'])} | "
            f"{fmt_kib(s['reqmem_kib'])} | {fmt_pct(s['mem_eff']['p95'])} | "
            f"{rec.get('mem', '-')} | "
            f"{str(rec.get('time_min', '-')) + 'm' if 'time_min' in rec else '-'} | "
            f"{rec.get('chunk_size', '-')} |")
    return "\n".join(rows)


def read_ledger(jobdir, name_filter=None, limit=20):
    """Return the last `limit` records from <jobdir>/submitted_jobs.jsonl (written by
    run_jobs_and_log at submit time), newest last, optionally filtered by job name."""
    path = os.path.join(jobdir, "submitted_jobs.jsonl")
    if not os.path.exists(path):
        print(f"No submission ledger at {path}.\n"
              "(It is written by run_jobs_and_log on submit; jobs from before that was "
              "wired up won't be here -- use --name/-S to find those via sacct.)",
              file=sys.stderr)
        return None, path
    recs = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            if name_filter and r.get("jobname") not in name_filter:
                continue
            recs.append(r)
    return recs[-limit:], path


def render_ledger(recs, path):
    print(f"Recent submissions (from {path}):\n")
    print(f"{'submitted':<20} {'job':<12} {'ids':<26} dates")
    print("-" * 76)
    for r in recs:
        ids = ",".join(r.get("job_ids") or []) or "(none captured)"
        dates = ",".join(r.get("dates") or []) if r.get("dates") else "-"
        print(f"{r.get('ts', '-'):<20} {r.get('jobname', '-'):<12} {ids:<26} {dates}")
    print("\nProfile one with:  python -m bb_hpc.slurm_report --jobs <id> --markdown")


def write_csv(path, tasks):
    cols = ["jobid", "name", "state", "elapsed_s", "timelimit_s",
            "reqmem_kib", "maxrss_kib", "ncpus", "totalcpu_s", "exitcode"]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in tasks.values():
            w.writerow(r)


# --------------------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Report SLURM job resource usage and recommend right-sized settings.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("-u", "--user", default=os.environ.get("USER", ""),
                   help="SLURM user (ignored when --jobs is given)")
    p.add_argument("-S", "--since", default="now-3days",
                   help="start of window, passed to sacct -S (e.g. 2026-06-01, now-7days)")
    p.add_argument("-E", "--until", default=None, help="end of window, passed to sacct -E")
    p.add_argument("--name", action="append", help="job name filter (repeatable)")
    p.add_argument("--jobs", nargs="+", help="explicit job/array ids (a bare array id "
                   "expands to all _N tasks)")
    p.add_argument("--chunk-size", action="append", metavar="NAME=N|N",
                   help="declare chunk_size for per-video math (e.g. rpi=45); a bare int "
                        "is the default for unmapped names")
    p.add_argument("--target-walltime-min", type=int, default=180,
                   help="target per-task wall-time for the chunk_size recommendation")
    p.add_argument("--time-margin", type=float, default=1.5)
    p.add_argument("--mem-margin", type=float, default=1.3)
    p.add_argument("--state", default=None, help="sacct --state filter, e.g. COMPLETED,TIMEOUT")
    p.add_argument("--list", nargs="?", const=20, type=int, metavar="N",
                   help="list the last N submitted arrays from the ledger (default 20) and "
                        "exit; combine with --name to filter. Use this to find a job id.")
    p.add_argument("--jobdir", default=None,
                   help="dir holding submitted_jobs.jsonl (default: settings.jobdir_hpc)")
    p.add_argument("--markdown", action="store_true", help="also emit an email-ready table")
    p.add_argument("--csv", default=None, help="write per-task rows to this CSV path")
    p.add_argument("-v", "--verbose", action="store_true", help="print the sacct command")
    return p.parse_args()


def main():
    args = parse_args()

    # --list: just show what we've submitted (and the array ids) from the ledger
    if args.list is not None:
        jobdir = args.jobdir or (getattr(_settings, "jobdir_hpc", None) if _settings else None)
        if not jobdir:
            print("ERROR: pass --jobdir (could not import settings.jobdir_hpc).", file=sys.stderr)
            sys.exit(2)
        recs, path = read_ledger(jobdir, set(args.name) if args.name else None, args.list)
        if not recs:
            sys.exit(1)
        render_ledger(recs, path)
        return

    if not args.jobs and not args.user:
        print("ERROR: set $USER or pass -u/--user (or use --jobs).", file=sys.stderr)
        sys.exit(2)

    tasks = parse_tasks(run_sacct(args))
    if not tasks:
        print("No matching jobs found. Widen --since, check --name, or run on the "
              "curta login node.", file=sys.stderr)
        sys.exit(1)

    chunk_map, default_chunk = build_chunk_map(args)

    by_name = defaultdict(list)
    for r in tasks.values():
        by_name[r["name"]].append(r)

    summaries = []
    for name in sorted(by_name):
        chunk = chunk_map.get(name, default_chunk)
        summaries.append(summarize(name, by_name[name], chunk, args))

    print(render_text(summaries, args))
    if args.markdown:
        print("\n--- markdown (paste into the HPC-team email) ---\n")
        print(render_markdown(summaries))
    if args.csv:
        write_csv(args.csv, tasks)
        print(f"\n[csv] wrote {len(tasks)} task rows -> {args.csv}", file=sys.stderr)


if __name__ == "__main__":
    main()
