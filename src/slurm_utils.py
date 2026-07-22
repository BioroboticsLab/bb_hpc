# bb_hpc/src/slurm_utils.py
from datetime import datetime, timedelta
import os
import io
import re
import json
import contextlib
import subprocess

def _normalize_mem(mem):
    """Convert '6GB' -> '6G'. Slurm also accepts integer MiB."""
    if not mem:
        return None
    mem = str(mem).strip().upper()
    if mem.endswith("GB"):
        return mem[:-2] + "G"
    return mem

def _parse_gres_to_ngpus(gres_value):
    """Accept 'gpu:1', 'gpu:2', 1, 2, or None; return int count."""
    if gres_value is None:
        return 0
    if isinstance(gres_value, int):
        return gres_value
    m = re.match(r"^\s*gpu:(\d+)\s*$", str(gres_value), flags=re.IGNORECASE)
    return int(m.group(1)) if m else 0

def deep_merge(base: dict, override: dict) -> dict:
    """Shallow + nested dict merge where override wins."""
    out = dict(base or {})
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def resolve_slurm_config(global_slurm: dict, specific_settings: dict) -> dict:
    """
    Return a single slurm config where specific overrides global.
    Supports either a subdict 'slurm' or job-level fields at the top of specific_settings.
    """
    global_slurm = dict(global_slurm or {})
    specific_settings = dict(specific_settings or {})
    specific_slurm = dict(specific_settings.get("slurm", {}))

    # Allow a few commonly used job-level keys in the tool settings to override SLURM directly.
    # (Keeps it general without hard-coding per submitter.)
    mappings = {
        "jobtime_minutes": ("jobtime_minutes", int),
        "max_memory": ("max_memory", str),
        "partition": ("partition", str),
        "qos": ("qos", str),
        "nodes": ("nodes", int),
        "cpus_per_task": ("n_cpus", int),
        "ntasks_per_node": ("ntasks_per_node", int),
        "n_gpus": ("n_gpus", int),            # or use gres if you prefer
        "gres": ("gres", str),
        "nice": ("nice", int),
        "concurrent_job_limit": ("concurrent_job_limit", int),
        "max_job_array_size": ("max_job_array_size", int),
        "exports": ("exports", str),
        "custom_preamble": ("custom_preamble", str),
    }
    lifted = {}
    for src_key, (dst_key, caster) in mappings.items():
        if src_key in specific_settings and specific_settings[src_key] is not None:
            try:
                lifted[dst_key] = caster(specific_settings[src_key])
            except Exception:
                lifted[dst_key] = specific_settings[src_key]

    # Merge: global <- specific.slurm <- lifted-from-top-level
    merged = deep_merge(global_slurm, specific_slurm)
    merged = deep_merge(merged, lifted)
    return merged

def apply_slurm_to_job(job, slurm_cfg: dict):
    """Apply a merged slurm config to a SLURMJob instance."""
    # Time limit (minutes -> timedelta)
    minutes = slurm_cfg.get("jobtime_minutes", slurm_cfg.get("time_minutes", None))
    if minutes is not None:
        job.time_limit = timedelta(minutes=int(minutes))

    # Standard fields
    job.partition            = slurm_cfg.get("partition", job.partition)
    job.qos                  = slurm_cfg.get("qos", job.qos)
    job.custom_preamble      = slurm_cfg.get("custom_preamble", job.custom_preamble or "")
    job.max_memory           = _normalize_mem(slurm_cfg.get("max_memory", job.max_memory))
    job.n_nodes              = int(slurm_cfg.get("nodes", getattr(job, "n_nodes", 1)))
    job.n_cpus               = int(slurm_cfg.get("n_cpus", getattr(job, "n_cpus", 1)))
    job.n_tasks              = slurm_cfg.get("ntasks_per_node", getattr(job, "n_tasks", None))
    job.nice                 = slurm_cfg.get("nice", getattr(job, "nice", None))
    job.concurrent_job_limit = slurm_cfg.get("concurrent_job_limit", getattr(job, "concurrent_job_limit", None))
    job.max_job_array_size   = slurm_cfg.get("max_job_array_size", getattr(job, "max_job_array_size", "auto"))
    job.exports              = slurm_cfg.get("exports", getattr(job, "exports", ""))

    # GPUs: prefer explicit n_gpus, else parse gres
    n_gpus = slurm_cfg.get("n_gpus", None)
    if n_gpus is None:
        n_gpus = _parse_gres_to_ngpus(slurm_cfg.get("gres"))
    job.n_gpus = int(n_gpus or 0)


# --------------------------------------------------------------------------------------
# Submission logging: record the array job IDs we submit so they can be monitored later
# with `python -m bb_hpc.slurm_report --jobs <id>`. Best-effort; never blocks a submit.
# --------------------------------------------------------------------------------------
def _append_submitted_jobs(jobdir, record):
    """Append one JSON record to <jobdir>/submitted_jobs.jsonl."""
    try:
        os.makedirs(jobdir, exist_ok=True)
        with open(os.path.join(jobdir, "submitted_jobs.jsonl"), "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except Exception as e:
        print(f"[warn] could not write submitted_jobs.jsonl: {e}")


def _queued_job_ids(user, jobname):
    """Set of this user's currently-queued job IDs for ``jobname`` ('' on any error).

    Note ``%A`` yields one ID per *array element*, so a single 426-task array comes
    back as 426 IDs -- which is why the fallback must diff, never report the raw set.
    """
    try:
        out = subprocess.run(
            ["squeue", "-h", "-u", user, "-n", jobname, "-o", "%A"],
            capture_output=True, text=True, check=False).stdout
        return set(out.split())
    except Exception:
        return set()


def _format_job_ids(job_ids, max_shown=12):
    """Render IDs for the console, eliding the middle of a long list."""
    if len(job_ids) <= max_shown:
        return ", ".join(job_ids)
    head = ", ".join(job_ids[: max_shown // 2])
    tail = ", ".join(job_ids[-(max_shown // 2):])
    return f"{head}, ... ({len(job_ids) - max_shown} more) ..., {tail}"


def run_jobs_and_log(job, jobdir, jobname, dates, user=None):
    """
    Run ``job.run_jobs()`` while recording the submitted array job IDs.

    Captures stdout to scrape sbatch's ``Submitted batch job <id>`` lines (re-printing
    everything so console output is unchanged). If nothing is captured -- e.g. slurmhelper
    writes directly to the OS fd -- falls back to the *difference* between ``squeue``
    snapshots taken before and after the submit, so prior same-name arrays that are still
    queued are not misattributed to this submission. The IDs (plus timestamp/jobname/dates)
    are appended to ``<jobdir>/submitted_jobs.jsonl``. Wrapped in try/except throughout so
    logging never blocks a submission.

    Returns the list of job IDs found (possibly empty).
    """
    user = user or os.environ.get("USER", "")
    before = _queued_job_ids(user, jobname)

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            job.run_jobs()
    finally:
        captured = buf.getvalue()
        if captured:
            print(captured, end="")

    job_ids = re.findall(r"Submitted batch job (\d+)", captured)
    source = "sbatch"
    if not job_ids:
        # Only IDs that appeared while we were submitting belong to this run. Tasks that
        # start and finish inside the submit window are missed; that is preferable to
        # reporting every queued element of an unrelated array.
        new_ids = _queued_job_ids(user, jobname) - before
        # squeue normally yields bare integers; sort numerically but never let an
        # unexpected format (e.g. '123_4') turn logging into a submit failure.
        try:
            job_ids = sorted(new_ids, key=int)
        except ValueError:
            job_ids = sorted(new_ids)
        source = "squeue(fallback,diff)"

    try:
        _append_submitted_jobs(jobdir, {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "jobname": jobname,
            "dates": list(dates) if dates is not None else None,
            "job_ids": job_ids,
            "source": source,
        })
        if job_ids:
            print(f"[submitted] {jobname}: {_format_job_ids(job_ids)} "
                  f"({source}, {len(job_ids)} id(s)) "
                  f"-> logged to {os.path.join(jobdir, 'submitted_jobs.jsonl')}")
        else:
            print(f"[submitted] {jobname}: no new job IDs detected ({source})")
    except Exception as e:
        print(f"[warn] submission logging failed: {e}")
    return job_ids