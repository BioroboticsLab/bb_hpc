# bb_hpc/src/slurm_utils.py
from datetime import timedelta
import re

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