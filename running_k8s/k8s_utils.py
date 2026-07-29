#!/usr/bin/env python3
"""Shared helpers for the Kubernetes submit scripts."""

from bb_hpc import settings


def resolve_workers(var: str, cli_value=None, job_settings: dict = None, default: int = 2) -> int:
    """
    Resolve a worker-count knob for one job type.

    Precedence (first non-empty wins):
      1. CLI flag                     (--workers)
      2. per-job setting              e.g. save_detect_settings["workers_per_pod"]
      3. global env                   k8s["env"]["WORKERS_PER_POD"]
      4. global job default           k8s["job"]["workers_per_pod"]
      5. `default` (2)

    `var` is the env var name ("WORKERS_PER_POD" / "WORKERS_PER_GPU"); the
    matching settings key is its lowercase form.
    """
    key = var.lower()
    k = settings.k8s
    candidates = (
        cli_value,
        (job_settings or {}).get(key),
        k.get("env", {}).get(var),
        k.get("job", {}).get(key),
    )
    for c in candidates:
        if c is None or str(c).strip() == "":
            continue
        return int(c)
    return int(default)


def apply_workers_env(env_list: list, var: str, workers: int) -> list:
    """Return env_list with `var` replaced by the resolved worker count."""
    out = [e for e in env_list if e.get("name") != var]
    out.append({"name": var, "value": str(workers)})
    return out
