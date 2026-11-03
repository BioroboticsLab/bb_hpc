#!/usr/bin/env python3
"""
Submit a bb_metrics run (1min / 5min / 60min metrics per hive-hour pair) as an Indexed Job.

The run directory is prepared by bb_metrics (metrics_pipeline.prepare_metrics_run, or
submit_metrics_cluster, which calls this script). It holds RUN.json (all parameters) and
pairs.jsonl (the pending pairs). This script only chunks the pairs and submits; each pod
runs `python -m bb_metrics.run_metrics --run-dir <run_dir> <shard>` on WORKERS_PER_POD shards.

The run directory, the config and the trajectory files must be visible at the same paths
in the pods (e.g. /mnt/share). bb_metrics is imported from settings.bb_metrics_dir_hpc.
"""
import argparse, json, re, subprocess
from pathlib import Path
from datetime import datetime

from bb_hpc import settings
from bb_hpc.running_k8s.k8s_utils import resolve_workers, apply_workers_env, job_ttl_spec


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True,
                   help="bb_metrics run directory with RUN.json and pairs.jsonl")
    p.add_argument("--workers", type=int, default=None,
                   help="Workers per pod. Overrides metrics_settings['workers_per_pod'].")
    p.add_argument("--chunk-size", type=int, default=None,
                   help="Pairs per completion index. Overrides metrics_settings['chunk_size'].")
    p.add_argument("--parallelism", type=int, default=None,
                   help="Pods at once. Overrides metrics_settings['parallelism'] and k8s['job']['parallelism'].")
    p.add_argument("--dry-run", action="store_true",
                   help="Write filelists & Job spec, but do not kubectl apply.")
    return p.parse_args()


def _sanitize_job_name(name: str) -> str:
    """
    Convert arbitrary strings to a DNS-1123 compliant name for K8s metadata.name.
    """
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "job"


def write_filelists(filelist_dir: Path, lines, chunk_size: int):
    filelist_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for idx, i0 in enumerate(range(0, len(lines), chunk_size)):
        f = filelist_dir / f"pairs_{idx:05d}.jsonl"
        f.write_text("".join(lines[i0:i0 + chunk_size]))
        files.append(f)
    return files


def make_indexed_job(job_name: str, completions: int, parallelism: int,
                     run_dir: str, bb_metrics_parent: str, workers: int):
    k = settings.k8s
    env_list = [{"name": k_, "value": str(v_)} for k_, v_ in k.get("env", {}).items()]
    # CPU-only: metrics never need a GPU
    resources = k.get("resources_metrics", k.get("resources_save_detect"))
    if resources is None:
        raise SystemExit("settings.k8s needs 'resources_metrics' (CPU-only) for metrics jobs")
    env_list = apply_workers_env(env_list, "WORKERS_PER_POD", workers)

    bash = f"""\
set -euo pipefail

# Activate conda env
if [ -f /opt/conda/etc/profile.d/conda.sh ]; then
  source /opt/conda/etc/profile.d/conda.sh
else
  eval "$(/opt/conda/bin/conda shell.bash hook)"
fi
conda activate beesbook || true

# bb_metrics from the checkout on shared storage
export PYTHONPATH="{bb_metrics_parent}:${{PYTHONPATH:-}}"

# Pick filelist for this completion index
idx="${{JOB_COMPLETION_INDEX:-0}}"
printf -v idx5 "%05d" "$idx"
fl="{run_dir}/filelists/pairs_${{idx5}}.jsonl"
echo "JOB_COMPLETION_INDEX=$idx -> filelist=$fl"

WORKERS="${{WORKERS_PER_POD:-2}}"
echo "WORKERS_PER_POD=${{WORKERS}}"

tmpd=$(mktemp -d)
split -n "l/${{WORKERS}}" -d -a 2 "$fl" "${{tmpd}}/i${{idx5}}_shard_"

pids=()
for shard in "${{tmpd}}"/i*_shard_*; do
  [ -s "$shard" ] || continue
  echo "Starting worker on shard: $shard"
  python -u -m bb_metrics.run_metrics --run-dir "{run_dir}" "$shard" &
  pids+=($!)
done

rc=0
for p in "${{pids[@]}}"; do
  if ! wait "$p"; then
    rc=1
  fi
done
exit $rc
"""

    pod_spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": k["image_pull_secret"]}],
        "containers": [{
            "name": "metrics",
            "image": k["image"],
            "command": ["bash", "-lc", bash],
            "resources": resources,
            "env": env_list,
            "volumeMounts": k["volume_mounts"],
        }],
        "volumes": k["volumes"],
    }
    if k.get("affinity"):
        pod_spec["affinity"] = k["affinity"]

    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {
            "name": job_name,
            "namespace": k["namespace"],
            "labels": {"app": "metrics", "bbhpc": "metrics-indexed"},
        },
        "spec": {
            "backoffLimit": int(k["job"].get("backoff_limit", 1)),
            **job_ttl_spec(),
            "completions": int(completions),
            "parallelism": int(min(max(1, parallelism), max(1, completions))),
            "completionMode": "Indexed",
            "template": {
                "metadata": {
                    "labels": {
                        "app": "metrics",
                        "bbhpc": "metrics-indexed",
                        "job-name": job_name
                    }
                },
                "spec": pod_spec
            }
        }
    }


def main():
    args = parse_args()
    run_dir = Path(args.run_dir).resolve()
    run = json.loads((run_dir / "RUN.json").read_text())
    lines = [l if l.endswith("\n") else l + "\n"
             for l in (run_dir / "pairs.jsonl").read_text().splitlines() if l.strip()]
    if not lines:
        print("No pairs to submit.")
        return

    s = getattr(settings, "metrics_settings", {})
    bb_metrics_dir = getattr(settings, "bb_metrics_dir_hpc", None)
    if not bb_metrics_dir:
        raise SystemExit("settings.bb_metrics_dir_hpc is not set (bb_metrics checkout visible to the pods)")
    chunk_size = int(args.chunk_size or s.get("chunk_size", 20))
    workers = resolve_workers("WORKERS_PER_POD", args.workers, s)
    par = int(args.parallelism or s.get("parallelism") or settings.k8s["job"].get("parallelism", 1))

    filelists = write_filelists(run_dir / "filelists", lines, chunk_size)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    jobname = s.get("jobname", "metrics")
    full_job_name = _sanitize_job_name(f"{jobname}-{run['time_division']}-{stamp}")

    job_dict = make_indexed_job(
        job_name          = full_job_name,
        completions       = len(filelists),
        parallelism       = par,
        run_dir           = str(run_dir),
        bb_metrics_parent = str(Path(bb_metrics_dir).parent),
        workers           = workers,
    )

    spec_path = run_dir / f"{full_job_name}.json"
    spec_path.write_text(json.dumps(job_dict, indent=2))
    print(f"Wrote {spec_path}  (pairs={len(lines)}, completions={len(filelists)}, "
          f"chunk_size={chunk_size}, parallelism={par}, workers_per_pod={workers})")

    run.update({
        "k8s_job_name": full_job_name,
        "k8s_namespace": settings.k8s["namespace"],
        "k8s_completions": len(filelists),
        "chunk_size": chunk_size,
        "workers_per_pod": workers,
        "parallelism": par,
        "dry_run": bool(args.dry_run),
    })
    (run_dir / "RUN.json").write_text(json.dumps(run, indent=2))

    if args.dry_run:
        print("Dry run: not applying Job.")
        return

    subprocess.run(["kubectl", "apply", "-f", str(spec_path),
                    "-n", settings.k8s["namespace"]], check=True)
    print(f"Submitted Job {full_job_name} in ns {settings.k8s['namespace']}")
    print(f"Run dir: {run_dir}")


if __name__ == "__main__":
    main()
