#!/usr/bin/env python3
"""Start / stop a small CPU-only inspection pod with the cluster volumes mounted.

For storage that only the pods mount (e.g. cephfs under /abyss): checking free
space, counting extracted frames and backgrounds, copying results to the share.
Uses the comb-background image (it has the cell-seg tools) and the
settings.k8s volumes/mounts, so paths look exactly as they do to the jobs.
The pod counts against the namespace pod quota until you take it down.

  python -m bb_hpc.running_k8s.inspect_pod --up     # create; prints the exec line
  python -m bb_hpc.running_k8s.inspect_pod --down   # delete
"""
import argparse
import json
import re
import subprocess
from pathlib import Path

from bb_hpc import settings


def parse_args():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--up", action="store_true", help="Create the pod.")
    g.add_argument("--down", action="store_true", help="Delete the pod.")
    p.add_argument("--name", default=None,
                   help="Pod name (default: bbhpc-inspect-<submitter>).")
    p.add_argument("--image", default=None,
                   help="Image (default: settings.k8s image_comb, else image).")
    return p.parse_args()


def _sanitize_name(name: str) -> str:
    name = name.lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    return name or "bbhpc-inspect"


def make_pod(name, image):
    k = settings.k8s
    env_list = [{"name": kk, "value": str(vv)} for kk, vv in k.get("env", {}).items()]
    pod_spec = {
        "restartPolicy": "Never",
        "imagePullSecrets": [{"name": k["image_pull_secret"]}],
        "containers": [{
            "name": "inspect",
            "image": image,
            "command": ["sleep", "infinity"],
            "resources": {
                "requests": {"cpu": "1", "memory": "2Gi"},
                "limits": {"cpu": "2", "memory": "4Gi"},
            },
            "env": env_list,
            "volumeMounts": k["volume_mounts"],
        }],
        "volumes": k["volumes"],
    }
    if k.get("affinity"):
        pod_spec["affinity"] = k["affinity"]

    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": name,
            "namespace": k["namespace"],
            "labels": {"app": "inspect", "bbhpc": "inspect"},
        },
        "spec": pod_spec,
    }


def main():
    args = parse_args()
    k = settings.k8s
    ns = k["namespace"]
    name = _sanitize_name(args.name or f"bbhpc-inspect-{settings.submitter_tag}")

    if args.down:
        subprocess.run(["kubectl", "delete", "pod", name, "-n", ns,
                        "--ignore-not-found", "--wait=false"], check=True)
        return

    image = args.image or k.get("image_comb", k["image"])
    spec_path = Path(settings.jobdir_local) / "k8s" / "inspect" / f"{name}.json"
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    with open(spec_path, "w") as f:
        json.dump(make_pod(name, image), f, indent=2)

    subprocess.run(["kubectl", "apply", "-f", str(spec_path), "-n", ns], check=True)
    print(f"Pod {name} in ns {ns} (spec: {spec_path}). Once it is Running:")
    print(f"  kubectl wait --for=condition=Ready pod/{name} -n {ns} --timeout=10m")
    print(f"  kubectl exec -it {name} -n {ns} -- bash")
    print(f"Remove it when done: python -m bb_hpc.running_k8s.inspect_pod --down")


if __name__ == "__main__":
    main()
