#!/usr/bin/env python3
import os, socket

# Who is submitting? default = short hostname, override with env BB_SUBMITTER
submitter_tag = os.environ.get("BB_SUBMITTER", socket.gethostname().split(".")[0])

## DIRECTORY SETTINGS

# Local (submission machine)
bb_hpc_dir_local    = '/path/to/your/bb_hpc/'
videodir_local      = '/path/to/your/beesbook_data/'
pipeline_root_local = '/path/to/your/pipeline_repo/'
resultdir_local     = '/path/to/your/results/'
jobdir_local        = '/path/to/your/jobs/'
pi_videodir_local   = '/path/to/your/pi_videos/'

# HPC / cluster (Slurm or K8s or docker nodes)
bb_hpc_dir_hpc    = "/path/to/your/bb_hpc/"
videodir_hpc      = '/path/to/your/beesbook_data/'
pipeline_root_hpc = '/path/to/your/pipeline_repo/'
resultdir_hpc     = '/path/to/your/results/'
jobdir_hpc        = '/path/to/your/jobs/'
pi_videodir_hpc   = '/path/to/your/pi_videos/'

## detect
detect_settings = {
    "chunk_size": 32,
    "jobtime_minutes": 60,
    "maxjobs": None,
    "jobname": "detect",
    # per-job Slurm overrides (optional)
    "slurm": {
        "max_memory": "6GB",
        "gres": None,
        "exports": (
            "TF_CPP_MIN_LOG_LEVEL=1,"
            "OMP_NUM_THREADS=1,"
            "MKL_NUM_THREADS=1,"
            "TF_NUM_INTRAOP_THREADS=1,"
            "TF_NUM_INTEROP_THREADS=1,"
            "TF_XLA_FLAGS=--tf_xla_enable_xla_devices=0,"
            "KMP_BLOCKTIME=0,"
            "KMP_AFFINITY=granularity=fine,verbose,scatter,"
            "XLA_FLAGS=--xla_gpu_strict_conv_algorithm_picker=false"
        ),
    },
}

## save_detect
save_detect_settings = {
    "jobtime_minutes": 180,
    "chunk_size": 50,
    "maxjobs": None,
    "jobname": "save_detect",
    # per-job Slurm overrides (optional)
    "slurm": {
        "max_memory": "8GB",
        # can override others too, e.g. n_cpus/qos/exports
    },
}

## tracking
track_settings = {
    "temp_path": '/path/to/your/tracking-tmpfiles/',
    "maxjobs": 100,
    "gpu": False,
}

## rpi
rpi_detect_settings = {
    "jobname": 'rpi',
    "jobtime_minutes": 360,
    "chunk_size": 150,
    "maxjobs": None,          # None = as many as needed
    "use_clahe": True,        # True -> "-c"; False -> "-nc"    
}

#-- SLURM-specific settings
slurm = {
    "qos": "standard",
    "partition": "dev",
    "custom_preamble": "",
    "n_cpus": 1,
    "exports": "OMP_NUM_THREADS=1,MKL_NUM_THREADS=1",
    "gres": "gpu:1",
}

# ---- Kubernetes runtime & submission knobs ----
# NOTE: we now use jobdir_local and jobdir_hpc above for filelists/specs.
k8s = {
    "namespace": os.environ.get("K8S_NAMESPACE", "your-namespace"),
    "image_pull_secret": os.environ.get("K8S_IMAGE_PULL_SECRET", "your-image-pull-secret"),
    "image": os.environ.get("K8S_IMAGE", "your-registry.example.com/your-username/beesbook:latest"),

    # The runner script path **inside the cluster** (must exist in the pod via mounts or image)
    "runner_path": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_videos.py"),
    "save_detect_runner_path": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_save_detect.py"),
    "tracking_runner_path": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_tracking.py"),

    "job": {
        "parallelism": 4,
        "backoff_limit": 1,
    },

    "resources": {
        "requests": {
            "cpu": "16",
            "memory": "20Gi",
            "nvidia.com/gpu": "1",
        },
        "limits": {
            "cpu": "24",
            "memory": "48Gi",
            "nvidia.com/gpu": "1",
        },
    },

    # CPU-only resources just for save-detect
    "resources_save_detect": {
        "requests": {"cpu": "4",  "memory": "8Gi"},
        "limits":   {"cpu": "8",  "memory": "16Gi"},
    },    

    "env": {
        "WORKERS_PER_GPU": "4",
        "WORKERS_PER_POD": "4",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "TF_XLA_FLAGS": "--tf_xla_auto_jit=0",
        "TF_ENABLE_ONEDNN_OPTS": "0",
        "TF_NUM_INTRAOP_THREADS": "1",
        "TF_NUM_INTEROP_THREADS": "1",
        "PYTHONPATH": bb_hpc_dir_hpc,  # additional path to include
    },

    "volume_mounts": [
        {"name": "data-volume",  "mountPath": "/mnt/data"},
        # Add your volume mounts here
        # Example:
        # {"name": "shared-storage", "mountPath": "/mnt/share"},
    ],
    "volumes": [
        {
            "name": "data-volume",
            "hostPath": {"path": "/path/to/host/data", "type": "Directory"},
        },
        # Add your volumes here
        # Examples:
        #
        # NFS volume:
        # {
        #     "name": "nfs-volume",
        #     "nfs": {
        #         "server": "nfs.example.com",
        #         "path": "/export/path",
        #     },
        # },
        #
        # CIFS/SMB volume (requires flexVolume driver):
        # {
        #     "name": "cifs-volume",
        #     "flexVolume": {
        #         "driver": "fstab/cifs",
        #         "fsType": "cifs",
        #         "secretRef": {"name": "cifs-secret"},  # Create this secret in your namespace
        #         "options": {
        #             "networkPath": "//your-server.example.com/share-name",
        #             "mountOptions": "vers=3.0,dir_mode=0755,file_mode=0644,noperm,domain=YOUR_DOMAIN",
        #         },
        #     },
        # },
        #
        # PersistentVolumeClaim:
        # {
        #     "name": "pvc-volume",
        #     "persistentVolumeClaim": {"claimName": "your-pvc-name"},
        # },
    ],

    "submit": {
        "max_apply_qps": 5,
    },
}

# ---- Local Docker runtime knobs ----
docker = {
    # Reuse the same image & runner used in k8s
    "image": os.environ.get("DOCKER_IMAGE", "jacobdavidson/beesbook:latest"),
    "runtime": "nvidia",
    "runner_path": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_videos.py"),
    "runner_path_rpi": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_rpi_videos.py"),
    "runner_path_save_detect": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_save_detect.py"),
    "runner_path_tracking": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_tracking.py"),

    # Bind mounts: list of (host_path, container_path) so the container sees HPC-style paths.
    # IMPORTANT: Make sure these cover:
    #  - the dataset root
    #  - the jobdir root
    "binds": [
        ("/path/to/your/data/", "/mnt/share/"),
        # Add more bind mounts as needed
    ],

    # Environment inside the container (reuse k8s env)
    "env": {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MALLOC_ARENA_MAX": "2",
        "TF_FORCE_GPU_ALLOW_GROWTH": "true",
        "TF_CPP_MIN_LOG_LEVEL": "1",
        "PYTHONPATH": bb_hpc_dir_hpc,  # additional path to include
    },

    # Concurrency: one container per GPU by default
    "gpus": "auto",          # "auto" = detect via nvidia-smi; or list like "0,1"
    "containers_per_gpu": 8, # keep 1 unless you're sure the model is very light
}
