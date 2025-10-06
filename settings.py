import os, socket

# Who is submitting? default = short hostname, override with env BB_SUBMITTER
submitter_tag = os.environ.get("BB_SUBMITTER", socket.gethostname().split(".")[0])

## DIRECTORY SETTINGS

# Local (submission machine)
bb_hpc_dir_local    = '/mnt/share/jacob/bb_hpc/'
videodir_local      = '/mnt/share/beesbook2025/'
pipeline_root_local = '/mnt/share/beesbook2025/pipeline_repo/'
resultdir_local     = '/mnt/share/beesbook2025/results/'
jobdir_local        = '/mnt/share/beesbook2025/jobs/'
pi_videodir_local   = '/mnt/share/beesbook2025/pi/'

# HPC / cluster (Slurm or K8s or docker nodes)
bb_hpc_dir_hpc    = "/mnt/share/jacob/bb_hpc/"
videodir_hpc      = '/mnt/share/beesbook2025/'
pipeline_root_hpc = '/mnt/share/beesbook2025/pipeline_repo/'
resultdir_hpc     = '/mnt/share/beesbook2025/results/'
jobdir_hpc        = '/mnt/share/beesbook2025/jobs/'
pi_videodir_hpc   = '/mnt/share/beesbook2025/pi/'

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
    "temp_path": '/home/jdavidson/beesbook2025/tracking-tmpfiles/',
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
    "namespace": "user-jacob-davidson",
    "image_pull_secret": "beesbooksecret",
    "image": "ccu-k8s.inf.uni-konstanz.de:32250/jacob.davidson/beesbook:latest",

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
        {"name": "cephfs-home",  "mountPath": "/abyss/home"},
        {"name": "cephfs-shared","mountPath": "/abyss/shared"},
        {"name": "samba-share",  "mountPath": "/mnt/share"},
    ],
    "volumes": [
        {
            "name": "cephfs-home",
            "hostPath": {"path": "/cephfs/abyss/home/jacob-davidson", "type": "Directory"},
        },
        {
            "name": "cephfs-shared",
            "hostPath": {"path": "/cephfs/abyss/shared", "type": "Directory"},
        },
        {
            "name": "samba-share",
            "flexVolume": {
                "driver": "fstab/cifs",
                "fsType": "cifs",
                "secretRef": {"name": "cifs-secret"},
                "options": {
                    "networkPath":  "//timon.cascb.uni-konstanz.de/L21-18-0000",
                    "mountOptions": "vers=3.0,dir_mode=0755,file_mode=0644,noperm,domain=CASCB",
                },
            },
        },
    ],

    "submit": {
        "max_apply_qps": 5,
    },
}

# ---- Local Docker runtime knobs ----
docker = {
    # Reuse the same image & runner used in k8s
    "image": "jacobdavidson/beesbook:latest",
    "runner_path": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_videos.py"),
    "runner_path_rpi": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_rpi_videos.py"),
    "runner_path_save_detect": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_save_detect.py"),
    "runner_path_tracking": os.path.join(bb_hpc_dir_hpc, "running_k8s/run_tracking.py"),


    # Bind mounts: list of (host_path, container_path) so the container sees HPC-style paths.
    # IMPORTANT: Make sure these cover:
    #  - the dataset root
    #  - the jobdir root
    "binds": [
        ("/mnt/share/", "/mnt/share/"),
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
    "containers_per_gpu": 8, # keep 1 unless you’re sure the model is very light
}