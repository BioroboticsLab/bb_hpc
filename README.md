# bb_hpc

A lightweight helper package for managing BeesBook HPC and distributed jobs across
Docker, Kubernetes, and SLURM. It provides unified job generation, submission,
and execution scripts for various processing stages (detect, save-detect,
tracking, and RPi-detect).

---

## 1. Installation and Settings

### Clone or link locally
Clone the repository into your working directory, and install in editable mode (needed for editing settings) in your Python environment
```bash
git clone https://github.com/BioroboticsLab/bb_hpc.git 
pip install -e bb_hpc
```

### Settings file

The configuration is expected in:
```text
bb_hpc/settings.py
```

Edit ```settings.example.py``` and save it as ```settings.py``` to create a first version of this file.

This file contains all environment-specific paths and options, such as:
- local and HPC directory mappings
- SLURM and Kubernetes parameters
- Docker settings
- job defaults (chunk size, time limits, etc.)

**Note:** This file must be identical on both the **local** system and the **HPC/Docker**
mount, so copy or sync it if needed.

---

## 2. FileInfo (Prerequisite)

**Run FileInfo before Save-Detect or Tracking.** It creates caches of what's already processed in your bb_binary repositories to prevent redundant work.

**Detect does NOT require FileInfo.**

### When to run FileInfo

Always rerun FileInfo when:
- New videos or detections have been added to the repo
- You've reprocessed detections and want Save-Detect/Tracking to update only changed periods

### Basic usage

```bash
python -m bb_hpc.get_fileinfo
```

This updates all caches using paths from `settings.py`.

### Optional flags

```
--paths {auto, local, hpc}    # Which directory paths to use (default: auto)
--what  {all, bbb, outputs}   # Which caches to rebuild (default: all)
```

**Examples:**

```bash
# Update only bb_binary cache using HPC paths (e.g., on FU Berlin Curta)
python -m bb_hpc.get_fileinfo --paths hpc --what bbb

# Update only outputs using local paths (quick refresh before new jobs)
python -m bb_hpc.get_fileinfo --paths local --what outputs
```

After FileInfo is updated, Save-Detect and Tracking will automatically skip already-processed windows.

---

## 3. Running Processing Stages

All scripts accept **--dates** (YYYYMMDD, UTC). Many also support **--dry-run** to stage work only.

**GPU support:** Kubernetes and Docker support GPU acceleration (significantly speeds up Detect; may help RPi-Detect). SLURM runs CPU-only in this setup.

### SLURM (CPU-only)
```bash
# Detect
python -m bb_hpc.running_slurm.detect_submit --dates 20251001
# Save-Detect
python -m bb_hpc.running_slurm.save_detect_submit --dates 20251001
# Tracking
python -m bb_hpc.running_slurm.tracking_submit --dates 20251001
# RPi-Detect
python -m bb_hpc.running_slurm.detect_rpi_submit --dates 20251001
```

### Kubernetes
```bash
# Detect (Indexed Job; GPU requests come from settings.k8s)
python -m bb_hpc.running_k8s.detect_submit --dates 20251001
# Save-Detect (CPU only)
python -m bb_hpc.running_k8s.save_detect_submit --dates 20251001
# Tracking (GPU support)
python -m bb_hpc.running_k8s.tracking_submit --dates 20251001
# RPi-Detect (GPU support)
python -m bb_hpc.running_k8s.detect_rpi_submit --dates 20251001
```

### Docker (local)
```bash
# Detect (GPU support)
python -m bb_hpc.running_docker.detect_submit --dates 20251001
# Save-Detect (CPU only)
python -m bb_hpc.running_docker.save_detect_submit --dates 20251001
# Tracking (GPU support)
python -m bb_hpc.running_docker.tracking_submit --dates 20251001
# RPi-Detect (GPU support)
python -m bb_hpc.running_docker.detect_rpi_submit --dates 20251001
```

### Docker extra options (for local flexibility)

**Detect:**
- `--gpus {auto,all,0,1,...}` — GPU selection
- `--containers-per-gpu N` — Parallel containers per GPU

**Save-Detect:**
- `--workers N` — Number of parallel containers

**Tracking:**
- `--gpus {auto,all,0,1,...}` — GPU selection
- `--containers-per-gpu N` — Parallel containers per GPU

**RPi-Detect:**
- `--gpus {auto,all,0,1,...}` — GPU selection
- `--containers-per-gpu N` — Parallel containers per GPU
- `--clahe / --no-clahe` — CLAHE preprocessing toggle
- `--chunk-size N` — Processing chunk size

All Docker scripts support `--dry-run`.

**Note:** SLURM/K8s resource knobs (cpus, mem, time, gpu, parallelism, etc.) are configured in `settings.slurm` / `settings.k8s`.

---

## 4. Additional Notes

**Job directories:** Each submission creates a structured job directory with logs, outputs, and job definitions.

**Resource configuration:** Memory, runtime, CPU/GPU counts are defined in `settings.slurm` or `settings.k8s`.

**Manual job execution:** To manually re-run jobs, use the generated files in the `jobs/` folder.

**Multi-system workflows:** When working across local + HPC systems, ensure `settings.py` is synchronized and the same bb_hpc version is installed in both environments.

---

**Tip:** Always run FileInfo before Save-Detect or Tracking to ensure efficient incremental processing.