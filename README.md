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
python -m pip install git+https://github.com/walachey/slurmhelper
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
--what  {all, bbb, outputs, rpi}  # Which caches to rebuild (default: all)
--use-cache / --no-use-cache  # Use incremental BBB cache (default: use)
--check-read-bbb              # Read .bbb files to validate (adds is_valid; slower)
--deep-check-bbb              # Read all frames for each .bbb (slowest; catches premature EOF)
```

**Examples:**

```bash
# Update only bb_binary cache using HPC paths (e.g., on FU Berlin Curta)
python -m bb_hpc.get_fileinfo --paths hpc --what bbb

# Update only outputs using local paths (quick refresh before new jobs)
python -m bb_hpc.get_fileinfo --paths local --what outputs

# Force a full re-scan and validate .bbb files
python -m bb_hpc.get_fileinfo --what bbb --no-use-cache --check-read-bbb

# Deep validation (scan all frames)
python -m bb_hpc.get_fileinfo --what bbb --no-use-cache --deep-check-bbb
```

After FileInfo is updated, Save-Detect and Tracking will automatically skip already-processed windows.

### VideoInfo (catalog of raw videos)

`get_fileinfo` catalogs *outputs*. `get_videoinfo` catalogs the *raw videos*, writing
`<resultdir>/bbb_fileinfo/video_info_all.parquet`. It is what `detect_submit --use-fileinfo`
reads to decide which videos still need detection, and what the progress report uses as
the denominator for detect coverage.

```bash
python -m bb_hpc.get_videoinfo
```

It is **incremental** by default: per-day catalogs are cached under
`bbb_fileinfo/daily/video_YYYYMMDD.parquet`, and a day is rescanned only when it contains
files newer than its cache. During a season this makes a refresh cost seconds instead of a
full recursive walk of every day.

```
--use-cache / --no-use-cache   # Incremental (default) vs full rescan of every day
--force-recent-days N          # Always rescan the N newest days (default: 2)
--exts .mp4 [...]              # Extensions to catalog (default: .mp4)
--outfile PATH                 # Override the output parquet path
```

The two newest days are always rescanned because videos are still landing in them; the
mtime comparison alone can race with a file written during the scan.

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
- `--video-glob PATTERN` — Override video glob (default: `cam-*--*Z.mp4`)

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

## 3b. Cell-seg heavy preprocessing (frame extraction + background generation)

Two GPU stages that wrap the heavy "do-once-first" steps of the
[honeybee_cell_segmentation_pipeline](https://github.com/BioroboticsLab/honeybee_cell_segmentation_pipeline)
(`heavy_preprocessing/frame_extractor` and `heavy_preprocessing/background_generator`)
so they scale across the cluster like detect/tracking. Run **frame extraction first**,
then **background generation** for the same dates.

**Prerequisite:** the cell-seg tools must be importable in the run environment.
On SLURM this is the `beesbook` conda env (installed by
`bb_main/code/install_update_beesbook_pipeline.sh`); on K8s/Docker use the
`comb-background` image (`building_docker/Dockerfile-comb-background`).

```bash
# SLURM
python -m bb_hpc.running_slurm.frame_extract_submit --dates 20250603 20250604
python -m bb_hpc.running_slurm.background_submit    --dates 20250603 20250604

# Kubernetes (GPU)
python -m bb_hpc.running_k8s.frame_extract_submit --dates 20250603
python -m bb_hpc.running_k8s.background_submit    --dates 20250603
```

Both accept `--dates` and `--dry-run`. Work is sharded by **(date, camera)**;
`chunk_size` bundles several units per task. All parameters live in
`settings.frame_extract_settings` / `settings.background_settings`:

- **Frame extraction:** `interval_in_sec` (seconds between frames — the key
  knob), `fps`, `file_format`, `max_workers`, `decoder` (`hevc_cuvid` for
  NVIDIA NVDEC, or `None`/`"none"` for CPU software decode).
- **Background generation:** `frame_interval_sec` (subsample frames, e.g. compare
  5-min vs 10-min backgrounds), `background_window` (`"hour"`/`"day"`/seconds —
  one background per window; `None` = count-based rolling mode), plus
  `window_size`, `num_median_images`, `mask_dilation`, `median_computation`,
  `device`, etc.

**Skip / "what's left to do":** both stages skip already-done `(date, camera)`
units by checking the **expected output filenames**. For extraction this means a
coarser interval that is a multiple of a finished finer run schedules nothing
(e.g. 10-min after 5-min). Backgrounds encode the interval/window config in the
output path (`data_backgrounds/<date>/cam-N/<config-tag>/`), so different configs
are distinct, comparable products and repeat configs skip. Use `--dry-run` to see
the pending unit count before submitting.

**Outputs:** `settings.frames_dir_*` (e.g. `results/data_extracted_frames/<date>/cam-N/`)
and `settings.backgrounds_dir_*` (e.g. `results/data_backgrounds/<date>/cam-N/<config-tag>/`).

**GPU note:** both stages request `gres=gpu:1`. Frame extraction's default
`hevc_cuvid` needs NVDEC (a CUDA build of ffmpeg); on CPU-only nodes set
`decoder=None` and drop the gres. Background generation needs a CUDA-12 runtime
for `cupy-cuda12x` (or set `median_computation="masked_array"`, `device="cpu"`).

Tune sizing after a first batch:
`python -m bb_hpc.slurm_report --name frame_extract --target-walltime-min 480`.

---

## 4. Progress reporting

See what has processed, what has not, and get the exact commands to fill the gaps.

```bash
python -m bb_hpc.progress_report
```

This reads the fileinfo catalogs, classifies **every work unit of every stage** as
done / pending / skipped, prints a one-screen report, and saves an atomic snapshot to
`<resultdir>/bbb_fileinfo/progress/latest/`.

Completion is decided by the same predicates the submitters use (`bb_hpc.src.generate`),
so a unit reported pending here is exactly a unit `<stage>_submit` would schedule.
`test/test_progress.py` pins that invariant.

| stage | work unit | statuses |
|---|---|---|
| `detect` | one raw video | `done` / `zero_byte_stub` / `missing` |
| `save_detect` | `(cam_id, hour window)` | `done` / `stale` / `missing` |
| `tracking` | `(cam_id, hour window)` | `done` / `stale` / `missing` |
| `frame_extract` | `(date, cam)` | `done` / `pending` / `skipped_no_txt` |
| `background` | `(date, cam)` | `done` / `pending` / `skipped_no_frames` / `skipped_min_frames` |
| `rpi` | one RPi video | `done` / `missing` |

`stale` means the output is older than a `.bbb` in its window, so it will be redone.
*Skipped* units can never be produced (a camera with no frames cannot yield a background),
so they are excluded from the percentage and reported separately rather than dragging
coverage down forever.

### Options

```
--dates YYYYMMDD [...]           # Explicit dates (default: every date in the catalogs)
--since / --until YYYYMMDD       # Inclusive date range
--last-days N                    # Only the N most recent dates
--stages detect tracking ...     # Subset of stages (default: all)
--backend {k8s, slurm, docker}   # Which running_<backend> module the printed commands target
--refresh {none, videos, fileinfo, all}   # Rebuild catalogs first (default: none)
--paths {auto, local, hpc}       # Which settings path set to use
--markdown                       # Render the report as markdown
--csv PATH                       # Also write the per-day tables to CSV
--no-save                        # Print only; write nothing
--exit-nonzero-if-pending        # Exit 1 when anything is pending (for cron alerting)
```

**Examples:**

```bash
# fast: read whatever the fileinfo cron produced
python -m bb_hpc.progress_report

# refresh the (incremental) video catalog first, look at the last week
python -m bb_hpc.progress_report --refresh videos --last-days 7

# only the comb stages, targeting the slurm submitters
python -m bb_hpc.progress_report --stages frame_extract background --backend slurm
```

The report **prints** resubmit commands and writes them to `commands.sh`. It never submits.

### Artifacts

```
<resultdir>/bbb_fileinfo/progress/
├── latest/                    # rewritten in place each run (never a symlink: EIO on CIFS)
│   ├── summary.json           # one-glance totals per stage
│   ├── report.md              # the rendered report
│   ├── commands.sh            # resubmit commands for everything unfinished
│   ├── <stage>_units.parquet  # one row per work unit, with its status
│   └── <stage>_by_day.parquet # day, total, done, pending, skipped, pct
└── history.jsonl              # append-only: one summary row per run -> progress over time
```

### Notebook

`notebooks/pipeline_progress.ipynb` **loads** the snapshot — it recomputes nothing, so its
cells can never disagree with each other. Use it to drill into `stage.units`, list pending
days, and copy resubmit commands.

```python
from bb_hpc.src.progress import load_latest, load_history

rep = load_latest()
rep.summary()
rep.stages["tracking"].pending_units()
rep.commands()                       # or rep.commands(backend="docker")
load_history()                       # % complete per stage, over time
```

A typical during-season cadence is a daily cron running
`python -m bb_hpc.progress_report --refresh videos`, then opening the notebook when
something looks wrong.

---

## 5. Additional Notes

**Job directories:** Each submission creates a structured job directory with logs, outputs, and job definitions.

**Resource configuration:** Memory, runtime, CPU/GPU counts are defined in `settings.slurm` or `settings.k8s`.

**Manual job execution:** To manually re-run jobs, use the generated files in the `jobs/` folder.

**Multi-system workflows:** When working across local + HPC systems, ensure `settings.py` is synchronized and the same bb_hpc version is installed in both environments.

**BBB validity checks:** Detect submitters accept `--use-fileinfo` and `--check-read-bbb` to avoid reprocessing videos with missing/invalid `.bbb` outputs (slower, but safer).

**Cleanup invalid BBB files (by date):**
```bash
# Dry-run scan for a specific day
python -m bb_hpc.scan_and_remove_invalid_bbb_files --dates 20160819 --dry-run

# Remove unreadable .bbb files for one or more days
python -m bb_hpc.scan_and_remove_invalid_bbb_files --dates 20160819 20160820

# Deep scan (reads through all frames)
python -m bb_hpc.scan_and_remove_invalid_bbb_files --dates 20160819 --deep-check-bbb
```

---

**Tip:** Always run FileInfo before Save-Detect or Tracking to ensure efficient incremental processing.
