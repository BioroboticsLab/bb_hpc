import os, glob
from datetime import timedelta, datetime
import pandas as pd
import fnmatch

from bb_binary.parsing import parse_video_fname
from bb_binary import Repository
from bb_hpc.src.fileinfo import get_bbb_file_path, get_pending_videos, is_bbb_file_valid_basicmatch

#################################################################
##### DETECT 
#################################################################
def generate_jobs_detect(
    video_root_dir,
    repo_output_path,
    RESULTDIR=None,              #  where bbb_fileinfo lives
    slurmdir=None,               # SLURM input-jobs dir
    chunk_size=4,
    maxjobs=None,
    datestring="20*",
    verbose=False,
    video_glob_pattern="cam-*--*Z.mp4",
    use_fileinfo=False,          # use RESULTDIR/bbb_fileinfo/*.parquet
    check_read_bbb=False,        # if True, read .bbb files to verify validity
):
    """
    Yield shards of videos to run detection on.

    Parameters
    ----------
    video_root_dir : str
        Root containing YYYYMMDD/…/cam-*--*Z.mp4
    repo_output_path : str
        Root of the pipeline repo where .bbb files are written.
    RESULTDIR : str or None
        Directory containing bbb_fileinfo/ (with bbb_info_*.parquet). Required if
        use_fileinfo=True. If None and use_fileinfo=True, the function falls back
        to on-disk existence checks.
    slurmdir : str or None
        If set, already-queued videos there are skipped (get_pending_videos()).
    chunk_size : int
        Videos per job shard.
    maxjobs : int or None
        Optional cap: keep only chunk_size * maxjobs videos.
    datestring : str | list[str]
        One or more YYYYMMDD strings (or wildcard like "202506*").
    verbose : bool
        Verbose prints.
    video_glob_pattern : str
        Glob for videos inside each date directory.
    use_fileinfo : bool
        If True, use RESULTDIR/bbb_fileinfo/video_info_all.parquet to source raw
        videos and bbb_info_*.parquet to skip videos whose .bbb already exists.
        Otherwise, fall back to glob + os.path.exists().
    check_read_bbb : bool
        If True, verify .bbb files by reading them (slower). Requires fileinfo
        catalogs with is_valid, or falls back to per-file disk checks.

    Yields
    ------
    dict with keys:
        - "video_paths": list[str]
        - "repo_output_path": str
    """
    import os, glob
    from pathlib import Path
    from itertools import chain
    import pandas as pd
    from bb_binary.parsing import parse_video_fname

    # Expect these helpers to be available in your module:
    # - get_pending_videos(slurmdir) -> set[str]
    # - get_bbb_file_path(video_basename) -> repo-relative .bbb path

    # -------------------------------
    # 1) what's already queued
    # -------------------------------
    pending = set()
    if slurmdir is not None:
        print("getting already submitted videos")
        pending = get_pending_videos(slurmdir)

    # -------------------------------
    # 2) (optional) load BBB catalog
    # -------------------------------
    existing_relpaths = None
    existing_fnames   = None
    if use_fileinfo:
        try:
            if RESULTDIR is None:
                if verbose:
                    print("[fileinfo] RESULTDIR not provided; falling back to disk checks")
            else:
                bbb_info_glob = os.path.join(RESULTDIR, "bbb_fileinfo", "bbb_info_*.parquet")
                info_files = sorted(glob.glob(bbb_info_glob))
                if not info_files:
                    if verbose:
                        print(f"[fileinfo] no catalogs found: {bbb_info_glob} (fallback to disk checks)")
                else:
                    latest = info_files[-1]
                    df = pd.read_parquet(latest)
                    catalog_ok = True
                    if check_read_bbb:
                        if "is_valid" not in df.columns:
                            catalog_ok = False
                            if verbose:
                                print("[fileinfo] missing is_valid; falling back to disk checks")
                        else:
                            df = df[df["is_valid"].fillna(False) == True]
                    # The canonical schema produced by list_bbb_files/build_bbb_info_parquet is:
                    #   file_name, full_path, starttime, endtime, modified_time
                    # We want repo-relative paths; derive them from full_path if they live under repo_output_path.
                    if catalog_ok:
                        existing_relpaths = set()
                        if "full_path" in df.columns:
                            for fp in df["full_path"].astype(str):
                                # Only derive a relpath if the file is actually under repo_output_path
                                # (this avoids ValueError for paths from other repositories).
                                try:
                                    relp = os.path.relpath(fp, repo_output_path)
                                    # relpath() will happily create a path with ".." if fp is outside;
                                    # ensure it's a subpath by rejecting anything that starts with ".."
                                    if not relp.startswith(".."):
                                        existing_relpaths.add(os.path.normpath(relp))
                                except Exception:
                                    pass
                        # Also keep a fallback set of file basenames in case relpath derivation fails
                        if "file_name" in df.columns:
                            existing_fnames = set(df["file_name"].astype(str).tolist())

                    if verbose:
                        print(f"[fileinfo] loaded {latest} rows={len(df)} "
                              f"derived_relpaths={len(existing_relpaths)} "
                              f"(repo_root={repo_output_path})")
        except Exception as e:
            if verbose:
                print(f"[fileinfo] error loading catalog: {e!r} — falling back to disk checks")
            existing_relpaths = None
            existing_fnames   = None  # fallback

    # -------------------------------
    # 3) gather raw mp4s
    # -------------------------------
    all_videos = None
    if use_fileinfo and RESULTDIR is not None:
        video_info_path = os.path.join(RESULTDIR, "bbb_fileinfo", "video_info_all.parquet")
        try:
            df_videofiles = pd.read_parquet(video_info_path)
            if verbose:
                print(f"[fileinfo] loaded {len(df_videofiles)} rows from {video_info_path}")

            # Filter by requested dates using parsed starttime
            dates = set(datestring if isinstance(datestring, (list, tuple)) else [datestring])
            if "starttime" in df_videofiles.columns:
                start_dt = pd.to_datetime(df_videofiles["starttime"], errors="coerce", utc=True)
                mask = start_dt.dt.strftime("%Y%m%d").isin(dates)
                df_videofiles = df_videofiles.loc[mask]
            if "full_path" in df_videofiles.columns:
                all_videos = df_videofiles["full_path"].dropna().astype(str).tolist()
            if not all_videos:
                print(f"[fileinfo] no matching entries for dates {dates}; falling back to glob")
                all_videos = None
        except FileNotFoundError:
            print(f"[fileinfo] catalog not found: {video_info_path}; falling back to glob")
        except Exception as e:
            print(f"[fileinfo] error reading {video_info_path}: {e}; falling back to glob")

    if all_videos is None:
        print("getting new mp4 files")
        if isinstance(datestring, (list, tuple)):
            days = list(datestring)
        else:
            days = [datestring]

        patterns = (
            os.path.join(video_root_dir, day, "**", video_glob_pattern)
            for day in days
        )
        all_videos = list(chain.from_iterable(glob.iglob(p, recursive=True) for p in patterns))
        print("found", len(all_videos), "files")

    # -------------------------------
    # 4) filter (pending + already processed)
    # -------------------------------
    print("making list to submit")
    to_schedule = []
    join = os.path.join
    for v in all_videos:
        if v in pending:
            continue

        rel = get_bbb_file_path(os.path.basename(v))  # repo-relative expected .bbb

        # Prefer fast lookups from catalog if available
        if (existing_relpaths is not None) or (existing_fnames is not None):
            already = (existing_relpaths is not None and rel in existing_relpaths) \
                      or (existing_fnames is not None and os.path.basename(rel) in existing_fnames)
            if already:
                if verbose:
                    print("already exists (catalog):", rel)
                continue
        else:
            # Disk check fallback
            full_bbb = join(repo_output_path, rel)
            if is_bbb_file_valid_basicmatch(full_bbb, check_read_file=check_read_bbb):
                if verbose:
                    print("already exists (disk):", rel)
                continue

        if verbose:
            print("adding:", v)
        to_schedule.append(v)

    print("number of videos to schedule:", len(to_schedule))

    # Sort quick, then apply global cap
    to_schedule.sort()
    if maxjobs is not None:
        to_schedule = to_schedule[: int(chunk_size) * int(maxjobs)]
    print("number of videos kept with max jobs", len(to_schedule))

    # -------------------------------
    # 5) parse timestamps, sort properly, and chunk
    # -------------------------------
    video_tuples = []
    for v in to_schedule:
        cam_id, dt_start, _ = parse_video_fname(os.path.basename(v))
        video_tuples.append((dt_start, cam_id, v))
    video_tuples.sort()

    # -------------------------------
    # 6) yield batches
    # -------------------------------
    for i in range(0, len(video_tuples), int(chunk_size)):
        chunk = video_tuples[i : i + int(chunk_size)]
        yield {
            "video_paths": [path for _, _, path in chunk],
            "repo_output_path": repo_output_path,
        }     


#################################################################
##### SAVE DETECT (date-based, daily windows)
#################################################################
def generate_jobs_save_detect(
    RESULTDIR,
    PIPELINE_ROOT,
    datestring,                 # e.g. ["20250901","20250902", ...]
    chunk_size=50,
    maxjobs=None,
    interval_hours: int = 1,
):
    """
    Generate save-detect jobs per camera over fixed hourly windows inside the
    given YYYYMMDD dates (default: 1-hour windows).
    Uses fileinfo parquet(s) to see what's in the repo and compares against
    previously saved outputs to avoid redoing work.

    Yields dicts: { "job_args_list": [ {repo_path, save_path, from_dt, to_dt, cam_id}, ... ] }
    """
    import os, glob
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    from bb_binary import parse_video_fname

    # Where aggregated parquet outputs live
    save_dir = os.path.join(RESULTDIR, 'data_alldetections/')
    os.makedirs(save_dir, exist_ok=True)

    # Fileinfo & cache
    bbb_info_glob      = os.path.join(RESULTDIR, 'bbb_fileinfo/bbb_info_*.parquet')
    detect_cache_file  = os.path.join(RESULTDIR, 'bbb_fileinfo/save_detect_outinfo.parquet')

    if (len(glob.glob(bbb_info_glob)) == 0) or (not os.path.exists(detect_cache_file)):
        print('Please run fileinfo first to create bbb_info and save_detect_outinfo files')
        return

    df_bbb = pd.read_parquet(sorted(glob.glob(bbb_info_glob))[-1])   # latest snapshot
    df_out = pd.read_parquet(detect_cache_file)

    # Map of already-produced windows: (cam_id, from_dt, to_dt) -> modified_time
    out_index = {
        (r['cam_id'], r['from_dt'], r['to_dt']): r['modified_time']
        for _, r in df_out.iterrows()
    }

    # Ensure cam_id exists in df_bbb (derive from filename if needed)
    if 'cam_id' not in df_bbb.columns:
        try:
            df_bbb = df_bbb.copy()
            df_bbb['cam_id'] = df_bbb['file_name'].apply(lambda fn: parse_video_fname(fn)[0])
        except Exception:
            # Fallback: try to parse simple "cam-<id>_..." prefixes
            import re
            def _fallback_cam(fn):
                m = re.match(r'(?:cam-)?(\d+)_', fn)
                return int(m.group(1)) if m else -1
            df_bbb = df_bbb.copy()
            df_bbb['cam_id'] = df_bbb['file_name'].apply(_fallback_cam)

    # Build daily windows from datestring
    def _to_utc_day(dstr: str):
        dt0 = datetime.strptime(dstr, "%Y%m%d").replace(tzinfo=timezone.utc)
        return dt0, dt0 + timedelta(days=1)

    # Hourly (or configurable) windows within a day
    interval = timedelta(hours=int(interval_hours))
    if interval.total_seconds() <= 0:
        raise ValueError("interval_hours must be positive")

    def _iter_windows(day_start, day_end):
        t = day_start
        while t < day_end:
            nxt = min(t + interval, day_end)
            yield t, nxt
            t = nxt

    # Collect candidates (per-date, per-cam) where:
    # - there is at least one repo file overlapping that day for that cam
    # - output either missing or older than latest source
    candidates = []
    for d in datestring:
        day_start, day_end = _to_utc_day(d)
        # Videos overlapping this day window
        sel = df_bbb[(df_bbb['starttime'] < day_end) & (df_bbb['endtime'] > day_start)]
        if sel.empty:
            continue

        for cam_id, subset in sel.groupby('cam_id'):
            for win_start, win_end in _iter_windows(day_start, day_end):
                # Only schedule windows that overlap at least one repo file
                window_data = subset[(subset['starttime'] < win_end) & (subset['endtime'] > win_start)]
                if window_data.empty:
                    continue

                latest_src = window_data['modified_time'].max()
                key = (cam_id, win_start, win_end)
                existing = out_index.get(key)
                if (existing is None) or (latest_src > existing):
                    candidates.append({
                        'repo_path': PIPELINE_ROOT,
                        'save_path': save_dir,
                        'from_dt':   win_start,
                        'to_dt':     win_end,
                        'cam_id':    cam_id,
                    })

    # Sort date-ordered (by from_dt) then cam_id
    candidates.sort(key=lambda x: (x['from_dt'], x['cam_id']), reverse=False)

    # Optional cap like detect(): only keep first chunk_size * maxjobs
    if maxjobs is not None:
        keep = int(chunk_size) * int(maxjobs)
        candidates = candidates[:keep]

    # Emit in chunks
    print(f"[save_detect] dates={len(datestring)} candidates={len(candidates)} "
          f"chunk_size={chunk_size} maxjobs={maxjobs} interval_hours={interval_hours}")
    for i in range(0, len(candidates), int(chunk_size)):
        yield {"job_args_list": candidates[i:i+int(chunk_size)]}

#################################################################
##### TRACKING (date-based, hourly windows, chunked)
#################################################################
def generate_jobs_tracking(
    RESULTDIR: str,
    PIPELINE_ROOT: str,
    TEMP_DIR: str,
    datestring,                 # e.g. ["20160719", "20160720", ...]
    chunk_size: int = 50,
    maxjobs=None,               # None = unlimited; else cap to chunk_size * maxjobs
    interval_hours: int = 1,    # tracking window size (hours)
):
    """
    Build tracking jobs per camera over HOURLY windows inside the given YYYYMMDD dates.

    Reads:
      - Latest BBB catalog parquet:   {RESULTDIR}/bbb_fileinfo/bbb_info_*.parquet
      - Tracking out-info parquet:    {RESULTDIR}/bbb_fileinfo/save_tracking_outinfo.parquet

    Emits batches like:
      { "job_args_list": [
          { "repo_path": PIPELINE_ROOT, "save_path": <RESULTDIR>/data_tracked/,
            "from_dt": <UTC datetime>, "to_dt": <UTC datetime>, "cam_id": <int> },
          ...
        ]
      }
    """
    import os, glob
    import pandas as pd
    from datetime import datetime, timedelta, timezone
    from bb_binary.parsing import parse_video_fname

    # ---- Paths (from settings via args) ----
    save_dir = os.path.join(RESULTDIR, "data_tracked")
    os.makedirs(save_dir, exist_ok=True)

    bbb_info_glob     = os.path.join(RESULTDIR, "bbb_fileinfo", "bbb_info_*.parquet")
    track_cache_file  = os.path.join(RESULTDIR, "bbb_fileinfo", "save_tracking_outinfo.parquet")

    # ---- Required inputs ----
    info_files = sorted(glob.glob(bbb_info_glob))
    if not info_files or not os.path.exists(track_cache_file):
        print("Please run get_fileinfo.py first (bbb_info_*.parquet and save_tracking_outinfo.parquet).")
        return

    df_bbb = pd.read_parquet(info_files[-1])          # latest snapshot
    df_out = pd.read_parquet(track_cache_file)

    # Ensure we have cam_id in df_bbb
    if "cam_id" not in df_bbb.columns:
        try:
            df_bbb = df_bbb.copy()
            df_bbb["cam_id"] = df_bbb["file_name"].apply(lambda fn: parse_video_fname(fn)[0])
        except Exception:
            # very conservative fallback (cam-<id>_...)
            import re
            def _fallback_cam(fn):
                m = re.match(r"(?:cam-)?(\d+)_", str(fn))
                return int(m.group(1)) if m else -1
            df_bbb = df_bbb.copy()
            df_bbb["cam_id"] = df_bbb["file_name"].apply(_fallback_cam)

    # Build “already produced” index
    out_index = {
        (row["cam_id"], row["from_dt"], row["to_dt"]): row["modified_time"]
        for _, row in df_out.iterrows()
    }

    # ---- Build hourly windows across the requested dates (UTC) ----
    interval = timedelta(hours=int(interval_hours))

    def _day_bounds_utc(yyyymmdd: str):
        # Full UTC day window [00:00, 24:00) and already hour-aligned
        start = datetime.strptime(yyyymmdd, "%Y%m%d").replace(tzinfo=timezone.utc)
        end   = start + timedelta(days=1)
        return start, end

    windows = []  # list of (win_start, win_end)
    for d in datestring:
        day_start, day_end = _day_bounds_utc(d)
        cur = day_start
        while cur < day_end:
            win_start = cur
            win_end   = min(day_end, cur + interval)
            # Hour alignment (defensive; day_start/day_end already hour-aligned)
            win_start = (pd.Timestamp(win_start).floor("h")).to_pydatetime().replace(tzinfo=timezone.utc)
            win_end   = (pd.Timestamp(win_end).floor("h")).to_pydatetime().replace(tzinfo=timezone.utc)
            if win_end > win_start:
                windows.append((win_start, win_end))
            cur += interval

    # Sort oldest-first, to process in order (note: reverse=False would process newest first)
    windows.sort(key=lambda w: w[0], reverse=False)

    # ---- Collect candidates: per-hour per-cam if missing/stale ----
    candidates = []
    for win_start, win_end in windows:
        sel = df_bbb[(df_bbb["starttime"] < win_end) & (df_bbb["endtime"] > win_start)]
        if sel.empty:
            continue

        for cam_id, subset in sel.groupby("cam_id"):
            if subset.empty:
                continue

            latest_src = subset["modified_time"].max()
            key = (cam_id, win_start, win_end)
            existing = out_index.get(key)

            if (existing is None) or (latest_src > existing):
                candidates.append({
                    "repo_path": PIPELINE_ROOT,
                    "save_path": save_dir,
                    "temp_path": TEMP_DIR,  
                    "from_dt":   win_start,
                    "to_dt":     win_end,
                    "cam_id":    cam_id,
                })

    # Oldest-first, stable by cam_id
    candidates.sort(key=lambda x: (x["from_dt"], x["cam_id"]), reverse=False)

    # Optional cap to first chunk_size * maxjobs
    if maxjobs is not None:
        keep = int(chunk_size) * int(maxjobs)
        candidates = candidates[:keep]

    # ---- Emit in chunks ----
    print(f"[tracking] dates={len(datestring)} windows={len(windows)} "
          f"candidates={len(candidates)} chunk_size={chunk_size} maxjobs={maxjobs} "
          f"interval_h={interval_hours}")

    for i in range(0, len(candidates), int(chunk_size)):
        yield {"job_args_list": candidates[i:i + int(chunk_size)]}

#################################################################
##### RPI - DETECT
#################################################################
def _resolve_model_type(cam_id, cam_model_rules):
    """Return model type string for a cam_id using prefix matching (first match wins)."""
    for prefix, model_type in cam_model_rules.items():
        if cam_id.startswith(prefix):
            return model_type
    return "default"


def _parse_rpi_timestamp(basename):
    """Parse timestamp from RPi video basename; fallback to None."""
    try:
        ts_str = os.path.splitext(basename)[0].rsplit("_", 1)[-1]
        return datetime.strptime(ts_str, "%Y-%m-%d-%H-%M-%S")
    except Exception:
        return None


def generate_jobs_rpi_detect(
    video_root_dir,
    dates,
    chunk_size=150,
    maxjobs=None,
    video_glob_pattern="*.h264",
    clahe=True,
    cam_model_rules=None,
):
    """
    Yields shards of RPi video paths under date directories using fast globbing.
    Excludes videos already processed (suffix depends on CLAHE and model type).

    When cam_model_rules is provided, videos are partitioned by model type so each
    shard contains only one type.  Each yielded dict includes a "model_type" key.

    - Oldest videos first (sorted by timestamp parsed from filename).
    - If parsing fails, falls back to file mtime.
    - If maxjobs is set, only keeps chunk_size * maxjobs videos per model type.
    """
    from collections import defaultdict

    # 1) glob all raw h264’s for requested dates
    print("getting new h264 files")
    all_videos = []
    for day in dates:  # e.g., "20250710"
        pattern = os.path.join(video_root_dir, day, "**", video_glob_pattern)
        print(pattern)
        all_videos.extend(glob.glob(pattern, recursive=True))
    print("found", len(all_videos), "files")

    # 2) partition by model type and filter already-processed
    print("making list to submit")
    clahe_suffix = "-c" if clahe else "-nc"
    by_model = defaultdict(list)

    for v in all_videos:
        basename = os.path.basename(v)
        cam_id = basename.split("_")[0]
        model_type = _resolve_model_type(cam_id, cam_model_rules) if cam_model_rules else "default"

        # Check if already processed (suffix depends on model type)
        if model_type == "polo":
            det_file = os.path.splitext(v)[0] + f"-detections-polo{clahe_suffix}.parquet"
        else:
            det_file = os.path.splitext(v)[0] + f"-detections{clahe_suffix}.parquet"

        if not os.path.exists(det_file):
            by_model[model_type].append(v)

    for model_type, videos in sorted(by_model.items()):
        print(f"  model_type={model_type}: {len(videos)} videos to schedule")

    # 3) per model type: sort by timestamp, apply maxjobs cap, yield shards
    for model_type, videos in sorted(by_model.items()):
        video_tuples = []
        for v in videos:
            base = os.path.basename(v)
            ts = _parse_rpi_timestamp(base)
            if ts is None:
                try:
                    ts = datetime.fromtimestamp(os.path.getmtime(v))
                except Exception:
                    ts = datetime.min
            video_tuples.append((ts, v))

        # oldest first
        video_tuples.sort(key=lambda t: t[0], reverse=False)

        if maxjobs is not None:
            keep = chunk_size * int(maxjobs)
            video_tuples = video_tuples[:keep]

        print(f"  model_type={model_type}: {len(video_tuples)} videos kept after maxjobs")

        for i in range(0, len(video_tuples), chunk_size):
            chunk = video_tuples[i : i + chunk_size]
            yield {
                "video_paths": [path for _, path in chunk],
                "model_type": model_type,
            }


#################################################################
##### FRAME EXTRACT (cell-seg heavy preprocessing)
#################################################################
def generate_jobs_frame_extract(
    video_root_dir,        # e.g. settings.videodir_hpc
    frames_root_dir,       # e.g. settings.frames_dir_hpc
    datestring,            # list of YYYYMMDD (or a single string)
    interval_in_sec=60,
    fps=3,
    file_format="png",
    decoder="hevc_cuvid",
    max_workers=2,
    chunk_size=4,
    maxjobs=None,
    verbose=False,
):
    """
    Yield chunks of (date, camera) frame-extraction work units.

    Enumerates ``video_root/<date>/cam-N/`` subtrees (the same layout detect
    uses) and skips a (date, cam) only when EVERY expected output frame already
    exists -- a per-filename check via
    ``frame_extractor.naming.expected_frame_filenames``. Because a coarser
    interval that is a multiple of a finer one produces a subset of the finer
    run's filenames, re-running a finished date at a coarser interval schedules
    nothing. Falls back to a presence check if the cell-seg package is not
    importable on the submitter.

    Yields
    ------
    dict: {"work_units": [ {date, cam, video_root, frames_root, interval_in_sec,
                            fps, file_format, decoder, max_workers}, ... ]}
    """
    import os, glob
    from pathlib import Path

    try:
        from frame_extractor.naming import expected_frame_filenames
    except Exception as e:
        expected_frame_filenames = None
        print(f"[frame_extract] frame_extractor not importable ({e!r}); using presence-based skip")

    days = list(datestring) if isinstance(datestring, (list, tuple)) else [datestring]

    units = []
    for date in days:
        date_dir = os.path.join(video_root_dir, date)
        if not os.path.isdir(date_dir):
            if verbose:
                print(f"[frame_extract] no date dir: {date_dir}")
            continue

        cam_dirs = sorted(d for d in glob.glob(os.path.join(date_dir, "cam-*")) if os.path.isdir(d))
        for cam_dir in cam_dirs:
            cam = os.path.basename(cam_dir)
            mp4s = sorted(glob.glob(os.path.join(cam_dir, "*.mp4")))
            txts = [Path(v).with_suffix(".txt") for v in mp4s]
            txts = [t for t in txts if t.exists()]
            if not txts:
                continue

            out_dir = os.path.join(frames_root_dir, date, cam)

            done = False
            if expected_frame_filenames is not None:
                try:
                    expected = expected_frame_filenames(txts, interval_in_sec, fps, file_format)
                    if expected:
                        existing = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
                        done = expected.issubset(existing)
                except Exception as e:
                    if verbose:
                        print(f"[frame_extract] skip-check error {cam_dir}: {e!r}")
            else:
                done = os.path.isdir(out_dir) and any(
                    fn.endswith(f".{file_format}") for fn in os.listdir(out_dir)
                )

            if done:
                if verbose:
                    print(f"[frame_extract] done: {date}/{cam}")
                continue

            units.append({
                "date": date,
                "cam": cam,
                "video_root": video_root_dir,
                "frames_root": frames_root_dir,
                "interval_in_sec": interval_in_sec,
                "fps": fps,
                "file_format": file_format,
                "decoder": decoder,
                "max_workers": max_workers,
            })

    units.sort(key=lambda u: (u["date"], u["cam"]))
    if maxjobs is not None:
        units = units[: int(chunk_size) * int(maxjobs)]

    print(f"[frame_extract] dates={len(days)} units={len(units)} "
          f"chunk_size={chunk_size} maxjobs={maxjobs} interval={interval_in_sec}s")

    for i in range(0, len(units), int(chunk_size)):
        yield {"work_units": units[i:i + int(chunk_size)]}


#################################################################
##### BACKGROUND (cell-seg heavy preprocessing)
#################################################################
def generate_jobs_background(
    frames_root_dir,        # e.g. settings.frames_dir_hpc
    backgrounds_root_dir,   # e.g. settings.backgrounds_dir_hpc
    datestring,             # list of YYYYMMDD (or a single string)
    frame_interval_sec=None,
    background_window=None,
    window_size=10,
    num_median_images=200,
    max_cycles=None,
    jump_size=1,
    apply_clahe="post",
    mask_dilation=15,
    median_computation="cupy",
    device="cuda",
    memmap_dir=None,
    chunk_size=2,
    maxjobs=None,
    verbose=False,
):
    """
    Yield chunks of (date, camera) background-generation work units over cameras
    whose extracted frames exist under ``frames_root/<date>/cam-N/``.

    The output config tag (interval/window settings) is encoded in the output
    path, so a (date, cam) is skipped when the backgrounds for THIS config
    already exist:
      - window mode: every expected per-window ``background_<wstart>.png`` exists
        (via ``background_generator.windowing.expected_background_names``)
      - count mode: the config-tag dir already contains >= 1 ``background_*.png``

    Yields
    ------
    dict: {"work_units": [ {date, cam, frames_root, backgrounds_root, <bg knobs>}, ... ]}
    """
    import os, glob

    try:
        from background_generator import windowing
    except Exception as e:
        windowing = None
        print(f"[background] background_generator not importable ({e!r}); using presence-based skip")

    days = list(datestring) if isinstance(datestring, (list, tuple)) else [datestring]

    if windowing is not None:
        tag = windowing.config_tag(frame_interval_sec, background_window, window_size, num_median_images)
    else:
        interval = frame_interval_sec or 0
        tag = (f"int{interval}s_win{background_window}" if background_window
               else f"count_w{window_size}_n{num_median_images}_int{interval}s")

    units = []
    for date in days:
        date_dir = os.path.join(frames_root_dir, date)
        if not os.path.isdir(date_dir):
            if verbose:
                print(f"[background] no frames for date: {date_dir}")
            continue

        cam_dirs = sorted(d for d in glob.glob(os.path.join(date_dir, "cam-*")) if os.path.isdir(d))
        for cam_dir in cam_dirs:
            cam = os.path.basename(cam_dir)
            frame_names = [fn for fn in os.listdir(cam_dir) if fn.endswith(".png")]
            if not frame_names:
                continue

            out_dir = os.path.join(backgrounds_root_dir, date, cam, tag)

            done = False
            if background_window and windowing is not None:
                expected = windowing.expected_background_names(frame_names, frame_interval_sec, background_window)
                if expected:
                    existing = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
                    done = expected.issubset(existing)
            else:
                done = os.path.isdir(out_dir) and any(
                    fn.startswith("background_") and fn.endswith(".png") for fn in os.listdir(out_dir)
                )

            if done:
                if verbose:
                    print(f"[background] done: {date}/{cam} [{tag}]")
                continue

            units.append({
                "date": date,
                "cam": cam,
                "frames_root": frames_root_dir,
                "backgrounds_root": backgrounds_root_dir,
                "frame_interval_sec": frame_interval_sec,
                "background_window": background_window,
                "window_size": window_size,
                "num_median_images": num_median_images,
                "max_cycles": max_cycles,
                "jump_size": jump_size,
                "apply_clahe": apply_clahe,
                "mask_dilation": mask_dilation,
                "median_computation": median_computation,
                "device": device,
                "memmap_dir": memmap_dir,
            })

    units.sort(key=lambda u: (u["date"], u["cam"]))
    if maxjobs is not None:
        units = units[: int(chunk_size) * int(maxjobs)]

    print(f"[background] dates={len(days)} units={len(units)} chunk_size={chunk_size} "
          f"maxjobs={maxjobs} tag={tag}")

    for i in range(0, len(units), int(chunk_size)):
        yield {"work_units": units[i:i + int(chunk_size)]}
