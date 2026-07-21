import os, glob
from datetime import timedelta, datetime
import pandas as pd
import fnmatch

from bb_binary.parsing import parse_video_fname
from bb_binary import Repository
from bb_hpc.src.fileinfo import (get_pending_videos, is_bbb_file_valid_basicmatch,
                                 get_bbb_start_key, get_bbb_bucket_and_prefix)

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
    # - get_bbb_start_key(basename) -> "<cam>|<start second>" match key
    # - get_bbb_bucket_and_prefix(basename) -> (bucket dir, .bbb name prefix)

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
    existing_start_keys = None
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
                    # We key on cam id + start second rather than on the filename or
                    # a derived relpath: bb_binary names its output from the ACTUAL
                    # frame timestamps, so a gappy recording's .bbb has an end the
                    # video's name never predicted and an exact-name match would
                    # re-schedule an already-detected video forever.
                    if catalog_ok:
                        existing_start_keys = set(bbb_start_key_series(df).dropna())

                    if verbose:
                        print(f"[fileinfo] loaded {latest} rows={len(df)} "
                              f"start_keys={len(existing_start_keys or ())} "
                              f"(repo_root={repo_output_path})")
        except Exception as e:
            if verbose:
                print(f"[fileinfo] error loading catalog: {e!r} — falling back to disk checks")
            existing_start_keys = None  # fallback

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

        base = os.path.basename(v)

        # Prefer fast lookups from catalog if available
        if existing_start_keys is not None:
            if _safe_start_key(base) in existing_start_keys:   # O(1)
                if verbose:
                    print("already exists (catalog):", base)
                continue
        else:
            # Disk check fallback: glob the bucket for this cam id + start second
            bucket, prefix = get_bbb_bucket_and_prefix(base)
            if _disk_has_primary(join(repo_output_path, bucket), prefix, check_read_bbb):
                if verbose:
                    print("already exists (disk):", base)
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
##### SHARED COMPLETION PREDICATES (hourly-window stages)
#################################################################
# These are the single source of truth for "is this (cam, hour) work unit done?".
# generate_jobs_save_detect / generate_jobs_tracking build their job dicts from
# window_candidates(); bb_hpc.src.progress reports on the same records. Keeping
# one implementation is what stops the progress report from drifting away from
# what the submitters actually schedule.
#
# All functions here are WRITE-FREE, so a read-only report can call them.

_CAM_RE = r"^[Cc]am[_-](\d+)"


def _parse_cam_id(file_name):
    """cam_id from a video/.bbb basename; None when unparsable."""
    try:
        return int(parse_video_fname(str(file_name))[0])
    except Exception:
        return None


def ensure_cam_id(df, column="file_name"):
    """
    Return a copy of df with an int cam_id column.

    The bbb catalog (list_bbb_files) does not store cam_id, and its incremental
    daily cache is trusted on mtime alone -- so a cached cam_id column would be a
    silent mix of real values and NaN. We always derive it here instead: one
    vectorized regex over the filename, falling back to parse_video_fname only
    for the rows the regex misses.
    """
    if "cam_id" in df.columns and df["cam_id"].notna().all():
        return df

    df = df.copy()
    cam = pd.to_numeric(
        df[column].astype(str).str.extract(_CAM_RE, expand=False), errors="coerce"
    )
    missing = cam.isna()
    if missing.any():
        cam.loc[missing] = df.loc[missing, column].map(_parse_cam_id)
    df["cam_id"] = cam
    return df


def bbb_start_key_series(df, column="file_name"):
    """
    Vectorized "cam id + start second" key for a bbb or video catalog.

    This is the join key for "is this video already detected?". It deliberately
    excludes the end timestamp: bb_binary names its output from the ACTUAL frame
    timestamps, so a gappy recording's .bbb carries an end the video's filename
    never predicted (see src/fileinfo.get_bbb_bucket_and_prefix for the full why).

    One vectorized pass, not one parse per row: both catalogs already carry a
    parsed `starttime` (fileinfo.list_bbb_files / list_video_day), and
    ensure_cam_id derives cam_id with a single regex. Keeping this vectorized is
    what preserves the O(1)-per-video lookup in the submit scanner.
    """
    if df is None or len(df) == 0:
        return pd.Series(dtype=object)

    d = ensure_cam_id(df, column=column)
    start = pd.to_datetime(d["starttime"], utc=True, errors="coerce") \
        if "starttime" in d.columns else pd.Series(pd.NaT, index=d.index)
    cam = pd.to_numeric(d["cam_id"], errors="coerce")

    # Legacy catalogs predate the starttime column. Without this fallback every
    # key would be NA and every video would look un-detected and re-schedule.
    if start.isna().all() and column in d.columns:
        print("[fileinfo] catalog has no usable 'starttime'; deriving start keys "
              "from filenames (slower, one-time)")
        return d[column].astype(str).map(
            lambda fn: _safe_start_key(fn)
        )

    out = pd.Series(pd.NA, index=d.index, dtype=object)
    ok = start.notna() & cam.notna()
    if ok.any():
        out.loc[ok] = (cam[ok].astype("int64").astype(str) + "|"
                       + start[ok].dt.strftime("%Y-%m-%dT%H:%M:%S"))
    return out


def _safe_start_key(name):
    try:
        return get_bbb_start_key(str(name))
    except Exception:
        return pd.NA


def _disk_has_primary(bucket_dir, prefix, check_read_bbb=False):
    """Disk fallback: any valid .bbb in the bucket sharing cam id + start second."""
    return any(
        is_bbb_file_valid_basicmatch(c, check_read_file=check_read_bbb)
        for c in sorted(glob.glob(os.path.join(glob.escape(bucket_dir), prefix + "*.bbb")))
    )


def hour_windows(dates, interval_hours=1):
    """
    Hour-aligned [start, end) windows covering each YYYYMMDD date (UTC), oldest first.

    Both generate_jobs_save_detect and generate_jobs_tracking previously built
    these independently; for interval_hours=1 they produced identical windows.
    """
    from datetime import datetime, timedelta, timezone

    interval = timedelta(hours=int(interval_hours))
    if interval.total_seconds() <= 0:
        raise ValueError("interval_hours must be positive")

    days = list(dates) if isinstance(dates, (list, tuple)) else [dates]
    windows = []
    for d in days:
        day_start = datetime.strptime(str(d), "%Y%m%d").replace(tzinfo=timezone.utc)
        day_end = day_start + timedelta(days=1)
        cur = day_start
        while cur < day_end:
            win_end = min(day_end, cur + interval)
            if win_end > cur:
                windows.append((cur, win_end))
            cur += interval
    windows.sort(key=lambda w: w[0])
    return windows


def build_out_index(df_out):
    """(cam_id, from_dt, to_dt) -> modified_time, from a *_outinfo.parquet frame."""
    if df_out is None or len(df_out) == 0:
        return {}
    return dict(
        zip(
            zip(df_out["cam_id"], df_out["from_dt"], df_out["to_dt"]),
            df_out["modified_time"],
        )
    )


def window_candidates(df_bbb, out_index, dates, interval_hours=1):
    """
    Classify every (cam_id, hour-window) work unit implied by the BBB catalog.

    A window is *expected* for a camera when at least one .bbb OVERLAPS it
    (starttime < win_end AND endtime > win_start). Note this is an overlap test on
    both ends, not a floor() of starttime: a .bbb spanning an hour boundary belongs
    to BOTH hours, and the submitters schedule it in both.

    Returns one dict per expected unit:
        cam_id, from_dt, to_dt, latest_src_mtime, out_mtime, status
    where status is:
        'missing' -- no output exists
        'stale'   -- output is older than the newest .bbb in the window
        'done'    -- output exists and is at least as new as every source .bbb

    Units with status != 'done' are exactly what the generators schedule.
    """
    if df_bbb is None or len(df_bbb) == 0:
        return []

    df_bbb = ensure_cam_id(df_bbb)
    df_bbb = df_bbb.dropna(subset=["cam_id", "starttime", "endtime"])

    out = []
    for win_start, win_end in hour_windows(dates, interval_hours):
        sel = df_bbb[(df_bbb["starttime"] < win_end) & (df_bbb["endtime"] > win_start)]
        if sel.empty:
            continue

        for cam_id, subset in sel.groupby("cam_id"):
            latest_src = subset["modified_time"].max()
            existing = out_index.get((int(cam_id), win_start, win_end))

            if existing is None:
                status = "missing"
            elif latest_src > existing:
                status = "stale"
            else:
                status = "done"

            out.append({
                "cam_id": int(cam_id),
                "from_dt": win_start,
                "to_dt": win_end,
                "latest_src_mtime": latest_src,
                "out_mtime": existing,
                "status": status,
            })

    out.sort(key=lambda u: (u["from_dt"], u["cam_id"]))
    return out


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

    units = window_candidates(df_bbb, build_out_index(df_out), datestring, interval_hours)
    candidates = [
        {
            'repo_path': PIPELINE_ROOT,
            'save_path': save_dir,
            'from_dt':   u['from_dt'],
            'to_dt':     u['to_dt'],
            'cam_id':    u['cam_id'],
        }
        for u in units if u['status'] != 'done'
    ]

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

    windows = hour_windows(datestring, interval_hours)
    units = window_candidates(df_bbb, build_out_index(df_out), datestring, interval_hours)
    candidates = [
        {
            "repo_path": PIPELINE_ROOT,
            "save_path": save_dir,
            "temp_path": TEMP_DIR,
            "from_dt":   u["from_dt"],
            "to_dt":     u["to_dt"],
            "cam_id":    u["cam_id"],
        }
        for u in units if u["status"] != "done"
    ]

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
##### CELL-SEG SHARED HELPERS (frame_extract + background)
#################################################################
# Single source of truth for "is this (date, cam) unit done?". The generators
# below and bb_hpc.src.progress both call these, so a progress report can never
# disagree with what the submitters actually schedule.
#
# All functions here are WRITE-FREE.

def _import_frame_naming():
    """frame_extractor.naming.expected_frame_filenames, or None on a bare submit host."""
    try:
        from frame_extractor.naming import expected_frame_filenames
        return expected_frame_filenames
    except Exception:
        return None


def _import_windowing():
    """background_generator.windowing, or None on a bare submit host."""
    try:
        from background_generator import windowing
        return windowing
    except Exception:
        return None


#################################################################
##### FRAME EXTRACT (cell-seg heavy preprocessing)
#################################################################
def iter_frame_extract_units(video_root_dir, frames_root_dir, datestring, verbose=False):
    """
    Yield every (date, cam) frame-extraction work unit that has source videos.

    Yields dicts: {date, cam, cam_dir, out_dir, txts}. A unit with an empty
    ``txts`` list has no usable source (the engine needs the per-video .txt
    timestamp sidecars) and is *skipped* by the generator -- neither done nor
    pending. Callers that report coverage must account for that third state.
    """
    import os, glob
    from pathlib import Path

    days = list(datestring) if isinstance(datestring, (list, tuple)) else [datestring]
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
            yield {
                "date": date,
                "cam": cam,
                "cam_dir": cam_dir,
                "out_dir": os.path.join(frames_root_dir, date, cam),
                "txts": txts,
            }


def frame_extract_is_done(out_dir, txts, interval_in_sec, fps, file_format, verbose=False):
    """
    True when every expected output frame already exists.

    Uses the engine's own ``expected_frame_filenames`` when importable (a coarser
    interval that is a multiple of a finer completed run yields a subset of the
    finer run's names, so re-running a finished date schedules nothing). Falls
    back to a presence check when the cell-seg package is absent.
    """
    import os

    expected_frame_filenames = _import_frame_naming()
    if expected_frame_filenames is not None:
        try:
            expected = expected_frame_filenames(txts, interval_in_sec, fps, file_format)
            if not expected:
                return False
            existing = set(os.listdir(out_dir)) if os.path.isdir(out_dir) else set()
            return expected.issubset(existing)
        except Exception as e:
            if verbose:
                print(f"[frame_extract] skip-check error {out_dir}: {e!r}")
            return False

    return os.path.isdir(out_dir) and any(
        fn.endswith(f".{file_format}") for fn in os.listdir(out_dir)
    )


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
    uses) via iter_frame_extract_units, and skips a (date, cam) only when EVERY
    expected output frame already exists (frame_extract_is_done).

    Yields
    ------
    dict: {"work_units": [ {date, cam, video_root, frames_root, interval_in_sec,
                            fps, file_format, decoder, max_workers}, ... ]}
    """
    if _import_frame_naming() is None:
        print("[frame_extract] frame_extractor not importable; using presence-based skip")

    days = list(datestring) if isinstance(datestring, (list, tuple)) else [datestring]

    units = []
    for u in iter_frame_extract_units(video_root_dir, frames_root_dir, days, verbose=verbose):
        if not u["txts"]:
            continue
        if frame_extract_is_done(u["out_dir"], u["txts"], interval_in_sec, fps, file_format, verbose):
            if verbose:
                print(f"[frame_extract] done: {u['date']}/{u['cam']}")
            continue

        units.append({
            "date": u["date"],
            "cam": u["cam"],
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
def _expected_background_names_fallback(frame_names, frame_interval_sec, background_window):
    """Dependency-free mirror of ``background_generator.windowing.expected_background_names``.

    Lets the SUBMIT HOST compute the exact per-window background output filenames
    for the skip check even when the heavy cell-seg package is not installed
    there (e.g. Konstanz Docker/k8s submit, where the work runs in the container).
    Must stay in sync with the fork's windowing.py, which is the canonical version
    and is used instead whenever it is importable.
    """
    import re as _re
    from datetime import datetime as _dt, timedelta as _td

    def _ts(name):
        m = _re.search(r"(\d{8}T\d{6})", name)
        if not m:
            return None
        try:
            return _dt.strptime(m.group(1), "%Y%m%dT%H%M%S")
        except ValueError:
            return None

    def _bucket(ts):
        if background_window == "hour":
            return ts.replace(minute=0, second=0, microsecond=0)
        if background_window == "day":
            return ts.replace(hour=0, minute=0, second=0, microsecond=0)
        sec = int(background_window)
        day0 = ts.replace(hour=0, minute=0, second=0, microsecond=0)
        off = int((ts - day0).total_seconds())
        return day0 + _td(seconds=(off // sec) * sec)

    names = sorted(frame_names)
    if frame_interval_sec:
        kept, last = [], None
        for n in names:
            ts = _ts(n)
            if ts is None:
                continue
            if last is None or (ts - last).total_seconds() >= int(frame_interval_sec):
                kept.append(n)
                last = ts
        names = kept

    buckets = set()
    for n in names:
        ts = _ts(n)
        if ts is not None:
            buckets.add(_bucket(ts))
    return {"background_" + b.strftime("%Y%m%dT%H%M%S") + ".000000.000Z.png" for b in buckets}


def _config_tag_fallback(frame_interval_sec, background_window, window_size, num_median_images):
    """Dependency-free mirror of ``background_generator.windowing.config_tag``.

    The config tag names the output subdirectory
    ``<backgrounds_root>/<date>/<cam>/<tag>/``, so getting it wrong makes the
    skip check stat a directory that never exists and every (date, cam) look
    pending forever. Must stay in sync with the fork's windowing.py, which is
    canonical and used whenever importable. test/test_progress.py asserts the two
    agree when background_generator is installed.
    """
    interval = frame_interval_sec or 0
    if background_window:
        return f"int{interval}s_win{background_window}"
    return f"count_w{window_size}_n{num_median_images}_int{interval}s"


def background_config_tag(frame_interval_sec, background_window, window_size, num_median_images):
    """Canonical config tag when background_generator is importable, else the mirror."""
    w = _import_windowing()
    if w is not None:
        return w.config_tag(frame_interval_sec, background_window, window_size, num_median_images)
    return _config_tag_fallback(frame_interval_sec, background_window, window_size, num_median_images)


def list_background_tag_dirs(out_cam_dir):
    """Config-tag subdirectories that actually exist under <output>/<cam>/."""
    import os
    if not os.path.isdir(out_cam_dir):
        return []
    return sorted(d for d in os.listdir(out_cam_dir) if os.path.isdir(os.path.join(out_cam_dir, d)))


def resolve_background_tag(out_cam_dir, frame_interval_sec, background_window,
                           window_size, num_median_images):
    """
    Return (tag, note) -- the config-tag dir to check for existing backgrounds.

    When background_generator is importable its config_tag is authoritative and we
    use it. When it is NOT (a bare submit host), our mirror is a guess: if the
    guessed dir is absent but exactly one tag dir exists on disk, that one was
    created by the engine for this product, so we use it and return a note. With
    two or more we cannot disambiguate, so we keep the guess and warn. With none,
    nothing has been produced yet and the guess is harmless.

    Without this, a wrong guess makes background_submit reschedule finished work
    on every run.
    """
    import os

    w = _import_windowing()
    if w is not None:
        return w.config_tag(frame_interval_sec, background_window, window_size, num_median_images), None

    tag = _config_tag_fallback(frame_interval_sec, background_window, window_size, num_median_images)
    if os.path.isdir(os.path.join(out_cam_dir, tag)):
        return tag, None

    existing = list_background_tag_dirs(out_cam_dir)
    if len(existing) == 1:
        return existing[0], (
            f"background_generator not importable: guessed config tag {tag!r} is absent under "
            f"{out_cam_dir}; using the only tag dir present ({existing[0]!r})."
        )
    if len(existing) > 1:
        return tag, (
            f"background_generator not importable: guessed config tag {tag!r} is absent under "
            f"{out_cam_dir} and {len(existing)} tag dirs exist ({', '.join(existing)}); cannot "
            f"disambiguate. Install background_generator on this host to get the canonical tag."
        )
    return tag, None


def background_expected_names(frame_names, frame_interval_sec, background_window):
    """Exact per-window background_*.png names the engine will produce."""
    w = _import_windowing()
    if w is not None:
        return w.expected_background_names(frame_names, frame_interval_sec, background_window)
    return _expected_background_names_fallback(frame_names, frame_interval_sec, background_window)


def background_kept_count(frame_names, frame_interval_sec):
    """Frames surviving interval subsampling -- mirrors what the engine will use."""
    w = _import_windowing()
    if frame_interval_sec and w is not None:
        return len(w.select_by_interval(sorted(frame_names), frame_interval_sec))
    return len(frame_names)


def background_day_of(name):
    """YYYYMMDD of a frame filename, or None."""
    import re
    w = _import_windowing()
    if w is not None:
        ts = w.parse_ts_from_name(name)
        return ts.strftime("%Y%m%d") if ts is not None else None
    m = re.search(r"(\d{8})T\d{6}", name)
    return m.group(1) if m else None


def background_is_done(out_cam_tag_dir, frame_names, frame_interval_sec, background_window):
    """
    Window mode: every expected per-window background_<ts>.png exists.
    Count/rolling mode (background_window=None): any background_*.png is 'done enough'.
    """
    import os

    if background_window:
        expected = background_expected_names(frame_names, frame_interval_sec, background_window)
        if not expected:
            return False
        existing = set(os.listdir(out_cam_tag_dir)) if os.path.isdir(out_cam_tag_dir) else set()
        return expected.issubset(existing)

    return os.path.isdir(out_cam_tag_dir) and any(
        fn.startswith("background_") and fn.endswith(".png") for fn in os.listdir(out_cam_tag_dir)
    )


def iter_background_scopes(frames_root_dir, backgrounds_root_dir, datestring,
                           source_dir=None, label=None, out_dir=None):
    """
    (scan_dir, output_path, scope_id, explicit) per scope.

    Date mode: one scope per date. Explicit-dir mode (source_dir set): a single
    scope with no date level, output under <out_dir or backgrounds_root>/<label>.
    """
    import os

    if source_dir is not None:
        out_base = out_dir or backgrounds_root_dir
        lbl = label or os.path.basename(str(source_dir).rstrip("/")) or "frames"
        return [(str(source_dir), os.path.join(out_base, lbl), lbl, True)]

    days = list(datestring) if isinstance(datestring, (list, tuple)) else [datestring]
    return [(os.path.join(frames_root_dir, d), os.path.join(backgrounds_root_dir, d), d, False)
            for d in days]


def iter_background_cams(scan_dir, cams=None):
    """Yield (cam, cam_dir, frame_names) for each cam-N dir under scan_dir."""
    import os, glob

    if not os.path.isdir(scan_dir):
        return
    cam_filter = set(cams) if cams else None
    for cam_dir in sorted(d for d in glob.glob(os.path.join(scan_dir, "cam-*")) if os.path.isdir(d)):
        cam = os.path.basename(cam_dir)
        if cam_filter is not None and cam not in cam_filter:
            continue
        yield cam, cam_dir, [fn for fn in os.listdir(cam_dir) if fn.endswith(".png")]


def generate_jobs_background(
    frames_root_dir,        # e.g. settings.frames_dir_hpc
    backgrounds_root_dir,   # e.g. settings.backgrounds_dir_hpc
    datestring,             # list of YYYYMMDD (or a single string); ignored in source_dir mode
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
    source_dir=None,        # explicit frames dir containing cam-N (NO date level)
    label=None,             # output sub-key under the output base for source_dir mode
    out_dir=None,           # explicit output base (defaults to backgrounds_root_dir)
    cams=None,              # optional camera filter, e.g. ["cam-0", "cam-1"]
    dates=None,             # optional YYYYMMDD day filter; source_dir mode -> one (cam, day) unit each
    min_frames=3,           # skip a (cam, day) with fewer frames than the engine needs (windowed mode)
):
    """
    Yield chunks of (scope, camera) background-generation work units.

    Two enumeration modes:
      - Date mode (default): cameras under ``frames_root/<date>/cam-N/`` for each
        date; output under ``backgrounds_root/<date>``.
      - Explicit-dir mode (``source_dir`` set): cameras under ``source_dir/cam-N/``
        directly (no date level -- e.g. bb_monitor's ``single_video_frames/``);
        output under ``<out_dir or backgrounds_root>/<label>``.

    The output config tag (interval/window settings) is encoded in the output
    path, so a (scope, cam) is skipped when the backgrounds for THIS config
    already exist -- see background_is_done / resolve_background_tag.

    Yields
    ------
    dict: {"work_units": [ {date, cam, frames_root, backgrounds_root, <bg knobs>,
                            (source_path, output_path in explicit-dir mode)}, ... ]}

    Day-sharding (``dates`` + ``source_dir``): emits one work unit per (cam, day)
    so each task masks/backgrounds only that day. Days with fewer than
    ``min_frames`` frames are skipped (the engine could not produce a background
    for them, so they must not be scheduled forever).
    """
    import os

    if _import_windowing() is None:
        print("[background] background_generator not importable; using mirrored skip logic")

    # Day filter is only meaningful in explicit-dir mode (date mode already shards
    # per date via scopes). Dedup so a repeated date does not emit duplicate units.
    dates_filter = list(dict.fromkeys(dates)) if (dates and source_dir is not None) else None

    def _knobs():
        return {
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
            "min_frames": min_frames,
        }

    def _make_unit(scope_id, cam, scan_dir, output_path, explicit, day=None):
        unit = {
            "date": scope_id,   # YYYYMMDD in date mode, label in explicit-dir mode
            "cam": cam,
            "dates": [day] if day else None,
            "frames_root": frames_root_dir,
            "backgrounds_root": backgrounds_root_dir,
        }
        if explicit:
            # Already-resolved cluster paths; job_for_background_chunk uses these
            # verbatim instead of reconstructing frames_root/<date>.
            unit["source_path"] = scan_dir
            unit["output_path"] = output_path
        if day:
            unit["day"] = day
        unit.update(_knobs())
        return unit

    scopes = iter_background_scopes(frames_root_dir, backgrounds_root_dir, datestring,
                                    source_dir=source_dir, label=label, out_dir=out_dir)
    tag = background_config_tag(frame_interval_sec, background_window, window_size, num_median_images)

    units = []
    seen_notes = set()
    for scan_dir, output_path, scope_id, explicit in scopes:
        if not os.path.isdir(scan_dir):
            if verbose:
                print(f"[background] no frames dir: {scan_dir}")
            continue

        for cam, cam_dir, frame_names in iter_background_cams(scan_dir, cams):
            if not frame_names:
                continue

            out_cam_dir = os.path.join(output_path, cam)
            cam_tag, note = resolve_background_tag(
                out_cam_dir, frame_interval_sec, background_window, window_size, num_median_images)
            if note and note not in seen_notes:
                seen_notes.add(note)
                print(f"[background] WARNING: {note}")
            out_cam_tag_dir = os.path.join(out_cam_dir, cam_tag)

            if dates_filter is not None:
                # One unit per requested day (each task masks/backgrounds that day only).
                by_day = {}
                for fn in frame_names:
                    d = background_day_of(fn)
                    if d is not None:
                        by_day.setdefault(d, []).append(fn)
                for d in dates_filter:
                    day_frames = by_day.get(d, [])
                    if background_kept_count(day_frames, frame_interval_sec) < int(min_frames):
                        if verbose:
                            print(f"[background] skip {scope_id}/{cam} {d}: "
                                  f"{len(day_frames)} frame(s) < min_frames={min_frames}")
                        continue
                    if background_is_done(out_cam_tag_dir, day_frames, frame_interval_sec, background_window):
                        if verbose:
                            print(f"[background] done: {scope_id}/{cam} {d} [{cam_tag}]")
                        continue
                    units.append(_make_unit(scope_id, cam, scan_dir, output_path, explicit, day=d))
            else:
                if background_is_done(out_cam_tag_dir, frame_names, frame_interval_sec, background_window):
                    if verbose:
                        print(f"[background] done: {scope_id}/{cam} [{cam_tag}]")
                    continue
                units.append(_make_unit(scope_id, cam, scan_dir, output_path, explicit))

    units.sort(key=lambda u: (u["date"], u["cam"], u.get("day") or ""))
    if maxjobs is not None:
        units = units[: int(chunk_size) * int(maxjobs)]

    print(f"[background] scopes={len(scopes)} units={len(units)} chunk_size={chunk_size} "
          f"maxjobs={maxjobs} tag={tag}"
          + (f" source_dir={source_dir}" if source_dir else "")
          + (f" dates={dates_filter}" if dates_filter is not None else ""))

    for i in range(0, len(units), int(chunk_size)):
        yield {"work_units": units[i:i + int(chunk_size)]}
