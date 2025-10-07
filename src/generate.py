import os, glob
from datetime import timedelta, datetime
import pandas as pd
import fnmatch

from bb_binary.parsing import parse_video_fname
from bb_binary import Repository
from bb_hpc.src.fileinfo import get_bbb_file_path, get_pending_videos

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
    video_name_format="basler",
    use_fileinfo=False,          # use RESULTDIR/bbb_fileinfo/bbb_info_*.parquet
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
    video_name_format : str
        Passed to parse_video_fname() (e.g., "basler").
    use_fileinfo : bool
        If True, use RESULTDIR/bbb_fileinfo/bbb_info_*.parquet to skip videos
        whose .bbb is already present. Otherwise, fall back to os.path.exists().

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
                    # The canonical schema produced by list_bbb_files/build_bbb_info_parquet is:
                    #   file_name, full_path, starttime, endtime, modified_time
                    # We want repo-relative paths; derive them from full_path if they live under repo_output_path.
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
    # 3) glob raw mp4s (fast)
    # -------------------------------
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
            if os.path.exists(full_bbb):
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
        cam_id, dt_start, _ = parse_video_fname(os.path.basename(v), format=video_name_format)
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
):
    """
    Generate daily save-detect jobs per camera for the given YYYYMMDD dates.
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
            latest_src = subset['modified_time'].max()
            key = (cam_id, day_start, day_end)
            existing = out_index.get(key)
            if (existing is None) or (latest_src > existing):
                candidates.append({
                    'repo_path': PIPELINE_ROOT,
                    'save_path': save_dir,
                    'from_dt':   day_start,
                    'to_dt':     day_end,
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
          f"chunk_size={chunk_size} maxjobs={maxjobs}")
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
def generate_jobs_rpi_detect(
    video_root_dir,
    dates,
    chunk_size=150,
    maxjobs=None,
    video_glob_pattern="*.h264",
    clahe=True,
):
    """
    Yields shards of RPi video paths under date directories using fast globbing.
    Excludes videos already processed (suffix depends on CLAHE).
    - Newest videos first (sorted by timestamp parsed from filename like ..._YYYY-MM-DD-HH-MM-SS.h264).
    - If parsing fails, falls back to file mtime.
    - If maxjobs is set, only keeps the newest chunk_size * maxjobs videos.
    """

    # 1) glob all raw h264’s for requested dates
    print("getting new h264 files")
    all_videos = []
    for day in dates:  # e.g., "20250710"
        pattern = os.path.join(video_root_dir, day, "**", video_glob_pattern)
        print(pattern)
        all_videos.extend(glob.glob(pattern, recursive=True))
    print("found", len(all_videos), "files")

    # 2) filter out videos already processed (suffix depends on CLAHE)
    print("making list to submit")
    suffix = "-c" if clahe else "-nc"
    to_schedule = []
    for v in all_videos:
        det_file = os.path.splitext(v)[0] + f"-detections{suffix}.parquet"
        if not os.path.exists(det_file):
            to_schedule.append(v)

    print("number of videos to schedule:", len(to_schedule))

    # 3) newest-first ordering
    #    try to parse ..._<YYYY-MM-DD-HH-MM-SS>.h264 from basename; fallback: mtime
    video_tuples = []
    for v in to_schedule:
        base = os.path.basename(v)
        ts = None
        # Expected pattern: <cam>_<YYYY-MM-DD-HH-MM-SS>.h264
        # Get the last "_" segment without extension
        try:
            ts_str = os.path.splitext(base)[0].rsplit("_", 1)[-1]
            ts = datetime.strptime(ts_str, "%Y-%m-%d-%H-%M-%S")
        except Exception:
            # fallback to file modification time if parse fails
            try:
                ts = datetime.fromtimestamp(os.path.getmtime(v))
            except Exception:
                ts = datetime.min
        video_tuples.append((ts, v))

    # oldest first
    video_tuples.sort(key=lambda t: t[0], reverse=False)

    # 4) keep only videos for max number of jobs (if set)
    if maxjobs is not None:
        keep = chunk_size * int(maxjobs)
        video_tuples = video_tuples[:keep]

    print("number of videos kept with max jobs", len(video_tuples))

    # 5) yield in batches of size chunk_size
    for i in range(0, len(video_tuples), chunk_size):
        chunk = video_tuples[i : i + chunk_size]
        yield {
            "video_paths": [path for _, path in chunk],
        }