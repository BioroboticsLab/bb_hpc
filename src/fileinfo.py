import os, glob
import pandas as pd
from bb_binary.parsing import parse_video_fname
from pathlib import Path

import sys, shlex, subprocess

import bb_binary
from datetime import datetime, timezone

bbb = bb_binary.common.bbb

#################################################################
##### listing functions for making database of existing and already processed videos
#################################################################

def _latest_div_mtime(day_dir: Path) -> datetime | None:
    """
    For a given day directory (YYYY/MM/DD), return the newest mtime among all
    *minute-div* directories located at: YYYY/MM/DD/HH/<minute-div>.
    We use GNU find to avoid Python per-entry stat calls on large trees.

    Returns a timezone-aware UTC datetime, or None if none found.
    """
    import subprocess, shlex
    try:
        # From the day_dir, minute-div dirs are depth 2: HH (1) / minute-div (2)
        cmd = (
            f"find {shlex.quote(str(day_dir))} "
            f"-mindepth 2 -maxdepth 2 -type d -printf '%T@\\n'"
        )
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        newest = None
        for line in out.splitlines():
            try:
                t = float(line.decode("utf-8", "replace").strip())
                dt = datetime.fromtimestamp(t, tz=timezone.utc)
                if newest is None or dt > newest:
                    newest = dt
            except Exception:
                continue
        return newest
    except Exception:
        # Fallback: try the day_dir mtime; if that fails, return None
        try:
            return datetime.fromtimestamp(day_dir.stat().st_mtime, tz=timezone.utc)
        except Exception:
            return None

def _max_file_mtime_python(day_dir: Path) -> datetime | None:
    """
    Portable fallback for _latest_file_mtime: newest mtime among all files under
    day_dir, via scandir. Slower than GNU find, but CORRECT everywhere -- notably
    on BSD/macOS, where `find -printf` does not exist.

    Do NOT fall back to day_dir.stat() instead: a new video lands in
    <day>/cam-N/, which bumps the *camera* directory's mtime, not the day's, so a
    day-mtime check would silently trust a stale cache forever.
    """
    newest = None
    stack = [str(day_dir)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for entry in it:
                    try:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                            continue
                        mt = entry.stat(follow_symlinks=False).st_mtime
                    except OSError:
                        continue
                    if newest is None or mt > newest:
                        newest = mt
        except (FileNotFoundError, PermissionError, NotADirectoryError, OSError):
            continue
    return datetime.fromtimestamp(newest, tz=timezone.utc) if newest is not None else None


def _latest_file_mtime(day_dir: Path) -> datetime | None:
    """
    Return the newest file mtime anywhere under day_dir (recursive).
    Used for RPi and raw-video daily caches, where files live in per-camera subdirs.
    """
    import subprocess, shlex
    try:
        cmd = f"find {shlex.quote(str(day_dir))} -type f -printf '%T@\\n'"
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        newest = None
        for line in out.splitlines():
            try:
                t = float(line.decode("utf-8", "replace").strip())
                dt = datetime.fromtimestamp(t, tz=timezone.utc)
                if newest is None or dt > newest:
                    newest = dt
            except Exception:
                continue
        if newest is not None:
            return newest
        # GNU find succeeded but printed nothing usable (e.g. empty dir): fall through.
    except Exception:
        pass
    return _max_file_mtime_python(day_dir)

def _safe_subdirs(p: Path):
    try:
        for q in p.iterdir():
            try:
                if q.is_dir():
                    yield q
            except OSError:
                continue
    except (FileNotFoundError, PermissionError, OSError):
        return

def _is_day_dirname(name: str) -> bool:
    """True for a top-level day directory: YYYYMMDD or YYYY-MM-DD."""
    if len(name) == 8 and name.isdigit():
        return True
    if len(name) == 10 and name[4] == "-" and name[7] == "-":
        y, m, d = name.split("-")
        return y.isdigit() and m.isdigit() and d.isdigit()
    return False

def _day_key(name: str) -> str:
    """Normalize a day dir name to YYYYMMDD (sortable, cache-file safe)."""
    return name.replace("-", "")

def _list_rpi_day(day_dir: Path, day_str: str, video_glob_pattern: str) -> pd.DataFrame:
    """
    Build a per-day catalog of RPi videos and whether CLAHE / no-CLAHE detections exist.
    """
    rows = []
    pattern = os.path.join(str(day_dir), "**", video_glob_pattern)
    for path in glob.glob(pattern, recursive=True):
        base = os.path.basename(path)
        cam = base.rsplit("_", 1)[0] if "_" in base else ""
        root, _ = os.path.splitext(path)
        rows.append({
            "date": day_str,
            "cam": cam,
            "video_name": base,
            "full_path": path,
            "detections_clahe": os.path.exists(root + "-detections-c.parquet") or os.path.exists(root + "-detections-polo-c.parquet"),
            "detections_noclahe": os.path.exists(root + "-detections-nc.parquet") or os.path.exists(root + "-detections-polo-nc.parquet"),
        })
    cols = ["date", "cam", "video_name", "full_path", "detections_clahe", "detections_noclahe"]
    return pd.DataFrame(rows, columns=cols)

def list_bbb_files_incremental(
    pipeline_root: str,
    cache_dir: str,
    check_read_bbb: bool = False,
    deep_check_bbb: bool = False,
) -> pd.DataFrame:
    """
    Cache daily catalogs under cache_dir/daily/bbb_YYYYMMDD.parquet.
    Only rescan days that are missing or whose *minute-div* directory mtime
    (YYYY/MM/DD/HH/<minute-div>) is newer than the cached parquet.
    If deep_check_bbb is True, cached daily files must include valid_check="deep"
    or they will be regenerated.
    """
    if deep_check_bbb:
        check_read_bbb = True
    daily_dir = Path(cache_dir) / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    # Discover day folders (YYYY/MM/DD) without deep walking.
    day_paths: list[Path] = []
    pr = Path(pipeline_root)
    for y in sorted(_safe_subdirs(pr)):
        for m in sorted(_safe_subdirs(y)):
            for d in sorted(_safe_subdirs(m)):
                day_paths.append(d)

    dfs = []
    for d in day_paths:
        day_str = "/".join(d.parts[-3:])  # YYYY/MM/DD
        print(day_str)
        out_pq = daily_dir / f"bbb_{day_str.replace('/','')}.parquet"

        # Compute deepest relevant mtime: newest of any minute-div dir under the day.
        deep_mtime = _latest_div_mtime(d)

        # If cache exists and is at least as new as deepest mtime, trust it.
        try:
            if out_pq.exists():
                pq_mtime = datetime.fromtimestamp(out_pq.stat().st_mtime, tz=timezone.utc)
                if deep_mtime is None or pq_mtime >= deep_mtime:
                    try:
                        df_cached = pd.read_parquet(out_pq)
                        if check_read_bbb:
                            if "is_valid" not in df_cached.columns:
                                raise ValueError("cached file missing is_valid")
                            if deep_check_bbb:
                                if "valid_check" not in df_cached.columns:
                                    raise ValueError("cached file missing valid_check")
                                if not (df_cached["valid_check"] == "deep").all():
                                    raise ValueError("cached file has non-deep validity checks")
                        dfs.append(df_cached)
                        continue
                    except Exception:
                        pass
        except Exception:
            pass

        # Rescan this day only
        print("...reindexing")
        df_day = list_bbb_files(str(d), check_read_bbb=check_read_bbb, deep_check_bbb=deep_check_bbb)
        try:
            df_day.to_parquet(out_pq, index=False)
        except Exception:
            pass
        dfs.append(df_day)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    cols = ["file_name", "full_path", "starttime", "endtime", "modified_time", "file_size"]
    if check_read_bbb:
        cols.extend(["is_valid", "valid_check"])
    return pd.DataFrame(columns=cols)

def list_rpi_files_incremental(video_root: str, cache_dir: str, video_glob_pattern: str = "*.h264") -> pd.DataFrame:
    """
    Cache daily catalogs of RPi videos under cache_dir/daily/rpi_YYYYMMDD.parquet.
    Only rescan a day if the newest file mtime under that day is newer than the cache.
    """
    daily_dir = Path(cache_dir) / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    vr = Path(video_root)
    day_paths: list[Path] = []
    for d in sorted(_safe_subdirs(vr)):
        name = d.name
        if len(name) == 8 and name.replace("-", "").isdigit():
            day_paths.append(d)

    dfs = []
    for d in day_paths:
        day_str = d.name
        out_pq = daily_dir / f"rpi_{day_str.replace('-', '')}.parquet"

        deep_mtime = _latest_file_mtime(d)
        try:
            if out_pq.exists():
                pq_mtime = datetime.fromtimestamp(out_pq.stat().st_mtime, tz=timezone.utc)
                if deep_mtime is None or pq_mtime >= deep_mtime:
                    try:
                        dfs.append(pd.read_parquet(out_pq))
                        continue
                    except Exception:
                        pass
        except Exception:
            pass

        print(day_str)
        print("...reindexing RPi day")
        df_day = _list_rpi_day(d, day_str, video_glob_pattern)
        try:
            df_day.to_parquet(out_pq, index=False)
        except Exception:
            pass
        dfs.append(df_day)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["date", "cam", "video_name", "full_path", "detections_clahe", "detections_noclahe"])

VIDEO_INFO_COLUMNS = ["file_name", "full_path", "starttime", "endtime", "cam"]

def list_video_day(day_dir, exts=(".mp4",)) -> pd.DataFrame:
    """
    Catalog every raw video under one day directory (recursive).

    Columns: file_name, full_path, starttime, endtime, cam.
    Unparsable filenames keep the row with None time/cam fields, matching the
    historical behavior of get_videoinfo.build_video_info_df.
    """
    exts_lower = tuple(e.lower() for e in exts)
    rows = []
    for root, _, files in os.walk(str(day_dir)):
        for fn in files:
            if not fn.lower().endswith(exts_lower):
                continue
            cam = start = end = None
            try:
                cam, start, end = parse_video_fname(fn)
            except Exception:
                pass
            rows.append({
                "file_name": fn,
                "full_path": os.path.join(root, fn),
                "starttime": start,
                "endtime": end,
                "cam": cam,
            })
    return pd.DataFrame(rows, columns=VIDEO_INFO_COLUMNS)


def list_video_files_incremental(
    video_root: str,
    cache_dir: str,
    exts=(".mp4",),
    force_recent_days: int = 2,
) -> pd.DataFrame:
    """
    Cache daily catalogs of raw videos under cache_dir/daily/video_YYYYMMDD.parquet.
    Only rescan a day if the newest file mtime under that day is newer than the cache.

    Mirrors list_rpi_files_incremental, with two additions:

    - The written daily parquet's mtime is back-dated to the *pre-scan* sample of
      the day's newest file mtime. Both timestamps then come from the same
      (shared-storage) clock, and any video that lands *during* our os.walk is
      guaranteed to be newer than the cache -- so it is picked up on the next run
      instead of being silently lost. Without this, a file written between the
      find() sample and the parquet write would be masked forever.
    - force_recent_days unconditionally rescans the N newest day dirs, which is
      cheap (one day of one season) and covers filesystems with coarse mtime
      granularity. Set 0 to rely purely on the mtime comparison.
    """
    daily_dir = Path(cache_dir) / "daily"
    daily_dir.mkdir(parents=True, exist_ok=True)

    day_paths = sorted(
        (d for d in _safe_subdirs(Path(video_root)) if _is_day_dirname(d.name)),
        key=lambda d: _day_key(d.name),
    )

    n_force = max(0, int(force_recent_days))
    forced = {d.name for d in day_paths[len(day_paths) - n_force:]} if n_force else set()

    dfs = []
    for d in day_paths:
        out_pq = daily_dir / f"video_{_day_key(d.name)}.parquet"

        # Sample the day's newest file mtime BEFORE scanning it.
        deep_mtime = _latest_file_mtime(d)

        if d.name not in forced:
            try:
                if out_pq.exists():
                    pq_mtime = datetime.fromtimestamp(out_pq.stat().st_mtime, tz=timezone.utc)
                    if deep_mtime is None or pq_mtime >= deep_mtime:
                        try:
                            dfs.append(pd.read_parquet(out_pq))
                            continue
                        except Exception:
                            pass
            except Exception:
                pass

        print(f"{d.name}\n...reindexing video day")
        df_day = list_video_day(d, exts)
        try:
            df_day.to_parquet(out_pq, index=False)
            if deep_mtime is not None:
                ts = deep_mtime.timestamp()
                os.utime(out_pq, (ts, ts))
        except Exception:
            pass
        dfs.append(df_day)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=VIDEO_INFO_COLUMNS)

def list_bbb_files(
    pipeline_root: str,
    check_read_bbb: bool = False,
    deep_check_bbb: bool = False,
) -> pd.DataFrame:
    """
    Fast scanner for .bbb files using GNU find, dramatically reducing metadata round-trips.
    Falls back to a scandir-based walk if GNU find is unavailable.
    Returns a DataFrame with columns:
      ['file_name','full_path','starttime','endtime','modified_time','file_size']
      plus optional 'is_valid' and 'valid_check' when check_read_bbb=True.
    """
    if deep_check_bbb:
        check_read_bbb = True

    # Declare the schema up front: pd.DataFrame([]) has ZERO columns, so an empty
    # repo would produce a catalog that KeyErrors in every reader.
    cols = ["file_name", "full_path", "starttime", "endtime", "modified_time", "file_size"]
    if check_read_bbb:
        cols = cols + ["is_valid", "valid_check"]

    def _from_find(root: str) -> pd.DataFrame:
        # %P  = path relative to the starting point
        # %T@ = mtime as seconds since epoch (float)
        # %s  = file size (bytes)
        cmd = (
            f"find {shlex.quote(root)} -type f -name '*.bbb' "
            r"-printf '%P\t%T@\t%s\n'"
        )
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        rows = []
        append = rows.append
        for line in out.splitlines():
            try:
                rel, mtime_epoch, size_str = line.decode("utf-8", "replace").rstrip("\n").split("\t", 2)
                full = os.path.join(root, rel)
                fname = os.path.basename(rel)
                # parse start/end from BBB filename (mirrors video naming)
                _cam, start, end = parse_video_fname(fname)
                mtime = datetime.fromtimestamp(float(mtime_epoch), tz=timezone.utc)
                try:
                    file_size = int(size_str)
                except Exception:
                    file_size = None
                row = {
                    "file_name": fname,
                    "full_path": full,
                    "starttime": start,
                    "endtime": end,
                    "modified_time": mtime,
                    "file_size": file_size,
                }
                if check_read_bbb:
                    if deep_check_bbb:
                        row["is_valid"] = is_bbb_file_valid_deep(full)
                        row["valid_check"] = "deep"
                    else:
                        if file_size == 0:
                            row["is_valid"] = False
                        else:
                            row["is_valid"] = is_bbb_file_valid_basicmatch(full, check_read_file=True)
                        row["valid_check"] = "read"
                append(row)
            except Exception:
                # Skip any malformed lines/filenames rather than stalling the run
                continue
        return pd.DataFrame(rows, columns=cols)

    def _fallback_scandir(root: str) -> pd.DataFrame:
        # If GNU find isn't available, use scandir (faster than os.walk + os.stat)
        rows = []
        append = rows.append
        stack = [root]
        while stack:
            d = stack.pop()
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            stack.append(entry.path)
                        elif entry.is_file(follow_symlinks=False) and entry.name.endswith(".bbb"):
                            fname = entry.name
                            full = entry.path
                            try:
                                st = entry.stat(follow_symlinks=False)
                                mtime = datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                                _cam, start, end = parse_video_fname(fname)
                                file_size = int(getattr(st, "st_size", 0))
                                row = {
                                    "file_name": fname,
                                    "full_path": full,
                                    "starttime": start,
                                    "endtime": end,
                                    "modified_time": mtime,
                                    "file_size": file_size,
                                }
                                if check_read_bbb:
                                    if deep_check_bbb:
                                        row["is_valid"] = is_bbb_file_valid_deep(full)
                                        row["valid_check"] = "deep"
                                    else:
                                        if file_size == 0:
                                            row["is_valid"] = False
                                        else:
                                            row["is_valid"] = is_bbb_file_valid_basicmatch(full, check_read_file=True)
                                        row["valid_check"] = "read"
                                append(row)
                            except Exception:
                                continue
            except PermissionError:
                continue
        return pd.DataFrame(rows, columns=cols)

    # Try fast path (GNU find). If that fails, use Python fallback.
    try:
        return _from_find(pipeline_root)
    except Exception:
        return _fallback_scandir(pipeline_root)


OUTINFO_COLUMNS = ["cam_id", "from_dt", "to_dt", "modified_time"]

def build_outinfo(output_dir, CACHE_DIR, extension, outname):
    # Helper to read existing outputs
    rows = []
    pattern = os.path.join(output_dir, f"*.{extension}")
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        cam_id, start, end = parse_video_fname(fname.replace(f'.{extension}',''))
        mtime = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        rows.append({
            'cam_id': cam_id,
            'from_dt': start,
            'to_dt': end,
            'modified_time': mtime
        })
    # Always declare the schema: pd.DataFrame([]) has ZERO columns, and a
    # column-less parquet makes every reader KeyError. That is the normal state
    # early in a season, before the first output of this stage exists.
    df = pd.DataFrame(rows, columns=OUTINFO_COLUMNS)
    outpath = os.path.join(CACHE_DIR, f'{outname}.parquet')
    df.to_parquet(outpath, index=False)
    print(f"Wrote {len(df)} rows to {outpath}")
    return outpath

#################################################################
##### bbb file info functions, for detection jobs
#################################################################
def get_bbb_file_path(video_basename):
    """
    Given a video file path, get the expected .bbb detection file path.
    Args:
        video_file_path (str): The full path to the video file (e.g. mp4).
    Returns:
        str: The expected .bbb file path subdirectory structure
    """
    from bb_binary.parsing import parse_video_fname, get_video_fname
    import numpy as np
    # Parse the video filename to extract the cam_id and timestamp using bb_binary functions
    cam_id, start, end = parse_video_fname(video_basename)
    # Convert the parsed timestamp to the expected directory structure (assumes UTC time)
    year = start.year
    month = str(start.month).zfill(2)
    day = str(start.day).zfill(2)
    hour = str(start.hour).zfill(2)
    minute = str(int(np.floor(start.minute / 20.0) * 20)).zfill(2)
    # Create the expected directory path for the .bbb file
    bbb_dir = f"{year}/{month}/{day}/{hour}/{minute}/"
    # Generate the expected .bbb file name
    bbb_file_name = get_video_fname(cam_id,start,end)+'.bbb'
    # Combine the base directory with the expected .bbb filename
    return bbb_dir+bbb_file_name

#################################################################
##### matching a video to its .bbb by camera + start second
#################################################################
# WHY the end timestamp cannot be part of the match key:
# bb_binary names its output from the ACTUAL frame timestamps, not from the
# source video's filename. Repository.add() passes frameContainer.fromTimestamp /
# toTimestamp (bb_binary repository.py:77-85), which bb_pipeline fills from the
# first/last decoded frame (pipeline/io.py:195-200); those timestamps come from
# the companion .txt sidecar (pipeline/io.py:123), not the video container.
# For a gappy recording (dropped frames) the content end and the filename end
# legitimately differ -- observed in production by +58s and +100s -- so the full
# start--end stem is not a usable identity. The START matched to the microsecond
# in every observed case.
#
# Truncated to whole SECONDS, not microseconds, because dt_to_str
# (bb_binary parsing.py:21-32) omits the fractional part entirely when
# microsecond == 0; an exact-microsecond key would fail to match every .bbb that
# lands on a whole second. Second-truncation also absorbs float round-trip drift.
#
# NOTE: src/jobfunctions.py duplicates this logic inline (see _primary_locator
# there). It cannot import from here -- slurmhelper copies each job function's
# source verbatim, so job functions must not reference module-level definitions
# (see the banner in jobfunctions.py and src/repo_guard.py). Keep the two in sync.

def get_bbb_bucket_and_prefix(video_basename, minute_step=20):
    """
    Locate the .bbb bb_binary will write for a video, by camera + start second.

    Args:
        video_basename (str): video file name (e.g. cam-0_...mp4).
        minute_step (int): bb_binary Repository bucket width in minutes.
    Returns:
        tuple[str, str]: (repo-relative bucket dir, .bbb filename prefix).
                         Glob `<repo>/<bucket>/<prefix>*.bbb` to find it.
    """
    import re
    from bb_binary.parsing import parse_video_fname, get_fname
    cam_id, start, _end = parse_video_fname(video_basename)
    minute = int(start.minute // minute_step * minute_step)
    bucket = (f"{start.year}/{start.month:02d}/{start.day:02d}/"
              f"{start.hour:02d}/{minute:02d}")
    # get_fname gives exactly "Cam_{id}_{dt_to_str(start)}" -- strip the optional
    # ".<microseconds>" and the trailing "Z" to get a second-resolution prefix.
    # The "_" after the cam id anchors it, so Cam_1_ cannot match Cam_11_.
    return bucket, re.sub(r"(\.\d+)?Z$", "", get_fname(cam_id, start))


def bbb_start_key_from_parts(cam_id, start):
    """Canonical hashable identity for a video/.bbb pair: cam id + start second."""
    from datetime import timezone
    if getattr(start, "tzinfo", None) is not None:
        start = start.astimezone(timezone.utc)
    return f"{int(cam_id)}|{start.strftime('%Y-%m-%dT%H:%M:%S')}"


def get_bbb_start_key(basename):
    """
    Start key for a video basename OR a .bbb basename -- the two must agree.

    This is the join key that replaces `basename(get_bbb_file_path(v))` for
    "is this video already detected?" checks.
    """
    from bb_binary.parsing import parse_video_fname
    cam_id, start, _end = parse_video_fname(basename)
    return bbb_start_key_from_parts(cam_id, start)


def get_pending_videos(input_dir, log_step=500):
    """
    This was used to check if there are pending videos in the 'jobs' directory already queued up.  
    I stopped using this procedure because of complexity, but leave this function here for reference
    """    
    import os, dill
    pending = set()

    # correct: use `f` consistently inside the generator expression
    dill_files = sorted(f for f in os.listdir(input_dir) if f.endswith(".dill"))

    for i, fn in enumerate(dill_files):
        if i % log_step == 0:
            print(f"[{i:>6}] {fn}")
        with open(os.path.join(input_dir, fn), "rb") as fobj:
            pending.update(dill.load(fobj)["video_paths"])
    return pending    

    
#################################################################
##### not used anymore / reference
#################################################################

def is_bbb_file_valid_globfilematch(bbb_file):
    """
    Checks if a .bbb file is valid by attempting to open and read it.
    Uses glob to replace the fractional seconds with *, so that it would still match

    Args:
        bbb_file (str): Path to the .bbb file.

    Returns:
        bool: True if the file is valid and can be read, False otherwise.
    """
    import os
    import glob
    import re
    import bb_binary
    bbb = bb_binary.common.bbb

    # Get the directory and file components
    dir_path, bbb_filename = os.path.split(bbb_file)

    # Build the glob pattern by replacing the fractional seconds with '*'
    # Fractional seconds are the digits after the '.' and before 'Z'
    glob_pattern = re.sub(r'\.\d+Z', r'.*Z', bbb_filename)
    glob_path = os.path.join(dir_path, glob_pattern)

    # Search for matching files using glob
    matching_files = glob.glob(glob_path)

    for full_path in matching_files:
        # Check if the file exists and is valid (can be read)
        try:
            with open(full_path, 'rb') as f:
                message = bbb.FrameContainer.read(f)
                return True  # Return True if the file can be read successfully
        except Exception as e:
            print(f"Error reading file {full_path}: {e}")
            return False
    return False  # Return False if no valid matching file is found

def is_bbb_file_valid_basicmatch(bbb_file, check_read_file = False):
    """
    Checks if a .bbb file is valid by attempting to open and read it.

    Args:
        bbb_file (str): Path to the .bbb file.

    Returns:
        bool: True if the file is valid and can be read, False otherwise.
    """
    if not os.path.exists(bbb_file):
        return False
    else:
        if not check_read_file:
            # Fast check: exists AND non-zero. A valid .bbb always has capnp
            # framing, so a 0-byte file is an aborted/partial write (e.g. the
            # bb_binary stub left when a cross-boundary symlink hit EIO) and must
            # NOT count as "already done" — otherwise re-submits skip it forever.
            try:
                return os.path.getsize(bbb_file) != 0
            except OSError:
                return False
        else:
            try:
                if os.path.getsize(bbb_file) == 0:
                    return False
            except Exception:
                pass
            # Attempt to open and read the file
            try:
                with open(bbb_file, 'rb') as f:
                    bbb.FrameContainer.read(f)
                return True  # File is valid
            except Exception:
                return False  # File is invalid or cannot be read


def is_bbb_file_valid_deep(bbb_file):
    """
    Deep validation: attempt to read all FrameContainers until EOF.
    Returns False on premature EOF / decode errors.

    This runs the validation in a subprocess to handle Cap'n Proto C++ exceptions
    that would otherwise crash the main process (e.g., "Premature EOF" from kj library).
    """
    if not os.path.exists(bbb_file):
        return False
    try:
        size = os.path.getsize(bbb_file)
        if size == 0:
            return False
    except Exception:
        pass

    # Run validation in subprocess to isolate from Cap'n Proto C++ crashes
    script = f'''
import sys
import os
import bb_binary
bbb = bb_binary.common.bbb

bbb_file = {bbb_file!r}
try:
    size = os.path.getsize(bbb_file)
except Exception:
    size = None

try:
    with open(bbb_file, "rb") as f:
        while True:
            pos = f.tell()
            if size is not None:
                if pos == size:
                    sys.exit(0)  # valid
                if size - pos < 8:
                    sys.exit(1)  # invalid
            try:
                bbb.FrameContainer.read(f)
                if f.tell() <= pos:
                    sys.exit(1)  # invalid
            except EOFError:
                if size is not None:
                    if f.tell() < size:
                        sys.exit(1)  # invalid
                sys.exit(0)  # valid
            except Exception:
                if size is not None:
                    if f.tell() >= size:
                        sys.exit(0)  # valid
                sys.exit(1)  # invalid
except Exception:
    sys.exit(1)  # invalid
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            timeout=60,
            capture_output=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False


def is_dill_file_valid(dill_file):
    """
    Basic validity check for a tracking .dill file.

    A tracking output is a sequence of pickled batches written via
    `dill.dump(batch, f)` (see bb_hpc.src.jobfunctions.job_for_tracking).
    Considered invalid if the file does not exist, is 0 bytes, or cannot
    be opened.

    Use is_dill_file_valid_deep() to also verify the dill stream parses
    cleanly to end-of-file.
    """
    if not os.path.exists(dill_file):
        return False
    try:
        if os.path.getsize(dill_file) == 0:
            return False
    except Exception:
        return False
    try:
        with open(dill_file, "rb") as f:
            f.read(1)
    except Exception:
        return False
    return True


def is_dill_file_valid_deep(dill_file, max_end_gap_seconds=300.0):
    """
    Deep validation for a tracking .dill file. Two checks:

      1. Structural: every pickled batch parses cleanly and the stream ends at
         a terminal EOF (catches 0-byte and truncated-mid-pickle files).
      2. Timestamp coverage: the latest detection timestamp reaches within
         `max_end_gap_seconds` of the file's nominal end time (parsed from the
         filename). This catches "cleanly written but short" zombie files left
         behind when a tracking run silently stopped early -- those parse fine
         structurally but only cover the first part of the time window.

    The default 300 s (5 min) gap tolerates legitimate end-of-window effects
    (missing/removed trailing minute .bbb files, brief obscured-hive periods,
    track finalization) while still flagging the dramatically-short zombie
    files, which are typically short by tens of minutes. Pass a smaller value
    for a stricter check.

    Runs in a subprocess with a timeout to isolate from pathological loads.
    """
    if not is_dill_file_valid(dill_file):
        return False

    # Nominal end time from the filename: Cam_<id>_<start>--<end>.dill
    to_dt_posix = None
    try:
        from bb_binary.parsing import parse_video_fname
        base = os.path.basename(dill_file)
        if base.endswith(".dill"):
            base = base[:-5]
        _cam, _start, _end = parse_video_fname(base)
        try:
            to_dt_posix = _end.timestamp()
        except AttributeError:
            to_dt_posix = float(_end)
    except Exception:
        to_dt_posix = None  # cannot determine end -> skip the coverage check

    script = f'''
import sys
import os
import dill

dill_file = {dill_file!r}
to_dt_posix = {to_dt_posix!r}
max_end_gap = {float(max_end_gap_seconds)!r}

try:
    size = os.path.getsize(dill_file)
except Exception:
    sys.exit(1)

def _as_posix(v):
    # batch tuple element [0] is det.timestamp: a tz-aware datetime or float epoch
    try:
        return v.timestamp()
    except AttributeError:
        try:
            return float(v)
        except Exception:
            return None

max_ts = None
try:
    with open(dill_file, "rb") as f:
        while True:
            pos = f.tell()
            if pos == size:
                break  # clean EOF at end-of-file
            try:
                batch = dill.load(f)
                if f.tell() <= pos:
                    sys.exit(1)  # no progress -> invalid
            except EOFError:
                if f.tell() == size:
                    break  # clean terminal EOF
                sys.exit(1)  # premature EOF -> invalid
            except Exception:
                sys.exit(1)  # decode error -> invalid
            try:
                for row in batch:
                    ts = _as_posix(row[0])
                    if ts is not None and (max_ts is None or ts > max_ts):
                        max_ts = ts
            except TypeError:
                pass  # unexpected batch shape -> ignore for coverage
except Exception:
    sys.exit(1)

# Timestamp-coverage check
if to_dt_posix is not None:
    if max_ts is None:
        sys.exit(1)  # parsed cleanly but contains no usable detections
    if (to_dt_posix - max_ts) > max_end_gap:
        sys.exit(1)  # tracking stopped well before the nominal end of the window

sys.exit(0)
'''
    try:
        result = subprocess.run(
            [sys.executable, "-c", script],
            timeout=180,
            capture_output=True,
        )
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        return False
    except Exception:
        return False
