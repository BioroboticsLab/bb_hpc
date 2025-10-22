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

def list_bbb_files_incremental(pipeline_root: str, cache_dir: str) -> pd.DataFrame:
    """
    Cache daily catalogs under cache_dir/daily/bbb_YYYYMMDD.parquet.
    Only rescan days that are missing or whose *minute-div* directory mtime
    (YYYY/MM/DD/HH/<minute-div>) is newer than the cached parquet.
    """
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
                        dfs.append(pd.read_parquet(out_pq))
                        continue
                    except Exception:
                        pass
        except Exception:
            pass

        # Rescan this day only
        print("...reindexing")
        df_day = list_bbb_files(str(d))
        try:
            df_day.to_parquet(out_pq, index=False)
        except Exception:
            pass
        dfs.append(df_day)

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=["file_name","full_path","starttime","endtime","modified_time"])

def list_bbb_files(pipeline_root: str) -> pd.DataFrame:
    """
    Fast scanner for .bbb files using GNU find, dramatically reducing metadata round-trips.
    Falls back to a scandir-based walk if GNU find is unavailable.
    Returns a DataFrame with columns:
      ['file_name','full_path','starttime','endtime','modified_time']
    """
    def _from_find(root: str) -> pd.DataFrame:
        # %P  = path relative to the starting point
        # %T@ = mtime as seconds since epoch (float)
        cmd = (
            f"find {shlex.quote(root)} -type f -name '*.bbb' "
            r"-printf '%P\t%T@\n'"
        )
        out = subprocess.check_output(["bash", "-lc", cmd], stderr=subprocess.DEVNULL)
        rows = []
        append = rows.append
        for line in out.splitlines():
            try:
                rel, mtime_epoch = line.decode("utf-8", "replace").rstrip("\n").split("\t", 1)
                full = os.path.join(root, rel)
                fname = os.path.basename(rel)
                # parse start/end from BBB filename (mirrors video naming)
                _cam, start, end = parse_video_fname(fname)
                mtime = datetime.fromtimestamp(float(mtime_epoch), tz=timezone.utc)
                append({
                    "file_name": fname,
                    "full_path": full,
                    "starttime": start,
                    "endtime": end,
                    "modified_time": mtime,
                })
            except Exception:
                # Skip any malformed lines/filenames rather than stalling the run
                continue
        return pd.DataFrame(rows)    

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
                                append({
                                    "file_name": fname,
                                    "full_path": full,
                                    "starttime": start,
                                    "endtime": end,
                                    "modified_time": mtime,
                                })
                            except Exception:
                                continue
            except PermissionError:
                continue
        return pd.DataFrame(rows)

    # Try fast path (GNU find). If that fails, use Python fallback.
    try:
        return _from_find(pipeline_root)
    except Exception:
        return _fallback_scandir(pipeline_root)


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
    df = pd.DataFrame(rows)
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
    cam_id, start, end = parse_video_fname(video_basename,format='basler')
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
        if not check_read_file:  # this is the simple and fast check: just see if the file exists.
            return True
        else:
            # Attempt to open and read the file
            try:
                with open(bbb_file, 'rb') as f:
                    bbb.FrameContainer.read(f)
                return True  # File is valid
            except Exception:
                return False  # File is invalid or cannot be read