import os, glob
import pandas as pd
from bb_binary.parsing import parse_video_fname

import bb_binary
from datetime import datetime 

bbb = bb_binary.common.bbb

#################################################################
##### listing functions for making database of existing and already processed videos
#################################################################
def list_bbb_files(pipeline_root):
    """
    Scan the entire bb_binary repository for .bbb files,
    record their time windows and modification times,
    and write the full catalog to a parquet.
    Intended to run every 8 hours (e.g. via cron).
    """
    rows = []
    for root, _, files in os.walk(pipeline_root):
        print(root)
        for f in files:
            if not f.endswith('.bbb'):
                continue
            path = os.path.join(root, f)
            mtime = datetime.fromtimestamp(os.stat(path).st_mtime)
            _, start, end = parse_video_fname(f)
            rows.append({
                'file_name': f,
                'full_path': path,
                'starttime': start,
                'endtime': end,
                'modified_time': mtime
            })
    return pd.DataFrame(rows)


def build_outinfo(output_dir, CACHE_DIR, extension, outname):
    # Helper to read existing outputs
    rows = []
    pattern = os.path.join(output_dir, f"*.{extension}")
    for path in glob.glob(pattern):
        fname = os.path.basename(path)
        cam_id, start, end = parse_video_fname(fname.replace(f'.{extension}',''))
        mtime = datetime.fromtimestamp(os.path.getmtime(path))
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