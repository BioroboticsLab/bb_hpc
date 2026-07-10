"""Guards against stray directories inside a bb_binary repository root.

bb_binary lays its repository out as YYYY/MM/DD/HH/MM (`Repository._DIR_FORMAT`),
and `Repository._get_directory` walks it by taking `min()`/`max()` over *every*
subdirectory it finds, then `int()`-parsing the winner's path components. It
applies no name filter. So a single non-numeric directory at the repo root -- a
`_detect_failures`, a `lost+found`, an NFS `.snapshot` -- gets selected as the
"earliest" or "latest" path and makes every read raise

    ValueError: invalid literal for int() with base 10: '<that dir name>'

`iter_fnames` hits this unconditionally, before any camera or time-window
filtering, so one stray directory takes down tracking and save_detect entirely.

`assert_clean_repo_root` is for submit scripts: it turns that failure into one
actionable message at submit time instead of an identical cryptic traceback in
every pod, hours later.

Note: the in-pod counterpart -- monkeypatching `Repository._get_directory` to
skip non-numeric names -- cannot live here. Job functions in `src/jobfunctions.py`
are copied verbatim into slurmhelper run directories and must not depend on any
module-level import, so that patch is defined inline in each job function.
"""

import os


def stray_repo_root_dirs(repo_root):
    """Return the sorted non-timestamp directory names at `repo_root`.

    Mirrors the `.isdigit()` convention already used by the repo scanners in
    check_symlinks_and_fill_in.py and get_videoinfo.py. Files are ignored --
    bb_binary's walk only considers directories, so its own
    `.bbb_repo_config.json` is already harmless.

    A root that does not exist yet (fresh season, before detect has run) has no
    stray dirs; let the downstream code report the missing path.
    """
    root = os.path.abspath(repo_root)
    if not os.path.isdir(root):
        return []
    return sorted(
        d for d in os.listdir(root)
        if not d.isdigit() and os.path.isdir(os.path.join(root, d))
    )


def assert_clean_repo_root(repo_root):
    """Fail fast if `repo_root` holds anything bb_binary's walk cannot int()-parse."""
    root = os.path.abspath(repo_root)
    stray = stray_repo_root_dirs(root)
    if not stray:
        return
    raise SystemExit(
        f"[repo_guard] bb_binary repository root {root} contains "
        f"non-timestamp directories: {stray}\n"
        f"[repo_guard] Every read (iter_fnames -> _get_latest_path) will fail with "
        f"\"ValueError: invalid literal for int() with base 10\".\n"
        f"[repo_guard] Move them out of the repo root -- do not rename them in place, "
        f"and do not hide them behind a dot (a leading '.' sorts below the digits and "
        f"poisons _get_earliest_path() instead)."
    )
