"""
Find save_detect outputs that the submitters count as done but that are not.

The submitters decide completion by mtime alone (generate.window_candidates): an
output newer than every .bbb in its window is 'done' and never scheduled again.
Before save_all_detections made read errors fatal, a bb_binary read error ended the
window quietly and the empty (or cut-short) result was saved like a finished hour.
Such files are invisible to progress_report and the submitters, so this module
opens each 'done' output's parquet footer and compares it with the .bbb catalog.

Statuses, one per (cam_id, window) unit that has .bbb data in the latest catalog:

    ok             output exists, is newer than its .bbb, and covers them
    ok_empty_tail  last detection ends early, but the .bbb frames after it have no
                   detections either (camera dark or covered, empty hive): complete
    empty_hour     0 rows WITH the column schema: a clean read that found no detections
    empty_stub     0 rows and NO columns: written by the old error handling        -> redo
    truncated      the .bbb frames after the last detection DO have detections      -> redo
    unreadable     the parquet footer cannot be read                                -> redo
    missing        no output; the submitter schedules it by itself
    stale          output older than its newest .bbb; the submitter schedules it by itself

A parquet has no row for a frame without detections, so "the last detection is
early" alone does not mean the file is cut short. Outputs whose last detection is
more than tail_tol_min before the end of the .bbb coverage are therefore checked
against the .bbb frames after that detection (verify_bbb, on by default).

The three 'redo' statuses must be moved aside before resubmitting, since the
submitter skips them as done. Read-only: nothing here moves, deletes or submits.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence

import pandas as pd

from bb_hpc.src import generate as G
from bb_hpc.src.progress import _ensure_columns, _read_latest, _utc, dates_from_catalog

#: Statuses whose file must be moved aside before resubmitting (the submitter counts them as done).
FLAG_STATUSES = ("empty_stub", "truncated", "unreadable")
#: Statuses the submitter schedules once the flagged files are gone.
REDO_STATUSES = FLAG_STATUSES + ("missing", "stale")

UNIT_COLUMNS = ["day", "cam_id", "from_dt", "to_dt", "status", "path", "n_rows",
                "last_detection", "bbb_coverage_end", "tail_gap_min", "bbb_dets_after_last",
                "latest_src_mtime", "out_mtime"]

# Detection timestamps are the frame's float posix time rounded to microseconds, so
# the frame holding the last detection can compare a hair later than it. Far below
# one frame interval (>= 1/6 s).
_FRAME_TOL = pd.Timedelta(milliseconds=10)


def bbb_dates(resultdir: str) -> list[str]:
    """Every date with .bbb data, from the latest bbb catalog.

    Not progress.catalog_dates: that prefers the raw-video catalog, which only lists
    videos still on disk, so once old videos are cleared it silently drops the early
    season -- exactly where the empty stubs were.
    """
    df = _read_latest(os.path.join(resultdir, "bbb_fileinfo"), "bbb_info_*.parquet")
    return sorted(set(dates_from_catalog(df, "starttime")) | set(dates_from_catalog(df, "endtime")))


def list_outputs(detect_dir: str) -> pd.DataFrame:
    """Stat every *.parquet in detect_dir directly (not the daily outinfo cache),
    so files moved aside today are already gone. Same parsing as build_outinfo."""
    from bb_binary.parsing import parse_video_fname

    rows = []
    if os.path.isdir(detect_dir):
        with os.scandir(detect_dir) as it:
            for entry in it:
                if not entry.name.endswith(".parquet") or not entry.is_file():
                    continue
                try:
                    cam_id, start, end = parse_video_fname(entry.name.replace(".parquet", ""))
                except Exception:
                    continue
                rows.append({
                    "cam_id": int(cam_id),
                    "from_dt": start,
                    "to_dt": end,
                    "modified_time": pd.Timestamp(entry.stat().st_mtime, unit="s", tz="UTC"),
                    "path": entry.path,
                })
    df = pd.DataFrame(rows, columns=["cam_id", "from_dt", "to_dt", "modified_time", "path"])
    for c in ("from_dt", "to_dt", "modified_time"):
        df[c] = _utc(df[c])
    return df


def bbb_coverage_end(df_bbb: pd.DataFrame, interval_hours: int = 1) -> dict:
    """(cam_id, window_start) -> latest .bbb end inside that window, clipped to the window.

    Uses the same overlap rule as window_candidates: a .bbb spanning a window
    boundary belongs to both windows.
    """
    if 24 % int(interval_hours):
        raise ValueError("interval_hours must divide 24 (windows are day-aligned)")
    if df_bbb is None or df_bbb.empty:
        return {}
    step = pd.Timedelta(hours=int(interval_hours))
    freq = f"{int(interval_hours)}h"
    df = df_bbb.dropna(subset=["cam_id", "starttime", "endtime"])
    df = df[df["endtime"] > df["starttime"]]
    first = df["starttime"].dt.floor(freq)
    last = (df["endtime"] - pd.Timedelta(microseconds=1)).dt.floor(freq)
    n = ((last - first) // step).astype(int) + 1
    rep = df.loc[df.index.repeat(n), ["cam_id", "endtime"]]
    offset = rep.groupby(level=0).cumcount().to_numpy()
    win_start = first.loc[rep.index].reset_index(drop=True) + pd.to_timedelta(offset * int(interval_hours), unit="h")
    rep = rep.reset_index(drop=True)
    rep["win_start"] = win_start
    rep["cov_end"] = rep["endtime"].where(rep["endtime"] < rep["win_start"] + step, rep["win_start"] + step)
    agg = rep.groupby(["cam_id", "win_start"])["cov_end"].max()
    return {(int(c), w): e for (c, w), e in agg.items()}


def read_footer(path: str) -> dict:
    """Row count, column count and the latest 'timestamp' from a parquet footer only.

    Falls back to reading the one column when the footer has no statistics.
    """
    import pyarrow.parquet as pq

    try:
        pf = pq.ParquetFile(path)
        md = pf.metadata
        names = pf.schema_arrow.names
        out = {"n_rows": md.num_rows, "n_cols": len(names), "last_detection": pd.NaT}
        if md.num_rows == 0 or "timestamp" not in names:
            return out
        i = names.index("timestamp")
        maxes = []
        for r in range(md.num_row_groups):
            st = md.row_group(r).column(i).statistics
            if st is None or not st.has_min_max:
                maxes = None
                break
            maxes.append(st.max)
        if maxes is None:
            maxes = [pq.read_table(path, columns=["timestamp"]).column("timestamp").to_pandas().max()]
        ts = pd.Timestamp(max(pd.Timestamp(m) for m in maxes))
        out["last_detection"] = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
        return out
    except Exception as e:
        return {"n_rows": None, "n_cols": None, "last_detection": pd.NaT, "error": f"{type(e).__name__}: {e}"}


def tail_detections(bbb_paths: Sequence[str], after: pd.Timestamp, until: pd.Timestamp,
                    loader=None) -> int:
    """Detections in .bbb frames with after < t < until. Stops at the first frame that has any.

    `bbb_paths` are the catalog's full_path values for the camera, oldest first.
    """
    if loader is None:
        from bb_binary import load_frame_container as loader
    lo = (after + _FRAME_TOL).timestamp()
    hi = until.timestamp()
    for path in bbb_paths:
        for frame in loader(path).frames:
            if lo < frame.timestamp < hi:
                n = len(frame.detectionsDP) + len(frame.detectionsBees)
                if n:
                    return n
    return 0


def check_save_detect(resultdir: str, dates: Sequence[str], tail_tol_min: float = 3.0,
                      interval_hours: int = 1, detect_dir: str | None = None,
                      max_workers: int = 16, verify_bbb: bool = True, loader=None) -> pd.DataFrame:
    """Classify every save_detect unit on `dates`. Returns one row per unit (UNIT_COLUMNS).

    verify_bbb=False skips reading .bbb frames and flags every early-ending output as
    'truncated' (fast, but sparse or dark hours become false positives).
    """
    cache_dir = os.path.join(resultdir, "bbb_fileinfo")
    detect_dir = detect_dir or os.path.join(resultdir, "data_alldetections")

    df_bbb = _read_latest(cache_dir, "bbb_info_*.parquet")
    if df_bbb is None:
        raise FileNotFoundError(f"no bbb_info_*.parquet in {cache_dir} (run `python -m bb_hpc.get_fileinfo`)")
    df_bbb = _ensure_columns(df_bbb, ["file_name", "starttime", "endtime", "modified_time"])
    for c in ("starttime", "endtime", "modified_time"):
        df_bbb[c] = _utc(df_bbb[c])
    df_bbb = G.ensure_cam_id(df_bbb)

    df_out = list_outputs(detect_dir)
    path_of = dict(zip(zip(df_out["cam_id"], df_out["from_dt"], df_out["to_dt"]), df_out["path"]))

    # Same predicate the submitters use for missing / stale / done.
    records = G.window_candidates(df_bbb, G.build_out_index(df_out), list(dates), interval_hours)
    if not records:
        return pd.DataFrame(columns=UNIT_COLUMNS)
    units = pd.DataFrame(records)
    for c in ("from_dt", "to_dt", "latest_src_mtime", "out_mtime"):
        units[c] = _utc(units[c])
    units["path"] = [path_of.get((c, f, t)) for c, f, t in zip(units["cam_id"], units["from_dt"], units["to_dt"])]
    cov = bbb_coverage_end(df_bbb, interval_hours)
    units["bbb_coverage_end"] = _utc(pd.Series([cov.get((c, f)) for c, f in zip(units["cam_id"], units["from_dt"])],
                                               index=units.index, dtype="object"))

    # Only 'done' units need their footer read: the submitter redoes the rest anyway.
    done = units.index[units["status"] == "done"]
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        footers = list(ex.map(read_footer, units.loc[done, "path"]))
    units["n_rows"] = pd.Series(dtype="float64")
    units["last_detection"] = pd.Series(dtype="datetime64[ns, UTC]")
    tol = pd.Timedelta(minutes=float(tail_tol_min))
    for idx, ft in zip(done, footers):
        units.at[idx, "n_rows"] = ft["n_rows"]
        units.at[idx, "last_detection"] = ft["last_detection"]
        if "error" in ft:
            units.at[idx, "status"] = "unreadable"
        elif ft["n_rows"] == 0:
            units.at[idx, "status"] = "empty_hour" if ft["n_cols"] else "empty_stub"
        elif pd.notna(ft["last_detection"]) and units.at[idx, "bbb_coverage_end"] - ft["last_detection"] > tol:
            units.at[idx, "status"] = "truncated"
        else:
            units.at[idx, "status"] = "ok"

    # An early last detection is only a truncation if the .bbb has detections after it.
    units["bbb_dets_after_last"] = pd.Series(dtype="float64")
    suspects = units.index[units["status"] == "truncated"]
    if verify_bbb and len(suspects) and "full_path" in df_bbb.columns:
        # The catalog can list a file twice; read each once.
        cat = df_bbb.drop_duplicates("full_path")

        def verify(idx):
            u = units.loc[idx]
            sel = cat[(cat["cam_id"] == u["cam_id"]) & (cat["endtime"] > u["last_detection"])
                      & (cat["starttime"] < u["to_dt"])].sort_values("starttime")
            try:
                return tail_detections(sel["full_path"].tolist(), u["last_detection"], u["to_dt"], loader)
            except Exception as e:
                # Cannot tell; keep it flagged (a fresh save_detect run fails loudly on it).
                print(f"[check_save_detect] could not read .bbb for cam={u['cam_id']} {u['from_dt']}: "
                      f"{type(e).__name__}: {e}", flush=True)
                return None

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            found = list(ex.map(verify, suspects))
        for idx, n in zip(suspects, found):
            units.at[idx, "bbb_dets_after_last"] = n
            if n == 0:
                units.at[idx, "status"] = "ok_empty_tail"

    units["tail_gap_min"] = ((units["bbb_coverage_end"] - units["last_detection"]).dt.total_seconds() / 60).round(1)
    units["day"] = units["from_dt"].dt.strftime("%Y%m%d")
    return units[UNIT_COLUMNS].sort_values(["from_dt", "cam_id"]).reset_index(drop=True)


def by_day(units: pd.DataFrame) -> pd.DataFrame:
    """Per-day counts of every status, for days with anything that is not ok."""
    if units.empty:
        return pd.DataFrame()
    t = pd.crosstab(units["day"], units["status"])
    bad = [c for c in t.columns if c in REDO_STATUSES]
    return t[t[bad].sum(axis=1) > 0] if bad else t.iloc[0:0]


def fix_commands(units: pd.DataFrame, resultdir: str, out_dir: str, backend: str = "k8s",
                 paths_flag: str = "", today: str | None = None) -> list[str]:
    """Shell lines that repair what `units` found. Empty when nothing needs doing."""
    flagged = units[units["status"].isin(FLAG_STATUSES)]
    redo_days = sorted(units.loc[units["status"].isin(REDO_STATUSES), "day"].unique())
    if not redo_days:
        return []
    today = today or pd.Timestamp.now(tz="UTC").strftime("%Y%m%d")
    lines = ["set -euo pipefail", ""]
    if len(flagged):
        q = os.path.join(resultdir, f"data_alldetections_quarantine_{today}")
        lines += [
            f"# 1. move {len(flagged)} output(s) the submitter counts as done aside (not deleted)",
            f'Q="{q}"',
            'mkdir -p "$Q"',
            f"xargs -d '\\n' -a \"{os.path.join(out_dir, 'flagged_files.txt')}\" mv -t \"$Q\"",
            "",
            "# 2. rebuild the outputs catalog the submitter reads",
            f"python -m bb_hpc.get_fileinfo --what outputs{paths_flag}",
            "",
        ]
    lines += [
        f"# {'3' if len(flagged) else '1'}. resubmit: schedules only missing/stale units on these dates",
        f"python -m bb_hpc.running_{backend}.save_detect_submit --dates " + " ".join(redo_days),
        "",
        "# after the jobs finish, check again (expect no empty_stub / truncated / unreadable)",
        f"python -m bb_hpc.check_save_detect --resultdir {resultdir}",
    ]
    return lines


def write_outputs(units: pd.DataFrame, out_dir: str, commands: list[str]) -> None:
    os.makedirs(out_dir, exist_ok=True)
    units.to_csv(os.path.join(out_dir, "units.csv"), index=False)
    flagged = units.loc[units["status"].isin(FLAG_STATUSES), "path"].dropna()
    with open(os.path.join(out_dir, "flagged_files.txt"), "w") as f:
        f.write("".join(p + "\n" for p in flagged))
    redo_days = sorted(units.loc[units["status"].isin(REDO_STATUSES), "day"].unique())
    with open(os.path.join(out_dir, "redo_dates.txt"), "w") as f:
        f.write(" ".join(redo_days) + ("\n" if redo_days else ""))
    with open(os.path.join(out_dir, "commands.sh"), "w") as f:
        f.write("\n".join(commands) + ("\n" if commands else ""))
