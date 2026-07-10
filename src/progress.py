#!/usr/bin/env python3
"""
Pipeline progress: what has processed, what has not, and how to fill the gaps.

This module classifies every work unit of every stage into done / pending /
skipped, using the SAME completion predicates the submitters use (imported from
bb_hpc.src.generate). If a unit shows up as pending here, `<stage>_submit` will
schedule it; if it shows as done, the submitter will skip it. test/test_progress.py
pins that invariant.

It is WRITE-FREE apart from ProgressReport.save(). In particular it never calls
generate_jobs_save_detect / generate_jobs_tracking, which os.makedirs() their
output dirs before their first yield.

Typical use:

    # compute a snapshot (the CLI does this for you)
    from bb_hpc.src.progress import build_report, resolve_paths
    rep = build_report(**resolve_paths(), dates=["20260708", "20260709"])
    print(rep.render())
    rep.save()

    # later, from a notebook: load what the CLI wrote, no recomputation
    from bb_hpc.src.progress import load_latest
    rep = load_latest()
    rep.summary()
    rep.stages["tracking"].pending_units()
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Iterable, Sequence

import pandas as pd

from bb_hpc.src import generate as G
from bb_hpc.src.fileinfo import get_bbb_file_path

STAGES = ("detect", "save_detect", "tracking", "frame_extract", "background", "rpi")
BACKENDS = ("k8s", "slurm", "docker")

#: Statuses that mean "the submitter will schedule this unit".
PENDING_STATUSES = frozenset({"missing", "stale", "pending", "zero_byte_stub"})
#: Statuses that mean "the submitter will not schedule this, and it isn't done either".
SKIPPED_STATUSES = frozenset({"skipped_no_txt", "skipped_no_frames", "skipped_min_frames"})

_PROGRESS_SUBDIR = os.path.join("bbb_fileinfo", "progress")


# --------------------------------------------------------------------------- #
# Paths / catalogs
# --------------------------------------------------------------------------- #
def resolve_paths(prefer: str | None = None) -> dict:
    """
    Resolve the roots this report reads, from bb_hpc.settings.

    prefer: 'local' | 'hpc' | None (auto: prefer *_local when it exists on disk).
    Mirrors get_fileinfo._pick_paths, extended to the video/frames/backgrounds roots.
    """
    from bb_hpc import settings

    def pick(name: str) -> str:
        local = getattr(settings, f"{name}_local", "") or ""
        hpc = getattr(settings, f"{name}_hpc", "") or ""
        if prefer == "local" and local:
            return local
        if prefer == "hpc" and hpc:
            return hpc
        if local and os.path.exists(local):
            return local
        if hpc and os.path.exists(hpc):
            return hpc
        return local or hpc

    return {
        "resultdir": pick("resultdir"),
        "pipeline_root": pick("pipeline_root"),
        "videodir": pick("videodir"),
        "pi_videodir": pick("pi_videodir"),
        "frames_dir": pick("frames_dir"),
        "backgrounds_dir": pick("backgrounds_dir"),
    }


def _cache_dir(resultdir: str) -> str:
    return os.path.join(resultdir, "bbb_fileinfo")


def default_out_dir(resultdir: str) -> str:
    return os.path.join(resultdir, _PROGRESS_SUBDIR)


def _read_latest(cache_dir: str, pattern: str) -> pd.DataFrame | None:
    """Newest catalog matching `pattern` (dated snapshots sort lexicographically)."""
    files = sorted(glob.glob(os.path.join(cache_dir, pattern)))
    if not files:
        return None
    try:
        return pd.read_parquet(files[-1])
    except Exception:
        return None


def _read_one(cache_dir: str, name: str) -> pd.DataFrame | None:
    path = os.path.join(cache_dir, name)
    if not os.path.exists(path):
        return None
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def _utc(series: pd.Series) -> pd.Series:
    """Coerce a column to tz-aware UTC datetimes."""
    out = pd.to_datetime(series, utc=True, errors="coerce")
    return out


def _ensure_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """
    Add any missing columns as all-NA.

    An empty catalog written by an older build_outinfo / list_bbb_files has ZERO
    columns (pd.DataFrame([]) declares no schema), which is the normal state early
    in a season before a stage has produced its first output. Readers must not
    KeyError on that.
    """
    df = df.copy()
    for c in columns:
        if c not in df.columns:
            df[c] = pd.Series(dtype="object")
    return df


def _day_str(ts) -> str | None:
    ts = pd.Timestamp(ts)
    return None if pd.isna(ts) else ts.strftime("%Y%m%d")


def dates_from_catalog(df: pd.DataFrame | None, column: str = "starttime") -> list[str]:
    """Every YYYYMMDD present in a catalog, ascending. Used as the default date range."""
    if df is None or len(df) == 0 or column not in df.columns:
        return []
    days = _utc(df[column]).dt.strftime("%Y%m%d").dropna().unique().tolist()
    return sorted(days)


def catalog_dates(resultdir: str) -> list[str]:
    """
    Every date the pipeline knows about: the raw-video catalog if present, else the
    bbb catalog. This is the default date range for a report, so it works mid-season
    with no manual date list.
    """
    cache_dir = _cache_dir(resultdir)
    dates = dates_from_catalog(_read_one(cache_dir, "video_info_all.parquet"))
    if not dates:
        dates = dates_from_catalog(_read_latest(cache_dir, "bbb_info_*.parquet"))
    return dates


def filter_dates(dates: Sequence[str], since: str | None = None,
                 until: str | None = None, last_days: int | None = None) -> list[str]:
    out = sorted(dates)
    if since:
        out = [d for d in out if d >= since]
    if until:
        out = [d for d in out if d <= until]
    if last_days:
        out = out[-int(last_days):]
    return out


# --------------------------------------------------------------------------- #
# Result containers
# --------------------------------------------------------------------------- #
@dataclass
class StageProgress:
    """One row per work unit, plus per-day rollups and the dates needing a resubmit."""
    stage: str
    units: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    available: bool = True
    reason: str = ""

    def pending_units(self) -> pd.DataFrame:
        if self.units.empty:
            return self.units
        return self.units[self.units["status"].isin(PENDING_STATUSES)]

    @property
    def by_day(self) -> pd.DataFrame:
        """day, total, done, pending, skipped, pct  (pct excludes skipped units)."""
        cols = ["day", "total", "done", "pending", "skipped", "pct"]
        if self.units.empty:
            return pd.DataFrame(columns=cols)

        u = self.units
        g = u.groupby("day", dropna=False)
        out = pd.DataFrame({
            "total": g.size(),
            "done": g["status"].apply(lambda s: (s == "done").sum()),
            "pending": g["status"].apply(lambda s: s.isin(PENDING_STATUSES).sum()),
            "skipped": g["status"].apply(lambda s: s.isin(SKIPPED_STATUSES).sum()),
        }).reset_index()
        # A skipped unit can never be produced, so it must not drag the percentage
        # down forever; it is excluded from the denominator and surfaced separately.
        # A day where everything is skipped has nothing left to do -> 100%.
        denom = out["done"] + out["pending"]
        out["pct"] = (100.0 * out["done"] / denom.where(denom > 0)).fillna(100.0)
        return out.sort_values("day").reset_index(drop=True)[cols]

    @property
    def pending_dates(self) -> list[str]:
        p = self.pending_units()
        if p.empty:
            return []
        return sorted(p["day"].dropna().unique().tolist())

    def totals(self) -> dict:
        n = len(self.units)
        if n == 0:
            return {"total": 0, "done": 0, "pending": 0, "skipped": 0, "pct": None}
        s = self.units["status"]
        done = int((s == "done").sum())
        pending = int(s.isin(PENDING_STATUSES).sum())
        skipped = int(s.isin(SKIPPED_STATUSES).sum())
        denom = done + pending
        # Skipped units are excluded from the denominator (they can never be
        # produced). All-skipped therefore means nothing is left to do -> 100%.
        # Only a stage with no units at all has an undefined percentage.
        return {
            "total": n,
            "done": done,
            "pending": pending,
            "skipped": skipped,
            "pct": round(100.0 * done / denom, 2) if denom else 100.0,
        }

    def status_counts(self) -> dict:
        if self.units.empty:
            return {}
        return {k: int(v) for k, v in self.units["status"].value_counts().items()}


@dataclass
class ProgressReport:
    generated_at: datetime
    config: dict
    dates: list[str]
    stages: dict[str, StageProgress]
    backend: str = "k8s"

    # ---------------- summary / commands ----------------
    def summary(self) -> dict:
        out = {
            "generated_at": self.generated_at.isoformat(),
            "backend": self.backend,
            "n_dates": len(self.dates),
            "date_first": self.dates[0] if self.dates else None,
            "date_last": self.dates[-1] if self.dates else None,
        }
        for name, sp in self.stages.items():
            if not sp.available:
                out[name] = {"available": False, "reason": sp.reason}
                continue
            t = sp.totals()
            t["available"] = True
            t["status_counts"] = sp.status_counts()
            t["pending_dates"] = sp.pending_dates
            if sp.notes:
                t["notes"] = sp.notes
            out[name] = t
        return out

    def commands(self, backend: str | None = None) -> dict[str, str]:
        backend = backend or self.backend
        cmds = {}
        for name, sp in self.stages.items():
            if not sp.available:
                continue
            cmd = build_command(name, sp, backend)
            if cmd:
                cmds[name] = cmd
        return cmds

    # ---------------- rendering ----------------
    def render(self, markdown: bool = False, max_days: int = 14) -> str:
        return _render(self, markdown=markdown, max_days=max_days)

    # ---------------- persistence ----------------
    def save(self, out_dir: str | None = None) -> str:
        """
        Write an atomic snapshot to <out_dir>/latest/ and append one row to
        <out_dir>/history.jsonl. Never creates symlinks (they raise EIO on the
        CIFS share), so `latest/` is a real directory rewritten in place.
        """
        out_dir = out_dir or default_out_dir(self.config["resultdir"])
        latest = os.path.join(out_dir, "latest")
        os.makedirs(latest, exist_ok=True)

        meta = {
            "generated_at": self.generated_at.isoformat(),
            "backend": self.backend,
            "dates": self.dates,
            "config": self.config,
            "stages": {n: {"available": sp.available, "reason": sp.reason, "notes": sp.notes}
                       for n, sp in self.stages.items()},
        }
        with open(os.path.join(latest, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2, default=str)

        with open(os.path.join(latest, "summary.json"), "w") as f:
            json.dump(self.summary(), f, indent=2, default=str)

        for name, sp in self.stages.items():
            if not sp.available:
                continue
            sp.units.to_parquet(os.path.join(latest, f"{name}_units.parquet"), index=False)
            sp.by_day.to_parquet(os.path.join(latest, f"{name}_by_day.parquet"), index=False)

        with open(os.path.join(latest, "report.md"), "w") as f:
            f.write(self.render(markdown=True))

        cmds = self.commands()
        with open(os.path.join(latest, "commands.sh"), "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(f"# generated {self.generated_at.isoformat()} by bb_hpc.progress_report\n")
            f.write("# Resubmit commands for everything that has not finished.\n\n")
            if not cmds:
                f.write("# nothing pending\n")
            detect_sp = self.stages.get("detect")
            if detect_sp is not None and detect_sp.available:
                cleanup = cleanup_command(detect_sp)
                if cleanup:
                    f.write("# --- remove 0-byte .bbb stubs before resubmitting detect ---\n")
                    f.write(f"{cleanup}\n\n")
            for name, cmd in cmds.items():
                f.write(f"# --- {name} ---\n{cmd}\n\n")

        row = {k: v for k, v in self.summary().items()}
        with open(os.path.join(out_dir, "history.jsonl"), "a") as f:
            f.write(json.dumps(row, default=str) + "\n")

        return latest


# --------------------------------------------------------------------------- #
# Loading a saved snapshot (what the notebook uses)
# --------------------------------------------------------------------------- #
def load_latest(out_dir: str | None = None, prefer: str | None = None) -> ProgressReport:
    if out_dir is None:
        out_dir = default_out_dir(resolve_paths(prefer)["resultdir"])
    latest = os.path.join(out_dir, "latest")
    meta_path = os.path.join(latest, "meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(
            f"No progress snapshot at {latest}. Run `python -m bb_hpc.progress_report` first."
        )
    with open(meta_path) as f:
        meta = json.load(f)

    stages = {}
    for name, info in meta["stages"].items():
        units_path = os.path.join(latest, f"{name}_units.parquet")
        units = pd.read_parquet(units_path) if os.path.exists(units_path) else pd.DataFrame()
        stages[name] = StageProgress(
            stage=name, units=units, notes=info.get("notes", []),
            available=info.get("available", True), reason=info.get("reason", ""),
        )

    return ProgressReport(
        generated_at=datetime.fromisoformat(meta["generated_at"]),
        config=meta["config"], dates=meta["dates"], stages=stages,
        backend=meta.get("backend", "k8s"),
    )


def load_history(out_dir: str | None = None, prefer: str | None = None) -> pd.DataFrame:
    """Every past run's summary as a DataFrame -- progress over time."""
    if out_dir is None:
        out_dir = default_out_dir(resolve_paths(prefer)["resultdir"])
    path = os.path.join(out_dir, "history.jsonl")
    if not os.path.exists(path):
        return pd.DataFrame()

    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue

    flat = []
    for r in rows:
        out = {"generated_at": r.get("generated_at")}
        for stage in STAGES:
            s = r.get(stage)
            if isinstance(s, dict) and s.get("available"):
                out[f"{stage}_pct"] = s.get("pct")
                out[f"{stage}_pending"] = s.get("pending")
        flat.append(out)
    df = pd.DataFrame(flat)
    if "generated_at" in df.columns:
        df["generated_at"] = pd.to_datetime(df["generated_at"], errors="coerce")
    return df.sort_values("generated_at").reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Stage classifiers
# --------------------------------------------------------------------------- #
def _unavailable(stage: str, reason: str) -> StageProgress:
    return StageProgress(stage=stage, units=pd.DataFrame(), available=False, reason=reason)


def detect_progress(resultdir: str, dates: Sequence[str]) -> StageProgress:
    """
    Unit = one raw video. Done iff its expected .bbb is catalogued AND non-empty.

    A 0-byte .bbb is an aborted write (the stub bb_binary leaves when a
    cross-boundary symlink hits EIO). is_bbb_file_valid_basicmatch rejects it, so
    we must too -- counting it as done is how the old notebook reported 100% for
    days that still had work.
    """
    cache_dir = _cache_dir(resultdir)
    df_vid = _read_one(cache_dir, "video_info_all.parquet")
    if df_vid is None:
        return _unavailable("detect", "video_info_all.parquet missing (run `python -m bb_hpc.get_videoinfo`)")
    df_bbb = _read_latest(cache_dir, "bbb_info_*.parquet")
    if df_bbb is None:
        return _unavailable("detect", "bbb_info_*.parquet missing (run `python -m bb_hpc.get_fileinfo`)")

    df_vid = _ensure_columns(df_vid, ["file_name", "starttime", "cam"])
    df_vid["starttime"] = _utc(df_vid["starttime"])
    df_vid["day"] = df_vid["starttime"].dt.strftime("%Y%m%d")
    df_vid = df_vid[df_vid["day"].isin(set(dates))]
    if df_vid.empty:
        return StageProgress("detect", pd.DataFrame(columns=["day", "cam", "file_name", "status"]))

    df_bbb = _ensure_columns(df_bbb, ["file_name"])
    good = df_bbb
    if "is_valid" in df_bbb.columns:
        good = good[good["is_valid"].fillna(False)]
    elif "file_size" in df_bbb.columns:
        good = good[good["file_size"].fillna(0) > 0]
    done_names = set(good["file_name"].astype(str))

    all_names = set(df_bbb["file_name"].astype(str))
    stub_names = all_names - done_names

    expected = df_vid["file_name"].astype(str).map(
        lambda fn: os.path.basename(get_bbb_file_path(fn))
    )

    status = pd.Series("missing", index=df_vid.index, dtype=object)
    status[expected.isin(done_names)] = "done"
    status[expected.isin(stub_names)] = "zero_byte_stub"

    units = pd.DataFrame({
        "day": df_vid["day"].values,
        "cam": df_vid["cam"].values if "cam" in df_vid.columns else None,
        "file_name": df_vid["file_name"].values,
        "expected_bbb": expected.values,
        "status": status.values,
    })

    notes = []
    n_stub = int((units["status"] == "zero_byte_stub").sum())
    if n_stub:
        notes.append(
            f"{n_stub} expected .bbb file(s) are 0-byte stubs (aborted writes). Plain "
            f"`--use-fileinfo` skips them by name; the emitted command adds --check-read-bbb."
        )
    return StageProgress("detect", units, notes=notes)


def _window_stage(stage: str, resultdir: str, dates: Sequence[str],
                  outinfo_name: str, interval_hours: int = 1) -> StageProgress:
    cache_dir = _cache_dir(resultdir)
    df_bbb = _read_latest(cache_dir, "bbb_info_*.parquet")
    if df_bbb is None:
        return _unavailable(stage, "bbb_info_*.parquet missing (run `python -m bb_hpc.get_fileinfo`)")
    df_out = _read_one(cache_dir, outinfo_name)
    if df_out is None:
        return _unavailable(stage, f"{outinfo_name} missing (run `python -m bb_hpc.get_fileinfo`)")

    df_bbb = _ensure_columns(df_bbb, ["file_name", "starttime", "endtime", "modified_time"])
    for c in ("starttime", "endtime", "modified_time"):
        df_bbb[c] = _utc(df_bbb[c])
    df_out = _ensure_columns(df_out, ["cam_id", "from_dt", "to_dt", "modified_time"])
    for c in ("from_dt", "to_dt", "modified_time"):
        df_out[c] = _utc(df_out[c])

    # Same predicate the submitters use -- and write-free, unlike the generators.
    records = G.window_candidates(df_bbb, G.build_out_index(df_out), list(dates), interval_hours)
    if not records:
        return StageProgress(stage, pd.DataFrame(columns=["day", "cam_id", "from_dt", "to_dt", "status"]))

    units = pd.DataFrame(records)
    units["day"] = units["from_dt"].map(_day_str)
    cols = ["day", "cam_id", "from_dt", "to_dt", "latest_src_mtime", "out_mtime", "status"]
    return StageProgress(stage, units[cols])


def save_detect_progress(resultdir: str, dates: Sequence[str], interval_hours: int = 1) -> StageProgress:
    return _window_stage("save_detect", resultdir, dates, "save_detect_outinfo.parquet", interval_hours)


def tracking_progress(resultdir: str, dates: Sequence[str], interval_hours: int = 1) -> StageProgress:
    return _window_stage("tracking", resultdir, dates, "save_tracking_outinfo.parquet", interval_hours)


def frame_extract_progress(videodir: str, frames_dir: str, dates: Sequence[str],
                           settings_dict: dict | None = None) -> StageProgress:
    """Unit = (date, cam). A cam with no .txt sidecars is `skipped_no_txt`: the engine cannot run it."""
    s = settings_dict or {}
    if not videodir or not os.path.isdir(videodir):
        return _unavailable("frame_extract", f"videodir not found: {videodir}")

    interval = int(s.get("interval_in_sec", 60))
    fps = int(s.get("fps", 3))
    fmt = s.get("file_format", "png")

    rows = []
    for u in G.iter_frame_extract_units(videodir, frames_dir, list(dates)):
        if not u["txts"]:
            status = "skipped_no_txt"
        elif G.frame_extract_is_done(u["out_dir"], u["txts"], interval, fps, fmt):
            status = "done"
        else:
            status = "pending"
        rows.append({"day": u["date"], "cam": u["cam"], "n_videos": len(u["txts"]),
                     "out_dir": u["out_dir"], "status": status})

    notes = []
    if G._import_frame_naming() is None:
        notes.append("frame_extractor not importable here; using a presence-based (coarser) done-check.")
    return StageProgress("frame_extract", pd.DataFrame(rows), notes=notes)


def background_progress(frames_dir: str, backgrounds_dir: str, dates: Sequence[str],
                        settings_dict: dict | None = None) -> StageProgress:
    """
    Unit = (date, cam). Statuses mirror generate_jobs_background exactly, including
    the two skips it makes silently: a cam dir with no frames, and (windowed mode)
    a cam with fewer than min_frames usable frames.
    """
    s = settings_dict or {}
    if not frames_dir or not os.path.isdir(frames_dir):
        return _unavailable("background", f"frames_dir not found: {frames_dir}")

    interval = s.get("frame_interval_sec", None)
    window = s.get("background_window", None)
    wsize = int(s.get("window_size", 10))
    nmed = int(s.get("num_median_images", 200))
    min_frames = int(s.get("min_frames", 3))

    rows, notes = [], []
    scopes = G.iter_background_scopes(frames_dir, backgrounds_dir, list(dates))
    for scan_dir, output_path, scope_id, _explicit in scopes:
        if not os.path.isdir(scan_dir):
            continue
        for cam, _cam_dir, frame_names in G.iter_background_cams(scan_dir):
            if not frame_names:
                rows.append({"day": scope_id, "cam": cam, "n_frames": 0, "tag": None,
                             "status": "skipped_no_frames"})
                continue

            out_cam_dir = os.path.join(output_path, cam)
            tag, note = G.resolve_background_tag(out_cam_dir, interval, window, wsize, nmed)
            if note and note not in notes:
                notes.append(note)
            out_tag_dir = os.path.join(out_cam_dir, tag)

            kept = G.background_kept_count(frame_names, interval)
            if window and kept < min_frames:
                status = "skipped_min_frames"
            elif G.background_is_done(out_tag_dir, frame_names, interval, window):
                status = "done"
            else:
                status = "pending"

            rows.append({"day": scope_id, "cam": cam, "n_frames": len(frame_names),
                         "tag": tag, "status": status})

    if G._import_windowing() is None and not notes:
        # No per-cam note fired (nothing on disk to reconcile against), but the
        # config tag is still only a guess here -- say so once.
        notes.append("background_generator not importable here; the config tag is a "
                     "mirrored guess. Install it on this host for the canonical tag.")
    return StageProgress("background", pd.DataFrame(rows), notes=notes)


def rpi_progress(resultdir: str, dates: Sequence[str], clahe: bool = True) -> StageProgress:
    """Unit = one RPi video; done iff its detections parquet exists for the chosen CLAHE mode."""
    cache_dir = _cache_dir(resultdir)
    df = _read_latest(cache_dir, "rpi_info_*.parquet")
    if df is None:
        return _unavailable("rpi", "rpi_info_*.parquet missing (run `python -m bb_hpc.get_fileinfo --what rpi`)")

    df = _ensure_columns(df, ["date", "cam", "video_name", "detections_clahe", "detections_noclahe"])
    df["day"] = df["date"].astype(str).str.replace("-", "", regex=False)
    df = df[df["day"].isin(set(dates))]
    if df.empty:
        return StageProgress("rpi", pd.DataFrame(columns=["day", "cam", "video_name", "status"]))

    col = "detections_clahe" if clahe else "detections_noclahe"
    units = pd.DataFrame({
        "day": df["day"].values,
        "cam": df["cam"].values,
        "video_name": df["video_name"].values,
        "detections_clahe": df["detections_clahe"].values,
        "detections_noclahe": df["detections_noclahe"].values,
        "status": ["done" if v else "missing" for v in df[col].fillna(False)],
    })
    return StageProgress("rpi", units, notes=[f"done-ness measured against {col}"])


# --------------------------------------------------------------------------- #
# Resubmit commands
# --------------------------------------------------------------------------- #
_SUBMIT_MODULE = {
    "detect": "detect_submit",
    "save_detect": "save_detect_submit",
    "tracking": "tracking_submit",
    "frame_extract": "frame_extract_submit",
    "background": "background_submit",
    "rpi": "detect_rpi_submit",
}


def build_command(stage: str, sp: StageProgress, backend: str = "k8s") -> str | None:
    """The exact command that will redo this stage's pending units. None when nothing pends."""
    if backend not in BACKENDS:
        raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
    dates = sp.pending_dates
    if not dates:
        return None

    mod = f"bb_hpc.running_{backend}.{_SUBMIT_MODULE[stage]}"
    flags = ""
    if stage == "detect":
        # Plain --use-fileinfo skips a 0-byte stub by name, so it would never redo
        # one. --check-read-bbb forces a real validity check.
        has_stub = (not sp.units.empty) and (sp.units["status"] == "zero_byte_stub").any()
        flags = " --use-fileinfo --check-read-bbb" if has_stub else " --use-fileinfo"

    return f"python -m {mod}{flags} --dates " + " ".join(dates)


def cleanup_command(sp: StageProgress) -> str | None:
    """For detect: the command that removes 0-byte .bbb stubs before resubmitting."""
    if sp.stage != "detect" or sp.units.empty:
        return None
    stubs = sp.units[sp.units["status"] == "zero_byte_stub"]
    if stubs.empty:
        return None
    dates = sorted(stubs["day"].dropna().unique().tolist())
    return "python -m bb_hpc.scan_and_remove_invalid_bbb_files --dates " + " ".join(dates)


# --------------------------------------------------------------------------- #
# Build
# --------------------------------------------------------------------------- #
def build_report(
    *,
    resultdir: str,
    pipeline_root: str = "",
    videodir: str = "",
    pi_videodir: str = "",
    frames_dir: str = "",
    backgrounds_dir: str = "",
    dates: Sequence[str] | None = None,
    stages: Iterable[str] = STAGES,
    backend: str = "k8s",
    frame_extract_settings: dict | None = None,
    background_settings: dict | None = None,
    rpi_clahe: bool = True,
    interval_hours: int = 1,
) -> ProgressReport:
    """Compute one atomic snapshot across `stages` for `dates`."""
    stages = tuple(stages)

    if dates is None:
        dates = catalog_dates(resultdir)
    dates = sorted(set(dates))

    out: dict[str, StageProgress] = {}
    if "detect" in stages:
        out["detect"] = detect_progress(resultdir, dates)
    if "save_detect" in stages:
        out["save_detect"] = save_detect_progress(resultdir, dates, interval_hours)
    if "tracking" in stages:
        out["tracking"] = tracking_progress(resultdir, dates, interval_hours)
    if "frame_extract" in stages:
        out["frame_extract"] = frame_extract_progress(videodir, frames_dir, dates, frame_extract_settings)
    if "background" in stages:
        out["background"] = background_progress(frames_dir, backgrounds_dir, dates, background_settings)
    if "rpi" in stages:
        out["rpi"] = rpi_progress(resultdir, dates, clahe=rpi_clahe)

    config = {
        "resultdir": resultdir, "pipeline_root": pipeline_root, "videodir": videodir,
        "pi_videodir": pi_videodir, "frames_dir": frames_dir, "backgrounds_dir": backgrounds_dir,
        "interval_hours": interval_hours,
    }
    return ProgressReport(
        generated_at=datetime.now(timezone.utc), config=config,
        dates=list(dates), stages=out, backend=backend,
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #
def _bar(pct: float | None, width: int = 24) -> str:
    if pct is None:
        return " " * width
    filled = int(round(width * pct / 100.0))
    return "█" * filled + "·" * (width - filled)


def _render(rep: ProgressReport, markdown: bool = False, max_days: int = 14) -> str:
    L: list[str] = []
    h1, h2 = ("# ", "## ") if markdown else ("", "")
    rule = "" if markdown else "-" * 78

    L.append(f"{h1}bb_hpc pipeline progress")
    L.append("")
    L.append(f"generated : {rep.generated_at.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    if rep.dates:
        L.append(f"dates     : {rep.dates[0]} .. {rep.dates[-1]}  ({len(rep.dates)} days)")
    L.append(f"resultdir : {rep.config.get('resultdir')}")
    L.append(f"backend   : {rep.backend}")
    L.append("")

    # ---- overview ----
    L.append(f"{h2}Overview" if markdown else "OVERVIEW")
    if markdown:
        L.append("")
        L.append("| stage | done | total | pct | pending | skipped |")
        L.append("|---|---:|---:|---:|---:|---:|")
    else:
        L.append(rule)

    for name in STAGES:
        sp = rep.stages.get(name)
        if sp is None:
            continue
        if not sp.available:
            if markdown:
                L.append(f"| {name} | — | — | — | — | n/a: {sp.reason} |")
            else:
                L.append(f"  {name:<14} n/a  ({sp.reason})")
            continue
        t = sp.totals()
        pct = t["pct"]
        pcts = "  —  " if pct is None else f"{pct:5.1f}%"
        if markdown:
            L.append(f"| {name} | {t['done']} | {t['total']} | {pcts.strip()} | {t['pending']} | {t['skipped']} |")
        else:
            extra = f"  skipped={t['skipped']}" if t["skipped"] else ""
            L.append(f"  {name:<14} {_bar(pct)} {pcts}  "
                     f"{t['done']:>7,}/{t['total']:<7,} pending={t['pending']:<6,}{extra}")
    L.append("")

    # ---- attention: only days with pending work ----
    L.append(f"{h2}Needs attention" if markdown else "NEEDS ATTENTION")
    if not markdown:
        L.append(rule)
    any_pending = False
    for name in STAGES:
        sp = rep.stages.get(name)
        if sp is None or not sp.available:
            continue
        bd = sp.by_day
        if bd.empty:
            continue
        bad = bd[bd["pending"] > 0]
        if bad.empty:
            continue
        any_pending = True
        L.append("")
        L.append(f"  {name}: {len(bad)} day(s) with pending work")
        shown = bad.tail(max_days)
        if len(bad) > max_days:
            L.append(f"    … {len(bad) - max_days} earlier day(s) omitted; see {name}_by_day.parquet")
        for _, r in shown.iterrows():
            L.append(f"    {r['day']}  {r['pct']:5.1f}%  "
                     f"done={int(r['done']):>6,} pending={int(r['pending']):>6,}")
    if not any_pending:
        L.append("")
        L.append("  Nothing pending. All stages complete for the selected dates.")
    L.append("")

    # ---- notes ----
    notes = [(n, note) for n in STAGES
             for note in (rep.stages[n].notes if n in rep.stages and rep.stages[n].available else [])]
    if notes:
        L.append(f"{h2}Notes" if markdown else "NOTES")
        if not markdown:
            L.append(rule)
        for name, note in notes:
            L.append(f"  [{name}] {note}")
        L.append("")

    # ---- commands ----
    L.append(f"{h2}Resubmit commands" if markdown else f"RESUBMIT COMMANDS (backend={rep.backend})")
    if not markdown:
        L.append(rule)
    cmds = rep.commands()
    if not cmds:
        L.append("  (nothing to resubmit)")
    else:
        if markdown:
            L.append("")
            L.append("```bash")
        detect_sp = rep.stages.get("detect")
        if detect_sp is not None and detect_sp.available:
            cleanup = cleanup_command(detect_sp)
            if cleanup:
                L.append("# remove 0-byte .bbb stubs first:")
                L.append(cleanup)
                L.append("")
        for name, cmd in cmds.items():
            L.append(f"# {name}")
            L.append(cmd)
            L.append("")
        if markdown:
            L.append("```")
    return "\n".join(L)
