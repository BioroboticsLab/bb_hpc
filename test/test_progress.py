#!/usr/bin/env python3
"""
Hermetic tests for the progress report (no bb_binary / cell-seg deps needed).

The load-bearing test here is ANTI-DRIFT: for the same inputs, the set of units
bb_hpc.src.progress reports as pending must equal the set of work units the
corresponding generate_jobs_* generator actually yields. That invariant is what
lets you trust the report -- if generate.py's completion rules change, these
tests fail rather than the report silently lying.

Also covers the specific defects the old bbb_and_tracking_progress.ipynb had:
  - a 0-byte .bbb counted as processed
  - an hour-boundary-spanning .bbb attributed to only one hour
  - background/frame_extract "skipped" units miscounted as done
  - a wrong config_tag guess making finished background work look pending forever

Run:  pytest test/test_progress.py
      python test/test_progress.py
"""
import os
import sys
import types
from datetime import datetime, timezone

import pytest

# --------------------------------------------------------------------------- #
# Install fake bb_binary BEFORE importing any bb_hpc module.
# --------------------------------------------------------------------------- #
_TS = "%Y%m%dT%H%M%S"
UTC = timezone.utc


def _parse_video_fname(basename, format=None):
    """cam-<id>__<startTS>__<endTS>.<ext> -> (cam_id:int, start:dt, end:dt), UTC."""
    stem = basename
    for ext in (".mp4", ".bbb", ".parquet", ".dill", ".h264"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    cam_s, s, e = stem.split("__")
    cam = int(cam_s.replace("cam-", "").replace("Cam_", ""))
    return (cam,
            datetime.strptime(s, _TS).replace(tzinfo=UTC),
            datetime.strptime(e, _TS).replace(tzinfo=UTC))


def _get_video_fname(cam, start, end):
    return f"Cam_{cam}__{start.strftime(_TS)}__{end.strftime(_TS)}"


def _install_fakes():
    bb = types.ModuleType("bb_binary")
    parsing = types.ModuleType("bb_binary.parsing")
    parsing.parse_video_fname = _parse_video_fname
    parsing.get_video_fname = _get_video_fname
    common = types.ModuleType("bb_binary.common")
    common.bbb = object()
    repository = types.ModuleType("bb_binary.repository")
    repository.Repository = type("Repository", (), {})
    bb.parsing = parsing
    bb.common = common
    bb.repository = repository
    bb.parse_video_fname = _parse_video_fname
    bb.get_video_fname = _get_video_fname
    bb.Repository = repository.Repository
    sys.modules["bb_binary"] = bb
    sys.modules["bb_binary.parsing"] = parsing
    sys.modules["bb_binary.common"] = common
    sys.modules["bb_binary.repository"] = repository


_install_fakes()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + "/..")

import pandas as pd  # noqa: E402

from bb_hpc.src import generate as G  # noqa: E402
from bb_hpc.src import progress as P  # noqa: E402
from bb_hpc.src.fileinfo import list_video_files_incremental  # noqa: E402


def dt(day, h, m=0):
    return datetime(2026, 7, day, h, m, tzinfo=UTC)


def bbb_row(cam, start, end, mtime, size=4096):
    return {
        "file_name": _get_video_fname(cam, start, end) + ".bbb",
        "full_path": f"/repo/{_get_video_fname(cam, start, end)}.bbb",
        "starttime": start, "endtime": end,
        "modified_time": mtime, "file_size": size,
    }


def outinfo(rows):
    return pd.DataFrame(rows, columns=["cam_id", "from_dt", "to_dt", "modified_time"])


# --------------------------------------------------------------------------- #
# window_candidates: the shared save_detect / tracking predicate
# --------------------------------------------------------------------------- #
def test_hour_boundary_bbb_makes_both_hours_expected():
    """A .bbb spanning 09:50-10:10 belongs to BOTH hour 9 and hour 10.

    The old notebook floored starttime to the hour, so hour 10 vanished from the
    denominator and genuine pending work there read as complete.
    """
    df = pd.DataFrame([bbb_row(0, dt(1, 9, 50), dt(1, 10, 10), dt(1, 12))])
    units = G.window_candidates(df, {}, ["20260701"])
    assert sorted((u["cam_id"], u["from_dt"].hour) for u in units) == [(0, 9), (0, 10)]


def test_window_status_missing_stale_done():
    df = pd.DataFrame([
        bbb_row(0, dt(1, 9), dt(1, 9, 30), dt(1, 12)),
        bbb_row(1, dt(1, 9), dt(1, 9, 30), dt(1, 12)),
    ])
    out_index = {
        (0, dt(1, 9), dt(1, 10)): dt(1, 13),   # newer than source -> done
        (1, dt(1, 9), dt(1, 10)): dt(1, 11),   # older than source -> stale
    }
    got = {(u["cam_id"]): u["status"] for u in G.window_candidates(df, out_index, ["20260701"])}
    assert got == {0: "done", 1: "stale"}

    got = {(u["cam_id"]): u["status"] for u in G.window_candidates(df, {}, ["20260701"])}
    assert got == {0: "missing", 1: "missing"}


def test_ensure_cam_id_derives_int_without_catalog_column():
    """The bbb catalog has no cam_id column and its daily cache is never schema-invalidated,
    so cam_id must always be derived at read time."""
    df = pd.DataFrame([bbb_row(3, dt(1, 9), dt(1, 9, 30), dt(1, 12))])
    assert "cam_id" not in df.columns
    out = G.ensure_cam_id(df)
    assert out["cam_id"].tolist() == [3]
    assert out["cam_id"].dtype.kind == "i"


def test_hour_windows_covers_the_day():
    w = G.hour_windows(["20260701"])
    assert len(w) == 24
    assert w[0][0] == dt(1, 0) and w[-1][1] == datetime(2026, 7, 2, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# ANTI-DRIFT: progress pending set == what the generators actually schedule
# --------------------------------------------------------------------------- #
def _seed_window_catalogs(tmp_path):
    resultdir = tmp_path / "results"
    cache = resultdir / "bbb_fileinfo"
    cache.mkdir(parents=True)

    rows = []
    for cam in (0, 1):
        for hour in (9, 10):
            rows.append(bbb_row(cam, dt(1, hour), dt(1, hour, 30), dt(1, 12)))
    pd.DataFrame(rows).to_parquet(cache / "bbb_info_20260701.parquet", index=False)

    # cam 0 hour 9 -> done; cam 1 hour 9 -> stale; hour 10 -> missing (absent)
    out = outinfo([
        (0, dt(1, 9), dt(1, 10), dt(1, 13)),
        (1, dt(1, 9), dt(1, 10), dt(1, 11)),
    ])
    out.to_parquet(cache / "save_detect_outinfo.parquet", index=False)
    out.to_parquet(cache / "save_tracking_outinfo.parquet", index=False)
    return str(resultdir)


def test_antidrift_save_detect(tmp_path):
    resultdir = _seed_window_catalogs(tmp_path)

    reported = P.save_detect_progress(resultdir, ["20260701"]).pending_units()
    reported_keys = {(int(r.cam_id), r.from_dt.to_pydatetime()) for r in reported.itertuples()}

    scheduled = [c for ch in G.generate_jobs_save_detect(resultdir, "/repo", ["20260701"], chunk_size=99)
                 for c in ch["job_args_list"]]
    scheduled_keys = {(c["cam_id"], c["from_dt"]) for c in scheduled}

    assert reported_keys == scheduled_keys
    assert len(reported_keys) == 3  # cam1@9 stale + cam0@10 + cam1@10 missing


def test_antidrift_tracking(tmp_path):
    resultdir = _seed_window_catalogs(tmp_path)

    reported = P.tracking_progress(resultdir, ["20260701"]).pending_units()
    reported_keys = {(int(r.cam_id), r.from_dt.to_pydatetime()) for r in reported.itertuples()}

    scheduled = [c for ch in G.generate_jobs_tracking(resultdir, "/repo", "/tmp", ["20260701"], chunk_size=99)
                 for c in ch["job_args_list"]]
    scheduled_keys = {(c["cam_id"], c["from_dt"]) for c in scheduled}

    assert reported_keys == scheduled_keys


def _seed_frames(tmp_path):
    videodir = tmp_path / "videos"
    frames = tmp_path / "frames"
    # cam-0 has videos + .txt sidecars and extracted frames -> done
    # cam-1 has videos + .txt but no frames                 -> pending
    # cam-2 has an .mp4 but NO .txt                         -> skipped_no_txt
    for cam, with_txt in ((0, True), (1, True), (2, False)):
        d = videodir / "20260701" / f"cam-{cam}"
        d.mkdir(parents=True)
        stem = f"cam-{cam}__20260701T090000__20260701T090500"
        (d / f"{stem}.mp4").write_text("v")
        if with_txt:
            (d / f"{stem}.txt").write_text("ts")

    fd = frames / "20260701" / "cam-0"
    fd.mkdir(parents=True)
    (fd / "frame_20260701T090000.png").write_text("f")
    return str(videodir), str(frames)


def test_antidrift_frame_extract(tmp_path):
    videodir, frames = _seed_frames(tmp_path)

    sp = P.frame_extract_progress(videodir, frames, ["20260701"])
    reported = {(r.day, r.cam) for r in sp.pending_units().itertuples()}

    scheduled = [u for ch in G.generate_jobs_frame_extract(videodir, frames, ["20260701"], chunk_size=99)
                 for u in ch["work_units"]]
    scheduled_keys = {(u["date"], u["cam"]) for u in scheduled}

    assert reported == scheduled_keys == {("20260701", "cam-1")}

    status = dict(zip(sp.units["cam"], sp.units["status"]))
    assert status == {"cam-0": "done", "cam-1": "pending", "cam-2": "skipped_no_txt"}


def test_frame_extract_skipped_is_not_counted_as_done(tmp_path):
    """done = glob_total - pending would call cam-2 'done'; it can never run."""
    videodir, frames = _seed_frames(tmp_path)
    sp = P.frame_extract_progress(videodir, frames, ["20260701"])
    t = sp.totals()
    assert t == {"total": 3, "done": 1, "pending": 1, "skipped": 1, "pct": 50.0}


def _seed_background(tmp_path, engine_tag="count_w10_n200"):
    frames = tmp_path / "frames"
    bg = tmp_path / "bg"
    for cam, n in (("cam-0", 5), ("cam-1", 5), ("cam-2", 0)):
        d = frames / "20260701" / cam
        d.mkdir(parents=True)
        for i in range(n):
            (d / f"frame_20260701T09{i:02d}00.png").write_text("f")
    # only cam-0 has a produced background, under the tag the ENGINE chose
    done = bg / "20260701" / "cam-0" / engine_tag
    done.mkdir(parents=True)
    (done / "background_20260701T090000.000000.000Z.png").write_text("bg")
    return str(frames), str(bg)


def test_antidrift_background(tmp_path):
    frames, bg = _seed_background(tmp_path)
    s = {"frame_interval_sec": None, "background_window": None,
         "window_size": 10, "num_median_images": 200, "min_frames": 3}

    sp = P.background_progress(frames, bg, ["20260701"], s)
    reported = {(r.day, r.cam) for r in sp.pending_units().itertuples()}

    scheduled = [u for ch in G.generate_jobs_background(
        frames, bg, ["20260701"], window_size=10, num_median_images=200, chunk_size=99)
        for u in ch["work_units"]]
    scheduled_keys = {(u["date"], u["cam"]) for u in scheduled}

    assert reported == scheduled_keys == {("20260701", "cam-1")}

    status = dict(zip(sp.units["cam"], sp.units["status"]))
    assert status == {"cam-0": "done", "cam-1": "pending", "cam-2": "skipped_no_frames"}


def test_background_zero_frame_cam_is_skipped_not_done(tmp_path):
    frames, bg = _seed_background(tmp_path)
    s = {"frame_interval_sec": None, "background_window": None,
         "window_size": 10, "num_median_images": 200, "min_frames": 3}
    t = P.background_progress(frames, bg, ["20260701"], s).totals()
    assert t == {"total": 3, "done": 1, "pending": 1, "skipped": 1, "pct": 50.0}


# --------------------------------------------------------------------------- #
# config_tag: the submit-host bug
# --------------------------------------------------------------------------- #
def test_config_tag_fallback_matches_engine_when_available():
    """When background_generator IS installed, our mirror must agree with it."""
    windowing = G._import_windowing()
    if windowing is None:
        pytest.skip("background_generator not installed")
    for interval, window, wsize, nmed in [
        (None, None, 10, 200), (60, None, 5, 100), (None, "hour", 10, 200), (30, "day", 4, 50),
    ]:
        assert G._config_tag_fallback(interval, window, wsize, nmed) == \
            windowing.config_tag(interval, window, wsize, nmed)


def test_resolve_background_tag_heals_a_wrong_guess(tmp_path):
    """Without background_generator, a wrong tag guess would make finished work
    look pending forever. Resolve against the single tag dir on disk instead."""
    if G._import_windowing() is not None:
        pytest.skip("background_generator installed; the guess path is not exercised")

    frames, bg = _seed_background(tmp_path, engine_tag="count_w10_n200")
    out_cam = os.path.join(bg, "20260701", "cam-0")

    guess = G._config_tag_fallback(None, None, 10, 200)
    assert guess != "count_w10_n200", "test needs a guess that differs from the engine tag"

    tag, note = G.resolve_background_tag(out_cam, None, None, 10, 200)
    assert tag == "count_w10_n200"
    assert note and "not importable" in note
    assert G.background_is_done(os.path.join(out_cam, tag), [], None, None)


def test_resolve_background_tag_keeps_guess_when_ambiguous(tmp_path):
    if G._import_windowing() is not None:
        pytest.skip("background_generator installed")
    frames, bg = _seed_background(tmp_path, engine_tag="count_w10_n200")
    out_cam = os.path.join(bg, "20260701", "cam-0")
    os.mkdir(os.path.join(out_cam, "another_tag"))

    guess = G._config_tag_fallback(None, None, 10, 200)
    tag, note = G.resolve_background_tag(out_cam, None, None, 10, 200)
    assert tag == guess
    assert note and "cannot" in note


def test_resolve_background_tag_no_output_yet(tmp_path):
    if G._import_windowing() is not None:
        pytest.skip("background_generator installed")
    tag, note = G.resolve_background_tag(str(tmp_path / "nope"), None, None, 10, 200)
    assert tag == G._config_tag_fallback(None, None, 10, 200)
    assert note is None


# --------------------------------------------------------------------------- #
# detect: 0-byte stubs
# --------------------------------------------------------------------------- #
def _seed_detect(tmp_path, stub_size=0):
    resultdir = tmp_path / "results"
    cache = resultdir / "bbb_fileinfo"
    cache.mkdir(parents=True)

    vids, bbbs = [], []
    for i, minute in enumerate((0, 5, 10)):
        st, en = dt(1, 9, minute), dt(1, 9, minute + 5)
        stem = f"cam-0__{st.strftime(_TS)}__{en.strftime(_TS)}"
        vids.append({"file_name": f"{stem}.mp4", "full_path": f"/v/{stem}.mp4",
                     "starttime": st, "endtime": en, "cam": 0})
        if i == 2:
            continue  # third video has no .bbb at all -> missing
        bbbs.append(bbb_row(0, st, en, dt(1, 12), size=(stub_size if i == 0 else 4096)))

    pd.DataFrame(vids).to_parquet(cache / "video_info_all.parquet", index=False)
    pd.DataFrame(bbbs).to_parquet(cache / "bbb_info_20260701.parquet", index=False)
    return str(resultdir)


def test_zero_byte_bbb_is_pending_not_done(tmp_path):
    """The catalog lists 0-byte stubs; is_bbb_file_valid_basicmatch rejects them.
    The old notebook's `basename in catalog` check called them processed."""
    resultdir = _seed_detect(tmp_path, stub_size=0)
    sp = P.detect_progress(resultdir, ["20260701"])
    assert sp.status_counts() == {"zero_byte_stub": 1, "done": 1, "missing": 1}
    assert sp.totals()["pending"] == 2
    assert any("0-byte" in n for n in sp.notes)


def test_nonzero_bbb_is_done(tmp_path):
    resultdir = _seed_detect(tmp_path, stub_size=4096)
    sp = P.detect_progress(resultdir, ["20260701"])
    assert sp.status_counts() == {"done": 2, "missing": 1}


def test_bbb_start_key_ignores_the_end_timestamp():
    """The invariant the whole gappy-recording fix rests on.

    A video and its .bbb must produce the SAME key even when the .bbb's end
    differs, because bb_binary names its output from the actual frame timestamps
    rather than from the video's filename.
    """
    st = dt(1, 9, 0)
    vid = pd.DataFrame([{"file_name": f"cam-0__{st.strftime(_TS)}__{dt(1, 9, 5).strftime(_TS)}.mp4",
                         "starttime": st}])
    # same start, end 3 minutes later than the video name implies
    bbb = pd.DataFrame([bbb_row(0, st, dt(1, 9, 8), dt(1, 12))])
    assert G.bbb_start_key_series(vid).iloc[0] == G.bbb_start_key_series(bbb).iloc[0]

    # ...but a different camera or a different start second must NOT collide
    other_cam = pd.DataFrame([bbb_row(1, st, dt(1, 9, 5), dt(1, 12))])
    other_start = pd.DataFrame([bbb_row(0, dt(1, 9, 1), dt(1, 9, 5), dt(1, 12))])
    assert G.bbb_start_key_series(vid).iloc[0] != G.bbb_start_key_series(other_cam).iloc[0]
    assert G.bbb_start_key_series(vid).iloc[0] != G.bbb_start_key_series(other_start).iloc[0]


def test_detect_matches_bbb_with_gappy_end_timestamp(tmp_path):
    """A .bbb whose end differs from the video filename's is still that video's.

    Real case: a gappy recording produced a .bbb ending 58s past what the video
    name implied. Matching the full start--end stem reported it missing forever.
    """
    resultdir = tmp_path / "results"
    cache = resultdir / "bbb_fileinfo"
    cache.mkdir(parents=True)

    st, en = dt(1, 9, 0), dt(1, 9, 5)
    stem = f"cam-0__{st.strftime(_TS)}__{en.strftime(_TS)}"
    pd.DataFrame([{"file_name": f"{stem}.mp4", "full_path": f"/v/{stem}.mp4",
                   "starttime": st, "endtime": en, "cam": 0}]) \
        .to_parquet(cache / "video_info_all.parquet", index=False)
    # detection succeeded, but the last frame landed 3 minutes past the name's end
    pd.DataFrame([bbb_row(0, st, dt(1, 9, 8), dt(1, 12), size=4096)]) \
        .to_parquet(cache / "bbb_info_20260701.parquet", index=False)

    sp = P.detect_progress(str(resultdir), ["20260701"])
    assert sp.status_counts() == {"done": 1}
    # and the report shows the ACTUAL .bbb name, not the predicted one
    assert sp.units["expected_bbb"].iloc[0] == bbb_row(0, st, dt(1, 9, 8), dt(1, 12))["file_name"]


def test_detect_command_adds_check_read_bbb_only_when_stubs_exist(tmp_path):
    """Plain --use-fileinfo skips a stub by name, so it would never redo one."""
    sp = P.detect_progress(_seed_detect(tmp_path, stub_size=0), ["20260701"])
    assert "--check-read-bbb" in P.build_command("detect", sp, "k8s")
    assert P.cleanup_command(sp) == \
        "python -m bb_hpc.scan_and_remove_invalid_bbb_files --dates 20260701"

    sp2 = P.detect_progress(_seed_detect(tmp_path / "b", stub_size=4096), ["20260701"])
    cmd = P.build_command("detect", sp2, "k8s")
    assert "--use-fileinfo" in cmd and "--check-read-bbb" not in cmd
    assert P.cleanup_command(sp2) is None


def test_commands_target_the_requested_backend(tmp_path):
    sp = P.detect_progress(_seed_detect(tmp_path, stub_size=4096), ["20260701"])
    for backend in ("k8s", "slurm", "docker"):
        assert f"bb_hpc.running_{backend}.detect_submit" in P.build_command("detect", sp, backend)
    with pytest.raises(ValueError):
        P.build_command("detect", sp, "nonsense")


def test_no_pending_means_no_command(tmp_path):
    resultdir = tmp_path / "results"
    (resultdir / "bbb_fileinfo").mkdir(parents=True)
    st, en = dt(1, 9), dt(1, 9, 5)
    stem = f"cam-0__{st.strftime(_TS)}__{en.strftime(_TS)}"
    pd.DataFrame([{"file_name": f"{stem}.mp4", "full_path": "/v", "starttime": st,
                   "endtime": en, "cam": 0}]).to_parquet(
        resultdir / "bbb_fileinfo" / "video_info_all.parquet", index=False)
    pd.DataFrame([bbb_row(0, st, en, dt(1, 12))]).to_parquet(
        resultdir / "bbb_fileinfo" / "bbb_info_20260701.parquet", index=False)

    sp = P.detect_progress(str(resultdir), ["20260701"])
    assert sp.totals()["pct"] == 100.0
    assert P.build_command("detect", sp, "k8s") is None


# --------------------------------------------------------------------------- #
# Missing catalogs degrade gracefully
# --------------------------------------------------------------------------- #
def test_missing_catalog_marks_stage_unavailable(tmp_path):
    resultdir = tmp_path / "results"
    (resultdir / "bbb_fileinfo").mkdir(parents=True)
    sp = P.detect_progress(str(resultdir), ["20260701"])
    assert not sp.available
    assert "video_info_all.parquet" in sp.reason
    assert sp.totals()["total"] == 0


def test_build_outinfo_declares_schema_when_empty(tmp_path):
    """pd.DataFrame([]) has zero columns; a column-less parquet KeyErrors every reader.
    That is the normal state before a stage's first output exists."""
    from bb_hpc.src.fileinfo import OUTINFO_COLUMNS, build_outinfo

    out_dir = tmp_path / "data_alldetections"
    out_dir.mkdir()
    cache = tmp_path / "cache"
    cache.mkdir()
    path = build_outinfo(str(out_dir), str(cache), "parquet", "save_detect_outinfo")
    df = pd.read_parquet(path)
    assert list(df.columns) == OUTINFO_COLUMNS and df.empty


def test_window_stage_survives_a_columnless_outinfo(tmp_path):
    """Older get_fileinfo runs already wrote column-less parquets; still readable."""
    resultdir = tmp_path / "results"
    cache = resultdir / "bbb_fileinfo"
    cache.mkdir(parents=True)

    pd.DataFrame([bbb_row(0, dt(1, 9), dt(1, 9, 30), dt(1, 12))]).to_parquet(
        cache / "bbb_info_20260701.parquet", index=False)
    pd.DataFrame([]).to_parquet(cache / "save_detect_outinfo.parquet", index=False)  # zero columns

    sp = P.save_detect_progress(str(resultdir), ["20260701"])
    assert sp.available
    assert sp.status_counts() == {"missing": 1}   # nothing produced yet -> all pending


def test_empty_bbb_catalog_yields_no_units(tmp_path):
    resultdir = tmp_path / "results"
    cache = resultdir / "bbb_fileinfo"
    cache.mkdir(parents=True)
    pd.DataFrame([]).to_parquet(cache / "bbb_info_20260701.parquet", index=False)
    pd.DataFrame([]).to_parquet(cache / "save_tracking_outinfo.parquet", index=False)

    sp = P.tracking_progress(str(resultdir), ["20260701"])
    assert sp.available and sp.totals()["total"] == 0
    assert P.build_command("tracking", sp, "k8s") is None


# --------------------------------------------------------------------------- #
# Incremental video catalog
# --------------------------------------------------------------------------- #
def _mkvid(videodir, day, cam, minute):
    d = videodir / day / f"cam-{cam}"
    d.mkdir(parents=True, exist_ok=True)
    stem = f"cam-{cam}__{day}T09{minute:02d}00__{day}T09{minute + 5:02d}00"
    (d / f"{stem}.mp4").write_text("v")


def _rescanned(videodir, cache, capsys, **kw):
    df = list_video_files_incremental(str(videodir), str(cache), **kw)
    lines = capsys.readouterr().out.splitlines()
    days = [lines[i - 1] for i, l in enumerate(lines) if l == "...reindexing video day"]
    return df, days


def test_video_catalog_reuses_cache_and_detects_new_files(tmp_path, capsys):
    videodir = tmp_path / "videos"
    cache = tmp_path / "cache"
    cache.mkdir()
    for day in ("20260701", "20260702"):
        for minute in (0, 5):
            _mkvid(videodir, day, 0, minute)

    df, days = _rescanned(videodir, cache, capsys, force_recent_days=0)
    assert len(df) == 4 and days == ["20260701", "20260702"]

    df, days = _rescanned(videodir, cache, capsys, force_recent_days=0)
    assert len(df) == 4 and days == []          # cache reused

    # A new video lands deep inside a cam dir; the DAY dir's own mtime does not
    # change, so this only works because we compare against the newest FILE mtime.
    os.utime(videodir / "20260701" / "cam-0", None)
    _mkvid(videodir, "20260701", 0, 10)
    os.utime(videodir / "20260701" / "cam-0" / "cam-0__20260701T091000__20260701T091500.mp4",
             (2_000_000_000, 2_000_000_000))

    df, days = _rescanned(videodir, cache, capsys, force_recent_days=0)
    assert days == ["20260701"], days
    assert len(df) == 5

    df, days = _rescanned(videodir, cache, capsys, force_recent_days=2)
    assert days == ["20260701", "20260702"]     # forced regardless of mtime

    assert set(df.columns) == {"file_name", "full_path", "starttime", "endtime", "cam"}


def test_video_catalog_empty_root(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    df = list_video_files_incremental(str(tmp_path / "nothing"), str(cache))
    assert df.empty and list(df.columns) == ["file_name", "full_path", "starttime", "endtime", "cam"]


# --------------------------------------------------------------------------- #
# by_day rollup semantics
# --------------------------------------------------------------------------- #
def test_by_day_excludes_skipped_from_the_percentage():
    units = pd.DataFrame([
        {"day": "20260701", "status": "done"},
        {"day": "20260701", "status": "pending"},
        {"day": "20260701", "status": "skipped_no_frames"},
    ])
    bd = P.StageProgress("background", units).by_day
    row = bd.iloc[0]
    assert (row["total"], row["done"], row["pending"], row["skipped"]) == (3, 1, 1, 1)
    assert row["pct"] == 50.0   # 1 done / (1 done + 1 pending), skipped excluded


def test_all_skipped_is_100_pct_not_nan_in_both_rollups():
    """Nothing schedulable is left, so it is complete -- and by_day must agree with totals()."""
    units = pd.DataFrame([{"day": "20260701", "status": "skipped_no_frames"}])
    sp = P.StageProgress("background", units)
    assert sp.by_day.iloc[0]["pct"] == 100.0
    assert sp.totals()["pct"] == 100.0


def test_stage_with_no_units_has_undefined_pct():
    sp = P.StageProgress("background", pd.DataFrame(columns=["day", "status"]))
    assert sp.totals()["pct"] is None
    assert sp.by_day.empty


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
