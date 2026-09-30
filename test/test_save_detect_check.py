#!/usr/bin/env python3
"""
Hermetic tests for save_detect: the check that finds outputs the submitters wrongly
count as done (bb_hpc.src.save_detect_check), and the job function's handling of
read errors (job_for_save_detect_chunk). No bb_binary / bb_tracking needed.

The job-function tests pin the 2026 failure: iterate_bb_binary_repository is a
plain generator, so a read error kills it, and the old loop turned that into an
empty or cut-short parquet saved as a finished hour.

Run:  pytest test/test_save_detect_check.py
"""
import os
import sys
import types
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

# Importing test_progress installs the fake bb_binary (fill-missing only) and
# puts the bb_hpc package on sys.path; reuse its helpers so names match.
from bb_hpc.test.test_progress import _gvf, bbb_row  # noqa: E402

from bb_hpc.src import save_detect_check as C  # noqa: E402

UTC = timezone.utc


def dt(day, h, m=0, s=0):
    return datetime(2026, 7, day, h, m, s, tzinfo=UTC)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _det_frame(start, end, n=5):
    ts = pd.date_range(start, end, periods=n)
    return pd.DataFrame({
        "timestamp": ts, "cam_id": 0, "detection_type": 2, "x_pixels": 1.0, "y_pixels": 2.0,
        "orientation_pixels": 0.0, "localizer_saliency": 0.5, "bee_id": float("nan"),
        "bee_id_confidence": float("nan"),
    })


def _write_out(detect_dir, cam, hour, df=None, raw=None, mtime=None):
    path = detect_dir / (_gvf(cam, dt(1, hour), dt(1, hour + 1)) + ".parquet")
    if raw is not None:
        path.write_bytes(raw)
    else:
        df.to_parquet(path)
    if mtime is not None:
        ts = mtime.timestamp()
        os.utime(path, (ts, ts))
    return str(path)


def _seed(tmp_path):
    """One unit per status. .bbb in every window cover minutes 0-30, modified 07-01 12:00."""
    resultdir = tmp_path / "results"
    cache = resultdir / "bbb_fileinfo"
    detect_dir = resultdir / "data_alldetections"
    cache.mkdir(parents=True)
    detect_dir.mkdir()

    rows = [bbb_row(cam, dt(1, h), dt(1, h, 30), dt(1, 12)) for cam in (0, 1) for h in (9, 10, 11, 12)]
    pd.DataFrame(rows).to_parquet(cache / "bbb_info_20260701.parquet", index=False)

    empty_with_schema = _det_frame(dt(1, 10), dt(1, 10, 1)).iloc[0:0]
    paths = {
        "ok":         _write_out(detect_dir, 0, 9, _det_frame(dt(1, 9), dt(1, 9, 29, 59))),
        "empty_stub": _write_out(detect_dir, 1, 9, pd.DataFrame()),
        "empty_hour": _write_out(detect_dir, 0, 10, empty_with_schema),
        "truncated":  _write_out(detect_dir, 1, 10, _det_frame(dt(1, 10), dt(1, 10, 5))),
        "unreadable": _write_out(detect_dir, 0, 11, raw=b"not a parquet file"),
        # cam 1 hour 11: no output -> missing
        "stale":      _write_out(detect_dir, 0, 12, _det_frame(dt(1, 12), dt(1, 12, 29)), mtime=dt(1, 11)),
        "ok_12":      _write_out(detect_dir, 1, 12, _det_frame(dt(1, 12), dt(1, 12, 29, 30))),
    }
    return str(resultdir), paths


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #
def test_bbb_coverage_end_clips_to_window_and_spans_the_boundary():
    df = pd.DataFrame([bbb_row(0, dt(1, 9, 50), dt(1, 10, 10), dt(1, 12))])
    df["starttime"] = pd.to_datetime(df["starttime"], utc=True)
    df["endtime"] = pd.to_datetime(df["endtime"], utc=True)
    df["cam_id"] = 0
    cov = C.bbb_coverage_end(df)
    assert cov == {
        (0, pd.Timestamp(dt(1, 9))): pd.Timestamp(dt(1, 10)),
        (0, pd.Timestamp(dt(1, 10))): pd.Timestamp(dt(1, 10, 10)),
    }


def test_every_status_is_classified(tmp_path):
    resultdir, paths = _seed(tmp_path)
    units = C.check_save_detect(resultdir, ["20260701"], tail_tol_min=3, verify_bbb=False)
    got = {(int(r.cam_id), r.from_dt.hour): r.status for r in units.itertuples()}
    assert got == {
        (0, 9): "ok", (1, 9): "empty_stub",
        (0, 10): "empty_hour", (1, 10): "truncated",
        (0, 11): "unreadable", (1, 11): "missing",
        (0, 12): "stale", (1, 12): "ok",
    }
    trunc = units[units["status"] == "truncated"].iloc[0]
    assert trunc["tail_gap_min"] == 25.0


def test_truncation_tolerance(tmp_path):
    resultdir, _ = _seed(tmp_path)
    units = C.check_save_detect(resultdir, ["20260701"], tail_tol_min=30)
    assert "truncated" not in set(units["status"])


def test_fix_commands_move_flagged_then_resubmit(tmp_path):
    resultdir, paths = _seed(tmp_path)
    units = C.check_save_detect(resultdir, ["20260701"], verify_bbb=False)
    out_dir = str(tmp_path / "out")
    cmds = C.fix_commands(units, resultdir, out_dir, backend="k8s", today="20260929")
    text = "\n".join(cmds)
    assert f"{resultdir}/data_alldetections_quarantine_20260929" in text
    assert "mv -t" in text
    assert "python -m bb_hpc.get_fileinfo --what outputs" in text
    assert "python -m bb_hpc.running_k8s.save_detect_submit --dates 20260701" in text
    # the move must come before the catalog rebuild, which must come before the submit
    assert text.index("mv -t") < text.index("get_fileinfo") < text.index("save_detect_submit")

    C.write_outputs(units, out_dir, cmds)
    flagged = open(os.path.join(out_dir, "flagged_files.txt")).read().split()
    assert sorted(flagged) == sorted([paths["empty_stub"], paths["truncated"], paths["unreadable"]])
    assert open(os.path.join(out_dir, "redo_dates.txt")).read().split() == ["20260701"]


# --------------------------------------------------------------------------- #
# Early-ending outputs are verified against the .bbb frames after the last detection
# --------------------------------------------------------------------------- #
def _frame(ts, n_dets):
    return types.SimpleNamespace(timestamp=ts.timestamp(), detectionsDP=[], detectionsBees=[object()] * n_dets)


def _loader_for(frames_by_path, calls=None):
    def load(path):
        if calls is not None:
            calls.append(path)
        return types.SimpleNamespace(frames=frames_by_path.get(path, []))
    return load


def _truncated_bbb_path():
    # _seed: cam 1, hour 10 has one .bbb 10:00-10:30 and a parquet ending at 10:05
    return f"/repo/{_gvf(1, dt(1, 10), dt(1, 10, 30))}.bbb"


def test_early_end_with_empty_bbb_frames_after_is_complete(tmp_path):
    """Berlin 2026-05 / 07-29: detections stop but the camera keeps recording empty
    (dark / covered / beeless) frames. A parquet has no row for such frames."""
    resultdir, paths = _seed(tmp_path)
    frames = [_frame(dt(1, 10, m), 0) for m in range(6, 30)]
    units = C.check_save_detect(resultdir, ["20260701"], loader=_loader_for({_truncated_bbb_path(): frames}))
    row = units[(units["cam_id"] == 1) & (units["from_dt"].dt.hour == 10)].iloc[0]
    assert row["status"] == "ok_empty_tail"
    assert row["bbb_dets_after_last"] == 0
    assert paths["truncated"] not in set(units.loc[units["status"].isin(C.FLAG_STATUSES), "path"])


def test_early_end_with_detections_after_is_truncated(tmp_path):
    resultdir, _ = _seed(tmp_path)
    frames = [_frame(dt(1, 10, 6), 0), _frame(dt(1, 10, 7), 42)]
    units = C.check_save_detect(resultdir, ["20260701"], loader=_loader_for({_truncated_bbb_path(): frames}))
    row = units[(units["cam_id"] == 1) & (units["from_dt"].dt.hour == 10)].iloc[0]
    assert row["status"] == "truncated"
    assert row["bbb_dets_after_last"] == 42


def test_the_last_detections_own_frame_is_not_counted_as_after(tmp_path):
    """The parquet stores the frame time rounded to microseconds, so the float frame
    time of that same frame can compare slightly later."""
    resultdir, _ = _seed(tmp_path)
    same_frame = types.SimpleNamespace(timestamp=dt(1, 10, 5).timestamp() + 4e-7,
                                       detectionsDP=[], detectionsBees=[object()] * 564)
    units = C.check_save_detect(resultdir, ["20260701"], loader=_loader_for({_truncated_bbb_path(): [same_frame]}))
    row = units[(units["cam_id"] == 1) & (units["from_dt"].dt.hour == 10)].iloc[0]
    assert row["status"] == "ok_empty_tail"


def test_a_file_listed_twice_in_the_catalog_is_read_once(tmp_path):
    resultdir, _ = _seed(tmp_path)
    cat_path = os.path.join(resultdir, "bbb_fileinfo", "bbb_info_20260701.parquet")
    cat = pd.read_parquet(cat_path)
    pd.concat([cat, cat[cat["full_path"] == _truncated_bbb_path()]]).to_parquet(cat_path, index=False)
    calls = []
    C.check_save_detect(resultdir, ["20260701"], loader=_loader_for({}, calls))
    assert calls == [_truncated_bbb_path()]


def test_unreadable_bbb_keeps_the_output_flagged(tmp_path):
    resultdir, _ = _seed(tmp_path)

    def broken(path):
        raise OSError("simulated read error")

    units = C.check_save_detect(resultdir, ["20260701"], loader=broken)
    row = units[(units["cam_id"] == 1) & (units["from_dt"].dt.hour == 10)].iloc[0]
    assert row["status"] == "truncated"


def test_all_ok_means_no_commands(tmp_path):
    resultdir = tmp_path / "results"
    (resultdir / "bbb_fileinfo").mkdir(parents=True)
    detect_dir = resultdir / "data_alldetections"
    detect_dir.mkdir()
    pd.DataFrame([bbb_row(0, dt(1, 9), dt(1, 9, 30), dt(1, 12))]).to_parquet(
        resultdir / "bbb_fileinfo" / "bbb_info_20260701.parquet", index=False)
    _write_out(detect_dir, 0, 9, _det_frame(dt(1, 9), dt(1, 9, 29, 59)))
    units = C.check_save_detect(str(resultdir), ["20260701"])
    assert set(units["status"]) == {"ok"}
    assert C.fix_commands(units, str(resultdir), str(tmp_path / "out")) == []


def test_default_dates_come_from_the_bbb_catalog_not_the_video_catalog(tmp_path):
    """Old videos get cleared from disk, so the video catalog starts late in the season.
    Taking dates from it (progress.catalog_dates) hid every early-season stub."""
    from bb_hpc.src.progress import catalog_dates

    resultdir, _ = _seed(tmp_path)
    cache = os.path.join(resultdir, "bbb_fileinfo")
    late = datetime(2026, 7, 20, 9, tzinfo=UTC)
    pd.DataFrame({"starttime": [late], "endtime": [late + timedelta(minutes=1)]}).to_parquet(
        os.path.join(cache, "video_info_all.parquet"), index=False)
    assert catalog_dates(resultdir) == ["20260720"]
    assert C.bbb_dates(resultdir) == ["20260701"]


def test_moved_files_count_as_missing_immediately(tmp_path):
    """The check stats the directory, not the daily outinfo cache, so a file moved
    aside today is already 'missing' (what the submitter will see after get_fileinfo)."""
    resultdir, paths = _seed(tmp_path)
    os.remove(paths["empty_stub"])
    units = C.check_save_detect(resultdir, ["20260701"])
    row = units[(units["cam_id"] == 1) & (units["from_dt"].dt.hour == 9)].iloc[0]
    assert row["status"] == "missing"


# --------------------------------------------------------------------------- #
# job_for_save_detect_chunk: read errors are fatal, nothing half-written
# --------------------------------------------------------------------------- #
class _DetType:
    value = 2   # untagged


class _Det:
    detection_type = _DetType()
    x_pixels = 1.0
    y_pixels = 2.0
    orientation_pixels = 0.0
    localizer_saliency = 0.5

    def __init__(self, ts):
        self.timestamp = ts


def _fake_iterator(n_frames, raise_after=None, dets_per_frame=3):
    def iterate_bb_binary_repository(repository_path, dt_begin, dt_end, **kwargs):
        for i in range(n_frames):
            if raise_after is not None and i == raise_after:
                raise ValueError("simulated bb_binary read error")
            ts = dt_begin + timedelta(seconds=i)
            yield (kwargs.get("cam_id"), i, ts, [_Det(ts) for _ in range(dets_per_frame)], None)
    return iterate_bb_binary_repository


@pytest.fixture
def run_chunk(tmp_path, monkeypatch):
    """Returns run(iterator_fn, windows) -> (exit_code, save_dir)."""
    repo = tmp_path / "repo"
    (repo / "2026").mkdir(parents=True)
    save_dir = tmp_path / "out"
    save_dir.mkdir()

    bb_utils = types.ModuleType("bb_utils")
    ids = types.ModuleType("bb_utils.ids")
    ids.BeesbookID = object
    bb_utils.ids = ids
    monkeypatch.setitem(sys.modules, "bb_utils", bb_utils)
    monkeypatch.setitem(sys.modules, "bb_utils.ids", ids)

    def run(iterator_fn, windows, repo_path=str(repo)):
        bb_tracking = types.ModuleType("bb_tracking")
        dw = types.ModuleType("bb_tracking.data_walker")
        dw.iterate_bb_binary_repository = iterator_fn
        bb_tracking.data_walker = dw
        monkeypatch.setitem(sys.modules, "bb_tracking", bb_tracking)
        monkeypatch.setitem(sys.modules, "bb_tracking.data_walker", dw)

        from bb_hpc.src.jobfunctions import job_for_save_detect_chunk
        jobs = [dict(repo_path=repo_path, save_path=str(save_dir), from_dt=dt(1, h), to_dt=dt(1, h + 1), cam_id=0)
                for h in windows]
        try:
            job_for_save_detect_chunk(jobs)
            return 0, save_dir
        except SystemExit as e:
            return e.code, save_dir
    return run


def _out(save_dir, hour):
    return save_dir / (_gvf(0, dt(1, hour), dt(1, hour + 1)) + ".parquet")


def test_clean_read_writes_all_detections(run_chunk):
    code, save_dir = run_chunk(_fake_iterator(10), [9])
    assert code == 0
    df = pd.read_parquet(_out(save_dir, 9))
    assert len(df) == 30
    assert not list(save_dir.glob("*.tmp"))


def test_read_error_writes_nothing_and_exits_nonzero(run_chunk):
    """The old loop caught this, the dead generator then StopIteration'd, and 3
    frames were saved as the whole hour."""
    code, save_dir = run_chunk(_fake_iterator(10, raise_after=3), [9])
    assert code == 1
    assert not _out(save_dir, 9).exists()
    assert not list(save_dir.iterdir())


def test_read_error_at_the_first_frame_writes_no_stub(run_chunk):
    """The 2026 Konstanz case: the repo walk failed before the first frame and a
    column-less 1 KB parquet was saved for every hour."""
    code, save_dir = run_chunk(_fake_iterator(10, raise_after=0), [9])
    assert code == 1
    assert not list(save_dir.iterdir())


def test_frames_without_detections_write_an_empty_file_with_schema(run_chunk):
    code, save_dir = run_chunk(_fake_iterator(10, dets_per_frame=0), [9])
    assert code == 0
    df = pd.read_parquet(_out(save_dir, 9))
    assert len(df) == 0
    assert "timestamp" in df.columns and "x_pixels" in df.columns


def test_missing_repo_root_is_an_error_not_an_empty_hour(run_chunk, tmp_path):
    code, save_dir = run_chunk(_fake_iterator(10), [9], repo_path=str(tmp_path / "not_mounted"))
    assert code == 1
    assert not list(save_dir.iterdir())


def test_one_bad_window_does_not_stop_the_rest_of_the_chunk(run_chunk):
    calls = {"n": 0}
    good = _fake_iterator(10)
    bad = _fake_iterator(10, raise_after=2)

    def iterator_fn(repository_path, dt_begin, dt_end, **kwargs):
        calls["n"] += 1
        fn = bad if dt_begin.hour == 9 else good
        return fn(repository_path, dt_begin, dt_end, **kwargs)

    code, save_dir = run_chunk(iterator_fn, [9, 10])
    assert calls["n"] == 2
    assert code == 1
    assert not _out(save_dir, 9).exists()
    assert len(pd.read_parquet(_out(save_dir, 10))) == 30
