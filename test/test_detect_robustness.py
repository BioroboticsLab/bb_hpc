#!/usr/bin/env python3
"""
Hermetic tests for the detect-robustness hardening (no bb_binary/pipeline needed).

Covers:
- job_for_process_videos retry + clean-slate + post-write validation + runtime
  skip + durable failure list + sys.exit(1)  (src/jobfunctions.py)
- is_bbb_file_valid_basicmatch rejects 0-byte even with check_read_file=False
  (src/fileinfo.py)
- check_symlinks_and_fill_in._source_is_valid 0-byte guard
  (check_symlinks_and_fill_in.py)

The real deps (bb_binary, pipeline) are replaced with fakes that reproduce the
relevant behavior: a process_video that mimics bb_binary writing a 0-byte primary
stub first and (optionally) failing before writing content — exactly the CIFS-EIO
failure mode this hardening targets.

Run directly:  python test/test_detect_robustness.py
Or via pytest:  pytest test/test_detect_robustness.py
"""
import os
import sys
import types
import errno
import tempfile
import shutil
import importlib.util
from datetime import datetime
from pathlib import Path

# --------------------------------------------------------------------------- #
# Install fake bb_binary / pipeline BEFORE importing any bb_hpc module.
# --------------------------------------------------------------------------- #
_TS = "%Y%m%dT%H%M%S"


def _parse_video_fname(basename, format=None):
    """Fake of bb_binary.parsing.parse_video_fname for our test naming:
    cam-<id>__<startTS>__<endTS>.<ext>  ->  (cam_id:int, start:dt, end:dt)."""
    stem = basename
    for ext in (".mp4", ".bbb", ".avi"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    cam_s, s, e = stem.split("__")
    cam = int(cam_s.replace("cam-", "").replace("Cam_", ""))
    return cam, datetime.strptime(s, _TS), datetime.strptime(e, _TS)


def _get_video_fname(cam, start, end):
    """Fake of bb_binary.parsing.get_video_fname (deterministic stem)."""
    return f"Cam_{cam}__{start.strftime(_TS)}__{end.strftime(_TS)}"


def _install_fakes():
    bb = types.ModuleType("bb_binary")
    parsing = types.ModuleType("bb_binary.parsing")
    parsing.parse_video_fname = _parse_video_fname
    parsing.get_video_fname = _get_video_fname
    common = types.ModuleType("bb_binary.common")
    common.bbb = object()  # fileinfo does `bbb = bb_binary.common.bbb`
    repository = types.ModuleType("bb_binary.repository")
    repository.Repository = type("Repository", (), {})  # stub; unused by our tests
    bb.parsing = parsing
    bb.common = common
    bb.repository = repository
    sys.modules["bb_binary"] = bb
    sys.modules["bb_binary.parsing"] = parsing
    sys.modules["bb_binary.common"] = common
    sys.modules["bb_binary.repository"] = repository

    pipeline = types.ModuleType("pipeline")
    scripts = types.ModuleType("pipeline.scripts")
    bb_pipeline = types.ModuleType("pipeline.scripts.bb_pipeline")
    bb_pipeline.process_video = _process_video  # set below
    pipeline.scripts = scripts
    scripts.bb_pipeline = bb_pipeline
    sys.modules["pipeline"] = pipeline
    sys.modules["pipeline.scripts"] = scripts
    sys.modules["pipeline.scripts.bb_pipeline"] = bb_pipeline


# --- fault-injectable fake process_video --------------------------------- #
# state: how many times called, and which call numbers must fail with EIO.
_STATE = {"calls": 0, "fail_set": set(), "fail_all": False}


def _expected_primary(repo_root, video_basename):
    cam, start, end = _parse_video_fname(video_basename)
    minute = int(start.minute // 20 * 20)
    rel = (
        f"{start.year}/{str(start.month).zfill(2)}/{str(start.day).zfill(2)}/"
        f"{str(start.hour).zfill(2)}/{str(minute).zfill(2)}/"
        f"{_get_video_fname(cam, start, end)}.bbb"
    )
    return os.path.join(repo_root, rel)


def _process_video(args):
    """Mimic bb_binary: create 0-byte primary stub, maybe fail (EIO) before
    writing content; otherwise write non-zero content (truncating)."""
    _STATE["calls"] += 1
    n = _STATE["calls"]
    primary = _expected_primary(args.repo_output_path, os.path.basename(args.video_path))
    os.makedirs(os.path.dirname(primary), exist_ok=True)
    open(primary, "w").close()  # 0-byte stub, like bb_binary's add()
    if _STATE["fail_all"] or n in _STATE["fail_set"]:
        raise OSError(errno.EIO, "Input/output error")
    with open(primary, "w") as f:  # truncating write -> non-zero, no double-append
        f.write("DETECTIONS")


def _bucket_path(repo_root, video_basename, dt):
    """Path of the .bbb in the 20-min bucket containing `dt`."""
    cam, start, end = _parse_video_fname(video_basename)
    m = dt.minute // 20 * 20
    rel = (
        f"{dt.year}/{str(dt.month).zfill(2)}/{str(dt.day).zfill(2)}/"
        f"{str(dt.hour).zfill(2)}/{str(m).zfill(2)}/"
        f"{_get_video_fname(cam, start, end)}.bbb"
    )
    return os.path.join(repo_root, rel)


def _process_video_boundary(args):
    """Mimic bb_binary for a boundary-spanning video: 0-byte stub, THEN the
    cross-boundary os.symlink (which the runner's shim intercepts), THEN write
    content. Reproduces the exact ordering that loses data when symlink raises."""
    _STATE["calls"] += 1
    name = os.path.basename(args.video_path)
    _cam, start, end = _parse_video_fname(name)
    primary = _bucket_path(args.repo_output_path, name, start)
    secondary = _bucket_path(args.repo_output_path, name, end)
    os.makedirs(os.path.dirname(primary), exist_ok=True)
    open(primary, "w").close()  # 0-byte stub, like bb_binary's add()
    if os.path.dirname(primary) != os.path.dirname(secondary):
        rel = os.path.relpath(primary, os.path.dirname(secondary))
        os.makedirs(os.path.dirname(secondary), exist_ok=True)
        os.symlink(rel, secondary)  # intercepted by the runner shim -> EIO -> deferred
    with open(primary, "w") as f:
        f.write("DETECTIONS")


# If the real bb_binary/pipeline are installed (e.g. on the cluster), this
# hermetic test would clobber them for sibling tests in a shared pytest session,
# so skip it there — test_pipeline_single_video.py covers the real-deps path.
_REAL_DEPS = all(importlib.util.find_spec(m) is not None for m in ("bb_binary", "pipeline"))

# repo parent on path so `import bb_hpc...` resolves to this checkout
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO.parent))

if _REAL_DEPS:
    try:
        import pytest
        pytestmark = pytest.mark.skip(reason="hermetic detect-robustness test; real bb_binary/pipeline present")
    except Exception:
        pytestmark = None
else:
    _install_fakes()
    from bb_hpc.src.jobfunctions import job_for_process_videos  # noqa: E402
    from bb_hpc.src import fileinfo  # noqa: E402
    from bb_hpc import check_symlinks_and_fill_in as csf  # noqa: E402


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _reset(fail_set=None, fail_all=False):
    _STATE["calls"] = 0
    _STATE["fail_set"] = set(fail_set or ())
    _STATE["fail_all"] = fail_all


def _video_name(cam=1, start="20260622T221901", end="20260622T222001"):
    # spans the 22:19 -> 22:20 boundary (bucket 00 -> would need 20)
    return f"cam-{cam}__{start}__{end}.mp4"


def _run(repo, videos, **kw):
    kw.setdefault("retry_backoff_sec", 0)  # no real sleeping in tests
    kw.setdefault("copy_local", False)
    return job_for_process_videos(video_paths=videos, repo_output_path=repo, **kw)


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_transient_failure_is_retried_and_succeeds():
    repo = tempfile.mkdtemp()
    try:
        _reset(fail_set={1})  # first attempt EIO, second OK
        name = _video_name()
        res = _run(repo, [str(Path(repo) / name)])
        primary = _expected_primary(repo, name)
        assert res is True
        assert _STATE["calls"] == 2, "should have retried exactly once"
        assert os.path.getsize(primary) > 0, "primary must be non-zero after retry"
        assert os.path.getsize(primary) == len("DETECTIONS"), "no double-append"
        assert not Path(repo, "_detect_failures").exists(), "no failure file on success"
        print("ok: transient failure retried and succeeded")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_persistent_failure_exits_nonzero_and_records():
    repo = tempfile.mkdtemp()
    try:
        _reset(fail_all=True)
        name = _video_name()
        src = str(Path(repo) / name)
        raised = None
        try:
            _run(repo, [src], max_attempts=3)
        except SystemExit as e:
            raised = e
        assert raised is not None, "should sys.exit on unrecoverable failure"
        assert raised.code == 1
        assert _STATE["calls"] == 3, "should have tried max_attempts times"
        # primary left absent or 0-byte (never a bogus non-zero file)
        primary = _expected_primary(repo, name)
        assert (not os.path.exists(primary)) or os.path.getsize(primary) == 0
        # durable failure list written with the source path
        fdir = Path(repo, "_detect_failures")
        files = list(fdir.glob("detect_failures_*.txt")) if fdir.exists() else []
        assert files, "expected a durable failure-list file"
        body = files[0].read_text()
        assert src in body
        print("ok: persistent failure exits 1 and records failure list")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_skip_existing_skips_nonzero_primary():
    repo = tempfile.mkdtemp()
    try:
        _reset()
        name = _video_name()
        primary = _expected_primary(repo, name)
        os.makedirs(os.path.dirname(primary), exist_ok=True)
        with open(primary, "w") as f:
            f.write("ALREADY")
        res = _run(repo, [str(Path(repo) / name)], skip_existing=True)
        assert res is True
        assert _STATE["calls"] == 0, "non-zero primary should be skipped, process_video not called"
        print("ok: skip_existing skips a non-zero primary")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_zero_byte_primary_is_not_skipped():
    repo = tempfile.mkdtemp()
    try:
        _reset()  # processing succeeds this time
        name = _video_name()
        primary = _expected_primary(repo, name)
        os.makedirs(os.path.dirname(primary), exist_ok=True)
        open(primary, "w").close()  # pre-existing 0-byte orphan
        res = _run(repo, [str(Path(repo) / name)], skip_existing=True)
        assert res is True
        assert _STATE["calls"] == 1, "0-byte primary must NOT be skipped"
        assert os.path.getsize(primary) > 0
        print("ok: 0-byte primary is reprocessed, not skipped")
    finally:
        shutil.rmtree(repo, ignore_errors=True)


def test_is_bbb_basicmatch_rejects_zero_byte():
    d = tempfile.mkdtemp()
    try:
        zero = os.path.join(d, "z.bbb")
        nonzero = os.path.join(d, "n.bbb")
        missing = os.path.join(d, "missing.bbb")
        open(zero, "w").close()
        with open(nonzero, "w") as f:
            f.write("x")
        assert fileinfo.is_bbb_file_valid_basicmatch(zero) is False, "0-byte must be invalid (default args)"
        assert fileinfo.is_bbb_file_valid_basicmatch(nonzero) is True
        assert fileinfo.is_bbb_file_valid_basicmatch(missing) is False
        print("ok: is_bbb_file_valid_basicmatch rejects 0-byte by default")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_fillin_source_guard_rejects_zero_byte():
    d = tempfile.mkdtemp()
    try:
        zero = Path(d, "z.bbb")
        nonzero = Path(d, "n.bbb")
        missing = Path(d, "missing.bbb")
        open(zero, "w").close()
        with open(nonzero, "w") as f:
            f.write("x")
        assert csf._source_is_valid(zero, validate_source=False) is False
        assert csf._source_is_valid(nonzero, validate_source=False) is True
        assert csf._source_is_valid(missing, validate_source=False) is False
        print("ok: fill-in _source_is_valid rejects 0-byte source")
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_boundary_symlink_eio_writes_primary_and_copies():
    repo = tempfile.mkdtemp()
    real_symlink = os.symlink
    bbp = sys.modules["pipeline.scripts.bb_pipeline"]
    saved_pv = bbp.process_video
    try:
        _reset()

        def _eio_symlink(target, link, *a, **k):   # this CIFS share's behavior
            raise OSError(errno.EIO, "Input/output error")

        os.symlink = _eio_symlink
        bbp.process_video = _process_video_boundary

        name = _video_name(cam=0, start="20260624T221913", end="20260624T222013")  # :19 -> :20
        res = _run(repo, [str(Path(repo) / name)])

        primary = _bucket_path(repo, name, datetime(2026, 6, 24, 22, 19, 13))
        secondary = _bucket_path(repo, name, datetime(2026, 6, 24, 22, 20, 13))
        assert res is True
        assert _STATE["calls"] == 1, "should succeed on first attempt (symlink tolerated, not retried)"
        assert os.path.getsize(primary) > 0, "primary must be written despite the symlink EIO"
        assert os.path.exists(secondary) and os.path.getsize(secondary) > 0, \
            "cross-boundary entry must be materialized as a real copy"
        assert not Path(repo, "_detect_failures").exists(), "no failure on a tolerated symlink"
        print("ok: boundary symlink EIO -> primary written + cross-boundary copy created")
    finally:
        os.symlink = real_symlink
        bbp.process_video = saved_pv
        shutil.rmtree(repo, ignore_errors=True)


def test_materialize_overwrites_zero_byte_secondary():
    repo = tempfile.mkdtemp()
    real_symlink = os.symlink
    bbp = sys.modules["pipeline.scripts.bb_pipeline"]
    saved_pv = bbp.process_video
    try:
        _reset()

        def _eio_symlink(target, link, *a, **k):
            raise OSError(errno.EIO, "Input/output error")

        os.symlink = _eio_symlink
        bbp.process_video = _process_video_boundary

        name = _video_name(cam=0, start="20260624T221913", end="20260624T222013")
        secondary = _bucket_path(repo, name, datetime(2026, 6, 24, 22, 20, 13))
        os.makedirs(os.path.dirname(secondary), exist_ok=True)
        open(secondary, "w").close()  # legacy 0-byte secondary stub from a prior partial run

        res = _run(repo, [str(Path(repo) / name)])
        assert res is True
        assert os.path.getsize(secondary) > 0, "pre-existing 0-byte secondary must be overwritten with a real copy"
        print("ok: materialize overwrites a pre-existing 0-byte secondary")
    finally:
        os.symlink = real_symlink
        bbp.process_video = saved_pv
        shutil.rmtree(repo, ignore_errors=True)


_TESTS = [
    test_transient_failure_is_retried_and_succeeds,
    test_persistent_failure_exits_nonzero_and_records,
    test_skip_existing_skips_nonzero_primary,
    test_zero_byte_primary_is_not_skipped,
    test_is_bbb_basicmatch_rejects_zero_byte,
    test_fillin_source_guard_rejects_zero_byte,
    test_boundary_symlink_eio_writes_primary_and_copies,
    test_materialize_overwrites_zero_byte_secondary,
]


def main():
    if _REAL_DEPS:
        print("skipped (real bb_binary/pipeline installed; use test_pipeline_single_video.py)")
        return
    failed = 0
    for t in _TESTS:
        try:
            t()
        except Exception as e:
            failed += 1
            print(f"FAIL: {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(_TESTS) - failed}/{len(_TESTS)} passed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
