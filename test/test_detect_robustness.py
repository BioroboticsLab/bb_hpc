#!/usr/bin/env python3
"""
Hermetic tests for the detect-robustness hardening (no bb_binary/pipeline needed).

Covers:
- job_for_process_videos retry + clean-slate + post-write validation + runtime
  skip + durable failure list + sys.exit(1)  (src/jobfunctions.py)
- the durable failure list lands *beside* the bb_binary repo root, never inside
  it, and assert_clean_repo_root rejects a root that already holds a stray dir
  (src/jobfunctions.py, src/repo_guard.py)
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
from datetime import datetime, timedelta
from pathlib import Path

# --------------------------------------------------------------------------- #
# Install fake bb_binary / pipeline BEFORE importing any bb_hpc module.
# --------------------------------------------------------------------------- #
_TS = "%Y%m%dT%H%M%S"        # video-side timestamp:  20260622T221901
_BTS = "%Y-%m-%dT%H_%M_%S"   # bb_binary-side stamp:  2026-06-22T22_19_01


def _strptime(s, fmt):
    """strptime that tolerates an optional .<microseconds> tail."""
    return datetime.strptime(s, fmt + ".%f" if "." in s else fmt)


def _parse_video_fname(basename, format=None):
    """Fake of bb_binary.parsing.parse_video_fname.

    Handles both shapes the real one does:
      video: cam-<id>__<startTS>__<endTS>.mp4          (this test's convention)
      bbb:   Cam_<id>_<start>--<end>.bbb               (real bb_binary shape)
    """
    stem = basename
    for ext in (".mp4", ".bbb", ".avi"):
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
    if "--" in stem:  # bb_binary output name
        head, e = stem.split("--", 1)
        _cam_label, cam_s, s = head.split("_", 2)
        return (int(cam_s),
                _strptime(s.rstrip("Z"), _BTS),
                _strptime(e.rstrip("Z"), _BTS))
    cam_s, s, e = stem.split("__")
    cam = int(cam_s.replace("cam-", "").replace("Cam_", ""))
    return cam, _strptime(s, _TS), _strptime(e, _TS)


def _dt_to_str(dt):
    """Fake of bb_binary.parsing.dt_to_str: microseconds ONLY when non-zero.

    That omission is load-bearing -- it is why the runner's search prefix is
    truncated to whole seconds rather than matching exact microseconds.
    """
    s = dt.strftime(_BTS)
    if dt.microsecond:
        s += f".{dt.microsecond:06d}"
    return s + "Z"


def _get_fname(cam, start):
    """Fake of bb_binary.parsing.get_fname -> "Cam_{id}_{dt_to_str(start)}"."""
    return f"Cam_{cam}_{_dt_to_str(start)}"


def _get_video_fname(cam, start, end):
    """Fake of bb_binary.parsing.get_video_fname -- REAL bb_binary shape.

    The '--' separator and the optional '.<us>' tail are load-bearing: the runner
    derives its search prefix from get_fname(cam, start) and globs '<prefix>*.bbb'.
    The old 'Cam_{cam}__{ts}__{ts}' stem made the gappy-end bug untestable by
    construction, because no part of the name distinguished start from end.
    """
    return f"{_get_fname(cam, start)}--{_dt_to_str(end)}"


def _install_fakes():
    bb = types.ModuleType("bb_binary")
    parsing = types.ModuleType("bb_binary.parsing")
    parsing.parse_video_fname = _parse_video_fname
    parsing.get_video_fname = _get_video_fname
    parsing.get_fname = _get_fname
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
_STATE = {"calls": 0, "fail_set": set(), "fail_all": False, "end_shift_sec": 0}


def _primary_path(repo_root, video_basename, end_override=None):
    """Path of the primary .bbb.

    The BUCKET always comes from the start timestamp (as in bb_binary's
    _path_for_dt), but the NAME comes from the content timestamps -- which is
    exactly why `end_override` exists: a gappy recording's last frame lands past
    the end the video's filename advertises.
    """
    cam, start, end = _parse_video_fname(video_basename)
    minute = int(start.minute // 20 * 20)
    rel = (
        f"{start.year}/{str(start.month).zfill(2)}/{str(start.day).zfill(2)}/"
        f"{str(start.hour).zfill(2)}/{str(minute).zfill(2)}/"
        f"{_get_video_fname(cam, start, end_override or end)}.bbb"
    )
    return os.path.join(repo_root, rel)


def _content_end(video_basename):
    """The end timestamp bb_binary would actually record, given end_shift_sec."""
    _cam, _start, end = _parse_video_fname(video_basename)
    return end + timedelta(seconds=_STATE["end_shift_sec"])


def _process_video(args):
    """Mimic bb_binary: create 0-byte primary stub, maybe fail (EIO) before
    writing content; otherwise write non-zero content (truncating)."""
    _STATE["calls"] += 1
    n = _STATE["calls"]
    name = os.path.basename(args.video_path)
    primary = _primary_path(args.repo_output_path, name, end_override=_content_end(name))
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
def _reset(fail_set=None, fail_all=False, end_shift_sec=0):
    _STATE["calls"] = 0
    _STATE["fail_set"] = set(fail_set or ())
    _STATE["fail_all"] = fail_all
    _STATE["end_shift_sec"] = end_shift_sec


def _video_name(cam=1, start="20260622T221901", end="20260622T222001"):
    # spans the 22:19 -> 22:20 boundary (bucket 00 -> would need 20)
    return f"cam-{cam}__{start}__{end}.mp4"


def _run(repo, videos, **kw):
    kw.setdefault("retry_backoff_sec", 0)  # no real sleeping in tests
    kw.setdefault("copy_local", False)
    return job_for_process_videos(video_paths=videos, repo_output_path=repo, **kw)


def _mkrepo():
    """Build <parent>/pipeline_repo and return both.

    The default failure-list dir is a *sibling* of the repo root, so the repo must
    have a dedicated parent: with a bare mkdtemp() repo the sibling would be the
    shared system temp dir, where one test's failure list would leak into the next
    test's absence assertions.
    """
    parent = tempfile.mkdtemp()
    repo = os.path.join(parent, "pipeline_repo")
    os.makedirs(repo)
    return parent, repo


def _failure_dir(repo):
    """Mirror of job_for_process_videos' default: a sibling of the repo root."""
    return Path(os.path.dirname(os.path.abspath(repo)), "detect_failures")


def _assert_repo_root_clean(repo):
    """The repo root must contain only bb_binary's numeric YYYY buckets.

    bb_binary int()-parses every directory name at the root, so anything else
    there breaks all reads (see src/repo_guard.py).
    """
    stray = sorted(d for d in os.listdir(repo)
                   if not d.isdigit() and os.path.isdir(os.path.join(repo, d)))
    assert not stray, f"non-numeric dirs in repo root would break bb_binary reads: {stray}"


# --------------------------------------------------------------------------- #
# tests
# --------------------------------------------------------------------------- #
def test_transient_failure_is_retried_and_succeeds():
    parent, repo = _mkrepo()
    try:
        _reset(fail_set={1})  # first attempt EIO, second OK
        name = _video_name()
        res = _run(repo, [str(Path(repo) / name)])
        primary = _primary_path(repo, name)
        assert res is True
        assert _STATE["calls"] == 2, "should have retried exactly once"
        assert os.path.getsize(primary) > 0, "primary must be non-zero after retry"
        assert os.path.getsize(primary) == len("DETECTIONS"), "no double-append"
        assert not _failure_dir(repo).exists(), "no failure file on success"
        _assert_repo_root_clean(repo)
        print("ok: transient failure retried and succeeded")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_persistent_failure_exits_nonzero_and_records():
    parent, repo = _mkrepo()
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
        primary = _primary_path(repo, name)
        assert (not os.path.exists(primary)) or os.path.getsize(primary) == 0
        # durable failure list written with the source path, beside the repo root
        fdir = _failure_dir(repo)
        files = list(fdir.glob("detect_failures_*.txt")) if fdir.exists() else []
        assert files, f"expected a durable failure-list file under {fdir}"
        body = files[0].read_text()
        assert src in body
        print("ok: persistent failure exits 1 and records failure list")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_failure_list_default_is_outside_repo_root():
    """Regression: a failure list inside the repo root breaks every bb_binary read.

    `_detect_failures` used to be written to <repo>/_detect_failures. Because '_'
    sorts above every digit, Repository._get_directory(max) selected it as the
    "latest path" and int()-parsed it, so tracking and save_detect died with
    ValueError: invalid literal for int() with base 10: '_detect_failures'.
    """
    parent, repo = _mkrepo()
    try:
        _reset(fail_all=True)
        name = _video_name()
        src = str(Path(repo) / name)
        try:
            _run(repo, [src], max_attempts=1, failure_list_dir=None)
        except SystemExit:
            pass

        _assert_repo_root_clean(repo)
        assert not Path(repo, "_detect_failures").exists(), \
            "failure list must never be written inside the bb_binary repo root"
        assert not Path(repo, "detect_failures").exists(), \
            "not even under a numeric-safe name -- it still is not a timestamp dir"

        fdir = _failure_dir(repo)
        assert fdir.is_dir(), f"failure list should be a sibling of the repo root, at {fdir}"
        files = list(fdir.glob("detect_failures_*.txt"))
        assert files, "sibling failure dir should hold the recorded failure"
        assert src in files[0].read_text()
        print("ok: default failure list lands beside the repo root, not inside it")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_explicit_failure_list_dir_is_honored():
    parent, repo = _mkrepo()
    try:
        _reset(fail_all=True)
        name = _video_name()
        src = str(Path(repo) / name)
        explicit = Path(parent, "somewhere_else")
        try:
            _run(repo, [src], max_attempts=1, failure_list_dir=str(explicit))
        except SystemExit:
            pass
        files = list(explicit.glob("detect_failures_*.txt"))
        assert files, f"explicit failure_list_dir {explicit} should have been used"
        assert src in files[0].read_text()
        assert not _failure_dir(repo).exists(), "explicit dir should override the default"
        print("ok: explicit failure_list_dir overrides the default")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_skip_existing_skips_nonzero_primary():
    parent, repo = _mkrepo()
    try:
        _reset()
        name = _video_name()
        primary = _primary_path(repo, name)
        os.makedirs(os.path.dirname(primary), exist_ok=True)
        with open(primary, "w") as f:
            f.write("ALREADY")
        res = _run(repo, [str(Path(repo) / name)], skip_existing=True)
        assert res is True
        assert _STATE["calls"] == 0, "non-zero primary should be skipped, process_video not called"
        print("ok: skip_existing skips a non-zero primary")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_zero_byte_primary_is_not_skipped():
    parent, repo = _mkrepo()
    try:
        _reset()  # processing succeeds this time
        name = _video_name()
        primary = _primary_path(repo, name)
        os.makedirs(os.path.dirname(primary), exist_ok=True)
        open(primary, "w").close()  # pre-existing 0-byte orphan
        res = _run(repo, [str(Path(repo) / name)], skip_existing=True)
        assert res is True
        assert _STATE["calls"] == 1, "0-byte primary must NOT be skipped"
        assert os.path.getsize(primary) > 0
        print("ok: 0-byte primary is reprocessed, not skipped")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_gappy_end_timestamp_still_counts_as_done():
    """The outage of 2026-07-21: a valid .bbb whose END differs from the video's.

    bb_binary names its output from the ACTUAL frame timestamps, not the video's
    filename. Real case: cam-1_...T081529.782248...--...T081812.855549....mp4
    produced Cam_1_2026-07-16T08_15_29.782248Z--2026-07-16T08_19_11.135227Z.bbb --
    a valid 11.9 MB file whose end is 58s past what the filename implied, because
    the recording dropped frames. Predicting the full start--end stem called that
    success a failure, burned all 3 retries, exited the pod 1, and with
    backoffLimit: 1 marked the entire 141-index Indexed Job Failed.
    """
    parent, repo = _mkrepo()
    try:
        _reset(end_shift_sec=59)
        name = _video_name(cam=1, start="20260716T081529.782248",
                                  end="20260716T081812.855549")
        res = _run(repo, [str(Path(repo) / name)])
        written = _primary_path(repo, name, end_override=_content_end(name))
        assert res is True
        assert _STATE["calls"] == 1, "a valid .bbb with a gappy end must not be retried"
        assert os.path.getsize(written) > 0
        assert written != _primary_path(repo, name), \
            "the test is only meaningful if the written name differs from the predicted one"
        assert not _failure_dir(repo).exists(), "a written .bbb is not a failure"
        _assert_repo_root_clean(repo)
        print("ok: gappy end timestamp still counts as done")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_gappy_end_is_skipped_on_rerun():
    """skip_existing must find the gappy-named .bbb too.

    Otherwise every rerun redoes the video and the submit-time scanner
    re-schedules it forever -- the slow-burn half of the same bug.
    """
    parent, repo = _mkrepo()
    try:
        _reset(end_shift_sec=59)
        name = _video_name(cam=1, start="20260716T081529.782248",
                                  end="20260716T081812.855549")
        assert _run(repo, [str(Path(repo) / name)]) is True
        assert _STATE["calls"] == 1

        _reset(end_shift_sec=59)  # second submit, same video, output already there
        assert _run(repo, [str(Path(repo) / name)], skip_existing=True) is True
        assert _STATE["calls"] == 0, "a gappy-named .bbb must be recognized as already done"
        print("ok: gappy-named .bbb is skipped on rerun")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_zero_byte_gappy_stub_is_cleared_before_retry():
    """The 0-byte stub is named from the CONTENT timestamps too.

    So the clean-slate step must clear any 0-byte match in the bucket -- the old
    exact-name unlink could only ever find the predicted name, leaking mis-named
    stubs on disk forever.
    """
    parent, repo = _mkrepo()
    try:
        _reset(end_shift_sec=59)
        name = _video_name(cam=1, start="20260716T081529.782248",
                                  end="20260716T081812.855549")
        stub = _primary_path(repo, name, end_override=_content_end(name))
        os.makedirs(os.path.dirname(stub), exist_ok=True)
        open(stub, "w").close()  # 0-byte orphan at the gappy name

        res = _run(repo, [str(Path(repo) / name)], skip_existing=True)
        assert res is True
        assert _STATE["calls"] == 1, "a 0-byte gappy stub must NOT be treated as done"
        assert os.path.getsize(stub) > 0, "stub should have been cleared and rewritten"
        print("ok: 0-byte gappy stub is cleared, not skipped")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_start_prefix_does_not_match_a_different_camera():
    """Cam_1_<t> must not match Cam_11_<t>: the '_' after the cam id anchors it."""
    parent, repo = _mkrepo()
    try:
        _reset()
        name = _video_name(cam=1, start="20260716T081529.782248",
                                  end="20260716T081812.855549")
        # a decoy from camera 11, same bucket, same start second
        decoy_src = _video_name(cam=11, start="20260716T081529.782248",
                                        end="20260716T081812.855549")
        decoy = _primary_path(repo, decoy_src)
        os.makedirs(os.path.dirname(decoy), exist_ok=True)
        with open(decoy, "w") as f:
            f.write("SOMEONE ELSE'S DETECTIONS")

        res = _run(repo, [str(Path(repo) / name)], skip_existing=True)
        assert res is True
        assert _STATE["calls"] == 1, "cam 11's output must not satisfy cam 1's check"
        print("ok: start prefix does not match a different camera")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


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
    parent, repo = _mkrepo()
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
        assert not _failure_dir(repo).exists(), "no failure on a tolerated symlink"
        _assert_repo_root_clean(repo)
        print("ok: boundary symlink EIO -> primary written + cross-boundary copy created")
    finally:
        os.symlink = real_symlink
        bbp.process_video = saved_pv
        shutil.rmtree(parent, ignore_errors=True)


def test_materialize_overwrites_zero_byte_secondary():
    parent, repo = _mkrepo()
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
        shutil.rmtree(parent, ignore_errors=True)


def test_repo_guard_rejects_stray_root_dir():
    from bb_hpc.src.repo_guard import assert_clean_repo_root, stray_repo_root_dirs

    parent, repo = _mkrepo()
    try:
        os.makedirs(os.path.join(repo, "2026", "06", "10"))
        assert stray_repo_root_dirs(repo) == []
        assert_clean_repo_root(repo)  # clean root: must not raise

        # bb_binary's own config file is a file, not a dir -- already harmless.
        open(os.path.join(repo, ".bbb_repo_config.json"), "w").close()
        assert stray_repo_root_dirs(repo) == []

        os.makedirs(os.path.join(repo, "_detect_failures"))
        assert stray_repo_root_dirs(repo) == ["_detect_failures"]
        raised = None
        try:
            assert_clean_repo_root(repo)
        except SystemExit as e:
            raised = e
        assert raised is not None, "a stray root dir must be rejected at submit time"
        assert "_detect_failures" in str(raised.code)
        print("ok: repo_guard rejects a stray dir in the repo root")
    finally:
        shutil.rmtree(parent, ignore_errors=True)


def test_stray_root_dir_poisons_both_ends_of_the_sort():
    """Why the failure dir must move *out*, not be renamed or hidden.

    Repository._get_directory takes max() for the latest path and min() for the
    earliest, over the raw child names. '_' (0x5F) sorts above every digit and a
    leading '.' (0x2E) sorts below, so neither an underscore nor a dot escapes --
    one poisons _get_latest_path, the other _get_earliest_path.
    """
    years = ["2025", "2026"]
    assert max(years + ["_detect_failures"]) == "_detect_failures"
    assert min(years + [".detect_failures"]) == ".detect_failures"
    # and both are unparseable as a bb_binary timestamp component
    for bad in ("_detect_failures", ".detect_failures"):
        raised = None
        try:
            int(bad)
        except ValueError as e:
            raised = e
        assert raised is not None
    print("ok: stray dirs poison max() and min() alike -- move them out of the root")


_TESTS = [
    test_transient_failure_is_retried_and_succeeds,
    test_persistent_failure_exits_nonzero_and_records,
    test_failure_list_default_is_outside_repo_root,
    test_explicit_failure_list_dir_is_honored,
    test_skip_existing_skips_nonzero_primary,
    test_zero_byte_primary_is_not_skipped,
    test_gappy_end_timestamp_still_counts_as_done,
    test_gappy_end_is_skipped_on_rerun,
    test_zero_byte_gappy_stub_is_cleared_before_retry,
    test_start_prefix_does_not_match_a_different_camera,
    test_is_bbb_basicmatch_rejects_zero_byte,
    test_fillin_source_guard_rejects_zero_byte,
    test_boundary_symlink_eio_writes_primary_and_copies,
    test_materialize_overwrites_zero_byte_secondary,
    test_repo_guard_rejects_stray_root_dir,
    test_stray_root_dir_poisons_both_ends_of_the_sort,
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
