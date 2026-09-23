"""
Validity checks for .bbb files, against real bb_binary.

These pin two failures that both showed up as "the scan tool is useless on exactly
the files it exists to find":

  * Reading capnp framing from a corrupt .bbb aborts the process on some capnp
    builds -- a C++ terminate(), not a Python exception. Inside a
    multiprocessing.Pool worker that killed the worker, and the parent then waited
    forever for a result that was never coming.
  * is_bbb_file_valid_deep treated "read blew up after consuming to EOF" as a clean
    end of stream, so every truncated file passed.

Everything here runs in child processes on purpose. Importing bb_binary (or anything
under bb_hpc that pulls it in) at module scope would install the real module into
sys.modules for the whole session, and test_progress.py depends on getting its fake
in first -- whichever module imports first wins, and pytest collects this file first.
Staying out of the parent also means a regression to a hanging pool fails on a
timeout here instead of wedging the entire test run.
"""
import json
import os
import subprocess
import sys

import pytest

REPO_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_MAKE_FIXTURES = '''
import os, struct, sys
sys.path.insert(0, {parent!r})
import bb_binary
bbb = bb_binary.common.bbb

day = {day!r}
os.makedirs(day, exist_ok=True)

def write_containers(path, n):
    with open(path, "wb") as f:
        for i in range(n):
            fc = bbb.FrameContainer.new_message()
            fc.fromTimestamp = float(i)
            fc.toTimestamp = float(i + 1)
            fc.camId = 0
            fc.init("frames", 1)
            fc.write(f)

write_containers(os.path.join(day, "v1.bbb"), 1)
write_containers(os.path.join(day, "v3.bbb"), 3)

# Truncated mid-message: what an interrupted detect run leaves behind.
whole = open(os.path.join(day, "v3.bbb"), "rb").read()
open(os.path.join(day, "cut.bbb"), "wb").write(whole[:-40])

# Header promises 1024 words (8192 bytes) and delivers 100.
open(os.path.join(day, "hdr.bbb"), "wb").write(struct.pack("<II", 0, 1024) + b"\\x00" * 100)

open(os.path.join(day, "zero.bbb"), "wb").close()
'''

_CHECK = '''
import json, os, sys
sys.path.insert(0, {parent!r})
from bb_hpc.src.fileinfo import is_bbb_file_valid_basicmatch, is_bbb_file_valid_deep

day = {day!r}
out = {{}}
for name in ("v1", "v3", "cut", "hdr", "zero"):
    p = os.path.join(day, name + ".bbb")
    out[name] = {{
        "basic": is_bbb_file_valid_basicmatch(p, check_read_file=True),
        "deep": is_bbb_file_valid_deep(p),
    }}
print(json.dumps(out))
'''


def _run(code, timeout=180):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=timeout)


@pytest.fixture(scope="module")
def day_dir(tmp_path_factory):
    root = tmp_path_factory.mktemp("repo")
    day = root / "2026" / "09" / "04"
    r = _run(_MAKE_FIXTURES.format(parent=REPO_PARENT, day=str(day)))
    if r.returncode != 0:
        pytest.skip(f"cannot build .bbb fixtures (needs real bb_binary): "
                    f"{r.stderr.decode()[-300:]}")
    return root, day


@pytest.fixture(scope="module")
def verdicts(day_dir):
    _root, day = day_dir
    r = _run(_CHECK.format(parent=REPO_PARENT, day=str(day)))
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    return json.loads(r.stdout)


def test_deep_check_rejects_truncated_files(verdicts):
    """The regression: a file cut mid-message used to come back valid."""
    assert verdicts["v1"]["deep"] is True
    assert verdicts["v3"]["deep"] is True
    assert verdicts["cut"]["deep"] is False
    assert verdicts["hdr"]["deep"] is False
    assert verdicts["zero"]["deep"] is False


def test_basic_check_survives_corrupt_framing(verdicts):
    """Corrupt framing must return False, not abort the process doing the check."""
    assert verdicts["v1"]["basic"] is True
    assert verdicts["hdr"]["basic"] is False
    assert verdicts["zero"]["basic"] is False
    # The child exiting 0 at all is the real assertion: an unisolated read aborts
    # the interpreter, and the check fixture would have failed on returncode.


def test_basic_check_is_shallow_by_design(verdicts):
    """basicmatch reads only the first container, so a later cut is deep's job."""
    assert verdicts["cut"]["basic"] is True
    assert verdicts["cut"]["deep"] is False


@pytest.mark.parametrize("workers", [1, 4])
def test_scan_terminates_on_corrupt_files(day_dir, workers):
    """The scan must finish rather than hang when corrupt files are present."""
    root, _day = day_dir
    cmd = [sys.executable, "-m", "bb_hpc.scan_and_remove_invalid_bbb_files",
           "--pipeline-root", str(root), "--dates", "20260904",
           "--deep-check-bbb", "--dry-run", "--num-workers", str(workers)]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=180, cwd=REPO_PARENT)
    except subprocess.TimeoutExpired:
        pytest.fail(f"scan hung with --num-workers {workers} (the Pool regression)")

    assert r.returncode == 0, r.stderr.decode()[-2000:]
    out = r.stdout.decode()
    assert "[total] files=5 invalid=3 (dry-run)" in out, out
