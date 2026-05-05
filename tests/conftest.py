"""Shared fixtures for example smoke tests."""

import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


@pytest.fixture
def policy_server_ac(request):
    """Boot examples/policy_server_ac.py on a port; yield the port; terminate on teardown.

    Override the port via ``@pytest.mark.parametrize`` indirect=True if a test
    needs a non-default port (e.g. to avoid collision when running in parallel).
    """
    port = getattr(request, "param", 8765)
    proc = subprocess.Popen(
        [sys.executable, "-u", str(EXAMPLES / "policy_server_ac.py"),
         "--port", str(port)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
    )
    time.sleep(2.0)  # let the server bind + start serving
    if proc.poll() is not None:
        out, _ = proc.communicate(timeout=5)
        pytest.fail(f"policy_server_ac.py died during boot:\n{out}")

    try:
        yield port
    finally:
        proc.terminate()
        try:
            out, _ = proc.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            out, _ = proc.communicate()
        print("\n=== SERVER STDOUT ===")
        print(out)
