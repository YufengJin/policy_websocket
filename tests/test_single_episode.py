"""Smoke: examples/policy_client.py against examples/policy_server_ac.py for one episode."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"


@pytest.mark.parametrize("policy_server_ac", [8765], indirect=True)
def test_single_episode(policy_server_ac):
    port = policy_server_ac
    result = subprocess.run(
        [sys.executable, "-u", str(EXAMPLES / "policy_client.py"),
         "--host", "localhost", "--port", str(port),
         "--episodes", "1", "--steps", "8"],
        capture_output=True, text=True, timeout=30,
    )
    print("=== CLIENT STDOUT ===")
    print(result.stdout)
    if result.stderr:
        print("=== CLIENT STDERR ===")
        print(result.stderr)

    assert result.returncode == 0, f"client exited {result.returncode}"
    out = result.stdout
    assert "Connected. Server metadata:" in out
    assert out.count("reset ack received") == 1
    assert out.count("Init infer:") == 1
    assert "Done. 1 episode(s)" in out
