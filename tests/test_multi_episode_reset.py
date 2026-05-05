"""Smoke: three episodes, verify policy.reset() round-trip fires for each."""

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = REPO_ROOT / "examples"

EPISODES = 3
STEPS = 8


@pytest.mark.parametrize("policy_server_ac", [8766], indirect=True)
def test_three_episodes_each_reset(policy_server_ac):
    port = policy_server_ac
    result = subprocess.run(
        [sys.executable, "-u", str(EXAMPLES / "policy_client.py"),
         "--host", "localhost", "--port", str(port),
         "--episodes", str(EPISODES), "--steps", str(STEPS)],
        capture_output=True, text=True, timeout=60,
    )
    print("=== CLIENT STDOUT ===")
    print(result.stdout)
    if result.stderr:
        print("=== CLIENT STDERR ===")
        print(result.stderr)

    assert result.returncode == 0, f"client exited {result.returncode}"
    out = result.stdout
    assert out.count("reset ack received") == EPISODES
    assert out.count("Init infer:") == EPISODES
    for ep in range(1, EPISODES + 1):
        assert f"Episode {ep}/{EPISODES}: reset ack received" in out
    assert f"Done. {EPISODES} episode(s)" in out
