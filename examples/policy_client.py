#!/usr/bin/env python3
"""Policy client: connects to a policy server and runs a simulated episode.

Usage:
    python examples/policy_client.py --host localhost --port 8000 [--steps 24]
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

from policy_websocket import WebsocketClientPolicy


def make_init_obs() -> Dict:
    """First infer: episode init (no images)."""
    return {
        "action_dim": 7,
        "action_low": np.full(7, -1.0, dtype=np.float64),
        "action_high": np.full(7, 1.0, dtype=np.float64),
        "task_description": "example task",
    }


def make_step_obs(step: int, h: int = 64, w: int = 64) -> Dict:
    """Step infer: full obs with placeholder images."""
    return {
        "primary_image": np.zeros((h, w, 3), dtype=np.uint8),
        "secondary_image": np.zeros((h, w, 3), dtype=np.uint8),
        "wrist_image": np.zeros((h, w, 3), dtype=np.uint8),
        "proprio": np.zeros(14, dtype=np.float64),
        "task_description": "example task",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--steps", type=int, default=24, help="Number of control steps")
    args = parser.parse_args()

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    print(f"Connected. Server metadata: {metadata}")

    # First infer: episode init
    init_obs = make_init_obs()
    init_action = policy.infer(init_obs)
    actions = init_action["actions"]
    print(f"Init infer: actions shape {actions.shape}, sample {actions[:3]}")

    # Step infers
    for step in range(args.steps):
        obs = make_step_obs(step)
        action_dict = policy.infer(obs)
        actions = action_dict["actions"]
        if step < 3 or step == args.steps - 1:
            print(f"  Step {step}: actions shape {actions.shape}, sample {actions[:3]}")

    policy.close()
    print(f"Done. Completed {args.steps + 1} infers (1 init + {args.steps} steps).")


if __name__ == "__main__":
    main()
