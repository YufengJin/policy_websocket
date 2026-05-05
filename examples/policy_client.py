#!/usr/bin/env python3
"""Policy client: connects to a policy server and runs a simulated episode.

Usage:
    python examples/policy_client.py --host localhost --port 8000 [--steps 24]
                                     [--episodes 1]
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
    """First infer: episode init (env action spec, no images)."""
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
    parser.add_argument("--steps", type=int, default=24, help="Steps per episode")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    args = parser.parse_args()

    policy = WebsocketClientPolicy(host=args.host, port=args.port)
    metadata = policy.get_server_metadata()
    print(f"Connected. Server metadata: {metadata}")

    for ep in range(args.episodes):
        # Sends a reset request to the policy server. Required when the server
        # holds episode-scoped state — e.g. an ActionChunkBroker would need to clean its
        # cached chunk from the previous episode.
        policy.reset()
        print(f"Episode {ep + 1}/{args.episodes}: reset ack received")

        init_action = policy.infer(make_init_obs())["actions"]
        print(f"  Init infer: actions shape {init_action.shape}, sample {init_action[:3]}")

        for step in range(args.steps):
            obs = make_step_obs(step)
            action_dict = policy.infer(obs)
            actions = action_dict["actions"]
            if step < 3 or step == args.steps - 1:
                print(f"  Step {step}: actions shape {actions.shape}, sample {actions[:3]}")

    policy.close()
    print(f"Done. {args.episodes} episode(s), {args.steps} step(s) each.")


if __name__ == "__main__":
    main()
