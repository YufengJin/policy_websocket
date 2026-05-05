#!/usr/bin/env python3
"""Single-step policy server (no Action Chunk).

Usage:
    python examples/policy_server.py [--port 8000]
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

from policy_websocket import BasePolicy, WebsocketPolicyServer


class SimplePolicy(BasePolicy):
    """Returns one action (7,) per infer."""

    def infer(self, obs: Dict) -> Dict:
        action_dim = obs.get("action_dim", 7)
        action = np.random.randn(action_dim).astype(np.float64) * 0.1
        return {"actions": action}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    policy = SimplePolicy()
    server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port)

    print(f"Single-step policy server on ws://0.0.0.0:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
