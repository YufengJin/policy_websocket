#!/usr/bin/env python3
"""Action Chunk policy server: predict 16 steps, execute 8 per chunk.

Usage:
    python examples/policy_server_ac.py [--port 8000]
"""

import argparse
import os
import sys
from typing import Dict

import numpy as np

_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_script_dir, "..", "src"))

from policy_websocket import BasePolicy, ActionChunkBroker, WebsocketPolicyServer


class ChunkPolicy(BasePolicy):
    """Returns action chunk shape (16, 7) — predict 16, broker yields 8 per chunk."""

    def infer(self, obs: Dict) -> Dict:
        # Simulate: always return 16-step chunk
        action_dim = obs.get("action_dim", 7)
        chunk = np.random.randn(16, action_dim).astype(np.float64) * 0.1
        return {"actions": chunk}

    def reset(self) -> None:
        pass


class ResetOnInitPolicy(BasePolicy):
    """Calls inner policy reset() when first infer has action_dim but no images."""

    def __init__(self, policy: BasePolicy) -> None:
        self._policy = policy

    def infer(self, obs: Dict) -> Dict:
        if "action_dim" in obs and "primary_image" not in obs:
            self._policy.reset()
        return self._policy.infer(obs)

    def reset(self) -> None:
        self._policy.reset()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    chunk_policy = ChunkPolicy()
    broker = ActionChunkBroker(policy=chunk_policy, action_horizon=8)
    reset_policy = ResetOnInitPolicy(broker)
    server = WebsocketPolicyServer(policy=reset_policy, host="0.0.0.0", port=args.port)

    print(f"Action Chunk policy server on ws://0.0.0.0:{args.port} (predict 16, execute 8)")
    server.serve_forever()


if __name__ == "__main__":
    main()
