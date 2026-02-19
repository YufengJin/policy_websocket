"""Wraps a policy to return action chunks one step at a time.

If a policy returns an action chunk of shape (H, action_dim), this broker
yields one row per ``infer`` call and re-queries the inner policy only after
the chunk is exhausted.
"""

from typing import Dict, Optional

import numpy as np

from policy_websocket.base_policy import BasePolicy


class ActionChunkBroker(BasePolicy):
    def __init__(self, policy: BasePolicy, action_horizon: int):
        self._policy = policy
        self._action_horizon = action_horizon
        self._cur_step: int = 0
        self._last_results: Optional[Dict] = None

    def infer(self, obs: Dict) -> Dict:
        if self._last_results is None:
            self._last_results = self._policy.infer(obs)
            self._cur_step = 0

        results = _slice_step(self._last_results, self._cur_step)
        self._cur_step += 1

        if self._cur_step >= self._action_horizon:
            self._last_results = None

        return results

    def reset(self) -> None:
        self._policy.reset()
        self._last_results = None
        self._cur_step = 0


def _slice_step(d: Dict, step: int) -> Dict:
    """Recursively slice the step-th element from arrays in a nested dict."""
    out = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray) and v.ndim >= 2:
            out[k] = v[step, ...]
        elif isinstance(v, dict):
            out[k] = _slice_step(v, step)
        else:
            out[k] = v
    return out
