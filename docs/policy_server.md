# Policy Server Setup Guide

This document describes how to wrap any policy (including PyTorch `nn.Module`) as a Policy Server, supporting both single-step output and Action Chunk. Compatible with RoboCasa `run_demo`/`run_eval`, openpi, and other clients.

---

## 1. Prerequisites: Policy Interface Requirements

### 1.1 BasePolicy Interface

Any policy that serves as a Policy Server must implement `BasePolicy`:

```python
from policy_websocket import BasePolicy

class MyPolicy(BasePolicy):
    def infer(self, obs: Dict) -> Dict:
        """Observation → action dict. Must implement."""
        ...
        return {"actions": action_array}

    def reset(self) -> None:
        """Reset when a new episode starts. Optional, defaults to pass."""
        pass
```

### 1.2 Observation `obs` Format (from client)

| Field | Type | Description |
|------|------|------|
| `primary_image` | `np.ndarray` (H,W,3) | Primary camera RGB |
| `secondary_image` | `np.ndarray` (H,W,3) | Secondary camera RGB |
| `wrist_image` | `np.ndarray` (H,W,3) | Wrist camera RGB |
| `proprio` | `np.ndarray` (D,) | Proprioception (gripper, end-effector pose, etc.) |
| `task_description` | `str` | Natural language task description |
| `action_dim` / `action_low` / `action_high` | First infer only | Action spec for episode initialization |

The first infer contains only `action_dim`, `action_low`, `action_high`, `task_name`, `task_description`; no images.

### 1.3 Return `action` Format

| Field | Type | Description |
|------|------|------|
| `actions` | `np.ndarray` | `(action_dim,)` or `(7,)`. Client auto-pads 7-dim to env dimension |

---

## 2. Policy Server Requirements

The following conditions are sufficient to start a Policy Server:

1. Implement `BasePolicy`'s `infer(obs) -> Dict`, returning `{"actions": np.ndarray}`
2. Return `actions` as `np.float64`, shape `(action_dim,)` or `(7,)`
3. (Recommended) Handle first-infer action spec: `action_dim`, `action_low`, `action_high`

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer

policy = MyPolicy()
server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=8000)
server.serve_forever()
```

---

## 3. Action Chunk Requirements

For "predict H steps, execute K steps" (e.g., predict 16, execute 8), you must:

1. Policy returns **chunk**: `{"actions": np.ndarray}` with shape `(H, action_dim)`
2. Wrap with `ActionChunkBroker`, set `action_horizon=K`
3. (Recommended) Use `ResetOnInitPolicy` to call `reset()` at episode start

```python
from policy_websocket import BasePolicy, ActionChunkBroker, WebsocketPolicyServer

class ChunkPolicy(BasePolicy):
    def infer(self, obs):
        return {"actions": model(obs)}  # shape (16, 7)

chunk_policy = ChunkPolicy()
broker = ActionChunkBroker(policy=chunk_policy, action_horizon=8)
policy = ResetOnInitPolicy(broker)
server = WebsocketPolicyServer(policy=policy, port=8000)
server.serve_forever()
```

---

## 4. Building Policy Server from PyTorch nn.Module

### 4.1 Single-Step Policy (inference per step)

```python
import torch
import numpy as np
from policy_websocket import BasePolicy, WebsocketPolicyServer

class TorchPolicy(BasePolicy):
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device
        self.model.eval()

    def infer(self, obs: dict) -> dict:
        primary = torch.from_numpy(obs["primary_image"]).float().permute(2, 0, 1)
        secondary = torch.from_numpy(obs["secondary_image"]).float().permute(2, 0, 1)
        wrist = torch.from_numpy(obs["wrist_image"]).float().permute(2, 0, 1)
        proprio = torch.from_numpy(obs["proprio"]).float()

        primary = primary.unsqueeze(0).to(self.device) / 255.0
        secondary = secondary.unsqueeze(0).to(self.device) / 255.0
        wrist = wrist.unsqueeze(0).to(self.device) / 255.0
        proprio = proprio.unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.model(primary, secondary, wrist, proprio)

        action = action.cpu().numpy().squeeze().astype(np.float64)
        if action.ndim == 0:
            action = np.array([float(action)])
        return {"actions": action}

    def reset(self) -> None:
        pass


if __name__ == "__main__":
    model = YourTorchModel().cuda().eval()
    policy = TorchPolicy(model)
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=8000,
        metadata={"policy_name": "YourModel", "action_dim": 7},
    )
    server.serve_forever()
```

### 4.2 Action Chunk Policy (predict multiple steps at once)

```python
class TorchChunkPolicy(BasePolicy):
    def __init__(self, model: torch.nn.Module, chunk_size: int = 16, device: str = "cuda"):
        self.model = model
        self.chunk_size = chunk_size
        self.device = device
        self.model.eval()

    def infer(self, obs: dict) -> dict:
        ...
        with torch.no_grad():
            actions = self.model(...)  # shape (batch, chunk_size, action_dim)
        actions = actions.cpu().numpy().squeeze().astype(np.float64)
        return {"actions": actions}

    def reset(self) -> None:
        pass


from policy_websocket import ActionChunkBroker

chunk_policy = TorchChunkPolicy(model, chunk_size=16)
broker = ActionChunkBroker(policy=chunk_policy, action_horizon=8)
policy = ResetOnInitPolicy(broker)
server = WebsocketPolicyServer(policy=policy, port=8000)
server.serve_forever()
```

### 4.3 ResetOnInitPolicy (Episode boundary reset)

The client does not explicitly send reset; a new episode is detected by "first infer has no images":

```python
class ResetOnInitPolicy(BasePolicy):
    def __init__(self, policy: BasePolicy):
        self._policy = policy

    def infer(self, obs: dict) -> dict:
        if "action_dim" in obs and "primary_image" not in obs:
            self._policy.reset()
        return self._policy.infer(obs)

    def reset(self) -> None:
        self._policy.reset()
```

---

## 5. Quick Setup Checklist

- [ ] Inherit `BasePolicy`, implement `infer(obs) -> {"actions": np.ndarray}`
- [ ] `actions` as `np.float64`, shape `(action_dim,)` or `(7,)`; for chunk: `(H, action_dim)`
- [ ] Handle first-infer `action_dim` / `action_low` / `action_high` (if needed)
- [ ] If using chunk: wrap with `ActionChunkBroker` and use `ResetOnInitPolicy`
- [ ] Start server with `WebsocketPolicyServer`

---

## 6. Full Template: Single-Step PyTorch Policy Server

```python
#!/usr/bin/env python3
"""Template: Wrap a PyTorch model as a Policy Server."""

import argparse
import numpy as np
import torch

from policy_websocket import BasePolicy, WebsocketPolicyServer


def load_model(checkpoint_path: str, device: str = "cuda") -> torch.nn.Module:
    model = ...
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.to(device)


class TorchPolicyAdapter(BasePolicy):
    def __init__(self, model: torch.nn.Module, device: str = "cuda"):
        self.model = model
        self.device = device

    def infer(self, obs: dict) -> dict:
        inp = self._preprocess(obs)
        with torch.no_grad():
            out = self.model(**inp)
        action = out.cpu().numpy().squeeze().astype(np.float64)
        if action.ndim == 0:
            action = np.array([float(action)])
        return {"actions": action}

    def _preprocess(self, obs: dict):
        raise NotImplementedError

    def reset(self) -> None:
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model = load_model(args.checkpoint, args.device)
    policy = TorchPolicyAdapter(model, args.device)
    server = WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata={"policy_name": "TorchModel", "action_dim": 7},
    )
    print(f"Serving on ws://0.0.0.0:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
```

---

## 7. Run and Test

```bash
# Terminal 1: Start Policy Server
python your_policy_server.py --port 8000

# Terminal 2: Use RoboCasa client
python scripts/run_demo.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab
# or
python scripts/run_eval.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab --num_trials 5
```

---

## 8. RoboCasa Reference Implementations

- `tests/test_random_policy_server.py` — Single-step random policy
- `tests/test_ac_policy_server.py` — Action Chunk (predict 16, execute 8) + ResetOnInitPolicy
