# Policy Server 搭建指南

本文档说明如何将任意策略（含 PyTorch `nn.Module`）封装为 Policy Server，支持单步输出和 Action Chunk。可与 RoboCasa `run_demo`/`run_eval`、openpi 等客户端对接。

---

## 1. 前置条件：Policy 需满足的接口

### 1.1 BasePolicy 接口

任何要作为 Policy Server 的策略必须实现 `BasePolicy`：

```python
from policy_websocket import BasePolicy

class MyPolicy(BasePolicy):
    def infer(self, obs: Dict) -> Dict:
        """观测 → 动作字典。必须实现。"""
        ...
        return {"actions": action_array}

    def reset(self) -> None:
        """新 episode 开始时的重置。可选，默认 pass。"""
        pass
```

### 1.2 观测 `obs` 格式（客户端传入）

| 字段 | 类型 | 说明 |
|------|------|------|
| `primary_image` | `np.ndarray` (H,W,3) | 主相机 RGB |
| `secondary_image` | `np.ndarray` (H,W,3) | 副相机 RGB |
| `wrist_image` | `np.ndarray` (H,W,3) | 腕部相机 RGB |
| `proprio` | `np.ndarray` (D,) | 本体感知（夹爪、末端位姿等） |
| `task_description` | `str` | 任务自然语言描述 |
| `action_dim` / `action_low` / `action_high` | 首次 infer 时 | Episode 初始化用的 action spec |

首次 infer 只有 `action_dim`, `action_low`, `action_high`, `task_name`, `task_description`，没有图像。

### 1.3 返回 `action` 格式

| 字段 | 类型 | 说明 |
|------|------|------|
| `actions` | `np.ndarray` | `(action_dim,)` 或 `(7,)`。7 维时客户端会自动 pad 到 env 维度 |

---

## 2. Policy Server 的条件

满足以下即可启动 Policy Server：

1. 实现 `BasePolicy` 的 `infer(obs) -> Dict`，返回 `{"actions": np.ndarray}`
2. 返回值中 `actions` 为 `np.float64`，shape `(action_dim,)` 或 `(7,)`
3. （推荐）处理首次 infer 的 action spec：`action_dim`, `action_low`, `action_high`

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer

policy = MyPolicy()
server = WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=8000)
server.serve_forever()
```

---

## 3. Action Chunk 的条件

若希望「预测 H 步，执行 K 步」（例如 predict 16, execute 8），需满足：

1. 策略返回 **chunk**：`{"actions": np.ndarray}`，shape 为 `(H, action_dim)`
2. 用 `ActionChunkBroker` 包装，设置 `action_horizon=K`
3. （推荐）用 `ResetOnInitPolicy` 在 episode 开始时调用 `reset()`

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

## 4. 从 PyTorch nn.Module 搭建 Policy Server

### 4.1 单步策略（每步推理）

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

### 4.2 Action Chunk 策略（一次预测多步）

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

### 4.3 ResetOnInitPolicy（Episode 边界 reset）

客户端不会显式发送 reset，通过「首次 infer 没有图像」判断新 episode：

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

## 5. 快速搭建清单

- [ ] 继承 `BasePolicy`，实现 `infer(obs) -> {"actions": np.ndarray}`
- [ ] `actions` 为 `np.float64`，shape `(action_dim,)` 或 `(7,)`；chunk 时为 `(H, action_dim)`
- [ ] 首次 infer 处理 `action_dim` / `action_low` / `action_high`（如需）
- [ ] 若用 chunk：用 `ActionChunkBroker` 包装，并搭配 `ResetOnInitPolicy`
- [ ] 用 `WebsocketPolicyServer` 启动服务

---

## 6. 完整模板：单步 PyTorch Policy Server

```python
#!/usr/bin/env python3
"""模板：将 PyTorch 模型封装为 Policy Server。"""

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

## 7. 运行与测试

```bash
# 终端 1：启动 Policy Server
python your_policy_server.py --port 8000

# 终端 2：使用 RoboCasa 客户端
python scripts/run_demo.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab
# 或
python scripts/run_eval.py --policy_server_addr localhost:8000 --task_name PnPCounterToCab --num_trials 5
```

---

## 8. RoboCasa 参考实现

- `tests/test_random_policy_server.py` — 单步随机策略
- `tests/test_ac_policy_server.py` — Action Chunk（predict 16, execute 8）+ ResetOnInitPolicy
