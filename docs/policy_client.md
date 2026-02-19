# policy_websocket 模块说明

`policy_websocket` 是 WebSocket 策略客户端/服务端库，用于将远程策略服务器作为本地策略的替代。 与 openpi、robocasa 等机器人环境兼容。

## 安装

```bash
pip install git+https://github.com/YufengJin/policy_websocket.git
```

## 模块结构

```
policy_websocket/
├── base_policy.py           # 策略抽象基类
├── websocket_client.py      # WebSocket 客户端策略
├── websocket_server.py      # WebSocket 策略服务端
├── action_chunk_broker.py   # 动作块代理
└── msgpack_numpy.py         # NumPy 数组的 msgpack 序列化
```

---

## 1. base_policy.py — 策略抽象基类

**作用**：定义所有策略的统一接口，本地策略与远程策略均需实现此接口。

**核心接口**：

| 方法 | 说明 |
|------|------|
| `infer(obs: Dict) -> Dict` | 给定观测字典，返回动作字典。**必须实现**。 |
| `reset() -> None` | 新 episode 开始时重置内部状态。默认空实现。 |

**观测 `obs` 常用字段**：

- `primary_image` / `secondary_image` / `wrist_image`：相机图像 (H, W, 3)
- `proprio`： proprioception（夹爪、末端位姿等）
- `task_description`：任务自然语言描述
- `action_dim` / `action_low` / `action_high`：首次 `infer` 时由环境提供

**返回字典 `action` 需包含**：

- `actions`：`np.ndarray`， shape `(action_dim,)` 或 `(7,)`（7 维会被自动 pad）

---

## 2. websocket_client.py — WebSocket 客户端策略

**作用**：将 `BasePolicy` 实现为 WebSocket 客户端，通过发送观测并接收动作与远程服务通信。

**主要逻辑**：

1. **连接**：构造 `ws://host:port`，循环等待连接成功
2. **握手**：连接后接收服务端 `metadata`（如 `policy_name`, `action_dim`）
3. **推理**：`infer(obs)` 将 `obs` 序列化、发送，接收响应后反序列化并返回

**参数**：

- `host`：主机或完整 `ws://...` URL
- `port`：端口（可选）
- `api_key`：可选认证头

**方法**：

- `infer(obs)`：调用远程策略推理
- `get_server_metadata()`：返回握手阶段收到的 metadata
- `close()`：关闭 WebSocket 连接
- `reset()`：当前为空实现

---

## 3. websocket_server.py — WebSocket 策略服务端

**作用**：将任意 `BasePolicy` 封装为 WebSocket 服务端，供 `WebsocketClientPolicy` 连接使用。

**主要逻辑**：

1. **启动**：绑定指定 host:port，开启 WebSocket 服务
2. **连接处理**：每个新连接先发送 `metadata`，再进入请求/响应循环
3. **推理循环**：接收观测 → 调用 `policy.infer(obs)` → 附加 `server_timing` → 返回动作

**参数**：

- `policy`：实现 `BasePolicy` 的策略实例
- `host`：监听地址，默认 `0.0.0.0`
- `port`：监听端口，默认 `8000`
- `metadata`：握手时发给客户端的字典

**行为**：

- 使用 `SO_REUSEADDR`，重启后端口可立即复用
- 支持 SIGINT/SIGTERM 优雅退出
- `/healthz` 返回 HTTP 200
- 异常时发送 traceback，并以 `INTERNAL_ERROR` 关闭连接

---

## 4. action_chunk_broker.py — 动作块代理

**作用**：将返回 `(H, action_dim)` 动作序列的策略包装成按步返回 `(action_dim,)` 的 `BasePolicy`，避免每步重新推理。

**机制**：

- 首次 `infer`：调用内部策略，得到 `actions` shape `(H, action_dim)`
- 后续 `infer`：按步索引取出 `actions[step]` 返回
- 当 `step >= action_horizon`：下次 `infer` 再重新调用内部策略

**参数**：

- `policy`：内部策略（可能返回 chunk）
- `action_horizon`：动作序列长度 H

---

## 5. msgpack_numpy.py — NumPy 序列化

**作用**：在 msgpack 基础上增加对 NumPy 数组和标量的编码，用于 WebSocket 传输观测和动作。

**选择 msgpack 的原因**：

- 安全：无任意代码执行（相较 pickle）
- 无 schema：灵活（相较 protobuf）
- 性能：大数组下约比 pickle 快 4 倍

---

## 使用示例

### 自定义策略服务端

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer
import numpy as np

class MyPolicy(BasePolicy):
    def infer(self, obs):
        return {"actions": np.zeros(7)}

server = WebsocketPolicyServer(policy=MyPolicy(), host="0.0.0.0", port=8000)
server.serve_forever()
```

### 客户端使用

```python
from policy_websocket import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=8000)
action_dict = policy.infer(obs_dict)
action = action_dict["actions"]
```

---

## 数据流示意

```
┌─────────────────┐     WebSocket (msgpack)     ┌─────────────────────┐
│  Client         │  obs (images, proprio,     │  WebsocketPolicy    │
│  (env/runtime)  │  task_desc, action_spec)   │  Server             │
│                 │ ────────────────────────► │                     │
│  Websocket      │                             │  policy.infer(obs)  │
│  ClientPolicy   │  action (actions, timing)   │  (BasePolicy)       │
│                 │ ◄────────────────────────  │                     │
└─────────────────┘                             └─────────────────────┘
```
