# policy_websocket Module Reference

`policy_websocket` is a WebSocket policy client/server library for using a remote policy server as a drop-in replacement for a local policy. Compatible with robot environments such as openpi and RoboCasa.

## Installation

```bash
pip install git+https://github.com/YufengJin/policy_websocket.git
```

## Module Structure

```
policy_websocket/
├── base_policy.py           # Policy abstract base class
├── websocket_client.py      # WebSocket client policy
├── websocket_server.py      # WebSocket policy server
├── action_chunk_broker.py   # Action chunk broker
└── msgpack_numpy.py         # NumPy array msgpack serialization
```

---

## 1. base_policy.py — Policy Abstract Base Class

**Purpose**: Defines the unified interface for all policies; both local and remote policies must implement this interface.

**Core interface**:

| Method | Description |
|------|------|
| `infer(obs: Dict) -> Dict` | Given observation dict, returns action dict. **Must implement**. |
| `reset() -> None` | Resets internal state when a new episode starts. Default empty implementation. |

**Common `obs` fields**:

- `primary_image` / `secondary_image` / `wrist_image`: Camera images (H, W, 3)
- `proprio`: Proprioception (gripper, end-effector pose, etc.)
- `task_description`: Natural language task description
- `action_dim` / `action_low` / `action_high`: Provided by environment on first `infer`

**Return dict `action` must contain**:

- `actions`: `np.ndarray`, shape `(action_dim,)` or `(7,)` (7-dim is auto-padded)

---

## 2. websocket_client.py — WebSocket Client Policy

**Purpose**: Implements `BasePolicy` as a WebSocket client, communicating with a remote server by sending observations and receiving actions.

**Main logic**:

1. **Connect**: Build `ws://host:port`, retry until connection succeeds
2. **Handshake**: After connect, receive server `metadata` (e.g., `policy_name`, `action_dim`)
3. **Inference**: `infer(obs)` serializes and sends `obs`, deserializes response and returns

**Parameters**:

- `host`: Hostname or full `ws://...` URL
- `port`: Port (optional)
- `api_key`: Optional auth header

**Methods**:

- `infer(obs)`: Call remote policy inference
- `get_server_metadata()`: Returns metadata received during handshake
- `close()`: Close WebSocket connection
- `reset()`: Send `{"__command__": "reset"}` and block on ack. Call between episodes.

---

## 3. websocket_server.py — WebSocket Policy Server

**Purpose**: Wraps any `BasePolicy` as a WebSocket server for `WebsocketClientPolicy` to connect to.

**Main logic**:

1. **Start**: Bind to host:port, start WebSocket server
2. **Connection handling**: For each new connection, send `metadata` first, then enter request/response loop
3. **Dispatch loop**: For frames with `"__command__": "reset"` call `policy.reset()` (reply ack); otherwise call `policy.infer(obs)` → attach `server_timing` → reply action

**Parameters**:

- `policy`: Policy instance implementing `BasePolicy`
- `host`: Listen address, default `0.0.0.0`
- `port`: Listen port, default `8000`
- `metadata`: Dict sent to client during handshake

**Behavior**:

- Uses `SO_REUSEADDR`, port can be reused immediately after restart
- Graceful shutdown on SIGINT/SIGTERM
- `/healthz` returns HTTP 200
- On error: send traceback, close connection with `INTERNAL_ERROR`

---

## 4. action_chunk_broker.py — Action Chunk Broker

**Purpose**: Wraps a policy that returns `(H, action_dim)` action sequences into a `BasePolicy` that yields `(action_dim,)` per step, avoiding re-inference each step.

**Mechanism**:

- First `infer`: Call inner policy, get `actions` shape `(H, action_dim)`
- Later `infer`: Return `actions[step]` by step index
- When `step >= action_horizon`: Next `infer` re-calls inner policy
- `reset()`: Clears the cached chunk and forwards to inner policy

**Parameters**:

- `policy`: Inner policy (may return chunks)
- `action_horizon`: Action sequence length H

---

## 5. msgpack_numpy.py — NumPy Serialization

**Purpose**: Extends msgpack with encoding for NumPy arrays and scalars, used for WebSocket transport of observations and actions.

**Why msgpack**:

- Safe: No arbitrary code execution (vs pickle)
- Schema-free: Flexible (vs protobuf)
- Performance: ~4x faster than pickle on large arrays

---

## Usage Examples

### Custom Policy Server

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer
import numpy as np

class MyPolicy(BasePolicy):
    def infer(self, obs):
        return {"actions": np.zeros(7)}

server = WebsocketPolicyServer(policy=MyPolicy(), host="0.0.0.0", port=8000)
server.serve_forever()
```

### Client Usage

```python
from policy_websocket import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=8000)
action_dict = policy.infer(obs_dict)
action = action_dict["actions"]
```

---

## Data Flow Diagram

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
