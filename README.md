# policy-websocket

WebSocket-based policy client/server for robot learning. Provides a minimal, dependency-light interface for running policies remotely over WebSocket.

Compatible with [openpi](https://github.com/Physical-Intelligence/openpi), [RoboCasa](https://github.com/robocasa/robocasa), and other robot environments.

## Installation

```bash
pip install policy-websocket
```

## Components

| Class | Description |
|-------|-------------|
| `BasePolicy` | Abstract base: `infer(obs) -> dict`, `reset()` |
| `WebsocketClientPolicy` | Client that sends obs to a remote server, returns actions |
| `WebsocketPolicyServer` | Server that wraps any `BasePolicy` and serves over WebSocket |
| `ActionChunkBroker` | Wraps chunk-returning policies to yield one action per step |

## Usage

### Server (wrap your policy)

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer
import numpy as np

class MyPolicy(BasePolicy):
    def infer(self, obs):
        return {"actions": np.zeros(7)}

server = WebsocketPolicyServer(policy=MyPolicy(), host="0.0.0.0", port=8000)
server.serve_forever()
```

### Client

```python
from policy_websocket import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=8000)
action_dict = policy.infer(obs_dict)
action = action_dict["actions"]
```

### ActionChunkBroker (predict N, execute M)

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer, ActionChunkBroker

# Inner policy returns (16, action_dim) chunks
chunk_policy = MyChunkPolicy()
broker = ActionChunkBroker(policy=chunk_policy, action_horizon=8)
server = WebsocketPolicyServer(policy=broker, port=8000)
```

## Protocol

- **Transport**: WebSocket
- **Serialization**: msgpack with NumPy array support
- **Flow**: Client sends `obs` dict → Server calls `policy.infer(obs)` → Returns action dict
- **Health**: GET `/healthz` returns 200 OK

## Dependencies

- `websockets>=11.0`
- `msgpack>=1.0.5`
- `numpy>=1.22.4,<2.0.0`

## License

MIT
