# policy-websocket

<!-- BANNER IMAGE: see image generation prompt at bottom of this file -->
<p align="center">
  <img src="./assets/banner.png" alt="policy-websocket banner" width="800" />
</p>

<p align="center">
  <em>Lightweight WebSocket bridge between robot policies and environments — serialize once, infer anywhere.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/policy-websocket"><img src="https://img.shields.io/pypi/v/policy-websocket" alt="PyPI" /></a>
  <a href="https://github.com/YufengJin/policy_websocket/blob/main/LICENSE"><img src="https://img.shields.io/github/license/YufengJin/policy_websocket" alt="License" /></a>
  <a href="https://python.org"><img src="https://img.shields.io/pypi/pyversions/policy-websocket" alt="Python" /></a>
</p>

## Why policy-websocket?

Running learned policies on a different machine from the robot controller is common — GPU server for inference, real-time PC for control. The typical options are heavyweight (ROS services, gRPC with protobuf schemas) or slow (REST + JSON serializing large NumPy arrays every step).

**policy-websocket** gives you a two-line integration with three design choices that matter for robot learning:

- **WebSocket** — persistent connection, no per-step HTTP overhead, ~0.3 ms round-trip on localhost
- **msgpack + NumPy** — binary serialization that handles multi-camera RGB-D observations without Base64 bloat
- **ActionChunkBroker** — built-in predict-N-execute-M pattern so chunk-based policies (ACT, Diffusion Policy, …) work out of the box

Minimal dependencies (`websockets`, `msgpack`, `numpy`), no framework lock-in.

## Architecture

<!-- ARCHITECTURE DIAGRAM: see image generation prompt at bottom of this file -->
<p align="center">
  <img src="./assets/architecture.png" alt="architecture diagram" width="700" />
</p>

```
┌─────────────────────┐         WebSocket / msgpack         ┌─────────────────────────────┐
│   Robot / Env        │  ──── obs dict ──────────────────▶  │  WebsocketPolicyServer      │
│                      │                                     │  ┌───────────────────────┐  │
│  WebsocketClient     │  ◀─── action dict ────────────────  │  │ BasePolicy.infer(obs) │  │
│  Policy.infer(obs)   │                                     │  └───────────────────────┘  │
└─────────────────────┘                                      │         or                   │
                                                             │  ┌───────────────────────┐  │
                                                             │  │ ActionChunkBroker     │  │
                                                             │  │ predict 16, execute 8 │  │
                                                             │  └───────────────────────┘  │
                                                             └─────────────────────────────┘
```

## Quick Start

```bash
pip install policy-websocket
```

```bash
# Terminal 1 — start server
python examples/policy_server.py --port 8000

# Terminal 2 — run client
python examples/policy_client.py --host localhost --port 8000 --steps 10
```

<details>
<summary>Install from source (development)</summary>

```bash
git clone https://github.com/YufengJin/policy_websocket.git && cd policy_websocket
pip install -e .
```

</details>

## Usage

### Server — wrap your policy

```python
from policy_websocket import BasePolicy, WebsocketPolicyServer
import numpy as np

class MyPolicy(BasePolicy):
    def infer(self, obs):
        return {"actions": np.zeros(7)}

server = WebsocketPolicyServer(policy=MyPolicy(), host="0.0.0.0", port=8000)
server.serve_forever()
```

### Client — two lines to connect

```python
from policy_websocket import WebsocketClientPolicy

policy = WebsocketClientPolicy(host="localhost", port=8000)
action = policy.infer(obs_dict)["actions"]
```

### ActionChunkBroker — predict N, execute M

```python
from policy_websocket import WebsocketPolicyServer, ActionChunkBroker

broker = ActionChunkBroker(policy=MyChunkPolicy(), action_horizon=8)
server = WebsocketPolicyServer(policy=broker, port=8000)
```

## Components

| Class | Role |
|-------|------|
| `BasePolicy` | Abstract interface — implement `infer(obs) → dict` and `reset()` |
| `WebsocketClientPolicy` | Drop-in policy that forwards obs to a remote server |
| `WebsocketPolicyServer` | Wraps any `BasePolicy`, serves over WebSocket |
| `ActionChunkBroker` | Buffers chunk predictions, yields one action per step |

## Performance

<!-- TODO: paste your stress_test.py results here -->
<!-- Example:
| Scenario | FPS | Latency (p50) | Latency (p99) |
|----------|-----|---------------|---------------|
| 3×RGB-D 720p, localhost | X | X ms | X ms |
| 3×RGB-D 720p, LAN | X | X ms | X ms |
-->

Benchmark: 3-view RGB-D 720p throughput and per-step latency (see [Policy Server Setup Guide](docs/policy_server.md) for stress test details).

## Protocol

| Aspect | Detail |
|--------|--------|
| Transport | WebSocket (persistent connection) |
| Serialization | msgpack with NumPy array support |
| Flow | Client sends obs dict → Server calls `policy.infer(obs)` → returns action dict |
| Health check | `GET /healthz` → 200 OK |

## Documentation

- [Policy Server Setup Guide](docs/policy_server.md) — PyTorch models, Action Chunk, RoboCasa integration
- [Module Reference](docs/policy_client.md) — API details, data flow
- [Examples](examples/) — [single-step server](examples/policy_server.py), [Action Chunk server](examples/policy_server_ac.py), [client](examples/policy_client.py)

Compatible with [openpi](https://github.com/Physical-Intelligence/openpi), [RoboCasa](https://github.com/robocasa/robocasa), and other robot learning environments.

## Contributing

Issues and PRs welcome. For development:

```bash
pip install -e .
python examples/policy_client.py --host localhost --port 8000 --steps 5  # with server running
```

## License

[MIT](LICENSE)

---

<!--
================================================================================
IMAGE GENERATION PROMPTS
================================================================================

1. BANNER IMAGE (assets/banner.png)
   - Recommended size: 1600×400 px (4:1)
   - Style: minimal, dark background (#0d1117 or similar dark navy/charcoal),
     clean tech aesthetic, no photorealism
   - Content: A simplified, iconic illustration showing two nodes connected by
     a glowing line/stream. Left node represents a robot arm (simple geometric
     silhouette), right node represents a GPU/server (circuit-board pattern or
     chip icon). Between them, a flowing data stream rendered as small dots or
     particles traveling along a WebSocket connection line. The stream should
     feel lightweight and fast.
   - Text: "policy-websocket" in a modern monospace font (e.g. JetBrains Mono
     style), centered or slightly right, white or light gray.
   - Color palette: dark background, accent color teal/cyan (#00d4aa) for the
     data stream and connection line, subtle gray (#8b949e) for secondary elements.
   - Do NOT include: photos of real robots, complex 3D renders, corporate logos,
     gradients that look like stock art.
   - Vibe: "developer tool for robotics" — clean like a GitHub README banner
     for a well-maintained open source project. Think: similar energy to
     FastAPI, Ruff, or uv project banners.

2. ARCHITECTURE DIAGRAM (assets/architecture.png)
   - Recommended size: 1400×600 px
   - Style: clean technical diagram on white or very light gray background,
     flat design, no 3D effects, no drop shadows.
   - Content: Reproduce the ASCII architecture in the README as a polished diagram:
     * Left box: "Robot / Environment" with a small robot arm icon, containing
       "WebsocketClientPolicy"
     * Right box: "GPU Server" with a chip icon, containing two sub-boxes stacked:
       "BasePolicy.infer(obs)" and "ActionChunkBroker (predict 16, execute 8)"
     * Between them: two horizontal arrows:
       - Top arrow (left→right): labeled "obs dict (msgpack + NumPy)"
       - Bottom arrow (right→left): labeled "action dict"
     * The arrows should have a WebSocket icon or "ws://" label near the middle
   - Typography: clean sans-serif (Inter, SF Pro, or similar), code elements in
     monospace. Labels should be readable at 50% zoom.
   - Color palette: white background, boxes with light gray (#f6f8fa) fill and
     thin border (#d0d7de), arrows in teal/cyan (#00d4aa) matching the banner,
     text in dark gray (#1f2328).
   - Do NOT include: complex UML notation, excessive detail, decorative elements.
   - Vibe: the kind of diagram you'd see in a well-written engineering blog post.
     Clear enough that someone unfamiliar with the project gets the idea in 5 seconds.

================================================================================
-->