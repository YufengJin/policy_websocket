"""WebSocket policy server — serves a BasePolicy over WebSocket.

Each client frame is either a control frame (top-level ``"__command__"`` key)
or an inference frame (an obs dict). See ``_handler`` for the dispatch.

Usage::

    from policy_websocket import WebsocketPolicyServer

    server = WebsocketPolicyServer(policy=my_policy, port=8000)
    server.serve_forever()
"""

import asyncio
import http
import logging
import signal
import socket
import time
import traceback
from typing import Dict, Optional

import websockets.asyncio.server as _server
import websockets.frames

from policy_websocket import base_policy as _base_policy
from policy_websocket import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a BasePolicy via WebSocket.  Pairs with WebsocketClientPolicy."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int = 8000,
        metadata: Optional[Dict] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        self._server: Optional[_server.Server] = None
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        try:
            asyncio.run(self._run())
        except KeyboardInterrupt:
            logger.info("Server interrupted, shutting down")
        finally:
            self._server = None

    async def _run(self):
        loop = asyncio.get_running_loop()

        # Graceful shutdown on SIGINT / SIGTERM
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, lambda s=sig: asyncio.ensure_future(self._shutdown(s)))

        # SO_REUSEADDR so the port is immediately available after restart
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((self._host, self._port))
        sock.listen()
        sock.setblocking(False)

        async with _server.serve(
            self._handler,
            sock=sock,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            self._server = server
            logger.info("PolicyServer listening on ws://%s:%s", self._host, self._port)
            await server.serve_forever()

    async def _shutdown(self, sig):
        logger.info("Received signal %s, shutting down ...", sig.name)
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task():
                task.cancel()
        asyncio.get_running_loop().stop()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("Connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                msg = msgpack_numpy.unpackb(await websocket.recv())

                # Control frames bypass policy.infer().
                if isinstance(msg, dict) and "__command__" in msg:
                    cmd = msg["__command__"]
                    if cmd == "reset":
                        self._policy.reset()
                        await websocket.send(packer.pack({"__command__": "reset", "ok": True}))
                    else:
                        await websocket.send(packer.pack({
                            "__command__": cmd, "ok": False,
                            "error": f"unknown command: {cmd!r}",
                        }))
                    continue

                obs = msg
                infer_time = time.monotonic()
                action = self._policy.infer(obs)
                infer_time = time.monotonic() - infer_time

                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("Connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request):
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None
