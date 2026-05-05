"""WebSocket policy client — implements BasePolicy over a WebSocket connection.

Usage::

    from policy_websocket import WebsocketClientPolicy

    policy = WebsocketClientPolicy(host="localhost", port=8000)
    action = policy.infer(obs_dict)
"""

import logging
import time
from typing import Dict, Optional, Tuple

import websockets.sync.client

from policy_websocket import base_policy as _base_policy
from policy_websocket import msgpack_numpy

logger = logging.getLogger(__name__)


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Sends observations to a remote WebsocketPolicyServer and returns actions."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if host.startswith("ws"):
            self._uri = host
        else:
            self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"

        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        self._server_metadata: Dict = {}
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logger.info("Waiting for server at %s ...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                logger.info("Connected to server at %s", self._uri)
                return conn, metadata
            except ConnectionRefusedError:
                logger.info("Still waiting for server ...")
                time.sleep(2)

    def infer(self, obs: Dict) -> Dict:
        """Send an obs to the server, return the action dict."""
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from policy server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        """Close the WebSocket connection, releasing the port on both sides."""
        if self._ws is not None:
            try:
                self._ws.close()
            except Exception:
                pass
            self._ws = None

    def reset(self) -> None:
        """Send a reset request to the server; block until ack."""
        self._ws.send(self._packer.pack({"__command__": "reset"}))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error from policy server during reset:\n{response}")
        ack = msgpack_numpy.unpackb(response)
        if not (isinstance(ack, dict) and ack.get("ok") is True
                and ack.get("__command__") == "reset"):
            raise RuntimeError(f"Unexpected reset ack: {ack!r}")

    def __del__(self):
        self.close()
