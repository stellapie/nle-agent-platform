"""Python SDK for connecting external agents to the NLE Agent Platform."""
import asyncio
import json
import logging
from typing import Any, Optional

import httpx
import websockets

logger = logging.getLogger(__name__)


class NLEAgentClient:
    """Async client for interacting with an NLE game session."""

    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        agent_name: str = "sdk-agent",
        config: Optional[dict] = None,
    ):
        self.server_url = server_url.rstrip("/")
        self.agent_name = agent_name
        self.config = config or {}

        self._session_id: Optional[str] = None
        self._agent_token: Optional[str] = None
        self._spectate_url: Optional[str] = None
        self._ws: Optional[websockets.WebSocketClientProtocol] = None
        self._http: Optional[httpx.AsyncClient] = None

    @property
    def session_id(self) -> Optional[str]:
        return self._session_id

    @property
    def spectate_url(self) -> Optional[str]:
        return self._spectate_url

    async def connect(self) -> str:
        """Create a session via REST and connect the agent WebSocket.

        Returns the spectate URL for browser viewing.
        """
        self._http = httpx.AsyncClient(base_url=self.server_url, timeout=30.0)

        resp = await self._http.post("/api/sessions", json={
            "agent_name": self.agent_name,
            "config": self.config,
        })
        resp.raise_for_status()
        data = resp.json()

        self._session_id = data["session_id"]
        self._agent_token = data["agent_token"]
        self._spectate_url = data["spectate_url"]

        ws_scheme = "wss" if self.server_url.startswith("https") else "ws"
        ws_host = self.server_url.replace("http://", "").replace("https://", "")
        ws_url = f"{ws_scheme}://{ws_host}/ws/agent/{self._session_id}?token={self._agent_token}"

        self._ws = await websockets.connect(ws_url)
        logger.info(
            "Connected to session %s (spectate: %s)",
            self._session_id, self._spectate_url,
        )
        return self._spectate_url

    async def get_observation(self) -> dict:
        """Wait for and return the next observation from the server."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        raw = await self._ws.recv()
        return json.loads(raw)

    async def send_action(self, action: str) -> dict:
        """Send a single text action and wait for the resulting observation."""
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        await self._ws.send(json.dumps({
            "type": "action",
            "action": action,
        }))
        raw = await self._ws.recv()
        return json.loads(raw)

    async def send_plan(self, plan: dict) -> dict:
        """Send a multi-action plan and wait for the aggregated result.

        Args:
            plan: dict with "actions" key containing a list of text actions,
                  and optionally "reasoning" with the plan rationale.
        """
        if self._ws is None:
            raise RuntimeError("Not connected. Call connect() first.")
        await self._ws.send(json.dumps({
            "type": "plan",
            "plan": plan,
        }))
        raw = await self._ws.recv()
        return json.loads(raw)

    async def disconnect(self):
        """Close WebSocket and optionally delete the session."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None

        if self._http and self._session_id:
            try:
                await self._http.delete(f"/api/sessions/{self._session_id}")
            except Exception as e:
                logger.warning("Failed to delete session: %s", e)

        if self._http:
            await self._http.aclose()
            self._http = None

        logger.info("Disconnected from session %s", self._session_id)
        self._session_id = None
        self._agent_token = None

    async def __aenter__(self):
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
        return False
