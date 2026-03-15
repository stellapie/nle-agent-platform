import asyncio
import logging
from typing import Dict, Set
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class WSManager:
    """Manage WebSocket connections grouped by session."""

    def __init__(self):
        self._spectators: Dict[str, Set[WebSocket]] = {}
        self._agent_ws: Dict[str, WebSocket] = {}

    async def add_spectator(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self._spectators.setdefault(session_id, set()).add(ws)
        logger.info("Spectator connected to session %s (total: %d)",
                    session_id, len(self._spectators[session_id]))

    def remove_spectator(self, session_id: str, ws: WebSocket):
        if session_id in self._spectators:
            self._spectators[session_id].discard(ws)
            if not self._spectators[session_id]:
                del self._spectators[session_id]

    async def set_agent_ws(self, session_id: str, ws: WebSocket):
        await ws.accept()
        self._agent_ws[session_id] = ws

    def remove_agent_ws(self, session_id: str):
        self._agent_ws.pop(session_id, None)

    async def broadcast_to_spectators(self, session_id: str, data: dict):
        spectators = self._spectators.get(session_id, set()).copy()
        dead = []
        for ws in spectators:
            try:
                await ws.send_json(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.remove_spectator(session_id, ws)

    def spectator_count(self, session_id: str) -> int:
        return len(self._spectators.get(session_id, set()))
