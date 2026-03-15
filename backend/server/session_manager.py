"""Manage multiple concurrent NLE game sessions."""
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from env.scorable_env import ScorableNLEEnv, ScoreConfig
from server.nle_event_converter import NLEEventConverter
from memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)

MAX_SESSIONS = 20
IDLE_TIMEOUT_SECONDS = 30 * 60  # 30 minutes


class SessionStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FINISHED = "finished"


@dataclass
class GameSession:
    session_id: str
    agent_name: str
    agent_token: str
    status: SessionStatus = SessionStatus.IDLE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_active: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    env: Optional[ScorableNLEEnv] = field(default=None, repr=False)
    converter: Optional[NLEEventConverter] = field(default=None, repr=False)
    memory: Optional[MemorySystem] = field(default=None, repr=False)
    config: dict = field(default_factory=dict)
    total_steps: int = 0

    def touch(self):
        self.last_active = datetime.now(timezone.utc).isoformat()

    def is_idle_timeout(self) -> bool:
        last = datetime.fromisoformat(self.last_active)
        elapsed = (datetime.now(timezone.utc) - last).total_seconds()
        return elapsed > IDLE_TIMEOUT_SECONDS

    def summary(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_name": self.agent_name,
            "status": self.status.value,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "total_steps": self.total_steps,
            "config": self.config,
        }


class SessionManager:
    """Create, track, and clean up NLE game sessions."""

    def __init__(self, server_base_url: str = "http://localhost:8000"):
        self._sessions: dict[str, GameSession] = {}
        self._token_to_session: dict[str, str] = {}
        self.server_base_url = server_base_url.rstrip("/")

    def create_session(
        self,
        agent_name: str,
        config: Optional[dict] = None,
    ) -> dict:
        if len(self._sessions) >= MAX_SESSIONS:
            self._cleanup_idle()
            if len(self._sessions) >= MAX_SESSIONS:
                raise RuntimeError(
                    f"Maximum number of sessions ({MAX_SESSIONS}) reached"
                )

        session_id = str(uuid.uuid4())
        agent_token = secrets.token_urlsafe(32)
        config = config or {}

        score_cfg = ScoreConfig(
            initial_score=config.get("initial_score", 1000),
            death_penalty=config.get("death_penalty", 200),
            max_deaths=config.get("max_deaths"),
        )
        env = ScorableNLEEnv(
            config=score_cfg,
            env_id=config.get("env_id", "NetHackChallenge-v0"),
        )
        converter = NLEEventConverter()
        memory = MemorySystem()

        session = GameSession(
            session_id=session_id,
            agent_name=agent_name,
            agent_token=agent_token,
            env=env,
            converter=converter,
            memory=memory,
            config=config,
        )
        self._sessions[session_id] = session
        self._token_to_session[agent_token] = session_id

        spectate_url = f"{self.server_base_url}/ws/spectate/{session_id}"
        logger.info(
            "Session created: id=%s agent=%s", session_id, agent_name
        )
        return {
            "session_id": session_id,
            "agent_token": agent_token,
            "spectate_url": spectate_url,
        }

    def get_session(self, session_id: str) -> Optional[GameSession]:
        return self._sessions.get(session_id)

    def get_session_by_token(self, token: str) -> Optional[GameSession]:
        sid = self._token_to_session.get(token)
        if sid:
            return self._sessions.get(sid)
        return None

    def delete_session(self, session_id: str) -> bool:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        self._token_to_session.pop(session.agent_token, None)
        try:
            if session.env:
                session.env.close()
        except Exception as e:
            logger.warning("Error closing env for session %s: %s", session_id, e)
        session.status = SessionStatus.FINISHED
        logger.info("Session deleted: %s", session_id)
        return True

    def list_sessions(self) -> list[dict]:
        return [s.summary() for s in self._sessions.values()]

    def _cleanup_idle(self) -> int:
        idle_ids = [
            sid for sid, s in self._sessions.items() if s.is_idle_timeout()
        ]
        for sid in idle_ids:
            logger.info("Auto-cleaning idle session: %s", sid)
            self.delete_session(sid)
        return len(idle_ids)

    def cleanup_all(self):
        for sid in list(self._sessions.keys()):
            self.delete_session(sid)
        logger.info("All sessions cleaned up")
