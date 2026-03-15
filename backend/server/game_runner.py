import asyncio
import logging
import gymnasium as gym
import nle  # noqa: F401 -- registers NLE envs

from server.state_serializer import StateSerializer
from server.ws_manager import WSManager
from server.config import ServerConfig

logger = logging.getLogger(__name__)


class GameRunner:
    """Run an NLE instance and broadcast state over WebSocket."""

    def __init__(self, session_id: str, ws_manager: WSManager,
                 config: ServerConfig, env_id: str = "NetHackScore-v0"):
        self.session_id = session_id
        self.ws_manager = ws_manager
        self.config = config
        self.env_id = env_id
        self.serializer = StateSerializer()
        self.env = None
        self.running = False
        self._task = None

    async def start(self):
        self.env = gym.make(self.env_id)
        self.running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Game started for session %s", self.session_id)

    async def stop(self):
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        if self.env:
            self.env.close()
            self.env = None
        logger.info("Game stopped for session %s", self.session_id)

    async def _run_loop(self):
        obs, info = self.env.reset()
        self.serializer.reset()

        full_state = self.serializer.serialize_full(obs)
        await self.ws_manager.broadcast_to_spectators(self.session_id, full_state)

        step_count = 0
        while self.running:
            action = self.env.action_space.sample()
            obs, reward, done, truncated, info = self.env.step(action)
            step_count += 1

            if step_count % self.config.keyframe_interval == 0:
                state = self.serializer.serialize_full(obs, reward, done)
            else:
                state = self.serializer.serialize_delta(obs, reward, done)

            await self.ws_manager.broadcast_to_spectators(self.session_id, state)

            if done:
                obs, info = self.env.reset()
                self.serializer.reset()
                full_state = self.serializer.serialize_full(obs)
                await self.ws_manager.broadcast_to_spectators(self.session_id, full_state)

            await asyncio.sleep(self.config.step_delay)
