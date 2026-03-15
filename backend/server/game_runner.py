import asyncio
import logging
import random
import gymnasium as gym
import nle  # noqa: F401
import nle.nethack as nethack

from server.nle_event_converter import NLEEventConverter
from server.ws_manager import WSManager
from server.config import ServerConfig

logger = logging.getLogger(__name__)

DIR = nethack.actions.CompassDirection
MISC = nethack.actions.MiscDirection
CMD = nethack.actions.Command

SAFE_ACTIONS = [
    DIR.N, DIR.E, DIR.S, DIR.W,
    DIR.NE, DIR.SE, DIR.SW, DIR.NW,
    MISC.WAIT,
    CMD.SEARCH,
    CMD.OPEN,
    CMD.PICKUP,
    CMD.EAT,
    CMD.LOOK,
]


class GameRunner:
    """Run an NLE instance and broadcast RuntimeEvent-format messages."""

    def __init__(self, session_id: str, ws_manager: WSManager,
                 config: ServerConfig, env_id: str = "NetHackScore-v0"):
        self.session_id = session_id
        self.ws_manager = ws_manager
        self.config = config
        self.env_id = env_id
        self.converter = NLEEventConverter()
        self.env = None
        self.running = False
        self._task = None
        self._action_indices = None

    async def start(self):
        self.env = gym.make(self.env_id)
        self._build_action_index()
        self.running = True
        self._task = asyncio.create_task(self._run_loop())
        logger.info("Game started for session %s", self.session_id)

    def _build_action_index(self):
        """Map safe action codes to env action-space indices."""
        env_actions = list(self.env.unwrapped.actions)
        code_to_idx = {int(a): i for i, a in enumerate(env_actions)}
        self._action_indices = [
            code_to_idx[int(a)] for a in SAFE_ACTIONS if int(a) in code_to_idx
        ]
        if not self._action_indices:
            self._action_indices = list(range(len(env_actions)))

    def _sample_action(self):
        return random.choice(self._action_indices)

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
        self.converter.reset()

        await self._send_ready()
        events = self.converter.obs_to_events(obs, full=True)
        self.ws_manager.cache_full_state(self.session_id, events)
        await self._broadcast_events(events)

        step_count = 0
        while self.running:
            action = self._sample_action()
            obs, reward, done, truncated, info = self.env.step(action)
            step_count += 1

            is_full = (step_count % self.config.keyframe_interval == 0)
            events = self.converter.obs_to_events(obs, reward=reward, done=done, full=is_full)
            await self._broadcast_events(events)

            if done:
                obs, info = self.env.reset()
                self.converter.reset()
                events = self.converter.obs_to_events(obs, full=True)
                await self._broadcast_events(events)

            await asyncio.sleep(self.config.step_delay)

    async def _send_ready(self):
        await self.ws_manager.broadcast_to_spectators(
            self.session_id, {"type": "runtime_ready"})

    async def _broadcast_events(self, events: list[dict]):
        if not events:
            return
        if len(events) == 1:
            await self.ws_manager.broadcast_to_spectators(self.session_id, events[0])
        else:
            batch = {"type": "runtime_event_batch", "events": [
                e["event"] for e in events if e.get("type") == "runtime_event"
            ]}
            non_event = [e for e in events if e.get("type") != "runtime_event"]
            for ne in non_event:
                await self.ws_manager.broadcast_to_spectators(self.session_id, ne)
            if batch["events"]:
                await self.ws_manager.broadcast_to_spectators(self.session_id, batch)
