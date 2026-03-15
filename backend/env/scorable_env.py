"""Score-based NLE environment wrapper with death/respawn mechanics."""
import gymnasium as gym
import nle  # noqa: F401
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScoreConfig:
    initial_score: int = 1000
    death_penalty: int = 200
    kill_bonus_multiplier: float = 1.0
    exploration_bonus: float = 0.5
    gold_bonus_multiplier: float = 0.1
    max_deaths: Optional[int] = None


class ScorableNLEEnv:
    """NLE with persistent scoring across deaths instead of permadeath."""

    def __init__(self, config: ScoreConfig = None,
                 env_id: str = "NetHackChallenge-v0", **env_kwargs):
        self.config = config or ScoreConfig()
        self.env = gym.make(env_id, **env_kwargs)
        self.total_score = self.config.initial_score
        self.deaths = 0
        self.total_steps = 0
        self.current_episode_steps = 0
        self.episode_count = 0
        self.death_history: list[dict] = []
        self.last_obs = None
        self.last_info = None

    def reset(self):
        obs, info = self.env.reset()
        self.total_score = self.config.initial_score
        self.deaths = 0
        self.total_steps = 0
        self.current_episode_steps = 0
        self.episode_count = 0
        self.death_history = []
        self.last_obs = obs
        info.update(self._get_meta())
        self.last_info = info
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.total_steps += 1
        self.current_episode_steps += 1
        self.total_score += reward
        self.last_obs = obs

        if done:
            import numpy as np
            msg = obs.get("message", b"")
            if isinstance(msg, np.ndarray):
                msg = bytes(msg).decode("ascii", errors="ignore").strip("\x00").strip()
            elif isinstance(msg, bytes):
                msg = msg.decode("ascii", errors="ignore").strip()
            else:
                msg = str(msg)

            self.death_history.append({
                "death_number": self.deaths + 1,
                "episode_steps": self.current_episode_steps,
                "total_steps": self.total_steps,
                "score_at_death": self.total_score,
                "last_message": msg,
            })
            self.deaths += 1
            self.total_score -= self.config.death_penalty

            game_over = self.total_score <= 0
            if self.config.max_deaths and self.deaths >= self.config.max_deaths:
                game_over = True

            if not game_over:
                obs, env_info = self.env.reset()
                self.episode_count += 1
                self.current_episode_steps = 0
                done = False
                self.last_obs = obs
                info.update(env_info)
            else:
                done = True

        info.update(self._get_meta())
        self.last_info = info
        return obs, reward, done, truncated, info

    def _get_meta(self) -> dict:
        return {
            "total_score": self.total_score,
            "deaths": self.deaths,
            "total_steps": self.total_steps,
            "episode_count": self.episode_count,
            "current_episode_steps": self.current_episode_steps,
            "death_history": self.death_history,
            "game_over_reason": "score_depleted" if self.total_score <= 0 else None,
        }

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()
