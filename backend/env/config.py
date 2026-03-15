from dataclasses import dataclass
from typing import Optional


@dataclass
class EnvConfig:
    env_id: str = "NetHackScore-v0"
    initial_score: int = 1000
    death_penalty: int = 200
    max_deaths: Optional[int] = None
    max_steps: int = 100_000
