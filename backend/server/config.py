import os
from dataclasses import dataclass


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    step_delay: float = 0.2
    keyframe_interval: int = 50
    max_spectators_per_session: int = 10
    cors_origins: list = None

    def __post_init__(self):
        self.host = os.getenv("NLE_HOST", self.host)
        self.port = int(os.getenv("NLE_PORT", self.port))
        self.step_delay = float(os.getenv("NLE_STEP_DELAY", self.step_delay))
        if self.cors_origins is None:
            self.cors_origins = ["*"]
