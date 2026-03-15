import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    OBSERVATION = "observation"
    ACTION = "action"
    PLAN = "plan"
    DEATH = "death"
    REFLECTION = "reflection"


class NoteCategory(Enum):
    STRATEGY = "strategy"
    PATTERN = "pattern"
    RISK = "risk"
    ITEM = "item"


@dataclass
class EpisodeMemory:
    content: str
    event_type: str = "observation"
    importance: int = 5
    step_number: int = 0
    tags: list[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class NoteMemory:
    content: str
    category: str = "strategy"
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    source_episodes: list[int] = field(default_factory=list)


REFLECTION_SYSTEM_PROMPT = """You are a reflective analyst for a NetHack game agent.
Based on recent game episodes, derive useful insights.
Respond with a JSON array of objects, each with keys: content (string), category (one of: strategy, pattern, risk, item), confidence (float 0-1)."""

REFLECTION_PROMPT_TEMPLATE = """Recent episodes:
{episodes}

Existing notes:
{notes}

Analyze these episodes and generate 2-5 useful insights as a JSON array."""


class MemorySystem:
    REFLECTION_EPISODE_THRESHOLD = 15
    REFLECTION_IMPORTANCE_THRESHOLD = 50

    def __init__(self, llm: Optional[object] = None):
        self.episodes: list[EpisodeMemory] = []
        self.notes: list[NoteMemory] = []
        self.llm = llm
        self._importance_since_reflection = 0
        self._episodes_since_reflection = 0

    def record_observation(
        self,
        content: str,
        importance: int = 5,
        step_number: int = 0,
        tags: Optional[list[str]] = None,
    ) -> EpisodeMemory:
        ep = EpisodeMemory(
            content=content,
            event_type=EventType.OBSERVATION.value,
            importance=min(max(importance, 1), 10),
            step_number=step_number,
            tags=tags or [],
        )
        self.episodes.append(ep)
        self._importance_since_reflection += ep.importance
        self._episodes_since_reflection += 1
        return ep

    def record_plan(self, content: str, step_number: int = 0) -> EpisodeMemory:
        ep = EpisodeMemory(
            content=content,
            event_type=EventType.PLAN.value,
            importance=4,
            step_number=step_number,
        )
        self.episodes.append(ep)
        self._episodes_since_reflection += 1
        self._importance_since_reflection += ep.importance
        return ep

    def record_death(self, content: str, step_number: int = 0) -> EpisodeMemory:
        ep = EpisodeMemory(
            content=content,
            event_type=EventType.DEATH.value,
            importance=10,
            step_number=step_number,
            tags=["death"],
        )
        self.episodes.append(ep)
        self._importance_since_reflection += ep.importance
        self._episodes_since_reflection += 1
        return ep

    def retrieve(self, n: int = 10, event_type: Optional[str] = None) -> list[EpisodeMemory]:
        episodes = self.episodes
        if event_type:
            episodes = [e for e in episodes if e.event_type == event_type]
        return episodes[-n:]

    def get_recent_reflections(self, n: int = 5) -> list[NoteMemory]:
        return self.notes[-n:]

    def should_reflect(self) -> bool:
        return (
            self._episodes_since_reflection >= self.REFLECTION_EPISODE_THRESHOLD
            or self._importance_since_reflection >= self.REFLECTION_IMPORTANCE_THRESHOLD
        )

    async def generate_reflections(self) -> list[NoteMemory]:
        if self.llm is None:
            logger.warning("No LLM configured for reflection")
            return []

        recent = self.retrieve(n=20)
        episodes_text = "\n".join(
            f"[step {e.step_number}] ({e.event_type}) {e.content}" for e in recent
        )
        notes_text = "\n".join(
            f"[{n.category}] {n.content} (confidence={n.confidence})" for n in self.notes[-10:]
        )

        prompt = REFLECTION_PROMPT_TEMPLATE.format(episodes=episodes_text, notes=notes_text)

        try:
            data = await self.llm.generate_json(prompt, system_prompt=REFLECTION_SYSTEM_PROMPT)
            if isinstance(data, dict):
                data = data.get("insights", [data])

            new_notes = []
            for item in data:
                note = NoteMemory(
                    content=item.get("content", ""),
                    category=item.get("category", "strategy"),
                    confidence=float(item.get("confidence", 0.5)),
                    source_episodes=[e.step_number for e in recent[-5:]],
                )
                self.notes.append(note)
                new_notes.append(note)

            self._episodes_since_reflection = 0
            self._importance_since_reflection = 0
            logger.info("Generated %d reflections", len(new_notes))
            return new_notes
        except Exception as e:
            logger.error("Reflection generation failed: %s", e)
            return []

    def save(self, filepath: str) -> None:
        data = {
            "episodes": [asdict(e) for e in self.episodes],
            "notes": [asdict(n) for n in self.notes],
            "episodes_since_reflection": self._episodes_since_reflection,
            "importance_since_reflection": self._importance_since_reflection,
        }
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Memory saved to %s (%d episodes, %d notes)", filepath, len(self.episodes), len(self.notes))

    def load(self, filepath: str) -> None:
        with open(filepath, "r") as f:
            data = json.load(f)

        self.episodes = [
            EpisodeMemory(**ep) for ep in data.get("episodes", [])
        ]
        self.notes = [
            NoteMemory(**n) for n in data.get("notes", [])
        ]
        self._episodes_since_reflection = data.get("episodes_since_reflection", 0)
        self._importance_since_reflection = data.get("importance_since_reflection", 0)
        logger.info("Memory loaded from %s (%d episodes, %d notes)", filepath, len(self.episodes), len(self.notes))
