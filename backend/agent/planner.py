import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Callable

from agent.llm_interface import LLMInterface

try:
    from agent.action_translator import NLEActionTranslator
except ImportError:
    NLEActionTranslator = None

logger = logging.getLogger(__name__)

AVAILABLE_ACTIONS = [
    "move north", "move south", "move east", "move west",
    "move northeast", "move northwest", "move southeast", "move southwest",
    "kick", "open", "close", "search", "look", "eat",
    "drink", "read", "wear", "wield", "take off", "put on",
    "drop", "throw", "zap", "cast", "pray", "apply",
    "inventory", "pick up", "go up stairs", "go down stairs",
    "wait", "engrave",
]

if NLEActionTranslator is not None:
    try:
        _translator = NLEActionTranslator()
        AVAILABLE_ACTIONS = _translator.available_actions()
    except Exception:
        pass


class InterruptType(Enum):
    LOW_HP = "low_hp"
    DEATH = "death"
    ENEMY_SPOTTED = "enemy_spotted"
    ITEM_FOUND = "item_found"
    CUSTOM = "custom"


@dataclass
class InterruptCondition:
    interrupt_type: InterruptType
    check_fn: Callable[..., bool]
    priority: int = 0
    description: str = ""


@dataclass
class Plan:
    goal: str
    reasoning: str
    steps: list[str] = field(default_factory=list)
    max_steps: int = 10
    steps_executed: int = 0

    @property
    def current_step(self) -> Optional[str]:
        if self.steps_executed < len(self.steps):
            return self.steps[self.steps_executed]
        return None

    @property
    def is_complete(self) -> bool:
        return self.steps_executed >= len(self.steps)


@dataclass
class ExecutionResult:
    plan: Plan
    completed: bool = False
    interrupted: bool = False
    interrupt_reason: str = ""
    final_observation: str = ""
    steps_taken: int = 0
    total_reward: float = 0.0


PLANNER_SYSTEM_PROMPT = """You are a strategic planner for a NetHack agent.
Given the current game state, generate a short-term plan as a JSON object.
Respond ONLY with valid JSON, no other text."""

PLANNER_PROMPT_TEMPLATE = """Current observation:
{observation}

Recent memories:
{memories}

Reflections / notes:
{reflections}

Game stats: score={score}, deaths={deaths}

Available actions:
{actions}

Generate a plan as JSON with keys: goal (string), reasoning (string), steps (list of action strings, max {max_steps})."""


class Planner:
    def __init__(self, llm: LLMInterface, max_steps: int = 8):
        self.llm = llm
        self.max_steps = max_steps

    async def generate_plan(
        self,
        observation: str,
        memories: str = "",
        reflections: str = "",
        score: int = 0,
        deaths: int = 0,
    ) -> Plan:
        actions_str = ", ".join(AVAILABLE_ACTIONS)
        prompt = PLANNER_PROMPT_TEMPLATE.format(
            observation=observation,
            memories=memories,
            reflections=reflections,
            score=score,
            deaths=deaths,
            actions=actions_str,
            max_steps=self.max_steps,
        )

        try:
            data = await self.llm.generate_json(prompt, system_prompt=PLANNER_SYSTEM_PROMPT)
            plan = Plan(
                goal=data.get("goal", "explore"),
                reasoning=data.get("reasoning", ""),
                steps=data.get("steps", ["look", "move north"]),
                max_steps=self.max_steps,
            )
            logger.info("Plan generated: goal=%s, steps=%d", plan.goal, len(plan.steps))
            return plan
        except Exception as e:
            logger.warning("Plan generation failed (%s), using fallback", e)
            return self._fallback_plan()

    def _fallback_plan(self) -> Plan:
        return Plan(
            goal="explore surroundings",
            reasoning="LLM planning failed; defaulting to basic exploration.",
            steps=["look", "move north", "look", "move east"],
            max_steps=self.max_steps,
        )
