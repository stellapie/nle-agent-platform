import logging
from typing import Any, Optional

from agent.llm_interface import LLMInterface
from agent.planner import Planner, ExecutionResult
from agent.executor import PlanExecutor
from agent.text_observer import NLETextObserver
from agent.action_translator import NLEActionTranslator
from memory.memory_system import MemorySystem

logger = logging.getLogger(__name__)


class AgentController:
    def __init__(
        self,
        llm: LLMInterface,
        env: Any,
        memory: MemorySystem,
        action_translator: Any = None,
        text_observer: Any = None,
        max_plans: int = 50,
        max_steps_per_plan: int = 8,
    ):
        self.llm = llm
        self.env = env
        self.memory = memory
        self.max_plans = max_plans

        if text_observer is None:
            text_observer = NLETextObserver()
        self.text_observer = text_observer

        if action_translator is None:
            env_actions = getattr(getattr(env, "env", None), "unwrapped", env)
            try:
                action_translator = NLEActionTranslator(
                    env_actions=env_actions.actions if hasattr(env_actions, "actions") else None
                )
            except Exception:
                action_translator = NLEActionTranslator()
        self.action_translator = action_translator

        self.planner = Planner(llm, max_steps=max_steps_per_plan)
        self.executor = PlanExecutor(
            env=env,
            action_translator=action_translator,
            memory=memory,
            observation_formatter=self._format_observation,
        )

        self._total_reward = 0.0
        self._total_steps = 0
        self._deaths = 0
        self._game_over = False

    def _format_observation(self, obs: Any) -> str:
        if isinstance(obs, dict) and "glyphs" in obs:
            meta = None
            if hasattr(self.env, "_get_meta"):
                meta = self.env._get_meta()
            return self.text_observer.observe_as_prompt(obs, meta=meta)
        return str(obs)

    def _format_memories_for_prompt(self, n: int = 10) -> str:
        recent = self.memory.retrieve(n=n)
        if not recent:
            return "No memories yet."
        return "\n".join(
            f"[step {e.step_number}] {e.content}" for e in recent
        )

    def _format_reflections_for_prompt(self, n: int = 5) -> str:
        notes = self.memory.get_recent_reflections(n=n)
        if not notes:
            return "No reflections yet."
        return "\n".join(
            f"[{n.category}] {n.content}" for n in notes
        )

    async def run(self, initial_observation: Optional[str] = None) -> dict:
        obs, info = self.env.reset()
        obs_text = initial_observation or self._format_observation(obs)

        logger.info("=== Agent Controller started ===")
        self.memory.record_observation(
            content=f"Game started. Initial observation: {obs_text[:200]}",
            importance=6,
            step_number=0,
        )

        for plan_idx in range(self.max_plans):
            if self._game_over:
                logger.info("Game over. Stopping.")
                break

            # Sync score/deaths from ScorableNLEEnv if available
            if hasattr(self.env, "total_score"):
                self._total_reward = self.env.total_score
                self._deaths = self.env.deaths

            memories_text = self._format_memories_for_prompt()
            reflections_text = self._format_reflections_for_prompt()

            plan = await self.planner.generate_plan(
                observation=obs_text,
                memories=memories_text,
                reflections=reflections_text,
                score=int(self._total_reward),
                deaths=self._deaths,
            )

            self.memory.record_plan(
                content=f"Plan #{plan_idx + 1}: {plan.goal} - Steps: {plan.steps}",
                step_number=self._total_steps,
            )

            logger.info(
                "Plan #%d: goal=%s, steps=%d",
                plan_idx + 1, plan.goal, len(plan.steps),
            )

            result: ExecutionResult = await self.executor.execute(plan)

            self._total_steps += result.steps_taken
            obs_text = result.final_observation

            if result.interrupted:
                logger.info(
                    "Plan interrupted: %s (steps=%d, reward=%.1f)",
                    result.interrupt_reason, result.steps_taken, result.total_reward,
                )
                if result.interrupt_reason == "Agent died":
                    self._deaths += 1
                    # Check ScorableNLEEnv game over
                    if hasattr(self.env, "total_score") and self.env.total_score <= 0:
                        self._game_over = True
                    elif hasattr(self.env, "config") and self.env.config.max_deaths:
                        if self._deaths >= self.env.config.max_deaths:
                            self._game_over = True
            elif result.completed:
                logger.info(
                    "Plan completed (steps=%d, reward=%.1f)",
                    result.steps_taken, result.total_reward,
                )
            else:
                logger.info(
                    "Plan reached max steps (steps=%d, reward=%.1f)",
                    result.steps_taken, result.total_reward,
                )

            if self.memory.should_reflect():
                logger.info("Triggering reflection...")
                await self.memory.generate_reflections()

        # Final score from env
        if hasattr(self.env, "total_score"):
            self._total_reward = self.env.total_score
            self._deaths = self.env.deaths

        summary = {
            "total_reward": self._total_reward,
            "total_steps": self._total_steps,
            "deaths": self._deaths,
            "episodes_recorded": len(self.memory.episodes),
            "notes_generated": len(self.memory.notes),
        }
        logger.info("=== Agent Controller finished === %s", summary)
        return summary
