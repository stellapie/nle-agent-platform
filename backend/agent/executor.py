import logging
from typing import Any, Optional

from agent.planner import Plan, ExecutionResult, InterruptCondition, InterruptType

logger = logging.getLogger(__name__)


def _default_interrupts() -> list[InterruptCondition]:
    def _check_low_hp(obs: dict, **kw: Any) -> bool:
        hp = obs.get("blstats", {}).get("hitpoints", 100) if isinstance(obs, dict) else 100
        max_hp = obs.get("blstats", {}).get("max_hitpoints", 100) if isinstance(obs, dict) else 100
        if max_hp == 0:
            return False
        return (hp / max_hp) < 0.3

    def _check_death(obs: dict, **kw: Any) -> bool:
        done = kw.get("done", False)
        return bool(done)

    return [
        InterruptCondition(
            interrupt_type=InterruptType.LOW_HP,
            check_fn=_check_low_hp,
            priority=10,
            description="HP below 30%",
        ),
        InterruptCondition(
            interrupt_type=InterruptType.DEATH,
            check_fn=_check_death,
            priority=100,
            description="Agent died",
        ),
    ]


class PlanExecutor:
    def __init__(
        self,
        env: Any,
        action_translator: Any,
        memory: Any,
        observation_formatter: Any = None,
        interrupt_conditions: Optional[list[InterruptCondition]] = None,
    ):
        self.env = env
        self.action_translator = action_translator
        self.memory = memory
        self.observation_formatter = observation_formatter
        self.interrupt_conditions = interrupt_conditions or _default_interrupts()
        self.interrupt_conditions.sort(key=lambda ic: ic.priority, reverse=True)

    def _format_observation(self, obs: Any) -> str:
        if self.observation_formatter is not None:
            return self.observation_formatter(obs)
        if isinstance(obs, dict):
            parts = []
            if "text_observation" in obs:
                parts.append(obs["text_observation"])
            if "message" in obs:
                parts.append(f"Message: {obs['message']}")
            return "\n".join(parts) if parts else str(obs)
        return str(obs)

    def _check_interrupts(self, obs: Any, done: bool = False) -> Optional[InterruptCondition]:
        for ic in self.interrupt_conditions:
            try:
                if ic.check_fn(obs, done=done):
                    return ic
            except Exception as e:
                logger.warning("Interrupt check %s failed: %s", ic.interrupt_type.value, e)
        return None

    async def execute(self, plan: Plan) -> ExecutionResult:
        total_reward = 0.0
        last_obs_text = ""

        while not plan.is_complete and plan.steps_executed < plan.max_steps:
            step_text = plan.current_step
            if step_text is None:
                break

            action_indices = self.action_translator.translate(step_text)
            if not action_indices:
                logger.warning("Cannot translate action: %s, skipping", step_text)
                plan.steps_executed += 1
                continue

            step_reward = 0.0
            obs = None
            done = False

            for action_idx in action_indices:
                obs, reward, done, info = self.env.step(action_idx)
                step_reward += reward
                if done:
                    break

            total_reward += step_reward
            plan.steps_executed += 1

            obs_text = self._format_observation(obs)
            last_obs_text = obs_text

            self.memory.record_observation(
                content=f"Action: {step_text} -> {obs_text}",
                importance=3,
                step_number=plan.steps_executed,
            )

            triggered = self._check_interrupts(obs, done=done)
            if triggered is not None:
                logger.info("Interrupt triggered: %s", triggered.description)
                if done:
                    self.memory.record_death(
                        content=f"Died after action: {step_text}. Observation: {obs_text}",
                        step_number=plan.steps_executed,
                    )
                return ExecutionResult(
                    plan=plan,
                    completed=False,
                    interrupted=True,
                    interrupt_reason=triggered.description,
                    final_observation=last_obs_text,
                    steps_taken=plan.steps_executed,
                    total_reward=total_reward,
                )

        return ExecutionResult(
            plan=plan,
            completed=plan.is_complete,
            interrupted=False,
            final_observation=last_obs_text,
            steps_taken=plan.steps_executed,
            total_reward=total_reward,
        )
