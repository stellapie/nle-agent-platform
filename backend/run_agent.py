"""Run the NLE Agent with a configurable LLM backend."""
import asyncio
import sys
import os
import logging

sys.path.insert(0, os.path.dirname(__file__))

from agent.llm_interface import OpenAILLM, DeepSeekLLM, AnthropicLLM
from agent.controller import AgentController
from env.scorable_env import ScorableNLEEnv, ScoreConfig
from memory.memory_system import MemorySystem

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def create_llm(provider: str = None) -> object:
    provider = provider or os.getenv("LLM_PROVIDER", "deepseek")
    if provider == "openai":
        return OpenAILLM(model=os.getenv("LLM_MODEL", "gpt-4o"))
    elif provider == "deepseek":
        return DeepSeekLLM(model=os.getenv("LLM_MODEL", "deepseek-chat"))
    elif provider == "anthropic":
        return AnthropicLLM(model=os.getenv("LLM_MODEL", "claude-sonnet-4-20250514"))
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def main():
    llm = create_llm()
    env = ScorableNLEEnv(ScoreConfig(
        initial_score=int(os.getenv("INITIAL_SCORE", "1000")),
        death_penalty=int(os.getenv("DEATH_PENALTY", "200")),
    ))
    memory = MemorySystem(save_dir="./memory_data")
    memory.load()

    agent = AgentController(llm=llm, env=env, memory=memory)
    max_plans = int(os.getenv("MAX_PLANS", "50"))

    try:
        await agent.run(max_plans=max_plans)
    except KeyboardInterrupt:
        agent.stop()
    finally:
        memory.save()
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
