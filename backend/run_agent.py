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

MEMORY_FILE = os.path.join(os.path.dirname(__file__), "memory_data", "memory.json")


def create_llm(provider: str = None):
    provider = provider or os.getenv("LLM_PROVIDER", "deepseek")
    api_key = os.getenv("LLM_API_KEY", "")
    model = os.getenv("LLM_MODEL")

    if not api_key:
        raise ValueError(
            "LLM_API_KEY environment variable is required. "
            "Set it before running: export LLM_API_KEY=your_key"
        )

    if provider == "openai":
        return OpenAILLM(api_key=api_key, model=model or "gpt-4o")
    elif provider == "deepseek":
        return DeepSeekLLM(api_key=api_key, model=model or "deepseek-chat")
    elif provider == "anthropic":
        return AnthropicLLM(api_key=api_key, model=model or "claude-sonnet-4-20250514")
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")


async def main():
    llm = create_llm()

    env = ScorableNLEEnv(ScoreConfig(
        initial_score=int(os.getenv("INITIAL_SCORE", "1000")),
        death_penalty=int(os.getenv("DEATH_PENALTY", "200")),
    ))

    memory = MemorySystem(llm=llm)
    if os.path.exists(MEMORY_FILE):
        try:
            memory.load(MEMORY_FILE)
            logging.info("Loaded memory from %s", MEMORY_FILE)
        except Exception as e:
            logging.warning("Failed to load memory: %s", e)

    max_plans = int(os.getenv("MAX_PLANS", "50"))
    agent = AgentController(
        llm=llm,
        env=env,
        memory=memory,
        max_plans=max_plans,
    )

    try:
        result = await agent.run()
        logging.info("Agent finished: %s", result)
    except KeyboardInterrupt:
        logging.info("Interrupted by user")
    finally:
        os.makedirs(os.path.dirname(MEMORY_FILE), exist_ok=True)
        memory.save(MEMORY_FILE)
        env.close()


if __name__ == "__main__":
    asyncio.run(main())
