import json
import re
import logging
from abc import ABC, abstractmethod
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)


class LLMInterface(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        ...

    async def generate_json(self, prompt: str, system_prompt: str = "", temperature: float = 0.3, max_tokens: int = 2048) -> dict:
        raw = await self.generate(prompt, system_prompt, temperature, max_tokens)
        return self._extract_json(raw)

    @staticmethod
    def _extract_json(text: str) -> dict:
        json_match = re.search(r'```(?:json)?\s*(.+?)```', text, re.DOTALL)
        if json_match:
            text = json_match.group(1).strip()

        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[:-3]

        try:
            return json.loads(text.strip())
        except json.JSONDecodeError:
            brace = text.find("{")
            bracket = text.find("[")
            if brace == -1 and bracket == -1:
                raise
            start = min(i for i in (brace, bracket) if i >= 0)
            candidate = text[start:]
            return json.loads(candidate)


class OpenAILLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "gpt-4o", base_url: str = "https://api.openai.com/v1"):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/chat/completions", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["choices"][0]["message"]["content"]


class DeepSeekLLM(OpenAILLM):
    def __init__(self, api_key: str, model: str = "deepseek-chat"):
        super().__init__(api_key=api_key, model=model, base_url="https://api.deepseek.com/v1")


class AnthropicLLM(LLMInterface):
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1"

    async def generate(self, prompt: str, system_prompt: str = "", temperature: float = 0.7, max_tokens: int = 2048) -> str:
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json",
        }
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system_prompt:
            payload["system"] = system_prompt

        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(f"{self.base_url}/messages", json=payload, headers=headers)
            resp.raise_for_status()
            data = resp.json()

        return data["content"][0]["text"]
