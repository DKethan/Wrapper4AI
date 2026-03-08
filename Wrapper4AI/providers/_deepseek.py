"""DeepSeek provider via HTTP API (requests)."""

from __future__ import annotations

from collections.abc import Iterator
import json

import requests

from wrapper4ai.config import DEFAULTS, resolve_api_key
from wrapper4ai.exceptions import AuthenticationError, APIError, RateLimitError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages

BASE_URL = "https://api.deepseek.com/v1"


class DeepSeekProvider(BaseProvider):
    """DeepSeek chat models via official HTTP API."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        key = resolve_api_key("deepseek", explicit_key=api_key)
        if not key:
            raise AuthenticationError(
                "DeepSeek API key is required", provider="deepseek"
            )
        self._api_key = key
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def generate(self, messages: list[dict[str, str]]) -> str:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        payload = {
            "model": self.model,
            "messages": truncated,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=int(DEFAULTS.get("timeout", 60)),
        )
        if r.status_code == 429:
            raise RateLimitError("DeepSeek rate limit exceeded", provider="deepseek")
        r.raise_for_status()
        data = r.json()
        choices = data.get("choices")
        if not choices or not choices[0].get("message", {}).get("content"):
            raise APIError("Empty response from DeepSeek", status_code=r.status_code)
        return (choices[0]["message"]["content"] or "").strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        payload = {
            "model": self.model,
            "messages": truncated,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        r = requests.post(
            f"{BASE_URL}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=int(DEFAULTS.get("timeout", 60)),
            stream=True,
        )
        if r.status_code == 429:
            raise RateLimitError("DeepSeek rate limit exceeded", provider="deepseek")
        r.raise_for_status()
        for line in r.iter_lines():
            if not line or line.strip() != line:
                continue
            if line.startswith(b"data: "):
                data = line[6:]
                if data.strip() == b"[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = obj.get("choices")
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                content = delta.get("content")
                if content:
                    yield content

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") // 4 + 3 for m in messages)
