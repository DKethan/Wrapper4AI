"""OpenAI provider using the official SDK (lazy import)."""

from __future__ import annotations

from collections.abc import Iterator

from wrapper4ai.config import DEFAULTS, resolve_api_key
from wrapper4ai.exceptions import AuthenticationError, MissingDependencyError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages


class OpenAIProvider(BaseProvider):
    """OpenAI chat completions (GPT-4o, GPT-4.1, etc.)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        try:
            import openai
        except ImportError as e:
            raise MissingDependencyError("openai", "openai") from e
        key = resolve_api_key("openai", explicit_key=api_key)
        if not key:
            raise AuthenticationError("OpenAI API key is required", provider="openai")
        self._client = openai.OpenAI(api_key=key)
        try:
            import tiktoken
            self._encoding = tiktoken.get_encoding("cl100k_base")
        except Exception:
            self._encoding = None
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))

    def generate(self, messages: list[dict[str, str]]) -> str:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=truncated,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        stream = self._client.chat.completions.create(
            model=self.model,
            messages=truncated,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        if self._encoding is None:
            return sum(len(m.get("content", "")) // 4 + 3 for m in messages)
        n = 0
        for m in messages:
            n += 3 + len(self._encoding.encode(m.get("content", "") or ""))
        return n
