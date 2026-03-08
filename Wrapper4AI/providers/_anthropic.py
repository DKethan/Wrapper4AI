"""Anthropic Claude provider using the official SDK (lazy import)."""

from __future__ import annotations

from collections.abc import Iterator

from wrapper4ai.config import DEFAULTS, resolve_api_key
from wrapper4ai.exceptions import AuthenticationError, MissingDependencyError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages


def _extract_system(messages: list[dict[str, str]]) -> tuple[str, list[dict[str, str]]]:
    system_parts: list[str] = []
    rest: list[dict[str, str]] = []
    for m in messages:
        if m.get("role") == "system":
            system_parts.append(m.get("content") or "")
        else:
            rest.append(m)
    return "\n".join(system_parts), rest


class AnthropicProvider(BaseProvider):
    """Anthropic Claude models via official SDK."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        try:
            import anthropic
        except ImportError as e:
            raise MissingDependencyError("anthropic", "anthropic") from e
        key = resolve_api_key("anthropic", explicit_key=api_key)
        if not key:
            raise AuthenticationError(
                "Anthropic API key is required", provider="anthropic"
            )
        self._client = anthropic.Anthropic(api_key=key)
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))

    def generate(self, messages: list[dict[str, str]]) -> str:
        system, rest = _extract_system(messages)
        truncated = truncate_messages(rest, self._max_history, preserve_system=False)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        kwargs: dict = {
            "model": self.model,
            "messages": truncated,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if system:
            kwargs["system"] = system
        resp = self._client.messages.create(**kwargs)
        if not resp.content:
            return ""
        return (resp.content[0].text or "").strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        system, rest = _extract_system(messages)
        truncated = truncate_messages(rest, self._max_history, preserve_system=False)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        kwargs: dict = {
            "model": self.model,
            "messages": truncated,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        if system:
            kwargs["system"] = system
        with self._client.messages.stream(**kwargs) as stream:
            for text in stream.text_stream:
                yield text

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") // 4 + 3 for m in messages)
