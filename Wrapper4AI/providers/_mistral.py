"""Mistral AI provider using the official SDK (lazy import)."""

from __future__ import annotations

from collections.abc import Iterator

from wrapper4ai.config import DEFAULTS, resolve_api_key
from wrapper4ai.exceptions import AuthenticationError, MissingDependencyError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages


def _to_mistral_messages(messages: list[dict[str, str]]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        if role == "assistant":
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "user"
        content = m.get("content") or ""
        out.append({"role": role, "content": content})
    return out


class MistralProvider(BaseProvider):
    """Mistral AI models via mistralai SDK."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        try:
            from mistralai import Mistral
        except ImportError as e:
            raise MissingDependencyError("mistral", "mistralai") from e
        key = resolve_api_key("mistral", explicit_key=api_key)
        if not key:
            raise AuthenticationError(
                "Mistral API key is required", provider="mistral"
            )
        self._client = Mistral(api_key=key)
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))

    def generate(self, messages: list[dict[str, str]]) -> str:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        msgs = _to_mistral_messages(truncated)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        resp = self._client.chat.complete(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        if not resp.choices or not resp.choices[0].message.content:
            return ""
        return resp.choices[0].message.content.strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        msgs = _to_mistral_messages(truncated)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        for chunk in self._client.chat.stream(
            model=self.model,
            messages=msgs,
            temperature=temperature,
            max_tokens=max_tokens,
        ):
            data = getattr(chunk, "data", chunk)
            choices = getattr(data, "choices", [])
            if choices:
                delta = getattr(choices[0], "delta", None)
                if delta:
                    content = getattr(delta, "content", None)
                    if content:
                        yield content

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") // 4 + 3 for m in messages)
