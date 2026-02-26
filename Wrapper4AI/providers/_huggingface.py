"""Hugging Face Inference API provider (InferenceClient)."""

from __future__ import annotations

from collections.abc import Iterator

from wrapper4ai.config import DEFAULTS, resolve_api_key
from wrapper4ai.exceptions import AuthenticationError, MissingDependencyError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages


def _format_chat_prompt(messages: list[dict[str, str]]) -> str:
    """Turn messages into a single prompt string for text_generation."""
    parts: list[str] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        content = m.get("content") or ""
        if role == "user":
            parts.append(f"User: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(content)
    if messages and (messages[-1].get("role") or "user").lower() == "user":
        parts.append("Assistant:")
    return "\n".join(parts)


class HuggingFaceProvider(BaseProvider):
    """Hugging Face models via InferenceClient (text generation)."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        try:
            from huggingface_hub import InferenceClient
        except ImportError as e:
            raise MissingDependencyError("huggingface", "huggingface_hub") from e
        key = resolve_api_key("huggingface", explicit_key=api_key)
        self._client = InferenceClient(
            model=model,
            token=key,
        )
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))
        self._max_new_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))

    def generate(self, messages: list[dict[str, str]]) -> str:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        prompt = _format_chat_prompt(truncated)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        out = self._client.text_generation(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=temperature,
        )
        text = (out or "").strip()
        if "Assistant:" in text:
            text = text.split("Assistant:")[-1]
        if "User:" in text:
            text = text.split("User:")[0]
        return text.strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        prompt = _format_chat_prompt(truncated)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        for token in self._client.text_generation(
            prompt,
            max_new_tokens=self._max_new_tokens,
            temperature=temperature,
            stream=True,
        ):
            if token:
                yield token

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") // 4 + 3 for m in messages)
