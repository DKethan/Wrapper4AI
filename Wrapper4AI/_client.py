"""
Unified client for multi-provider LLM chat.

Uses the provider registry to resolve backends and manages history,
streaming, and token counting.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from wrapper4ai.config import DEFAULTS, get_default_model
from wrapper4ai.exceptions import ConfigurationError
from wrapper4ai.providers import get_provider
from wrapper4ai._utils import count_message_tokens, count_tokens

VALID_ROLES = frozenset({"user", "assistant", "system"})


class Client:
    """
    Unified LLM client with history, streaming, and token counting.

    Use connect(provider, model, **kwargs) or Client(provider, model=..., **kwargs).
    """

    def __init__(
        self,
        provider: str,
        *,
        model: str | None = None,
        api_key: str | None = None,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.provider_name = provider.lower()
        self.model = model or get_default_model(provider)
        if not self.model:
            raise ConfigurationError(f"No model specified for provider: {provider}")
        self._system_prompt = system_prompt
        self._history: list[dict[str, str]] = []
        self._provider_instance = get_provider(
            provider,
            model=self.model,
            api_key=api_key,
            **kwargs,
        )

    @property
    def provider(self) -> str:
        return self.provider_name

    def _messages(self, extra: list[dict[str, str]] | None = None) -> list[dict[str, str]]:
        """Build message list: optional system prompt + history + optional extra."""
        out: list[dict[str, str]] = []
        if self._system_prompt:
            out.append({"role": "system", "content": self._system_prompt})
        out.extend(self._history)
        if extra:
            out.extend(extra)
        return out

    def chat(self, prompt: str, *, use_history: bool = True) -> str:
        """Send a user message and return the assistant reply. Optionally update history."""
        if use_history:
            self._history.append({"role": "user", "content": prompt})
            messages = self._messages()
            response = self._provider_instance.generate(messages)
            self._history.append({"role": "assistant", "content": response})
            return response
        messages = self._messages(extra=[{"role": "user", "content": prompt}])
        return self._provider_instance.generate(messages)

    def stream(self, prompt: str) -> Iterator[str]:
        """Stream the assistant reply for the given user message; updates history."""
        self._history.append({"role": "user", "content": prompt})
        messages = self._messages()
        chunks: list[str] = []
        for chunk in self._provider_instance.stream(messages):
            chunks.append(chunk)
            yield chunk
        self._history.append({"role": "assistant", "content": "".join(chunks)})

    def complete(self, messages: list[dict[str, str]]) -> str:
        """Get a single completion for the given messages. Does not update history."""
        return self._provider_instance.generate(messages)

    def complete_stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream a completion for the given messages. Does not update history."""
        return self._provider_instance.stream(messages)

    @property
    def history(self) -> list[dict[str, str]]:
        """Current conversation history (user + assistant messages)."""
        return list(self._history)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self._history.clear()

    def add_message(self, role: str, content: str) -> None:
        """Append a message to history. Role must be 'user', 'assistant', or 'system'."""
        if role not in VALID_ROLES:
            raise ConfigurationError(f"Invalid role: {role}. Must be one of {sorted(VALID_ROLES)}")
        self._history.append({"role": role, "content": content})

    def count_tokens(self, text_or_messages: str | list[dict[str, str]]) -> int:
        """Count tokens for a string or a list of messages."""
        if isinstance(text_or_messages, str):
            return count_tokens(text_or_messages)
        return count_message_tokens(text_or_messages)

    @property
    def total_tokens_used(self) -> int:
        """Total tokens in current history (approximate)."""
        return count_message_tokens(self._history)

    def generate_title(self, prompt: str) -> str:
        """Generate a short title for the given prompt."""
        return self._provider_instance.generate_title(prompt)

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: object) -> None:
        self.clear_history()

    def __repr__(self) -> str:
        return f"Client(provider={self.provider_name!r}, model={self.model!r})"
