"""Google Gemini provider using google-genai SDK (lazy import)."""

from __future__ import annotations

from collections.abc import Iterator

from wrapper4ai.config import DEFAULTS, resolve_api_key
from wrapper4ai.exceptions import AuthenticationError, MissingDependencyError
from wrapper4ai.providers import BaseProvider
from wrapper4ai._utils import truncate_messages


def _messages_to_contents(messages: list[dict[str, str]]) -> list[dict]:
    """Convert role/content messages to google-genai contents format."""
    out: list[dict] = []
    for m in messages:
        role = (m.get("role") or "user").lower()
        if role == "assistant":
            role = "model"
        if role == "system":
            role = "user"
        content = m.get("content") or ""
        out.append({"role": role, "parts": [{"text": content}]})
    return out


class GoogleProvider(BaseProvider):
    """Google Gemini models via google-genai SDK."""

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(model, api_key=api_key, **kwargs)
        try:
            from google import genai
        except ImportError as e:
            raise MissingDependencyError("google", "google-genai") from e
        key = resolve_api_key("google", explicit_key=api_key)
        if not key:
            raise AuthenticationError("Google API key is required", provider="google")
        self._client = genai.Client(api_key=key)
        self._max_history = int(DEFAULTS.get("max_history_tokens", 8000))

    def generate(self, messages: list[dict[str, str]]) -> str:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        contents = _messages_to_contents(truncated)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        resp = self._client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        if not resp or not resp.text:
            return ""
        return resp.text.strip()

    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        truncated = truncate_messages(
            messages,
            self._max_history,
            preserve_system=True,
        )
        contents = _messages_to_contents(truncated)
        temperature = float(self._kwargs.get("temperature", DEFAULTS["temperature"]))
        max_tokens = int(self._kwargs.get("max_tokens", DEFAULTS["max_tokens"]))
        config = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        for chunk in self._client.models.generate_content_stream(
            model=self.model,
            contents=contents,
            config=config,
        ):
            if chunk and chunk.text:
                yield chunk.text

    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        return sum(len(m.get("content", "") or "") // 4 + 3 for m in messages)
