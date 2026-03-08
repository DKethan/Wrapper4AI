"""
Provider registry and base class for LLM backends.

Register built-in and custom providers, then use via Client(provider=..., model=...).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from wrapper4ai.config import DEFAULTS, get_default_model
from wrapper4ai.exceptions import ProviderNotFoundError

if TYPE_CHECKING:
    from typing import Type

_registry: dict[str, type] = {}


class BaseProvider(ABC):
    """
    Abstract base for all LLM providers.

    Subclasses must implement generate(), stream(), and count_tokens().
    """

    def __init__(
        self,
        model: str,
        *,
        api_key: str | None = None,
        **kwargs: object,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._kwargs = kwargs

    @abstractmethod
    def generate(self, messages: list[dict[str, str]]) -> str:
        """Return a single completion for the given messages."""
        ...

    @abstractmethod
    def stream(self, messages: list[dict[str, str]]) -> Iterator[str]:
        """Stream completion chunks for the given messages."""
        ...

    @abstractmethod
    def count_tokens(self, messages: list[dict[str, str]]) -> int:
        """Return total token count for the messages."""
        ...

    def generate_title(self, prompt: str) -> str:
        """Generate a short title for the prompt. Default uses generate()."""
        response = self.generate(
            [
                {
                    "role": "system",
                    "content": "Generate a 3-5 word title. Respond only with the title.",
                },
                {"role": "user", "content": prompt},
            ]
        )
        return (response or "").strip().strip('"')


def register_provider(name: str, provider_class: type) -> None:
    """Register a provider class under the given name."""
    _registry[name.lower()] = provider_class


def get_provider(
    name: str,
    model: str | None = None,
    *,
    api_key: str | None = None,
    **kwargs: object,
) -> BaseProvider:
    """
    Resolve and instantiate a provider by name.

    Args:
        name: Provider name (e.g. 'openai', 'anthropic').
        model: Model id; uses get_default_model(name) if not set.
        api_key: Optional API key (overrides env).
        **kwargs: Passed to the provider constructor.

    Returns:
        An instance of the provider.

    Raises:
        ProviderNotFoundError: If the provider is not registered.
    """
    key = name.lower()
    if key not in _registry:
        raise ProviderNotFoundError(f"Unknown provider: {name}")
    model = model or get_default_model(name)
    if not model:
        raise ProviderNotFoundError(f"No default model for provider: {name}")
    cls = _registry[key]
    return cls(model=model, api_key=api_key, **kwargs)


def list_providers() -> list[str]:
    """Return the list of registered provider names."""
    return sorted(_registry.keys())


# Register built-in providers (lazy SDK import happens inside each class).
def _register_builtins() -> None:
    from wrapper4ai.providers._anthropic import AnthropicProvider
    from wrapper4ai.providers._bedrock import BedrockProvider
    from wrapper4ai.providers._deepseek import DeepSeekProvider
    from wrapper4ai.providers._google import GoogleProvider
    from wrapper4ai.providers._huggingface import HuggingFaceProvider
    from wrapper4ai.providers._mistral import MistralProvider
    from wrapper4ai.providers._openai import OpenAIProvider
    from wrapper4ai.providers._perplexity import PerplexityProvider

    register_provider("openai", OpenAIProvider)
    register_provider("anthropic", AnthropicProvider)
    register_provider("google", GoogleProvider)
    register_provider("mistral", MistralProvider)
    register_provider("deepseek", DeepSeekProvider)
    register_provider("bedrock", BedrockProvider)
    register_provider("huggingface", HuggingFaceProvider)
    register_provider("perplexity", PerplexityProvider)


_register_builtins()
