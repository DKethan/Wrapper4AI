"""
Configuration and environment resolution for wrapper4ai.

Resolves API keys and regions from explicit arguments or environment variables.
"""

from __future__ import annotations

import os

# Default values used across providers when not overridden.
DEFAULTS: dict[str, int | float] = {
    "temperature": 0.7,
    "max_tokens": 4096,
    "max_history_tokens": 8000,
    "timeout": 60,
    "max_retries": 3,
}

# Provider name -> env var name for API key.
_API_KEY_ENV: dict[str, str] = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "mistral": "MISTRAL_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "huggingface": "HUGGINGFACE_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "bedrock": "AWS_ACCESS_KEY_ID",
    "mock": "MOCK_API_KEY",
}

# Provider name -> env var for secret (e.g. AWS secret key).
_SECRET_KEY_ENV: dict[str, str] = {
    "bedrock": "AWS_SECRET_ACCESS_KEY",
    "mock": "MOCK_SECRET_KEY",
}

# Provider name -> env var for region (e.g. AWS region).
_REGION_ENV: dict[str, str] = {
    "bedrock": "AWS_DEFAULT_REGION",
    "mock": "MOCK_REGION",
}

# Default region when env is not set.
_DEFAULT_REGION = "us-east-1"

# Default model per provider (latest stable).
_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o",
    "anthropic": "claude-sonnet-4-20250514",
    "google": "gemini-2.0-flash",
    "mistral": "mistral-large-latest",
    "deepseek": "deepseek-chat",
    "bedrock": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "huggingface": "HuggingFaceH4/zephyr-7b-beta",
    "perplexity": "sonar-pro",
    "mock": "test-model",
}


def resolve_api_key(provider: str, *, explicit_key: str | None = None) -> str | None:
    """
    Resolve API key for a provider: explicit key overrides environment variable.

    Args:
        provider: Provider name (e.g. 'openai', 'anthropic').
        explicit_key: If set, this value is returned regardless of env.

    Returns:
        API key string, or None if not set.
    """
    if explicit_key is not None:
        return explicit_key
    env_var = _API_KEY_ENV.get(provider.lower())
    if not env_var:
        return None
    return os.environ.get(env_var)


def resolve_secret_key(provider: str, *, explicit_key: str | None = None) -> str | None:
    """
    Resolve secret key for a provider (e.g. AWS secret access key).

    Args:
        provider: Provider name (e.g. 'bedrock').
        explicit_key: If set, this value is returned.

    Returns:
        Secret key string, or None if not set.
    """
    if explicit_key is not None:
        return explicit_key
    env_var = _SECRET_KEY_ENV.get(provider.lower())
    if not env_var:
        return None
    return os.environ.get(env_var)


def resolve_region(provider: str, *, explicit_region: str | None = None) -> str:
    """
    Resolve region for a provider (e.g. AWS region).

    Args:
        provider: Provider name (e.g. 'bedrock').
        explicit_region: If set, this value is returned.

    Returns:
        Region string; defaults to us-east-1 if not set.
    """
    if explicit_region is not None:
        return explicit_region
    env_var = _REGION_ENV.get(provider.lower())
    if not env_var:
        return _DEFAULT_REGION
    return os.environ.get(env_var, _DEFAULT_REGION)


def get_default_model(provider: str) -> str:
    """
    Return the default model ID for a provider.

    Args:
        provider: Provider name (e.g. 'openai').

    Returns:
        Default model string; empty string for unknown provider.
    """
    return _DEFAULT_MODELS.get(provider.lower(), "")
