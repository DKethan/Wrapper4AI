"""
Wrapper4AI: unified multi-provider LLM client.

Usage:

    from wrapper4ai import connect, Client, list_providers

    client = connect("openai", "gpt-4o", api_key="...")
    response = client.chat("Hello!")

    # Or with explicit Client
    client = Client("anthropic", model="claude-sonnet-4-20250514")
    for chunk in client.stream("Tell me a story"):
        print(chunk, end="")
"""

from wrapper4ai._client import Client
from wrapper4ai.providers import list_providers


def connect(
    provider: str,
    model: str | None = None,
    **kwargs: object,
) -> Client:
    """
    Create a client for the given provider and model.

    Args:
        provider: Provider name (e.g. 'openai', 'anthropic', 'google').
        model: Model id; uses provider default if omitted.
        **kwargs: Passed to Client (api_key, system_prompt, temperature, etc.).

    Returns:
        Client instance.
    """
    return Client(provider, model=model, **kwargs)


__all__ = ["Client", "connect", "list_providers"]
