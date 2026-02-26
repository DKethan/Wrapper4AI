"""
Internal utilities for token counting and message truncation.

Uses tiktoken when available for OpenAI-compatible tokenization; falls back to
a simple character-based estimate otherwise.
"""

from __future__ import annotations

from typing import Any

# Lazy tiktoken encoder (cl100k_base) for accurate counting when available.
_encoder: Any = None
_NO_ENCODER = object()  # Sentinel so we don't re-import on every call after failure.


def _get_encoder() -> Any:
    """Return tiktoken encoder, or None if tiktoken is not installed."""
    global _encoder
    if _encoder is not None and _encoder is not _NO_ENCODER:
        return _encoder
    if _encoder is _NO_ENCODER:
        return None
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
    except Exception:
        _encoder = _NO_ENCODER
        return None
    return _encoder


def count_tokens(text: str) -> int:
    """
    Count tokens in a string.

    Uses tiktoken (cl100k_base) when available; otherwise approximates with ~4 chars per token.

    Args:
        text: Input string.

    Returns:
        Token count (>= 0).
    """
    if not text:
        return 0
    enc = _get_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(0, (len(text) + 3) // 4)


def count_message_tokens(messages: list[dict[str, str]]) -> int:
    """
    Count total tokens for a list of chat messages (role + content).

    Args:
        messages: List of dicts with 'role' and 'content' keys.

    Returns:
        Total token count (>= 0).
    """
    if not messages:
        return 0
    enc = _get_encoder()
    overhead = 3  # per-message overhead for chat format
    total = 0
    for m in messages:
        content = m.get("content", "") or ""
        if enc is not None:
            total += len(enc.encode(content)) + overhead
        else:
            total += max(0, (len(content) + 3) // 4) + overhead
    return total


def truncate_messages(
    messages: list[dict[str, str]],
    max_tokens: int,
    *,
    preserve_system: bool = False,
) -> list[dict[str, str]]:
    """
    Trim messages to fit within a token budget, keeping the most recent messages.

    Optionally keeps the first system message even when over limit.

    Args:
        messages: List of dicts with 'role' and 'content' keys.
        max_tokens: Maximum total tokens to retain.
        preserve_system: If True, always include the first system message.

    Returns:
        New list of messages (subset, most recent retained when over limit).
    """
    if not messages or max_tokens <= 0:
        return []

    enc = _get_encoder()
    overhead = 3

    def token_count(content: str) -> int:
        if enc is not None:
            return len(enc.encode(content)) + overhead
        return max(0, (len(content) + 3) // 4) + overhead

    system_msg: dict[str, str] | None = None
    if preserve_system and messages and messages[0].get("role") == "system":
        system_msg = messages[0]
        messages = messages[1:]
        system_tokens = token_count(system_msg.get("content", "") or "")
    else:
        system_tokens = 0

    budget = max(0, max_tokens - system_tokens)
    if not messages:
        return [system_msg] if system_msg else []

    trimmed: list[dict[str, str]] = []
    total = 0
    # Traverse in reverse to keep most recent messages
    for msg in reversed(messages):
        content = msg.get("content", "") or ""
        n = token_count(content)
        if total + n > budget:
            break
        trimmed.insert(0, msg)
        total += n

    if system_msg:
        trimmed.insert(0, system_msg)
    return trimmed
