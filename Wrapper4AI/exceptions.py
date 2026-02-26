"""
Exception hierarchy for wrapper4ai.

All provider and client errors inherit from Wrapper4AIError for easy catching.
"""

from __future__ import annotations


class Wrapper4AIError(Exception):
    """Base exception for all wrapper4ai errors."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class ProviderNotFoundError(Wrapper4AIError):
    """Raised when the requested provider is not registered."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class AuthenticationError(Wrapper4AIError):
    """Raised when API authentication fails (e.g. invalid or missing API key)."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class MissingDependencyError(Wrapper4AIError):
    """Raised when a provider's optional dependency is not installed."""

    def __init__(self, provider: str, package: str) -> None:
        self.provider = provider
        self.package = package
        message = (
            f"Provider '{provider}' requires the '{package}' package. "
            f"Install it with: pip install {package}"
        )
        super().__init__(message)


class ConfigurationError(Wrapper4AIError):
    """Raised when configuration is invalid (e.g. invalid role, missing required setting)."""

    def __init__(self, message: str) -> None:
        super().__init__(message)


class RateLimitError(Wrapper4AIError):
    """Raised when the provider returns a rate limit (429) response."""

    def __init__(self, message: str, provider: str | None = None) -> None:
        super().__init__(message)
        self.provider = provider


class APIError(Wrapper4AIError):
    """Raised when the provider API returns an error."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code
