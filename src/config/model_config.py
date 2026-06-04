from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class ModelProvider(StrEnum):
    """Supported LLM backend providers."""

    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM = "custom"


@dataclass(frozen=True, slots=True)
class ModelProfile:
    """Static capability profile for a known model."""

    name: str
    provider: ModelProvider
    context_window: int
    supports_tools: bool = True
    supports_streaming: bool = True


# ---------------------------------------------------------------------------
# Well-known model profiles
# ---------------------------------------------------------------------------

DEFAULT_PROFILES: dict[str, ModelProfile] = {
    # Anthropic
    "claude-sonnet-4-20250514": ModelProfile(
        name="claude-sonnet-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
    ),
    "claude-opus-4-20250514": ModelProfile(
        name="claude-opus-4-20250514",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
    ),
    "claude-3-5-haiku-20241022": ModelProfile(
        name="claude-3-5-haiku-20241022",
        provider=ModelProvider.ANTHROPIC,
        context_window=200_000,
    ),
    # OpenAI
    "gpt-4o": ModelProfile(
        name="gpt-4o",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
    ),
    "gpt-4o-mini": ModelProfile(
        name="gpt-4o-mini",
        provider=ModelProvider.OPENAI,
        context_window=128_000,
    ),
    "o3-mini": ModelProfile(
        name="o3-mini",
        provider=ModelProvider.OPENAI,
        context_window=200_000,
        supports_streaming=False,
    ),
    # Ollama (local)
    "llama3.1:70b": ModelProfile(
        name="llama3.1:70b",
        provider=ModelProvider.OLLAMA,
        context_window=131_072,
        supports_tools=False,
    ),
    "deepseek-coder-v2": ModelProfile(
        name="deepseek-coder-v2",
        provider=ModelProvider.OLLAMA,
        context_window=131_072,
        supports_tools=False,
    ),
}


def _fallback_profile(model_name: str) -> ModelProfile:
    """Construct a best-effort profile for an unknown model name."""
    if "claude" in model_name.lower():
        provider = ModelProvider.ANTHROPIC
        ctx = 200_000
    elif "gpt" in model_name.lower() or "o3" in model_name.lower():
        provider = ModelProvider.OPENAI
        ctx = 128_000
    else:
        provider = ModelProvider.CUSTOM
        ctx = 8_192

    return ModelProfile(
        name=model_name,
        provider=provider,
        context_window=ctx,
        supports_tools=provider != ModelProvider.OLLAMA,
    )


def get_model_profile(model_name: str) -> ModelProfile:
    """Look up a model profile by name, falling back to heuristics."""
    return DEFAULT_PROFILES.get(model_name) or _fallback_profile(model_name)
