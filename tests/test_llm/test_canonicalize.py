from __future__ import annotations

import os
import pytest
import litellm
from src.llm.client import canonicalize_model_name


def test_canonicalize_valid_model() -> None:
    # A valid litellm model like gpt-4 should not be changed, regardless of environment variables
    assert canonicalize_model_name("gpt-4") == "gpt-4"


def test_canonicalize_known_families() -> None:
    # Model names containing "claude" or "gemini" should be mapped to the correct provider
    assert canonicalize_model_name("claude-3-5-sonnet-20240620") == "anthropic/claude-3-5-sonnet-20240620"
    assert canonicalize_model_name("gemini-1.5-pro") == "gemini/gemini-1.5-pro"



def test_canonicalize_provider_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test LITELLM_PROVIDER
    monkeypatch.setenv("LITELLM_PROVIDER", "huggingface")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    
    # Qwen/Qwen2.5-32B-Instruct is unrecognized on its own, but huggingface/Qwen/Qwen2.5-32B-Instruct is recognized by litellm
    assert canonicalize_model_name("Qwen/Qwen2.5-32B-Instruct") == "huggingface/Qwen/Qwen2.5-32B-Instruct"

    # Test LLM_PROVIDER
    monkeypatch.delenv("LITELLM_PROVIDER", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openrouter")
    assert canonicalize_model_name("Qwen/Qwen2.5-32B-Instruct") == "openrouter/Qwen/Qwen2.5-32B-Instruct"


def test_canonicalize_openai_api_base(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test OPENAI_API_BASE when LITELLM_PROVIDER/LLM_PROVIDER are not set
    monkeypatch.delenv("LITELLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    
    assert canonicalize_model_name("Qwen/Qwen2.5-32B-Instruct") == "openai/Qwen/Qwen2.5-32B-Instruct"


def test_canonicalize_fallback_original_when_unrecognized(monkeypatch: pytest.MonkeyPatch) -> None:
    # Test fallback if all env vars are unset
    monkeypatch.delenv("LITELLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    
    assert canonicalize_model_name("Qwen/Qwen2.5-32B-Instruct") == "Qwen/Qwen2.5-32B-Instruct"
