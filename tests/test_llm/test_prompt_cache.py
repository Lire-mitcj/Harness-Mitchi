from __future__ import annotations

import json
from src.llm.prompt_cache import apply_prompt_cache, mark_cache_breakpoint


def test_apply_prompt_cache_strips_markers_when_disabled() -> None:
    messages = [
        mark_cache_breakpoint({"role": "system", "content": "x" * 5000}),
    ]
    out = apply_prompt_cache(messages, model="claude-sonnet-4-20250514", enabled=False)
    assert "_mitkii_cache_breakpoint" not in out[0]
    assert "cache_control" not in out[0]


def test_apply_prompt_cache_skips_small_blocks() -> None:
    messages = [
        mark_cache_breakpoint({"role": "system", "content": "tiny"}),
    ]
    out = apply_prompt_cache(
        messages,
        model="claude-sonnet-4-20250514",
        enabled=True,
        min_tokens=1024,
    )
    assert "cache_control" not in out[0]


def test_llm_client_enables_parallel_tool_calls() -> None:
    from src.llm.client import LLMClient

    client = LLMClient(model="test-model")
    kwargs = client._build_kwargs(  # noqa: SLF001 - intentional unit coverage
        [{"role": "user", "content": "x"}],
        [{"type": "function", "function": {"name": "grep_search"}}],
    )

    assert kwargs["tool_choice"] == "auto"
    assert kwargs["parallel_tool_calls"] is True


def test_llm_client_allows_per_request_budget_override() -> None:
    from src.llm.client import LLMClient

    client = LLMClient(model="test-model", request_timeout=180)
    kwargs = client._build_kwargs(  # noqa: SLF001 - intentional unit coverage
        [{"role": "user", "content": "summarize"}],
        [],
        max_tokens=512,
        timeout=30,
        response_format={"type": "json_object"},
    )

    assert kwargs["max_tokens"] == 512
    assert kwargs["timeout"] == 30
    assert kwargs["response_format"] == {"type": "json_object"}
    assert "tools" not in kwargs
