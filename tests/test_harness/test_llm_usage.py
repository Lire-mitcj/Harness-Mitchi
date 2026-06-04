from __future__ import annotations

from src.harness.probe.llm_usage import estimate_cost_for_model, estimate_usage_from_text


def test_estimate_usage_from_text() -> None:
    usage = estimate_usage_from_text(
        [{"role": "user", "content": "hello world"}],
        "response text",
    )
    assert usage.prompt_tokens > 0
    assert usage.completion_tokens > 0


def test_estimate_cost_for_model_zero_when_no_tokens() -> None:
    assert estimate_cost_for_model(0, 0, "openai/gpt-4o") == 0.0
