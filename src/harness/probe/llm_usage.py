from __future__ import annotations

import logging
from typing import Any

import litellm

from src.agent.types import TokenUsage
from src.harness.probe.token_budget import TokenBudget

log = logging.getLogger(__name__)


def estimate_cost_for_model(
    prompt_tokens: int,
    completion_tokens: int,
    model: str,
) -> float:
    if prompt_tokens <= 0 and completion_tokens <= 0:
        return 0.0
    try:
        return float(
            litellm.completion_cost(
                model=model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
            )
        )
    except Exception:
        pass
    for candidate in _model_price_keys(model):
        cost = TokenBudget.estimate_cost(prompt_tokens, completion_tokens, candidate)
        if cost > 0:
            return cost
    return 0.0


def usage_from_litellm_response(response: Any | None) -> TokenUsage | None:
    if response is None:
        return None
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    from src.llm.prompt_cache import extract_cache_usage

    prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion = int(getattr(usage, "completion_tokens", 0) or 0)
    total = int(getattr(usage, "total_tokens", 0) or 0) or (prompt + completion)
    cache_created, cache_read = extract_cache_usage(usage)
    if prompt == 0 and completion == 0:
        return None
    return TokenUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
        cache_creation_input_tokens=cache_created,
        cache_read_input_tokens=cache_read,
    )


def estimate_usage_from_text(
    messages: list[dict[str, Any]],
    completion_text: str,
    *,
    model: str = "gpt-4o",
) -> TokenUsage:
    budget = TokenBudget(model=model)
    prompt_tokens = budget.count_tokens(messages)
    completion_tokens = budget.count_text_tokens(completion_text) if completion_text else 0
    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


def record_litellm_completion(
    metrics: Any,
    response: Any,
    *,
    model: str,
    messages: list[dict[str, Any]] | None = None,
    completion_text: str = "",
) -> None:
    """Record one LLM call into UsageMetrics (probe)."""
    usage = usage_from_litellm_response(response)
    if usage is None and messages is not None:
        usage = estimate_usage_from_text(messages, completion_text, model=model)
    if usage is None:
        return
    cost: float | None = None
    try:
        cost = float(litellm.completion_cost(completion_response=response))
    except Exception:
        cost = None
    metrics.record(
        model,
        usage.prompt_tokens,
        usage.completion_tokens,
        cost=cost,
        cache_creation_input_tokens=usage.cache_creation_input_tokens,
        cache_read_input_tokens=usage.cache_read_input_tokens,
    )


def _model_price_keys(model: str) -> list[str]:
    keys = [model]
    if "/" in model:
        keys.append(model.split("/", 1)[1])
        keys.append(model.split("/")[-1])
    lowered = model.lower()
    if "deepseek" in lowered:
        keys.append("deepseek-coder-v2")
    if "qwen" in lowered and "7b" in lowered:
        keys.append("gpt-4o-mini")
    return list(dict.fromkeys(keys))
