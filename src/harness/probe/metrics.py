from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from src.harness.probe.token_budget import TokenBudget


@dataclass
class CallRecord:
    model: str
    prompt_tokens: int
    completion_tokens: int
    cost: float
    timestamp: float
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class UsageMetrics:
    """Tracks per-call and cumulative token usage across a session."""

    def __init__(self) -> None:
        self._records: list[CallRecord] = []
        self._by_model: dict[str, list[CallRecord]] = {}
        self._session_start = time.time()

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        *,
        cost: float | None = None,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
    ) -> CallRecord:
        from src.harness.probe.llm_usage import estimate_cost_for_model

        if cost is None:
            cost = estimate_cost_for_model(prompt_tokens, completion_tokens, model)
        rec = CallRecord(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            timestamp=time.time(),
            cache_creation_input_tokens=cache_creation_input_tokens,
            cache_read_input_tokens=cache_read_input_tokens,
        )
        self._records.append(rec)
        self._by_model.setdefault(model, []).append(rec)
        return rec

    def get_summary(self) -> dict[str, Any]:
        total_prompt = sum(r.prompt_tokens for r in self._records)
        total_completion = sum(r.completion_tokens for r in self._records)
        total_cost = sum(r.cost for r in self._records)
        total_cache_created = sum(r.cache_creation_input_tokens for r in self._records)
        total_cache_read = sum(r.cache_read_input_tokens for r in self._records)

        per_model: dict[str, dict[str, Any]] = {}
        for model, recs in self._by_model.items():
            per_model[model] = {
                "calls": len(recs),
                "prompt_tokens": sum(r.prompt_tokens for r in recs),
                "completion_tokens": sum(r.completion_tokens for r in recs),
                "cost": round(sum(r.cost for r in recs), 6),
                "cache_creation_input_tokens": sum(
                    r.cache_creation_input_tokens for r in recs
                ),
                "cache_read_input_tokens": sum(
                    r.cache_read_input_tokens for r in recs
                ),
            }

        cache_hit_ratio = (
            round(total_cache_read / total_prompt, 4)
            if total_prompt and total_cache_read
            else 0.0
        )

        return {
            "total_calls": len(self._records),
            "total_prompt_tokens": total_prompt,
            "total_completion_tokens": total_completion,
            "total_tokens": total_prompt + total_completion,
            "total_cost": round(total_cost, 6),
            "cache_creation_input_tokens": total_cache_created,
            "cache_read_input_tokens": total_cache_read,
            "cache_hit_ratio": cache_hit_ratio,
            "session_duration_s": round(time.time() - self._session_start, 2),
            "per_model": per_model,
        }

    def get_session_cost(self) -> float:
        return sum(r.cost for r in self._records)

    @property
    def total_calls(self) -> int:
        return len(self._records)

    @property
    def last_record(self) -> CallRecord | None:
        return self._records[-1] if self._records else None
