from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable, Awaitable

from src.harness.probe.token_budget import TokenBudget
from src.harness.probe.metrics import UsageMetrics
from src.harness.probe import trimmer

log = logging.getLogger(__name__)


class ContextProbe:
    """Intercepts LLM calls to enforce token budgets and track usage.

    Generic trim cascade (tool output truncate → file evict → summarize → hard cut).
    Executor subtasks use ``HarnessEngine.before_executor_llm_call`` first for
    digest fold/compact, then fall through to ``before_call`` here.
    """

    def __init__(
        self,
        max_tokens: int = 128_000,
        budget_ratio: float = 0.75,
        llm_summarizer: Callable[..., Awaitable[str]] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.budget_ratio = budget_ratio
        self.budget = int(max_tokens * budget_ratio)
        self._budget_calc = TokenBudget()
        self._metrics = UsageMetrics()
        self._llm_summarizer = llm_summarizer

    @property
    def metrics(self) -> UsageMetrics:
        return self._metrics

    async def before_call(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Trim *messages* if total tokens exceed the budget."""
        current = await asyncio.to_thread(self._budget_calc.count_tokens, messages)
        if current <= self.budget:
            return messages

        log.info(
            "Token count %d exceeds budget %d — applying trim strategies",
            current,
            self.budget,
        )
        trimmed = await asyncio.to_thread(self._apply_trim_strategies_sync, messages)
        if self._llm_summarizer is not None and not self._under_budget_sync(trimmed):
            trimmed = await trimmer.summarize_old_turns(trimmed, self._llm_summarizer)
            if not self._under_budget_sync(trimmed):
                trimmed = await asyncio.to_thread(
                    trimmer.hard_truncate,
                    trimmed,
                    self.budget,
                )
        return trimmed

    async def after_call(self, response: Any, usage: Any) -> None:
        """Record token usage after an LLM response is received."""
        if usage is None:
            return

        model = getattr(response, "model", None) or "unknown"
        prompt_tokens = getattr(usage, "prompt_tokens", 0)
        completion_tokens = getattr(usage, "completion_tokens", 0)
        self._metrics.record(model, prompt_tokens, completion_tokens)

    def _apply_trim_strategies_sync(
        self,
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        result = trimmer.truncate_tool_outputs(messages)
        if self._under_budget_sync(result):
            return result

        result = trimmer.evict_irrelevant_files(result)
        if self._under_budget_sync(result):
            return result

        return trimmer.hard_truncate(result, self.budget)

    def _under_budget_sync(self, messages: list[dict[str, Any]]) -> bool:
        return self._budget_calc.count_tokens(messages) <= self.budget
