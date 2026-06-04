from __future__ import annotations

import json
from typing import Any

import tiktoken

MODEL_PRICING: dict[str, tuple[float, float]] = {
    # (prompt $/1M tokens, completion $/1M tokens)
    "claude-sonnet-4-20250514": (3.00, 15.00),
    "claude-opus-4-20250514": (15.00, 75.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o3-mini": (1.10, 4.40),
    "deepseek-coder-v2": (0.14, 0.28),
}

_TOKENS_PER_MESSAGE = 4
_TOKENS_PER_NAME = -1


class TokenBudget:
    """Accurate token counting via tiktoken with cost estimation."""

    def __init__(self, model: str = "gpt-4o") -> None:
        try:
            self._enc = tiktoken.encoding_for_model(model)
        except KeyError:
            self._enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += self.count_message_tokens(msg)
        total += 3  # every reply is primed with <|start|>assistant<|message|>
        return total

    def count_message_tokens(self, message: dict[str, Any]) -> int:
        num_tokens = _TOKENS_PER_MESSAGE
        for key, value in message.items():
            if value is None:
                continue
            text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            num_tokens += len(self._enc.encode(text))
            if key == "name":
                num_tokens += _TOKENS_PER_NAME
        return num_tokens

    def count_text_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    @staticmethod
    def estimate_cost(
        prompt_tokens: int,
        completion_tokens: int,
        model: str,
    ) -> float:
        pricing = MODEL_PRICING.get(model)
        if pricing is None:
            return 0.0
        prompt_rate, completion_rate = pricing
        return (prompt_tokens * prompt_rate + completion_tokens * completion_rate) / 1_000_000
