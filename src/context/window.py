from __future__ import annotations

from dataclasses import dataclass, field

import tiktoken

from src.agent.types import Message

_ENCODER: tiktoken.Encoding | None = None


def _get_encoder() -> tiktoken.Encoding:
    global _ENCODER  # noqa: PLW0603
    if _ENCODER is None:
        _ENCODER = tiktoken.get_encoding("cl100k_base")
    return _ENCODER


def count_tokens(text: str) -> int:
    return len(_get_encoder().encode(text, disallowed_special=()))


@dataclass
class ContextWindow:
    """Manages messages that fit within a token budget.

    Messages are appended in order.  When the budget is exceeded the oldest
    non-system messages are evicted to make room.
    """

    max_tokens: int = 128_000
    _messages: list[Message] = field(default_factory=list)
    _token_counts: list[int] = field(default_factory=list)

    def add(self, message: Message) -> None:
        tokens = count_tokens(message.content or "")
        self._messages.append(message)
        self._token_counts.append(tokens)
        self._evict_if_needed()

    def get_messages(self) -> list[Message]:
        return list(self._messages)

    def get_token_count(self) -> int:
        return sum(self._token_counts)

    def remaining_budget(self) -> int:
        return max(0, self.max_tokens - self.get_token_count())

    def clear(self) -> None:
        self._messages.clear()
        self._token_counts.clear()

    def _evict_if_needed(self) -> None:
        while self.get_token_count() > self.max_tokens and len(self._messages) > 1:
            # Never evict the system message (index 0)
            if self._messages[0].role == "system":
                if len(self._messages) < 2:
                    break
                self._messages.pop(1)
                self._token_counts.pop(1)
            else:
                self._messages.pop(0)
                self._token_counts.pop(0)

    def __len__(self) -> int:
        return len(self._messages)
