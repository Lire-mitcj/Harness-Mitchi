from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


@dataclass(slots=True)
class TokenUsage:
    """Token counts returned from a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0

    @classmethod
    def zero(cls) -> TokenUsage:
        return cls(prompt_tokens=0, completion_tokens=0, total_tokens=0)

    def __add__(self, other: TokenUsage) -> TokenUsage:
        return TokenUsage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            total_tokens=self.total_tokens + other.total_tokens,
            cache_creation_input_tokens=(
                self.cache_creation_input_tokens + other.cache_creation_input_tokens
            ),
            cache_read_input_tokens=(
                self.cache_read_input_tokens + other.cache_read_input_tokens
            ),
        )


@dataclass(slots=True)
class ToolCall:
    """A single tool invocation requested by the model."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(slots=True)
class ToolResult:
    """Result returned after executing a tool."""

    success: bool
    output: str
    error: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class Message:
    """A single message in the conversation history.

    Follows the OpenAI-style role convention so that the same structure can be
    serialized for any LLM provider supported by litellm.
    """

    role: str
    content: str
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    cache_breakpoint: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert to the dict format expected by litellm / OpenAI."""
        msg: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.cache_breakpoint:
            msg["_mitkii_cache_breakpoint"] = True
        if self.name is not None:
            msg["name"] = self.name
        if self.tool_calls is not None:
            msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    # OpenAI-compatible format requires arguments to be a JSON string.
                    "function": {"name": tc.name, "arguments": json.dumps(tc.arguments, ensure_ascii=False)},
                }
                for tc in self.tool_calls
            ]
        if self.tool_call_id is not None:
            msg["tool_call_id"] = self.tool_call_id
        return msg


@dataclass(slots=True)
class LLMResponse:
    """Parsed response from an LLM completion call."""

    content: str | None
    tool_calls: list[ToolCall] | None
    usage: TokenUsage | None
    model: str


class RiskLevel(StrEnum):
    """How dangerous a proposed action is considered."""

    SAFE = "safe"
    MODERATE = "moderate"
    DANGEROUS = "dangerous"


@dataclass
class AgentState:
    """Mutable state carried across turns within a single agent session.

    This is the central bookkeeping object: the agent loop reads and mutates it
    on every turn, and checkpointing serializes it to disk.
    """

    messages: list[Message] = field(default_factory=list)
    file_changes: list[str] = field(default_factory=list)
    current_plan: str | None = None
    turn_count: int = 0
    total_tokens_used: int = 0
    total_cost: float = 0.0

    def record_usage(self, usage: TokenUsage, cost: float) -> None:
        """Accumulate token / cost counters after an LLM call."""
        self.total_tokens_used += usage.total_tokens
        self.total_cost += cost

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    def advance_turn(self) -> int:
        """Increment the turn counter and return the new value."""
        self.turn_count += 1
        return self.turn_count


# ---------------------------------------------------------------------------
# Message construction helpers
# ---------------------------------------------------------------------------


def system_message(content: str, *, cache_breakpoint: bool = False) -> Message:
    return Message(role="system", content=content, cache_breakpoint=cache_breakpoint)


def harness_nudge(content: str) -> Message:
    """Dynamic harness hint — user role so cached prefix stays stable."""
    return user_message(f"[Harness]\n{content}")


def user_message(content: str) -> Message:
    return Message(role="user", content=content)


def assistant_message(
    content: str,
    tool_calls: list[ToolCall] | None = None,
) -> Message:
    return Message(role="assistant", content=content, tool_calls=tool_calls)


def tool_message(tool_call_id: str, content: str) -> Message:
    return Message(role="tool", content=content, tool_call_id=tool_call_id)
