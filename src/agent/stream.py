from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from src.agent.types import LLMResponse, TokenUsage, ToolCall

log = logging.getLogger(__name__)


class StreamCollector:
    """Collects streaming LLM chunks into a complete response.

    Accumulates content deltas, tool call fragments, and usage stats from
    the incremental stream produced by litellm's ``acompletion`` (or any
    provider that yields OpenAI-style SSE chunks).
    """

    def __init__(self) -> None:
        self._content_parts: list[str] = []
        self._tool_calls: dict[int, dict[str, Any]] = {}
        self._usage: TokenUsage | None = None
        self._model: str = "unknown"
        self._finish_reason: str | None = None

    async def collect(
        self, stream: AsyncIterator[Any]
    ) -> tuple[str, LLMResponse]:
        """Consume the full stream and return ``(full_text, response)``.

        Yields nothing; callers who want per-chunk events should use
        :meth:`collect_with_events` instead.
        """
        async for chunk in stream:
            self._process_chunk(chunk)
        return self._finalize()

    async def collect_with_events(
        self, stream: AsyncIterator[Any]
    ) -> AsyncIterator[dict[str, Any]]:
        """Consume the stream, yielding normalized event dicts per chunk.

        The final yield is ``{"type": "response", "response": LLMResponse}``.
        """
        async for chunk in stream:
            delta = self._process_chunk(chunk)
            if delta:
                yield delta

        full_text, response = self._finalize()
        yield {"type": "response", "response": response}

    def _process_chunk(self, chunk: Any) -> dict[str, Any] | None:
        """Handle a single streaming chunk and return a normalized event or None."""
        self._model = getattr(chunk, "model", self._model) or self._model

        if hasattr(chunk, "usage") and chunk.usage is not None:
            u = chunk.usage
            self._usage = TokenUsage(
                prompt_tokens=getattr(u, "prompt_tokens", 0),
                completion_tokens=getattr(u, "completion_tokens", 0),
                total_tokens=getattr(u, "total_tokens", 0),
            )

        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return None

        choice = choices[0]
        delta = getattr(choice, "delta", None)
        if delta is None:
            return None

        self._finish_reason = getattr(choice, "finish_reason", None) or self._finish_reason

        content = getattr(delta, "content", None)
        if content:
            self._content_parts.append(content)
            return {"type": "content", "content": content}

        tool_calls = getattr(delta, "tool_calls", None)
        if tool_calls:
            for tc_delta in tool_calls:
                idx = getattr(tc_delta, "index", 0)
                entry = self._tool_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if getattr(tc_delta, "id", None):
                    entry["id"] = tc_delta.id
                fn = getattr(tc_delta, "function", None)
                if fn:
                    if getattr(fn, "name", None):
                        entry["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        entry["arguments"] += fn.arguments
            return {"type": "tool_call_delta"}

        return None

    def _finalize(self) -> tuple[str, LLMResponse]:
        full_text = "".join(self._content_parts)

        tool_calls: list[ToolCall] | None = None
        if self._tool_calls:
            tool_calls = []
            for _, raw in sorted(self._tool_calls.items()):
                args_str = raw["arguments"]
                try:
                    args = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    log.warning("Failed to parse tool call arguments: %s", args_str[:200])
                    args = {"_raw": args_str}
                tool_calls.append(ToolCall(id=raw["id"], name=raw["name"], arguments=args))

        response = LLMResponse(
            content=full_text or None,
            tool_calls=tool_calls,
            usage=self._usage,
            model=self._model,
        )
        return full_text, response

    def reset(self) -> None:
        self._content_parts.clear()
        self._tool_calls.clear()
        self._usage = None
        self._model = "unknown"
        self._finish_reason = None
