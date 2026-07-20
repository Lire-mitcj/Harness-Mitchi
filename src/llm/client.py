from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import litellm
import tiktoken

from src.agent.types import LLMResponse, TokenUsage, ToolCall
from src.llm.dsml import split_dsml_tool_calls
from src.llm.prompt_cache import CacheTTL, apply_prompt_cache

log = logging.getLogger(__name__)

litellm.drop_params = True
setattr(litellm, "set_verbose", False)


def _extract_tool_names(tools: list[dict[str, Any]] | None) -> set[str] | None:
    if not tools:
        return None
    names = set()
    for t in tools:
        if isinstance(t, dict):
            if t.get("type") == "function" and "function" in t:
                name = t["function"].get("name")
                if name:
                    names.add(name)
            elif "name" in t:
                names.add(t["name"])
    return names


class LLMClient:
    """Unified LLM client backed by litellm.

    Supports streaming and non-streaming chat completions with optional
    function/tool calling.  Tracks cumulative token usage and cost.
    """

    def __init__(
        self,
        model: str,
        temperature: float = 0,
        max_tokens: int = 8192,
        *,
        request_timeout: float = 180,
        stream_idle_timeout: float = 60,
        stream_reasoning_timeout: float = 180,
        prompt_cache_enabled: bool = True,
        prompt_cache_min_tokens: int = 1024,
        prompt_cache_ttl: CacheTTL = "5m",
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.request_timeout = request_timeout
        self.stream_idle_timeout = stream_idle_timeout
        # Absolute ceiling for reasoning-before-first-content. Reasoning deltas
        # keep the stream alive (so a slow-but-healthy reasoning model is not
        # misread as a dead connection), but they must not remove the cap
        # entirely — otherwise a model that "thinks" for minutes hangs the edit.
        self.stream_reasoning_timeout = stream_reasoning_timeout
        self.prompt_cache_enabled = prompt_cache_enabled
        self.prompt_cache_min_tokens = prompt_cache_min_tokens
        self.prompt_cache_ttl = prompt_cache_ttl

        self._total_usage = TokenUsage.zero()
        self._total_cost: float = 0.0

        try:
            self._encoder = tiktoken.encoding_for_model(model)
        except KeyError:
            self._encoder = tiktoken.get_encoding("cl100k_base")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        stream: bool = True,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> LLMResponse | AsyncIterator[str]:
        """Send a chat completion request.

        When *stream* is ``False`` returns a resolved ``LLMResponse``.
        When *stream* is ``True`` returns an ``AsyncIterator`` that yields
        content-delta strings; call ``collect_stream`` to drain the iterator
        and get the final ``LLMResponse``.
        """
        kwargs = self._build_kwargs(
            messages,
            tools,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
        )

        if not stream:
            return await self._chat_sync(kwargs)

        return self._chat_stream_iter(kwargs)

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> AsyncIterator[tuple[str, LLMResponse | None]]:
        """Convenience streaming helper.

        Yields ``(chunk, None)`` while content is arriving, then a final
        ``("", LLMResponse)`` with the fully-assembled response.
        """
        import asyncio

        connect_timeout = timeout if timeout is not None else self.request_timeout
        idle_timeout = self.stream_idle_timeout
        kwargs = self._build_kwargs(
            messages,
            tools,
            max_tokens=max_tokens,
            timeout=connect_timeout,
            response_format=response_format,
        )

        import time

        content_parts: list[str] = []
        tool_calls_raw: dict[int, dict[str, Any]] = {}
        usage: TokenUsage | None = None
        last_chunk: Any = None

        timeout_stage = "connection/first content"
        try:
            # Do not pass a provider-level absolute timeout for streams. Local
            # guards bound connection + first *content* latency and each idle gap.
            # Empty/role-only SSE events must not reset the first-content budget.
            kwargs.pop("timeout", None)
            stream_started_at = time.monotonic()
            first_content_deadline = stream_started_at + connect_timeout
            reasoning_deadline = stream_started_at + max(
                connect_timeout, self.stream_reasoning_timeout
            )
            response = await asyncio.wait_for(
                litellm.acompletion(**kwargs, stream=True),
                timeout=connect_timeout,
            )
            iterator = response.__aiter__()
            awaiting_first_content = True
            seen_reasoning = False
            while True:
                if awaiting_first_content:
                    if seen_reasoning:
                        # Alive (model is reasoning) but still absolutely capped.
                        timeout_stage = "reasoning (no visible content)"
                        remaining = reasoning_deadline - time.monotonic()
                        chunk_timeout = min(remaining, idle_timeout)
                    else:
                        timeout_stage = "connection/first content"
                        remaining = first_content_deadline - time.monotonic()
                        chunk_timeout = remaining
                    if remaining <= 0:
                        raise TimeoutError()
                else:
                    timeout_stage = "chunk idle"
                    chunk_timeout = idle_timeout
                try:
                    chunk = await asyncio.wait_for(
                        anext(iterator),
                        timeout=chunk_timeout,
                    )
                except StopAsyncIteration:
                    break
                last_chunk = chunk
                chunk_usage = self._extract_usage(chunk)
                if chunk_usage is not None:
                    usage = chunk_usage
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta is None:
                    continue

                # Reasoning models may stream private thinking for a long time
                # before emitting user-visible content. Treat those deltas as
                # real model activity so the absolute first-content deadline
                # does not kill a healthy stream. Do not expose or append the
                # reasoning text to the final assistant response.
                reasoning_content = getattr(delta, "reasoning_content", None)
                if not reasoning_content:
                    model_extra = getattr(delta, "model_extra", None)
                    if isinstance(model_extra, dict):
                        reasoning_content = model_extra.get("reasoning_content")
                if reasoning_content:
                    # Keep alive + switch to the (bounded) reasoning deadline, but
                    # do NOT clear awaiting_first_content: reasoning is not content.
                    seen_reasoning = True

                if delta.content:
                    awaiting_first_content = False
                    content_parts.append(delta.content)
                    yield (delta.content, None)

                if delta.tool_calls:
                    awaiting_first_content = False
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_raw:
                            tool_calls_raw[idx] = {
                                "id": tc_delta.id or "",
                                "name": "",
                                "arguments": "",
                            }
                        entry = tool_calls_raw[idx]
                        if tc_delta.id:
                            entry["id"] = tc_delta.id
                        if tc_delta.function:
                            if tc_delta.function.name:
                                entry["name"] = tc_delta.function.name
                            if tc_delta.function.arguments:
                                entry["arguments"] += tc_delta.function.arguments
        except TimeoutError:
            if timeout_stage == "connection/first content":
                limit = connect_timeout
            elif timeout_stage == "reasoning (no visible content)":
                limit = max(connect_timeout, self.stream_reasoning_timeout)
            else:
                limit = idle_timeout
            log.error("LLM stream %s timed out after %.0fs", timeout_stage, limit)
            yield (
                "",
                LLMResponse(
                    content=(
                        f"LLM stream {timeout_stage} timed out after {int(limit)}s. "
                        "Retry without discarding prior streamed progress."
                    ),
                    tool_calls=None,
                    usage=None,
                    model="error",
                ),
            )
            return

        if usage is None and last_chunk is not None:
            usage = self._extract_usage(last_chunk)
        completion_text = "".join(content_parts)
        known_tools = _extract_tool_names(tools)
        cleaned_text, dsml_calls = split_dsml_tool_calls(
            completion_text,
            known_tool_names=known_tools,
        )
        if usage is None:
            from src.harness.probe.llm_usage import estimate_usage_from_text

            usage = estimate_usage_from_text(
                messages, completion_text, model=self.model
            )
        cost = self._estimate_cost(usage, model=self.model)
        if usage:
            self._total_usage = self._total_usage + usage
            self._total_cost += cost

        final = LLMResponse(
            content=cleaned_text,
            tool_calls=(self._parse_tool_calls(tool_calls_raw) + dsml_calls) or None,
            usage=usage,
            model=self.model,
        )
        yield ("", final)

    # ------------------------------------------------------------------
    # Token helpers
    # ------------------------------------------------------------------

    def count_tokens(self, text: str) -> int:
        return len(self._encoder.encode(text))

    def count_messages_tokens(self, messages: list[dict[str, Any]]) -> int:
        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            for value in msg.values():
                if isinstance(value, str):
                    total += self.count_tokens(value)
        total += 2  # priming
        return total

    @property
    def total_usage(self) -> TokenUsage:
        return self._total_usage

    @property
    def total_cost(self) -> float:
        return self._total_cost

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int | None = None,
        timeout: float | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        request_timeout = timeout if timeout is not None else self.request_timeout
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": apply_prompt_cache(
                messages,
                model=self.model,
                enabled=self.prompt_cache_enabled,
                min_tokens=self.prompt_cache_min_tokens,
                ttl=self.prompt_cache_ttl,
            ),
            "temperature": self.temperature,
            "max_tokens": max_tokens if max_tokens is not None else self.max_tokens,
            "timeout": request_timeout,
            "stream_options": {"include_usage": True},
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
            kwargs["parallel_tool_calls"] = True
        elif response_format is not None:
            kwargs["response_format"] = response_format
        return kwargs

    async def _chat_sync(self, kwargs: dict[str, Any]) -> LLMResponse:
        response = await litellm.acompletion(**kwargs, stream=False)
        choice = response.choices[0]

        tool_calls_raw: dict[int, dict[str, Any]] = {}
        if choice.message.tool_calls:
            for i, tc in enumerate(choice.message.tool_calls):
                tool_calls_raw[i] = {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                }

        usage = self._extract_usage(response)
        cost = self._estimate_cost(usage, model=response.model or self.model)
        if usage:
            self._total_usage = self._total_usage + usage
            self._total_cost += cost

        cleaned_text, dsml_calls = split_dsml_tool_calls(
            choice.message.content,
            known_tool_names=_extract_tool_names(kwargs.get("tools")),
        )
        return LLMResponse(
            content=cleaned_text,
            tool_calls=(self._parse_tool_calls(tool_calls_raw) + dsml_calls) or None,
            usage=usage,
            model=response.model or self.model,
        )

    async def _chat_stream_iter(self, kwargs: dict[str, Any]) -> AsyncIterator[str]:
        """Low-level streaming iterator that yields content deltas."""
        response = await litellm.acompletion(**kwargs, stream=True)
        async for chunk in response:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content

    @staticmethod
    def _parse_tool_calls(raw: dict[int, dict[str, Any]]) -> list[ToolCall]:
        calls: list[ToolCall] = []
        for idx in sorted(raw):
            entry = raw[idx]
            try:
                arguments = json.loads(entry["arguments"])
            except (json.JSONDecodeError, TypeError):
                arguments = {"_raw": entry["arguments"]}
            calls.append(ToolCall(
                id=entry["id"],
                name=entry["name"],
                arguments=arguments,
            ))
        return calls

    @staticmethod
    def _extract_usage(response: Any) -> TokenUsage | None:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None
        from src.llm.prompt_cache import extract_cache_usage

        cache_created, cache_read = extract_cache_usage(usage)
        return TokenUsage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            total_tokens=getattr(usage, "total_tokens", 0) or 0,
            cache_creation_input_tokens=cache_created,
            cache_read_input_tokens=cache_read,
        )

    @staticmethod
    def _estimate_cost(usage: TokenUsage | None, *, model: str) -> float:
        if usage is None:
            return 0.0
        from src.harness.probe.llm_usage import estimate_cost_for_model

        return estimate_cost_for_model(
            usage.prompt_tokens,
            usage.completion_tokens,
            model,
        )
