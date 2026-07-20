from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from src.llm.client import LLMClient


def _chunk(text: str):
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))],
    )


@pytest.mark.asyncio
async def test_stream_has_no_absolute_total_deadline(monkeypatch) -> None:
    captured = {}

    async def completion(**kwargs):
        captured.update(kwargs)

        async def stream():
            for text in ("one", " two", " three", " four"):
                await asyncio.sleep(0.02)
                yield _chunk(text)

        return stream()

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(
        model="test-model",
        request_timeout=0.05,
        stream_idle_timeout=0.05,
    )
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]
    assert "".join(text for text, response in chunks if response is None) == "one two three four"
    assert chunks[-1][1] is not None
    assert chunks[-1][1].model == "test-model"
    assert "timeout" not in captured


def _role_only_chunk():
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=SimpleNamespace(content=None, tool_calls=None))],
    )


def _reasoning_chunk(text: str, *, via_model_extra: bool = False):
    delta = SimpleNamespace(content=None, tool_calls=None)
    if via_model_extra:
        delta.model_extra = {"reasoning_content": text}
    else:
        delta.reasoning_content = text
    return SimpleNamespace(
        usage=None,
        choices=[SimpleNamespace(delta=delta)],
    )


@pytest.mark.asyncio
async def test_stream_reports_connection_first_content_timeout(monkeypatch) -> None:
    async def completion(**_kwargs):
        await asyncio.sleep(0.03)
        return None

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(model="test-model", request_timeout=0.01, stream_idle_timeout=0.01)
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]
    assert chunks[-1][1].model == "error"
    assert "connection/first content" in chunks[-1][1].content


@pytest.mark.asyncio
async def test_stream_empty_events_do_not_extend_first_content_deadline(
    monkeypatch,
) -> None:
    """Role/empty SSE events must not reset the absolute first-content budget."""

    async def completion(**_kwargs):
        async def stream():
            for _ in range(5):
                yield _role_only_chunk()
                await asyncio.sleep(0.02)
            yield _chunk("late-content")

        return stream()

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(model="test-model", request_timeout=0.05, stream_idle_timeout=1.0)
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]
    assert chunks[-1][1].model == "error"
    assert "connection/first content" in chunks[-1][1].content
    assert not any(text for text, _response in chunks if text)


@pytest.mark.asyncio
@pytest.mark.parametrize("via_model_extra", [False, True])
async def test_reasoning_keeps_stream_alive_past_first_content_deadline(
    monkeypatch,
    via_model_extra: bool,
) -> None:
    """DeepSeek-style reasoning keeps the stream alive past the first-content
    deadline (no false timeout), and must not leak into content."""

    async def completion(**_kwargs):
        async def stream():
            await asyncio.sleep(0.01)
            yield _reasoning_chunk("private thought", via_model_extra=via_model_extra)
            # Past the tiny connect_timeout, but within the reasoning ceiling and
            # per-chunk idle timeout after reasoning starts.
            await asyncio.sleep(0.03)
            yield _chunk("answer")

        return stream()

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(
        model="test-model",
        request_timeout=0.02,
        stream_idle_timeout=0.05,
        stream_reasoning_timeout=5.0,
    )
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]

    assert "".join(text for text, response in chunks if response is None) == "answer"
    assert chunks[-1][1].model == "test-model"
    assert chunks[-1][1].content == "answer"


@pytest.mark.asyncio
async def test_endless_reasoning_is_bounded_by_reasoning_ceiling(monkeypatch) -> None:
    """Reasoning must not remove the cap entirely: a model that only ever
    reasons (never emits content) must still time out at the ceiling."""

    async def completion(**_kwargs):
        async def stream():
            while True:
                yield _reasoning_chunk("still thinking...")
                await asyncio.sleep(0.01)

        return stream()

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(
        model="test-model",
        request_timeout=0.05,
        stream_idle_timeout=1.0,
        stream_reasoning_timeout=0.08,
    )
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]

    assert chunks[-1][1].model == "error"
    assert "reasoning" in chunks[-1][1].content
    assert not any(text for text, _response in chunks if text)


@pytest.mark.asyncio
async def test_stream_reports_idle_timeout_between_chunks(monkeypatch) -> None:
    async def completion(**_kwargs):
        async def stream():
            yield _chunk("started")
            await asyncio.sleep(0.04)
            yield _chunk("late")

        return stream()

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(model="test-model", request_timeout=0.02, stream_idle_timeout=0.01)
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]
    assert chunks[0] == ("started", None)
    assert chunks[-1][1].model == "error"
    assert "chunk idle" in chunks[-1][1].content
