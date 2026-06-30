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


@pytest.mark.asyncio
async def test_stream_reports_connection_first_token_timeout(monkeypatch) -> None:
    async def completion(**_kwargs):
        await asyncio.sleep(0.03)
        return None

    monkeypatch.setattr("litellm.acompletion", completion)
    client = LLMClient(model="test-model", request_timeout=0.01, stream_idle_timeout=0.01)
    chunks = [item async for item in client.chat_stream([{"role": "user", "content": "x"}])]
    assert chunks[-1][1].model == "error"
    assert "connection/first token" in chunks[-1][1].content


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
