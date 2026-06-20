from __future__ import annotations

import asyncio
import json

import pytest

from src.agent.cursor_contracts import RetrievalResult
from src.agent.cursor_query_bridge import QueryBridgeResult
from src.harness.cursor.retrieval_guardrail import (
    CursorRetrievalGuardrail,
    RetrievalGuardrailPolicy,
)


def test_guardrail_light_dedup_and_query_cap() -> None:
    guardrail = CursorRetrievalGuardrail(query_cap=3)

    queries = guardrail.normalize((" View ", "view", "component", "page", "screen"))

    assert queries == ("View", "component", "page")


def test_guardrail_does_not_apply_semantic_deduplication() -> None:
    guardrail = CursorRetrievalGuardrail(query_cap=8)

    queries = guardrail.normalize(
        ("view", "views", "screen", "ui", "component", "page")
    )

    assert queries == ("view", "views", "screen", "ui", "component", "page")


def _fallback_bridge() -> QueryBridgeResult:
    return QueryBridgeResult(
        intent="list",
        expanded_terms=["view", "component"],
        keywords=[],
        symbols=[],
        file_hints=["ui"],
    )


def test_bridge_guardrail_enforces_schema_and_query_cap() -> None:
    guardrail = CursorRetrievalGuardrail(query_cap=3)
    content = json.dumps({
        "intent": "list",
        "expanded_terms": ["view", "component", "page", "screen"],
        "keywords": [],
        "symbols": [],
        "file_hints": [],
    })

    result = guardrail.validate_bridge_json(content, _fallback_bridge())

    assert result.bridge.search_terms() == ("view", "component", "page")
    assert result.repaired is False
    assert result.used_fallback is False
    assert result.missing_keys == ()


def test_bridge_guardrail_repairs_malformed_json() -> None:
    guardrail = CursorRetrievalGuardrail(query_cap=4)
    content = """
    ```json
    {intent: 'list', expanded_terms: ['view', 'page',],
     keywords: [], symbols: [], file_hints: ['ui'],}
    ```
    """

    result = guardrail.validate_bridge_json(content, _fallback_bridge())

    assert result.bridge.expanded_terms == ["view", "page"]
    assert result.repaired is True
    assert result.used_fallback is False


def test_bridge_guardrail_fills_missing_and_invalid_fields() -> None:
    guardrail = CursorRetrievalGuardrail(query_cap=4)
    content = json.dumps({
        "intent": "invalid",
        "expanded_terms": "not-an-array",
        "symbols": [],
        "file_hints": [],
    })

    result = guardrail.validate_bridge_json(content, _fallback_bridge())

    assert result.bridge.intent == "list"
    assert result.bridge.expanded_terms == ["view", "component"]
    assert result.bridge.keywords == []
    assert result.missing_keys == ("keywords",)
    assert result.repaired is True


@pytest.mark.parametrize("content", (None, "", "[]", "not json", "{}"))
def test_bridge_guardrail_never_returns_empty_retrieval_input(
    content: str | None,
) -> None:
    guardrail = CursorRetrievalGuardrail(query_cap=4)
    empty_fallback = QueryBridgeResult(
        intent="explain",
        expanded_terms=[],
        keywords=[],
        symbols=[],
        file_hints=[],
    )

    result = guardrail.validate_bridge_json(content, empty_fallback)

    assert result.bridge.search_terms() == ("code", "module", "class", "function")
    assert result.used_fallback is True


@pytest.mark.asyncio
async def test_retrieval_guardrail_never_calls_retriever_with_empty_input() -> None:
    calls: list[tuple[str, ...]] = []

    async def retrieve(terms: tuple[str, ...]) -> RetrievalResult:
        calls.append(terms)
        return RetrievalResult()

    guardrail = CursorRetrievalGuardrail(query_cap=2)

    result = await guardrail.run((), retrieve)

    assert calls == [("code", "module")]
    assert result.queries == ("code", "module")


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("query_cap", 0),
        ("fan_out", 0),
        ("timeout", 0),
        ("early_stop_candidates", 0),
    ),
)
def test_guardrail_policy_rejects_non_positive_limits(field: str, value: int) -> None:
    values: dict[str, int | float] = {
        "query_cap": 12,
        "fan_out": 4,
        "timeout": 12.0,
        "early_stop_candidates": 8,
    }
    values[field] = value

    with pytest.raises(ValueError, match="must be positive"):
        RetrievalGuardrailPolicy(**values)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_guardrail_controls_fan_out_and_stops_early() -> None:
    calls: list[tuple[str, ...]] = []

    async def retrieve(terms: tuple[str, ...]) -> RetrievalResult:
        calls.append(terms)
        return RetrievalResult(files=(f"{terms[0]}.py",))

    guardrail = CursorRetrievalGuardrail(
        query_cap=8,
        fan_out=2,
        timeout=1.0,
        early_stop_candidates=2,
    )

    result = await guardrail.run(("a1", "a2", "b1", "b2", "c1"), retrieve)

    assert calls == [("a1", "a2"), ("b1", "b2")]
    assert result.batches_started == 2
    assert result.stopped_early is True
    assert result.retrieval.files == ("a1.py", "b1.py")


@pytest.mark.asyncio
async def test_guardrail_timeout_returns_partial_results() -> None:
    calls = 0

    async def retrieve(terms: tuple[str, ...]) -> RetrievalResult:
        nonlocal calls
        calls += 1
        if calls == 1:
            return RetrievalResult(files=("first.py",))
        await asyncio.sleep(1)
        return RetrievalResult(files=("late.py",))

    guardrail = CursorRetrievalGuardrail(
        query_cap=4,
        fan_out=1,
        timeout=0.02,
        early_stop_candidates=4,
    )

    result = await guardrail.run(("first", "slow"), retrieve)

    assert result.timed_out is True
    assert result.retrieval.files == ("first.py",)
    assert calls == 2
