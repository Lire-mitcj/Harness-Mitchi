from __future__ import annotations

from types import SimpleNamespace

from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency
from src.agent.types import ToolResult
from src.hooks.reallocate_tools import determine_allowed_tools
from src.hooks.retrieval_convergence import (
    is_duplicate_retrieval_result,
    retrieval_round_all_duplicate,
    view_round_all_duplicate,
)


def test_is_duplicate_retrieval_result_mock_success() -> None:
    result = ToolResult(
        success=True,
        output="cached",
        metadata={"is_mock_success": True},
    )
    assert is_duplicate_retrieval_result("view_symbol_code", result) is True


def test_is_duplicate_retrieval_result_empty_store_with_refresh() -> None:
    result = ToolResult(
        success=True,
        output="replay",
        metadata={
            "raw_evidence_store": [],
            "refresh_evidence_store": [{"file": "a.py", "span": [1, 2]}],
        },
    )
    assert is_duplicate_retrieval_result("view_symbol_code", result) is True


def test_is_duplicate_retrieval_result_empty_grep_is_not_duplicate() -> None:
    result = ToolResult(success=True, output='{"matches": []}', metadata={})
    assert is_duplicate_retrieval_result("grep_search", result) is False


def test_view_round_all_duplicate_requires_view_calls_and_all_duplicate() -> None:
    dup = ToolResult(success=True, output="", metadata={"is_mock_success": True})
    fresh = ToolResult(
        success=True,
        output="new",
        metadata={"raw_evidence_store": [{"file": "a.py", "span": [1, 2], "code": "x"}]},
    )
    assert view_round_all_duplicate([("view_symbol_code", dup)]) is True
    assert view_round_all_duplicate([("grep_search", fresh)]) is False
    assert view_round_all_duplicate(
        [("view_symbol_code", dup), ("grep_search", fresh)]
    ) is True


def test_retrieval_round_all_duplicate_requires_every_retrieval_duplicate() -> None:
    dup = ToolResult(success=True, output="", metadata={"is_mock_success": True})
    fresh = ToolResult(
        success=True,
        output="new",
        metadata={"raw_evidence_store": [{"file": "a.py", "span": [1, 2], "code": "x"}]},
    )
    assert retrieval_round_all_duplicate([("view_symbol_code", dup)]) is True
    assert retrieval_round_all_duplicate(
        [("view_symbol_code", dup), ("grep_search", fresh)]
    ) is False


def test_one_no_gain_all_view_duplicate_round_keeps_grep() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="grounded",
                need="grounded",
                file="target.py",
                symbol="target",
                status="SATISFIED",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    run_state = SimpleNamespace(
        task_mode="edit",
        manifest=manifest,
        validation=SimpleNamespace(status="not_run"),
        changes=SimpleNamespace(files=()),
        retrieval_no_gain_rounds=1,
        view_last_round_all_duplicate=True,
    )
    state = SimpleNamespace(run_state=run_state)
    default_tools = frozenset(
        {"grep_search", "decision_edit", "codebase_retrieve", "view_symbol_code"}
    )

    allowed = determine_allowed_tools(state, None, default_tools)

    assert allowed == frozenset({"decision_edit"})


def test_format_duplicate_receipt_for_preflight_block() -> None:
    from src.agent.types import ToolResult
    from src.hooks.retrieval_convergence import format_duplicate_retrieval_receipt

    result = ToolResult(
        success=False,
        output="Error: BLOCK: Symbol 'handler' is already present in CURRENT_CONTEXT.",
        error="BLOCK: Symbol 'handler' is already present in CURRENT_CONTEXT.",
    )
    receipt = format_duplicate_retrieval_receipt(
        "view_symbol_code",
        result,
        arguments={"target_file": "api.py", "symbol": "handler"},
    )

    assert "NOT a tool failure" in receipt
    assert "already present" in receipt
