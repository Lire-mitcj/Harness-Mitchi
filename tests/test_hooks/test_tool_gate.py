"""Tests for unified tool_gate policy."""

from __future__ import annotations

from types import SimpleNamespace

from src.agent.manifest import EvidenceItem, StepManifest, Sufficiency
from src.hooks.tool_gate import (
    actionable_stale,
    edit_ready_closed,
    files_to_invalidate,
    normalize_view_symbol,
    scorer_should_boost_view_for_stale,
)
from src.hooks.tool_scorer import ToolScoringContext


def test_normalize_view_symbol_full_file_aliases() -> None:
    assert normalize_view_symbol("(full file)") == "*"
    assert normalize_view_symbol("full file") == "*"
    assert normalize_view_symbol("FULL_FILE") == "*"
    assert normalize_view_symbol("*") == "*"
    assert normalize_view_symbol("NoisePolicy") == "NoisePolicy"


def test_files_to_invalidate_only_edited_file() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="a",
                need="a",
                type="symbol",
                role="observed",
                file="a.py",
                symbol="A",
                status="SATISFIED",
            ),
            EvidenceItem(
                id="b",
                need="b",
                type="symbol",
                role="observed",
                file="b.py",
                symbol="B",
                status="SATISFIED",
            ),
        )
    )
    assert files_to_invalidate("conf/noise_policy.yaml", manifest) == frozenset(
        {"conf/noise_policy.yaml"}
    )
    assert files_to_invalidate("a.py", manifest) == frozenset({"a.py"})


def test_actionable_stale_empty_when_edit_ready() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="obs",
                need="x",
                type="symbol",
                role="observed",
                file="other.py",
                symbol="X",
                status="STALE",
            ),
        ),
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
    )
    run_state = SimpleNamespace(
        manifest=manifest,
        changes=SimpleNamespace(files=("edited.py",)),
    )
    assert (
        actionable_stale(
            run_state,
            manifest,
            sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
            has_missing=False,
        )
        == ()
    )
    assert edit_ready_closed(
        task_mode="edit",
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        has_missing=False,
        has_actionable_stale_flag=False,
        blocks_edit=False,
    )


def test_scorer_skips_view_boost_when_edit_ready() -> None:
    ctx = ToolScoringContext(
        sufficiency=Sufficiency.SUFFICIENT_FOR_EDIT,
        has_missing=False,
        stale_ratio=1.0,
        has_stale=True,
    )
    assert scorer_should_boost_view_for_stale(ctx) is False
