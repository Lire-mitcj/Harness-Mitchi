from __future__ import annotations

from src.agent.manifest import (
    EvidenceItem,
    StepManifest,
    missing_focus_symbols,
)
from src.hooks.before_tool import inspect_tool_request_async


async def test_decision_edit_passes_probe_intent_to_decision_llm() -> None:
    """Harness does not judge insertion/new-symbol placement — DecisionLLM does."""
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_notice",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(128, 140),
                symbol="ticket_notice",
                status="SATISFIED",
            ),
        ),
    )
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "db/init/init.sql",
            "intent": "查看 init.sql 中是否已有 bill_info 表以及完整的表结构",
            "focus_symbols": ["bill_info"],
            "context_window": [{"file": "db/init/init.sql", "span": [1, 200]}],
        },
        allowed_tools={"decision_edit"},
        manifest=manifest,
    )
    assert err is None


async def test_decision_edit_allows_unloaded_focus_symbols() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_notice",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(128, 140),
                symbol="ticket_notice",
                status="SATISFIED",
            ),
        ),
    )
    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "db/init/init.sql",
            "intent": "Compare bill_info columns against ticket_order",
            "focus_symbols": ["bill_info", "order_timeline"],
            "context_window": [{"file": "db/init/init.sql", "span": [100, 120]}],
        },
        allowed_tools={"decision_edit"},
        manifest=manifest,
    )
    assert err is None


def test_missing_focus_symbols_helper() -> None:
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="x",
                need="x",
                file="db/init/init.sql",
                symbol="ticket_notice",
                status="SATISFIED",
            ),
        ),
    )
    missing = missing_focus_symbols(
        manifest,
        "db/init/init.sql",
        ["ticket_notice", "bill_info"],
    )
    assert missing == ("bill_info",)
