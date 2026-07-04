from __future__ import annotations

from pathlib import Path

import pytest

from src.agent.manifest import observations_from_edited_file, reconcile_observations
from src.hooks.before_tool import inspect_tool_request_async


@pytest.mark.asyncio
async def test_context_window_rejects_span_beyond_file_length(tmp_path: Path) -> None:
    sql_path = tmp_path / "list.py"
    sql_path.write_text("line1\ndef order_timeline():\n    pass\n", encoding="utf-8")

    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "Replace stub with real query",
            "focus_symbols": ["order_timeline"],
            "context_window": [
                {"file": "list.py", "span": [345, 348], "reason": "stub"},
            ],
        },
        allowed_tools={"decision_edit", "view_symbol_code"},
        project_root=tmp_path,
    )

    assert err is not None
    assert "345" in err


@pytest.mark.asyncio
async def test_context_window_allows_anchor_span_without_focus_symbol(tmp_path: Path) -> None:
    from src.agent.manifest import EvidenceItem, StepManifest

    (tmp_path / "list.py").write_text(
        "async def build_router():\n    return router\n\nasync def helper():\n    pass\n",
        encoding="utf-8",
    )
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.symbol:list.py:build_router",
                need="handler",
                type="symbol",
                role="observed",
                file="list.py",
                span=(1, 2),
                symbol="build_router",
                status="SATISFIED",
            ),
        ),
    )

    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "Modify build_router to add validation",
            "focus_symbols": ["build_router"],
            "context_window": [
                {"file": "list.py", "span": [4, 5], "reason": "insertion anchor"},
            ],
        },
        allowed_tools={"decision_edit"},
        project_root=tmp_path,
        manifest=manifest,
    )

    assert err is None


@pytest.mark.asyncio
async def test_context_window_allows_new_table_after_anchor_span(tmp_path: Path) -> None:
    from src.agent.manifest import EvidenceItem, StepManifest

    sql = (
        "CREATE TABLE IF NOT EXISTS `ticket_order` (\n"
        "  order_id INT PRIMARY KEY\n"
        ");\n\n"
        "CREATE TABLE IF NOT EXISTS `ticket_notice` (\n"
        "  notice_id INT PRIMARY KEY\n"
        ");\n"
    )
    (tmp_path / "db" / "init").mkdir(parents=True)
    (tmp_path / "db" / "init" / "init.sql").write_text(sql, encoding="utf-8")
    manifest = StepManifest(
        required_items=(
            EvidenceItem(
                id="observed.schema:db/init/init.sql:ticket_notice",
                need="schema",
                type="schema",
                role="observed",
                file="db/init/init.sql",
                span=(5, 7),
                symbol="ticket_notice",
                status="SATISFIED",
            ),
        ),
    )

    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "db/init/init.sql",
            "intent": "在 ticket_notice 表之后添加 order_timeline 表",
            "focus_symbols": ["order_timeline"],
            "context_window": [
                {"file": "db/init/init.sql", "span": [5, 7], "reason": "insert after ticket_notice"},
                {"file": "db/init/init.sql", "span": [1, 3], "reason": "fk target ticket_order"},
            ],
        },
        allowed_tools={"decision_edit"},
        project_root=tmp_path,
        manifest=manifest,
    )

    assert err is None


@pytest.mark.asyncio
async def test_context_window_accepts_valid_target_span(tmp_path: Path) -> None:
    (tmp_path / "list.py").write_text(
        "async def order_timeline():\n    raise HTTPException(501)\n",
        encoding="utf-8",
    )

    err = await inspect_tool_request_async(
        "decision_edit",
        {
            "target_file": "list.py",
            "intent": "Replace 501 stub with real query",
            "focus_symbols": ["order_timeline"],
            "context_window": [
                {"file": "list.py", "span": [1, 2], "reason": "stub"},
            ],
        },
        allowed_tools={"decision_edit"},
        project_root=tmp_path,
    )

    assert err is None


def test_observations_from_edited_sql_file() -> None:
    content = (
        "-- header\n"
        "CREATE TABLE ticket_order (\n"
        "  id INT PRIMARY KEY\n"
        ");\n\n"
        "CREATE TABLE order_timeline (\n"
        "  id INT PRIMARY KEY,\n"
        "  order_id INT NOT NULL\n"
        ");\n"
    )
    observations = observations_from_edited_file("db/init/init.sql", content)

    assert len(observations) == 2
    names = {str(item["symbol"]) for item in observations}
    assert names == {"ticket_order", "order_timeline"}
    timeline = next(item for item in observations if item["symbol"] == "order_timeline")
    assert timeline["span"] == [6, 9]
    assert "order_id INT NOT NULL" in str(timeline["code"])


def test_reconcile_observations_adds_new_schema_from_edit() -> None:
    from src.agent.manifest import StepManifest

    manifest = StepManifest(required_items=(), sufficiency="INSUFFICIENT")
    content = (
        "CREATE TABLE order_timeline (\n"
        "  id INT PRIMARY KEY\n"
        ");\n"
    )
    observations = observations_from_edited_file("db/init/init.sql", content)
    updated = reconcile_observations(manifest, observations)

    schema_items = [
        item for item in updated.required_items if item.symbol == "order_timeline"
    ]
    assert len(schema_items) == 1
    assert schema_items[0].type == "schema"
    assert schema_items[0].role == "observed"
