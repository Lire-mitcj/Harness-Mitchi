from __future__ import annotations

import pytest

from src.agent.types import tool_message, user_message
from src.harness.discovery.manifest import manifest_actionable
from src.harness.discovery.manifest_fallback import (
    build_manifest_from_tool_history,
    extract_manifest_from_raw,
    manifest_reject_reason,
    try_parse_manifest_with_fallback,
)
from src.harness.discovery.scout_preflight import derive_grep_patterns


def test_derive_grep_patterns_register_transaction() -> None:
    patterns = derive_grep_patterns("为什么注册接口报数据库事务异常")
    assert any("register" in p for p in patterns)
    assert any("transaction" in p or "commit" in p for p in patterns)


def test_manifest_actionable_with_victim_files_only() -> None:
    from src.harness.discovery.manifest import DiagnosticsManifest, VictimFile

    m = DiagnosticsManifest(
        user_request="fix register",
        victim_files=[VictimFile(path="app.py", lines=[11])],
    )
    assert manifest_actionable(m)


def test_manifest_reject_reason_parse_failed() -> None:
    from src.harness.discovery.manifest import DiagnosticsManifest

    m = DiagnosticsManifest(
        user_request="x",
        uncertainties=["Manifest JSON parse failed; Planner will rely on project context."],
    )
    assert manifest_reject_reason(m) == "json_parse_failed"


def test_build_manifest_from_grep_tool_output() -> None:
    messages = [
        user_message("fix registration transaction error"),
        tool_message(
            "tc1",
            "./app.py:42:    db.session.commit()\n./app.py:55: def register():",
        ),
    ]
    m = build_manifest_from_tool_history(messages, user_request="fix register")
    assert m is not None
    assert manifest_actionable(m)
    assert any(v.path == "app.py" for v in m.victim_files)


def test_extract_manifest_from_malformed_trace() -> None:
    raw = (
        '<discovery_trace">\n'
        '1. Task type: bug_fix"\n'
        '3. Evidence collected: "数据库事务异常" in "app.py"\n'
        "4. Victim candidates: app.py lines 11\n"
    )
    m = extract_manifest_from_raw(raw, user_request="注册接口事务异常")
    assert m is not None
    assert manifest_actionable(m)
    assert any(v.path == "app.py" for v in m.victim_files)


def test_try_parse_manifest_with_preflight_grep() -> None:
    preflight = "./app.py:11: raise TransactionError('数据库事务异常')\n"
    m = try_parse_manifest_with_fallback(
        "not json",
        user_request="注册接口事务异常",
        preflight_grep=preflight,
    )
    assert manifest_actionable(m)
    assert any(v.path == "app.py" for v in m.victim_files)


@pytest.mark.asyncio
async def test_run_scout_preflight_finds_register(tmp_path, monkeypatch) -> None:
    from src.harness.discovery.scout_preflight import run_scout_preflight
    from src.tools.registry import ToolRegistry
    from src.tools.search.grep import GrepSearchTool

    app = tmp_path / "app.py"
    app.write_text(
        "def register():\n    db.session.commit()\n",
        encoding="utf-8",
    )
    registry = ToolRegistry()
    registry.register(GrepSearchTool())
    monkeypatch.chdir(tmp_path)

    out = await run_scout_preflight(registry, "为什么注册接口报数据库事务异常")
    assert "app.py" in out
    assert "register" in out.lower() or "commit" in out.lower()
