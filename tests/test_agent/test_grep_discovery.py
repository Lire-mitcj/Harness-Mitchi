from __future__ import annotations

from src.agent.grep_discovery import (
    discovery_hint_lines,
    grep_patterns_for_task,
    grep_scope_for_task,
    themes_for_task,
    view_symbol_avoid,
)
from src.agent.manifest import execution_card
from src.agent.run_state import requirements_for_task


TASK = "你把统一数据库异常日志接口接到现有的与数据库有关的接口上"


def test_themes_for_db_exception_integration_task() -> None:
    themes = themes_for_task(TASK)
    assert "exception_handler" in themes
    assert "db_integration" in themes
    assert "endpoint" in themes
    assert "schema" in themes


def test_grep_patterns_prefers_handler_defs_not_import_names() -> None:
    patterns = grep_patterns_for_task(TASK)
    joined = " ".join(patterns)
    assert "@app\\.exception_handler" in joined or "add_exception_handler" in joined
    assert "async def handle_" in joined or "def _handle_" in joined
    assert "SQLAlchemyError" not in patterns
    assert "exception_handler" not in patterns


def test_grep_scope_targets_main_py_for_unified_handler() -> None:
    include, path = grep_scope_for_task(TASK)
    assert include == "main.py"
    assert path == "main.py"


def test_discovery_hints_include_task_batch_and_avoid_list() -> None:
    hints = discovery_hint_lines(TASK, sorted(requirements_for_task(TASK)))
    blob = "\n".join(hints)
    assert "task_batch" in blob
    assert "exception_handler" in blob
    assert "avoid_view_symbols" in blob
    assert "SQLAlchemyError" in blob
    assert "suggested_views" in blob


def test_execution_card_surfaces_task_derived_discovery_hints() -> None:
    card = execution_card(
        manifest=__import__("src.agent.manifest", fromlist=["StepManifest"]).StepManifest(),
        tools_available=["grep_search", "view_symbol_code"],
        task_slots=sorted(requirements_for_task(TASK)),
        task_text=TASK,
    )
    assert "discovery_hints" in card
    assert "task_batch" in card
    assert "avoid_view_symbols" in card
    assert "main.py" in card


def test_view_symbol_avoid_for_exception_tasks() -> None:
    avoid = view_symbol_avoid(TASK)
    assert "SQLAlchemyError" in avoid
    assert "logger" in avoid
