import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.decision import CursorDecisionLLM
from src.agent.executor import CursorExecutor
from src.agent.patch_applier import CursorPatchApplier
from src.agent.types import ToolResult
from src.agent.validator import CursorValidator, _collect_view_schemas, _sql_alias_safety
from src.harness.cursor.manager import CursorStateManager
from src.hooks.post_tool_context import apply_post_tool_context_hook


def test_validator_reports_exact_python_syntax_coordinates(tmp_path: Path) -> None:
    validator = CursorValidator(tmp_path)

    result = validator.validate_ast(
        "target.py",
        "def run():\n    return 1\n",
        "def run():\n  return (\n",
    )

    assert result["pass"] is False
    assert "python_syntax_error" in str(result["issues"])
    trace = result["trace"]
    assert trace["exception_type"] == "SyntaxError"
    assert trace["line_number"] == 2
    assert isinstance(trace["offset"], int)


def test_loop_cost_model_records_but_never_overrides_action() -> None:
    manager = CursorStateManager(max_bytes=8192)
    state = manager.initial("inspect", max_steps=3)

    state = manager.record_decision(
        state,
        action="ask_clarify",
        target_file="",
        can_answer=True,
    )

    assert state.retry_bias == 1
    assert state.decision_cost_total == 0.8
    assert state.decision_signatures[-1].startswith("ask_clarify|-")
    assert state.status == "running"
    retry = manager.mark_retry(state)
    assert retry.current_step == 2


@pytest.mark.asyncio
async def test_rolled_back_attempt_is_compiled_into_next_prompt(tmp_path: Path) -> None:
    target = tmp_path / "target.py"
    target.write_text("def run():\n    return 1\n", encoding="utf-8")
    patch = (
        "<<<<<<< SEARCH\n"
        "    return 1\n"
        "=======\n"
        "  return (\n"
        ">>>>>>> REPLACE"
    )
    executor = CursorExecutor(tmp_path, CursorPatchApplier(tmp_path))
    validator = CursorValidator(tmp_path, command=("true",))

    execution, validation, _ = await executor.execute_transaction(
        "target.py", patch, validator, user_intent="repair run",
    )

    assert execution.rolled_back is True
    assert target.read_text(encoding="utf-8") == "def run():\n    return 1\n"
    assert validation.success is False
    manager = CursorStateManager(max_bytes=8192)
    state = manager.initial("repair run", max_steps=3)
    state = manager.after_execution(state, "target.py", patch, execution)
    state = manager.after_validation(state, validation, patch=patch, execution=execution)
    state_text = manager.format_for_prompt(state)

    assert "EXECUTION TRACE LAYER" in state_text
    assert "SyntaxError" in state_text
    assert "line=2" in state_text
    assert "PATCH MEMORY LAYER" in state_text
    assert "STATE DIFF EVOLUTION LAYER" in state_text
    assert "target.py@base" in state_text
    assert "+  return (" in state_text

    messages = CursorDecisionLLM(MagicMock()).build_messages(
        state_text=state_text,
        context_pack=MagicMock(windows=(), candidate_files=()),
        hint=None,
    )
    assert "STATE DIFF EVOLUTION LAYER" in messages[1]["content"]


def test_validator_cache_refreshes_when_mtime_is_unchanged(tmp_path: Path) -> None:
    # 1. Create a schema file defining a view
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        "CREATE VIEW my_view AS SELECT id FROM my_table;\n", encoding="utf-8"
    )
    os.utime(schema_file, (1000, 1000))

    # 2. Collect view schemas -> should load my_view with field 'id'
    schemas = _collect_view_schemas(
        tmp_path, target_file="other.py", original_content="", patched_content=""
    )
    assert "my_view" in schemas
    assert schemas["my_view"]["fields"] == {"id"}

    # 3. Modify the schema file to add a column, keeping the same mtime to trick the st_mtime check
    schema_file.write_text(
        "CREATE VIEW my_view AS SELECT id, name FROM my_table;\n", encoding="utf-8"
    )
    os.utime(schema_file, (1000, 1000))

    # Collection must notice the edit even though st_mtime was forced back.
    schemas_cached = _collect_view_schemas(
        tmp_path, target_file="other.py", original_content="", patched_content=""
    )
    assert schemas_cached["my_view"]["fields"] == {"id", "name"}

    # 4. Invalidate cache for the schema file using the hook system
    apply_post_tool_context_hook(
        tool_name="decision_edit",
        arguments={"target_file": "schema.sql"},
        result=ToolResult(success=True, output="", error=None, metadata={}),
    )

    # Explicit hook invalidation remains supported as well.
    schemas_fresh = _collect_view_schemas(
        tmp_path, target_file="other.py", original_content="", patched_content=""
    )
    assert schemas_fresh["my_view"]["fields"] == {"id", "name"}


def test_validator_parses_view_expressions_with_commas(tmp_path: Path) -> None:
    schema_file = tmp_path / "schema.sql"
    schema_file.write_text(
        "CREATE VIEW order_view AS "
        "SELECT id, COALESCE(first_name, last_name) AS display_name "
        "FROM orders;\n",
        encoding="utf-8",
    )
    validator = CursorValidator(tmp_path)

    result = validator.validate_schema(
        target_file="api.py",
        patch="SELECT id, display_name FROM order_view",
        original_content="SELECT id, first_name, last_name FROM orders",
        patched_content="SELECT id, display_name FROM order_view",
    )

    assert result["pass"] is True
    assert result["missing_fields"] == []


def test_validator_schema_error_reports_missing_fields(tmp_path: Path) -> None:
    (tmp_path / "schema.sql").write_text(
        "CREATE VIEW order_view AS SELECT id FROM orders;\n",
        encoding="utf-8",
    )
    validator = CursorValidator(tmp_path)

    result = validator.validate_schema(
        target_file="api.py",
        patch="SELECT id, missing_name FROM order_view",
        original_content="",
        patched_content="SELECT id, missing_name FROM order_view",
    )

    assert result["issues"] == ["SELECT_FIELD_NOT_IN_VIEW"]
    assert result["missing_fields"] == ["missing_name"]


def test_sql_alias_safety_accepts_mysql_trigger_pseudo_rows() -> None:
    sql = """
    CREATE TRIGGER tg_order_timeline
    AFTER UPDATE ON ticket_order
    FOR EACH ROW
    BEGIN
        IF NEW.status <> OLD.status THEN
            INSERT INTO order_timeline (order_id, status)
            VALUES (NEW.order_id, NEW.status);
        END IF;
    END;
    """

    result = _sql_alias_safety(sql)

    assert result["pass"] is True
    assert result["trigger_pseudo_aliases"] == ["new", "old"]
    assert result["dead_aliases"] == []


def test_sql_alias_safety_does_not_allow_pseudo_rows_outside_trigger() -> None:
    result = _sql_alias_safety("SELECT NEW.status FROM ticket_order")

    assert result["pass"] is False
    assert result["trigger_pseudo_aliases"] == []
    assert result["dead_aliases"] == ["new"]


def test_sql_alias_safety_keeps_rejecting_real_dead_alias_inside_trigger() -> None:
    sql = """
    CREATE TRIGGER tg_order_timeline
    AFTER UPDATE ON ticket_order
    FOR EACH ROW
    BEGIN
        INSERT INTO order_timeline (order_id, status)
        VALUES (missing_alias.order_id, NEW.status);
    END;
    """

    result = _sql_alias_safety(sql)

    assert result["pass"] is False
    assert result["dead_aliases"] == ["missing_alias"]


def test_validator_schema_accepts_order_timeline_trigger(tmp_path: Path) -> None:
    sql = """
    CREATE TABLE IF NOT EXISTS order_timeline (
        timeline_id INT AUTO_INCREMENT PRIMARY KEY,
        order_id INT NOT NULL,
        status VARCHAR(20) NOT NULL
    );
    CREATE TRIGGER tg_order_timeline
    AFTER UPDATE ON ticket_order
    FOR EACH ROW
    BEGIN
        IF NEW.status <> OLD.status THEN
            INSERT INTO order_timeline (order_id, status)
            VALUES (NEW.order_id, NEW.status);
        END IF;
    END;
    """
    validator = CursorValidator(tmp_path)

    result = validator.validate_schema(
        target_file="db/init/init.sql",
        patch=sql,
        original_content="",
        patched_content=sql,
    )

    assert result["pass"] is True
    assert "DEAD_SQL_ALIAS" not in result["issues"]


def test_validator_schema_skips_python_import_only_patch(tmp_path: Path) -> None:
    original = (
        "from fastapi import APIRouter, HTTPException\n"
        "from sqlalchemy import text\n"
        "from sqlalchemy.engine import Engine\n\n"
        "def build_router(engine: Engine) -> APIRouter:\n"
        "    router = APIRouter()\n"
        "    with engine.connect() as conn:\n"
        '        conn.execute(text("SELECT o.id FROM ticket_order o"), {})\n'
        "    return router\n"
    )
    patched = (
        "import logging\n"
        "from fastapi import APIRouter, HTTPException\n"
        "from sqlalchemy import text\n"
        "from sqlalchemy.engine import Engine\n"
        "from sqlalchemy.exc import SQLAlchemyError\n\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def _handle_db_error(exc: Exception, route_name: str) -> None:\n"
        '    logger.exception("Database error in %s", route_name, exc)\n'
        '    raise HTTPException(status_code=500, detail="db error")\n\n'
        "def build_router(engine: Engine) -> APIRouter:\n"
        "    router = APIRouter()\n"
        "    with engine.connect() as conn:\n"
        '        conn.execute(text("SELECT o.id FROM ticket_order o"), {})\n'
        "    return router\n"
    )
    patch = (
        "<<<<<<< SEARCH\n"
        "from fastapi import APIRouter, HTTPException\n"
        "from sqlalchemy import text\n"
        "from sqlalchemy.engine import Engine\n\n"
        "def build_router(engine: Engine) -> APIRouter:\n"
        "=======\n"
        "import logging\n"
        "from fastapi import APIRouter, HTTPException\n"
        "from sqlalchemy import text\n"
        "from sqlalchemy.engine import Engine\n"
        "from sqlalchemy.exc import SQLAlchemyError\n\n"
        "logger = logging.getLogger(__name__)\n\n"
        "def _handle_db_error(exc: Exception, route_name: str) -> None:\n"
        '    logger.exception("Database error in %s", route_name, exc)\n'
        '    raise HTTPException(status_code=500, detail="db error")\n\n'
        "def build_router(engine: Engine) -> APIRouter:\n"
        ">>>>>>> REPLACE"
    )
    validator = CursorValidator(tmp_path)

    result = validator.validate_schema(
        target_file="list.py",
        patch=patch,
        original_content=original,
        patched_content=patched,
    )

    assert result["pass"] is True
    assert "DEAD_SQL_ALIAS" not in result["issues"]


def test_validator_schema_still_flags_dead_alias_inside_sql_literal(tmp_path: Path) -> None:
    original = (
        "from sqlalchemy import text\n\n"
        'SQL = "SELECT o.id FROM ticket_order o"\n'
    )
    patched = (
        "from sqlalchemy import text\n\n"
        'SQL = "SELECT bad_alias.id FROM ticket_order o"\n'
    )
    patch = (
        "<<<<<<< SEARCH\n"
        'SQL = "SELECT o.id FROM ticket_order o"\n'
        "=======\n"
        'SQL = "SELECT bad_alias.id FROM ticket_order o"\n'
        ">>>>>>> REPLACE"
    )
    validator = CursorValidator(tmp_path)

    result = validator.validate_schema(
        target_file="list.py",
        patch=patch,
        original_content=original,
        patched_content=patched,
    )

    assert "DEAD_SQL_ALIAS" in result["issues"]
