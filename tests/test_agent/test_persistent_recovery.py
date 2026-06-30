from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.decision import CursorDecisionLLM
from src.agent.executor import CursorExecutor
from src.agent.patch_applier import CursorPatchApplier
from src.agent.validator import CursorValidator
from src.harness.cursor.manager import CursorStateManager


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
