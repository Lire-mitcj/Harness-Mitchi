from __future__ import annotations

from unittest.mock import MagicMock
from src.agent.run_state import RunPhase
from src.hooks.reallocate_tools import (
    determine_allowed_tools,
    extract_actionable_plan_items,
)


class DummyState:
    def __init__(self, retrieval_complete: bool, phase: RunPhase, task_mode: str) -> None:
        self.run_state = MagicMock()
        self.run_state.retrieval_complete = retrieval_complete
        self.run_state.phase = phase
        self.run_state.task_mode = task_mode


def test_extracts_i_need_to_numbered_plan() -> None:
    content = "I need to:\n1. Create a table\n2. Implement the endpoint"

    assert extract_actionable_plan_items(content) == (
        "Create a table",
        "Implement the endpoint",
    )


def test_extracts_only_last_plan_section() -> None:
    content = (
        "1. Create an exploratory draft\n"
        "2. Update an exploratory draft\n\n"
        "## Plan\n"
        "1. Create the timeline table\n"
        "2. Implement the endpoint\n"
    )

    assert extract_actionable_plan_items(content) == (
        "Create the timeline table",
        "Implement the endpoint",
    )


def test_model_checkmarks_do_not_mark_unexecuted_plan_complete() -> None:
    content = (
        "**KANBAN_CHECKLIST:**\n"
        "1. ✅ Add the timeline table\n"
        "2. ⬜ Implement the endpoint\n"
    )

    assert extract_actionable_plan_items(content) == (
        "Add the timeline table",
        "Implement the endpoint",
    )


def test_determine_allowed_tools_not_complete() -> None:
    # 1. Retrieval not complete, not disabled, no compile error
    state = DummyState(retrieval_complete=False, phase=RunPhase.RETRIEVING, task_mode="edit")
    gravity_controller = MagicMock()
    gravity_controller.retrieval_disabled = False
    
    default_tools = frozenset({"grep_search", "decision_edit"})
    allowed = determine_allowed_tools(state, gravity_controller, default_tools, has_compile_error=False)
    
    assert allowed == default_tools


def test_determine_allowed_tools_complete_edit_mode() -> None:
    # 2. Retrieval complete (via retrieval_complete=True), task_mode="edit"
    state = DummyState(retrieval_complete=True, phase=RunPhase.ACTING, task_mode="edit")
    gravity_controller = MagicMock()
    gravity_controller.retrieval_disabled = False
    
    default_tools = frozenset({"grep_search", "decision_edit", "view_symbol_code"})
    allowed = determine_allowed_tools(state, gravity_controller, default_tools, has_compile_error=False)
    
    assert allowed == frozenset({"grep_search", "view_symbol_code", "decision_edit"})


def test_determine_allowed_tools_disabled_diagnose_mode() -> None:
    # 3. Retrieval disabled via gravity_controller, task_mode="diagnose"
    state = DummyState(retrieval_complete=False, phase=RunPhase.RETRIEVING, task_mode="diagnose")
    gravity_controller = MagicMock()
    gravity_controller.retrieval_disabled = True
    
    default_tools = frozenset({"grep_search", "decision_edit", "view_symbol_code"})
    allowed = determine_allowed_tools(state, gravity_controller, default_tools, has_compile_error=False)
    
    assert allowed == frozenset()


def test_determine_allowed_tools_disabled_edit_mode() -> None:
    state = DummyState(
        retrieval_complete=True,
        phase=RunPhase.ACTING,
        task_mode="edit",
    )
    allowed = determine_allowed_tools(
        state,
        MagicMock(retrieval_disabled=True),
        frozenset({"grep_search", "view_symbol_code", "decision_edit"}),
    )

    assert allowed == frozenset({"decision_edit"})


def test_determine_allowed_tools_responding_phase_diagnose_mode() -> None:
    # 4. Phase is RESPONDING, task_mode="diagnose"
    state = DummyState(retrieval_complete=False, phase=RunPhase.RESPONDING, task_mode="diagnose")
    gravity_controller = MagicMock()
    gravity_controller.retrieval_disabled = False
    
    default_tools = frozenset({"grep_search", "decision_edit", "view_symbol_code"})
    allowed = determine_allowed_tools(state, gravity_controller, default_tools, has_compile_error=False)
    
    assert allowed == frozenset({"grep_search", "view_symbol_code", "decision_edit"})


def test_determine_allowed_tools_exemption_with_compile_error() -> None:
    # 5. Retrieval is complete, but there is a compilation error -> must return all default tools (exemption)
    state = DummyState(retrieval_complete=True, phase=RunPhase.ACTING, task_mode="edit")
    gravity_controller = MagicMock()
    gravity_controller.retrieval_disabled = True
    
    default_tools = frozenset({"grep_search", "decision_edit", "codebase_retrieve", "view_symbol_code"})
    allowed = determine_allowed_tools(state, gravity_controller, default_tools, has_compile_error=True)
    
    assert allowed == default_tools


def test_redundant_retrieval_round_forces_edit() -> None:
    state = DummyState(retrieval_complete=True, phase=RunPhase.ACTING, task_mode="edit")
    allowed = determine_allowed_tools(
        state,
        MagicMock(retrieval_disabled=False),
        frozenset({"grep_search", "view_symbol_code", "decision_edit"}),
        retrieval_round_saturated=True,
    )

    assert allowed == frozenset({"decision_edit"})


def test_numbered_actionable_plan_forces_edit() -> None:
    state = DummyState(retrieval_complete=True, phase=RunPhase.ACTING, task_mode="edit")
    state.checklist = ("[ ] Create a table", "[ ] Implement the endpoint")
    allowed = determine_allowed_tools(
        state,
        MagicMock(retrieval_disabled=False),
        frozenset({"grep_search", "view_symbol_code", "decision_edit"}),
    )

    assert allowed == frozenset({"decision_edit"})
