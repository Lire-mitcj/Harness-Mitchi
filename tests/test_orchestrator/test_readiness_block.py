from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.orchestrator.orchestrator import _analysis_for_edit, _append_effective_edit_context, OrchestratorLoop
from src.harness.task_analysis import HarnessTaskAnalysis
from src.planner.task_tree import SubTaskKind, SubTaskNode, TaskTree
from src.agent.events import EventType


def test_analysis_for_edit_missing_intent() -> None:
    analysis = HarnessTaskAnalysis(
        intent="general_edit",
        confidence=1.0,
        edit_ready=False,
        edit_strategy="general_edit",
        complexity="high",
    )
    prior = {"st-1": "Diagnose finished."}
    
    data = _analysis_for_edit(analysis, prior)
    assert data["edit_ready"] is False
    assert data["readiness_checks"]["intent_resolved"] is False
    assert data["readiness_checks"]["targets_resolved"] is False


def test_analysis_for_edit_incomplete_intent() -> None:
    analysis = HarnessTaskAnalysis(
        intent="general_edit",
        confidence=1.0,
        edit_ready=False,
        edit_strategy="general_edit",
        complexity="high",
    )
    prior = {
        "st-1": "PATCH_INTENT_JSON\n{\n  \"edit_strategy\": \"general_edit\"\n}"
    }
    
    data = _analysis_for_edit(analysis, prior)
    assert data["edit_ready"] is False
    assert data["readiness_checks"]["targets_resolved"] is False


@pytest.mark.asyncio
async def test_run_edit_skill_executor_readiness_block() -> None:
    loop_mock = MagicMock(spec=OrchestratorLoop)
    loop_mock.state = MagicMock()
    loop_mock.state.task_analysis = HarnessTaskAnalysis(
        intent="general_edit",
        confidence=1.0,
        edit_ready=False,
        edit_strategy="general_edit",
        complexity="high",
    )
    loop_mock.state.subtask_summaries = {}
    
    # Mock code_search success return
    search_result = MagicMock()
    search_result.success = True
    search_result.metadata = {"edit_context_targets": "1", "search_output": "evidence"}
    loop_mock._skill_executor = MagicMock()
    loop_mock._skill_executor.run = AsyncMock(return_value=search_result)
    
    from src.agent.events import AgentEvent
    loop_mock._skill_failure_stream_end = MagicMock(
        return_value=AgentEvent(type=EventType.STREAM_END, data={"success": False})
    )
    
    task_tree = TaskTree(
        root_task="fix code",
        nodes=[
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="modify lines",
            )
        ]
    )
    
    events = []
    async for event in OrchestratorLoop._run_edit_skill_executor(
        loop_mock,
        user_msg="fix code",
        task_tree=task_tree,
        node=task_tree.nodes[0],
        project_structure="{}",
        context_pack=None,
    ):
        events.append(event)
        
    # The runner should have completed code_search but blocked code_edit execution
    assert loop_mock._skill_executor.run.call_count == 1
    assert loop_mock._skill_executor.run.call_args_list[0][0][0] == "code_search"
    
    status_events = [e for e in events if e.type == EventType.STATUS]
    assert len(status_events) >= 1
    assert "code_edit blocked: Harness edit_ready=false" in status_events[-1].content
    
    end_events = [e for e in events if e.type == EventType.STREAM_END]
    assert len(end_events) == 1
    assert end_events[0].data["success"] is False


def test_analysis_for_edit_global_summaries_fallback_and_list_acceptance() -> None:
    analysis = HarnessTaskAnalysis(
        intent="general_edit",
        confidence=1.0,
        edit_ready=False,
        edit_strategy="general_edit",
        complexity="high",
    )
    prior = {"st-1": "Diagnose finished."}
    
    global_summaries = {
        "st-1": "Diagnose finished.",
        "st-2": """PATCH_INTENT_JSON
{
  "edit_strategy": "general_edit",
  "edit_ready": true,
  "edit_targets": [
    {
      "file": "main.py",
      "symbol": "foo",
      "line_start": 10,
      "line_end": 20,
      "snippet": "def foo(): pass",
      "decision": "keep"
    }
  ],
  "dependencies": [],
  "acceptance_criteria": ["test passes"]
}
"""
    }
    
    data = _analysis_for_edit(analysis, prior, global_summaries)
    assert data["edit_ready"] is True
    assert data["readiness_checks"]["intent_resolved"] is True
    assert data["readiness_checks"]["targets_resolved"] is True
    assert data["readiness_checks"]["acceptance_resolved"] is True
    assert data["acceptance_contract"] == {"criteria": ["test passes"]}


def test_analysis_for_edit_strategy_override_and_design_ready() -> None:
    analysis = HarnessTaskAnalysis(
        intent="general_edit",
        confidence=1.0,
        edit_ready=False,
        edit_strategy="general_edit",
        complexity="high",
    )
    prior = {
        "st-1": """PATCH_INTENT_JSON
{
  "edit_strategy": "sql_view_rewrite",
  "edit_ready": true,
  "edit_targets": [
    {
      "file": "main.py",
      "symbol": "foo",
      "line_start": 10,
      "line_end": 20,
      "snippet": "def foo(): pass",
      "decision": "keep"
    }
  ],
  "dependencies": [{"kind": "database_view", "name": "view_a"}],
  "acceptance_criteria": ["test passes"],
  "target_view": "view_a"
}
"""
    }
    
    data = _analysis_for_edit(analysis, prior)
    assert data["edit_ready"] is True
    assert data["edit_strategy"] == "sql_view_rewrite"
    assert data["target_view"] == "view_a"
    assert data["patch_intent"]["target_view"] == "view_a"
    assert data["dependencies"][0]["name"] == "view_a"
    assert data["readiness_checks"]["intent_resolved"] is True
    assert data["readiness_checks"]["dependencies_resolved"] is True


def test_effective_edit_context_prefers_patch_intent_over_search_hydration() -> None:
    search_ctx = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "dependencies_resolved": False,
        "target_view": "",
        "resolved_dependencies": [],
        "editable_targets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 2,
            "current_code": "def query():\n    return 'SELECT * FROM orders'\n",
        }],
        "intended_change": "use view",
        "acceptance_criteria": ["query uses view"],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }
    patch_intent = {
        "edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "target_view": "view_ticket_report_detail",
        "edit_targets": [{
            "file": "main.py",
            "symbol": "query",
            "line_start": 1,
            "line_end": 2,
            "snippet": "def query(): pass",
            "decision": "use view_ticket_report_detail",
        }],
        "dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "view_ticket_report_detail",
            "columns": ["id"],
            "replaces_objects": ["orders"],
        }],
        "acceptance_criteria": ["query uses view"],
    }

    output = _append_effective_edit_context(
        "EDIT_CONTEXT_JSON\n" + json.dumps(search_ctx),
        handoff_contract={},
        edit_analysis={
            "edit_ready": True,
            "edit_strategy": "sql_view_rewrite",
            "patch_intent": patch_intent,
        },
    )
    payload = json.loads(output.split("EDIT_CONTEXT_JSON", 1)[1])

    assert payload["target_view"] == "view_ticket_report_detail"
    assert payload["dependencies_resolved"] is True
    assert payload["resolved_dependencies"][0]["name"] == "view_ticket_report_detail"
    assert payload["editable_targets"][0]["current_code"] == search_ctx["editable_targets"][0]["current_code"]
    assert payload["editable_targets"][0]["symbol"] == "query"
