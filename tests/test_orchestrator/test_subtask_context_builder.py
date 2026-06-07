from __future__ import annotations

from src.context.pack import ContextPack
from src.orchestrator.orchestrator import (
    _apply_context_pack_to_subtask,
    _append_prior_edit_context,
    _previous_handoff_from_summaries,
    _subtask_context_request,
)
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode


def test_context_pack_applies_only_to_current_subtask() -> None:
    current = SubTaskNode(
        id="st-1",
        kind=SubTaskKind.DIAGNOSE,
        description="find target",
    )
    other = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="edit target",
    )
    pack = ContextPack(
        user_request="find",
        candidate_files=({"file": "main.py", "score": 0.95, "reasons": ["symbol_match"]},),
    )

    _apply_context_pack_to_subtask(current, pack)

    assert current.context_files == ["main.py"]
    assert other.context_files == []


def test_context_pack_does_not_override_planner_scope() -> None:
    node = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="edit target",
        context_files=["planner.py"],
    )
    pack = ContextPack(
        user_request="edit",
        candidate_files=({"file": "main.py", "score": 0.95, "reasons": ["symbol_match"]},),
    )

    _apply_context_pack_to_subtask(node, pack)

    assert node.context_files == ["planner.py"]


def test_subtask_context_request_contains_subtask_contract() -> None:
    node = SubTaskNode(
        id="st-2",
        kind=SubTaskKind.EDIT,
        description="修改查询接口",
        acceptance_criteria="changed_files and validation_plan",
        context_files=["main.py"],
        depends_on=["st-1"],
    )

    text = _subtask_context_request("根任务", node)

    assert "Root task: 根任务" in text
    assert "Subtask [st-2]" in text
    assert "Acceptance:" in text
    assert "Context files: main.py" in text
    assert "Depends on: st-1" in text


def test_previous_handoff_from_summaries_extracts_json_evidence() -> None:
    handoff = _previous_handoff_from_summaries({
        "st-1": (
            '{"result":"found","acceptance_met":true,'
            '"evidence":[{"file":"main.py","symbol":"query_orders"}],'
            '"blocker":"no matches in tests"}'
        )
    })

    assert handoff["evidence"][0]["source_subtask"] == "st-1"
    assert handoff["known_negatives"][0]["reason"] == "no matches in tests"


def test_previous_handoff_from_agent_output_schema() -> None:
    handoff = _previous_handoff_from_summaries({
        "st-1": (
            '{"status":"need_more_context","changed_files":[],'
            '"validation":{"ran":[],"result":"skipped","summary":""},'
            '"risks":[],"handoff":{"facts":["checked"],'
            '"evidence":[{"path":"main.py","line":1,"symbol":"query_orders"}],'
            '"known_negatives":[{"query":"boarding_pass","reason":"no direct hit"}],'
            '"next_focus":["main.py"]}}'
        )
    })

    assert handoff["evidence"][0]["path"] == "main.py"
    assert handoff["known_negatives"][0]["query"] == "boarding_pass"
    assert handoff["facts"][0]["fact"] == "checked"
    assert handoff["next_focus"][0]["focus"] == "main.py"


def test_prior_edit_context_is_appended_to_edit_search_output() -> None:
    prior = {
        "st-1": (
            "Result: found\n\n"
            "EDIT_CONTEXT_JSON\n"
            '{"schema":"mitkii.edit_context.v1","code_edit_ready":true,'
            '"builder":"EditPlanBuilder",'
            '"snippets":[{"file":"app.py","start_line":1,'
            '"end_line":2,"current_code":"def x():\\n    pass",'
            '"intended_change":"change",'
            '"acceptance_criteria":["changed"]}],'
            '"editable_targets":[{"file":"app.py","start_line":1,'
            '"end_line":2,"current_code":"def x():\\n    pass",'
            '"intended_change":"change",'
            '"acceptance_criteria":["changed"]}],'
            '"intended_change":"change",'
            '"acceptance_criteria":["changed"],'
            '"tool_policy":{"allowed_tools":["edit_file"],"scope":["app.py"]}}'
        )
    }

    out = _append_prior_edit_context("current search", prior)

    assert "current search" in out
    assert "EDIT_CONTEXT_JSON" in out
    assert '"builder":"EditPlanBuilder"' in out
    assert '"snippets"' in out
    assert '"editable_targets"' in out


def test_prior_edit_context_is_merged_when_current_exists() -> None:
    prior = {
        "st-1": (
            "Result: found\n\n"
            "EDIT_CONTEXT_JSON\n"
            '{"schema":"mitkii.edit_context.v1","code_edit_ready":true,'
            '"builder":"EditPlanBuilder",'
            '"target_view":"v_order_detail",'
            '"available_views":["v_order_detail"]}'
        )
    }
    
    current_search_output = (
        "current search\n\n"
        "EDIT_CONTEXT_JSON\n"
        '{"schema":"mitkii.edit_context.v1","code_edit_ready":true,'
        '"builder":"EditPlanBuilder",'
        '"editable_targets":[]}'
    )
    
    out = _append_prior_edit_context(current_search_output, prior)
    
    assert "v_order_detail" in out
    assert '"target_view": "v_order_detail"' in out
    assert '"available_views": [\n    "v_order_detail"\n  ]' in out
