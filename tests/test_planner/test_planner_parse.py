from __future__ import annotations

import json

from src.planner.patch_plan_parse import parse_patch_plan_output
from src.planner.planner_node import _parse_task_tree, parse_planner_output
from src.planner.planner_parse import (
    build_task_tree_from_payload,
    normalize_planner_payload,
    validate_planner_payload,
)
from src.planner.task_tree import SubTaskKind


def test_normalize_fills_missing_depends_on() -> None:
    payload = {
        "root_task": "x",
        "nodes": [
            {
                "id": "st-1",
                "kind": "diagnose",
                "description": "search",
                "acceptance_criteria": "paths",
                "allowed_tools": ["grep_search"],
                "context_files": [],
            },
            {
                "id": "st-2",
                "kind": "edit",
                "description": "fix",
                "acceptance_criteria": "patched",
                "allowed_tools": ["edit_file"],
                "context_files": [],
            },
        ],
    }
    normalized = normalize_planner_payload(payload)
    errors = validate_planner_payload(normalized)
    assert normalized["nodes"][1]["depends_on"] == ["st-1"]
    assert normalized["nodes"][0]["needs_l1"] is False
    assert not errors


def test_validate_payload_allows_edit_first() -> None:
    payload = {
        "root_task": "Add health endpoint",
        "nodes": [{
            "id": "st-1",
            "kind": "edit",
            "description": "Create health route",
            "acceptance_criteria": "GET /health returns 200",
            "allowed_tools": ["write_file", "edit_file"],
            "context_files": ["app/health.py"],
            "depends_on": [],
        }],
    }
    errors = validate_planner_payload(payload)
    assert not errors


def test_parser_preserves_artifact_and_write_scope_fields() -> None:
    payload = {
        "root_task": "Switch report SQL to view",
        "nodes": [{
            "id": "st-1",
            "kind": "edit",
            "description": "Rewrite report SQL",
            "acceptance_criteria": "report uses view",
            "allowed_tools": ["context_search", "edit_file"],
            "context_files": [],
            "depends_on": [],
            "requires_artifacts": ["database_view"],
            "produces_artifacts": ["patch_intent"],
            "write_scope": ["app/report.py"],
        }],
    }

    normalized = normalize_planner_payload(payload)
    errors = validate_planner_payload(normalized)
    tree = build_task_tree_from_payload(normalized, fallback_task="fallback")

    assert not errors
    assert tree.nodes[0].requires_artifacts == ["database_view"]
    assert tree.nodes[0].produces_artifacts == ["patch_intent"]
    assert tree.nodes[0].write_scope == ["app/report.py"]


def test_parse_planner_output_no_silent_ok_on_bad_json() -> None:
    result = parse_planner_output("{not json", fallback_task="task")
    assert not result.ok
    assert not result.json_ok
    assert result.tree.nodes == []


def test_parse_planner_output_ok_minimal_plan() -> None:
    payload = {
        "root_task": "List views",
        "nodes": [{
            "id": "st-1",
            "kind": "diagnose",
            "description": "Find view files",
            "acceptance_criteria": "paths listed",
            "allowed_tools": ["list_dir", "grep_search"],
            "context_files": [],
            "depends_on": [],
        }],
    }
    result = parse_planner_output(json.dumps(payload), fallback_task="List views")
    assert result.ok
    assert result.tree.nodes[0].kind == SubTaskKind.DIAGNOSE


def test_legacy_parse_fallback_still_works() -> None:
    tree = _parse_task_tree("{bad", fallback_task="task")
    assert len(tree.nodes) == 1
    assert tree.nodes[0].kind == SubTaskKind.DIAGNOSE


def test_parse_patch_plan_output_ok() -> None:
    raw = json.dumps({
        "patch_plan": {
            "files_to_edit": ["main.py"],
            "target_symbols": ["query_orders"],
            "intended_changes": ["switch query to view"],
            "edits": [{
                "path": "main.py",
                "symbol": "query_orders",
                "old_string": "SELECT * FROM orders",
                "new_string": "SELECT * FROM view_ticket_report_detail",
            }],
            "validation_plan": ["pytest"],
            "requires_confirmation": False,
            "confidence": 0.91,
            "missing_info": [],
        }
    })

    result = parse_patch_plan_output(raw)

    assert result.ok
    assert result.patch_plan is not None
    assert result.patch_plan.is_executable()
    assert result.patch_plan.edits[0].path == "main.py"


def test_parse_patch_plan_rejects_missing_edits() -> None:
    result = parse_patch_plan_output(
        json.dumps({
            "patch_plan": {
                "files_to_edit": ["main.py"],
                "intended_changes": ["change query"],
                "confidence": 0.9,
                "missing_info": [],
            }
        })
    )

    assert not result.ok
    assert "edits" in "; ".join(result.all_errors)


def test_parse_patch_plan_reports_task_tree_shape() -> None:
    result = parse_patch_plan_output(
        json.dumps({
            "root_task": "fix",
            "nodes": [],
        })
    )

    assert not result.ok
    assert "TaskTree" in "; ".join(result.all_errors)
