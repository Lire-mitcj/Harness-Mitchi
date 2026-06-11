from __future__ import annotations

import json

from src.harness.subtask.artifacts import build_artifact_store


def test_artifact_store_extracts_explicit_artifacts() -> None:
    summary = json.dumps({
        "status": "success",
        "changed_files": [],
        "validation": {"ran": [], "result": "skipped", "summary": ""},
        "risks": [],
        "handoff": {
            "artifacts": [
                {
                    "kind": "database_view",
                    "canonical_id": "database_view:v_order_detail",
                    "name": "v_order_detail",
                    "columns": [{"canonical_name": "order_id", "type": "int"}],
                    "confidence": 0.9,
                }
            ]
        },
    })

    store = build_artifact_store({"st-1": summary})

    assert store["policy"]["artifacts_are_hints"] is True
    assert store["policy"]["may_not_block_edit"] is True
    assert store["artifacts"][0]["kind"] == "database_view"
    assert store["artifacts"][0]["name"] == "v_order_detail"


def test_artifact_store_marks_field_conflicts_as_warnings() -> None:
    first = json.dumps({
        "status": "success",
        "changed_files": [],
        "validation": {"ran": [], "result": "skipped", "summary": ""},
        "risks": [],
        "handoff": {
            "artifacts": [
                {
                    "kind": "database_view",
                    "canonical_id": "database_view:v_report",
                    "name": "v_report",
                    "columns": [{"canonical_name": "id", "type": "int"}],
                }
            ]
        },
    })
    second = json.dumps({
        "status": "success",
        "changed_files": [],
        "validation": {"ran": [], "result": "skipped", "summary": ""},
        "risks": [],
        "handoff": {
            "artifacts": [
                {
                    "kind": "database_view",
                    "canonical_id": "database_view:v_report",
                    "name": "v_report",
                    "columns": [{"canonical_name": "id", "type": "uuid"}],
                }
            ]
        },
    })

    store = build_artifact_store({"st-1": first, "st-2": second})

    assert store["artifacts"][0]["conflicts"][0]["column"] == "id"
    assert store["warnings"][0]["hint"].startswith("Verify this field locally")


def test_artifact_store_extracts_edit_context_marker() -> None:
    summary = (
        "Result: found target\n\n"
        "EDIT_CONTEXT_JSON\n"
        + json.dumps({
            "target_view": "v_ticket_report",
            "available_views": [{"name": "v_ticket_report", "fields": []}],
            "editable_targets": [{"file": "app.py", "symbol": "query"}],
        })
    )

    store = build_artifact_store({"st-1": summary})
    kinds = {artifact["kind"] for artifact in store["artifacts"]}

    assert "database_view" in kinds
    assert "patch_intent" in kinds
