from __future__ import annotations

from pathlib import Path
import pytest

from src.skills.design import DesignSkill
from src.skills.code_search import _is_editable_target


class DummyContext:
    def __init__(self, user_request: str, project_root: Path) -> None:
        self.user_request = user_request
        self.project_root = project_root
        self.context_pack = None
        self.patch_plan = None


def test_is_editable_target_relaxed_symbol_matching(tmp_path: Path) -> None:
    # Create a dummy python file where symbol is defined outside the target range
    display_file = "test_module.py"
    file_path = tmp_path / display_file
    file_path.write_text(
        "build_order_detail_sql = '''\n"
        "    SELECT id, name\n"
        "    FROM orders\n"
        "'''\n",
        encoding="utf-8"
    )

    current_code = "    SELECT id, name\n    FROM orders"
    intended_change = "replace build_order_detail_sql with a view"
    context = DummyContext(user_request=intended_change, project_root=tmp_path)

    # Case 1: Symbol not in snippet, but exists in file, is_sql_change is True
    # If we pass symbol in task_analysis target
    task_analysis = {
        "edit_strategy": "sql_view_rewrite",
        "editable_targets": [
            {"file": "test_module.py", "symbol": "build_order_detail_sql"}
        ]
    }

    # Should match because the symbol is in the file
    res = _is_editable_target(
        current_code,
        display_file,
        intended_change,
        context,  # type: ignore[arg-type]
        edit_strategy="sql_view_rewrite",
        task_analysis=task_analysis,
    )
    assert res is True


@pytest.mark.asyncio
async def test_design_skill_refinement_from_file(tmp_path: Path) -> None:
    display_file = "main.py"
    file_path = tmp_path / display_file
    file_path.write_text(
        "build_order_detail_sql = '''\n"
        "    SELECT id, name\n"
        "    FROM orders\n"
        "'''\n",
        encoding="utf-8"
    )

    context = DummyContext(
        user_request="replace build_order_detail_sql with view_ticket_report_detail",
        project_root=tmp_path
    )

    skill = DesignSkill()
    result = await skill.run(
        context,  # type: ignore[arg-type]
        handoff_contract={
            "must_modify": [{
                "file": "main.py",
                "line": 0,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "",
            }]
        },
        task_analysis={
            "edit_strategy": "sql_view_rewrite",
            "resolved_dependencies": [
                {"kind": "database_view", "name": "view_ticket_report_detail"}
            ]
        }
    )

    assert result.success
    # The output targets should have line_start, line_end, snippet, and decision refined from the file
    import json
    payload = json.loads(result.metadata["patch_intent_json"])
    assert payload["schema"] == "mitkii.handoff.v1"
    assert "must_modify" in payload
    assert payload["must_modify"][0]["file"] == "main.py"
    assert payload["must_modify"][0]["symbol_or_api"] == "build_order_detail_sql"
    assert payload["must_modify"][0]["line_start"] == 1
    assert payload["must_modify"][0]["line_end"] == 4
    assert "SELECT id, name" in payload["must_modify"][0]["current_code"]
    assert "view_ticket_report_detail" in payload["must_modify"][0]["should_change_to"]

    assert "available_views" in payload
    assert payload["available_views"][0]["name"] == "view_ticket_report_detail"

    assert "evidence" in payload
    assert any("main.py:1-4 build_order_detail_sql" in item for item in payload["evidence"])
    
    assert "view_ticket_report_detail" in payload["acceptance_criteria"][0]
    assert "py_compile passes" in payload["acceptance_criteria"]

