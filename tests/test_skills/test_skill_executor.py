from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.agent.types import ToolResult
from src.context.pack import ContextPack, SearchPlan
from src.planner.patch_plan import PatchEdit, PatchPlan
from src.skills import (
    CodeEditSkill,
    CodeSearchSkill,
    SkillContext,
    SkillExecutor,
    SkillResult,
    ValidatorSkill,
    VerifySkill,
)
from src.skills.code_edit import _edit_messages, _replacement_messages
from src.tools.search.context_search import ContextSearchTool
from src.tools.registry import create_default_registry


class FakeSkill:
    name = "validator"

    async def run(self, context: SkillContext, **kwargs: Any) -> SkillResult:
        return SkillResult(
            success=True,
            summary=f"validated: {context.user_request}",
            validation_result="ok",
        )


class FakeToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, name: str, params: dict[str, object]) -> ToolResult:
        self.calls.append((name, params))
        return ToolResult(success=True, output=f"{name}: ok")


class SnippetToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, name: str, params: dict[str, object]) -> ToolResult:
        self.calls.append((name, params))
        if name == "grep_search":
            return ToolResult(success=True, output="main.py:2: def target():")
        return ToolResult(success=True, output="No matches found.")


class RangeToolRegistry:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, name: str, params: dict[str, object]) -> ToolResult:
        self.calls.append((name, params))
        if name == "map_search":
            return ToolResult(
                success=True,
                output="main.py:2-4 function make_boarding_pass_pdf score=0.9",
            )
        return ToolResult(success=True, output="No matches found.")


class AbsoluteRangeToolRegistry:
    def __init__(self, target: Path) -> None:
        self.target = target
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call(self, name: str, params: dict[str, object]) -> ToolResult:
        self.calls.append((name, params))
        if name == "map_search":
            return ToolResult(
                success=True,
                output=f"{self.target}:2-4 function make_boarding_pass_pdf score=0.9",
            )
        return ToolResult(success=True, output="No matches found.")


class PipeHitToolRegistry:
    def __init__(self, target: Path) -> None:
        self.target = target

    async def call(self, name: str, params: dict[str, object]) -> ToolResult:
        if name == "map_search":
            return ToolResult(
                success=True,
                output=(
                    f"  - {self.target}:2 | 目标代码 | "
                    "def make_boarding_pass_pdf(order):"
                ),
            )
        return ToolResult(success=True, output="No matches found.")


async def fake_edit_complete(messages: list[dict[str, object]]) -> str:
    system = str(messages[0].get("content") or "")
    if "ReplacementBuilder" in system:
        return json.dumps({"new_string": "sql = 'SELECT * FROM view_ticket_report_detail'"})
    return json.dumps({
        "edits": [{
            "target_index": 0,
            "operation": "replace_sql_source",
            "source_table": "orders",
            "target_view": "view_ticket_report_detail",
            "change_summary": "switch query source to view",
        }],
        "confidence": 0.9,
        "missing_info": [],
    })


@pytest.mark.asyncio
async def test_skill_executor_runs_registered_skill() -> None:
    executor = SkillExecutor([FakeSkill()])

    result = await executor.run("validator", SkillContext(user_request="fix query"))

    assert result.success
    assert result.validation_result == "ok"
    assert not result.requires_fallback


@pytest.mark.asyncio
async def test_skill_executor_missing_skill_requires_fallback() -> None:
    executor = SkillExecutor()

    result = await executor.run("code_edit", SkillContext(user_request="fix query"))

    assert not result.success
    assert result.requires_fallback
    assert "skill:code_edit" in result.missing_info


@pytest.mark.asyncio
async def test_design_skill_outputs_patch_intent() -> None:
    from src.skills.design import DesignSkill

    executor = SkillExecutor([DesignSkill()])
    result = await executor.run(
        "design",
        SkillContext(user_request="重构订单详情处理，统一脱敏"),
        handoff_contract={
            "must_modify": [{
                "file": "main.py",
                "line": 10,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "apply the requested code change",
            }]
        },
        task_analysis={
            "intent": "function_refactor",
            "edit_strategy": "function_refactor",
            "edit_ready": False,
            "resolved_dependencies": [],
            "acceptance_contract": {"intent": "function_refactor"},
        },
    )
    assert result.success
    assert "PATCH_INTENT_JSON" in result.metadata["final_message"]
    assert "function_refactor" in result.metadata["final_message"]


@pytest.mark.asyncio
async def test_code_edit_skill_applies_exact_patch(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")
    plan = PatchPlan(
        files_to_edit=("main.py",),
        target_symbols=("query_orders",),
        intended_changes=("switch to view",),
        edits=(
            PatchEdit(
                path="main.py",
                old_string="SELECT * FROM orders",
                new_string="SELECT * FROM view_ticket_report_detail",
            ),
        ),
        confidence=0.9,
    )
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path)])

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query", patch_plan=plan),
    )

    assert result.success
    assert result.changed_files == ("main.py",)
    assert "view_ticket_report_detail" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_code_edit_skill_accepts_absolute_path_under_project(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")
    plan = PatchPlan(
        files_to_edit=(str(target),),
        intended_changes=("switch to view",),
        edits=(
            PatchEdit(
                path=str(target),
                old_string="SELECT * FROM orders",
                new_string="SELECT * FROM view_ticket_report_detail",
            ),
        ),
        confidence=0.9,
    )
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path)])

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query", patch_plan=plan),
    )

    assert result.success
    assert result.changed_files == ("main.py",)
    assert "view_ticket_report_detail" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_code_edit_skill_runs_focused_edit_without_patch_plan(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")
    executor = SkillExecutor([
        CodeEditSkill(project_root=tmp_path, llm_complete=fake_edit_complete)
    ])

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        instruction="把查询改成视图",
        search_output=(
            "main.py:1: sql = 'SELECT * FROM orders'\n\n"
            "EDIT_CONTEXT_JSON\n"
                + json.dumps({
                    "schema": "mitkii.edit_context.v1",
                    "code_edit_ready": True,
                    "edit_strategy": "sql_view_rewrite",
                    "target_view": "view_ticket_report_detail",
                "available_views": ["view_ticket_report_detail"],
                "editable_targets": [{
                    "file": "main.py",
                    "start_line": 1,
                    "end_line": 1,
                    "current_code": "sql = 'SELECT * FROM orders'",
                }],
                "intended_change": "把查询改成视图",
                "acceptance_criteria": "query uses view",
                "tool_policy": {
                    "allowed_tools": ["edit_file"],
                    "scope": ["main.py"],
                },
            })
        ),
    )

    assert result.success
    assert result.changed_files == ("main.py",)
    assert "view_ticket_report_detail" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_code_edit_skill_requires_edit_context_before_llm(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")
    calls = 0

    async def complete(_messages: list[dict[str, object]]) -> str:
        nonlocal calls
        calls += 1
        return '{"edits":[],"confidence":0.0,"missing_info":[]}'

    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        instruction="把查询改成视图",
        search_output="main.py:1: sql = 'SELECT * FROM orders'",
    )

    assert not result.success
    assert calls == 0
    assert "missing EDIT_CONTEXT_JSON" in result.summary


@pytest.mark.asyncio
async def test_code_edit_skill_rejects_absolute_file_outside_project_root(tmp_path) -> None:
    project_root = tmp_path / "harness"
    project_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    target = target_root / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")

    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        if "ReplacementBuilder" in system:
            return json.dumps({"new_string": "sql = 'SELECT * FROM view_ticket_report_detail'"})
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_sql_source",
                "source_table": "orders",
                "target_view": "view_ticket_report_detail",
                "change_summary": "switch query source to view",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })

    executor = SkillExecutor([
        CodeEditSkill(project_root=project_root, llm_complete=complete)
    ])
    edit_context = {
        "schema": "mitkii.edit_context.v1",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "target_view": "view_ticket_report_detail",
        "available_views": ["view_ticket_report_detail"],
        "editable_targets": [{
            "file": str(target),
            "start_line": 1,
            "end_line": 1,
            "current_code": "sql = 'SELECT * FROM orders'",
        }],
        "intended_change": "use view",
        "acceptance_criteria": "query uses view",
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": [str(target)],
        },
    }

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )

    assert not result.success
    assert "patch_validator cannot locate" in result.summary
    assert "view_ticket_report_detail" not in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_code_edit_skill_accepts_snippets_as_editable_targets(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")

    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        if "ReplacementBuilder" in system:
            return json.dumps({"new_string": "sql = 'SELECT * FROM view_ticket_report_detail'"})
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_sql_source",
                "source_table": "orders",
                "target_view": "view_ticket_report_detail",
                "change_summary": "switch query source to view",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })

    executor = SkillExecutor([
        CodeEditSkill(project_root=tmp_path, llm_complete=complete)
    ])
    edit_context = {
        "schema": "mitkii.edit_context.v1",
        "builder": "EditPlanBuilder",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "target_view": "view_ticket_report_detail",
        "available_views": ["view_ticket_report_detail"],
        "snippets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 1,
            "current_code": "sql = 'SELECT * FROM orders'",
            "intended_change": "use view",
            "acceptance_criteria": ["query uses view"],
        }],
        "intended_change": "use view",
        "acceptance_criteria": ["query uses view"],
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": ["main.py"],
        },
    }

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )

    assert result.success
    assert result.changed_files == ("main.py",)


@pytest.mark.asyncio
async def test_code_edit_skill_accepts_target_index_without_old_string(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text(
        "def get_boarding_pass():\n"
        "    sql = 'SELECT * FROM boarding_pass'\n"
        "    return sql\n",
        encoding="utf-8",
    )

    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        if "ReplacementBuilder" in system:
            return json.dumps({
                "new_string": (
                    "def get_boarding_pass():\n"
                    "    sql = 'SELECT * FROM view_ticket_report_detail'\n"
                    "    return sql"
                ),
            })
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_sql_source",
                "source_table": "boarding_pass",
                "target_view": "view_ticket_report_detail",
                "change_summary": "switch query source to view",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })

    executor = SkillExecutor([
        CodeEditSkill(project_root=tmp_path, llm_complete=complete)
    ])
    current_code = target.read_text(encoding="utf-8").rstrip()
    edit_context = {
        "schema": "mitkii.edit_context.v1",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "target_view": "view_ticket_report_detail",
        "available_views": ["view_ticket_report_detail"],
        "editable_targets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 3,
            "current_code": current_code,
        }],
        "intended_change": "use view",
        "acceptance_criteria": "query uses view",
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": ["main.py"],
        },
    }

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )

    assert result.success
    assert result.changed_files == ("main.py",)
    assert "view_ticket_report_detail" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_code_edit_skill_falls_back_from_non_json_to_sql_view_edit(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text(
        "def get_boarding_pass():\n"
        "    sql = 'SELECT * FROM boarding_pass WHERE id = :id'\n"
        "    return sql\n",
        encoding="utf-8",
    )

    async def complete(_messages: list[dict[str, object]]) -> str:
        return "我会把查询改成视图，但这里不是 JSON"

    executor = SkillExecutor([
        CodeEditSkill(project_root=tmp_path, llm_complete=complete)
    ])
    current_code = target.read_text(encoding="utf-8").rstrip()
    edit_context = {
        "schema": "mitkii.edit_context.v1",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "target_view": "view_ticket_report_detail",
        "available_views": ["view_ticket_report_detail"],
        "editable_targets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 3,
            "current_code": current_code,
        }],
        "intended_change": "把登机牌查询接口改成用视图查询",
        "acceptance_criteria": "query uses view",
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": ["main.py"],
        },
    }

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output=(
            "db/init/init.sql:394: CREATE VIEW view_ticket_report_detail AS\n"
            "EDIT_CONTEXT_JSON\n"
            + json.dumps(edit_context)
        ),
    )

    assert result.success
    assert result.changed_files == ("main.py",)
    assert "FROM view_ticket_report_detail" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_code_edit_skill_non_json_without_sql_fallback_reports_preview(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text(
        "def make_boarding_pass_pdf(order):\n"
        "    return order\n",
        encoding="utf-8",
    )

    calls = 0

    async def complete(_messages: list[dict[str, object]]) -> str:
        nonlocal calls
        calls += 1
        return "不是 JSON"

    executor = SkillExecutor([
        CodeEditSkill(project_root=tmp_path, llm_complete=complete)
    ])
    current_code = target.read_text(encoding="utf-8").rstrip()
    edit_context = {
        "schema": "mitkii.edit_context.v1",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "target_view": "view_ticket_report_detail",
        "available_views": ["view_ticket_report_detail"],
        "editable_targets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 2,
            "current_code": current_code,
        }],
        "intended_change": "把登机牌查询接口改成用视图查询",
        "acceptance_criteria": "query uses view",
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": ["main.py"],
        },
    }

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output=(
            "db/init/init.sql:394: CREATE VIEW view_ticket_report_detail AS\n"
            "EDIT_CONTEXT_JSON\n"
            + json.dumps(edit_context)
        ),
    )

    assert not result.success
    assert calls == 0
    assert "not a SQL/query target" in result.summary


@pytest.mark.asyncio
async def test_code_edit_skill_falls_back_when_old_string_not_unique(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\nx = 1\n", encoding="utf-8")
    plan = PatchPlan(
        files_to_edit=("main.py",),
        intended_changes=("change x",),
        edits=(PatchEdit(path="main.py", old_string="x = 1", new_string="x = 2"),),
        confidence=0.9,
    )
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path)])

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change x", patch_plan=plan),
    )

    assert not result.success
    assert result.requires_fallback


@pytest.mark.asyncio
async def test_validator_skill_compiles_changed_python_file(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("x = 1\n", encoding="utf-8")
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])

    result = await executor.run(
        "validator",
        SkillContext(user_request="validate"),
        changed_files=("main.py",),
    )

    assert result.success
    assert result.validation_result == "passed"


@pytest.mark.asyncio
async def test_validator_skill_reports_compile_error(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("def broken(:\n", encoding="utf-8")
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])

    result = await executor.run(
        "validator",
        SkillContext(user_request="validate"),
        changed_files=("main.py",),
    )

    assert not result.success
    assert result.validation_result == "failed"


@pytest.mark.asyncio
async def test_validator_skill_reports_undefined_name_via_ruff(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text("def test_func():\n    return undefined_variable_name\n", encoding="utf-8")
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    result = await executor.run(
        "validator",
        SkillContext(user_request="validate"),
        changed_files=("main.py",),
    )
    
    import shutil
    if shutil.which("ruff") is not None:
        assert not result.success
        assert any("undefined_variable_name" in err or "F821" in err for err in result.missing_info)


@pytest.mark.asyncio
async def test_code_search_skill_batches_map_and_grep(tmp_path) -> None:
    registry = FakeToolRegistry()
    executor = SkillExecutor([
        CodeSearchSkill(project_root=tmp_path, tools=registry),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
        task_analysis={"edit_strategy": "sql_view_rewrite", "intent": "sql_view_rewrite"},
    )

    assert result.success
    assert [name for name, _args in registry.calls[:3]] == [
        "map_search",
        "map_search",
        "map_search",
    ]
    grep_calls = [args for name, args in registry.calls if name == "grep_search"]
    assert [args["include"] for args in grep_calls] == ["*.py", "*.sql"]
    py_pattern = str(grep_calls[0]["pattern"])
    sql_pattern = str(grep_calls[1]["pattern"])
    assert "boarding_pass" in py_pattern
    assert "CREATE" in sql_pattern
    assert "VIEW" in sql_pattern
    assert "api" not in py_pattern.lower()
    assert "api" not in sql_pattern.lower()
    assert result.metadata["search_output"]


def test_code_edit_messages_include_hard_handoff_contract() -> None:
    messages = _edit_messages(
        instruction="修改登机牌查询接口",
        evidence="main.py:10: SELECT * FROM boarding_pass",
        handoff_contract={
            "must_modify": [{"file": "main.py", "line": 10}],
            "available_views": [{"name": "v_boarding_pass"}],
            "tool_policy": {"allowed_tools": ["edit_file"]},
        },
    )

    user_content = str(messages[1]["content"])
    assert "HARD_HANDOFF_CONTRACT_JSON" in user_content
    assert "v_boarding_pass" in user_content
    assert "main.py" in user_content
    assert "target_index" in str(messages[0]["content"])
    assert "must not generate replacement code" in str(messages[0]["content"])


def test_code_edit_messages_preserve_edit_context_after_large_search_log() -> None:
    messages = _edit_messages(
        instruction="修改登机牌查询接口",
        evidence=(
            "x" * 20_000
            + "\nEDIT_CONTEXT_JSON\n"
            + json.dumps({
                "code_edit_ready": True,
                "editable_targets": [{
                    "file": "app.py",
                    "current_code": "def make_boarding_pass_pdf(): pass",
                }],
            })
        ),
    )

    user_content = str(messages[1]["content"])
    assert "EDIT_CONTEXT_JSON" in user_content
    assert "def make_boarding_pass_pdf(): pass" in user_content


def test_code_edit_plan_messages_compact_large_current_code() -> None:
    large_code = "def target():\n" + "\n".join(f"    x_{idx} = {idx}" for idx in range(2000))
    messages = _edit_messages(
        instruction="把查询改成视图",
        evidence=(
            "EDIT_CONTEXT_JSON\n"
            + json.dumps({
                "code_edit_ready": True,
                "editable_targets": [{
                    "file": "app.py",
                    "start_line": 1,
                    "end_line": 2001,
                    "current_code": large_code,
                }],
            })
        ),
    )

    user_content = str(messages[1]["content"])
    assert "EDIT_CONTEXT_JSON_COMPACT" in user_content
    assert len(user_content) < 5_000
    assert "x_1999" not in user_content


def test_replacement_messages_focus_single_current_code() -> None:
    messages = _replacement_messages(
        instruction="把查询改成视图",
        file="app.py",
        target_index=0,
        current_code="def target():\n    return 'SELECT * FROM boarding_pass'",
        edit_plan={"target_index": 0, "target_view": "view_ticket_report_detail"},
        evidence="other.py:1: unrelated\nEDIT_CONTEXT_JSON\n{}",
    )

    user_content = str(messages[1]["content"])
    assert "CURRENT_CODE" in user_content
    assert "SELECT * FROM boarding_pass" in user_content
    assert "EDIT_CONTEXT_JSON" not in user_content
    assert "view_ticket_report_detail" in str(messages[0]["content"])


def test_replacement_messages_without_target_view_do_not_force_view() -> None:
    messages = _replacement_messages(
        instruction="重构订单详情处理并统一脱敏",
        file="app.py",
        target_index=0,
        current_code="def target():\n    return normalize_order_record(row)",
        edit_plan={"target_index": 0, "operation": "general_edit"},
        evidence="",
        target_view="",
    )

    system_content = str(messages[0]["content"])
    assert "MUST only use the replacement source" not in system_content
    assert "Do not invent replacement dependencies" in system_content


@pytest.mark.asyncio
async def test_code_search_skill_does_not_grep_need_words(tmp_path) -> None:
    registry = FakeToolRegistry()
    skill = CodeSearchSkill(project_root=tmp_path, tools=registry)  # type: ignore[arg-type]

    result = await skill.run(
        SkillContext(user_request="视图 view\nNeed: file:line symbol snippet"),
        extra_query="视图 view\nNeed: file:line symbol snippet",
        search_query="视图 view",
    )

    assert result.success
    grep_patterns = [
        str(args["pattern"])
        for name, args in registry.calls
        if name == "grep_search"
    ]
    assert grep_patterns
    assert all("snippet" not in pattern for pattern in grep_patterns)
    assert all("file" not in pattern for pattern in grep_patterns)
    assert any("CREATE" in pattern and "VIEW" in pattern for pattern in grep_patterns)


@pytest.mark.asyncio
async def test_code_search_skill_view_definition_uses_definition_pattern(tmp_path) -> None:
    registry = FakeToolRegistry()
    skill = CodeSearchSkill(project_root=tmp_path, tools=registry)  # type: ignore[arg-type]

    result = await skill.run(
        SkillContext(user_request="搜索项目中的视图定义"),
        extra_query="搜索项目中的视图定义",
        search_query="搜索项目中的视图定义",
    )

    assert result.success
    grep_patterns = [
        str(args["pattern"])
        for name, args in registry.calls
        if name == "grep_search"
    ]
    assert grep_patterns
    assert all(r"\bview\b" not in pattern for pattern in grep_patterns)
    assert any("CREATE" in pattern and "VIEW" in pattern for pattern in grep_patterns)


@pytest.mark.asyncio
async def test_code_search_skill_context_pack_search_plan_keeps_domain_pattern(tmp_path) -> None:
    registry = FakeToolRegistry()
    skill = CodeSearchSkill(project_root=tmp_path, tools=registry)  # type: ignore[arg-type]
    pack = ContextPack(
        user_request="把登机牌查询接口改成用视图查询",
        search_plan=(
            SearchPlan(
                module=".",
                files=("main.py",),
                patterns=("查询", "query", "接口", "api", "boarding_pass", "view"),
                globs=("*.py", "*.sql"),
            ),
        ),
    )

    result = await skill.run(
        SkillContext(
            user_request="把登机牌查询接口改成用视图查询",
            context_pack=pack,
        ),
    )

    assert result.success
    grep_patterns = [
        str(args["pattern"])
        for name, args in registry.calls
        if name == "grep_search"
    ]
    assert grep_patterns
    assert any("boarding_pass" in pattern for pattern in grep_patterns)
    assert any("CREATE" in pattern and "VIEW" in pattern for pattern in grep_patterns)
    assert all("api" not in pattern.lower() for pattern in grep_patterns)
    assert all("query" not in pattern.lower() for pattern in grep_patterns)
    assert any(name == "map_search" for name, _args in registry.calls)
    assert any(
        name == "grep_search" and args.get("path") == str(tmp_path.resolve())
        for name, args in registry.calls
    )


@pytest.mark.asyncio
async def test_code_search_skill_hydrates_bounded_snippets(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text(
        "x = 1\n"
        "def target():\n"
        "    return x\n",
        encoding="utf-8",
    )
    executor = SkillExecutor([
        CodeSearchSkill(project_root=tmp_path, tools=SnippetToolRegistry()),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="find target"),
    )

    assert result.success
    output = result.metadata["search_output"]
    assert '<snippet path="main.py"' in output
    assert "2: def target():" in output
    assert "EDIT_CONTEXT_JSON" in output
    assert '"editable_targets"' in output
    assert '"current_code"' in output


@pytest.mark.asyncio
async def test_code_search_skill_hydrates_map_range_as_edit_context(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text(
        "x = 1\n"
        "def make_boarding_pass_pdf(order):\n"
        "    sql = 'SELECT * FROM boarding_pass'\n"
        "    return sql\n"
        "y = 2\n",
        encoding="utf-8",
    )
    executor = SkillExecutor([
        CodeSearchSkill(project_root=tmp_path, tools=RangeToolRegistry()),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
        task_analysis={"edit_strategy": "sql_view_rewrite", "intent": "sql_view_rewrite"},
    )

    assert result.success
    output = result.metadata["search_output"]
    assert "EDIT_CONTEXT_JSON" in output
    assert '"file": "main.py"' in output
    assert '"builder": "EditPlanBuilder"' in output
    assert '"snippets"' in output
    assert '"editable_targets"' in output
    assert '"scope"' in output
    assert "SELECT * FROM boarding_pass" in output
    assert '"intended_change"' in output
    assert '"acceptance_criteria"' in output


@pytest.mark.asyncio
async def test_code_search_skill_keeps_edit_context_when_snippet_exceeds_budget(tmp_path) -> None:
    lines = ["x = 1", "def make_boarding_pass_pdf(order):"]
    lines.extend(
        f"    value_{idx} = 'SELECT * FROM boarding_pass " + ("x" * 120) + "'"
        for idx in range(1, 260)
    )
    lines.append("    return order")
    target = tmp_path / "app.py"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    class LargeRangeToolRegistry:
        async def call(self, name: str, params: dict[str, object]) -> ToolResult:
            if name == "map_search":
                return ToolResult(
                    success=True,
                    output="app.py:2-261 function make_boarding_pass_pdf score=0.9",
                )
            return ToolResult(success=True, output="No matches found.")

    executor = SkillExecutor([
        CodeSearchSkill(project_root=tmp_path, tools=LargeRangeToolRegistry()),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
    )

    output = result.metadata["search_output"]
    assert result.success
    assert result.metadata["edit_context_targets"] == "1"
    assert "EDIT_CONTEXT_JSON" in output
    assert '"current_code"' in output
    assert "snippet omitted due to budget" in result.metadata["hydration_failures"]


@pytest.mark.asyncio
async def test_code_search_skill_does_not_make_pdf_renderer_editable_for_query_view_task(tmp_path) -> None:
    target = tmp_path / "app.py"
    target.write_text(
        "def make_boarding_pass_pdf(order):\n"
        "    title = '电子登机牌'\n"
        "    return title\n",
        encoding="utf-8",
    )

    class PdfRangeToolRegistry:
        async def call(self, name: str, params: dict[str, object]) -> ToolResult:
            if name == "map_search":
                return ToolResult(
                    success=True,
                    output="app.py:1-3 function make_boarding_pass_pdf score=0.9",
                )
            return ToolResult(success=True, output="No matches found.")

    executor = SkillExecutor([
        CodeSearchSkill(project_root=tmp_path, tools=PdfRangeToolRegistry()),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
        task_analysis={"edit_strategy": "sql_view_rewrite", "intent": "sql_view_rewrite"},
    )

    assert result.success
    assert result.metadata["edit_context_targets"] == "0"
    assert "EDIT_CONTEXT_JSON" not in result.metadata["search_output"]
    assert "not editable for SQL/view query change" in result.metadata["hydration_failures"]


@pytest.mark.asyncio
async def test_code_search_skill_rejects_absolute_range_outside_project_root(tmp_path) -> None:
    harness_root = tmp_path / "harness"
    harness_root.mkdir()
    target_root = tmp_path / "target"
    target_root.mkdir()
    target = target_root / "main.py"
    target.write_text(
        "x = 1\n"
        "def make_boarding_pass_pdf(order):\n"
        "    sql = 'SELECT * FROM boarding_pass'\n"
        "    return sql\n",
        encoding="utf-8",
    )
    executor = SkillExecutor([
        CodeSearchSkill(
            project_root=harness_root,
            tools=AbsoluteRangeToolRegistry(target),  # type: ignore[arg-type]
        ),
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
    )

    assert result.success
    assert result.metadata["edit_context_targets"] == "0"
    output = result.metadata["search_output"]
    assert "EDIT_CONTEXT_JSON" not in output
    assert str(target) not in result.metadata.get("hydration_hit_paths", "")


@pytest.mark.asyncio
async def test_code_search_skill_hydrates_pipe_style_line_hits(tmp_path) -> None:
    harness_root = tmp_path / "harness"
    harness_root.mkdir()
    target = harness_root / "app.py"
    target.write_text(
        "x = 1\n"
        "def make_boarding_pass_pdf(order):\n"
        "    sql = 'SELECT * FROM boarding_pass'\n"
        "    return sql\n",
        encoding="utf-8",
    )
    executor = SkillExecutor([
        CodeSearchSkill(
            project_root=harness_root,
            tools=PipeHitToolRegistry(target),  # type: ignore[arg-type]
        ),
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
    )

    assert result.success
    assert result.metadata["edit_context_targets"] == "1"
    assert "EDIT_CONTEXT_JSON" in result.metadata["search_output"]


@pytest.mark.asyncio
async def test_code_search_skill_rejects_relative_hit_from_sibling_target_root(tmp_path) -> None:
    harness_root = tmp_path / "harness"
    harness_root.mkdir()
    target_root = tmp_path / "database-course-design"
    target_root.mkdir()
    target = target_root / "main.py"
    target.write_text(
        "x = 1\n"
        "def make_boarding_pass_pdf(order):\n"
        "    sql = 'SELECT * FROM boarding_pass'\n"
        "    return sql\n",
        encoding="utf-8",
    )
    executor = SkillExecutor([
        CodeSearchSkill(project_root=harness_root, tools=RangeToolRegistry()),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
    )

    assert result.success
    assert result.success
    assert result.metadata["edit_context_targets"] == "0"
    assert result.metadata["hydration_hits"] == "0"
    assert "not found under hydration root" in result.metadata["hydration_failures"]


@pytest.mark.asyncio
async def test_code_search_skill_does_not_guess_sibling_when_same_name_file_too_short(tmp_path) -> None:
    harness_root = tmp_path / "harness"
    harness_root.mkdir()
    (harness_root / "app.py").write_text("short\n", encoding="utf-8")
    target_root = tmp_path / "database-course-design"
    target_root.mkdir()
    target = target_root / "app.py"
    target.write_text(
        "\n".join(
            ["x = 1"] * 1583
            + [
                "def make_boarding_pass_pdf(order):",
                "    sql = 'SELECT * FROM boarding_pass'",
                "    return sql",
            ]
            + ["z = 3"] * 240
        ),
        encoding="utf-8",
    )

    class AppRangeToolRegistry:
        async def call(self, name: str, params: dict[str, object]) -> ToolResult:
            if name == "map_search":
                return ToolResult(
                    success=True,
                    output="app.py:1584-1817 function make_boarding_pass_pdf score=0.9",
                )
            return ToolResult(success=True, output="No matches found.")

    executor = SkillExecutor([
        CodeSearchSkill(project_root=harness_root, tools=AppRangeToolRegistry()),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
    )

    assert result.success
    assert result.metadata["edit_context_targets"] == "0"
    assert str(target) not in result.metadata["search_output"]
    assert "not found under hydration root" in result.metadata.get(
        "hydration_failures", ""
    )


@pytest.mark.asyncio
async def test_context_search_tool_uses_skill_snippets(tmp_path) -> None:
    target = tmp_path / "main.py"
    target.write_text(
        "x = 1\n"
        "def target():\n"
        "    return x\n",
        encoding="utf-8",
    )
    tool = ContextSearchTool(
        project_root=tmp_path,
        tools=SnippetToolRegistry(),  # type: ignore[arg-type]
    )

    result = await tool.execute(query="target", need="file:line and snippet", paths=["."])

    assert result.success
    assert '<snippet path="main.py"' in result.output


def test_default_registry_registers_context_search(tmp_path) -> None:
    registry = create_default_registry(project_root=tmp_path)

    assert registry.get("context_search") is not None


@pytest.mark.asyncio
async def test_verify_skill_no_framework_detected(tmp_path) -> None:
    executor = SkillExecutor([VerifySkill(project_root=tmp_path)])
    result = await executor.run(
        "verify",
        SkillContext(user_request="verify"),
        changed_files=("main.py",),
    )
    assert not result.success
    assert result.validation_result == "failed"
    assert (
        "No test framework detected" in result.summary
        or "pytest" in result.summary
    )


@pytest.mark.asyncio
async def test_verify_skill_fails_when_test_fails(tmp_path) -> None:
    conftest = tmp_path / "conftest.py"
    conftest.write_text("", encoding="utf-8")

    test_file = tmp_path / "test_fail.py"
    test_file.write_text("def test_one(): assert False\n", encoding="utf-8")

    executor = SkillExecutor([VerifySkill(project_root=tmp_path)])
    result = await executor.run(
        "verify",
        SkillContext(user_request="verify"),
        changed_files=("test_fail.py",),
    )
    assert not result.success
    assert result.validation_result == "failed"
    assert "pytest" in result.summary
    assert "FAILED" in result.summary


def test_infer_target_view_token_matching() -> None:
    from src.skills.code_search import _infer_target_view
    views = ["flight_load", "v_boarding_pass", "v_order_detail"]
    
    # Chinese and english matching for boarding pass
    v1 = _infer_target_view("使用登机牌视图", views, [])
    assert v1 == "v_boarding_pass"
    
    v2 = _infer_target_view("use boarding_pass view to replace table", views, [])
    assert v2 == "v_boarding_pass"
    
    # Token matching on target methods/code
    v3 = _infer_target_view(
        "use view to replace query method",
        views,
        [{"file": "app.py", "current_code": "def build_order_detail_sql():\n    pass"}]
    )
    assert v3 == "v_order_detail"


def test_validate_edit_context_ready_with_target_view(tmp_path) -> None:
    from src.skills.code_edit import _validate_edit_context_ready
    
    # 1. Successful validation when target_view matches available_views
    ctx_ok = {
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "intended_change": "use view to replace order query",
        "available_views": ["v_order_detail"],
        "target_view": "v_order_detail",
        "acceptance_criteria": "success",
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["app.py"]},
        "editable_targets": [{
            "file": "app.py",
            "start_line": 1,
            "end_line": 2,
            "current_code": "SELECT * FROM orders",
            "intended_change": "use view to replace order query",
            "acceptance_criteria": "success"
        }]
    }
    
    app_file = tmp_path / "app.py"
    app_file.write_text("SELECT * FROM orders\n", encoding="utf-8")
    
    err = _validate_edit_context_ready(tmp_path, ctx_ok)
    assert err == ""
    
    # 2. Stale/incomplete available_views should not block the editor; final
    # validator owns target_view existence checks.
    ctx_bad = dict(ctx_ok)
    ctx_bad["target_view"] = "hallucinated_view"
    err = _validate_edit_context_ready(tmp_path, ctx_bad)
    assert err == ""


@pytest.mark.asyncio
async def test_code_edit_allows_unresolved_dependencies_and_uses_bounded_sql_fallback(tmp_path) -> None:
    from src.skills.code_edit import CodeEditSkill

    target = tmp_path / "main.py"
    current_code = "def query():\n    return \"SELECT o.name AS order_name FROM orders o\"\n"
    fallback_code = "def query():\n    return \"SELECT v.id AS order_name FROM view_ticket_report_detail v\"\n"
    target.write_text(current_code, encoding="utf-8")

    calls: list[str] = []

    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        calls.append(system)
        if "SQL ReplacementBuilder" in system:
            user = str(messages[1].get("content") or "")
            assert "EDIT_CONTEXT_JSON" not in user
            assert current_code in user
            return json.dumps({"new_string": fallback_code})
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_sql_source",
                "target_view": "view_ticket_report_detail",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })

    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "dependencies_resolved": False,
        "target_view": "view_ticket_report_detail",
        "task_intent": {
            "operation": "replace_dependency",
            "target_symbol": "query",
            "goal": "use target view",
        },
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "view_ticket_report_detail",
            "columns": ["id"],
            "replaces_objects": ["orders"],
        }],
        "editable_targets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 2,
            "current_code": current_code,
            "intended_change": "use view_ticket_report_detail",
            "acceptance_criteria": ["query uses view"],
        }],
        "intended_change": "use view_ticket_report_detail",
        "acceptance_criteria": ["query uses view"],
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": ["main.py"],
        },
    }

    result = await executor.run(
        "code_edit",
        SkillContext(user_request="replace query with view"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )

    assert result.success
    assert any("SQL ReplacementBuilder" in call for call in calls)
    written = target.read_text(encoding="utf-8")
    assert "view_ticket_report_detail" in written
    assert "orders" not in written


def test_validate_sql_references_check(tmp_path) -> None:
    from src.skills.validator import validate_sql_references
    
    # Create a SQL file defining a view and a table, and a PY file defining another view
    sql_file = tmp_path / "schema.sql"
    sql_file.write_text(
        "CREATE VIEW v_boarding_pass AS SELECT * FROM boarding_pass;\n"
        "CREATE TABLE orders (id INT);\n"
        "CREATE TEMPORARY TABLE IF NOT EXISTS [dbo].[ticket_order] (id INT);\n"
        'CREATE TABLE "public"."v_order_summary" (id INT);\n',
        encoding="utf-8"
    )
    py_def = tmp_path / "db_setup.py"
    py_def.write_text(
        'cursor.execute("CREATE VIEW IF NOT EXISTS v_order_detail AS SELECT 1")\n',
        encoding="utf-8"
    )
    
    # Scenario A: Changed file references a defined table, SQL view, and PY view
    changed_file = tmp_path / "app.py"
    changed_file.write_text(
        "import datetime\n"
        "from fastapi import FastAPI\n"
        "import sqlalchemy\n"
        "def query(some_table_var):\n"
        "    \"\"\"\n"
        "    SELECT * FROM orders\n"
        "    from sqlalchemy.exc import SQLAlchemyError\n"
        "    \"\"\"\n"
        "    msg = \"Failed to load data from real_name\"\n"
        "    sql1 = \"SELECT * FROM orders JOIN v_boarding_pass JOIN v_order_detail JOIN ticket_order JOIN v_order_summary;\"\n"
        "    sql2 = f\"SELECT * FROM {some_table_var} JOIN {get_table_fn()};\"\n"
        "    return sql1 + sql2 + msg\n",
        encoding="utf-8"
    )
    
    errors = validate_sql_references(tmp_path, ["app.py"])
    assert not errors




    # Scenario B: References a non-existent database view/table in a string literal
    changed_file.write_text(
        "def query():\n"
        "    return 'SELECT * FROM nonexistent_table;'\n",
        encoding="utf-8"
    )
    errors = validate_sql_references(tmp_path, ["app.py"])
    assert len(errors) == 1
    assert "nonexistent_table" in errors[0]

    # Scenario C: ON DUPLICATE KEY UPDATE with column names should not be flagged as table references
    changed_file.write_text(
        "def query():\n"
        "    return 'INSERT INTO orders (id) VALUES (1) ON DUPLICATE KEY UPDATE real_name = VALUES(real_name);'\n",
        encoding="utf-8"
    )
    errors = validate_sql_references(tmp_path, ["app.py"])
    assert not errors



def test_find_all_views_robustness(tmp_path) -> None:
    from src.skills.code_search import _find_all_views
    
    sql_file = tmp_path / "test.sql"
    sql_file.write_text(
        "CREATE VIEW v_boarding_pass AS SELECT * FROM boarding_pass;\n"
        "CREATE TABLE IF NOT EXISTS orders (id INT);\n"
        "CREATE OR REPLACE VIEW [dbo].[v_order_detail] AS SELECT 1;\n"
        'CREATE ALGORITHM=UNDEFINED DEFINER=`root`@`localhost` SQL SECURITY DEFINER VIEW "public"."v_order_summary" AS SELECT 2;\n'
        "create  or  replace  view  `some_other_view`  as select 3;\n"
        "DROP VIEW IF EXISTS some_other_view;\n",
        encoding="utf-8"
    )
    
    views = _find_all_views(tmp_path)
    expected = {
        "v_boarding_pass",
        "dbo.v_order_detail",
        "v_order_detail",
        "public.v_order_summary",
        "v_order_summary",
        "some_other_view"
    }
    for item in expected:
        assert item in views
    assert "orders" not in views
    assert "if" not in views


def test_extract_json_payload_corrupted_robustness() -> None:
    from src.skills.code_edit import _extract_json_payload
    
    # Scenario 1: JSON corrupted by unescaped newlines in values (common LLM mistake)
    raw_json_1 = '{\n"new_string": "def hello():\\n    print(\'hi\')\\n    return 1"\n}'
    payload_1 = _extract_json_payload(raw_json_1)
    assert payload_1 is not None
    assert payload_1.get("new_string") == "def hello():\n    print('hi')\n    return 1"

    # Scenario 2: EditPlan JSON severely corrupted by unescaped quotes and code blocks inside source_table
    raw_json_2 = (
        '{\n'
        '  "edits": [\n'
        '    {\n'
        '      "target_index": 0,\n'
        '      "operation": "replace_sql_source",\n'
        '      "source_table": "def query(): return text(\\" SELECT FROM orders \\")",\n'
        '      "target_view": "v_order_detail"\n'
        '    }\n'
        '  ]\n'
        '}'
    )
    payload_2 = _extract_json_payload(raw_json_2)
    assert payload_2 is not None
    edits = payload_2.get("edits")
    assert isinstance(edits, list) and len(edits) == 1
    assert edits[0].get("target_index") == 0
    assert edits[0].get("target_view") == "v_order_detail"

    # Scenario 3: ReplacementBuilder JSON corrupted by unescaped quotes inside new_string
    raw_json_3 = (
        '{\n'
        '  "new_string": "def query():\\n    return text(\\"SELECT * FROM view_ticket_report_detail\\")"\n'
        '}'
    )
    payload_3 = _extract_json_payload(raw_json_3)
    assert payload_3 is not None
    assert "view_ticket_report_detail" in str(payload_3.get("new_string"))


def test_clean_llm_code() -> None:
    from src.skills.code_edit import _clean_llm_code
    
    # Test over-escaped triple double quotes
    assert _clean_llm_code('   \\"""管理员修改订单\\"""') == '   """管理员修改订单"""'
    assert _clean_llm_code('   \"\"\"管理员修改订单\"\"\"') == '   """管理员修改订单"""'
    
    # Test over-escaped triple single quotes
    assert _clean_llm_code("   \\'''管理员修改订单\\'''") == "   '''管理员修改订单'''"
    assert _clean_llm_code("   \'\'\'管理员修改订单\'\'\'") == "   '''管理员修改订单'''"


def test_deterministic_replace_sql_with_view() -> None:
    from src.skills.code_edit import deterministic_replace_sql_with_view
    
    # 1. SQL with WHERE clause
    sql_with_where = (
        "def query():\n"
        "    return text(\"\"\"\n"
        "        SELECT a, b\n"
        "        FROM ticket_order\n"
        "        WHERE a = 1\n"
        "    \"\"\")"
    )
    res_where = deterministic_replace_sql_with_view(sql_with_where, "v_my_view")
    assert res_where is not None
    assert "SELECT a, b" in res_where
    assert "FROM v_my_view" in res_where
    assert "WHERE a = 1" in res_where
    
    # 2. SQL without WHERE clause
    sql_no_where = (
        "def query():\n"
        "    return text(\"\"\"\n"
        "        SELECT a, b\n"
        "        FROM ticket_order\n"
        "    \"\"\")"
    )
    res_no_where = deterministic_replace_sql_with_view(sql_no_where, "v_my_view")
    assert res_no_where is not None
    assert "SELECT a, b" in res_no_where
    assert "FROM v_my_view" in res_no_where


def test_merge_ranges_hard_cap() -> None:
    from src.skills.code_search import _merge_ranges
    # Capped at max_lines = 100
    ranges = [(10, 159)]
    merged = _merge_ranges(ranges, gap=20, max_lines=100)
    assert len(merged) == 2
    assert merged[0] == (10, 109)
    assert merged[1] == (110, 159)

    # Capped at max_lines = 140
    ranges = [(10, 80), (90, 160)]
    merged = _merge_ranges(ranges, gap=20, max_lines=140)
    assert len(merged) == 2
    assert merged[0] == (10, 80)
    assert merged[1] == (90, 160)


def test_find_enclosing_sql_range() -> None:
    from src.skills.code_search import _find_enclosing_sql_range
    sql_lines = [
        "SELECT a, b",      # 1
        "FROM orders",       # 2
        "WHERE id = 1;",     # 3
        "",                  # 4
        "CREATE VIEW v_passengers AS", # 5
        "SELECT *",          # 6
        "FROM passengers;",  # 7
    ]
    start, end = _find_enclosing_sql_range(sql_lines, 2)
    assert start == 1
    assert end == 3

    start, end = _find_enclosing_sql_range(sql_lines, 6)
    assert start == 5
    assert end == 7


def test_deterministic_replace_sql_with_view_complex() -> None:
    from src.skills.code_edit import deterministic_replace_sql_with_view

    # 1. Complex subquery in SELECT
    sql_with_subquery_select = (
        "def query():\n"
        "    return text(\"\"\"\n"
        "        SELECT a, (SELECT count(*) FROM other WHERE other.a = ticket_order.a) as cnt\n"
        "        FROM ticket_order\n"
        "        WHERE a = 1\n"
        "    \"\"\")"
    )
    res = deterministic_replace_sql_with_view(sql_with_subquery_select, "v_my_view")
    assert res is not None
    assert "FROM v_my_view" in res
    assert "WHERE a = 1" in res
    assert "(SELECT count(*) FROM other" in res

    # 2. Subquery in FROM
    sql_with_subquery_from = (
        "def query():\n"
        "    return text(\"\"\"\n"
        "        SELECT a, b\n"
        "        FROM (SELECT * FROM ticket_order) as t\n"
        "        WHERE a = 1\n"
        "    \"\"\")"
    )
    res2 = deterministic_replace_sql_with_view(sql_with_subquery_from, "v_my_view")
    assert res2 is not None
    assert "FROM v_my_view" in res2
    assert "WHERE a = 1" in res2
    assert "(SELECT * FROM ticket_order)" not in res2


def test_extract_all_search_terms_mapping() -> None:
    from src.skills.code_search import _extract_all_search_terms
    terms1 = _extract_all_search_terms("修改登机牌的视图")
    assert "boarding" in terms1
    assert "pass" in terms1
    assert "ticket" in terms1

    terms2 = _extract_all_search_terms("查询机票订单")
    assert "order" in terms2
    assert "ticket" in terms2
    assert "passenger" in terms2


def test_code_search_view_replacement_intent_is_explicit() -> None:
    from src.skills.code_search import _is_sql_view_change

    assert not _is_sql_view_change("重构订单详情查询并统一脱敏")
    assert not _is_sql_view_change("fix SQL query fields")
    assert _is_sql_view_change("使用 view_ticket_report_detail 视图替换订单详情查询")
    assert _is_sql_view_change("replace query with view_ticket_report_detail view")


def test_extract_target_view_from_contract() -> None:
    from src.skills.code_edit import _extract_target_view_from_contract

    # Test extraction with a valid handoff contract
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 450,
                "should_change_to": "use view v_order_detail",
            }
        ]
    }
    view = _extract_target_view_from_contract(contract)
    assert view == "v_order_detail"

    # Test extraction with missing/invalid inputs
    assert _extract_target_view_from_contract(None) == ""
    assert _extract_target_view_from_contract({}) == ""
    assert _extract_target_view_from_contract({"must_modify": []}) == ""


def test_validator_helpers() -> None:
    from src.skills.validator import _find_symbol_ranges, _parse_modified_old_lines, _extract_identifiers

    # Test symbol ranges
    code = (
        "class MyClass:\n"
        "    pass\n"
        "\n"
        "def func_one():\n"
        "    a = 1\n"
        "    return a\n"
        "\n"
        "async def func_two():\n"
        "    pass\n"
    )
    ranges = _find_symbol_ranges(code)
    assert ranges["MyClass"] == (1, 2)
    assert ranges["func_one"] == (4, 6)
    assert ranges["func_two"] == (8, 9)

    # Test diff line parser
    diff = [
        "--- old",
        "+++ new",
        "@@ -1,3 +1,4 @@",
        " def main():",
        "-    print('hello')",
        "+    print('world')",
        "+    a = 1",
        "     return",
    ]
    lines = _parse_modified_old_lines(diff)
    assert lines == [2]

    # Test identifiers extraction
    idents = _extract_identifiers("SELECT * FROM ticket_order JOIN flight_info")
    assert "SELECT" in idents
    assert "ticket_order" in idents
    assert "flight_info" in idents
    assert "from" not in idents
    assert "return" not in idents


@pytest.mark.asyncio
async def test_patch_intent_validator_success(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM ticket_order JOIN passenger_info\"\n"
    )
    new_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM v_order_detail\"\n"
    )
    
    # Write new file
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    # Mock sql init file
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW v_order_detail AS SELECT * FROM ...", encoding="utf-8")
    
    # Mock git show content
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 1,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "use view v_order_detail",
            }
        ]
    }
    
    result = await executor.run(
        "validator",
        SkillContext(
            user_request="用已有的视图替换 build_order_detail_sql()",
            context_pack=None,
        ),
        changed_files=("main.py",),
        handoff_contract=contract,
    )
    
    assert result.success
    assert result.validation_result == "passed"
    assert "build_order_detail_sql" in result.summary
    assert "v_order_detail" in result.summary
    
    # Check details JSON
    details_json = result.metadata.get("validation_details")
    assert details_json is not None
    import json
    details = json.loads(details_json)
    assert len(details) == 1
    assert details[0]["changed_file"] == "main.py"
    assert "build_order_detail_sql" in details[0]["changed_symbols"]
    assert "v_order_detail" in details[0]["added_identifiers"]
    assert "ticket_order" in details[0]["removed_identifiers"]


@pytest.mark.asyncio
async def test_patch_intent_validator_failure_missing_view(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM ticket_order JOIN passenger_info\"\n"
    )
    new_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM v_hallucinated_view\"\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    # Write empty sql init file
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE TABLE other_table (id INT)", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 1,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "use view v_hallucinated_view",
            }
        ]
    }
    
    result = await executor.run(
        "validator",
        SkillContext(
            user_request="用已有的视图替换 build_order_detail_sql()",
            context_pack=None,
        ),
        changed_files=("main.py",),
        handoff_contract=contract,
    )
    
    assert not result.success
    assert result.validation_result == "failed"
    assert "does not exist in the repository's database schema" in result.summary


@pytest.mark.asyncio
async def test_patch_intent_validator_out_of_bounds(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    from src.context.pack import ContextPack, ContextSnippet
    
    old_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM ticket_order JOIN passenger_info\"\n"
    )
    new_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM v_order_detail\"\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW v_order_detail AS SELECT * FROM ...", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    # Snippet range is at lines 10-20, but the change happened at lines 1-2
    snippet = ContextSnippet(file_path="main.py", start_line=10, end_line=20, text="")
    context_pack = ContextPack(user_request="", focused_snippets=(snippet,))
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 1,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "use view v_order_detail",
            }
        ]
    }
    
    result = await executor.run(
        "validator",
        SkillContext(
            user_request="用已有的视图替换 build_order_detail_sql()",
            context_pack=context_pack,
        ),
        changed_files=("main.py",),
        handoff_contract=contract,
    )
    
    assert not result.success
    assert "outside any allowed snippet ranges" in result.summary


@pytest.mark.asyncio
async def test_patch_intent_validator_old_table_not_eliminated(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM ticket_order JOIN passenger_info\"\n"
    )
    # The new code still has primary table ticket_order JOINed!
    new_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM v_order_detail JOIN ticket_order\"\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW v_order_detail AS SELECT * FROM ...; CREATE TABLE ticket_order (id INT);", encoding="utf-8")

    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 1,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "use view v_order_detail",
            }
        ]
    }
    
    result = await executor.run(
        "validator",
        SkillContext(
            user_request="用已有的视图替换 build_order_detail_sql()",
            context_pack=None,
        ),
        changed_files=("main.py",),
        handoff_contract=contract,
    )
    
    assert not result.success
    assert "still referenced in the modified function" in result.summary


@pytest.mark.asyncio
async def test_code_edit_hash_check_no_modification(tmp_path, monkeypatch) -> None:
    from src.skills.code_edit import CodeEditSkill
    from src.planner.patch_plan import PatchPlan, PatchEdit
    
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")
    
    plan = PatchPlan(
        files_to_edit=("main.py",),
        intended_changes=("no-op change",),
        target_symbols=("some_symbol",),
        confidence=0.9,
        edits=(
            PatchEdit(
                path="main.py",
                old_string="SELECT * FROM orders",
                new_string="SELECT * FROM view_orders",
            ),
        ),
    )
    
    # Mock hashlib.sha256 to always return the same hash
    import hashlib
    class FakeHash:
        def hexdigest(self):
            return "same_hash"
    
    monkeypatch.setattr(hashlib, "sha256", lambda data: FakeHash())
    
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path)])
    
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="no-op", patch_plan=plan),
    )
    
    assert not result.success
    assert "file hash did not change" in result.summary


@pytest.mark.asyncio
async def test_validator_sql_semantic_checks_field_loss(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT o.order_id, o.p_id FROM ticket_order o\"\n"
    )
    # The new code is missing o.p_id!
    new_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT o.order_id FROM v_order_detail o\"\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW v_order_detail AS SELECT * FROM ...", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 1,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "use view v_order_detail",
            }
        ]
    }
    
    result = await executor.run(
        "validator",
        SkillContext(
            user_request="用已有的视图替换 build_order_detail_sql()",
            context_pack=None,
        ),
        changed_files=("main.py",),
        handoff_contract=contract,
    )
    
    assert not result.success
    assert "SQL semantic check failed: SELECT fields are lost: ['p_id']" in result.summary


@pytest.mark.asyncio
async def test_validator_sql_semantic_checks_select_star_error(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT o.order_id, o.p_id FROM ticket_order o\"\n"
    )
    # The new code uses SELECT *!
    new_code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return \"SELECT * FROM v_order_detail o\"\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW v_order_detail AS SELECT * FROM ...", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    contract = {
        "must_modify": [
            {
                "file": "main.py",
                "line": 1,
                "symbol_or_api": "build_order_detail_sql",
                "should_change_to": "use view v_order_detail",
            }
        ]
    }
    
    result = await executor.run(
        "validator",
        SkillContext(
            user_request="用已有的视图替换 build_order_detail_sql()",
            context_pack=None,
        ),
        changed_files=("main.py",),
        handoff_contract=contract,
    )
    
    assert not result.success
    assert "SQL semantic check failed: Avoid using SELECT *, please preserve the original SELECT field list." in result.summary


def test_replace_table_with_view_in_sql() -> None:
    from src.skills.code_edit import replace_table_with_view_in_sql
    
    sql = "SELECT o.order_id, o.p_id FROM ticket_order o JOIN passenger_info p ON p.p_id = o.p_id"
    res = replace_table_with_view_in_sql(sql, "v_order_detail")
    assert res == "SELECT o.order_id, o.p_id FROM v_order_detail o JOIN passenger_info p ON p.p_id = o.p_id"


def test_deterministic_replace_sql_with_view_preserves_structure() -> None:
    from src.skills.code_edit import deterministic_replace_sql_with_view
    
    code = (
        "def query():\n"
        "    return \"\"\"\n"
        "    SELECT o.id, o.num\n"
        "    FROM my_table o\n"
        "    JOIN other p ON p.id = o.id\n"
        "    \"\"\"\n"
    )
    res = deterministic_replace_sql_with_view(code, "my_view")
    assert "SELECT o.id, o.num" in res
    assert "FROM my_view o" in res
    assert "JOIN other p ON p.id = o.id" in res
    assert "SELECT *" not in res


def test_generate_sql_patch_uses_dependency_columns_and_removes_replaced_joins() -> None:
    from src.skills.code_edit import generate_sql_patch

    code = (
        "def query():\n"
        "    return \"\"\"\n"
        "    SELECT o.order_id, p.passenger_name, f.flight_no\n"
        "    FROM ticket_order o\n"
        "    JOIN passenger_info p ON p.p_id = o.p_id\n"
        "    JOIN keep_table k ON k.id = o.k_id\n"
        "    LEFT JOIN flight_info f ON f.flight_id = o.flight_id\n"
        "    WHERE o.status = :status\n"
        "    ORDER BY o.created_at DESC\n"
        "    LIMIT 10\n"
        "    \"\"\"\n"
    )
    ctx = {
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "view_ticket_report_detail",
            "columns": [
                "order_id", "passenger_name", "flight_no", "p_id",
                "k_id", "flight_id", "status", "created_at",
            ],
            "replaces_objects": ["ticket_order", "passenger_info", "flight_info"],
        }]
    }

    res = generate_sql_patch(code, ctx)
    assert res is not None
    assert "FROM view_ticket_report_detail v" in res
    assert "v.order_id" in res
    assert "v.passenger_name" in res
    assert "v.flight_no" in res
    assert "JOIN keep_table k" in res
    assert "passenger_info" not in res
    assert "flight_info" not in res
    assert "ticket_order" not in res
    assert "o." not in res
    assert "p." not in res
    assert "f." not in res
    assert "WHERE v.status = :status" in res
    assert "ORDER BY v.created_at DESC" in res
    assert "LIMIT 10" in res


def test_generate_sql_patch_raises_missing_column_mapping() -> None:
    from src.skills.code_edit import ProjectionMappingError, generate_sql_patch

    code = (
        "def query():\n"
        "    return \"SELECT o.order_id, p.unknown_value FROM ticket_order o "
        "JOIN passenger_info p ON p.p_id = o.p_id\"\n"
    )
    ctx = {
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "view_ticket_report_detail",
            "columns": ["order_id"],
            "replaces_objects": ["ticket_order", "passenger_info"],
        }]
    }

    with pytest.raises(ProjectionMappingError):
        generate_sql_patch(code, ctx)


def test_build_edit_context_includes_view_dependency_columns(tmp_path) -> None:
    from src.skills.code_search import _build_edit_plan_context

    (tmp_path / "schema.sql").write_text(
        "CREATE VIEW view_ticket_report_detail AS\n"
        "SELECT o.order_id, p.name AS passenger_name, f.flight_no\n"
        "FROM ticket_order o\n"
        "JOIN passenger_info p ON p.p_id = o.p_id\n"
        "JOIN flight_info f ON f.flight_id = o.flight_id;\n",
        encoding="utf-8",
    )
    ctx = _build_edit_plan_context(
        [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 3,
            "current_code": "def query():\n    return 'SELECT o.order_id FROM ticket_order o'",
        }],
        intended_change="use view_ticket_report_detail view",
        project_root=tmp_path,
        task_analysis={"edit_strategy": "sql_view_rewrite", "intent": "sql_view_rewrite"},
        target_view="view_ticket_report_detail",
    )

    assert ctx is not None
    deps = ctx["resolved_dependencies"]
    assert isinstance(deps, list)
    dep = deps[0]
    assert dep["name"] == "view_ticket_report_detail"
    assert dep["columns"] == ["order_id", "passenger_name", "flight_no"]
    assert dep["replaces_objects"] == ["ticket_order", "passenger_info", "flight_info"]


def test_hydrate_snippets_simplifies_long_sql_for_edit_context(tmp_path) -> None:
    from src.skills.code_search import _hydrate_snippets

    columns = ",\n        ".join(f"o.col_{idx} AS col_{idx}" for idx in range(60))
    code = (
        "def build_order_detail_sql(where_clause):\n"
        "    return f\"\"\"\n"
        f"    SELECT {columns}\n"
        "    FROM ticket_order o\n"
        "    JOIN passenger_info p ON p.p_id = o.p_id\n"
        "    WHERE {where_clause}\n"
        "    \"\"\"\n"
    )
    (tmp_path / "main.py").write_text(code, encoding="utf-8")

    hydrated = _hydrate_snippets(
        tmp_path,
        "main.py:1:def build_order_detail_sql(where_clause):",
        intended_change="使用 view_ticket_report_detail 视图替换 build_order_detail_sql",
        task_analysis={"edit_strategy": "sql_view_rewrite", "intent": "sql_view_rewrite"},
        target_view="view_ticket_report_detail",
    )

    assert hydrated.edit_context is not None
    targets = hydrated.edit_context["editable_targets"]
    assert isinstance(targets, list)
    current_code = targets[0]["current_code"]
    display_code = targets[0]["display_code"]
    assert "COLUMN MAPPING" in display_code
    assert "COLUMN MAPPING" not in current_code
    assert "FROM ticket_order o" in display_code
    assert "JOIN passenger_info p" in display_code
    assert "col_59 AS col_59" not in display_code
    assert "col_59 AS col_59" in current_code


def test_build_edit_context_does_not_infer_view_for_function_refactor(tmp_path) -> None:
    from src.skills.code_search import _build_edit_plan_context

    (tmp_path / "schema.sql").write_text(
        "CREATE VIEW view_ticket_report_detail AS SELECT order_id FROM ticket_order;\n",
        encoding="utf-8",
    )
    ctx = _build_edit_plan_context(
        [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 3,
            "current_code": "def query():\n    return 'SELECT order_id FROM ticket_order'",
        }],
        intended_change="重构订单详情处理，复用 normalize_order_record 做脱敏",
        project_root=tmp_path,
        task_analysis={"edit_strategy": "function_refactor", "intent": "function_refactor"},
    )

    assert ctx is not None
    assert ctx["edit_strategy"] == "function_refactor"
    assert ctx["target_view"] == ""
    assert ctx["resolved_dependencies"] == []


def test_build_edit_context_resolves_target_view_from_task_analysis_and_patch_intent(tmp_path) -> None:
    from src.skills.code_search import _build_edit_plan_context

    (tmp_path / "schema.sql").write_text(
        "CREATE VIEW view_ticket_report_detail AS SELECT order_id FROM ticket_order;\n",
        encoding="utf-8",
    )
    # Check 1: task_analysis has target_view directly
    ctx1 = _build_edit_plan_context(
        [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 3,
            "current_code": "def query():\n    return 'SELECT order_id FROM ticket_order'",
        }],
        intended_change="use view_ticket_report_detail view",
        project_root=tmp_path,
        task_analysis={
            "edit_strategy": "sql_view_rewrite", 
            "intent": "sql_view_rewrite",
            "target_view": "view_ticket_report_detail"
        },
    )
    assert ctx1 is not None
    assert ctx1["dependencies_resolved"] is True
    assert ctx1["target_view"] == "view_ticket_report_detail"

    # Check 2: task_analysis has patch_intent with target_view
    ctx2 = _build_edit_plan_context(
        [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 3,
            "current_code": "def query():\n    return 'SELECT order_id FROM ticket_order'",
        }],
        intended_change="use view_ticket_report_detail view",
        project_root=tmp_path,
        task_analysis={
            "edit_strategy": "sql_view_rewrite", 
            "intent": "sql_view_rewrite",
            "patch_intent": {
                "target_view": "view_ticket_report_detail"
            }
        },
    )
    assert ctx2 is not None
    assert ctx2["dependencies_resolved"] is True
    assert ctx2["target_view"] == "view_ticket_report_detail"



def test_validator_sql_replacement_contract_checks_aliases_and_columns() -> None:
    from src.skills.validator import _validate_sql_replacement_contract

    old_code = (
        "def query():\n"
        "    return \"SELECT o.order_id, p.name AS passenger_name "
        "FROM ticket_order o JOIN passenger_info p ON p.p_id = o.p_id "
        "WHERE p.status = 1\"\n"
    )
    new_code = (
        "def query():\n"
        "    return \"SELECT v.order_id, p.name AS passenger_name "
        "FROM view_ticket_report_detail v JOIN passenger_info p ON p.p_id = v.p_id "
        "WHERE p.status = 1\"\n"
    )

    errors = _validate_sql_replacement_contract(
        old_code,
        new_code,
        target_view="view_ticket_report_detail",
        replaces_objects=["ticket_order", "passenger_info"],
        dependency_columns=["order_id", "passenger_name", "p_id"],
    )
    joined = "; ".join(errors)
    assert "replaces_objects still referenced" in joined
    assert "old aliases still appear" in joined


@pytest.mark.asyncio
async def test_code_edit_skill_new_intent_schema(tmp_path) -> None:
    import json
    from src.skills.code_edit import CodeEditSkill
    
    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")
    
    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        if "ReplacementBuilder" in system:
            return json.dumps({"new_string": "sql = 'SELECT * FROM view_ticket_report_detail'"})
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_dependency",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })
        
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "task_intent": {
            "operation": "replace_dependency",
            "target_symbol": "query",
            "goal": "use view instead of table"
        },
        "edit_targets": [{
            "file": "main.py",
            "symbol": "query",
            "start_line": 1,
            "end_line": 1,
            "current_code": "sql = 'SELECT * FROM orders'",
        }],
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "view_ticket_report_detail",
            "evidence": [],
            "confidence": 0.95
        }],
        "constraints": ["Do not invent dependencies"],
        "acceptance": [{"type": "diff_must_touch_symbol"}],
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "scope": ["main.py"],
        },
        # backward compatibility
        "editable_targets": [{
            "file": "main.py",
            "start_line": 1,
            "end_line": 1,
            "current_code": "sql = 'SELECT * FROM orders'",
        }]
    }
    
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    assert result.success
    assert result.changed_files == ("main.py",)
    assert "view_ticket_report_detail" in target.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_validator_skill_new_intent_schema(tmp_path, monkeypatch) -> None:
    import json
    from src.skills.validator import ValidatorSkill
    
    old_code = "def query():\n    return \"SELECT o.id FROM orders o\"\n"
    new_code = "def query():\n    return \"SELECT o.id FROM view_ticket_report_detail o\"\n"
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW view_ticket_report_detail AS SELECT * FROM ...", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "task_intent": {
            "operation": "replace_dependency",
            "target_symbol": "query",
            "goal": "use view"
        },
        "edit_targets": [{
            "file": "main.py",
            "symbol": "query",
            "start_line": 1,
            "end_line": 2,
            "current_code": old_code,
        }],
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "view_ticket_report_detail",
            "evidence": [],
            "confidence": 0.95
        }],
    }
    
    result = await executor.run(
        "validator",
        SkillContext(user_request="replace table with view"),
        changed_files=("main.py",),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    
    assert result.success
    assert result.validation_result == "passed"


@pytest.mark.asyncio
async def test_code_edit_skill_metadata_driven_join_elimination(tmp_path) -> None:
    import json
    from src.skills.code_edit import CodeEditSkill
    
    target = tmp_path / "main.py"
    target.write_text(
        "sql = '''\n"
        "SELECT o.id FROM orders o\n"
        "JOIN passenger p ON p.id = o.p_id\n"
        "JOIN keep_table k ON k.id = o.k_id\n"
        "'''\n",
        encoding="utf-8"
    )
    
    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        if "ReplacementBuilder" in system:
            return json.dumps({"new_string": "sql = 'SELECT o.id FROM my_view o JOIN keep_table k ON k.id = o.k_id'"})
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_dependency",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })
        
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    targets = [{
        "file": "main.py",
        "start_line": 1,
        "end_line": 5,
        "current_code": (
            "sql = '''\n"
            "SELECT o.id FROM orders o\n"
            "JOIN passenger p ON p.id = o.p_id\n"
            "JOIN keep_table k ON k.id = o.k_id\n"
            "'''"
        ),
    }]
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "task_intent": {"operation": "replace_dependency", "goal": "use view"},
        "edit_targets": targets,
        "editable_targets": targets,
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger"],
        }],
        "constraints": ["Do not invent dependencies"],
        "acceptance": [{"type": "compile_or_syntax_check"}],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }
    
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    assert result.success
    content = target.read_text(encoding="utf-8")
    assert "my_view" in content
    assert "keep_table" in content
    assert "passenger" not in content


@pytest.mark.asyncio
async def test_validator_skill_metadata_driven_forbidden_tables(tmp_path, monkeypatch) -> None:
    import json
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def query():\n"
        "    return 'SELECT o.id FROM orders o JOIN passenger p ON p.id = o.p_id JOIN keep_table k'\n"
    )
    new_code = (
        "def query():\n"
        "    return 'SELECT o.id FROM my_view o JOIN keep_table k'\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW my_view AS SELECT * FROM ...", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "task_intent": {"operation": "replace_dependency", "target_symbol": "query"},
        "edit_targets": [{"file": "main.py", "symbol": "query", "current_code": old_code}],
        "resolved_dependencies": [{
            "role": "replacement_source",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger"],
        }],
    }
    
    result = await executor.run(
        "validator",
        SkillContext(user_request="replace table with view"),
        changed_files=("main.py",),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    
    assert result.success
    assert result.validation_result == "passed"


@pytest.mark.asyncio
async def test_code_edit_skill_projection_rewrite_success(tmp_path) -> None:
    import json
    from src.skills.code_edit import CodeEditSkill
    
    target = tmp_path / "main.py"
    target.write_text(
        "sql = '''\n"
        "SELECT o.id, p.name AS passenger_name, f.flight_no\n"
        "FROM orders o\n"
        "JOIN passenger p ON p.id = o.p_id\n"
        "LEFT JOIN flight f ON f.id = o.flight_id\n"
        "'''\n",
        encoding="utf-8"
    )
    
    async def complete(messages: list[dict[str, object]]) -> str:
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_dependency",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })
        
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    targets = [{
        "file": "main.py",
        "start_line": 1,
        "end_line": 6,
        "current_code": (
            "sql = '''\n"
            "SELECT o.id, p.name AS passenger_name, f.flight_no\n"
            "FROM orders o\n"
            "JOIN passenger p ON p.id = o.p_id\n"
            "LEFT JOIN flight f ON f.id = o.flight_id\n"
            "'''"
        ),
    }]
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "task_intent": {"operation": "replace_dependency", "goal": "use view"},
        "edit_targets": targets,
        "editable_targets": targets,
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger", "flight"],
            "columns": ["id", "passenger_name", "flight_no"],
        }],
        "constraints": ["Do not invent dependencies"],
        "acceptance": [{"type": "compile_or_syntax_check"}],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }
    
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    assert result.success
    content = target.read_text(encoding="utf-8")
    assert "my_view" in content
    assert "v.id" in content
    assert "v.passenger_name" in content
    assert "v.flight_no" in content
    assert "JOIN passenger" not in content
    assert "LEFT JOIN flight" not in content


@pytest.mark.asyncio
async def test_code_edit_skill_projection_rewrite_failure(tmp_path) -> None:
    import json
    from src.skills.code_edit import CodeEditSkill
    
    target = tmp_path / "main.py"
    target.write_text(
        "sql = '''\n"
        "SELECT o.id, p.name AS passenger_name, f.flight_no\n"
        "FROM orders o\n"
        "JOIN passenger p ON p.id = o.p_id\n"
        "LEFT JOIN flight f ON f.id = o.flight_id\n"
        "'''\n",
        encoding="utf-8"
    )
    
    async def complete(messages: list[dict[str, object]]) -> str:
        return json.dumps({
            "edits": [{
                "target_index": 0,
                "operation": "replace_dependency",
            }],
            "confidence": 0.9,
            "missing_info": [],
        })
        
    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    targets = [{
        "file": "main.py",
        "start_line": 1,
        "end_line": 6,
        "current_code": (
            "sql = '''\n"
            "SELECT o.id, p.name AS passenger_name, f.flight_no\n"
            "FROM orders o\n"
            "JOIN passenger p ON p.id = o.p_id\n"
            "LEFT JOIN flight f ON f.id = o.flight_id\n"
            "'''"
        ),
    }]
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "task_intent": {"operation": "replace_dependency", "goal": "use view"},
        "edit_targets": targets,
        "editable_targets": targets,
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger", "flight"],
            "columns": ["id", "passenger_name"],
        }],
        "constraints": ["Do not invent dependencies"],
        "acceptance": [{"type": "compile_or_syntax_check"}],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }
    
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change query"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    assert not result.success
    assert "diagnose_strategy_mismatch" in result.missing_info


@pytest.mark.asyncio
async def test_code_edit_rejects_sql_operation_for_function_refactor_strategy(tmp_path) -> None:
    import json
    from src.skills.code_edit import CodeEditSkill

    target = tmp_path / "main.py"
    target.write_text("sql = 'SELECT * FROM orders'\n", encoding="utf-8")

    async def complete(_messages: list[dict[str, object]]) -> str:
        return json.dumps({
            "edits": [{"target_index": 0, "operation": "replace_sql_source"}],
            "confidence": 0.9,
            "missing_info": [],
        })

    targets = [{
        "file": "main.py",
        "start_line": 1,
        "end_line": 1,
        "current_code": "sql = 'SELECT * FROM orders'",
    }]
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "function_refactor",
        "task_intent": {"operation": "general_edit", "goal": "refactor"},
        "edit_targets": targets,
        "editable_targets": targets,
        "constraints": ["Do not invent dependencies"],
        "acceptance": [{"type": "compile_or_syntax_check"}],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }

    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="重构订单详情处理，统一脱敏"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
        task_analysis={
            "edit_strategy": "function_refactor",
            "edit_ready": True,
            "acceptance_contract": {"intent": "function_refactor"},
        },
    )
    assert not result.success
    assert "diagnose_strategy_mismatch" in result.missing_info


@pytest.mark.asyncio
async def test_validator_skill_forbidden_aliases(tmp_path, monkeypatch) -> None:
    import json
    from src.skills.validator import ValidatorSkill
    
    old_code = (
        "def query():\n"
        "    return 'SELECT o.id, p.name FROM orders o JOIN passenger p ON p.id = o.p_id'\n"
    )
    new_code = (
        "def query():\n"
        "    return 'SELECT o.id, p.name FROM my_view o'\n"
    )
    
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    
    sql_dir = tmp_path / "db" / "init"
    sql_dir.mkdir(parents=True)
    (sql_dir / "init.sql").write_text("CREATE VIEW my_view AS SELECT * FROM ...", encoding="utf-8")
    
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)
    
    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    
    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "task_intent": {"operation": "replace_dependency", "target_symbol": "query"},
        "edit_targets": [{"file": "main.py", "symbol": "query", "current_code": old_code}],
        "resolved_dependencies": [{
            "role": "replacement_source",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger"],
        }],
    }
    
    result = await executor.run(
        "validator",
        SkillContext(user_request="replace table with view"),
        changed_files=("main.py",),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    
    assert not result.success
    assert any("legacy alias 'p' is still referenced" in err for err in result.missing_info)


@pytest.mark.asyncio
async def test_validator_function_refactor_requires_unified_helper(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill

    old_code = (
        "def query(row, other):\n"
        "    row['passenger_id_no'] = row['passenger_id_no'][:3] + '***'\n"
        "    other['passenger_id_no'] = other['passenger_id_no'][:3] + '***'\n"
    )
    new_code = (
        "def query(row, other):\n"
        "    row['passenger_id_no'] = row['passenger_id_no'][:3] + '***'\n"
        "    other['passenger_id_no'] = other['passenger_id_no'][:3] + '***'\n"
    )
    target_file = tmp_path / "main.py"
    target_file.write_text(new_code, encoding="utf-8")
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)

    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    result = await executor.run(
        "validator",
        SkillContext(user_request="重构订单详情处理，统一脱敏"),
        changed_files=("main.py",),
        task_analysis={"edit_strategy": "function_refactor", "intent": "function_refactor"},
    )

    assert not result.success
    assert "expected unified helper" in result.summary


def test_optimize_snippet_body_sql_columns() -> None:
    from src.skills.code_search import _optimize_snippet_body
    body = (
        "10: def get_order_sql():\n"
        "11:     return '''\n"
        "12:         SELECT\n"
        "13:             o.id AS order_id,\n"
        "14:             o.order_no,\n"
        "15:             o.passenger_id AS pass_id,\n"
        "16:             o.status,\n"
        "17:             o.amount\n"
        "18:         FROM orders o\n"
        "19:         WHERE o.id = 1\n"
        "20:     '''"
    )
    optimized = _optimize_snippet_body(body, "main.py")
    assert "COLUMN MAPPING" in optimized
    assert "order_id" in optimized
    assert "order_no" in optimized
    assert "pass_id" in optimized
    assert "status" in optimized
    assert "amount" in optimized
    assert "FROM orders o" in optimized
    assert "WHERE o.id = 1" in optimized


@pytest.mark.asyncio
async def test_code_edit_sql_fallback_success(tmp_path) -> None:
    from src.skills.code_edit import CodeEditSkill
    import json

    target = tmp_path / "main.py"
    # Function signature: def query():
    old_code = (
        "def query():\n"
        "    sql = \"\"\"\n"
        "    SELECT o.id, p.name AS passenger_name\n"
        "    FROM orders o\n"
        "    JOIN passenger p ON p.id = o.p_id\n"
        "    \"\"\"\n"
        "    return sql\n"
    )
    target.write_text(old_code, encoding="utf-8")

    # Mock llm_complete for two calls:
    # 1. EditPlanBuilder (returns plan payload)
    # 2. ReplacementBuilder (returns fallback string)
    call_count = 0
    async def complete(_messages: list[dict[str, object]]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({
                "edits": [{"target_index": 0, "operation": "replace_dependency", "target_view": "my_view"}],
                "confidence": 0.9,
                "missing_info": [],
            })
        else:
            # Fallback output that passes compilation, signature unchanged, body changed, references my_view, and removes legacy refs
            new_code = (
                "def query():\n"
                "    sql = \"\"\"\n"
                "    SELECT v.id, v.passenger_name\n"
                "    FROM my_view v\n"
                "    \"\"\"\n"
                "    return sql\n"
            )
            return json.dumps({"new_string": new_code})

    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "task_intent": {"operation": "replace_dependency", "target_symbol": "query", "goal": "use my_view"},
        "edit_targets": [{
            "file": "main.py",
            "symbol": "query",
            "current_code": old_code,
            "start_line": 1,
            "end_line": 7,
        }],
        "editable_targets": [{
            "file": "main.py",
            "symbol": "query",
            "current_code": old_code,
            "start_line": 1,
            "end_line": 7,
        }],
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger"],
            "columns": ["id"], # "passenger_name" is missing -> triggers ProjectionMappingError!
        }],
        "acceptance": [{"type": "compile_or_syntax_check"}],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }

    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="use my_view"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    assert result.success
    assert call_count == 2
    updated = target.read_text(encoding="utf-8")
    assert "my_view v" in updated
    assert "orders o" not in updated


@pytest.mark.asyncio
async def test_code_edit_sql_fallback_validation_failure(tmp_path) -> None:
    from src.skills.code_edit import CodeEditSkill
    import json

    target = tmp_path / "main.py"
    old_code = (
        "def query():\n"
        "    sql = \"\"\"\n"
        "    SELECT o.id, p.name AS passenger_name\n"
        "    FROM orders o\n"
        "    JOIN passenger p ON p.id = o.p_id\n"
        "    \"\"\"\n"
        "    return sql\n"
    )
    target.write_text(old_code, encoding="utf-8")

    call_count = 0
    async def complete(_messages: list[dict[str, object]]) -> str:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return json.dumps({
                "edits": [{"target_index": 0, "operation": "replace_dependency", "target_view": "my_view"}],
                "confidence": 0.9,
                "missing_info": [],
            })
        else:
            # Fallback output that fails validation (references old alias p.)
            new_code = (
                "def query():\n"
                "    sql = \"\"\"\n"
                "    SELECT v.id, p.passenger_name\n"
                "    FROM my_view v\n"
                "    \"\"\"\n"
                "    return sql\n"
            )
            return json.dumps({"new_string": new_code})

    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "sql_view_rewrite",
        "task_intent": {"operation": "replace_dependency", "target_symbol": "query", "goal": "use my_view"},
        "edit_targets": [{
            "file": "main.py",
            "symbol": "query",
            "current_code": old_code,
            "start_line": 1,
            "end_line": 7,
        }],
        "editable_targets": [{
            "file": "main.py",
            "symbol": "query",
            "current_code": old_code,
            "start_line": 1,
            "end_line": 7,
        }],
        "resolved_dependencies": [{
            "role": "replacement_source",
            "kind": "database_view",
            "name": "my_view",
            "replaces_objects": ["orders", "passenger"],
            "columns": ["id"], # missing passenger_name -> ProjectionMappingError!
        }],
        "acceptance": [{"type": "compile_or_syntax_check"}],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }

    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="use my_view"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )
    assert not result.success
    assert call_count == 2
    assert "diagnose_strategy_mismatch" in result.missing_info
    # Verify structured failure details
    err_msg = result.metadata["raw_preview"]
    assert "projection_mapping_failed" in err_msg
    assert "llm_replacement_failed" in err_msg
    assert "passenger_name" in err_msg # the missing column name from ProjectionMappingError


@pytest.mark.asyncio
async def test_code_edit_materializes_patch_intent_target_when_hydration_wrong_symbol(tmp_path) -> None:
    from src.skills.code_edit import CodeEditSkill
    import json

    target = tmp_path / "main.py"
    target.write_text(
        "def admin_update_order():\n"
        "    return 'wrong'\n\n"
        "def build_order_detail_sql():\n"
        "    return 'SELECT * FROM ticket_order'\n",
        encoding="utf-8",
    )

    async def complete(messages: list[dict[str, object]]) -> str:
        system = str(messages[0].get("content") or "")
        if "ReplacementBuilder" in system:
            return json.dumps({
                "new_string": (
                    "def build_order_detail_sql():\n"
                    "    return 'SELECT * FROM view_ticket_report_detail'\n"
                )
            })
        return json.dumps({
            "edits": [{
                "target_index": 0,
            }],
            "confidence": 0.9,
            "missing_info": [],
        })

    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "code_edit_ready": True,
        "edit_strategy": "general_edit",
        "patch_intent": {
            "edit_ready": True,
            "edit_strategy": "general_edit",
            "edit_targets": [{
                "file": "main.py",
                "symbol": "build_order_detail_sql",
            }],
            "acceptance_criteria": ["build_order_detail_sql changed"],
        },
        "editable_targets": [{
            "file": "main.py",
            "symbol": "admin_update_order",
            "start_line": 1,
            "end_line": 2,
            "current_code": "def admin_update_order():\n    return 'wrong'",
            "intended_change": "change order detail SQL",
            "acceptance_criteria": ["wrong hydration candidate"],
        }],
        "intended_change": "change order detail SQL",
        "acceptance_criteria": ["build_order_detail_sql changed"],
        "tool_policy": {"allowed_tools": ["edit_file"], "scope": ["main.py"]},
    }

    executor = SkillExecutor([CodeEditSkill(project_root=tmp_path, llm_complete=complete)])
    result = await executor.run(
        "code_edit",
        SkillContext(user_request="change build_order_detail_sql"),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
    )

    assert result.success
    updated = target.read_text(encoding="utf-8")
    assert "def admin_update_order():\n    return 'wrong'" in updated
    assert "view_ticket_report_detail" in updated


@pytest.mark.asyncio
async def test_validator_symbol_mismatch_requires_replan(tmp_path, monkeypatch) -> None:
    from src.skills.validator import ValidatorSkill
    import json

    old_code = (
        "def admin_update_order():\n"
        "    return 'wrong'\n\n"
        "def build_order_detail_sql():\n"
        "    return 'old'\n"
    )
    new_code = (
        "def admin_update_order():\n"
        "    return 'changed'\n\n"
        "def build_order_detail_sql():\n"
        "    return 'old'\n"
    )
    (tmp_path / "main.py").write_text(new_code, encoding="utf-8")
    monkeypatch.setattr("src.skills.validator.get_git_head_content", lambda root, rel: old_code)

    edit_context = {
        "schema": "mitkii.edit_context.v2",
        "edit_strategy": "general_edit",
        "editable_targets": [{
            "file": "main.py",
            "symbol": "build_order_detail_sql",
            "start_line": 4,
            "end_line": 5,
            "current_code": "def build_order_detail_sql():\n    return 'old'",
        }],
        "patch_intent": {
            "edit_targets": [{"file": "main.py", "symbol": "build_order_detail_sql"}],
        },
    }

    executor = SkillExecutor([ValidatorSkill(project_root=tmp_path)])
    result = await executor.run(
        "validator",
        SkillContext(user_request="change build_order_detail_sql"),
        changed_files=("main.py",),
        search_output="EDIT_CONTEXT_JSON\n" + json.dumps(edit_context),
        task_analysis={"edit_strategy": "general_edit", "intent": "general_edit"},
    )

    assert not result.success
    assert result.requires_fallback
    assert result.metadata["failure_code"] == "validator_context_mismatch"
    assert "target_symbol_mismatch" in result.metadata["structured_errors"]
