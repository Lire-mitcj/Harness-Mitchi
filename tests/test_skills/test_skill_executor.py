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
async def test_code_search_skill_batches_map_and_grep(tmp_path) -> None:
    registry = FakeToolRegistry()
    executor = SkillExecutor([
        CodeSearchSkill(project_root=tmp_path, tools=registry),  # type: ignore[arg-type]
    ])

    result = await executor.run(
        "code_search",
        SkillContext(user_request="把登机牌查询接口改成用视图查询"),
    )

    assert result.success
    assert [name for name, _args in registry.calls[:3]] == [
        "map_search",
        "map_search",
        "map_search",
    ]
    grep_calls = [args for name, args in registry.calls if name == "grep_search"]
    assert [args["include"] for args in grep_calls] == ["*.py", "*.sql"]
    pattern = str(grep_calls[0]["pattern"])
    assert "boarding_pass" in pattern
    assert "CREATE" in pattern
    assert "VIEW" in pattern
    assert "api" not in pattern.lower()
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
    assert all("CREATE" in pattern and "VIEW" in pattern for pattern in grep_patterns)


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
    assert all("CREATE" in pattern and "VIEW" in pattern for pattern in grep_patterns)


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
    assert all("boarding_pass" in pattern for pattern in grep_patterns)
    assert all("CREATE" in pattern and "VIEW" in pattern for pattern in grep_patterns)
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
    assert "snippet skipped by display token budget" in result.metadata["hydration_failures"]


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
