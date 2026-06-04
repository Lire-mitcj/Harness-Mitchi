from __future__ import annotations

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


async def fake_edit_complete(_messages: list[dict[str, object]]) -> str:
    return (
        '{"edits":[{"path":"main.py","old_string":"SELECT * FROM orders",'
        '"new_string":"SELECT * FROM view_ticket_report_detail"}],'
        '"confidence":0.9,"missing_info":[]}'
    )


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
        search_output="main.py:1: sql = 'SELECT * FROM orders'",
    )

    assert result.success
    assert result.changed_files == ("main.py",)
    assert "view_ticket_report_detail" in target.read_text(encoding="utf-8")


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
