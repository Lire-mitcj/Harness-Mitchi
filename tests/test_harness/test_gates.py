from __future__ import annotations

from pathlib import Path

from src.config.settings import MitKIISettings
from src.harness.gates.phase_metrics import PhaseMetrics
from src.harness.gates.plan_gate import (
    ReplanGateContext,
    find_replan_duplicates,
    validate_plan,
)
from src.harness.gates.preflight_probe import assess_preflight
from src.harness.gates.types import GateVerdict
from src.planner.task_tree import SubTaskKind, SubTaskNode, TaskTree


def test_plan_gate_blocks_empty_tree(tmp_path: Path) -> None:
    tree = TaskTree(root_task="x", nodes=[])
    result = validate_plan(tree, tmp_path)
    assert result.verdict == GateVerdict.BLOCK
    assert not result.passed


def test_plan_gate_blocks_duplicate_ids(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="x",
        nodes=[
            SubTaskNode(id="st-1", description="a"),
            SubTaskNode(id="st-1", description="b"),
        ],
    )
    result = validate_plan(tree, tmp_path)
    assert result.verdict == GateVerdict.BLOCK


def test_plan_gate_blocks_dependency_cycle(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="x",
        nodes=[
            SubTaskNode(id="a", description="step a", depends_on=["b"]),
            SubTaskNode(id="b", description="step b", depends_on=["a"]),
        ],
    )
    result = validate_plan(tree, tmp_path)
    assert result.verdict == GateVerdict.BLOCK


def test_plan_gate_passes_valid_tree(tmp_path: Path) -> None:
    f = tmp_path / "api.py"
    f.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix api",
        nodes=[
            SubTaskNode(
                id="st-1",
                description="grep api routes",
                kind=SubTaskKind.DIAGNOSE,
                allowed_tools=["grep_search", "list_dir"],
                acceptance_criteria="Output file:line, symbol, and route snippet/decision",
            ),
            SubTaskNode(
                id="st-2",
                description="edit api",
                kind=SubTaskKind.EDIT,
                context_files=["api.py"],
                allowed_tools=["read_file", "edit_file"],
                acceptance_criteria="api.py updated",
                depends_on=["st-1"],
            ),
        ],
    )
    result = validate_plan(tree, tmp_path)
    assert result.verdict in {GateVerdict.PASS, GateVerdict.WARN}


def test_plan_gate_allows_edit_first_subtask(tmp_path: Path) -> None:
    f = tmp_path / "health.py"
    f.write_text("# new\n")
    tree = TaskTree(
        root_task="add health",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="Create health route",
                context_files=["health.py"],
                allowed_tools=["write_file", "edit_file"],
                acceptance_criteria="route exists",
            ),
        ],
    )
    result = validate_plan(tree, tmp_path)
    assert result.verdict in {GateVerdict.PASS, GateVerdict.WARN}


def test_preflight_green_small_task(tmp_path: Path) -> None:
    (tmp_path / "small.py").write_text("print('hi')\n")
    settings = MitKIISettings(
        data_dir=tmp_path / ".mitkii",
        max_context_tokens=128_000,
        context_budget_ratio=0.75,
    )
    tree = TaskTree(
        root_task="demo",
        nodes=[SubTaskNode(id="st-1", description="read", context_files=["small.py"])],
    )
    node = tree.nodes[0]
    result = assess_preflight(
        subtask=node,
        task_tree=tree,
        project_root=tmp_path,
        settings=settings,
    )
    assert result.verdict == GateVerdict.PASS
    assert result.policy.tier == "green"


def test_phase_metrics_records_duration() -> None:
    pm = PhaseMetrics()
    pm.start("plan")
    pm.end("plan", verdict="PASS")
    summary = pm.get_summary()
    assert len(summary["phases"]) == 1
    assert summary["phases"][0]["phase"] == "plan"
    assert summary["phases"][0]["verdict"] == "PASS"


def test_plan_gate_blocks_replan_duplicate_edit(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text("x = 1\n")
    ctx = ReplanGateContext(
        failed_subtask_id="st-2",
        failed_kind=SubTaskKind.EDIT,
        failed_description="Fix boarding pass SELECT",
        failed_context_files=("app.py",),
    )
    tree = TaskTree(
        root_task="fix view",
        nodes=[
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.DIAGNOSE,
                description="grep boarding pass",
                allowed_tools=["grep_search"],
            ),
            SubTaskNode(
                id="st-4",
                kind=SubTaskKind.EDIT,
                description="Fix boarding pass SELECT",
                context_files=["app.py"],
                allowed_tools=["read_file", "edit_file"],
                depends_on=["st-3"],
            ),
        ],
    )
    dupes = find_replan_duplicates(tree, ctx)
    assert dupes
    result = validate_plan(tree, tmp_path, replan_context=ctx)
    assert result.verdict == GateVerdict.BLOCK
    assert "duplicates failed" in result.messages[0]


def test_plan_gate_allows_replan_with_wider_context(tmp_path: Path) -> None:
    f1 = tmp_path / "app.py"
    f2 = tmp_path / "routes.sql"
    f1.write_text("x = 1\n")
    f2.write_text("SELECT 1;\n")
    ctx = ReplanGateContext(
        failed_subtask_id="st-2",
        failed_kind=SubTaskKind.EDIT,
        failed_description="Fix boarding pass SELECT",
        failed_context_files=("app.py",),
    )
    tree = TaskTree(
        root_task="fix view",
        nodes=[
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.DIAGNOSE,
                description="grep boarding pass in app and sql",
                allowed_tools=["grep_search"],
                acceptance_criteria="Output file:line, symbol, and SQL snippet/decision",
            ),
            SubTaskNode(
                id="st-4",
                kind=SubTaskKind.EDIT,
                description="Update boarding pass view SQL",
                context_files=["routes.sql"],
                allowed_tools=["read_file", "edit_file"],
                acceptance_criteria="Boarding pass SQL uses the revised view",
                depends_on=["st-3"],
            ),
        ],
    )
    assert not find_replan_duplicates(tree, ctx)
    result = validate_plan(tree, tmp_path, replan_context=ctx)
    assert result.verdict in {GateVerdict.PASS, GateVerdict.WARN}


def test_plan_gate_blocks_low_value_read_then_edit(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix app",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="Read and analyze app.py",
                context_files=["app.py"],
                allowed_tools=["read_file"],
                acceptance_criteria="app.py analyzed",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="Fix app.py",
                context_files=["app.py"],
                allowed_tools=["read_file", "edit_file"],
                acceptance_criteria="app.py fixed",
                depends_on=["st-1"],
            ),
        ],
    )

    result = validate_plan(tree, tmp_path)

    assert result.verdict == GateVerdict.BLOCK
    assert any("low-value read/search/analyze" in msg for msg in result.messages)


def test_plan_gate_blocks_diagnose_to_edit_without_handoff_output(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix app",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="Locate app bug",
                allowed_tools=["grep_search"],
                acceptance_criteria="candidate files listed",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="Fix app bug",
                context_files=["app.py"],
                allowed_tools=["read_file", "edit_file"],
                acceptance_criteria="bug fixed",
                depends_on=["st-1"],
            ),
        ],
    )

    result = validate_plan(tree, tmp_path)

    assert result.verdict == GateVerdict.BLOCK
    assert any("file:line, symbol, and snippet/decision" in msg for msg in result.messages)


def test_plan_gate_accepts_chinese_diagnose_handoff_output(tmp_path: Path) -> None:
    f = tmp_path / "main.py"
    f.write_text("def query_orders():\n    pass\n")
    tree = TaskTree(
        root_task="把当前登机牌查询接口改成用视图查询",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="定位登机牌查询接口",
                allowed_tools=["map_search", "grep_search"],
                acceptance_criteria="输出文件:行号、符号和代码片段/决策",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="将登机牌查询改为使用视图",
                context_files=[],
                allowed_tools=["read_file", "edit_file"],
                acceptance_criteria="查询已改为使用目标视图",
                depends_on=["st-1"],
                needs_l1=True,
            ),
        ],
    )

    result = validate_plan(tree, tmp_path)

    assert result.verdict == GateVerdict.PASS


def test_plan_gate_blocks_too_many_milestones(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="too broad",
        nodes=[
            SubTaskNode(id=f"st-{i}", kind=SubTaskKind.DIAGNOSE, description=f"step {i}")
            for i in range(1, 6)
        ],
    )

    result = validate_plan(tree, tmp_path)

    assert result.verdict == GateVerdict.BLOCK
    assert any("max 4" in msg for msg in result.messages)


def test_plan_gate_blocks_replan_same_structure_with_new_words(tmp_path: Path) -> None:
    f = tmp_path / "app.py"
    f.write_text("x = 1\n")
    ctx = ReplanGateContext(
        failed_subtask_id="st-2",
        failed_kind=SubTaskKind.EDIT,
        failed_description="Fix old wording",
        failed_context_files=("app.py",),
        failed_acceptance_criteria="app.py fixed",
    )
    tree = TaskTree(
        root_task="fix app",
        nodes=[
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.EDIT,
                description="Patch the app bug differently",
                context_files=["app.py"],
                allowed_tools=["read_file", "edit_file"],
                acceptance_criteria="app.py fixed",
            ),
        ],
    )

    result = validate_plan(tree, tmp_path, replan_context=ctx)

    assert result.verdict == GateVerdict.BLOCK
    assert any("repeats failed" in msg for msg in result.messages)
