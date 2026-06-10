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
from src.harness.task_analysis import HarnessTaskAnalysis, analyze_task
from src.planner.task_tree import SubTaskKind, SubTaskNode, SubTaskStatus, TaskTree


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
                allowed_tools=["context_search"],
                acceptance_criteria="Output file:line, symbol, and route snippet/decision",
            ),
            SubTaskNode(
                id="st-2",
                description="edit api",
                kind=SubTaskKind.EDIT,
                context_files=["api.py"],
                allowed_tools=["context_search", "edit_file"],
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


def test_plan_gate_blocks_edit_first_for_function_refactor(tmp_path: Path) -> None:
    f = tmp_path / "main.py"
    f.write_text("x = 1\n")
    tree = TaskTree(
        root_task="重构订单详情处理，复用 normalize_order_record 做脱敏",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="重构订单详情处理",
                context_files=["main.py"],
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="订单详情查询和脱敏逻辑统一",
            ),
        ],
    )
    result = validate_plan(tree, tmp_path)
    assert result.verdict == GateVerdict.BLOCK
    assert "must start with a diagnose" in result.messages[0]


def test_plan_gate_requires_design_for_high_complexity_function_refactor(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="重构订单详情处理，统一脱敏",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="定位订单详情调用点",
                allowed_tools=["context_search"],
                acceptance_criteria="Output file:line, symbol, and snippet/decision",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="重构订单详情处理",
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="订单详情查询和脱敏逻辑统一",
                depends_on=["st-1"],
            ),
        ],
    )
    result = validate_plan(tree, tmp_path, task_analysis=analyze_task(tree.root_task))
    assert result.verdict == GateVerdict.BLOCK
    assert "requires design step" in "; ".join(result.messages)


def test_plan_gate_blocks_edit_first_when_harness_not_ready(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="重构订单详情处理，统一脱敏",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="重构订单详情处理",
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="订单详情查询和脱敏逻辑统一",
            )
        ],
    )
    result = validate_plan(tree, tmp_path, task_analysis=analyze_task(tree.root_task))
    assert result.verdict == GateVerdict.BLOCK
    assert "edit_ready=false" in "; ".join(result.messages)


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
                allowed_tools=["context_search"],
            ),
            SubTaskNode(
                id="st-4",
                kind=SubTaskKind.EDIT,
                description="Fix boarding pass SELECT",
                context_files=["app.py"],
                allowed_tools=["context_search", "edit_file"],
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
                allowed_tools=["context_search"],
                acceptance_criteria="Output file:line, symbol, and SQL snippet/decision",
            ),
            SubTaskNode(
                id="st-4",
                kind=SubTaskKind.EDIT,
                description="Update boarding pass view SQL",
                context_files=["routes.sql"],
                allowed_tools=["context_search", "edit_file"],
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
                allowed_tools=["context_search"],
                acceptance_criteria="app.py analyzed",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="Fix app.py",
                context_files=["app.py"],
                allowed_tools=["context_search", "edit_file"],
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
                allowed_tools=["context_search"],
                acceptance_criteria="candidate files listed",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="Fix app bug",
                context_files=["app.py"],
                allowed_tools=["context_search", "edit_file"],
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
                allowed_tools=["context_search"],
                acceptance_criteria="输出文件:行号、符号和代码片段/决策",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="将登机牌查询改为使用视图",
                context_files=[],
                allowed_tools=["context_search", "edit_file"],
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
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="app.py fixed",
            ),
        ],
    )

    result = validate_plan(tree, tmp_path, replan_context=ctx)

    assert result.verdict == GateVerdict.BLOCK
    assert any("repeats failed" in msg for msg in result.messages)


def test_plan_gate_blocks_multistep_root_task_copy_descriptions(tmp_path: Path) -> None:
    root = "登机牌的查询接口是不是没用到视图，你改成使用视图查询"
    tree = TaskTree(
        root_task=root,
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description=f"定位目标代码：{root}",
                allowed_tools=["context_search"],
                acceptance_criteria="输出文件:行号、符号和代码片段/决策",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description=root,
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="查询改为使用目标视图",
                depends_on=["st-1"],
                needs_l1=True,
            ),
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.VERIFY,
                description="验证相关行为",
                allowed_tools=["shell_exec"],
                acceptance_criteria="Relevant verification command exits 0",
                depends_on=["st-2"],
            ),
        ],
    )

    result = validate_plan(tree, tmp_path)

    assert result.verdict == GateVerdict.BLOCK
    assert any("description copies root_task" in msg for msg in result.messages)


def test_plan_gate_blocks_edit_replan_that_only_diagnoses(tmp_path: Path) -> None:
    ctx = ReplanGateContext(
        failed_subtask_id="st-1",
        failed_kind=SubTaskKind.EDIT,
        failed_description="将登机牌查询接口改为使用视图",
        failed_context_files=("app.py",),
    )
    tree = TaskTree(
        root_task="把登机牌查询的接口改成使用视图查询",
        nodes=[
            SubTaskNode(
                id="st-2a",
                kind=SubTaskKind.DIAGNOSE,
                description="定位登机牌查询接口和当前 SQL",
                allowed_tools=["context_search"],
                acceptance_criteria="输出文件:行号，符号，和 SQL 查询片段/决策",
            )
        ],
    )

    result = validate_plan(tree, tmp_path, replan_context=ctx)
    assert result.verdict == GateVerdict.BLOCK
    assert any("removed the edit objective" in msg for msg in result.messages)


def test_plan_gate_allows_replan_with_completed_steps(tmp_path: Path) -> None:
    from src.harness.task_analysis import HarnessTaskAnalysis
    from src.planner.task_tree import SubTaskStatus
    
    analysis = HarnessTaskAnalysis(
        intent="sql_view_rewrite",
        confidence=1.0,
        edit_ready=False,
        edit_strategy="sql_view_rewrite",
        complexity="high",
        resolved_dependencies=[{"kind": "database_view", "name": "v"}]
    )
    
    tree = TaskTree(
        root_task="把当前登机牌查询接口改成用视图查询",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                status=SubTaskStatus.SUCCESS,
                description="定位登机牌查询接口和当前 SQL",
                allowed_tools=["context_search"],
                acceptance_criteria="输出文件:行号，符号，和 SQL 查询片段/决策",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.DESIGN,
                status=SubTaskStatus.SUCCESS,
                description="设计视图替换补丁意图",
                allowed_tools=["context_search"],
                acceptance_criteria="输出 PATCH_INTENT_JSON",
                depends_on=["st-1"],
            ),
            SubTaskNode(
                id="st-3a",
                kind=SubTaskKind.DIAGNOSE,
                status=SubTaskStatus.PENDING,
                description="根据最新的失败证据，重新诊断并寻找代码变更点",
                allowed_tools=["context_search"],
                acceptance_criteria="输出文件:行号，符号，和 SQL 查询片段/决策",
                depends_on=["st-2"],
            ),
            SubTaskNode(
                id="st-3b",
                kind=SubTaskKind.DESIGN,
                status=SubTaskStatus.PENDING,
                description="在新的诊断基础上重构视图补丁意图设计方案",
                allowed_tools=["context_search"],
                acceptance_criteria="输出 PATCH_INTENT_JSON",
                depends_on=["st-3a"],
            ),
            SubTaskNode(
                id="st-3c",
                kind=SubTaskKind.EDIT,
                status=SubTaskStatus.PENDING,
                description="将登机牌查询改为使用视图",
                allowed_tools=["context_search", "edit_file"],
                acceptance_criteria="查询已改为使用目标视图",
                depends_on=["st-3b"],
            ),
            SubTaskNode(
                id="st-4",
                kind=SubTaskKind.VERIFY,
                status=SubTaskStatus.PENDING,
                description="验证登机牌查询接口",
                allowed_tools=["shell_exec"],
                acceptance_criteria="Relevant verification command exits 0",
                depends_on=["st-3c"],
            ),
        ],
    )
    
    result = validate_plan(tree, tmp_path, max_nodes=10, task_analysis=analysis)
    assert result.verdict in {GateVerdict.PASS, GateVerdict.WARN}


def test_plan_gate_skips_successful_nodes(tmp_path: Path) -> None:
    # A tree that would normally fail validation (e.g. edit step has no context files, or duplicates description)
    # but the failing node is marked SUCCESS, so it should not block validation.
    tree = TaskTree(
        root_task="add health",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                status=SubTaskStatus.SUCCESS, # SUCCESS!
                description="", # empty description!
                context_files=[], # empty context files!
                allowed_tools=[], # invalid tools!
                acceptance_criteria="", # empty acceptance!
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.VERIFY,
                status=SubTaskStatus.PENDING,
                description="Run validation",
                allowed_tools=["shell_exec"],
                acceptance_criteria="verification command exits 0",
            ),
        ],
    )
    result = validate_plan(tree, tmp_path)
    # st-1 is skipped, st-2 is valid, so this should pass/warn (not block)
    assert result.verdict in {GateVerdict.PASS, GateVerdict.WARN}


def test_plan_gate_blocks_edit_requiring_patch_intent_without_design_dependency(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="change target",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="定位目标",
                acceptance_criteria="输出 file:line、symbol、片段/决策",
                allowed_tools=["context_search"],
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.DESIGN,
                description="设计补丁意图",
                acceptance_criteria="输出 PATCH_INTENT_JSON",
                allowed_tools=["context_search"],
                depends_on=["st-1"],
                handoff_outputs=["PATCH_INTENT_JSON"],
            ),
            SubTaskNode(
                id="st-3",
                kind=SubTaskKind.EDIT,
                description="修改目标",
                acceptance_criteria="patched",
                allowed_tools=["context_search", "edit_file"],
                context_files=["main.py"],
                depends_on=["st-1"],
                requires_handoff=["PATCH_INTENT_JSON"],
            ),
        ],
    )
    (tmp_path / "main.py").write_text("def f(): pass\n", encoding="utf-8")

    result = validate_plan(tree, tmp_path, max_nodes=5)
    assert result.verdict == GateVerdict.BLOCK
    assert "requires PATCH_INTENT_JSON" in "; ".join(result.messages)


def test_plan_gate_allows_split_edits_for_multiple_harness_targets(tmp_path: Path) -> None:
    for name in ("a.py", "b.py"):
        (tmp_path / name).write_text("def f(): pass\n", encoding="utf-8")
    analysis = HarnessTaskAnalysis(
        intent="function_refactor",
        confidence=0.91,
        complexity="high",
        edit_ready=False,
        edit_strategy="function_refactor",
        readiness_checks={},
        editable_targets=(
            {"file": "a.py", "symbol": "a"},
            {"file": "b.py", "symbol": "b"},
        ),
    )
    tree = TaskTree(
        root_task="重构多个调用点",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="定位调用点",
                acceptance_criteria="输出 file:line、symbol、片段/决策",
                allowed_tools=["context_search"],
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.DESIGN,
                description="设计补丁意图",
                acceptance_criteria="输出 PATCH_INTENT_JSON",
                allowed_tools=["context_search"],
                depends_on=["st-1"],
                handoff_outputs=["PATCH_INTENT_JSON"],
            ),
            SubTaskNode(
                id="st-3a",
                kind=SubTaskKind.EDIT,
                description="修改 a",
                acceptance_criteria="patched a",
                allowed_tools=["context_search", "edit_file"],
                context_files=["a.py"],
                depends_on=["st-2"],
                requires_handoff=["PATCH_INTENT_JSON"],
            ),
            SubTaskNode(
                id="st-3b",
                kind=SubTaskKind.EDIT,
                description="修改 b",
                acceptance_criteria="patched b",
                allowed_tools=["context_search", "edit_file"],
                context_files=["b.py"],
                depends_on=["st-3a"],
                requires_handoff=["PATCH_INTENT_JSON"],
            ),
            SubTaskNode(
                id="st-4",
                kind=SubTaskKind.VERIFY,
                description="验证重构",
                acceptance_criteria="tests pass",
                allowed_tools=["shell_exec"],
                depends_on=["st-3b"],
            ),
        ],
    )

    result = validate_plan(tree, tmp_path, max_nodes=4, task_analysis=analysis)
    assert result.verdict in {GateVerdict.PASS, GateVerdict.WARN}
