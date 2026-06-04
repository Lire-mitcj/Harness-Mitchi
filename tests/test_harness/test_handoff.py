from __future__ import annotations

from pathlib import Path

from src.config.settings import MitKIISettings
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.handoff import (
    commit_subtask_failure,
    commit_subtask_success,
    prepare_executor_handoff,
    resolve_turn_tools,
)
from src.harness.subtask.prompt_builder import rebuild_executor_retry_messages
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


def test_prepare_handoff_builds_layered_messages(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="update x",
                context_files=["app.py"],
            ),
        ],
    )
    settings = MitKIISettings()
    bundle = prepare_executor_handoff(
        root_task="fix",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        settings=settings,
        truncation_policy=TruncationPolicy.green(),
    )
    assert len(bundle.initial_messages) >= 3
    system = "\n".join(m.content or "" for m in bundle.initial_messages if m.role == "system")
    assert '<file path="app.py">' in system
    assert bundle.ctx_config is not None
    assert bundle.ctx_runtime is not None


def test_query_style_edit_uses_incremental_read_not_splice(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text(
        "def query_orders():\n"
        "    sql = 'SELECT * FROM orders'\n"
        "    return sql\n",
        encoding="utf-8",
    )
    tree = TaskTree(
        root_task="将登机牌查询接口改为使用视图",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="定位登机牌查询接口",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="将登机牌查询接口改为使用视图",
                allowed_tools=["context_search", "edit_file"],
                context_files=["main.py"],
                depends_on=["st-1"],
            ),
        ],
    )

    bundle = prepare_executor_handoff(
        root_task=tree.root_task,
        task_tree=tree,
        subtask=tree.nodes[1],
        project_root=tmp_path,
        settings=MitKIISettings(),
        truncation_policy=TruncationPolicy.green(),
        prior_summaries={
            "st-1": (
                '接口端点: main.py:1 — @app.get("/api/orders/query")\n'
                "处理函数: main.py:1 — def query_orders(...)"
            ),
        },
    )

    assert bundle.edit_handoff is None
    assert bundle.ctx_runtime is not None
    assert bundle.ctx_runtime.edit_read_fallback is False
    assert bundle.ctx_runtime.active_runtime_tools == frozenset({"edit_file"})
    assert resolve_turn_tools(
        bundle,
        turns_used=1,
        tool_rounds=0,
        file_changes=[],
        active_runtime_tools=bundle.ctx_runtime.active_runtime_tools,
    ) == frozenset({"edit_file"})
    assert resolve_turn_tools(
        bundle,
        turns_used=2,
        tool_rounds=1,
        file_changes=[],
        active_runtime_tools=bundle.ctx_runtime.active_runtime_tools,
    ) == frozenset({"edit_file"})


def test_diagnose_turn_tools_prefer_batched_map_and_grep(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="locate boarding pass query",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="locate target",
                allowed_tools=["context_search", "git_status"],
            ),
        ],
    )
    bundle = prepare_executor_handoff(
        root_task=tree.root_task,
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        settings=MitKIISettings(),
        truncation_policy=TruncationPolicy.green(),
    )
    active = frozenset({"context_search", "git_status"})

    assert resolve_turn_tools(
        bundle,
        turns_used=1,
        tool_rounds=0,
        file_changes=[],
        active_runtime_tools=active,
    ) == frozenset({"context_search"})
    assert resolve_turn_tools(
        bundle,
        turns_used=3,
        tool_rounds=2,
        file_changes=[],
        active_runtime_tools=active,
    ) == frozenset()


def test_commit_success_propagates_diagnose_paths(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="find",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="edit",
                depends_on=["st-1"],
                context_files=[],
            ),
        ],
    )
    summaries: dict[str, str] = {}
    attempts: dict[str, int] = {"st-1": 2}
    digests: dict[str, str] = {"st-1": "grep hits"}
    result = commit_subtask_success(
        task_tree=tree,
        node=tree.nodes[0],
        exec_result={
            "final_message": "Target at app.py:10-20 in make_boarding_pass",
        },
        project_root=tmp_path,
        subtask_summaries=summaries,
        subtask_attempts=attempts,
        subtask_exploration_digests=digests,
    )
    assert result.summary_stored
    assert result.context_files_updated
    assert "st-1" in summaries
    assert "st-1" not in attempts
    assert "st-1" not in digests
    edit = tree.get("st-2")
    assert edit is not None
    assert "app.py" in edit.context_files


def test_commit_success_stores_exploration_digest_for_downstream(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.DIAGNOSE, description="find"),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="edit",
                depends_on=["st-1"],
            ),
        ],
    )
    summaries: dict[str, str] = {}

    commit_subtask_success(
        task_tree=tree,
        node=tree.nodes[0],
        exec_result={
            "final_message": "Target is app.py:1.",
            "exploration_digest": "Grep hits (sample):\n  - app.py:1:x = 1",
        },
        project_root=tmp_path,
        subtask_summaries=summaries,
        subtask_attempts={},
        subtask_exploration_digests={},
    )

    assert "Executor evidence digest" in summaries["st-1"]
    assert "app.py:1:x = 1" in summaries["st-1"]


def test_prepare_handoff_coordinate_handoff_restricts_verify_explore(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("".join(f"line_{i} = {i}\n" for i in range(30)))
    tree = TaskTree(
        root_task="verify focused target",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="locate target",
                context_files=["app.py"],
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.VERIFY,
                description="run focused verification",
                context_files=["app.py"],
                depends_on=["st-1"],
                allowed_tools=[
                    "shell_exec",
                    "context_search",
                ],
            ),
        ],
    )

    bundle = prepare_executor_handoff(
        root_task=tree.root_task,
        task_tree=tree,
        subtask=tree.nodes[1],
        project_root=tmp_path,
        settings=MitKIISettings(),
        truncation_policy=TruncationPolicy.green(),
        prior_summaries={
            "st-1": "Target at app.py:10 symbol handler with snippet/decision.",
        },
    )

    assert bundle.ctx_runtime is not None
    assert bundle.ctx_runtime.explore_restricted is True
    assert "shell_exec" in bundle.ctx_runtime.active_runtime_tools
    assert "context_search" not in bundle.ctx_runtime.active_runtime_tools
    assert "read_file" not in bundle.ctx_runtime.active_runtime_tools
    assert "grep_search" not in bundle.ctx_runtime.active_runtime_tools
    assert "map_search" not in bundle.ctx_runtime.active_runtime_tools
    system_text = "\n".join(
        m.content or "" for m in bundle.initial_messages if m.role == "system"
    )
    assert '<file path="app.py">' in system_text


def test_prepare_handoff_coordinate_handoff_restricts_diagnose_explore(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("".join(f"line_{i} = {i}\n" for i in range(30)))
    tree = TaskTree(
        root_task="check target",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="locate target",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.DIAGNOSE,
                description="check target usage",
                depends_on=["st-1"],
                allowed_tools=["context_search"],
            ),
        ],
    )

    bundle = prepare_executor_handoff(
        root_task=tree.root_task,
        task_tree=tree,
        subtask=tree.nodes[1],
        project_root=tmp_path,
        settings=MitKIISettings(),
        truncation_policy=TruncationPolicy.green(),
        prior_summaries={
            "st-1": "Target at app.py:10 symbol handler with snippet/decision.",
        },
    )

    assert bundle.ctx_runtime is not None
    assert bundle.ctx_runtime.explore_restricted is True
    assert bundle.ctx_runtime.active_runtime_tools == frozenset()
    assert "app.py" in bundle.whitelist_files
    system_text = "\n".join(
        m.content or "" for m in bundle.initial_messages if m.role == "system"
    )
    assert '<file path="app.py">' in system_text


def test_diagnose_context_search_is_single_round(tmp_path: Path) -> None:
    settings = MitKIISettings()
    tree = TaskTree(
        root_task="find",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="find target",
                allowed_tools=["context_search"],
            ),
        ],
    )
    bundle = prepare_executor_handoff(
        root_task=tree.root_task,
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        settings=settings,
        truncation_policy=TruncationPolicy.green(),
    )

    active = frozenset({"context_search"})

    assert resolve_turn_tools(
        bundle,
        turns_used=1,
        tool_rounds=0,
        file_changes=[],
        active_runtime_tools=active,
    ) == active
    assert resolve_turn_tools(
        bundle,
        turns_used=2,
        tool_rounds=1,
        file_changes=[],
        active_runtime_tools=active,
    ) == frozenset()
    assert resolve_turn_tools(
        bundle,
        turns_used=3,
        tool_rounds=2,
        file_changes=[],
        active_runtime_tools=active,
    ) == frozenset()


def test_commit_failure_preserves_digest() -> None:
    attempts: dict[str, int] = {}
    digests: dict[str, str] = {}
    node = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="x")
    attempt = commit_subtask_failure(
        node=node,
        exec_result={"exploration_digest": "Grep queries already run:\n  - 'x' in app.py\n"},
        subtask_attempts=attempts,
        subtask_exploration_digests=digests,
    )
    assert attempt == 1
    assert digests["st-2"].startswith("Grep")


def test_retry_messages_force_different_attempt(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("x = 1\n")
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="edit",
                context_files=["app.py"],
            ),
        ],
    )

    messages = rebuild_executor_retry_messages(
        root_task="fix",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        prior_summaries=None,
        error_trace=["edit_file: old_string not found in app.py"],
        context_files=["app.py"],
        runtime_tools=frozenset({"edit_file", "read_files"}),
        exploration_digest="Files already read: app.py",
    )
    text = "\n".join(m.content or "" for m in messages)

    assert '"failure_pattern": "edit_not_found"' in text
    assert "Do not repeat the same tool arguments" in text
    assert "Files already read: app.py" in text
