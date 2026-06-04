from __future__ import annotations

from pathlib import Path

from src.agent.types import user_message
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.context_pipeline import (
    ExecutorContextConfig,
    ExecutorContextSession,
    ExecutorRuntimeState,
)
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


def test_context_session_compact_before_turn(tmp_path: Path) -> None:
    tree = TaskTree(root_task="root", nodes=[SubTaskNode(id="st-1", kind=SubTaskKind.EDIT, description="x")])
    subtask = tree.nodes[0]
    memory = ExploreSessionMemory.create()
    config = ExecutorContextConfig(
        root_task="root",
        task_tree=tree,
        subtask=subtask,
        project_root=tmp_path,
        policy=TruncationPolicy.green(),
        prior_summaries=None,
        whitelist_files=[],
        whitelist_norm=frozenset(),
        diag_handoff=False,
        compact_token_threshold=10,
    )
    runtime = ExecutorRuntimeState(
        paths_only_mode=False,
        use_paths_only=False,
        preloaded_paths=frozenset(),
        truncated_paths=frozenset(),
        active_runtime_tools=frozenset({"edit_file"}),
        explore_restricted=False,
    )
    session = ExecutorContextSession(config=config, runtime=runtime, memory=memory)
    messages = [user_message("x" * 5000)]
    assert session.should_compact(session.estimate_tokens(messages))
    result = session.compact_before_turn(messages, [])
    assert result.changed
    assert any(ev.kind == "compact" for ev in result.events)
