from __future__ import annotations

from pathlib import Path

from src.agent.types import assistant_message, tool_message, user_message
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.context_pipeline import (
    ExecutorContextConfig,
    ExecutorContextSession,
    ExecutorRuntimeState,
)
from src.harness.subtask.prompt_builder import estimate_messages_tokens
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


def test_incremental_token_count_matches_full(tmp_path: Path) -> None:
    prefix = [user_message("bootstrap " * 100)]
    tail = [assistant_message("ok"), tool_message("t1", "grep hit " * 200)]
    full = prefix + tail
    prefix_tokens = estimate_messages_tokens(prefix)
    incremental = estimate_messages_tokens(
        full, prefix_len=len(prefix), prefix_tokens=prefix_tokens
    )
    assert incremental == estimate_messages_tokens(full)


def test_context_session_base_messages_cached(tmp_path: Path) -> None:
    tree = TaskTree(
        root_task="root",
        nodes=[SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")],
    )
    config = ExecutorContextConfig(
        root_task="root",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        policy=TruncationPolicy.green(),
        prior_summaries=None,
        whitelist_files=[],
        whitelist_norm=frozenset(),
        diag_handoff=False,
        compact_token_threshold=999_999,
    )
    runtime = ExecutorRuntimeState(
        paths_only_mode=True,
        use_paths_only=True,
        preloaded_paths=frozenset(),
        truncated_paths=frozenset(),
        active_runtime_tools=frozenset({"edit_file"}),
        explore_restricted=False,
    )
    session = ExecutorContextSession(
        config=config, runtime=runtime, memory=ExploreSessionMemory.create()
    )
    first = session._base_messages()
    second = session._base_messages()
    assert first is second


def test_compact_preserves_pinned_s1_s2(tmp_path: Path) -> None:
    from src.agent.types import system_message

    tree = TaskTree(
        root_task="root",
        nodes=[SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")],
    )
    config = ExecutorContextConfig(
        root_task="root",
        task_tree=tree,
        subtask=tree.nodes[0],
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
        active_runtime_tools=frozenset({"edit_file", "grep_search"}),
        explore_restricted=False,
    )
    session = ExecutorContextSession(
        config=config, runtime=runtime, memory=ExploreSessionMemory.create()
    )
    s1 = system_message("PINNED-S1-ROLE")
    s2 = system_message("PINNED-S2-TASK-TREE")
    s3 = system_message("PINNED-S3-SCOPE-WITH-FILES")
    bootstrap = user_message("Execute subtask now.")
    handoff = [s1, s2, s3, bootstrap]
    session.seed_prefix(handoff)

    fat_tail = [
        assistant_message("", tool_calls=[]),
        tool_message("t1", "grep hit " * 800),
    ]
    messages = handoff + fat_tail
    result = session.compact_before_turn(messages, [])

    assert result.changed
    assert result.messages[0] is s1
    assert result.messages[1] is s2
    assert result.messages[0].content == "PINNED-S1-ROLE"
    assert result.messages[1].content == "PINNED-S2-TASK-TREE"
    assert result.messages[3] is bootstrap
    assert result.messages[2] is not s3
    assert "PINNED-S3-SCOPE-WITH-FILES" not in (result.messages[2].content or "")
    assert len(result.messages) < len(messages)
    combined = "\n".join(m.content or "" for m in result.messages)
    assert "Context folded" in combined
