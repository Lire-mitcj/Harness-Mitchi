from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from src.agent.types import user_message
from src.config.settings import MitKIISettings
from src.harness.engine import HarnessEngine
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.context_pipeline import (
    ExecutorContextConfig,
    ExecutorContextSession,
    ExecutorRuntimeState,
)
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


@pytest.mark.asyncio
async def test_before_executor_llm_call_runs_digest_then_probe(tmp_path: Path) -> None:
    settings = MitKIISettings(max_context_tokens=8000, context_budget_ratio=0.75)
    harness = HarnessEngine.create(settings, project_root=tmp_path)
    tree = TaskTree(
        root_task="root",
        nodes=[SubTaskNode(id="st-1", kind=SubTaskKind.EDIT, description="x")],
    )
    memory = ExploreSessionMemory.create()
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
        active_runtime_tools=frozenset({"edit_file"}),
        explore_restricted=False,
    )
    session = ExecutorContextSession(config=config, runtime=runtime, memory=memory)
    messages = [user_message("x" * 5000)]

    trimmed, cp = await harness.before_executor_llm_call(session, messages, [])

    assert cp.changed
    assert any(ev.kind == "compact" for ev in cp.events)
    assert isinstance(trimmed, list)
    assert trimmed


@pytest.mark.asyncio
async def test_before_executor_llm_call_probe_trim_when_still_over_budget(
    tmp_path: Path,
) -> None:
    settings = MitKIISettings(max_context_tokens=400, context_budget_ratio=0.75)
    harness = HarnessEngine.create(settings, project_root=tmp_path)
    harness.probe.before_call = AsyncMock(
        side_effect=lambda msgs: msgs[:2] if len(msgs) > 2 else msgs
    )
    tree = TaskTree(
        root_task="root",
        nodes=[SubTaskNode(id="st-1", kind=SubTaskKind.EDIT, description="x")],
    )
    memory = ExploreSessionMemory.create()
    config = ExecutorContextConfig(
        root_task="root",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        policy=TruncationPolicy.green(),
        prior_summaries=None,
        whitelist_files=[],
        whitelist_norm=frozenset(),
        diag_handoff=True,
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
    messages = [user_message("y" * 2000)]

    trimmed, cp = await harness.before_executor_llm_call(session, messages, [])

    assert cp.changed
    assert any(ev.kind == "compact" for ev in cp.events)
    harness.probe.before_call.assert_awaited_once()
    assert len(trimmed) <= 2
