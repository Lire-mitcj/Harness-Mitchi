from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from src.agent.events import AgentEvent, EventType
from src.config.settings import MitKIISettings
from src.harness.gates.phase_metrics import PhaseMetrics
from src.orchestrator.orchestrator import OrchestratorLoop
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree


@pytest.mark.asyncio
async def test_parallel_diagnose_batch_commits_successes(tmp_path, monkeypatch) -> None:
    tree = TaskTree(
        root_task="find targets",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.DIAGNOSE,
                description="find api",
                acceptance_criteria="api evidence",
            ),
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.DIAGNOSE,
                description="find schema",
                acceptance_criteria="schema evidence",
            ),
        ],
    )
    loop = object.__new__(OrchestratorLoop)
    loop.harness = SimpleNamespace(
        project_root=tmp_path,
        phase_metrics=PhaseMetrics(),
        save_checkpoint=AsyncMock(side_effect=["cp-st-1", "cp-st-2"]),
    )
    loop.settings = MitKIISettings(data_dir=tmp_path / ".mitkii")
    loop.state = SimpleNamespace(
        subtask_summaries={},
        subtask_attempts={},
        subtask_exploration_digests={},
    )
    loop._sync_repo_map_after_exec = lambda node, result: None

    async def no_context_pack(*args, **kwargs):
        return None

    async def fake_diagnose(*, task_tree, node, context_pack) -> AsyncIterator[AgentEvent]:
        final = f"Result for {node.id}"
        yield AgentEvent(type=EventType.FINAL_ANSWER, content=final)
        yield AgentEvent(
            type=EventType.STREAM_END,
            data={
                "subtask_id": node.id,
                "success": True,
                "turns_used": 0,
                "changed_files": [],
                "file_diffs": {},
                "error_trace": [],
                "final_message": final,
                "failure_code": "",
                "quality_gate_failures": 0,
                "exploration_digest": f"digest {node.id}",
            },
        )

    monkeypatch.setattr(loop, "_context_pack_for_subtask", no_context_pack)
    monkeypatch.setattr(loop, "_run_diagnose_skill_executor", fake_diagnose)
    monkeypatch.setattr(
        loop,
        "_save_subtask_checkpoint",
        AsyncMock(side_effect=["cp-st-1", "cp-st-2"]),
    )

    handled, events = await loop._run_parallel_diagnose_batch(
        task_tree=tree,
        nodes=tree.nodes,
        repo_map=None,
    )

    assert handled is True
    assert [node.status for node in tree.nodes] == [
        SubTaskStatus.SUCCESS,
        SubTaskStatus.SUCCESS,
    ]
    assert [node.checkpoint_id for node in tree.nodes] == ["cp-st-1", "cp-st-2"]
    assert set(loop.state.subtask_summaries) == {"st-1", "st-2"}
    assert any(
        event.data and event.data.get("phase") == "dag_parallel"
        for event in events
    )


@pytest.mark.asyncio
async def test_parallel_diagnose_batch_falls_back_on_failure(tmp_path, monkeypatch) -> None:
    tree = TaskTree(
        root_task="find targets",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.DIAGNOSE, description="find api"),
            SubTaskNode(id="st-2", kind=SubTaskKind.DIAGNOSE, description="find schema"),
        ],
    )
    loop = object.__new__(OrchestratorLoop)
    loop.harness = SimpleNamespace(
        project_root=tmp_path,
        phase_metrics=PhaseMetrics(),
        save_checkpoint=AsyncMock(return_value="cp"),
    )
    loop.settings = MitKIISettings(data_dir=tmp_path / ".mitkii")
    loop.state = SimpleNamespace(
        subtask_summaries={},
        subtask_attempts={},
        subtask_exploration_digests={},
    )
    loop._sync_repo_map_after_exec = lambda node, result: None

    async def no_context_pack(*args, **kwargs):
        return None

    async def fake_diagnose(*, task_tree, node, context_pack) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STREAM_END,
            data={
                "subtask_id": node.id,
                "success": node.id == "st-1",
                "turns_used": 0,
                "error_trace": [] if node.id == "st-1" else ["failed"],
                "final_message": f"Result for {node.id}",
                "failure_code": "" if node.id == "st-1" else "skill_failed",
            },
        )

    monkeypatch.setattr(loop, "_context_pack_for_subtask", no_context_pack)
    monkeypatch.setattr(loop, "_run_diagnose_skill_executor", fake_diagnose)
    monkeypatch.setattr(loop, "_save_subtask_checkpoint", AsyncMock(return_value="cp"))

    handled, events = await loop._run_parallel_diagnose_batch(
        task_tree=tree,
        nodes=tree.nodes,
        repo_map=None,
    )

    assert handled is False
    assert [node.status for node in tree.nodes] == [
        SubTaskStatus.PENDING,
        SubTaskStatus.PENDING,
    ]
    assert loop.state.subtask_summaries == {}
    assert any("falling back to serial" in (event.content or "") for event in events)


@pytest.mark.asyncio
async def test_parallel_verify_batch_commits_successes(tmp_path, monkeypatch) -> None:
    tree = TaskTree(
        root_task="verify targets",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.VERIFY, description="verify api"),
            SubTaskNode(id="st-2", kind=SubTaskKind.VERIFY, description="verify schema"),
        ],
    )
    loop = object.__new__(OrchestratorLoop)
    loop.harness = SimpleNamespace(
        project_root=tmp_path,
        phase_metrics=PhaseMetrics(),
        save_checkpoint=AsyncMock(return_value="unused"),
    )
    loop.settings = MitKIISettings(data_dir=tmp_path / ".mitkii")
    loop.state = SimpleNamespace(
        subtask_summaries={},
        subtask_attempts={},
        subtask_exploration_digests={},
    )
    loop._sync_repo_map_after_exec = lambda node, result: None

    async def no_context_pack(*args, **kwargs):
        return None

    async def fake_verify(*, user_msg, task_tree, node, context_pack) -> AsyncIterator[AgentEvent]:
        final = f"Verified {node.id}"
        yield AgentEvent(type=EventType.FINAL_ANSWER, content=final)
        yield AgentEvent(
            type=EventType.STREAM_END,
            data={
                "subtask_id": node.id,
                "success": True,
                "turns_used": 0,
                "changed_files": [],
                "file_diffs": {},
                "error_trace": [],
                "final_message": final,
                "failure_code": "",
                "quality_gate_failures": 0,
            },
        )

    monkeypatch.setattr(loop, "_context_pack_for_subtask", no_context_pack)
    monkeypatch.setattr(loop, "_run_verify_skill_executor", fake_verify)
    monkeypatch.setattr(
        loop,
        "_save_subtask_checkpoint",
        AsyncMock(side_effect=["cp-st-1", "cp-st-2"]),
    )

    handled, events = await loop._run_parallel_verify_batch(
        user_msg="verify",
        task_tree=tree,
        nodes=tree.nodes,
        repo_map=None,
    )

    assert handled is True
    assert [node.status for node in tree.nodes] == [
        SubTaskStatus.SUCCESS,
        SubTaskStatus.SUCCESS,
    ]
    assert [node.checkpoint_id for node in tree.nodes] == ["cp-st-1", "cp-st-2"]
    assert set(loop.state.subtask_summaries) == {"st-1", "st-2"}
    assert any(
        event.data and event.data.get("kind") == "verify"
        for event in events
        if event.data
    )


@pytest.mark.asyncio
async def test_parallel_verify_batch_falls_back_on_failure(tmp_path, monkeypatch) -> None:
    tree = TaskTree(
        root_task="verify targets",
        nodes=[
            SubTaskNode(id="st-1", kind=SubTaskKind.VERIFY, description="verify api"),
            SubTaskNode(id="st-2", kind=SubTaskKind.VERIFY, description="verify schema"),
        ],
    )
    loop = object.__new__(OrchestratorLoop)
    loop.harness = SimpleNamespace(
        project_root=tmp_path,
        phase_metrics=PhaseMetrics(),
        save_checkpoint=AsyncMock(return_value="unused"),
    )
    loop.settings = MitKIISettings(data_dir=tmp_path / ".mitkii")
    loop.state = SimpleNamespace(
        subtask_summaries={},
        subtask_attempts={},
        subtask_exploration_digests={},
    )
    loop._sync_repo_map_after_exec = lambda node, result: None

    async def no_context_pack(*args, **kwargs):
        return None

    async def fake_verify(*, user_msg, task_tree, node, context_pack) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(
            type=EventType.STREAM_END,
            data={
                "subtask_id": node.id,
                "success": node.id == "st-1",
                "turns_used": 0,
                "error_trace": [] if node.id == "st-1" else ["failed"],
                "final_message": f"Verified {node.id}",
                "failure_code": "" if node.id == "st-1" else "verify_failed",
            },
        )

    monkeypatch.setattr(loop, "_context_pack_for_subtask", no_context_pack)
    monkeypatch.setattr(loop, "_run_verify_skill_executor", fake_verify)
    monkeypatch.setattr(loop, "_save_subtask_checkpoint", AsyncMock(return_value="cp"))

    handled, events = await loop._run_parallel_verify_batch(
        user_msg="verify",
        task_tree=tree,
        nodes=tree.nodes,
        repo_map=None,
    )

    assert handled is False
    assert [node.status for node in tree.nodes] == [
        SubTaskStatus.PENDING,
        SubTaskStatus.PENDING,
    ]
    assert loop.state.subtask_summaries == {}
    assert any("parallel verify failed" in (event.content or "") for event in events)
