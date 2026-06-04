from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.agent.explore_guard import ExploreCommandTracker
from src.agent.events import EventType
from src.agent.types import RiskLevel, ToolCall
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.harness.subtask.tool_pipeline import ExecutorToolPipeline, ToolPipelineContext
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode
from src.tools.base import ToolResult


@pytest.mark.asyncio
async def test_duplicate_grep_serves_cache_not_error(tmp_path: Path) -> None:
    memory = ExploreSessionMemory(tracker=ExploreCommandTracker(grep_dedup_limit=1))
    key = "grep:boarding@*"
    memory.put_output(key, "app.py:10: boarding_pass")
    memory.tracker.record_grep("boarding")

    subtask = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    ctx = ToolPipelineContext(
        subtask=subtask,
        root_task="task",
        project_root=tmp_path,
        runtime_tools=frozenset({"grep_search"}),
        preloaded_paths=frozenset(),
        paths_only_mode=True,
        truncated_paths=frozenset(),
        whitelist_files=["app.py"],
        policy=TruncationPolicy.green(),
        memory=memory,
        messages=[],
        error_trace=[],
        tool_failures=[0],
        shell_tracker=MagicMock(),
        pre_edit_snapshots={},
        file_changes=[],
    )

    tools = MagicMock()
    tools.get.return_value = None
    tools.call = AsyncMock(return_value=ToolResult(success=True, output="should not run"))

    pipeline = ExecutorToolPipeline(
        tools,
        MagicMock(),
        approval_waiter=AsyncMock(return_value=True),
        normalize_path=lambda _r, p: str(p),
        snapshot_before_edit=lambda *_a: None,
        collect_diff=lambda *_a: {},
    )

    response = MagicMock()
    response.tool_calls = [
        ToolCall(id="t1", name="grep_search", arguments={"pattern": "boarding", "path": "."}),
    ]

    events = []
    async for ev in pipeline.process_tool_calls(response, ctx):
        events.append(ev)

    tools.call.assert_not_called()
    assert ctx.tool_failures[0] == 0
    assert "[Harness cached explore" in ctx.messages[0].content
    assert pipeline.last_round_stats(ctx).explore_ok is True


@pytest.mark.asyncio
async def test_edit_file_yields_approval_before_waiting(tmp_path: Path) -> None:
    import asyncio

    subtask = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    ctx = ToolPipelineContext(
        subtask=subtask,
        root_task="task",
        project_root=tmp_path,
        runtime_tools=frozenset({"edit_file"}),
        preloaded_paths=frozenset({"app.py"}),
        paths_only_mode=False,
        truncated_paths=frozenset(),
        whitelist_files=["app.py"],
        policy=TruncationPolicy.green(),
        memory=ExploreSessionMemory(tracker=ExploreCommandTracker()),
        messages=[],
        error_trace=[],
        tool_failures=[0],
        shell_tracker=MagicMock(),
        pre_edit_snapshots={},
        file_changes=[],
    )

    edit_tool = MagicMock()
    edit_tool.risk_level = RiskLevel.MODERATE
    tools = MagicMock()
    tools.get.return_value = edit_tool
    tools.call = AsyncMock(return_value=ToolResult(success=True, output="ok"))

    permissions = MagicMock()
    check = MagicMock()
    check.allowed = False
    check.needs_prompt = True
    permissions.check.return_value = check

    approval_futures: dict[str, asyncio.Future[bool]] = {}
    prepare_called = asyncio.Event()

    def prepare_approval(action: str) -> None:
        approval_futures[action] = asyncio.get_running_loop().create_future()
        prepare_called.set()

    async def approval_waiter(action: str) -> bool:
        fut = approval_futures[action]
        return await asyncio.wait_for(fut, timeout=1.0)

    pipeline = ExecutorToolPipeline(
        tools,
        permissions,
        approval_waiter=approval_waiter,
        prepare_approval=prepare_approval,
        normalize_path=lambda _r, p: str(p),
        snapshot_before_edit=lambda *_a: None,
        collect_diff=lambda *_a: {},
    )

    response = MagicMock()
    response.tool_calls = [
        ToolCall(
            id="t1",
            name="edit_file",
            arguments={"path": "app.py", "old_string": "a", "new_string": "b"},
        ),
    ]

    gen = pipeline.process_tool_calls(response, ctx)
    tool_call_ev = await anext(gen)
    assert tool_call_ev.type == EventType.TOOL_CALL

    approval_ev = await anext(gen)
    assert approval_ev.type == EventType.APPROVAL_REQUEST
    await prepare_called.wait()
    assert "edit_file" in approval_futures
    approval_futures["edit_file"].set_result(True)

    async for _ in gen:
        pass

    tools.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_file_resolve_before_wait_matches_repl_flow(tmp_path: Path) -> None:
    """REPL resolves approval while the pipeline generator is still paused at yield."""
    import asyncio

    subtask = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    ctx = ToolPipelineContext(
        subtask=subtask,
        root_task="task",
        project_root=tmp_path,
        runtime_tools=frozenset({"edit_file"}),
        preloaded_paths=frozenset({"app.py"}),
        paths_only_mode=False,
        truncated_paths=frozenset(),
        whitelist_files=["app.py"],
        policy=TruncationPolicy.green(),
        memory=ExploreSessionMemory(tracker=ExploreCommandTracker()),
        messages=[],
        error_trace=[],
        tool_failures=[0],
        shell_tracker=MagicMock(),
        pre_edit_snapshots={},
        file_changes=[],
    )

    from src.config.permissions import PermissionManager

    permissions = PermissionManager()
    edit_tool = MagicMock()
    edit_tool.risk_level = RiskLevel.MODERATE
    tools = MagicMock()
    tools.get.return_value = edit_tool
    tools.call = AsyncMock(return_value=ToolResult(success=True, output="ok"))

    approval_futures: dict[str, asyncio.Future[bool]] = {}

    def prepare_approval(action: str) -> None:
        approval_futures[action] = asyncio.get_running_loop().create_future()

    async def resolve_approval(action: str, approved: bool) -> None:
        permissions.record_decision(action, approved)
        fut = approval_futures.get(action)
        if fut is not None and not fut.done():
            fut.set_result(approved)

    async def approval_waiter(action: str) -> bool:
        decision = permissions.session_decision(action)
        if decision is not None:
            approval_futures.pop(action, None)
            return decision
        fut = approval_futures.get(action)
        if fut is None:
            prepare_approval(action)
            fut = approval_futures[action]
        try:
            return await asyncio.wait_for(fut, timeout=1.0)
        finally:
            approval_futures.pop(action, None)

    pipeline = ExecutorToolPipeline(
        tools,
        permissions,
        approval_waiter=approval_waiter,
        prepare_approval=prepare_approval,
        normalize_path=lambda _r, p: str(p),
        snapshot_before_edit=lambda *_a: None,
        collect_diff=lambda *_a: {},
    )

    response = MagicMock()
    response.tool_calls = [
        ToolCall(
            id="t1",
            name="edit_file",
            arguments={"path": "app.py", "old_string": "a", "new_string": "b"},
        ),
    ]

    gen = pipeline.process_tool_calls(response, ctx)
    await anext(gen)  # tool_call
    approval_ev = await anext(gen)
    assert approval_ev.type == EventType.APPROVAL_REQUEST
    await resolve_approval("edit_file", True)

    async for _ in gen:
        pass

    tools.call.assert_awaited_once()


@pytest.mark.asyncio
async def test_edit_file_session_decision_wins_when_future_unset(tmp_path: Path) -> None:
    """If REPL records approval before the waiter runs, session decision must win."""
    import asyncio

    from src.config.permissions import PermissionManager

    permissions = PermissionManager()
    subtask = SubTaskNode(id="st-2", kind=SubTaskKind.EDIT, description="fix")
    ctx = ToolPipelineContext(
        subtask=subtask,
        root_task="task",
        project_root=tmp_path,
        runtime_tools=frozenset({"edit_file"}),
        preloaded_paths=frozenset({"app.py"}),
        paths_only_mode=False,
        truncated_paths=frozenset(),
        whitelist_files=["app.py"],
        policy=TruncationPolicy.green(),
        memory=ExploreSessionMemory(tracker=ExploreCommandTracker()),
        messages=[],
        error_trace=[],
        tool_failures=[0],
        shell_tracker=MagicMock(),
        pre_edit_snapshots={},
        file_changes=[],
    )

    edit_tool = MagicMock()
    edit_tool.risk_level = RiskLevel.MODERATE
    tools = MagicMock()
    tools.get.return_value = edit_tool
    tools.call = AsyncMock(return_value=ToolResult(success=True, output="ok"))

    approval_futures: dict[str, asyncio.Future[bool]] = {}

    def prepare_approval(action: str) -> None:
        approval_futures[action] = asyncio.get_running_loop().create_future()

    async def approval_waiter(action: str) -> bool:
        decision = permissions.session_decision(action)
        if decision is not None:
            approval_futures.pop(action, None)
            return decision
        fut = approval_futures.get(action)
        if fut is None:
            prepare_approval(action)
            fut = approval_futures[action]
        try:
            return await asyncio.wait_for(fut, timeout=1.0)
        finally:
            approval_futures.pop(action, None)

    pipeline = ExecutorToolPipeline(
        tools,
        permissions,
        approval_waiter=approval_waiter,
        prepare_approval=prepare_approval,
        normalize_path=lambda _r, p: str(p),
        snapshot_before_edit=lambda *_a: None,
        collect_diff=lambda *_a: {},
    )

    response = MagicMock()
    response.tool_calls = [
        ToolCall(
            id="t1",
            name="edit_file",
            arguments={"path": "app.py", "old_string": "a", "new_string": "b"},
        ),
    ]

    gen = pipeline.process_tool_calls(response, ctx)
    await anext(gen)
    await anext(gen)
    permissions.record_decision("edit_file", True)

    async for _ in gen:
        pass

    tools.call.assert_awaited_once()


def test_session_memory_duplicate_read_preload(tmp_path: Path) -> None:
    app = tmp_path / "app.py"
    app.write_text("".join(f"line {i}\n" for i in range(1, 101)))
    policy = TruncationPolicy.green()
    memory = ExploreSessionMemory(
        tracker=ExploreCommandTracker(read_dedup_limit=1),
        policy=policy,
        project_root=tmp_path,
    )
    memory.tracker.record_read("app.py", start_line=10, end_line=20)
    assert memory.is_duplicate_explore(
        "read_file",
        {"path": "app.py", "start_line": 10, "end_line": 20},
    )
    served = memory.serve_read_from_preload("app.py", start_line=10, end_line=20)
    assert served is not None
    assert "line 10" in served
