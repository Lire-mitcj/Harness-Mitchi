from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from src.agent.events import (
    AgentEvent,
    EventType,
    approval_event,
    file_edit_event,
    tool_call_event,
    tool_result_event,
)
from src.agent.explore_guard import parse_read_path_with_lines
from src.agent.framework_guard import (
    blocked_framework_browse,
    blocked_framework_reads,
    format_framework_browse_denial,
    format_framework_read_denial,
    normalize_rel_path,
)
from src.agent.shell_guard import ShellCommandTracker
from src.agent.types import (
    LLMResponse,
    Message,
    ToolCall,
    tool_message,
)
from src.config.permissions import PermissionManager
from src.executor.edit_guard import is_edit_recoverable_error
from src.executor.policy import (
    EXPLORE_TOOLS,
    explore_tool_denial,
    redundant_read_denial,
)
from src.harness.edit.apply import execute_replace_symbol
from src.harness.edit.target import EditTarget
from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.context_pipeline import ExecutorRuntimeState
from src.harness.subtask.handoff import SubtaskHandoffBundle
from src.harness.subtask.session_memory import ExploreSessionMemory
from src.harness.subtask.tool_recovery import apply_post_tool_recovery
from src.orchestrator.path_guard import (
    collect_paths_from_tool,
    format_whitelist_denial,
    is_path_allowed,
    should_apply_context_whitelist,
)
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode
from src.tools.arg_normalize import normalize_grep_search_args, normalize_shell_exec_args
from src.tools.base import ToolResult
from src.tools.registry import ToolRegistry

EDIT_TOOLS = frozenset({"edit_file", "write_file", "delete_file", "replace_symbol"})
PARALLEL_READ_ONLY_TOOLS = frozenset({
    "context_search",
    "read_file",
    "read_files",
    "grep_search",
    "map_search",
    "glob_files",
    "list_dir",
    "git_status",
})


@dataclass
class _PreparedToolCall:
    tc: ToolCall
    call_args: dict


@dataclass
class _ToolExecutionResult:
    prepared: _PreparedToolCall
    result: ToolResult


@dataclass
class ToolRoundStats:
    explore_ok: bool = False
    explore_used: bool = False


@dataclass
class ToolPipelineContext:
    subtask: SubTaskNode
    root_task: str
    project_root: Path
    runtime_tools: frozenset[str]
    preloaded_paths: frozenset[str]
    paths_only_mode: bool
    truncated_paths: frozenset[str]
    whitelist_files: list[str]
    policy: TruncationPolicy
    memory: ExploreSessionMemory
    messages: list[Message]
    error_trace: list[str]
    tool_failures: list[int]
    shell_tracker: ShellCommandTracker
    pre_edit_snapshots: dict[str, str]
    file_changes: list[str]
    max_tool_output_chars: int = 14_000
    edit_read_fallback: bool = False
    edit_targets: tuple[EditTarget, ...] = ()
    edit_splice_max_attempts: int = 2
    runtime: ExecutorRuntimeState | None = None
    splice_edit: bool = False
    round_stats: ToolRoundStats = field(default_factory=ToolRoundStats)


class ExecutorToolPipeline:
    """Harness tool middleware: guards, cache, truncate, digest, metrics."""

    def __init__(
        self,
        tools: ToolRegistry,
        permissions: PermissionManager,
        *,
        approval_waiter: Callable[[str], Awaitable[bool]],
        prepare_approval: Callable[[str], None] | None = None,
        normalize_path: Callable[[Path, object], str | None],
        snapshot_before_edit: Callable[[Path, ToolCall, dict[str, str]], None],
        collect_diff: Callable[[dict[str, str], Path, list[str]], dict[str, str]],
    ) -> None:
        self.tools = tools
        self.permissions = permissions
        self._approval_waiter = approval_waiter
        self._prepare_approval = prepare_approval
        self._normalize_path = normalize_path
        self._snapshot_before_edit = snapshot_before_edit
        self._collect_diff = collect_diff

    async def process_tool_round(
        self,
        response: LLMResponse,
        ctx: ToolPipelineContext,
    ) -> AsyncIterator[AgentEvent]:
        """Run tool calls then Harness post-round recovery (nudges, runtime policy)."""
        async for event in self.process_tool_calls(response, ctx):
            yield event

        if ctx.runtime is None:
            return

        recovery = apply_post_tool_recovery(
            subtask=ctx.subtask,
            runtime=ctx.runtime,
            error_trace=ctx.error_trace,
            splice_edit=ctx.splice_edit,
        )
        for nudge in recovery.nudges:
            ctx.messages.append(nudge)

        subtask_id = ctx.subtask.id
        for line in recovery.status_lines:
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Executor [{subtask_id}] · {line}",
                data={
                    "phase": "executor",
                    "subtask_id": subtask_id,
                    "executor_activity": True,
                },
            )

    async def process_tool_calls(
        self,
        response: LLMResponse,
        ctx: ToolPipelineContext,
    ) -> AsyncIterator[AgentEvent]:
        stats = ToolRoundStats()
        if not response.tool_calls:
            return

        if _can_parallelize_tool_round(response.tool_calls):
            async for event in self._process_parallel_read_only_calls(
                response.tool_calls,
                ctx,
                stats,
            ):
                yield event
            ctx.round_stats = stats
            return

        whitelist = ctx.whitelist_files
        truncated = ctx.truncated_paths
        allowed_read_paths = frozenset(_norm_rel_path(p) for p in whitelist)

        for tc in response.tool_calls:
            yield tool_call_event(tc.name, tc.arguments, phase="executor")

            if tc.name not in ctx.runtime_tools:
                deny = _tool_denial_for_context(tc.name, ctx)
                ctx.error_trace.append(deny)
                ctx.messages.append(tool_message(tc.id, deny))
                yield tool_result_event(tc.name, deny, success=False, phase="executor")
                continue

            if tc.name in EXPLORE_TOOLS:
                stats.explore_used = True

            deny = self._pre_execute_guards(tc, ctx, allowed_read_paths, truncated, whitelist)
            if deny is not None:
                content, success = deny
                ctx.messages.append(tool_message(tc.id, content))
                yield tool_result_event(tc.name, content, success=success, phase="executor")
                if not success:
                    ctx.tool_failures[0] += 1
                elif tc.name in EXPLORE_TOOLS:
                    stats.explore_ok = True
                continue

            call_args = self._normalize_call_args(tc, ctx)
            cached = self._serve_explore_cache(tc, call_args, ctx)
            if cached is not None:
                ctx.memory.cache_hits += 1
                duplicate = _duplicate_explore_denial(tc.name, call_args, ctx)
                content = duplicate or cached
                success = duplicate is None
                if duplicate:
                    ctx.error_trace.append(duplicate)
                    ctx.tool_failures[0] += 1
                ctx.messages.append(tool_message(tc.id, content))
                yield tool_result_event(tc.name, content, success=success, phase="executor")
                if success and tc.name in EXPLORE_TOOLS:
                    stats.explore_ok = True
                continue

            tool = self.tools.get(tc.name)
            approved = True
            if tool is not None:
                check = self.permissions.check(tc.name, tool.risk_level)
                if check.allowed:
                    approved = True
                elif not check.needs_prompt:
                    approved = False
                else:
                    if self._prepare_approval is not None:
                        self._prepare_approval(tc.name)
                    yield approval_event(tc.name, tool.risk_level.value)
                    approved = await self._approval_waiter(tc.name)
            if not approved:
                deny = f"Tool '{tc.name}' was denied by the user."
                ctx.error_trace.append(deny)
                ctx.tool_failures[0] += 1
                ctx.messages.append(tool_message(tc.id, deny))
                yield tool_result_event(tc.name, deny, success=False, phase="executor")
                continue

            if tc.name in EDIT_TOOLS:
                self._snapshot_before_edit(ctx.project_root, tc, ctx.pre_edit_snapshots)

            if tc.name == "replace_symbol":
                result = execute_replace_symbol(
                    project_root=ctx.project_root,
                    targets=list(ctx.edit_targets),
                    path=str(call_args.get("path", "")),
                    symbol=str(call_args.get("symbol", "")),
                    new_body=str(call_args.get("new_body", "")),
                    max_attempts=ctx.edit_splice_max_attempts,
                )
            else:
                result = await self.tools.call(tc.name, call_args)
                result = self._after_execute(tc, call_args, result, ctx)

            if tc.name == "shell_exec":
                cmd = tc.arguments.get("command")
                if isinstance(cmd, str):
                    ctx.shell_tracker.record_outcome(cmd, success=result.success)

            if tc.name in EDIT_TOOLS and result.success:
                path = tc.arguments.get("path", "unknown")
                ctx.file_changes.append(str(path))
                diffs = self._collect_diff(
                    ctx.pre_edit_snapshots, ctx.project_root, [str(path)]
                )
                diff = next(
                    iter(diffs.values()),
                    result.output[:4000] if result.output else "(edited)",
                )
                yield file_edit_event(str(path), diff)

            content = result.output if result.success else f"Error: {result.error}"
            if result.success and tc.name in EXPLORE_TOOLS:
                stats.explore_ok = True

            if not result.success and result.error:
                ctx.error_trace.append(f"{tc.name}: {result.error}")
                ctx.tool_failures[0] += 1
                if tc.name == "edit_file" and is_edit_recoverable_error(result.error):
                    ctx.messages.append(tool_message(tc.id, content))
                    yield tool_result_event(tc.name, content, success=False, phase="executor")
                    continue

            ctx.messages.append(tool_message(tc.id, content))
            yield tool_result_event(tc.name, content, success=result.success, phase="executor")

        ctx.round_stats = stats

    async def _process_parallel_read_only_calls(
        self,
        tool_calls: list[ToolCall],
        ctx: ToolPipelineContext,
        stats: ToolRoundStats,
    ) -> AsyncIterator[AgentEvent]:
        """Run one all-read-only tool round concurrently, preserving message order."""
        whitelist = ctx.whitelist_files
        truncated = ctx.truncated_paths
        allowed_read_paths = frozenset(_norm_rel_path(p) for p in whitelist)
        prepared: list[_PreparedToolCall] = []

        for tc in tool_calls:
            yield tool_call_event(tc.name, tc.arguments, phase="executor")

            if tc.name not in ctx.runtime_tools:
                deny = _tool_denial_for_context(tc.name, ctx)
                ctx.error_trace.append(deny)
                ctx.messages.append(tool_message(tc.id, deny))
                ctx.tool_failures[0] += 1
                yield tool_result_event(tc.name, deny, success=False, phase="executor")
                continue

            if tc.name in EXPLORE_TOOLS:
                stats.explore_used = True

            deny = self._pre_execute_guards(tc, ctx, allowed_read_paths, truncated, whitelist)
            if deny is not None:
                content, success = deny
                ctx.messages.append(tool_message(tc.id, content))
                yield tool_result_event(tc.name, content, success=success, phase="executor")
                if not success:
                    ctx.tool_failures[0] += 1
                elif tc.name in EXPLORE_TOOLS:
                    stats.explore_ok = True
                continue

            call_args = self._normalize_call_args(tc, ctx)
            cached = self._serve_explore_cache(tc, call_args, ctx)
            if cached is not None:
                ctx.memory.cache_hits += 1
                duplicate = _duplicate_explore_denial(tc.name, call_args, ctx)
                content = duplicate or cached
                success = duplicate is None
                if duplicate:
                    ctx.error_trace.append(duplicate)
                    ctx.tool_failures[0] += 1
                ctx.messages.append(tool_message(tc.id, content))
                yield tool_result_event(tc.name, content, success=success, phase="executor")
                if success and tc.name in EXPLORE_TOOLS:
                    stats.explore_ok = True
                continue

            tool = self.tools.get(tc.name)
            if tool is not None:
                check = self.permissions.check(tc.name, tool.risk_level)
                if not check.allowed:
                    deny = f"Tool '{tc.name}' was denied by policy."
                    ctx.error_trace.append(deny)
                    ctx.tool_failures[0] += 1
                    ctx.messages.append(tool_message(tc.id, deny))
                    yield tool_result_event(tc.name, deny, success=False, phase="executor")
                    continue

            prepared.append(_PreparedToolCall(tc=tc, call_args=call_args))

        if not prepared:
            return

        results = await asyncio.gather(
            *[
                self._execute_read_only_tool(prepared_call, ctx)
                for prepared_call in prepared
            ]
        )
        for item in results:
            tc = item.prepared.tc
            result = self._after_execute(tc, item.prepared.call_args, item.result, ctx)
            content = result.output if result.success else f"Error: {result.error}"
            if result.success and tc.name in EXPLORE_TOOLS:
                stats.explore_ok = True
            if not result.success and result.error:
                ctx.error_trace.append(f"{tc.name}: {result.error}")
                ctx.tool_failures[0] += 1
            ctx.messages.append(tool_message(tc.id, content))
            yield tool_result_event(tc.name, content, success=result.success, phase="executor")

    async def _execute_read_only_tool(
        self,
        prepared: _PreparedToolCall,
        ctx: ToolPipelineContext,
    ) -> _ToolExecutionResult:
        result = await self.tools.call(prepared.tc.name, prepared.call_args)
        return _ToolExecutionResult(prepared=prepared, result=result)

    def last_round_stats(self, ctx: ToolPipelineContext) -> ToolRoundStats:
        return ctx.round_stats

    def _serve_explore_cache(
        self,
        tc: ToolCall,
        call_args: dict,
        ctx: ToolPipelineContext,
    ) -> str | None:
        if tc.name not in EXPLORE_TOOLS:
            return None
        key = ctx.memory.explore_key(tc.name, call_args)
        if key is None:
            return None

        body = ctx.memory.get_output(key)
        if body:
            return ctx.memory.format_cached(body)

        if not ctx.memory.is_duplicate_explore(tc.name, call_args):
            return None

        if tc.name in {"read_file", "read_files"}:
            rel, start, end = self._read_range(call_args, tc.name)
            preload = ctx.memory.serve_read_from_preload(
                rel, start_line=start, end_line=end
            )
            if preload:
                ctx.memory.put_output(key, preload)
                return ctx.memory.format_cached(preload)

        fallback = ctx.memory.digest_fallback(key)
        return ctx.memory.format_cached(fallback)

    def _after_execute(
        self,
        tc: ToolCall,
        call_args: dict,
        result: ToolResult,
        ctx: ToolPipelineContext,
    ) -> ToolResult:
        ctx.memory.tool_calls += 1
        if not result.success:
            return result

        output = ExploreSessionMemory.truncate_output(
            result.output or "",
            max_chars=ctx.max_tool_output_chars,
        )
        key = ctx.memory.explore_key(tc.name, call_args)
        if key and tc.name in EXPLORE_TOOLS:
            ctx.memory.put_output(key, output)
            ctx.memory.record_explore(tc.name, call_args)
            if tc.name == "read_files":
                self._record_read_files(call_args, ctx)
            elif tc.name == "read_file":
                rel, start, end = self._read_range(call_args, "read_file")
                if rel:
                    ctx.memory.tracker.record_read(
                        rel,
                        start_line=start,
                        end_line=end,
                    )

        if result.output != output:
            return ToolResult(success=True, output=output, error=None)
        return result

    def _record_read_files(self, call_args: dict, ctx: ToolPipelineContext) -> None:
        paths = [p for p in (call_args.get("paths") or []) if isinstance(p, str)]
        start = call_args.get("start_line")
        end = call_args.get("end_line")
        for path_arg in paths:
            rel = self._normalize_path(ctx.project_root, path_arg) or normalize_rel_path(path_arg)
            ctx.memory.tracker.record_read(
                rel,
                start_line=start if isinstance(start, int) else None,
                end_line=end if isinstance(end, int) else None,
            )

    def _normalize_call_args(self, tc: ToolCall, ctx: ToolPipelineContext) -> dict:
        if tc.name == "grep_search":
            return normalize_grep_search_args(
                tc.arguments,
                subtask=ctx.subtask,
                hint_text=ctx.root_task,
            )
        if tc.name == "shell_exec":
            return normalize_shell_exec_args(tc.arguments, project_root=ctx.project_root)
        return dict(tc.arguments)

    def _pre_execute_guards(
        self,
        tc: ToolCall,
        ctx: ToolPipelineContext,
        allowed_read_paths: frozenset[str],
        truncated: frozenset[str],
        whitelist: list[str],
    ) -> tuple[str, bool] | None:
        project_root = ctx.project_root
        subtask = ctx.subtask

        if tc.name in {"read_file", "read_files"}:
            if (
                subtask.kind == SubTaskKind.EDIT
                and ctx.edit_read_fallback
                and ctx.runtime
                and ctx.runtime.edit_read_fallback
            ):
                if tc.name == "read_file" and "read_files" in ctx.runtime_tools:
                    return (
                        "read_file blocked for query-style edit: use one read_files call "
                        "with all needed context_files or the target range, then edit_file.",
                        False,
                    )
            raw_paths = (
                [tc.arguments["path"]]
                if tc.name == "read_file" and isinstance(tc.arguments.get("path"), str)
                else [p for p in (tc.arguments.get("paths") or []) if isinstance(p, str)]
            )
            rel_paths = [
                self._normalize_path(project_root, p) or normalize_rel_path(p)
                for p in raw_paths
            ]
            if subtask.kind == SubTaskKind.EDIT and ctx.preloaded_paths and not ctx.paths_only_mode:
                norm = [_norm_rel_path(p) for p in rel_paths if p]
                redundant = [p for p in norm if p in ctx.preloaded_paths and p not in truncated]
                if redundant:
                    start = tc.arguments.get("start_line")
                    end = tc.arguments.get("end_line")
                    if tc.name == "read_file" and isinstance(tc.arguments.get("path"), str):
                        _, ps, pe = parse_read_path_with_lines(tc.arguments["path"])
                        start = start if isinstance(start, int) else ps
                        end = end if isinstance(end, int) else pe
                    preload = ctx.memory.serve_read_from_preload(
                        redundant[0],
                        start_line=start if isinstance(start, int) else None,
                        end_line=end if isinstance(end, int) else None,
                    )
                    if preload:
                        key = (
                            ctx.memory.explore_key(tc.name, tc.arguments)
                            or f"read:{redundant[0]}"
                        )
                        ctx.memory.put_output(key, preload)
                        body = ctx.memory.format_cached(preload)
                        if ctx.edit_read_fallback:
                            body = (
                                "[Copy the lines below verbatim into edit_file old_string, "
                                "then supply your modified version as new_string]\n"
                                + body
                            )
                        return body, True
                    return (
                        redundant_read_denial(
                            redundant[0],
                            kind=subtask.kind,
                            paths_only=False,
                        ),
                        False,
                    )
            if subtask.kind == SubTaskKind.DIAGNOSE and ctx.preloaded_paths:
                norm = [_norm_rel_path(p) for p in rel_paths if p]
                redundant = [p for p in norm if p in ctx.preloaded_paths]
                if redundant and (tc.name == "read_file" or len(redundant) == len(norm)):
                    return redundant_read_denial(redundant[0]), True
            if subtask.kind == SubTaskKind.EDIT and ctx.paths_only_mode:
                norm = [_norm_rel_path(p) for p in rel_paths if p]
                out_of_scope = [p for p in norm if p and p not in allowed_read_paths]
                if out_of_scope:
                    return (
                        f"read_file blocked: '{out_of_scope[0]}' not in context_files "
                        f"({', '.join(whitelist)}).",
                        False,
                    )
            elif (
                subtask.kind == SubTaskKind.EDIT
                and ctx.preloaded_paths
                and not ctx.paths_only_mode
            ):
                norm = [_norm_rel_path(p) for p in rel_paths if p]
                out_of_scope = [p for p in norm if p and p not in ctx.preloaded_paths]
                if out_of_scope:
                    return (
                        f"read_file blocked: '{out_of_scope[0]}' is not in context_files. "
                        f"Allowed: {', '.join(sorted(ctx.preloaded_paths))}.",
                        False,
                    )
            blocked_fw = blocked_framework_reads(ctx.root_task, rel_paths)
            if blocked_fw:
                msg = format_framework_read_denial(blocked_fw)
                ctx.error_trace.append(msg)
                return msg, False

        if tc.name in {"list_dir", "glob_files", "grep_search"}:
            blocked_browse = blocked_framework_browse(ctx.root_task, tc.name, tc.arguments)
            if blocked_browse:
                msg = format_framework_browse_denial(blocked_browse, tc.name)
                ctx.error_trace.append(msg)
                return msg, False

        if tc.name == "context_search":
            raw_paths = [
                p for p in (tc.arguments.get("paths") or []) if isinstance(p, str)
            ]
            if raw_paths:
                blocked = [
                    p
                    for p in raw_paths
                    if not is_path_allowed(project_root, p, whitelist)
                ]
                if blocked:
                    msg = format_whitelist_denial(
                        tc.name,
                        blocked,
                        whitelist,
                        project_root=project_root,
                    )
                    ctx.error_trace.append(msg)
                    return msg, False

        if tc.name == "shell_exec":
            command = tc.arguments.get("command")
            if isinstance(command, str):
                deny = ctx.shell_tracker.check(command)
                if deny:
                    ctx.error_trace.append(deny)
                    return deny, False
                ctx.shell_tracker.record_run(command)

        if should_apply_context_whitelist(tc.name):
            blocked = [
                p
                for p in collect_paths_from_tool(tc.name, tc.arguments)
                if not is_path_allowed(project_root, p, whitelist)
            ]
            if blocked:
                msg = format_whitelist_denial(
                    tc.name, blocked, whitelist, project_root=project_root
                )
                ctx.error_trace.append(msg)
                return msg, False

        return None

    @staticmethod
    def _read_range(
        args: dict,
        tool_name: str,
    ) -> tuple[str, int | None, int | None]:
        if tool_name == "read_file":
            paths = [args.get("path")] if isinstance(args.get("path"), str) else []
        else:
            paths = [p for p in (args.get("paths") or []) if isinstance(p, str)]
        start = args.get("start_line")
        end = args.get("end_line")
        if not paths:
            return "", None, None
        parsed_path, ps, pe = parse_read_path_with_lines(str(paths[0]))
        if not isinstance(start, int) and ps is not None:
            start = ps
        if not isinstance(end, int) and pe is not None:
            end = pe
        rel = parsed_path.replace("\\", "/").lstrip("./")
        return rel, start if isinstance(start, int) else None, end if isinstance(end, int) else None


def build_tool_pipeline_context(
    *,
    bundle: SubtaskHandoffBundle,
    subtask: SubTaskNode,
    root_task: str,
    project_root: Path,
    turn_tools: frozenset[str],
    runtime: ExecutorRuntimeState,
    policy: TruncationPolicy,
    memory: ExploreSessionMemory,
    messages: list[Message],
    error_trace: list[str],
    tool_failures: list[int],
    shell_tracker: ShellCommandTracker,
    pre_edit_snapshots: dict[str, str],
    file_changes: list[str],
    max_tool_output_chars: int,
) -> ToolPipelineContext:
    """Harness builds tool middleware context — Executor only passes session state."""
    splice_edit = bool(bundle.edit_handoff and bundle.edit_handoff.active)
    return ToolPipelineContext(
        subtask=subtask,
        root_task=root_task,
        project_root=project_root,
        runtime_tools=turn_tools,
        preloaded_paths=runtime.preloaded_paths,
        paths_only_mode=runtime.paths_only_mode,
        truncated_paths=runtime.truncated_paths,
        whitelist_files=bundle.whitelist_files,
        policy=policy,
        memory=memory,
        messages=messages,
        error_trace=error_trace,
        tool_failures=tool_failures,
        shell_tracker=shell_tracker,
        pre_edit_snapshots=pre_edit_snapshots,
        file_changes=file_changes,
        max_tool_output_chars=max_tool_output_chars,
        edit_read_fallback=runtime.edit_read_fallback,
        edit_targets=tuple(bundle.edit_handoff.targets)
        if bundle.edit_handoff and bundle.edit_handoff.active
        else (),
        edit_splice_max_attempts=bundle.edit_splice_max_attempts,
        runtime=runtime,
        splice_edit=splice_edit,
    )


def _can_parallelize_tool_round(tool_calls: list[ToolCall]) -> bool:
    return len(tool_calls) > 1 and all(
        tc.name in PARALLEL_READ_ONLY_TOOLS for tc in tool_calls
    )


def _norm_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _tool_denial_for_context(tool_name: str, ctx: ToolPipelineContext) -> str:
    if ctx.subtask.kind == SubTaskKind.DIAGNOSE and not ctx.runtime_tools:
        return (
            f"Tool '{tool_name}' is disabled: diagnose context is Harness-owned. "
            "Use the context_search handoff or preloaded evidence; do not call "
            "raw exploration tools."
        )
    if not ctx.runtime_tools:
        return (
            f"Tool '{tool_name}' is disabled for this turn: Harness exposed no "
            "runtime tools because required context was already preloaded."
        )
    return explore_tool_denial(tool_name)


def _duplicate_explore_denial(
    tool_name: str,
    args: dict,
    ctx: ToolPipelineContext,
) -> str | None:
    if ctx.subtask.kind != SubTaskKind.DIAGNOSE:
        return None
    if tool_name == "context_search":
        query = args.get("query") or ""
        return (
            f"Blocked duplicate context_search in diagnose: {query!r}. "
            "Do not repeat equivalent searches. Summarize findings now."
        )
    return None
