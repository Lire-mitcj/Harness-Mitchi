from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from src.agent.events import (
    AgentEvent,
    EventType,
    approval_event,
    cost_event,
    error_event,
    final_answer_event,
    thinking_event,
    tool_call_event,
    tool_result_event,
    get_tool_status_text,
)
from src.agent.framework_guard import (
    blocked_framework_browse,
    blocked_framework_reads,
    format_framework_browse_denial,
    format_framework_read_denial,
    normalize_rel_path,
)
from src.agent.shell_guard import ShellCommandTracker
from src.agent.types import (
    AgentState,
    LLMResponse,
    Message,
    ToolCall,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from src.harness.quality_gate import build_diff, evaluate_quality_gate, scorer_signature
from src.llm.dsml import strip_dsml_text

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

EDIT_TOOLS = frozenset({"edit_file", "write_file", "delete_file"})
REWRITE_AUTO_APPROVE_TOOLS = frozenset({"edit_file", "write_file"})
# Halt auto-rewrite after this many consecutive scores with identical feedback.
STAGNANT_REWRITE_IDENTICAL_ROUNDS = 2


@runtime_checkable
class LLMClient(Protocol):
    """Expected interface for the LLM backend."""

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse: ...

    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[dict[str, Any]]: ...


@runtime_checkable
class ContextBuilder(Protocol):
    """Expected interface for building the initial system context."""

    async def build(self, user_message: str) -> list[Message]: ...


class AgentLoop:
    """Core Agent Loop — Think -> Act -> Observe cycle.

    This is the most important class in the entire system.
    All agent behavior flows through this loop.
    """

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        harness: HarnessEngine,
        context: ContextBuilder,
        permissions: PermissionManager,
        settings: MitKIISettings,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.harness = harness
        self.context = context
        self.permissions = permissions
        self.settings = settings
        self.state = AgentState()
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._quality_gate_failed = False
        self._rewrite_locked_files: set[str] | None = None
        self._rewrite_halted = False
        self._last_scorer_signature: str | None = None
        self._stagnant_identical_rounds = 0
        self._read_output_cache: dict[str, str] = {}
        self._shell_tracker = ShellCommandTracker(
            dedup_limit=settings.shell_dedup_limit,
            stagnant_limit=settings.shell_stagnant_limit,
        )

    async def run(self, user_msg: str) -> AsyncIterator[AgentEvent]:
        """Execute the Think -> Act -> Observe loop for a single user turn.

        Yields :class:`AgentEvent` instances as the agent progresses through
        reasoning, tool calls, and final answer generation.
        """
        context_messages = await self.context.build(user_msg)
        self.state.messages = context_messages
        self.state.add_message(user_message(user_msg))
        self._reset_rewrite_state()
        self._shell_tracker = ShellCommandTracker(
            dedup_limit=self.settings.shell_dedup_limit,
            stagnant_limit=self.settings.shell_stagnant_limit,
        )

        yield AgentEvent(type=EventType.STREAM_START)

        for _turn in range(self.settings.max_turns):
            self.state.advance_turn()
            turn_file_change_start = len(self.state.file_changes)

            # --- THINK ---
            trimmed = await self.harness.before_llm_call(
                [m.to_dict() for m in self.state.messages]
            )
            tool_schemas = self.tools.get_schemas()

            response_text = ""
            response: LLMResponse | None = None
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Agent · turn {self.state.turn_count} · calling model…",
                data={
                    "spinner_only": True,
                    "llm_loading": True,
                    "phase": "agent",
                },
            )

            self.harness.phase_metrics.start("core_llm", subtask_id=str(self.state.turn_count))
            try:
                async for chunk in self._stream_llm(trimmed, tool_schemas):
                    if chunk.get("type") == "content":
                        delta = chunk.get("content", "")
                        response_text += delta
                        yield thinking_event(delta)
                    elif chunk.get("type") == "response":
                        response = chunk["response"]
            finally:
                verdict = "ok" if response is not None and response.model != "error" else "error"
                self.harness.phase_metrics.end("core_llm", subtask_id=str(self.state.turn_count), verdict=verdict)

            if response is None:
                yield error_event("LLM returned no response")
                break

            if response.usage:
                cost = self.harness.probe.metrics.record(
                    response.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                ).cost
                self.state.record_usage(response.usage, cost)
                yield cost_event(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                )

            await self.harness.after_llm_call(response, response.usage)

            # --- ACT ---
            if response.tool_calls:
                self.state.add_message(
                    assistant_message(response.content or "", response.tool_calls)
                )
                async for event in self._process_tool_calls(response, self.state.messages):
                    yield event

                # After tool execution: auto-checkpoint
                cp_id = await self.harness.save_checkpoint(
                    "after_tool_calls", self.state
                )
                if cp_id:
                    yield AgentEvent(
                        type=EventType.CHECKPOINT_SAVED,
                        data={"checkpoint_id": cp_id},
                    )

                # If code was just edited, run scorer for feedback
                if self._just_edited_code():
                    changed_in_turn = self.state.file_changes[turn_file_change_start:]
                    score_event = await self._run_scorer(
                        changed_files=changed_in_turn,
                        user_msg=user_msg,
                    )
                    if score_event:
                        yield score_event
                        data = score_event.data or {}
                        if data.get("rewrite_halted"):
                            self._rewrite_halted = True
                            self._quality_gate_failed = False
                            halt_msg = data.get("halt_reason") or (
                                "Auto-rewrite halted: scorer feedback unchanged."
                            )
                            yield AgentEvent(type=EventType.STATUS, content=halt_msg)
                        elif data.get("auto_rewrite"):
                            self._quality_gate_failed = True
                            if changed_in_turn and self._rewrite_locked_files is None:
                                self._rewrite_locked_files = self._resolve_paths_under_project(
                                    changed_in_turn
                                )
                            rewrite_targets = sorted(
                                self._rewrite_locked_files
                                or self._resolve_paths_under_project(changed_in_turn)
                            )
                            rewrite_msg = self._build_rewrite_context(
                                changed_files=rewrite_targets,
                                feedback=data.get("feedback") or "",
                                user_msg=user_msg,
                                l0_passed=bool(data.get("l0_passed")),
                                l1_passed=bool(data.get("l1_passed")),
                                blockers=list(data.get("blockers") or []),
                            )
                            self.state.add_message(system_message(rewrite_msg))
                            yield AgentEvent(
                                type=EventType.STATUS,
                                content=(
                                    "L0/L1 failed; loaded file content for direct fix "
                                    "(no shell/read needed)."
                                ),
                            )
                        else:
                            self._quality_gate_failed = False
                            self._rewrite_locked_files = None
                            self._stagnant_identical_rounds = 0
                            self._last_scorer_signature = None

                continue

            # --- OBSERVE: no tool calls means final answer ---
            if self._rewrite_halted:
                halt_reason = (
                    "Auto-rewrite stopped: scorer feedback did not improve after "
                    f"{STAGNANT_REWRITE_IDENTICAL_ROUNDS} identical scoring rounds. "
                    "Please review and fix manually."
                )
                self.state.add_message(assistant_message(halt_reason))
                yield final_answer_event(halt_reason)
                yield AgentEvent(type=EventType.STREAM_END)
                return

            if self._quality_gate_failed:
                self.state.add_message(system_message(
                    "Quality gate is still failing. Do not finalize. "
                    "Make additional edits and rerun checks."
                ))
                yield AgentEvent(
                    type=EventType.STATUS,
                    content="Blocking final answer until L0/L1 pass.",
                )
                continue

            answer = response.content or response_text
            self.state.add_message(assistant_message(answer))
            yield final_answer_event(answer)
            yield AgentEvent(type=EventType.STREAM_END)
            return

        yield error_event(
            f"Reached maximum turns ({self.settings.max_turns})",
            {"turn_count": self.state.turn_count},
        )
        yield AgentEvent(type=EventType.STREAM_END)

    async def resolve_approval(self, action: str, approved: bool) -> None:
        """Called by the transport layer when the user responds to an approval request."""
        self.permissions.record_decision(action, approved)
        fut = self._approval_futures.pop(action, None)
        if fut and not fut.done():
            fut.set_result(approved)

    async def _stream_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
        """Wrap the LLM streaming call and yield normalized chunks."""
        try:
            async for chunk in self.llm.chat_stream(messages, tools=tools):
                if isinstance(chunk, dict):
                    yield chunk
                    continue
                content_chunk, final_response = chunk
                clean_chunk = strip_dsml_text(content_chunk)
                if clean_chunk:
                    yield {"type": "content", "content": clean_chunk}
                if final_response is not None:
                    yield {"type": "response", "response": final_response}
        except Exception as exc:
            from src.agent.error_recovery import ErrorRecovery

            recovery = ErrorRecovery()
            yield {
                "type": "response",
                "response": LLMResponse(
                    content=recovery.handle_tool_error(exc, "llm_call"),
                    tool_calls=None,
                    usage=None,
                    model="error",
                ),
            }

    async def _process_tool_calls(
        self,
        response: LLMResponse,
        messages: list[Message],
    ) -> AsyncIterator[AgentEvent]:
        """Execute each tool call from the LLM response, checking permissions."""
        if not response.tool_calls:
            return

        for tc in response.tool_calls:
            yield tool_call_event(tc.name, tc.arguments)

            if tc.name in {"read_file", "read_files"}:
                framework_deny = self._check_blocked_framework_reads(tc)
                if framework_deny:
                    messages.append(tool_message(tc.id, framework_deny))
                    yield tool_result_event(tc.name, framework_deny, success=False)
                    continue

            if tc.name in {"list_dir", "glob_files", "grep_search"}:
                user_msg = self._get_current_user_message()
                blocked = blocked_framework_browse(user_msg, tc.name, tc.arguments)
                if blocked:
                    deny = format_framework_browse_denial(blocked, tc.name)
                    messages.append(tool_message(tc.id, deny))
                    yield tool_result_event(tc.name, deny, success=False)
                    continue

            if tc.name == "shell_exec":
                command = tc.arguments.get("command")
                if isinstance(command, str):
                    deny = self._shell_tracker.check(command)
                    if deny:
                        messages.append(tool_message(tc.id, deny))
                        yield tool_result_event(tc.name, deny, success=False)
                        continue
                    self._shell_tracker.record_run(command)

            if (
                self._quality_gate_failed
                and not self._rewrite_halted
                and tc.name in {"read_file", "read_files"}
            ):
                deny_msg = (
                    f"{tc.name} skipped during quality-gate rewrite — "
                    "file content is already loaded in the last system message. "
                    "Fix directly with write_file/edit_file."
                )
                messages.append(tool_message(tc.id, deny_msg))
                yield tool_result_event(tc.name, deny_msg, success=False)
                continue

            if (
                self._quality_gate_failed
                and not self._rewrite_halted
                and tc.name == "shell_exec"
            ):
                deny_msg = (
                    "shell_exec is disabled during quality-gate rewrite. "
                    "Harness already ran L0/L1 scoring — do not re-run ruff/sed/pytest via shell. "
                    "Fix the file using scorer feedback with write_file/edit_file only."
                )
                messages.append(tool_message(tc.id, deny_msg))
                yield tool_result_event(tc.name, deny_msg, success=False)
                continue

            if self._quality_gate_failed and tc.name in EDIT_TOOLS:
                target_path = tc.arguments.get("path")
                if isinstance(target_path, str) and not self._is_rewrite_path_allowed(target_path):
                    allowed = ", ".join(sorted(self._rewrite_locked_files or [])) or "(none)"
                    deny_msg = (
                        "Quality-gate rewrite is file-locked. "
                        f"Requested path '{target_path}' is outside allowed set: {allowed}"
                    )
                    messages.append(tool_message(tc.id, deny_msg))
                    yield tool_result_event(tc.name, deny_msg, success=False)
                    continue

            approved = True
            tool = self.tools.get(tc.name)
            auto_approved_rewrite = (
                self._quality_gate_failed
                and not self._rewrite_halted
                and tc.name in REWRITE_AUTO_APPROVE_TOOLS
            )
            if auto_approved_rewrite:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=f"Auto-approved {tc.name} (quality-gate rewrite).",
                )
            elif tool is not None:
                check = self.permissions.check(tc.name, tool.risk_level)
                if check.allowed:
                    approved = True
                elif not check.needs_prompt:
                    approved = False
                else:
                    fut = asyncio.get_running_loop().create_future()
                    self._approval_futures[tc.name] = fut
                    yield approval_event(tc.name, tool.risk_level.value)
                    try:
                        approved = await asyncio.wait_for(fut, timeout=300.0)
                    except TimeoutError:
                        log.warning("Approval timeout for tool '%s'", tc.name)
                        approved = False

            if not approved:
                denied_msg = f"Tool '{tc.name}' was denied by the user."
                messages.append(tool_message(tc.id, denied_msg))
                yield tool_result_event(tc.name, denied_msg, success=False)
                continue

            if tc.name == "read_file":
                path_key = self._normalize_tool_path(tc.arguments.get("path"))
                if path_key and path_key in self._read_output_cache:
                    cached_content = self._read_output_cache[path_key]
                    messages.append(tool_message(tc.id, cached_content))
                    yield tool_result_event(
                        tc.name,
                        f"Read {tc.arguments.get('path', 'unknown')} (turn-cache, skipped disk)",
                        success=True,
                    )
                    continue

            # Yield dynamic thinking / loading status for this tool
            yield AgentEvent(
                type=EventType.STATUS,
                content=get_tool_status_text(tc.name, tc.arguments),
                data={"spinner_only": True, "phase": "executor"},
            )

            tool_args = dict(tc.arguments)
            if hasattr(self.state, "search_cache") and self.state.search_cache:
                tool_args["_search_cache"] = self.state.search_cache

            self.harness.phase_metrics.start(f"tool_{tc.name}", subtask_id=str(self.state.turn_count))
            success = False
            try:
                result = await self.tools.call(tc.name, tool_args)
                success = result.success
            except Exception as exc:
                success = False
                raise
            finally:
                self.harness.phase_metrics.end(
                    f"tool_{tc.name}",
                    subtask_id=str(self.state.turn_count),
                    verdict="success" if success else "fail",
                )

            if result.success and result.metadata and hasattr(self.state, "search_cache"):
                merged_cache = dict(self.state.search_cache)
                if "search_output" in result.metadata:
                    merged_cache["search_output"] = result.metadata["search_output"]
                if "edit_context" in result.metadata:
                    merged_cache["edit_context"] = result.metadata["edit_context"]
                if "snippets" in result.metadata:
                    merged_cache["snippets"] = result.metadata["snippets"]
                self.state.search_cache = merged_cache

            if tc.name == "shell_exec":
                cmd = tc.arguments.get("command")
                if isinstance(cmd, str):
                    self._shell_tracker.record_outcome(cmd, success=result.success)

            if tc.name in EDIT_TOOLS and result.success:
                path = tc.arguments.get("path", "unknown")
                self.state.file_changes.append(path)

            content = result.output if result.success else f"Error: {result.error}"
            # Keep full tool output in model context, but avoid dumping raw file contents in CLI.
            messages.append(tool_message(tc.id, content))

            display_content = content
            if tc.name == "read_file" and result.success:
                meta = result.metadata or {}
                path = tc.arguments.get("path", "unknown")
                path_key = self._normalize_tool_path(path)
                if path_key:
                    self._read_output_cache[path_key] = content
                display_content = self._format_read_display(
                    path,
                    meta.get("total_lines"),
                    bool(meta.get("cache_hit")),
                    meta.get("elapsed_ms"),
                )
            elif tc.name == "read_files" and result.success:
                meta = result.metadata or {}
                for entry in meta.get("per_file_outputs") or []:
                    if isinstance(entry, dict):
                        path_key = self._normalize_tool_path(entry.get("path"))
                        output = entry.get("output")
                        if path_key and isinstance(output, str):
                            self._read_output_cache[path_key] = output
                count = meta.get("file_count", 0)
                elapsed = meta.get("elapsed_ms", 0)
                paths = meta.get("paths") or []
                preview = ", ".join(Path(p).name for p in paths[:5])
                if len(paths) > 5:
                    preview += f", +{len(paths) - 5} more"
                display_content = f"Read {count} files [{preview}] ({elapsed}ms total)"

            yield tool_result_event(tc.name, display_content, success=result.success)

    def _just_edited_code(self) -> bool:
        """Check if the most recent tool call was a file edit/write."""
        for msg in reversed(self.state.messages):
            if msg.role == "tool":
                continue
            if msg.role == "assistant" and msg.tool_calls:
                return any(tc.name in EDIT_TOOLS for tc in msg.tool_calls)
            break
        return False

    async def _run_scorer(
        self,
        *,
        changed_files: list[str],
        user_msg: str,
    ) -> AgentEvent | None:
        """Run scorer on the current turn's edits and enforce quality gate."""
        result = await evaluate_quality_gate(
            self.harness,
            user_msg=user_msg,
            changed_files=changed_files,
        )
        if result is None:
            return None

        auto_rewrite = bool(result.get("auto_rewrite"))
        if auto_rewrite:
            signature = scorer_signature(result)
            if signature == self._last_scorer_signature:
                self._stagnant_identical_rounds += 1
            else:
                self._stagnant_identical_rounds = 1
                self._last_scorer_signature = signature

            if self._stagnant_identical_rounds >= STAGNANT_REWRITE_IDENTICAL_ROUNDS:
                result["rewrite_halted"] = True
                result["auto_rewrite"] = False
                result["halt_reason"] = (
                    "Auto-rewrite halted: scorer feedback unchanged for "
                    f"{STAGNANT_REWRITE_IDENTICAL_ROUNDS} consecutive rounds."
                )
                return AgentEvent(type=EventType.SCORE_RESULT, data=result)

        return AgentEvent(type=EventType.SCORE_RESULT, data=result)

    def _read_file_for_context(self, rel_path: str, *, max_lines: int = 300) -> str:
        """Load numbered file content for rewrite context (no tool call / no LLM round-trip)."""
        try:
            p = Path(rel_path)
            if not p.is_absolute():
                p = (self.harness.project_root / p).resolve()
            if not p.is_file():
                return f"(file not found: {rel_path})"
            lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
            if len(lines) > max_lines:
                numbered = [f"{i + 1:>6}|{line}" for i, line in enumerate(lines[:max_lines])]
                numbered.append(
                    f"\n... [{len(lines) - max_lines} more lines truncated for context]"
                )
                return "\n".join(numbered)
            return "\n".join(f"{i + 1:>6}|{line}" for i, line in enumerate(lines))
        except OSError as exc:
            return f"(could not read {rel_path}: {exc})"

    def _build_rewrite_context(
        self,
        *,
        changed_files: list[str],
        feedback: str,
        user_msg: str,
        l0_passed: bool,
        l1_passed: bool,
        blockers: list[str],
    ) -> str:
        """Pre-load changed files so the LLM can fix directly without shell/read loops."""
        parts = [
            "[Quality Gate FAIL — direct fix required]",
            f"Original user request:\n{user_msg}",
        ]
        if not l0_passed:
            parts.append(
                "L0 (lint/tests) FAILED. Fix programmatic errors shown in scorer feedback."
            )
        if not l1_passed:
            parts.append(
                "L1 (rubric/semantic) FAILED. Fix the issues below in the file content — "
                "do not re-diagnose with shell."
            )
            for blocker in blockers:
                parts.append(f"  • {blocker}")
        if feedback.strip():
            parts.append(f"Scorer feedback:\n{feedback.strip()}")
        parts.append(
            "\nCurrent file content (already loaded — do NOT call read_file or shell_exec):"
        )
        for rel in changed_files:
            parts.append(f"\n===== {rel} =====\n{self._read_file_for_context(rel)}")
        parts.append(
            "\nNext action: fix using write_file (preferred) or edit_file "
            "on the file(s) above only. "
            "Edits are auto-approved. Scoring will rerun after your edit."
        )
        return "\n".join(parts)

    async def _build_diff(self, changed_files: list[str]) -> str:
        """Build a scoped git diff for the currently changed files."""
        return await build_diff(self.harness, changed_files)

    def _resolve_paths_under_project(self, raw_paths: list[str]) -> set[str]:
        resolved: set[str] = set()
        for raw in raw_paths:
            try:
                p = Path(raw)
                if not p.is_absolute():
                    p = (self.harness.project_root / p).resolve()
                else:
                    p = p.resolve()
                rel = p.relative_to(self.harness.project_root)
                resolved.add(str(rel))
            except Exception:
                continue
        return resolved

    def _reset_rewrite_state(self) -> None:
        self._quality_gate_failed = False
        self._rewrite_locked_files = None
        self._rewrite_halted = False
        self._last_scorer_signature = None
        self._stagnant_identical_rounds = 0
        self._read_output_cache.clear()

    @staticmethod
    def _format_read_display(
        path: object,
        total_lines: object,
        cache_hit: bool,
        elapsed_ms: object,
    ) -> str:
        path_str = str(path) if path else "unknown"
        parts = [f"Read {path_str}"]
        if isinstance(total_lines, int):
            parts[0] += f" ({total_lines} lines)"
        timing: list[str] = []
        if isinstance(elapsed_ms, (int, float)):
            timing.append(f"{elapsed_ms}ms")
        if cache_hit:
            timing.append("disk-cache")
        if timing:
            parts.append(f"[{', '.join(timing)}]")
        return " ".join(parts)

    def _get_current_user_message(self) -> str:
        for msg in reversed(self.state.messages):
            if msg.role == "user":
                return msg.content or ""
        return ""

    def _collect_read_paths(self, tc: ToolCall) -> list[str]:
        if tc.name == "read_file":
            path = tc.arguments.get("path")
            return [path] if isinstance(path, str) else []
        if tc.name == "read_files":
            paths = tc.arguments.get("paths")
            if isinstance(paths, list):
                return [p for p in paths if isinstance(p, str)]
        return []

    def _check_blocked_framework_reads(self, tc: ToolCall) -> str | None:
        raw_paths = self._collect_read_paths(tc)
        if not raw_paths:
            return None

        rel_paths: list[str] = []
        for raw in raw_paths:
            rel = self._normalize_tool_path(raw)
            rel_paths.append(rel if rel else normalize_rel_path(raw))

        blocked = blocked_framework_reads(self._get_current_user_message(), rel_paths)
        if not blocked:
            return None
        return format_framework_read_denial(blocked)

    def _normalize_tool_path(self, path: object) -> str | None:
        if not isinstance(path, str) or not path.strip():
            return None
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (self.harness.project_root / p).resolve()
            else:
                p = p.resolve()
            return str(p.relative_to(self.harness.project_root))
        except Exception:
            return path.strip()

    @staticmethod
    def _scorer_signature(score_data: dict[str, Any]) -> str:
        return scorer_signature(score_data)

    def _is_rewrite_path_allowed(self, path: str) -> bool:
        if not self._rewrite_locked_files:
            return True
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (self.harness.project_root / p).resolve()
            else:
                p = p.resolve()
            rel = str(p.relative_to(self.harness.project_root))
            return rel in self._rewrite_locked_files
        except Exception:
            return False

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        """Expose checkpoint listing for CLI observability commands."""
        return await self.harness.checkpoint_store.list_checkpoints()

    def get_probe_metrics(self) -> dict[str, Any]:
        """Expose context-probe usage metrics summary."""
        summary = self.harness.probe.metrics.get_summary()
        summary.update(self.harness.phase_metrics.get_summary())
        return summary

    async def run_score_now(self) -> dict[str, Any] | None:
        """Run scorer on demand and return raw score payload."""
        event = await self._run_scorer(
            changed_files=self.state.file_changes[-10:],
            user_msg="manual /score",
        )
        return event.data if event else None
