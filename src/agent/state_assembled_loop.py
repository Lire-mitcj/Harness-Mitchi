from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.context_assembly import ContextAssembly
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
from src.agent.types import (
    AgentState,
    LLMResponse,
    Message,
    ToolCall,
    ToolResult,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from src.llm.dsml import strip_dsml_text

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

ASSEMBLED_TOOL_NAMES = frozenset({"codebase_retrieve", "decision_edit"})


from dataclasses import dataclass, field, replace

@dataclass(frozen=True, slots=True)
class AssembledState:
    """State tracking for StateAssembledLoop."""
    active_files: tuple[str, ...] = ()
    checklist: tuple[str, ...] = ()
    git_diff: str = ""
    validation_error: str | None = None
    messages_history: tuple[Message, ...] = ()
    current_step: int = 0
    max_steps: int = 20
    error_recovery_count: int = 0
    context_compact_count: int = 0
    stagnant_count: int = 0
    search_cache: dict[str, Any] = field(default_factory=dict)
    agent_state: AgentState = field(default_factory=AgentState)
    denied_by_user: bool = False
    patch_failed: bool = False
    validation_failed: bool = False
    empty_retrieve: bool = False

    def getMessagesAfterCompactBoundary(self) -> tuple[Message, ...]:
        for idx in range(len(self.messages_history) - 1, -1, -1):
            msg = self.messages_history[idx]
            if "[COMPACT_BOUNDARY]" in msg.content or msg.role == "compact_boundary":
                return self.messages_history[idx + 1:]
        return self.messages_history
class SystemLayerShaper:
    """Shaper 1: System prompt formatting and environment info assembly."""
    def shape(self, state: AssembledState, system_prompt: str) -> str:
        # Extract summaries from state.messages_history before the compact boundary
        summaries = []
        for msg in state.messages_history:
            if "STRUCTURED CONVERSATION SUMMARY" in msg.content:
                summaries.append(msg.content)
        
        if summaries:
            summary_section = "\n\n### HISTORICAL CONVERSATION SUMMARIES ###\n" + "\n\n".join(summaries)
            return f"{system_prompt}{summary_section}"
        return system_prompt

class ProjectConfigShaper:
    """Shaper 2: Compresses or prunes project rules and configuration layers."""
    def shape(self, state: AssembledState, project_rules: str) -> str:
        return project_rules

class MemoryShaper:
    """Shaper 3: Formats checklist and session facts."""
    def shape(self, state: AssembledState, checklist: tuple[str, ...]) -> tuple[str, ...]:
        return checklist

class ConversationShaper:
    """Shaper 4: Conversation history compaction shaper."""
    def shape(self, state: AssembledState) -> AssembledState:
        # Auto-compact if history is long (e.g., > 10 messages)
        # Check if messages need compaction
        messages = list(state.messages_history)
        if len(messages) <= 8:
            return state

        # Find the last compact boundary index if any
        last_boundary_idx = -1
        for idx in range(len(messages) - 1, -1, -1):
            if "[COMPACT_BOUNDARY]" in messages[idx].content:
                last_boundary_idx = idx
                break

        # Only compact if there are at least 6 new messages since last boundary or start
        new_msgs_count = len(messages) - (last_boundary_idx + 1)
        if new_msgs_count < 6:
            return state

        # We keep the messages before the boundary (if any) and summarize the ones between last boundary and keep-recent count
        # Let's keep the last 4 messages intact
        fold_end = len(messages) - 4
        fold_start = 0 if last_boundary_idx == -1 else last_boundary_idx + 1
        
        to_fold = messages[fold_start:fold_end]
        to_keep = messages[fold_end:]
        prior_part = messages[:fold_start]

        # Build structured summary of to_fold
        summary_lines = ["### STRUCTURED CONVERSATION SUMMARY ###"]
        for m in to_fold:
            content_preview = m.content[:150] + "..." if len(m.content) > 150 else m.content
            summary_lines.append(f"- **{m.role}**: {content_preview}")
            if m.tool_calls:
                for tc in m.tool_calls:
                    summary_lines.append(f"  - Called tool: {tc.name}")
        summary_text = "\n".join(summary_lines)

        summary_msg = Message(role="system", content=summary_text)
        boundary_msg = Message(role="system", content="[COMPACT_BOUNDARY] Context compressed.")

        new_history = tuple(prior_part) + (summary_msg, boundary_msg) + tuple(to_keep)
        return replace(
            state,
            messages_history=new_history,
            context_compact_count=state.context_compact_count + 1
        )

class RuntimeShaper:
    """Shaper 5: Active files and runtime validation error shaper."""
    def shape(self, state: AssembledState, active_files: tuple[str, ...]) -> tuple[str, ...]:
        # If we have a lot of active files, we can restrict them to the most recently retrieved or target ones
        if len(active_files) > 6:
            return active_files[-6:]
        return active_files



async def _get_git_diff(cwd: Path) -> str:
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "diff",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(cwd),
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)
        if proc.returncode == 0:
            return stdout.decode(errors="replace").strip()
    except Exception:
        pass
    return ""


class StateAssembledLoop:
    """A state-assembled agent loop driving coordinated tasks via high-level tools."""

    def __init__(
        self,
        llm: Any,
        tools: ToolRegistry,
        harness: HarnessEngine,
        context: Any,
        permissions: PermissionManager,
        settings: MitKIISettings,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.harness = harness
        self.permissions = permissions
        self.settings = settings

        self.context_assembly = ContextAssembly(harness.project_root)
        self.state = AssembledState(max_steps=settings.max_turns)
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}

    async def run(self, user_msg: str) -> AsyncIterator[AgentEvent]:
        system_prompt = self.context_assembly.load_system_prompt()
        user_context = user_msg
        permission_callback = self.permissions
        model_config = {
            "model": getattr(self.llm, "model", "default"),
            "max_steps": self.settings.max_turns,
        }
        async for event in self.queryLoop(
            system_prompt=system_prompt,
            user_context=user_context,
            permission_callback=permission_callback,
            model_config=model_config,
        ):
            yield event

    async def queryLoop(
        self,
        system_prompt: str,
        user_context: str,
        permission_callback: PermissionManager,
        model_config: dict[str, Any],
    ) -> AsyncIterator[AgentEvent]:
        # Initialize shapers
        sys_shaper = SystemLayerShaper()
        cfg_shaper = ProjectConfigShaper()
        mem_shaper = MemoryShaper()
        conv_shaper = ConversationShaper()
        run_shaper = RuntimeShaper()

        # Mutable state initialization (single State object)
        self.state = AssembledState(
            max_steps=model_config.get("max_steps", 20),
            messages_history=(),
            active_files=(),
            checklist=(),
            git_diff="",
            validation_error=None,
            current_step=0,
            error_recovery_count=0,
            context_compact_count=0,
            stagnant_count=0,
            search_cache={},
            agent_state=AgentState(),
        )

        yield AgentEvent(type=EventType.STREAM_START)
        self.harness.phase_metrics.reset_turn()

        step = 1
        while step <= self.state.max_steps:
            # Whole-object assignment: update step
            self.state = replace(self.state, current_step=step)

            # Update git diff
            git_diff = await _get_git_diff(self.harness.project_root)
            self.state = replace(self.state, git_diff=git_diff)

            # Update decision_edit tool with current active files
            edit_tool = self.tools.get("decision_edit")
            if edit_tool and hasattr(edit_tool, "set_active_files"):
                edit_tool.set_active_files(list(self.state.active_files))

            # --- RUN PRE-MODEL CONTEXT SHAPERS IN SEQUENCE ---
            # Shaper 4: Conversation compaction
            self.state = conv_shaper.shape(self.state)

            # Shaper 1: System Layer Shaper (injects history summaries if any)
            shaped_sys_prompt = sys_shaper.shape(self.state, system_prompt)

            # Shaper 2: Project Config Shaper
            rules_path = self.harness.project_root / ".mitkii" / "rules.md"
            rules_text = ""
            if rules_path.exists():
                try:
                    rules_text = rules_path.read_text(encoding="utf-8").strip()
                except OSError:
                    pass
            shaped_rules_text = cfg_shaper.shape(self.state, rules_text)

            # Shaper 3: Memory Shaper
            shaped_checklist = mem_shaper.shape(self.state, self.state.checklist)

            # Shaper 5: Runtime Shaper
            shaped_active_files = run_shaper.shape(self.state, self.state.active_files)

            # --- CONTEXT ASSEMBLY ---
            # Slice messages history after the compact boundary
            sliced_messages = list(self.state.getMessagesAfterCompactBoundary())

            # Now build the messages payload for LiteLLM/OpenAI
            system_content = shaped_sys_prompt
            if shaped_rules_text:
                system_content = f"{system_content}\n\n<project_rules>\n{shaped_rules_text}\n</project_rules>"

            checklist_str = "\n".join(f"- {item}" for item in shaped_checklist) or "- No checklist items"
            context_block = self.context_assembly.build_context_block(list(shaped_active_files))

            substituted = system_content
            has_new_slots = (
                "{{STATE.ACTIVE_FILES_LIST}}" in system_content or
                "{{STATE.CHECKLIST}}" in system_content or
                "{{STATE.GIT_DIFFS}}" in system_content or
                "{{STATE.BUILD_ERRORS}}" in system_content or
                "{{STATE.ACTIVE_FILES_BLOCKS}}" in system_content
            )

            if has_new_slots:
                active_files_list_str = ", ".join(shaped_active_files)
                git_diff_str = f"```diff\n{self.state.git_diff}\n```" if self.state.git_diff else "No unsaved changes."
                build_errors_str = f"```\n{self.state.validation_error}\n```" if self.state.validation_error else "No compile or build errors."
                active_files_blocks_str = context_block or "No active files in context yet."

                substituted = substituted.replace("{{STATE.ACTIVE_FILES_LIST}}", active_files_list_str)
                substituted = substituted.replace("{{STATE.CHECKLIST}}", checklist_str)
                substituted = substituted.replace("{{STATE.GIT_DIFFS}}", git_diff_str)
                substituted = substituted.replace("{{STATE.BUILD_ERRORS}}", build_errors_str)
                substituted = substituted.replace("{{STATE.ACTIVE_FILES_BLOCKS}}", active_files_blocks_str)

                assembled_sys_content = substituted
                user_instruction_block = f"Original Request: {user_context}"
            else:
                assembled_sys_content = system_content
                state_parts = [
                    "### CURRENT STATE ###",
                    f"Active Checklist:\n{checklist_str}",
                ]
                if self.state.git_diff:
                    state_parts.append(f"Git Diff:\n```diff\n{self.state.git_diff}\n```")
                if self.state.validation_error:
                    state_parts.append(f"Validation/Compiler Failures:\n```\n{self.state.validation_error}\n```")

                state_text = "\n\n".join(state_parts)
                context_text = f"### CURRENT CONTEXT ###\n\n{context_block}" if context_block else "### CURRENT CONTEXT ###\n\nNo active files in context yet."

                user_instruction_block = (
                    f"{state_text}\n\n"
                    f"{context_text}\n\n"
                    f"Original Request: {user_context}"
                )

            assembled_messages = []
            assembled_messages.append({"role": "system", "content": assembled_sys_content})
            for msg in sliced_messages:
                assembled_messages.append(msg.to_dict())
            assembled_messages.append({"role": "user", "content": user_instruction_block})

            # Stream thinking start status
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Agent · step {step}/{self.state.max_steps} · thinking…",
                data={
                    "spinner_only": True,
                    "llm_loading": True,
                    "phase": "agent",
                },
            )

            # Stream LLM coordinator thoughts & decisions
            response_text = ""
            response: LLMResponse | None = None
            tool_schemas = self.tools.get_schemas(include=ASSEMBLED_TOOL_NAMES)

            self.harness.phase_metrics.start("assembled_llm", subtask_id=str(step))
            try:
                async for chunk in self._stream_llm(assembled_messages, tool_schemas):
                    if chunk.get("type") == "content":
                        delta = chunk.get("content", "")
                        response_text += delta
                        yield thinking_event(delta)
                    elif chunk.get("type") == "response":
                        response = chunk["response"]
            finally:
                verdict = "ok" if response is not None and response.model != "error" else "error"
                self.harness.phase_metrics.end("assembled_llm", subtask_id=str(step), verdict=verdict)

            # -------------------------------------------------------------
            # Anchor 1: LLM Response Error/Empty Anchor
            # -------------------------------------------------------------
            if response is None or response.model == "error":
                err_msg = response.content if response else "LLM returned no response"
                # Whole-object assignment
                self.state = replace(
                    self.state,
                    error_recovery_count=self.state.error_recovery_count + 1,
                    messages_history=self.state.messages_history + (
                        assistant_message(f"Error recovery triggered: {err_msg}"),
                    )
                )
                yield error_event(f"LLM call failed: {err_msg}. Retrying step {step}...")
                if self.state.error_recovery_count > 3:
                    yield error_event("Too many consecutive LLM call failures. Aborting.")
                    break
                continue  # Retry Anchor 1

            self._trace_llm_response(step, response)

            # Record token usage & cost
            if response.usage:
                cost = self.harness.probe.metrics.record(
                    response.model,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                ).cost
                new_agent_state = AgentState(
                    messages=list(self.state.agent_state.messages),
                    file_changes=list(self.state.agent_state.file_changes),
                    current_plan=self.state.agent_state.current_plan,
                    turn_count=self.state.agent_state.turn_count,
                    total_tokens_used=self.state.agent_state.total_tokens_used,
                    total_cost=self.state.agent_state.total_cost,
                )
                new_agent_state.record_usage(response.usage, cost)
                self.state = replace(self.state, agent_state=new_agent_state)
                yield cost_event(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                )

            await self.harness.after_llm_call(response, response.usage)

            # Update checklist from thoughts/response dynamically
            matches = re.findall(r'-\s+\[( |x|X)\]\s+(.*)', response_text)
            if matches:
                checklist_items = []
                for status, task in matches:
                    check_char = "x" if status.lower() == "x" else " "
                    checklist_items.append(f"[{check_char}] {task.strip()}")
                self.state = replace(self.state, checklist=tuple(checklist_items))

            # Process actions (tool calls)
            if response.tool_calls:
                async for event in self._process_tool_calls(response):
                    yield event

                # Save checkpoint after executing step
                cp_id = await self.harness.save_checkpoint(
                    f"assembled_step_{step}", self.state.agent_state
                )
                if cp_id:
                    yield AgentEvent(
                        type=EventType.CHECKPOINT_SAVED,
                        data={"checkpoint_id": cp_id},
                    )

                # -------------------------------------------------------------
                # Anchor 2: User Permission Denial Anchor
                # -------------------------------------------------------------
                if self.state.denied_by_user:
                    yield AgentEvent(type=EventType.STATUS, content="Tool execution denied by user. Retrying...")
                    self.state = replace(self.state, stagnant_count=self.state.stagnant_count + 1)
                    step += 1
                    continue

                # -------------------------------------------------------------
                # Anchor 3: Tool Execution/Patch Generation Failure Anchor
                # -------------------------------------------------------------
                if self.state.patch_failed:
                    yield AgentEvent(type=EventType.STATUS, content="Patch generation failed. Retrying step...")
                    self.state = replace(self.state, stagnant_count=self.state.stagnant_count + 1)
                    step += 1
                    continue

                # -------------------------------------------------------------
                # Anchor 4: Compiler/Test Validation Failure Anchor
                # -------------------------------------------------------------
                if self.state.validation_failed:
                    yield AgentEvent(type=EventType.STATUS, content="Code validation failed. Retrying step...")
                    self.state = replace(self.state, stagnant_count=self.state.stagnant_count + 1)
                    step += 1
                    continue

                # -------------------------------------------------------------
                # Anchor 7: Search Retrieve Empty/Failed Anchor
                # -------------------------------------------------------------
                if self.state.empty_retrieve:
                    yield AgentEvent(type=EventType.STATUS, content="Codebase retrieve returned empty context. Retrying...")
                    self.state = replace(self.state, stagnant_count=self.state.stagnant_count + 1)
                    step += 1
                    continue

                # -------------------------------------------------------------
                # Anchor 6: Stagnant Scorer/Progress Anchor
                # -------------------------------------------------------------
                if self.state.stagnant_count > 3:
                    yield AgentEvent(type=EventType.STATUS, content="Progress is stagnant. Attempting alternate route...")
                    self.state = replace(
                        self.state,
                        stagnant_count=0,
                        messages_history=self.state.messages_history + (
                            system_message("System notice: Multiple retries occurred without progress. Please rethink your strategy."),
                        )
                    )
                    step += 1
                    continue

            else:
                # No tool calls means task is complete or clarification requested
                answer = response.content or response_text
                is_clarify = "clarification" in response_text.lower() or "clarify" in response_text.lower() or "?" in answer
                
                # -------------------------------------------------------------
                # Anchor 5: Clarification Request Anchor
                # -------------------------------------------------------------
                if is_clarify:
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            assistant_message(answer),
                        )
                    )
                    yield final_answer_event(answer)
                    yield AgentEvent(type=EventType.STREAM_END)
                    return

                self.state = replace(
                    self.state,
                    messages_history=self.state.messages_history + (
                        assistant_message(answer),
                    )
                )
                yield final_answer_event(answer)
                yield AgentEvent(type=EventType.STREAM_END)
                return

            step += 1

        yield error_event(
            f"Reached maximum steps ({self.state.max_steps})",
            {"step_count": self.state.current_step},
        )
        yield AgentEvent(type=EventType.STREAM_END)

    async def _process_tool_calls(
        self, response: LLMResponse
    ) -> AsyncIterator[AgentEvent]:
        # Reset orchestration flags
        self.state = replace(
            self.state,
            denied_by_user=False,
            patch_failed=False,
            validation_failed=False,
            empty_retrieve=False,
        )

        if not response.tool_calls:
            return

        self.state = replace(
            self.state,
            messages_history=self.state.messages_history + (
                assistant_message(response.content or "", response.tool_calls),
            )
        )

        all_succeeded = True
        validation_failed = False
        patch_failed = False
        denied_by_user = False
        empty_retrieve = False

        for tc in response.tool_calls:
            yield tool_call_event(tc.name, tc.arguments)

            if tc.name not in ASSEMBLED_TOOL_NAMES:
                denied_msg = (
                    f"Tool '{tc.name}' is not available in assembled mode. "
                    f"Allowed tools: {', '.join(sorted(ASSEMBLED_TOOL_NAMES))}."
                )
                self.state = replace(
                    self.state,
                    messages_history=self.state.messages_history + (
                        tool_message(tc.id, denied_msg),
                    )
                )
                self._trace_tool_result(tc.name, denied_msg, success=False)
                yield tool_result_event(tc.name, denied_msg, success=False)
                all_succeeded = False
                continue

            # Check permission
            approved = True
            tool = self.tools.get(tc.name)
            if tool is not None:
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
                self.state = replace(
                    self.state,
                    messages_history=self.state.messages_history + (
                        tool_message(tc.id, denied_msg),
                    )
                )
                self._trace_tool_result(tc.name, denied_msg, success=False)
                yield tool_result_event(tc.name, denied_msg, success=False)
                all_succeeded = False
                denied_by_user = True
                break

            # Yield dynamic thinking / loading status for this tool
            yield AgentEvent(
                type=EventType.STATUS,
                content=get_tool_status_text(tc.name, tc.arguments),
                data={"spinner_only": True, "phase": "executor"},
            )

            # Execute the tool
            tool_args = dict(tc.arguments)
            if self.state.search_cache:
                tool_args["_search_cache"] = self.state.search_cache

            self.harness.phase_metrics.start(f"tool_{tc.name}", subtask_id=str(self.state.current_step))
            tool_task = asyncio.create_task(
                self.tools.call(tc.name, tool_args),
                name=f"action-layer:{tc.name}:{tc.id}",
            )
            success = False
            try:
                result = await tool_task
                success = result.success
            except asyncio.CancelledError:
                self._trace_tool_result(tc.name, "Tool task cancelled", success=False)
                raise
            except Exception as exc:
                result = ToolResult(
                    success=False,
                    output="",
                    error=f"Tool task failed: {type(exc).__name__}: {exc}",
                )
            finally:
                self.harness.phase_metrics.end(
                    f"tool_{tc.name}",
                    subtask_id=str(self.state.current_step),
                    verdict="success" if success else "fail",
                )

            display_output = result.output if result.success else f"Error: {result.error}"
            self._trace_tool_result(tc.name, display_output, success=result.success)

            if result.success:
                # Capture search output/context to self.state.search_cache if returned
                if result.metadata:
                    merged_cache = dict(self.state.search_cache)
                    if "search_output" in result.metadata:
                        merged_cache["search_output"] = result.metadata["search_output"]
                    if "edit_context" in result.metadata:
                        merged_cache["edit_context"] = result.metadata["edit_context"]
                    if "snippets" in result.metadata:
                        merged_cache["snippets"] = result.metadata["snippets"]
                    self.state = replace(self.state, search_cache=merged_cache)

                if tc.name == "codebase_retrieve":
                    retrieved = result.metadata.get("retrieved_files") or []
                    if not retrieved:
                        empty_retrieve = True
                    
                    new_active = list(self.state.active_files)
                    for file in retrieved:
                        if file not in new_active:
                            new_active.append(file)
                    
                    summary = f"Retrieved codebase context. Loaded files: {list(retrieved)}"
                    self.state = replace(
                        self.state,
                        active_files=tuple(new_active),
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, summary),
                        )
                    )
                    display_output = f"Retrieved codebase context. Loaded {len(retrieved)} files."
                elif tc.name == "decision_edit":
                    target = tc.arguments.get("target_file", "unknown")
                    new_changes = list(self.state.agent_state.file_changes)
                    if target not in new_changes:
                        new_changes.append(target)
                    
                    new_agent_state = AgentState(
                        messages=list(self.state.agent_state.messages),
                        file_changes=new_changes,
                        current_plan=self.state.agent_state.current_plan,
                        turn_count=self.state.agent_state.turn_count,
                        total_tokens_used=self.state.agent_state.total_tokens_used,
                        total_cost=self.state.agent_state.total_cost,
                    )
                    
                    summary = f"decision_edit applied to {target}: validation passed."
                    self.state = replace(
                        self.state,
                        validation_error=None,
                        agent_state=new_agent_state,
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, summary),
                        )
                    )
                    display_output = "decision_edit validation passed."
                else:
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, result.output),
                        )
                    )
            else:
                all_succeeded = False
                if tc.name == "decision_edit":
                    exec_res = result.metadata.get("execution") if result.metadata else None
                    val_res = result.metadata.get("validation") if result.metadata else None
                    is_validation_failure = (exec_res and exec_res.success) and (val_res and not val_res.success)

                    if is_validation_failure:
                        validation_failed = True
                        val_err = val_res.error or result.error
                        summary = f"❌ 【自动化代码验证失败】：文件 `{tc.arguments.get('target_file')}` 补丁应用成功但验证未通过，修改已回滚。"
                        self.state = replace(
                            self.state,
                            validation_error=val_err,
                            messages_history=self.state.messages_history + (
                                tool_message(tc.id, summary),
                            )
                        )
                    else:
                        patch_failed = True
                        summary = f"❌ 【补丁生成失败】：无法匹配或应用补丁到 `{tc.arguments.get('target_file')}`。错误：{result.error or result.output}"
                        self.state = replace(
                            self.state,
                            validation_error=None,
                            messages_history=self.state.messages_history + (
                                tool_message(tc.id, summary),
                            )
                        )
                else:
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, f"Error: {result.error}"),
                        )
                    )

            yield tool_result_event(tc.name, display_output, success=result.success)

        # Update final orchestration flags on self.state
        self.state = replace(
            self.state,
            denied_by_user=denied_by_user,
            patch_failed=patch_failed,
            validation_failed=validation_failed,
            empty_retrieve=empty_retrieve,
        )

    async def resolve_approval(self, action: str, approved: bool) -> None:
        self.permissions.record_decision(action, approved)
        fut = self._approval_futures.pop(action, None)
        if fut and not fut.done():
            fut.set_result(approved)

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        return await self.harness.checkpoint_store.list_checkpoints()

    def get_probe_metrics(self) -> dict[str, Any]:
        summary = self.harness.probe.metrics.get_summary()
        summary.update(self.harness.phase_metrics.get_summary())
        return summary

    async def run_score_now(self) -> dict[str, Any] | None:
        return None

    async def _stream_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> AsyncIterator[dict[str, Any]]:
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

    @staticmethod
    def _trace_llm_response(step: int, response: LLMResponse) -> None:
        """Print one complete, machine-readable decision for each LLM step."""
        decision = {
            "step": step,
            "model": response.model,
            "content": response.content or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "name": tc.name,
                    "arguments": tc.arguments,
                }
                for tc in response.tool_calls or []
            ],
        }
        print(
            "[debug][llm][output-json]\n"
            + json.dumps(decision, ensure_ascii=False, indent=2, default=str),
            flush=True,
        )

    @staticmethod
    def _trace_tool_result(name: str, output: str, *, success: bool) -> None:
        state = "ok" if success else "error"
        print(
            f"[debug][action-layer][tool-result][{state}] {name}:\n{output}",
            flush=True,
        )
