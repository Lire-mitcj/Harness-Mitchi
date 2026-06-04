from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.agent.events import (
    AgentEvent,
    EventType,
    approval_event,
    cost_event,
    error_event,
    tool_call_event,
    tool_result_event,
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
    LLMResponse,
    Message,
    assistant_message,
    system_message,
    tool_message,
    user_message,
)
from src.harness.discovery.manifest import (
    DiagnosticsManifest,
    discovery_display_summary,
    extract_discovery_trace,
    manifest_actionable,
)
from src.harness.discovery.manifest_fallback import (
    manifest_reject_reason,
    try_parse_manifest_with_fallback,
)
from src.harness.discovery.scout_preflight import run_scout_preflight
from src.llm.dsml import strip_dsml_text

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.llm.client import LLMClient
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

_SCOUT_PROMPT_PATH = (
    Path(__file__).resolve().parent.parent.parent.parent / "prompts" / "scout_prompt.md"
)


_SCOUT_FALLBACK = (
    "You are MitKII Scout — read-only pre-Planner probe. Output discovery JSON only."
)


def load_scout_system_prompt() -> str:
    if _SCOUT_PROMPT_PATH.is_file():
        return _SCOUT_PROMPT_PATH.read_text(encoding="utf-8").strip()
    log.warning("scout_prompt.md missing, using embedded fallback")
    return _SCOUT_FALLBACK


SCOUT_TOOLS = frozenset({
    "read_file",
    "read_files",
    "grep_search",
    "glob_files",
    "list_dir",
    "shell_exec",
    "git_status",
})

EDIT_TOOLS = frozenset({"edit_file", "write_file", "delete_file", "git_commit", "git_stash"})

_MANIFEST_NUDGE_TRACE = (
    "Harness rejected manifest ({reason}). "
    "Reply: <discovery_trace> (5 one-liners) then ONE raw JSON (no fences)."
)
_MANIFEST_NUDGE_JSON = (
    "Harness rejected manifest ({reason}). "
    "Reply with ONE raw JSON only (root_cause and/or victim_files with lines)."
)


def _scout_manifest_instruction(settings: MitKIISettings) -> str:
    if settings.scout_trace:
        return (
            "Output <discovery_trace> (5 numbered one-liners) then ONE raw JSON manifest."
        )
    return "Output ONE raw JSON manifest only (no discovery_trace, no markdown fences)."


def _manifest_nudge(settings: MitKIISettings, reason: str) -> str:
    template = _MANIFEST_NUDGE_TRACE if settings.scout_trace else _MANIFEST_NUDGE_JSON
    return template.format(reason=reason)


class ScoutAgent:
    """Harness AOP pre-hook: read-only Scout before Planner (small LLM, manifest out)."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        harness: HarnessEngine,
        permissions: PermissionManager,
        settings: MitKIISettings,
    ) -> None:
        self.llm = llm
        self.tools = tools
        self.harness = harness
        self.permissions = permissions
        self.settings = settings
        self.last_manifest: DiagnosticsManifest | None = None
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._shell_tracker = ShellCommandTracker(
            dedup_limit=settings.shell_dedup_limit,
            stagnant_limit=settings.shell_stagnant_limit,
        )

    async def resolve_approval(self, action: str, approved: bool) -> None:
        self.permissions.record_decision(action, approved)
        fut = self._approval_futures.pop(action, None)
        if fut and not fut.done():
            fut.set_result(approved)

    def _scout_budget_label(self) -> tuple[int, int]:
        tool_cap = self.settings.scout_max_tool_turns
        manifest_cap = self.settings.scout_manifest_attempts
        total = tool_cap + manifest_cap
        return total, tool_cap

    async def run(
        self,
        user_request: str,
        project_structure: str,
    ) -> AsyncIterator[AgentEvent]:
        self._shell_tracker = ShellCommandTracker(
            dedup_limit=self.settings.shell_dedup_limit,
            stagnant_limit=self.settings.shell_stagnant_limit,
        )
        project_root = self.harness.project_root
        messages: list[Message] = [
            system_message(load_scout_system_prompt()),
            system_message(f"<project_structure>\n{project_structure}\n</project_structure>"),
            user_message(
                f"Discovery for Planner. Facts only.\n"
                f"{_scout_manifest_instruction(self.settings)}\n"
                f"Request: {user_request}"
            ),
        ]
        tool_schemas = self.tools.get_schemas(include=SCOUT_TOOLS)
        turns_used = 0
        final_text = ""
        manifest = DiagnosticsManifest(
            user_request=user_request,
            uncertainties=["Scout produced no manifest."],
        )
        total_budget, tool_cap = self._scout_budget_label()
        preflight_grep = ""

        yield AgentEvent(type=EventType.STREAM_START)
        yield AgentEvent(
            type=EventType.STATUS,
            content="Harness Scout (pre-Planner, read-only)...",
            data={"phase": "scout", "spinner_only": True},
        )

        yield AgentEvent(
            type=EventType.STATUS,
            content="Scout preflight grep (harness)...",
            data={"spinner_only": True, "phase": "scout"},
        )
        if self.settings.scout_preflight_grep:
            preflight_grep = await run_scout_preflight(self.tools, user_request)
            if preflight_grep:
                messages.append(
                    system_message(
                        "<scout_preflight_grep>\n"
                        "Harness ran grep before Scout LLM (use these facts in your manifest):\n"
                        f"{preflight_grep[:2500]}\n"
                        "</scout_preflight_grep>"
                    )
                )
                log.info("Scout preflight grep: %d chars", len(preflight_grep))
            else:
                log.warning("Scout preflight grep found no matches for: %s", user_request[:80])
        else:
            log.info("Scout preflight grep disabled — Scout LLM only")

        # Phase A — tool exploration (grep/read/shell); preflight grep is context only.
        for tool_turn in range(1, tool_cap + 1):
            turns_used += 1
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Scout exploring (tool turn {tool_turn}/{tool_cap})...",
                data={"spinner_only": True, "llm_loading": True, "phase": "scout"},
            )
            response = await self._call_llm(
                messages,
                tools=tool_schemas,
                max_tokens=self.settings.scout_max_tokens,
            )
            if response is None:
                yield error_event("Scout LLM returned no response")
                manifest = DiagnosticsManifest(
                    user_request=user_request,
                    uncertainties=["Scout LLM returned no response."],
                )
                break

            async for event in self._emit_usage(response, messages):
                yield event

            if response.tool_calls:
                messages.append(assistant_message(response.content or "", response.tool_calls))
                async for event in self._process_tools(
                    response,
                    messages,
                    user_request=user_request,
                    project_root=project_root,
                ):
                    yield event
                continue

            final_text = response.content or ""
            messages.append(assistant_message(final_text))
            candidate = try_parse_manifest_with_fallback(
                final_text,
                user_request=user_request,
                messages=messages,
                preflight_grep=preflight_grep,
            )
            if manifest_actionable(candidate):
                manifest = candidate
                break
            log.info(
                "Scout early manifest rejected during tool phase: %s",
                manifest_reject_reason(candidate),
            )

        # Phase B — manifest-only (no tools; avoids tool+JSON competing for one turn)
        if not manifest_actionable(manifest):
            messages.append(
                user_message(
                    "Tool exploration complete. Do NOT call tools. "
                    f"{_scout_manifest_instruction(self.settings)}"
                )
            )
            manifest_cap = self.settings.scout_manifest_attempts
            for attempt in range(1, manifest_cap + 1):
                turns_used += 1
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=(
                        f"Scout writing manifest (attempt {attempt}/{manifest_cap}, "
                        f"turn {turns_used}/{total_budget})..."
                    ),
                    data={"spinner_only": True, "llm_loading": True, "phase": "scout"},
                )
                response = await self._call_llm(
                    messages,
                    tools=None,
                    max_tokens=self.settings.scout_manifest_max_tokens,
                )
                if response is None:
                    yield error_event("Scout manifest LLM returned no response")
                    break

                async for event in self._emit_usage(response, messages):
                    yield event

                if response.tool_calls:
                    deny = (
                        "Manifest phase forbids tool calls. "
                        f"{_scout_manifest_instruction(self.settings)}"
                    )
                    messages.append(assistant_message(response.content or "", response.tool_calls))
                    for tc in response.tool_calls:
                        messages.append(tool_message(tc.id, deny))
                    messages.append(user_message(deny))
                    continue

                final_text = response.content or ""
                messages.append(assistant_message(final_text))
                candidate = try_parse_manifest_with_fallback(
                    final_text,
                    user_request=user_request,
                    messages=messages,
                    preflight_grep=preflight_grep,
                )
                if manifest_actionable(candidate):
                    manifest = candidate
                    break

                reason = manifest_reject_reason(candidate)
                log.info("Scout manifest rejected: %s", reason)
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=f"Scout manifest rejected ({reason}) — retrying...",
                    data={"phase": "scout"},
                )
                if attempt < manifest_cap:
                    messages.append(
                        user_message(_manifest_nudge(self.settings, reason))
                    )
                    continue
                manifest = candidate
                break

        if not manifest_actionable(manifest):
            harness_manifest = try_parse_manifest_with_fallback(
                final_text,
                user_request=user_request,
                messages=messages,
                preflight_grep=preflight_grep,
            )
            if manifest_actionable(harness_manifest):
                manifest = harness_manifest
                log.info("Scout manifest recovered from preflight/text extraction")

        trace = extract_discovery_trace(final_text)
        if trace:
            log.debug("Scout discovery trace:\n%s", trace)

        manifest.scout_turns_used = turns_used
        self.last_manifest = manifest

        yield AgentEvent(
            type=EventType.STATUS,
            content=discovery_display_summary(manifest),
            data={
                "discovery": manifest.to_dict(),
                "discovery_trace": trace,
                "milestone": "discovery_done",
            },
        )

    async def _call_llm(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None,
        max_tokens: int,
    ) -> LLMResponse | None:
        trimmed = await self.harness.before_llm_call([m.to_dict() for m in messages])
        response: LLMResponse | None = None
        async for chunk in self._stream_llm(trimmed, tools, max_tokens=max_tokens):
            if chunk.get("type") == "response":
                response = chunk["response"]
        if response is not None:
            await self.harness.after_llm_call(response, response.usage)
        return response

    async def _emit_usage(
        self,
        response: LLMResponse,
        messages: list[Message],
    ) -> AsyncIterator[AgentEvent]:
        trimmed = [m.to_dict() for m in messages]
        if response.usage:
            from src.harness.probe.llm_usage import record_litellm_completion

            record_litellm_completion(
                self.harness.probe.metrics,
                response,
                model=response.model or self.llm.model,
                messages=trimmed,
                completion_text=response.content or "",
            )
            cost = (
                self.harness.probe.metrics.last_record.cost
                if self.harness.probe.metrics.last_record
                else 0.0
            )
            usage = response.usage
            yield cost_event(usage.prompt_tokens, usage.completion_tokens, cost)
        elif trimmed:
            from src.harness.probe.llm_usage import estimate_cost_for_model, estimate_usage_from_text

            est = estimate_usage_from_text(
                trimmed,
                response.content or "",
                model=self.llm.model,
            )
            cost = estimate_cost_for_model(
                est.prompt_tokens,
                est.completion_tokens,
                self.llm.model,
            )
            self.harness.probe.metrics.record(
                self.llm.model,
                est.prompt_tokens,
                est.completion_tokens,
                cost=cost,
            )
            yield cost_event(est.prompt_tokens, est.completion_tokens, cost)

    async def _stream_llm(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        *,
        max_tokens: int,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            async for content_chunk, final_response in self.llm.chat_stream(
                messages,
                tools=tools,
                max_tokens=max_tokens,
            ):
                clean_chunk = strip_dsml_text(content_chunk)
                if clean_chunk:
                    yield {"type": "content", "content": clean_chunk}
                if final_response is not None:
                    yield {"type": "response", "response": final_response}
        except Exception as exc:
            yield {
                "type": "response",
                "response": LLMResponse(
                    content=f"Scout LLM error: {exc}",
                    tool_calls=None,
                    usage=None,
                    model="error",
                ),
            }

    async def _process_tools(
        self,
        response: LLMResponse,
        messages: list[Message],
        *,
        user_request: str,
        project_root: Path,
    ) -> AsyncIterator[AgentEvent]:
        if not response.tool_calls:
            return

        for tc in response.tool_calls:
            yield tool_call_event(tc.name, tc.arguments, phase="scout")

            if tc.name in EDIT_TOOLS:
                deny = (
                    f"Tool '{tc.name}' is forbidden during Scout discovery. "
                    "Use read/grep/shell only."
                )
                messages.append(tool_message(tc.id, deny))
                yield tool_result_event(tc.name, deny, success=False, phase="scout")
                continue

            if tc.name not in SCOUT_TOOLS:
                deny = f"Tool '{tc.name}' is not available to Scout."
                messages.append(tool_message(tc.id, deny))
                yield tool_result_event(tc.name, deny, success=False, phase="scout")
                continue

            if tc.name in {"read_file", "read_files"}:
                raw_paths = (
                    [tc.arguments["path"]]
                    if tc.name == "read_file" and isinstance(tc.arguments.get("path"), str)
                    else [p for p in (tc.arguments.get("paths") or []) if isinstance(p, str)]
                )
                rel_paths = [
                    self._normalize_path(project_root, p) or normalize_rel_path(p)
                    for p in raw_paths
                ]
                blocked = blocked_framework_reads(user_request, rel_paths)
                if blocked:
                    deny = format_framework_read_denial(blocked)
                    messages.append(tool_message(tc.id, deny))
                    yield tool_result_event(tc.name, deny, success=False, phase="scout")
                    continue

            if tc.name in {"list_dir", "glob_files", "grep_search"}:
                blocked = blocked_framework_browse(user_request, tc.name, tc.arguments)
                if blocked:
                    deny = format_framework_browse_denial(blocked, tc.name)
                    messages.append(tool_message(tc.id, deny))
                    yield tool_result_event(tc.name, deny, success=False, phase="scout")
                    continue

            if tc.name == "shell_exec":
                command = tc.arguments.get("command")
                if isinstance(command, str):
                    deny = self._shell_tracker.check(command)
                    if deny:
                        messages.append(tool_message(tc.id, deny))
                        yield tool_result_event(tc.name, deny, success=False, phase="scout")
                        continue
                    self._shell_tracker.record_run(command)

            approved = True
            tool = self.tools.get(tc.name)
            if self.settings.scout_auto_approve:
                approved = True
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
                        approved = False
            else:
                approved = False

            if not approved:
                deny = f"Tool '{tc.name}' was denied."
                messages.append(tool_message(tc.id, deny))
                yield tool_result_event(tc.name, deny, success=False, phase="scout")
                continue

            result = await self.tools.call(tc.name, tc.arguments)

            if tc.name == "shell_exec":
                cmd = tc.arguments.get("command")
                if isinstance(cmd, str):
                    self._shell_tracker.record_outcome(cmd, success=result.success)

            content = result.output if result.success else f"Error: {result.error}"
            messages.append(tool_message(tc.id, content))
            yield tool_result_event(tc.name, content, success=result.success, phase="scout")

    @staticmethod
    def _normalize_path(project_root: Path, path: object) -> str | None:
        if not isinstance(path, str) or not path.strip():
            return None
        try:
            p = Path(path)
            if not p.is_absolute():
                p = (project_root / p).resolve()
            else:
                p = p.resolve()
            return str(p.relative_to(project_root.resolve()))
        except Exception:
            return path.strip().replace("\\", "/")
