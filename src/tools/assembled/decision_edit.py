from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
from pathlib import Path
from typing import Any

from src.agent.contracts import ContextPack, ContextWindow
from src.agent.decision import CursorDecisionLLM
from src.agent.executor import CursorExecutor
from src.agent.patch_applier import CursorPatchApplier
from src.agent.types import RiskLevel, ToolResult
from src.agent.validator import CursorValidator
from src.llm.client import LLMClient
from src.tools.base import Tool

log = logging.getLogger(__name__)

_SPAN_MERGE_GAP = 5
_CHARS_PER_TIMEOUT_SECOND = 800.0
_TIMEOUT_PER_CONTEXT_SPAN = 12.0
_MAX_DECISION_TIMEOUT = 300.0
_MAX_FAILED_PATCH_FEEDBACK_CHARS = 6000


def _norm_file_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def merge_context_spans(
    spans: list[tuple[str, int, int]],
    *,
    gap: int = _SPAN_MERGE_GAP,
) -> list[tuple[str, int, int]]:
    """Merge overlapping or nearby spans on the same file (reduces window count)."""
    if not spans:
        return []

    canonical_path: dict[str, str] = {}
    by_file: dict[str, list[tuple[int, int]]] = {}
    file_order: list[str] = []
    for file_path, start, end in spans:
        norm = _norm_file_path(file_path)
        canonical_path.setdefault(norm, file_path)
        if norm not in by_file:
            by_file[norm] = []
            file_order.append(norm)
        by_file[norm].append((start, end))

    merged: list[tuple[str, int, int]] = []
    for norm in file_order:
        intervals = sorted(by_file[norm])
        current_start, current_end = intervals[0]
        for start, end in intervals[1:]:
            if start <= current_end + gap + 1:
                current_end = max(current_end, end)
            else:
                merged.append((canonical_path[norm], current_start, current_end))
                current_start, current_end = start, end
        merged.append((canonical_path[norm], current_start, current_end))
    return merged


def decision_timeout_for_context(
    base_timeout: float,
    *,
    context_chars: int,
    span_count: int,
) -> float:
    """First-token (connect) deadline for decision_edit streaming.

    Active streams are bounded by per-chunk idle timeout on the LLM client,
    not this value as a total wall-clock cap.
    """
    scaled = (
        base_timeout
        + (max(context_chars, 0) / _CHARS_PER_TIMEOUT_SECOND)
        + (max(span_count, 0) * _TIMEOUT_PER_CONTEXT_SPAN)
    )
    return min(_MAX_DECISION_TIMEOUT, max(base_timeout, scaled))


def is_mechanical_patch_error(error: str) -> bool:
    """True when patch applier rejected the patch without changing the file."""
    if not error:
        return False
    return error.startswith(("mismatch:", "invalid_patch:"))


def _patch_retry_hint(error: str) -> str:
    if "overlaps another block" in error:
        return (
            "Each SEARCH/REPLACE block must target disjoint line ranges. "
            "Shrink SEARCH to 3–8 lines per site; never reuse the same lines in two blocks."
        )
    if error.startswith("mismatch:"):
        return (
            "SEARCH must be verbatim code from CURRENT_CONTEXT (before edit). "
            "Put new code only in REPLACE; split independent sites into non-overlapping blocks."
        )
    if "patch produces no change" in error:
        return "REPLACE must differ from SEARCH; ensure the patch actually edits the file."
    if "expected SEARCH/REPLACE blocks only" in error:
        return "Return only SEARCH/REPLACE marker blocks in patch; no prose or Markdown fences."
    return (
        "Regenerate action=edit with corrected SEARCH/REPLACE blocks using CURRENT_CONTEXT only."
    )


def build_patch_retry_state_text(
    base_state_text: str,
    *,
    attempt: int,
    max_attempts: int,
    error: str,
    failed_patch: str,
) -> str:
    """Augment CURRENT_STATE for an inner decision_edit retry."""
    patch_excerpt = failed_patch.strip()
    if len(patch_excerpt) > _MAX_FAILED_PATCH_FEEDBACK_CHARS:
        patch_excerpt = (
            patch_excerpt[:_MAX_FAILED_PATCH_FEEDBACK_CHARS]
            + "\n... (truncated failed patch)"
        )
    return (
        f"{base_state_text.strip()}\n\n"
        f"PATCH_RETRY_FEEDBACK (attempt {attempt + 1}/{max_attempts})\n"
        f"The previous patch failed to apply:\n```\n{error.strip()}\n```\n"
        f"Failed patch excerpt:\n```\n{patch_excerpt}\n```\n"
        f"Fix: {_patch_retry_hint(error)}\n"
        "Return a new action=edit JSON with corrected patch only."
    )


def _build_state_text(
    target_file: str,
    intent: str,
    focus_symbols: list[str],
) -> str:
    """Pass Core plan step through to Decision LLM without internal re-batching."""
    lines = [
        f"Apply the following modification intent to target file `{target_file}`:",
        intent.strip(),
    ]
    symbols = [str(symbol).strip() for symbol in focus_symbols if str(symbol).strip()]
    if symbols:
        lines.append(f"\nFocus symbols for this plan step: {', '.join(symbols)}")
    return "\n".join(lines)


class DecisionEditTool(Tool):
    name = "decision_edit"
    description = (
        "Apply edits to exactly one file when STEP EVIDENCE shows edit_ready:yes "
        "and the change is grounded in loaded code anchors. One call per plan step "
        "and target file: pass intent, focus_symbols, and context_window from Core; "
        "Decision LLM consumes them in a single patch generation and validate pass. "
        "Split multi-step work across multiple decision_edit calls at the Core/plan "
        "layer — do not expect internal symbol batching inside this tool. "
        "Do NOT use to inspect, grep, or load code."
    )
    risk_level = RiskLevel.MODERATE
    parameters = {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "目标文件的相对路径。"
            },
            "intent": {
                "type": "string",
                "description": (
                    "本步（当前 plan 条目）对目标文件的修改意图，只描述这一步要做什么，"
                    "不要把多步 plan 的其他步骤写进来。多步任务由 Core 拆成多次 decision_edit。"
                ),
            },
            "focus_symbols": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "本步要改的符号名（函数/类/helper）。列出当前 step 涉及的全部符号；"
                    "Decision LLM 一次消费，不在工具内再分批。"
                ),
            },
            "context_window": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "file": {"type": "string", "description": "参考文件的相对路径"},
                        "span": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "minItems": 2,
                            "maxItems": 2,
                            "description": (
                                "起止行号，如 [10, 120]。来自 LOADED CODE ANCHORS，"
                                "由 Core 按本步需要选择最小充分 span。"
                            ),
                        },
                        "reason": {"type": "string", "description": "该参考片段被纳为上下文的原因/用途说明"}
                    },
                    "required": ["file", "span"]
                },
                "description": (
                    "已冻结的证据片段（来自 LOADED CODE ANCHORS）。同文件相邻 span "
                    "会在工具内机械合并后交给 Decision LLM。"
                ),
            },
            "task_id": {
                "type": "string",
                "description": "可选的任务包/步骤 ID。"
            },
            "mode": {
                "type": "string",
                "enum": ["edit"],
                "description": "任务模式，固定为 'edit'。"
            },
            "constraints": {
                "type": "object",
                "properties": {
                    "no_further_retrieval": {"type": "boolean", "default": True},
                    "patch_scope": {"type": "string", "default": "single_file_only"}
                },
                "description": "任务执行约束条件。"
            }
        },
        "required": ["target_file", "intent", "context_window"]
    }

    def __init__(
        self,
        *,
        project_root: Path,
        settings: Any,
        decision_llm: Any,
        harness: Any,
    ) -> None:
        self.project_root = project_root.resolve()
        self.settings = settings
        self.decision_llm = decision_llm
        self.harness = harness
        configured_timeout = getattr(settings, "cursor_decision_timeout", 120.0)
        self.decision_timeout_base = (
            float(configured_timeout)
            if isinstance(configured_timeout, (int, float))
            else 120.0
        )
        configured_patch_retries = getattr(settings, "cursor_decision_patch_retries", 2)
        self.patch_retry_max = (
            int(configured_patch_retries)
            if isinstance(configured_patch_retries, int)
            else 2
        )
        self.patch_retry_max = max(0, min(self.patch_retry_max, 3))

        self.decision = CursorDecisionLLM(self.decision_llm)
        self.patch_applier = CursorPatchApplier(self.project_root)
        self.executor = CursorExecutor(self.project_root, self.patch_applier)

        validator_llm = self.decision_llm
        if not settings.cursor_validator_model or settings.cursor_validator_model.lower() in ("none", "disabled", "false"):
            validator_llm = None
        elif getattr(validator_llm, "model", "") != settings.cursor_validator_model:
            validator_llm = LLMClient(
                model=settings.cursor_validator_model,
                request_timeout=float(settings.llm_request_timeout),
                prompt_cache_enabled=settings.prompt_cache_enabled,
                prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
                prompt_cache_ttl=(
                    "1h"
                    if settings.prompt_cache_ttl.strip().lower() in {"1h", "hour", "60m"}
                    else "5m"
                ),
            )
        self.validator = CursorValidator(
            self.project_root,
            command=tuple(settings.cursor_validator_command),
            timeout=settings.cursor_validator_timeout,
            max_error_chars=settings.cursor_observation_max_chars,
            semantic_llm=validator_llm,
            semantic_timeout=settings.cursor_validator_semantic_timeout,
        )
        self.active_files: list[str] = []

    def set_active_files(self, active_files: list[str]) -> None:
        self.active_files = list(active_files)

    def validate_params(self, params: dict[str, Any]) -> dict[str, Any]:
        if "context_window" not in params:
            params = {**params, "context_window": []}
        if "focus_symbols" not in params:
            params = {**params, "focus_symbols": []}
        return super().validate_params(params)

    def _read_file_span(self, file_path: str, start_line: int, end_line: int) -> str | None:
        abs_path = (self.project_root / file_path).resolve()
        if not abs_path.is_file():
            return None
        try:
            lines = abs_path.read_text(encoding="utf-8").splitlines()
            start = max(1, start_line)
            end = min(len(lines), end_line)
            return "\n".join(lines[start - 1 : end])
        except Exception as exc:
            log.warning(
                "Failed to read span %s:%d-%d from disk: %s",
                file_path,
                start_line,
                end_line,
                exc,
            )
            return None

    def _build_context_pack(
        self,
        target_file: str,
        search_cache: dict[str, Any] | None = None,
        context_window: list[dict[str, Any]] | None = None,
    ) -> ContextPack:
        windows = []

        norm_target = _norm_file_path(target_file)
        target_spans: list[tuple[str, int, int]] = []
        reference_spans: list[tuple[str, int, int]] = []

        if context_window is not None:
            for item in context_window:
                file_path = item.get("file")
                span = item.get("span")
                if not file_path or not span or len(span) < 2:
                    continue
                norm_file = _norm_file_path(str(file_path))
                entry = (str(file_path), int(span[0]), int(span[1]))
                if norm_file == norm_target:
                    target_spans.append(entry)
                else:
                    reference_spans.append(entry)

            target_spans = merge_context_spans(target_spans)
            reference_spans = merge_context_spans(reference_spans)

        if target_spans:
            for index, (file_path, start_line, end_line) in enumerate(target_spans):
                content = self._read_file_span(file_path, start_line, end_line)
                if content is None:
                    continue
                windows.append(
                    ContextWindow(
                        file=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        symbols=(),
                        semantic_tags=(),
                        role="target" if index == 0 else "reference",
                        mode="snippet",
                    )
                )
        else:
            abs_target_path = (self.project_root / target_file).resolve()
            if abs_target_path.exists():
                try:
                    lines = abs_target_path.read_text(encoding="utf-8").splitlines()
                    content = "\n".join(lines)
                    windows.append(
                        ContextWindow(
                            file=target_file,
                            start_line=1,
                            end_line=len(lines),
                            content=content,
                            symbols=(),
                            semantic_tags=(),
                            role="target",
                            mode="full",
                        )
                    )
                except Exception as exc:
                    log.warning("Failed to read target file %s: %s", target_file, exc)

        for file_path, start_line, end_line in reference_spans:
            content = self._read_file_span(file_path, start_line, end_line)
            if content is None:
                continue
            windows.append(
                ContextWindow(
                    file=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    symbols=(),
                    semantic_tags=(),
                    role="reference",
                    mode="snippet",
                )
            )

        if context_window is None:
            # Fall back to historical raw_evidence_store context loading (legacy behavior)
            raw_evidence = []
            if search_cache and "raw_evidence_store" in search_cache:
                raw_evidence = search_cache["raw_evidence_store"]

            for item in raw_evidence:
                file_path = item["file"]
                if file_path == target_file:
                    continue

                abs_path = (self.project_root / file_path).resolve()
                start_line, end_line = item["span"]

                content = None
                if abs_path.is_file():
                    try:
                        lines = abs_path.read_text(encoding="utf-8").splitlines()
                        start = max(1, start_line)
                        end = min(len(lines), end_line)
                        content = "\n".join(lines[start - 1 : end])
                    except Exception as exc:
                        log.warning("Failed to read span %s:%d-%d from disk: %s", file_path, start_line, end_line, exc)

                if content is None:
                    content = item["code"]

                windows.append(
                    ContextWindow(
                        file=file_path,
                        start_line=start_line,
                        end_line=end_line,
                        content=content,
                        symbols=(),
                        semantic_tags=(),
                        role="reference",
                        mode="snippet",
                    )
                )

        return ContextPack(windows=tuple(windows))

    @staticmethod
    def _build_evidence_flag(
        target_file: str,
        search_cache: dict[str, Any] | None,
        context_pack: ContextPack,
        *,
        context_window: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        norm_target = _norm_file_path(target_file)
        if context_window:
            frozen_spans: list[dict[str, Any]] = []
            for item in context_window:
                if not isinstance(item, dict):
                    continue
                file_path = str(item.get("file") or "")
                span = item.get("span")
                if not file_path or not isinstance(span, list) or len(span) < 2:
                    continue
                frozen = {
                    "file": file_path,
                    "span": [int(span[0]), int(span[1])],
                    "status": "frozen",
                }
                reason = item.get("reason")
                if isinstance(reason, str) and reason.strip():
                    frozen["reason"] = reason.strip()[:160]
                frozen_spans.append(frozen)
            return {
                "target_file": target_file,
                "context_mode": "frozen_context_window",
                "context_window_spans": frozen_spans,
                "available_context_files": list(context_pack.candidate_files),
                "can_edit": bool(context_pack.windows),
            }

        anchors: list[dict[str, Any]] = []
        if search_cache:
            raw = search_cache.get("raw_evidence_store") or []
            if isinstance(raw, list):
                anchors.extend(item for item in raw if isinstance(item, dict))
            encoded = search_cache.get("search_output")
            if isinstance(encoded, str) and encoded.strip():
                try:
                    decoded = json.loads(encoded)
                    if isinstance(decoded, list):
                        anchors.extend(item for item in decoded if isinstance(item, dict))
                except (TypeError, json.JSONDecodeError) as exc:
                    log.warning("Failed to parse code-anchor search_output: %s", exc)

        unique: dict[tuple[str, tuple[Any, ...]], dict[str, Any]] = {}
        for anchor in anchors:
            file_path = str(anchor.get("file") or "")
            span = tuple(anchor.get("span") or ())
            if file_path and len(span) == 2:
                unique[(file_path, span)] = anchor

        target_anchors = []
        reference_anchors = []
        related: dict[tuple[str, str, tuple[Any, ...]], dict[str, Any]] = {}
        for (file_path, span), anchor in unique.items():
            compact = {"file": file_path, "span": list(span), "status": "loaded"}
            if anchor.get("symbol"):
                compact["symbol"] = anchor["symbol"]
            (target_anchors if _norm_file_path(file_path) == norm_target else reference_anchors).append(compact)
            for function in anchor.get("related_functions") or []:
                if not isinstance(function, dict) or not function.get("name"):
                    continue
                function_span = tuple(function.get("span") or ())
                key = (str(function["name"]), str(function.get("file") or ""), function_span)
                related[key] = {
                    "name": function["name"],
                    "file": function.get("file"),
                    "span": list(function_span),
                }

        return {
            "target_file": target_file,
            "target_code_anchors": target_anchors,
            "reference_code_anchors": reference_anchors,
            "first_hop_functions": list(related.values()),
            "available_context_files": list(context_pack.candidate_files),
            "can_edit": bool(context_pack.windows),
        }

    async def _generate_patch(
        self,
        *,
        target_file: str,
        state_text: str,
        context_pack: ContextPack,
        evidence_flag: dict[str, Any],
        effective_timeout: float,
        started_at: float,
        batch_label: str = "",
    ) -> tuple[Any, Any]:
        """Call Decision LLM and return (parsed_decision, response)."""
        decision_messages = self.decision.build_messages(
            state_text=state_text,
            context_pack=context_pack,
            hint=None,
            evidence_flag=evidence_flag,
            edit_only=True,
        )
        trimmed_messages = await self.harness.before_llm_call(decision_messages)

        connect_timeout = effective_timeout
        stream_idle_timeout = float(
            getattr(self.decision_llm, "stream_idle_timeout", 60) or 60
        )
        use_stream = hasattr(self.decision_llm, "chat_stream")
        response = None
        content_chunks: list[str] = []

        def count_diff_lines(text: str) -> tuple[int, int]:
            normalized = text.replace("\\n", "\n")
            added = 0
            removed = 0
            state = "normal"
            for line in normalized.splitlines():
                clean_line = line.strip().strip('"\'')
                if "<<<<<<< SEARCH" in clean_line:
                    state = "search"
                elif "=======" in clean_line:
                    state = "replace"
                elif ">>>>>>> REPLACE" in clean_line:
                    state = "normal"
                else:
                    if state == "search":
                        removed += 1
                    elif state == "replace":
                        added += 1
            return added, removed

        label = f" {batch_label}" if batch_label else ""
        print(
            "[debug][decision-edit][decision-llm-start]"
            f"{label} file={target_file} connect_timeout={connect_timeout:g}s"
            f" idle_timeout={stream_idle_timeout:g}s",
            flush=True,
        )
        first_chunk_at: float | None = None

        async def _consume_stream() -> None:
            nonlocal response, first_chunk_at
            if (
                hasattr(self.harness, "progress_callback")
                and self.harness.progress_callback
            ):
                self.harness.progress_callback(f"正在编辑文件: {target_file}… [+0 -0]")

            stream_iter = self.decision_llm.chat_stream(
                trimmed_messages,
                tools=None,
                timeout=connect_timeout,
            )
            async for content_chunk, final_response in stream_iter:
                if content_chunk and first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                    print(
                        "[debug][decision-edit][decision-llm-ttft]"
                        f"{label} file={target_file} "
                        f"ttft={first_chunk_at - started_at:.2f}s",
                        flush=True,
                    )
                if content_chunk:
                    content_chunks.append(content_chunk)
                    added, removed = count_diff_lines("".join(content_chunks))
                    if (
                        hasattr(self.harness, "progress_callback")
                        and self.harness.progress_callback
                    ):
                        self.harness.progress_callback(
                            f"正在编辑文件: {target_file}… [+{added} -{removed}]"
                        )
                if final_response is not None:
                    if getattr(final_response, "model", "") == "error":
                        raise TimeoutError(final_response.content or "LLM stream timed out")
                    response = final_response

        if use_stream:
            probe = self.decision_llm.chat_stream(
                trimmed_messages,
                tools=None,
                timeout=connect_timeout,
            )
            if inspect.iscoroutine(probe):
                probe.close()
                use_stream = False

        if use_stream:
            await _consume_stream()
        else:
            async with asyncio.timeout(connect_timeout):
                response = await self.decision_llm.chat(
                    trimmed_messages,
                    tools=None,
                    stream=False,
                    timeout=connect_timeout,
                )

        if response is None and content_chunks:
            from src.agent.types import LLMResponse

            response = LLMResponse(
                content="".join(content_chunks),
                tool_calls=None,
                usage=None,
                model=getattr(self.decision_llm, "model", "unknown"),
            )

        if first_chunk_at is None and use_stream:
            print(
                "[debug][decision-edit][decision-llm-ttft]"
                f"{label} file={target_file} ttft=none (no stream chunks before deadline)",
                flush=True,
            )

        print(
            "[debug][decision-edit][decision-llm-done]"
            f"{label} file={target_file} elapsed={time.monotonic() - started_at:.2f}s",
            flush=True,
        )

        if response:
            await self.harness.after_llm_call(response, response.usage)

        session_id = getattr(self.harness, "session_id", "default_session")
        subtask_id = f"step_{getattr(self.harness, 'current_step', 1)}_edit"
        if hasattr(self.harness, "session_storage"):
            for msg in trimmed_messages:
                self.harness.session_storage.append_sidechain_message(
                    session_id, subtask_id, msg.get("role", "system"), msg.get("content", "")
                )
            if response:
                self.harness.session_storage.append_sidechain_message(
                    session_id, subtask_id, "assistant", response.content or ""
                )

        parsed_decision = self.decision.parse(
            response.content or "",
            context_pack.candidate_files,
            edit_only=True,
        )
        return parsed_decision, response

    async def execute(self, **params: Any) -> ToolResult:
        started_at = time.monotonic()
        validated = self.validate_params(params)
        target_file = validated["target_file"]
        intent = validated["intent"]
        focus_symbols = list(validated.get("focus_symbols") or [])
        search_cache = params.get("_search_cache")
        context_window = params.get("context_window")

        context_pack = self._build_context_pack(
            target_file,
            search_cache,
            context_window=context_window,
        )
        context_chars = sum(len(window.content) for window in context_pack.windows)
        context_lines = sum(
            len(window.content.splitlines()) for window in context_pack.windows
        )
        span_count = len(context_window or [])
        effective_timeout = decision_timeout_for_context(
            self.decision_timeout_base,
            context_chars=context_chars,
            span_count=span_count,
        )
        print(
            "[debug][decision-edit][context] "
            f"file={target_file} windows={len(context_pack.windows)} "
            f"lines={context_lines} chars={context_chars} spans={span_count} "
            f"focus_symbols={len(focus_symbols)} "
            f"connect_timeout={effective_timeout:.0f}s "
            f"elapsed={time.monotonic() - started_at:.2f}s",
            flush=True,
        )

        evidence_flag = self._build_evidence_flag(
            target_file,
            search_cache,
            context_pack,
            context_window=context_window,
        )
        state_text = _build_state_text(target_file, intent, focus_symbols)
        max_attempts = self.patch_retry_max + 1
        feedback_state = state_text
        parsed_decision = None
        execution = None
        validation = None
        pipeline_metrics = None
        patch_attempt = 0

        while patch_attempt < max_attempts:
            patch_attempt += 1
            try:
                parsed_decision, _response = await self._generate_patch(
                    target_file=target_file,
                    state_text=feedback_state,
                    context_pack=context_pack,
                    evidence_flag=evidence_flag,
                    effective_timeout=effective_timeout,
                    started_at=started_at,
                    batch_label=f"attempt={patch_attempt}",
                )
            except TimeoutError:
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        "DecisionLLM patch generation timed out after "
                        f"{effective_timeout:g}s for {target_file}."
                    ),
                    metadata={"patch_attempts": patch_attempt},
                )
            except Exception as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"DecisionLLM generation/parsing failed: {exc}",
                    metadata={"patch_attempts": patch_attempt},
                )

            if parsed_decision.action != "edit":
                return ToolResult(
                    success=False,
                    output=(
                        "Decision LLM did not generate an edit. "
                        f"Action taken: {parsed_decision.action}. "
                        f"Message: {parsed_decision.answer or parsed_decision.clarification}"
                    ),
                    error=f"Action is not edit: {parsed_decision.action}",
                    metadata={"patch_attempts": patch_attempt},
                )

            execution, validation, pipeline_metrics = await self.executor.execute_transaction(
                parsed_decision.target_file,
                parsed_decision.patch,
                self.validator,
                step=1,
                layer1=None,
                user_intent=intent,
            )

            print(
                "[debug][decision-edit][validation-done] "
                f"file={target_file} success={validation.success} "
                f"patch_attempt={patch_attempt}/{max_attempts} "
                f"apply_success={execution.success} "
                f"elapsed={time.monotonic() - started_at:.2f}s",
                flush=True,
            )

            if execution.success:
                break

            error_detail = execution.error or ""
            can_retry = (
                patch_attempt < max_attempts
                and is_mechanical_patch_error(error_detail)
            )
            if can_retry:
                print(
                    "[debug][decision-edit][patch-retry] "
                    f"file={target_file} attempt={patch_attempt}/{max_attempts} "
                    f"error={error_detail.splitlines()[0][:120]}",
                    flush=True,
                )
                feedback_state = build_patch_retry_state_text(
                    state_text,
                    attempt=patch_attempt,
                    max_attempts=max_attempts,
                    error=error_detail,
                    failed_patch=parsed_decision.patch,
                )
                continue

            task_id = params.get("task_id")
            task_label = f" 任务 `{task_id}`" if task_id else ""
            retry_hint = ""
            if error_detail.startswith("mismatch:"):
                retry_hint = (
                    "\n\n重试建议：① SEARCH 必须是文件中**现有**代码的逐字复制，"
                    "REPLACE 才是改后内容；② 同一文件多处修改请拆成多个互不重叠的 "
                    "SEARCH/REPLACE 块；③ 对照 Diagnostic 核对缩进与引号；"
                    "④ 缩小本步 focus_symbols 或拆成下一次 decision_edit plan step。"
                )
            elif "overlaps another block" in error_detail:
                retry_hint = (
                    "\n\n重试建议：① 每个 SEARCH/REPLACE 块的行范围必须互不重叠；"
                    "② 每块只包 3–8 行定位代码；③ 若改动点过多，请让 Core 拆成多次 decision_edit。"
                )
            inner_note = ""
            if patch_attempt > 1:
                inner_note = (
                    f"\n（Decision LLM 内层已重试 {patch_attempt - 1} 次，"
                    "仍无法应用补丁。）"
                )
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【补丁生成失败】：{task_label}对文件 `{target_file}` 的补丁无法匹配到物理内容。"
                    f"{inner_note}\n"
                    f"具体错误与诊断：\n```\n{error_detail}\n```"
                    f"{retry_hint}"
                ),
                error=error_detail,
                metadata={
                    "pipeline_metrics": pipeline_metrics,
                    "execution": execution,
                    "validation": validation,
                    "patch_attempts": patch_attempt,
                },
            )

        task_id = params.get("task_id")
        task_label = f" 任务 `{task_id}`" if task_id else ""

        if execution is None or validation is None:
            return ToolResult(
                success=False,
                output="",
                error="decision_edit: no patch execution result",
                metadata={"patch_attempts": patch_attempt},
            )

        if not validation.success:
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【自动化代码验证失败】：{task_label}对文件 `{target_file}` 的补丁导致编译、语法或测试错误。\n"
                    f"👉 已经自动将修改全部物理回滚（Rollback）。\n"
                    f"👉 具体的验证器报错如下，请根据此错误重新构思修改指令：\n"
                    f"```\n{validation.error}\n```"
                ),
                error=validation.error,
                metadata={
                    "pipeline_metrics": pipeline_metrics,
                    "execution": execution,
                    "validation": validation,
                    "patch_attempts": patch_attempt,
                },
            )

        return ToolResult(
            success=True,
            output=(
                f"✅ 【应用成功】：针对文件 `{target_file}`{task_label} "
                f"的补丁已通过编译与自动化测试验证，修改已持久化提交（Committed）。"
            ),
            metadata={
                "pipeline_metrics": pipeline_metrics,
                "execution": execution,
                "validation": validation,
                "patch_attempts": patch_attempt,
            },
        )
