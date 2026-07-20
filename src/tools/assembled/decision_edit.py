from __future__ import annotations

import asyncio
import inspect
import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from src.agent.contracts import ContextPack, ContextWindow
from src.agent.decision import CursorDecisionLLM, DecisionError
from src.agent.edit_errors import (
    EditErrorClass,
    RetryOwner,
    classify_edit_error,
    core_hint_for,
    edit_inner_retry_allowed,
    retry_owner_for,
)
from src.agent.edit_brief import format_applied_diff_summary
from src.agent.edit_materialize import (
    MaterializeError,
    materialize_edit_patch,
    sanitize_focus_symbols,
)
from src.agent.edit_plan import (
    MAX_PATCH_BLOCKS_PER_MINOR_STEP,
    PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP,
)
from src.agent.executor import CursorExecutor
from src.agent.patch_applier import CursorPatchApplier
from src.agent.types import RiskLevel, ToolResult
from src.agent.validator import CursorValidator
from src.indexer.project_stack import detect_project_stack
from src.llm.client import LLMClient
from src.tools.base import Tool

log = logging.getLogger(__name__)

_SPAN_MERGE_GAP = 5
_CHARS_PER_TIMEOUT_SECOND = 800.0
_TIMEOUT_PER_CONTEXT_SPAN = 12.0
_MAX_DECISION_TIMEOUT = 300.0
_MAX_FAILED_PATCH_FEEDBACK_CHARS = 6000
_SITE_SPAN_RE = re.compile(
    r"(?im)\bspan\s*=\s*(\d+)\s*[-:]\s*(\d+)"
)


def _norm_file_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def count_stream_diff_lines(text: str) -> tuple[int, int]:
    """Estimate ``[+added -removed]`` while EditLLM streams a patch.

    Supports:
    - Legacy SEARCH/REPLACE (count SEARCH lines as removed, REPLACE as added)
    - SITE + REPLACE-only (count REPLACE body as added; estimate removed from
      ``SITE: span=A-B`` when present — harness fills SEARCH after stream ends)
    """
    normalized = (text or "").replace("\\n", "\n")
    added = 0
    removed = 0
    state = "normal"
    saw_search = False
    for line in normalized.splitlines():
        clean_line = line.strip().strip("\"'")
        if "<<<<<<< SEARCH" in clean_line:
            state = "search"
            saw_search = True
        elif "<<<<<<< REPLACE" in clean_line:
            state = "replace"
        elif state == "search" and "=======" in clean_line:
            state = "replace"
        elif ">>>>>>> REPLACE" in clean_line:
            state = "normal"
        elif state == "search":
            removed += 1
        elif state == "replace":
            added += 1

    # SITE+REPLACE: SEARCH is synthesized later; approximate removals from span=.
    if not saw_search and removed == 0:
        for match in _SITE_SPAN_RE.finditer(normalized):
            start, end = int(match.group(1)), int(match.group(2))
            if end >= start:
                removed += end - start + 1
    return added, removed


def format_edit_progress(target_file: str, added: int, removed: int) -> str:
    """Rich markup for the decision_edit streaming status line."""
    return (
        f"正在编辑文件: [bold blue]{target_file}[/]… "
        f"[bold blue][+{added} -{removed}][/]"
    )


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
    """Absolute first-*content* deadline for decision_edit streaming.

    Scales lightly with context size. After the first content chunk, only the
    LLM client's per-chunk idle timeout applies — not this value as a total
    wall-clock cap for generation.
    """
    # Cap additive budget so small contexts are not misread as "must wait 160s+".
    char_budget = min(45.0, max(context_chars, 0) / _CHARS_PER_TIMEOUT_SECOND)
    span_budget = min(36.0, max(span_count, 0) * _TIMEOUT_PER_CONTEXT_SPAN)
    scaled = base_timeout + char_budget + span_budget
    return min(_MAX_DECISION_TIMEOUT, max(base_timeout, scaled))


def is_mechanical_patch_error(error: str) -> bool:
    """True when patch applier rejected the patch without changing the file."""
    klass = classify_edit_error(error, apply_succeeded=False)
    return klass in {EditErrorClass.E1_FORMAT, EditErrorClass.E2_LOCATE}


def is_format_validation_retry(error: str, attempted_content: str = "") -> bool:
    """True when validator failure is likely a patch-format corruption (Edit retry)."""
    klass = classify_edit_error(
        error,
        attempted_content=attempted_content,
        apply_succeeded=True,
    )
    return klass is EditErrorClass.E1_FORMAT


def _patch_retry_hint(error: str) -> str:
    klass = classify_edit_error(error, apply_succeeded=False)
    if klass is EditErrorClass.E1_FORMAT:
        if "decision_schema" in error or "invalid JSON" in error:
            return (
                "E1_FORMAT: return ACTION: edit with SITE + REPLACE (header-block). "
                "Do not wrap the patch in JSON. "
                f"Prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP} SITE "
                f"(max {MAX_PATCH_BLOCKS_PER_MINOR_STEP})."
            )
        if "SITE:" in error or "expected SITE" in error or "empty patch" in error:
            return (
                "E1_FORMAT: emit `SITE: symbol=<name>` then "
                "`<<<<<<< REPLACE` … `>>>>>>> REPLACE`. Do NOT emit SEARCH — "
                "harness fills SEARCH from disk. "
                f"Prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP} SITE "
                f"(max {MAX_PATCH_BLOCKS_PER_MINOR_STEP})."
            )
        if "nested SEARCH/REPLACE markers" in error or "stray ======= marker" in error:
            return (
                "E1_FORMAT: separate each SITE with a newline after >>>>>>> REPLACE "
                "before the next SITE. Never embed markers in code bodies. "
                f"Prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP} SITE "
                f"(max {MAX_PATCH_BLOCKS_PER_MINOR_STEP}) per 小步."
            )
        if "too many SEARCH/REPLACE blocks" in error or "too many SITE" in error:
            return (
                f"E1_FORMAT: at most {MAX_PATCH_BLOCKS_PER_MINOR_STEP} SITE blocks per 小步 "
                f"(prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP}). "
                "Enqueue remaining sites as later 小步."
            )
        if "patch produces no change" in error or "REPLACE equals on-disk" in error:
            return (
                "E1_FORMAT: your REPLACE echoed the on-disk SITE (no edit). "
                "REPLACE must be the AFTER-edit text with CURRENT_STATE intent "
                "applied — full symbol/span body, not a copy of CURRENT_CONTEXT. "
                "Do not repeat the previous identical REPLACE."
            )
        if "expected SEARCH/REPLACE blocks only" in error:
            return (
                "E1_FORMAT: return SITE + REPLACE blocks (or legacy SEARCH/REPLACE); "
                "no prose/fences."
            )
        return (
            "E1_FORMAT: regenerate SITE + REPLACE for this 小步 only "
            f"(prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP}, "
            f"max {MAX_PATCH_BLOCKS_PER_MINOR_STEP} SITE blocks). Do not emit SEARCH."
        )
    if klass is EditErrorClass.E2_LOCATE:
        if "overlaps another block" in error:
            return (
                "E2_LOCATE: SITE spans must be disjoint; one symbol/span per SITE. "
                "If sites >3, leave extras for later 小步."
            )
        return (
            "E2_LOCATE: fix SITE to an on-disk symbol from Did you mean… / "
            "valid_focus_symbols (ignore invented Core focus names). "
            "New helpers: insert via MODE+ANCHOR under an existing symbol — "
            "do not SITE a name that is not on disk yet. "
            "Do NOT emit SEARCH — harness copies on-disk text. "
            f"Prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP} SITE "
            f"(max {MAX_PATCH_BLOCKS_PER_MINOR_STEP}); if still failing Core will replan."
        )
    return (
        "Regenerate ACTION: edit with corrected SITE + REPLACE for this 小步 only. "
        f"Prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP} SITE "
        f"(max {MAX_PATCH_BLOCKS_PER_MINOR_STEP}). Do not emit SEARCH."
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
    error_text = (error or "").strip()
    # Replaying an echoed REPLACE teaches the model to emit the same no-op again.
    noop_echo = (
        "REPLACE equals on-disk" in error_text
        or "patch produces no change" in error_text
    )
    if noop_echo:
        patch_section = (
            "Do NOT copy the previous REPLACE body — it was identical to disk.\n"
            "Re-read CURRENT_STATE intent and emit the changed AFTER-edit text.\n"
        )
    else:
        patch_excerpt = failed_patch.strip()
        if len(patch_excerpt) > _MAX_FAILED_PATCH_FEEDBACK_CHARS:
            patch_excerpt = (
                patch_excerpt[:_MAX_FAILED_PATCH_FEEDBACK_CHARS]
                + "\n... (truncated failed patch)"
            )
        patch_section = f"Failed patch excerpt:\n```\n{patch_excerpt}\n```\n"
    return (
        f"{base_state_text.strip()}\n\n"
        f"PATCH_RETRY_FEEDBACK (attempt {attempt + 1}/{max_attempts})\n"
        f"The previous patch failed to apply:\n```\n{error_text}\n```\n"
        f"{patch_section}"
        f"Fix: {_patch_retry_hint(error_text)}\n"
        "Return ACTION: edit with corrected SITE (+ MODE/ANCHOR delta preferred) "
        "(header-block format; do NOT emit SEARCH; do NOT wrap in JSON)."
    )


def _build_state_text(
    target_file: str,
    intent: str,
    focus_symbols: list[str],
    *,
    dropped_focus: list[str] | None = None,
    remapped_focus: dict[str, str] | None = None,
) -> str:
    """Pass Core plan step through to Decision LLM without internal re-batching."""
    lines = [
        f"Apply the following modification intent to target file `{target_file}`:",
        intent.strip(),
    ]
    symbols = [str(symbol).strip() for symbol in focus_symbols if str(symbol).strip()]
    if symbols:
        lines.append(f"\nFocus symbols for this plan step: {', '.join(symbols)}")
    if remapped_focus:
        bits = [f"{src}→{dst}" for src, dst in remapped_focus.items()]
        lines.append(
            "Core focus typo remapped to on-disk names: " + ", ".join(bits)
        )
    if dropped_focus:
        lines.append(
            "Dropped unknown focus_symbols (not on disk — do not SITE them; "
            "insert new helpers via MODE+ANCHOR under an existing symbol): "
            + ", ".join(dropped_focus)
        )
    return "\n".join(lines)


class DecisionEditTool(Tool):
    name = "decision_edit"
    description = (
        "Apply one 小步 edit to exactly one file when edit_ready:yes. "
        "A 小步 is one decision_edit job from the Core edit_queue (one target_file; "
        f"prefer {PREFERRED_PATCH_BLOCKS_PER_MINOR_STEP} SITE/symbol edit, "
        f"hard max {MAX_PATCH_BLOCKS_PER_MINOR_STEP}). "
        "EditLLM emits SITE+REPLACE only; harness fills SEARCH from disk. "
        "大步 are checklist outcomes; Core expands each 大步 into multiple 小步. "
        "Mechanical failures retry inside Edit (E1/E2); exhausted or E4/E5/E6 return "
        "to Core. Do NOT use to inspect, grep, or load code."
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
                    "本步要改的、磁盘上已存在的符号名（1–3 个，必填）。"
                    "必须来自 STEP EVIDENCE available_symbols / loaded anchors。"
                    "不要把尚未写入的新函数名放进来；新符号由 EditLLM 用 MODE+ANCHOR 插入。"
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
        "required": ["target_file", "intent", "focus_symbols", "context_window"]
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
        configured_patch_retries = getattr(settings, "cursor_decision_patch_retries", 1)
        self.patch_retry_max = (
            int(configured_patch_retries)
            if isinstance(configured_patch_retries, int)
            else 1
        )
        self.patch_retry_max = max(0, min(self.patch_retry_max, 2))

        configured_max_blocks = getattr(settings, "cursor_decision_max_patch_blocks", 3)
        max_blocks = (
            int(configured_max_blocks)
            if isinstance(configured_max_blocks, int)
            else 3
        )
        sequential = bool(getattr(settings, "edit_sequential_patch", False))

        self.decision = CursorDecisionLLM(self.decision_llm)
        self.patch_applier = CursorPatchApplier(
            self.project_root,
            sequential=sequential,
            max_blocks=max_blocks,
        )
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
            stack=detect_project_stack(self.project_root),
            per_file_commands=settings.cursor_validator_auto,
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

    def _settings_int(self, name: str, default: int) -> int:
        """Read a positive int setting; ignore MagicMock / non-numeric stand-ins.

        ``unittest.mock.MagicMock.__int__`` returns 1, so never call ``int()``
        on arbitrary attribute objects.
        """
        value = getattr(self.settings, name, default)
        if isinstance(value, bool):
            return default
        if isinstance(value, int):
            return value if value > 0 else default
        if isinstance(value, float):
            parsed = int(value)
            return parsed if parsed > 0 else default
        if isinstance(value, str) and value.strip():
            try:
                parsed = int(value.strip(), 10)
            except ValueError:
                return default
            return parsed if parsed > 0 else default
        return default

    def _edit_context_budgets(self) -> tuple[int, int, int]:
        """Return (max_windows, chars_per_window, total_chars) for Edit LLM packs.

        Settings ``cursor_max_context_files`` / ``cursor_context_chars_per_file``
        were previously unused here, which let whole files (10k+ tokens) leak in.
        Edit 小步 packs are capped tighter than retrieve.
        """
        max_files = self._settings_int("cursor_max_context_files", 3)
        chars_per = self._settings_int("cursor_context_chars_per_file", 12_000)
        max_windows = max(1, min(max_files, 3))
        # Hard ceiling per window for Decision/Edit (retrieve may stay larger).
        chars_per_window = max(1_500, min(chars_per, 6_000))
        total_chars = max(chars_per_window, min(chars_per_window * max_windows, 12_000))
        return max_windows, chars_per_window, total_chars

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

    def _clip_window_content(
        self,
        content: str,
        *,
        start_line: int,
        max_chars: int,
    ) -> tuple[str, int]:
        """Clip content to max_chars; return (text, end_line_inclusive)."""
        if len(content) <= max_chars:
            return content, start_line + max(content.count("\n"), 0)
        clipped = content[:max_chars]
        # Prefer cutting on a line boundary.
        last_nl = clipped.rfind("\n")
        if last_nl > max_chars // 2:
            clipped = clipped[:last_nl]
        end_line = start_line + clipped.count("\n")
        return clipped + "\n… (truncated for Edit LLM context budget)", end_line

    def _spans_from_focus_symbols(
        self,
        target_file: str,
        focus_symbols: list[str],
        *,
        max_chars: int,
    ) -> list[tuple[str, int, int]]:
        """Best-effort symbol windows when Core omitted / undersized context_window."""
        if not focus_symbols:
            return []
        from src.agent.edit_materialize import locate_symbol_span

        spans: list[tuple[str, int, int]] = []
        window_lines = max(40, min(200, max_chars // 40))
        for symbol in focus_symbols[:3]:
            name = str(symbol).strip().split(".")[-1]
            if not name:
                continue
            located = locate_symbol_span(self.project_root, target_file, name)
            if located is not None:
                start, end = located
                # Pad a few lines after for insert_after ANCHOR visibility.
                abs_path = (self.project_root / target_file).resolve()
                try:
                    n_lines = len(abs_path.read_text(encoding="utf-8").splitlines())
                except OSError:
                    n_lines = end
                end = min(n_lines, max(end, start + window_lines - 1, end + 8))
                spans.append((target_file, max(1, start - 2), end))
                continue
            # Fallback: regex line scan (non-Python / odd defs).
            abs_path = (self.project_root / target_file).resolve()
            if not abs_path.is_file():
                continue
            try:
                lines = abs_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            hit = -1
            for index, line in enumerate(lines):
                stripped = line.strip()
                if (
                    stripped.startswith(f"def {name}")
                    or stripped.startswith(f"async def {name}")
                    or stripped.startswith(f"class {name}")
                ):
                    hit = index
                    break
            if hit < 0:
                continue
            start = max(1, hit + 1 - 2)
            end = min(len(lines), hit + window_lines)
            spans.append((target_file, start, end))
        return merge_context_spans(spans)

    def _spans_from_search_cache(
        self,
        target_file: str,
        search_cache: dict[str, Any] | None,
    ) -> list[tuple[str, int, int]]:
        if not search_cache:
            return []
        norm_target = _norm_file_path(target_file)
        spans: list[tuple[str, int, int]] = []
        for item in search_cache.get("raw_evidence_store") or []:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file") or "")
            span = item.get("span")
            if (
                _norm_file_path(file_path) == norm_target
                and isinstance(span, (list, tuple))
                and len(span) >= 2
            ):
                spans.append((file_path or target_file, int(span[0]), int(span[1])))
        for item in search_cache.get("symbol_projections") or []:
            if not isinstance(item, dict):
                continue
            file_path = str(item.get("file") or "")
            span = item.get("span")
            if (
                _norm_file_path(file_path) == norm_target
                and isinstance(span, (list, tuple))
                and len(span) >= 2
            ):
                spans.append((file_path or target_file, int(span[0]), int(span[1])))
        return merge_context_spans(spans)

    def _build_context_pack(
        self,
        target_file: str,
        search_cache: dict[str, Any] | None = None,
        context_window: list[dict[str, Any]] | None = None,
        focus_symbols: list[str] | None = None,
    ) -> ContextPack:
        max_windows, chars_per_window, total_chars = self._edit_context_budgets()
        windows: list[ContextWindow] = []

        norm_target = _norm_file_path(target_file)
        target_spans: list[tuple[str, int, int]] = []
        reference_spans: list[tuple[str, int, int]] = []

        # validate_params defaults missing context_window to [] — treat empty
        # the same as absent so we do not fall through to whole-file dumps.
        effective_window = context_window if context_window else None

        if effective_window is not None:
            for item in effective_window:
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

        # Expand undersized Core spans to full focus-symbol bodies. A 2-line
        # insert anchor caused EditLLM to delete strip_chat_noise while adding
        # is_bot_mentioned — it never saw the surrounding function.
        focus_spans = self._spans_from_focus_symbols(
            target_file,
            list(focus_symbols or []),
            max_chars=chars_per_window,
        )
        if focus_spans:
            from src.agent.edit_brief import MIN_FOCUS_CONTEXT_LINES

            expanded: list[tuple[str, int, int]] = []
            for file_path, start_line, end_line in target_spans:
                span_lines = end_line - start_line + 1
                if (
                    _norm_file_path(file_path) == norm_target
                    and span_lines < MIN_FOCUS_CONTEXT_LINES
                ):
                    # Prefer the focus symbol span that covers / is nearest.
                    best = focus_spans[0]
                    for cand in focus_spans:
                        c_start, c_end = cand[1], cand[2]
                        if c_start <= start_line <= c_end or start_line <= c_start <= end_line:
                            best = cand
                            break
                    # Union Core tip with full symbol (+ pad for insert_after).
                    abs_path = (self.project_root / target_file).resolve()
                    try:
                        n_lines = len(abs_path.read_text(encoding="utf-8").splitlines())
                    except OSError:
                        n_lines = best[2]
                    expanded.append(
                        (
                            target_file,
                            min(start_line, best[1]),
                            min(n_lines, max(end_line, best[2] + 8)),
                        )
                    )
                else:
                    expanded.append((file_path, start_line, end_line))
            if expanded:
                target_spans = merge_context_spans(expanded)

        if not target_spans:
            target_spans = focus_spans
        if not target_spans:
            target_spans = self._spans_from_search_cache(target_file, search_cache)

        def _append_window(
            *,
            file_path: str,
            start_line: int,
            end_line: int,
            content: str,
            role: str,
            mode: str,
        ) -> bool:
            if len(windows) >= max_windows:
                return False
            used = sum(len(window.content) for window in windows)
            budget = min(chars_per_window, total_chars - used)
            if budget < 200:
                return False
            clipped, clipped_end = self._clip_window_content(
                content, start_line=start_line, max_chars=budget
            )
            windows.append(
                ContextWindow(
                    file=file_path,
                    start_line=start_line,
                    end_line=min(end_line, clipped_end),
                    content=clipped,
                    symbols=(),
                    semantic_tags=(),
                    role=role,
                    mode=mode,
                )
            )
            return True

        if target_spans:
            for index, (file_path, start_line, end_line) in enumerate(target_spans):
                content = self._read_file_span(file_path, start_line, end_line)
                if content is None:
                    continue
                if not _append_window(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    role="target" if index == 0 else "reference",
                    mode="snippet",
                ):
                    break
        else:
            # Last resort: capped file head — never ship unbounded mode=full.
            abs_target_path = (self.project_root / target_file).resolve()
            if abs_target_path.exists():
                try:
                    lines = abs_target_path.read_text(encoding="utf-8").splitlines()
                    # ~chars_per_window lines of head only.
                    max_lines = max(40, min(len(lines), chars_per_window // 40))
                    content = "\n".join(lines[:max_lines])
                    _append_window(
                        file_path=target_file,
                        start_line=1,
                        end_line=max_lines,
                        content=content,
                        role="target",
                        mode="snippet",
                    )
                except Exception as exc:
                    log.warning("Failed to read target file %s: %s", target_file, exc)

        for file_path, start_line, end_line in reference_spans:
            content = self._read_file_span(file_path, start_line, end_line)
            if content is None:
                continue
            if not _append_window(
                file_path=file_path,
                start_line=start_line,
                end_line=end_line,
                content=content,
                role="reference",
                mode="snippet",
            ):
                break

        if effective_window is None and len(windows) < max_windows:
            # Bounded legacy evidence — only when Core did not freeze spans.
            raw_evidence = []
            if search_cache and "raw_evidence_store" in search_cache:
                raw_evidence = search_cache["raw_evidence_store"]

            for item in raw_evidence:
                if not isinstance(item, dict):
                    continue
                file_path = str(item.get("file") or "")
                span = item.get("span")
                if not file_path or file_path == target_file:
                    continue
                if not isinstance(span, (list, tuple)) or len(span) < 2:
                    continue
                start_line, end_line = int(span[0]), int(span[1])
                content = self._read_file_span(file_path, start_line, end_line)
                if content is None:
                    content = str(item.get("code") or "")
                if not content:
                    continue
                if not _append_window(
                    file_path=file_path,
                    start_line=start_line,
                    end_line=end_line,
                    content=content,
                    role="reference",
                    mode="snippet",
                ):
                    break

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

        prompt_chars = sum(
            len(str(message.get("content") or ""))
            for message in trimmed_messages
            if isinstance(message, dict)
        )
        label = f" {batch_label}" if batch_label else ""
        print(
            "[debug][decision-edit][decision-llm-start]"
            f"{label} file={target_file} first_content_deadline={connect_timeout:g}s"
            f" idle_timeout={stream_idle_timeout:g}s prompt_chars={prompt_chars}",
            flush=True,
        )
        first_chunk_at: float | None = None

        async def _consume_stream() -> None:
            nonlocal response, first_chunk_at
            if (
                hasattr(self.harness, "progress_callback")
                and self.harness.progress_callback
            ):
                self.harness.progress_callback(format_edit_progress(target_file, 0, 0))

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
                    added, removed = count_stream_diff_lines("".join(content_chunks))
                    if (
                        hasattr(self.harness, "progress_callback")
                        and self.harness.progress_callback
                    ):
                        self.harness.progress_callback(
                            format_edit_progress(target_file, added, removed)
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

        raw_content = response.content or ""
        try:
            parsed_decision = self.decision.parse(
                raw_content,
                context_pack.candidate_files,
                edit_only=True,
            )
        except DecisionError as exc:
            excerpt = raw_content.strip()
            if len(excerpt) > _MAX_FAILED_PATCH_FEEDBACK_CHARS:
                excerpt = (
                    excerpt[:_MAX_FAILED_PATCH_FEEDBACK_CHARS]
                    + "\n... (truncated raw response)"
                )
            raise DecisionError(
                f"{exc}\nRaw response excerpt:\n```\n{excerpt}\n```"
            ) from exc
        return parsed_decision, response

    async def execute(self, **params: Any) -> ToolResult:
        started_at = time.monotonic()
        validated = self.validate_params(params)
        target_file = validated["target_file"]
        intent = validated["intent"]
        focus_symbols = list(validated.get("focus_symbols") or [])
        search_cache = params.get("_search_cache")
        context_window = params.get("context_window")

        kept_focus, dropped_focus, remapped_focus = sanitize_focus_symbols(
            self.project_root,
            target_file,
            focus_symbols,
        )
        if dropped_focus or remapped_focus:
            print(
                "[debug][decision-edit][focus-sanitize] "
                f"file={target_file} "
                f"kept={kept_focus} dropped={dropped_focus} remapped={remapped_focus}",
                flush=True,
            )
        focus_symbols = kept_focus

        context_pack = self._build_context_pack(
            target_file,
            search_cache,
            context_window=context_window,
            focus_symbols=focus_symbols,
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
            f"first_content_deadline={effective_timeout:.0f}s "
            f"elapsed={time.monotonic() - started_at:.2f}s",
            flush=True,
        )

        evidence_flag = self._build_evidence_flag(
            target_file,
            search_cache,
            context_pack,
            context_window=context_window,
        )
        state_text = _build_state_text(
            target_file,
            intent,
            focus_symbols,
            dropped_focus=dropped_focus,
            remapped_focus=remapped_focus,
        )
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
            except DecisionError as exc:
                error_detail = f"invalid_patch: {exc}"
                retries_left = patch_attempt < max_attempts
                error_class = classify_edit_error(
                    error_detail, apply_succeeded=False
                )
                owner = retry_owner_for(
                    error_class, edit_retries_remaining=retries_left
                )
                can_retry = edit_inner_retry_allowed(
                    error_detail,
                    apply_succeeded=False,
                    edit_retries_remaining=retries_left,
                )
                if can_retry and owner is RetryOwner.EDIT:
                    print(
                        "[debug][decision-edit][patch-retry] "
                        f"file={target_file} attempt={patch_attempt}/{max_attempts} "
                        f"error_class={error_class.value} "
                        f"error={error_detail.splitlines()[0][:120]}",
                        flush=True,
                    )
                    feedback_state = build_patch_retry_state_text(
                        state_text,
                        attempt=patch_attempt,
                        max_attempts=max_attempts,
                        error=error_detail,
                        failed_patch=str(exc),
                    )
                    continue
                return ToolResult(
                    success=False,
                    output=(
                        f"❌ 【补丁生成失败】：对文件 `{target_file}` 的 Decision "
                        f"输出无法解析。\n"
                        f"ErrorClass={error_class.value} RetryOwner=core\n"
                        f"具体错误：\n```\n{error_detail}\n```\n"
                        f"👉 {core_hint_for(error_class)}"
                    ),
                    error=error_detail,
                    metadata={
                        "patch_attempts": patch_attempt,
                        "error_class": error_class.value,
                        "retry_owner": RetryOwner.CORE.value,
                    },
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

            try:
                concrete_patch = materialize_edit_patch(
                    self.project_root,
                    parsed_decision.target_file,
                    parsed_decision.patch,
                    focus_symbols=focus_symbols,
                    context_window=list(context_window or []),
                )
            except MaterializeError as exc:
                error_detail = str(exc)
                retries_left = patch_attempt < max_attempts
                error_class = classify_edit_error(
                    error_detail,
                    apply_succeeded=False,
                )
                owner = retry_owner_for(
                    error_class, edit_retries_remaining=retries_left
                )
                can_retry = edit_inner_retry_allowed(
                    error_detail,
                    apply_succeeded=False,
                    edit_retries_remaining=retries_left,
                )
                if can_retry and owner is RetryOwner.EDIT:
                    print(
                        "[debug][decision-edit][patch-retry] "
                        f"file={target_file} attempt={patch_attempt}/{max_attempts} "
                        f"error_class={error_class.value} "
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
                retry_hint = f"\n\n👉 {core_hint_for(error_class)}"
                return ToolResult(
                    success=False,
                    output=(
                        f"Edit 小步失败{task_label}。"
                        f"\nErrorClass={error_class.value}"
                        f"\n{error_detail}"
                        f"{retry_hint}"
                    ),
                    error=error_detail,
                    metadata={
                        "patch_attempts": patch_attempt,
                        "error_class": error_class.value,
                        "retry_owner": owner.value,
                    },
                )

            execution, validation, pipeline_metrics = await self.executor.execute_transaction(
                parsed_decision.target_file,
                concrete_patch,
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

            if execution.success and validation.success:
                break

            error_detail = execution.error or ""
            attempted = getattr(execution, "attempted_content", "") or ""
            format_validation_retry = (
                execution.success
                and not validation.success
                and is_format_validation_retry(validation.error or "", attempted)
            )
            if format_validation_retry:
                error_detail = (
                    "invalid_patch: applied patch left SEARCH/REPLACE markers or "
                    "format-corrupted syntax in the target file.\n"
                    f"Validator: {validation.error or 'unknown'}"
                )
            elif execution.success and not validation.success:
                error_detail = validation.error or error_detail or "validation failed"

            retries_left = patch_attempt < max_attempts
            error_class = classify_edit_error(
                error_detail,
                attempted_content=attempted,
                apply_succeeded=bool(execution.success),
            )
            owner = retry_owner_for(
                error_class, edit_retries_remaining=retries_left
            )
            can_retry = edit_inner_retry_allowed(
                error_detail,
                attempted_content=attempted,
                apply_succeeded=bool(execution.success),
                edit_retries_remaining=retries_left,
            )
            if can_retry and owner is RetryOwner.EDIT:
                print(
                    "[debug][decision-edit][patch-retry] "
                    f"file={target_file} attempt={patch_attempt}/{max_attempts} "
                    f"error_class={error_class.value} "
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

            if execution.success and not validation.success:
                # E3/E6 (or exhausted E1) — surface to Core; stop Edit inner loop.
                break

            task_id = params.get("task_id")
            task_label = f" 任务 `{task_id}`" if task_id else ""
            retry_hint = f"\n\n👉 {core_hint_for(error_class)}"
            inner_note = ""
            if patch_attempt > 1:
                inner_note = (
                    f"\n（Edit 内层已重试 {patch_attempt - 1} 次，"
                    f"ErrorClass={error_class.value} → 交还 Core 拆/改 小步。）"
                )
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【补丁生成失败】：{task_label}对文件 `{target_file}` 的补丁无法匹配到物理内容。"
                    f"{inner_note}\n"
                    f"ErrorClass={error_class.value} RetryOwner=core\n"
                    f"具体错误与诊断：\n```\n{error_detail}\n```"
                    f"{retry_hint}"
                ),
                error=error_detail,
                metadata={
                    "pipeline_metrics": pipeline_metrics,
                    "execution": execution,
                    "validation": validation,
                    "patch_attempts": patch_attempt,
                    "error_class": error_class.value,
                    "retry_owner": RetryOwner.CORE.value,
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
            attempted = getattr(execution, "attempted_content", "") or ""
            error_class = classify_edit_error(
                validation.error or "",
                attempted_content=attempted,
                apply_succeeded=True,
            )
            return ToolResult(
                success=False,
                output=(
                    f"❌ 【自动化代码验证失败】：{task_label}对文件 `{target_file}` 的补丁导致编译、语法或测试错误。\n"
                    f"ErrorClass={error_class.value} RetryOwner=core\n"
                    f"👉 已经自动将修改全部物理回滚（Rollback）。\n"
                    f"👉 {core_hint_for(error_class)}\n"
                    f"👉 验证器报错：\n"
                    f"```\n{validation.error}\n```"
                ),
                error=validation.error,
                metadata={
                    "pipeline_metrics": pipeline_metrics,
                    "execution": execution,
                    "validation": validation,
                    "patch_attempts": patch_attempt,
                    "error_class": error_class.value,
                    "retry_owner": RetryOwner.CORE.value,
                },
            )

        diff_summary = format_applied_diff_summary(
            getattr(execution, "original_content", "") or "",
            getattr(execution, "attempted_content", "") or "",
            target_file=target_file,
        )
        return ToolResult(
            success=True,
            output=(
                f"✅ 【应用成功】：针对文件 `{target_file}`{task_label} "
                f"的补丁已通过编译与自动化测试验证，修改已持久化提交（Committed）。\n"
                f"{diff_summary}"
            ),
            metadata={
                "pipeline_metrics": pipeline_metrics,
                "execution": execution,
                "validation": validation,
                "patch_attempts": patch_attempt,
                "applied_diff_summary": diff_summary,
            },
        )
