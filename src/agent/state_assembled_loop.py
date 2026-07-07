from __future__ import annotations

import ast
import asyncio
import copy
import hashlib
import json
import logging
import re
import textwrap
from collections.abc import AsyncIterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.state import StateLayer
from src.state.decision_gravity import evaluate_search_intent
from src.agent.context_assembly import (
    ContextAssembly,
    build_runtime_state_block,
    build_turn_context_block,
)
from src.agent.manifest import (
    Sufficiency,
    execution_card,
    manifest_metrics,
    observations_from_edited_file,
    project_manifest,
)
from src.agent.run_state import (
    ArtifactRefs,
    Evidence,
    RunEvent,
    RunPhase,
    RunState,
    detect_edit_mode,
    reduce_run_state,
    start_run,
)
from src.agent.turn_summary import build_turn_summary, summarize_tool_call
from src.agent.events import (
    PARALLEL_RETRIEVAL_TOOLS,
    AgentEvent,
    EventType,
    approval_event,
    cost_event,
    error_event,
    final_answer_event,
    get_tool_status_text,
    thinking_event,
    tool_call_event,
    tool_result_event,
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
)
from src.tools.grep_match_symbols import grep_search_fingerprint
from src.tools.grep_tool_args import prepare_grep_search_args
from src.hooks.after_tool import apply_after_tool_output_limit
from src.hooks.post_tool_context import apply_post_tool_context_hook
from src.hooks.reallocate_tools import determine_allowed_tools, post_edit_verification_ready
from src.hooks.retrieval_convergence import (
    RETRIEVAL_TOOLS,
    format_duplicate_retrieval_receipt,
    is_duplicate_retrieval_result,
    view_round_all_duplicate,
    retrieval_tool_signal_status,
)
from src.llm.client import LLMClient
from src.llm.dsml import contains_tool_call_markup, strip_dsml_text

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

ASSEMBLED_TOOL_NAMES = frozenset({"codebase_retrieve", "decision_edit", "view_symbol_code", "grep_search"})
LOADED_CODE_ANCHOR_LIMIT = 12


@dataclass(frozen=True, slots=True)
class ContextAnchors:
    """Durable context memory, deliberately separate from retrieval runtime cache."""
    code: tuple[dict[str, Any], ...] = ()
    summaries: dict[str, str] = field(default_factory=dict)
    purposes: dict[str, str] = field(default_factory=dict)
    created_steps: dict[str, int] = field(default_factory=dict)
    file_contracts: dict[str, dict[str, Any]] = field(default_factory=dict)
    file_facts: dict[str, tuple[str, ...]] = field(default_factory=dict)
    schema_contracts: dict[str, str] = field(default_factory=dict)
    last_updated_step: int | None = None


@dataclass(frozen=True, slots=True)
class AssembledState:
    """State tracking for StateAssembledLoop."""
    checklist: tuple[str, ...] = ()
    git_diff: str = ""
    messages_history: tuple[Message, ...] = ()
    search_cache: dict[str, Any] = field(default_factory=dict)
    context_anchors: ContextAnchors = field(default_factory=ContextAnchors)
    core_context_history: tuple[dict[str, Any], ...] = ()
    retrieval_outcome: dict[str, Any] = field(default_factory=dict)
    run_state: RunState = field(default_factory=lambda: start_run("", edit_mode=False))
    last_core_tools: frozenset[str] = frozenset()

    def getMessagesAfterCompactBoundary(self) -> tuple[Message, ...]:
        for idx in range(len(self.messages_history) - 1, -1, -1):
            msg = self.messages_history[idx]
            if "[COMPACT_BOUNDARY]" in msg.content or msg.role == "compact_boundary":
                return self.messages_history[idx + 1:]
        return self.messages_history
class SystemLayerShaper:
    """Shaper 1: System prompt formatting and environment info assembly."""
    def shape(self, state: AssembledState, system_prompt: str) -> str:
        summaries = [
            msg.content for msg in state.messages_history
            if "### TURN SUMMARY" in msg.content and "[CONTEXT COLLAPSE" not in msg.content
        ]
        if summaries:
            summary_section = "\n\n### HISTORICAL TURN SUMMARIES ###\n" + "\n\n".join(summaries[-4:])
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

        summary_text = build_turn_summary(
            to_fold,
            run_state=state.run_state,
            tools_available=state.last_core_tools or None,
        )

        summary_msg = Message(role="system", content=summary_text)
        boundary_msg = Message(role="system", content="[COMPACT_BOUNDARY] Context compressed.")

        new_history = tuple(prior_part) + (summary_msg, boundary_msg) + tuple(to_keep)
        return replace(
            state,
            messages_history=new_history,
        )

class RuntimeShaper:
    """Shaper 5: Active files and runtime validation error shaper."""
    def shape(self, state: AssembledState, active_files: tuple[str, ...]) -> tuple[str, ...]:
        # If we have a lot of active files, we can restrict them to the most recently retrieved or target ones
        if len(active_files) > 6:
            return active_files[-6:]
        return active_files


def _retrieval_tool_signal(
    tc: ToolCall,
    result: Any,
    *,
    arguments: dict[str, Any] | None = None,
) -> str:
    """Compact tool-round summary for RUNTIME STATE (includes failure reason)."""
    call = (
        ToolCall(id=tc.id, name=tc.name, arguments=arguments)
        if arguments is not None
        else tc
    )
    summary = summarize_tool_call(call)
    status = retrieval_tool_signal_status(tc.name, result)
    if status != "failed":
        return f"{summary} -> {status}"
    err = str(getattr(result, "error", "") or "").strip()
    if not err:
        err = str(getattr(result, "output", "") or "").strip()
    if err:
        short = err.splitlines()[0][:140]
        return f"{summary} -> failed: {short}"
    return f"{summary} -> failed"


def _prepare_tool_arguments(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    hint_text: str,
) -> dict[str, Any]:
    args = dict(arguments)
    if tool_name == "grep_search":
        args = prepare_grep_search_args(args, hint_text=hint_text)
    return args


def _merge_suggested_views(
    existing: list[dict[str, Any]],
    new_items: Any,
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {(str(i.get("file") or ""), str(i.get("symbol") or "")) for i in merged}
    if not isinstance(new_items, list):
        return merged
    for item in new_items:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file") or "").strip()
        symbol = str(item.get("symbol") or "").strip()
        if not file_path or not symbol:
            continue
        key = (file_path, symbol)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _observe_grep_round(
    tool_name: str,
    result: ToolResult,
    *,
    round_grep_error: str,
    round_suggested_views: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    if tool_name != "grep_search":
        return round_grep_error, round_suggested_views
    if not result.success:
        err = str(result.error or result.output or "").strip()
        return (round_grep_error or err), round_suggested_views
    views = (result.metadata or {}).get("suggested_views")
    return round_grep_error, _merge_suggested_views(round_suggested_views, views)


def _loaded_code_anchor_block(context_block: str) -> str:
    from src.agent.context_assembly import build_loaded_code_anchor_block

    return build_loaded_code_anchor_block(context_block)


def _build_runtime_state_block(
    *,
    active_files: list[str],
    checklist_str: str,
    git_diff: str,
    validation_error: str | None,
    last_tool_result: str | None,
    last_error: dict[str, Any] | None,
) -> str:
    return build_runtime_state_block(
        active_files=active_files,
        checklist_str=checklist_str,
        git_diff=git_diff,
        validation_error=validation_error,
        last_tool_result=last_tool_result,
        last_error=last_error,
    )


def _build_turn_context_block(
    *,
    loaded_anchors: str,
    execution_card_text: str,
) -> str:
    return build_turn_context_block(
        loaded_anchors=loaded_anchors,
        execution_card_text=execution_card_text,
    )


def _microcompact_retrieval_payload(content: str) -> str:
    try:
        payload = json.loads(content)
    except Exception:
        return content
    if not isinstance(payload, dict):
        return content
    if "dependencies" not in payload and "retrieval_graph" not in payload:
        return content

    compacted = dict(payload)
    compacted["dependencies"] = []
    compacted["retrieval_graph"] = {"nodes": [], "edges": []}
    return json.dumps(compacted, ensure_ascii=False, indent=2)


def _anchor_id(item: dict[str, Any]) -> str:
    span = item.get("span") or []
    if not item.get("file") or len(span) != 2:
        return ""
    return f"{item['file']}:{span[0]}-{span[1]}"


def _anchor_item_from_id(anchor_id: str) -> dict[str, Any] | None:
    match = re.match(r"^(?P<file>.+):(?P<start>\d+)-(?P<end>\d+)$", anchor_id)
    if not match:
        return None
    return {
        "file": match.group("file"),
        "span": [int(match.group("start")), int(match.group("end"))],
    }


def _span_overlap_ratio(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left.get("file") != right.get("file"):
        return 0.0
    left_span = left.get("span") or []
    right_span = right.get("span") or []
    if len(left_span) != 2 or len(right_span) != 2:
        return 0.0
    left_start, left_end = map(int, left_span)
    right_start, right_end = map(int, right_span)
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    shorter = min(left_end - left_start + 1, right_end - right_start + 1)
    return intersection / shorter if shorter > 0 else 0.0


def _anchor_quality(item: dict[str, Any]) -> tuple[int, int, int]:
    span = item.get("span") or [0, 0]
    span_size = int(span[1]) - int(span[0]) + 1 if len(span) == 2 else 0
    return (
        len(str(item.get("code") or "")),
        len(item.get("related_functions") or []),
        span_size,
    )


def _anchor_key(item: dict[str, Any]) -> tuple[Any, ...]:
    """Stable evidence identity: symbols never collapse into neighboring symbols."""
    file = str(item.get("file") or "").replace("\\", "/").lstrip("./")
    symbol = _anchor_symbol_name(item)
    content_hash = str(item.get("hash") or item.get("content_hash") or "")
    if symbol:
        return file, symbol, content_hash
    return file, tuple(item.get("span") or ()), content_hash


def _same_evidence_identity(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if str(left.get("file") or "").lstrip("./") != str(right.get("file") or "").lstrip("./"):
        return False
    left_symbol = _anchor_symbol_name(left)
    right_symbol = _anchor_symbol_name(right)
    if left_symbol or right_symbol:
        return bool(left_symbol and right_symbol and left_symbol == right_symbol)
    return tuple(left.get("span") or ()) == tuple(right.get("span") or ())


def _anchor_is_redundant(item: dict[str, Any], prior: dict[str, Any]) -> bool:
    item_hash = str(item.get("hash") or item.get("content_hash") or "")
    prior_hash = str(prior.get("hash") or prior.get("content_hash") or "")
    if item_hash and prior_hash and item_hash == prior_hash:
        return True
    item_span = item.get("span") or []
    prior_span = prior.get("span") or []
    if len(item_span) != 2 or len(prior_span) != 2:
        return False
    return int(prior_span[0]) <= int(item_span[0]) and int(item_span[1]) <= int(prior_span[1])


def _anchor_symbol_name(item: dict[str, Any]) -> str:
    if item.get("symbol"):
        return str(item["symbol"])
    signature = str(item.get("signature") or _extract_physical_signature(str(item.get("code") or "")))
    match = re.search(r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", signature)
    return match.group(1) if match else ""


def _dedupe_code_anchors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in items:
        duplicate_index = next(
            (index for index, existing in enumerate(kept)
             if _same_evidence_identity(existing, item)
             and _anchor_is_redundant(item, existing)
             and (
                 not _anchor_symbol_name(existing)
                 or not _anchor_symbol_name(item)
                 or _anchor_symbol_name(existing) == _anchor_symbol_name(item)
             )),
            None,
        )
        if duplicate_index is None:
            kept.append(item)
        elif _anchor_quality(item) > _anchor_quality(kept[duplicate_index]):
            kept[duplicate_index] = item
    return kept


def _touch_raw_evidence_lru(
    existing: list[dict[str, Any]],
    incoming: list[dict[str, Any]],
    *,
    limit: int = LOADED_CODE_ANCHOR_LIMIT,
) -> list[dict[str, Any]]:
    """Keep prompt-facing loaded anchors as a small LRU working set."""
    working = [item for item in existing if isinstance(item, dict) and _anchor_id(item)]
    for item in (entry for entry in incoming if isinstance(entry, dict) and _anchor_id(entry)):
        working = [
            prior for prior in working
            if not (
                _same_evidence_identity(prior, item)
                or (
                    prior.get("file") == item.get("file")
                    and _span_overlap_ratio(prior, item) > 0.90
                    and (
                        not _anchor_symbol_name(prior)
                        or not _anchor_symbol_name(item)
                        or _anchor_symbol_name(prior) == _anchor_symbol_name(item)
                    )
                )
            )
        ]
        working.append(item)
    return working[-max(1, limit):]


def _extract_physical_signature(code: str) -> str:
    match = re.search(r"(?m)^\s*(?:async\s+def|def|class)\s+", code)
    if not match:
        return ""
    start = match.start()
    depth = 0
    for index in range(start, len(code)):
        char = code[index]
        if char in "([{" :
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == ":" and depth == 0:
            return code[start:index + 1].strip()
    return code[start:].splitlines()[0].strip()


def _enrich_anchor_contract(project_root: Path, item: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(item)
    enriched["signature"] = _extract_physical_signature(str(item.get("code") or ""))
    imports: list[str] = []
    file_path = str(item.get("file") or "")
    path = (project_root / file_path).resolve()
    if path.is_file() and path.suffix == ".py":
        try:
            source = path.read_text(encoding="utf-8")
            enriched["file_hash"] = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
            tree = ast.parse(source)
            for node in tree.body:
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    segment = ast.get_source_segment(source, node)
                    if segment:
                        imports.append(segment.strip())
        except (OSError, SyntaxError, UnicodeError) as exc:
            log.warning("Failed to extract import contract from %s: %s", file_path, exc)
    enriched["top_level_imports"] = imports
    enriched["content_hash"] = hashlib.sha256(
        str(item.get("code") or "").encode("utf-8")
    ).hexdigest()[:12]
    enriched["evidence_id"] = "::".join(str(part) for part in _anchor_key(enriched))
    return enriched


def _anchor_memory_kind(item: dict[str, Any]) -> str:
    code = textwrap.dedent(str(item.get("code") or "")).strip()
    span = item.get("span") or []
    span_width = (
        int(span[1]) - int(span[0]) + 1
        if len(span) == 2 and all(isinstance(value, int) for value in span)
        else 0
    )
    source_complete = bool(code) and (
        span_width <= 0 or len(code.splitlines()) >= span_width
    )
    if item.get("symbol") and source_complete and not item.get("locator_only"):
        return "symbol"
    file_path = str(item.get("file") or "")
    if code and (
        file_path.endswith(".sql")
        or re.match(r"(?is)^CREATE\s+(?:TABLE|VIEW|PROCEDURE|TRIGGER)\b", code)
    ):
        return "schema"
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return "fact"
    nodes = [node for node in tree.body if not isinstance(node, ast.Expr) or code]
    if any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) for node in nodes):
        return "symbol"
    return "fact"


def _fact_text(item: dict[str, Any]) -> str:
    code = " ".join(str(item.get("code") or "").strip().split())
    span = item.get("span") or ["?", "?"]
    return f"`{span[0]}-{span[1]}` {code[:240]}"


def _symbol_contract(item: dict[str, Any]) -> tuple[str, str, list[str], str, str]:
    code = str(item.get("code") or "")
    signature = str(item.get("signature") or _extract_physical_signature(code))
    symbol = str(item.get("symbol") or "")
    kind = "code"
    decorators: list[str] = []
    parameters = "未解析"
    returns = "未声明"
    try:
        tree = ast.parse(textwrap.dedent(code))
        node = next(
            (entry for entry in tree.body if isinstance(entry, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))),
            None,
        )
        if node is not None:
            symbol = symbol or node.name
            kind = "class" if isinstance(node, ast.ClassDef) else "function"
            decorators = [f"@{ast.unparse(value)}" for value in node.decorator_list]
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                parameters = ast.unparse(node.args)
                returns = ast.unparse(node.returns) if node.returns is not None else "未声明"
    except SyntaxError:
        match = re.search(r"(?:async\s+def|def|class)\s+([A-Za-z_]\w*)", signature)
        if match:
            symbol = symbol or match.group(1)
            kind = "class" if signature.lstrip().startswith("class ") else "function"
    return symbol or "未解析", kind, decorators, parameters, returns


def _compose_context_collapse(
    item: dict[str, Any],
    purpose: str,
    semantic_markdown: str,
) -> str:
    anchor_id = _anchor_id(item)
    symbol, kind, decorators, parameters, returns = _symbol_contract(item)
    signature = str(item.get("signature") or _extract_physical_signature(str(item.get("code") or "")))
    content_hash = str(item.get("content_hash") or hashlib.sha256(
        str(item.get("code") or "").encode("utf-8")
    ).hexdigest()[:12])
    file_contract_ref = f"{item.get('file')}@{item.get('file_hash') or 'unknown'}"
    relation_rows = []
    for related in item.get("related_functions") or []:
        if not isinstance(related, dict) or not related.get("name"):
            continue
        span = related.get("span") or ["?", "?"]
        relation_rows.append(
            f"| `{related['name']}` | `{related.get('file') or '?'}` | "
            f"`{span[0]}-{span[1]}` | 一级强关联 |"
        )
    relation_table = "\n".join(relation_rows) or "| — | — | — | 无已解析关联 |"
    semantic = semantic_markdown.strip()
    semantic = re.sub(
        r"(?m)^\s*\[CONTEXT COLLAPSE[^\n]*\]\s*\n?",
        "",
        semantic,
    ).strip()
    if "## 3." not in semantic:
        semantic = "## 3. 行为契约\n" + semantic
    return (
        f"[CONTEXT COLLAPSE - {anchor_id}] [READ_LOCKED]\n\n"
        "## 1. 锚点身份\n"
        f"- 文件：`{item.get('file')}`\n"
        f"- Span：`{(item.get('span') or ['?', '?'])[0]}-{(item.get('span') or ['?', '?'])[1]}`\n"
        f"- Symbol：`{symbol}`\n"
        f"- 类型：`{kind}`\n"
        f"- 内容哈希：`{content_hash}`\n"
        "- 状态：已读；源码哈希未变化时禁止重复检索\n\n"
        "## 2. 物理接口契约（代码硬提取）\n"
        f"- 函数特征签名 DDL：`{signature or '未解析到声明签名'}`\n"
        f"- Decorators：{'; '.join(f'`{value}`' for value in decorators) or '无'}\n"
        f"- 参数契约：`{parameters}`\n"
        f"- 返回类型：`{returns}`\n"
        f"- 文件契约引用：`{file_contract_ref}`\n\n"
        f"{semantic}\n\n"
        "## 5. 一级强关联\n"
        "| Symbol | File | Span | 关系 |\n"
        "|---|---|---:|---|\n"
        f"{relation_table}\n\n"
        "## 6. 防重复检索闸门\n"
        f"- 已覆盖：`{anchor_id}`\n"
        "- 禁止：相同目的下再次读取重合度 >90% 的 span\n"
        "- 允许重开：文件 hash 变化、需要当前 span 外的新符号，或未确认项成为修改前置条件\n"
        f"- 原始读取目的：{purpose}"
    )


def _collapse_summary(item: dict[str, Any], purpose: str = "为当前任务建立代码事实") -> str:
    """Create a code-free durable anchor when a retrieval turn expires."""
    code = str(item.get("code") or "")
    outcomes = []
    if re.search(r"\braise\b", code):
        outcomes.append("可能抛出异常")
    if re.search(r"\breturn\b", code):
        outcomes.append("包含明确返回路径")
    if re.search(r"\byield\b", code):
        outcomes.append("产生迭代结果")
    if not outcomes:
        outcomes.append("通过副作用或下游调用完成处理")
    semantic = (
        "## 3. 行为契约\n"
        "- 输入：由物理签名定义\n"
        "- 输出：需结合调用点确认\n"
        "- 副作用：需结合实现确认\n"
        f"- 异常/控制流：{'；'.join(outcomes)}\n"
        "- 数据访问：未由本地兜底解析\n"
        "- 不变量：未确认\n\n"
        "## 4. 本次读取目的与结论\n"
        f"- 读取目的：{purpose}\n"
        "- 已确认：物理签名、imports 与一级关联已由代码提取\n"
        "- 解题结论：使用上述硬事实继续决策\n"
        "- 未确认：需要语义模型恢复后补充行为细节"
    )
    return _compose_context_collapse(item, purpose, semantic)


def _collapse_retrieval_turn(
    state: AssembledState,
    eligible_ids_or_summaries: Any = None,
    generated_summaries: dict[str, str] | None = None,
) -> AssembledState:
    """Move last step's detailed code anchor into the independent summary anchor."""
    if generated_summaries is None:
        # Backward compatibility mode
        if isinstance(eligible_ids_or_summaries, dict):
            summaries_dict = eligible_ids_or_summaries
            eligible_ids = set(summaries_dict.keys())
        else:
            # Auto-determine eligible_ids based on step < current_step - 1
            anchors = tuple(
                item for item in state.context_anchors.code
                if state.context_anchors.created_steps.get(
                    _anchor_id(item),
                    state.context_anchors.last_updated_step or state.run_state.step,
                ) < state.run_state.step - 1
            )
            eligible_ids = {_anchor_id(item) for item in anchors}
            summaries_dict = {}

        summaries = dict(state.context_anchors.summaries)
        purposes = dict(state.context_anchors.purposes)
        for item in state.context_anchors.code:
            aid = _anchor_id(item)
            if aid in eligible_ids:
                summaries[aid] = (
                    summaries_dict.get(aid)
                    or _collapse_summary(
                        item,
                        purposes.get(aid, "为当前任务建立代码事实"),
                    )
                )
    else:
        # New mode: grouped file summaries
        eligible_ids = eligible_ids_or_summaries or set()
        summaries = dict(state.context_anchors.summaries)
        for file_path, summary_text in generated_summaries.items():
            summaries[file_path] = summary_text

    cache = dict(state.search_cache)
    encoded = cache.get("search_output")
    if isinstance(encoded, str):
        try:
            payload = json.loads(encoded)
            if isinstance(payload, list):
                payload = [
                    item for item in payload
                    if not isinstance(item, dict) or _anchor_id(item) not in eligible_ids
                ]
                if payload:
                    cache["search_output"] = json.dumps(payload, ensure_ascii=False, indent=2)
                else:
                    cache.pop("search_output", None)
        except (TypeError, json.JSONDecodeError):
            cache.pop("search_output", None)
    cache.pop("context_projection", None)

    messages = []
    for msg in state.messages_history:
        if msg.role == "tool":
            try:
                payload = json.loads(msg.content)
            except Exception:
                payload = None
            
            is_single_dict = isinstance(payload, dict)
            items = [payload] if is_single_dict else (payload if isinstance(payload, list) else [])
            
            has_code = any(
                isinstance(x, dict) and ("code" in x or "observation_code" in x or "verbatim_code" in x) for x in items
            )
            if has_code and any(_anchor_id(x) in eligible_ids for x in items if isinstance(x, dict)):
                collapsed_text = "\n".join(
                    f"[ANCHOR MOVED TO MEMORY: {_anchor_id(x)}]"
                    for x in items
                    if isinstance(x, dict) and _anchor_id(x) in eligible_ids
                )
                fresh = [
                    x for x in items
                    if not isinstance(x, dict) or _anchor_id(x) not in eligible_ids
                ]
                
                if is_single_dict:
                    if fresh:
                        replacement = json.dumps(fresh[0], ensure_ascii=False, indent=2)
                    else:
                        replacement = collapsed_text
                else:
                    replacement = collapsed_text
                    if fresh:
                        replacement += "\n\n" + json.dumps(fresh, ensure_ascii=False, indent=2)
                
                messages.append(replace(msg, content=replacement))
                continue
        messages.append(msg)
    remaining_code = tuple(
        item for item in state.context_anchors.code if _anchor_id(item) not in eligible_ids
    )
    remaining_steps = {
        key: value for key, value in state.context_anchors.created_steps.items()
        if key not in eligible_ids
    }
    context_anchors = replace(
        state.context_anchors,
        code=remaining_code,
        summaries=summaries,
        created_steps=remaining_steps,
        last_updated_step=max(remaining_steps.values()) if remaining_steps else None,
    )
    return replace(
        state,
        search_cache=cache,
        context_anchors=context_anchors,
        messages_history=tuple(messages),
    )


def _search_cache_view(state: AssembledState) -> dict[str, Any]:
    """Compatibility projection for context builders and tools; never stored."""
    view = dict(state.search_cache)
    explicit_working_set = view.get("raw_evidence_store")
    if isinstance(explicit_working_set, list) and explicit_working_set:
        view["raw_evidence_store"] = _touch_raw_evidence_lru(
            [],
            [item for item in explicit_working_set if isinstance(item, dict)],
        )
    elif state.context_anchors.code:
        view["raw_evidence_store"] = _touch_raw_evidence_lru(
            [],
            list(state.context_anchors.code),
        )
    if state.context_anchors.summaries:
        view["summary_anchors"] = dict(state.context_anchors.summaries)
    if state.context_anchors.purposes:
        view["retrieval_purposes"] = dict(state.context_anchors.purposes)
    if state.context_anchors.created_steps:
        view["code_anchor_steps"] = dict(state.context_anchors.created_steps)
    if state.context_anchors.file_contracts:
        view["file_contracts"] = copy.deepcopy(state.context_anchors.file_contracts)
    if state.context_anchors.file_facts:
        view["file_facts"] = copy.deepcopy(state.context_anchors.file_facts)
    if state.context_anchors.schema_contracts:
        view["schema_contracts"] = dict(state.context_anchors.schema_contracts)
    if state.context_anchors.last_updated_step is not None:
        view["last_retrieval_step"] = state.context_anchors.last_updated_step
    return view


def _tool_read_purpose(tool_call: ToolCall) -> str:
    query = str(tool_call.arguments.get("query") or "").strip()
    if query:
        return query
    target = str(tool_call.arguments.get("target_file") or "").strip()
    symbol = str(tool_call.arguments.get("symbol") or "").strip()
    if target and symbol:
        return f"读取 {target} 中的 {symbol}，确认其精确实现与当前任务的关系"
    if target:
        return f"读取 {target}，确认其精确实现与当前任务的关系"
    return "为当前任务建立代码事实"





def _microcompact_retrieval_history(messages: tuple[Message, ...]) -> tuple[Message, ...]:
    compacted_messages = []
    changed = False
    for msg in messages:
        if msg.role == "tool":
            compacted_content = _microcompact_retrieval_payload(msg.content)
            if compacted_content != msg.content:
                changed = True
                compacted_messages.append(
                    Message(
                        role=msg.role,
                        content=compacted_content,
                        name=msg.name,
                        tool_calls=msg.tool_calls,
                        tool_call_id=msg.tool_call_id,
                        cache_breakpoint=msg.cache_breakpoint,
                    )
                )
                continue
        compacted_messages.append(msg)
    return tuple(compacted_messages) if changed else messages


def _microcompact_search_cache(search_cache: dict[str, Any]) -> dict[str, Any]:
    search_output = search_cache.get("search_output")
    if not isinstance(search_output, str):
        return search_cache

    compacted_output = _microcompact_retrieval_payload(search_output)
    if compacted_output == search_output:
        return search_cache

    compacted_cache = dict(search_cache)
    compacted_cache["search_output"] = compacted_output
    return compacted_cache


def _microcompact_context_state(state: AssembledState) -> AssembledState:
    messages_history = _microcompact_retrieval_history(state.messages_history)
    search_cache = _microcompact_search_cache(state.search_cache)
    if messages_history is state.messages_history and search_cache is state.search_cache:
        return state
    return replace(state, messages_history=messages_history, search_cache=search_cache)


def _tool_result_observation(result: ToolResult) -> str:
    if result.metadata:
        observation = result.metadata.get("llm_observation")
        if isinstance(observation, str) and observation.strip():
            return observation
        if isinstance(observation, dict):
            return json.dumps(observation, ensure_ascii=False, indent=2)
    return result.output


def _tool_history_receipt(tool_call: ToolCall, result: ToolResult) -> str:
    if tool_call.name in RETRIEVAL_TOOLS and is_duplicate_retrieval_result(
        tool_call.name, result
    ):
        return format_duplicate_retrieval_receipt(
            tool_call.name,
            result,
            arguments=tool_call.arguments,
        )

    observation = _tool_result_observation(result).strip()
    metadata = result.metadata or {}
    if observation and metadata.get("llm_observation"):
        return observation
    # Successful retrieval with novel evidence: code lands in LOADED CODE ANCHORS.
    if result.success:
        if tool_call.name in PARALLEL_RETRIEVAL_TOOLS:
            return (
                "[RETRIEVAL OK — NEW EVIDENCE STORED]\n"
                "Verbatim code was added to LOADED CODE ANCHORS for the next turn."
            )
        duplicate_replay = str(metadata.get("duplicate_anchor_replay") or "").strip()
        return duplicate_replay or observation
    error_text = (result.error or "").strip()
    if error_text:
        return f"{observation}\n\n[TOOL_ERROR]\n{error_text}".strip()
    return observation


def _fact_lock_replay_result(
    tool_call: ToolCall,
    payload_text: str,
    previous_outcome: dict[str, Any],
    unresolved: tuple[str, ...] = (),
) -> ToolResult:
    """Translate an internal cache replay into a minimal successful tool result."""
    try:
        payload = json.loads(payload_text)
    except (TypeError, json.JSONDecodeError):
        payload = {}

    output = ""
    if isinstance(payload, dict):
        output = str(
            payload.get("observation_code")
            or payload.get("verbatim_code")
            or payload.get("code")
            or ""
        ).strip()
    if not output:
        output = payload_text.strip()

    return ToolResult(
        success=True,
        output=output,
        metadata={
            "is_mock_success": True,
        },
    )


def _filter_grounded_slots(
    slots: tuple[str, ...] | list[str],
    refs: ArtifactRefs,
) -> tuple[str, ...]:
    """Grep locators must not satisfy schema slots without a verbatim schema anchor."""
    schema_only = frozenset({"relevant_schema"})
    filtered: list[str] = []
    for slot in slots:
        if slot in schema_only and not refs.schemas:
            continue
        filtered.append(str(slot))
    return tuple(filtered)


def _run_evidence_event(result: ToolResult) -> RunEvent | None:
    """Translate tool evidence metadata into the reducer's immutable event."""
    metadata = result.metadata or {}
    payload = metadata.get("run_event")
    if not isinstance(payload, dict) or payload.get("kind") != "evidence_discovered":
        return None

    raw_items = tuple(
        item
        for item in metadata.get("raw_evidence_store") or ()
        if isinstance(item, dict) and _anchor_id(item)
    )
    code_refs: list[str] = []
    schema_refs: list[str] = []
    fact_refs: list[str] = []
    for item in raw_items:
        artifact_id = _anchor_id(item)
        kind = _anchor_memory_kind(item)
        if kind == "symbol":
            code_refs.append(artifact_id)
        elif kind == "schema":
            schema_refs.append(artifact_id)
        else:
            fact_refs.append(artifact_id)
    refs = ArtifactRefs(
        code=tuple(code_refs),
        schemas=tuple(schema_refs),
        facts=tuple(fact_refs),
    )
    available_refs = refs.all

    evidence: list[Evidence] = []
    grounded_slots = payload.get("grounded_slots") or ()
    grounded_slots = _filter_grounded_slots(grounded_slots, refs)
    if available_refs:
        source = raw_items[0]
        artifact_id = next(iter(available_refs))
        for slot in grounded_slots:
            evidence.append(Evidence(
                slot=str(slot),
                artifact_id=artifact_id,
                file=str(source.get("file") or ""),
                symbol=str(source.get("symbol") or "") or None,
                evidence_type=(
                    "full_symbol" if artifact_id in refs.code
                    else "schema" if artifact_id in refs.schemas
                    else "exact_match"
                ),
            ))

    candidates = tuple(
        f"{item.get('file')}::{item.get('symbol')}"
        for item in payload.get("candidates") or ()
        if isinstance(item, dict) and item.get("file") and item.get("symbol")
    )
    observations = tuple(dict(item) for item in raw_items)
    if not evidence and not candidates and not available_refs and not observations:
        return None
    return RunEvent(
        "evidence_stored",
        evidence=tuple(evidence),
        candidates=candidates,
        artifact_refs=refs,
        observations=observations,
        reason="tool evidence ingested",
    )


def _duplicate_anchor_replay(items: list[dict[str, Any]]) -> str:
    """Return exact durable slices without implying full-file coverage."""
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in items:
        unique[_anchor_key(item)] = item
    lines = [
        "[DUPLICATE ANCHOR — EXISTING FACTS REPLAYED]",
        "The following exact code slices are already durable in the context:",
    ]
    for item in unique.values():
        code = str(item.get("code") or "")
        span = item.get("span") or ["?", "?"]
        symbol, kind, decorators, _parameters, returns = _symbol_contract(item)
        signature = str(
            item.get("signature") or _extract_physical_signature(code) or symbol
        )
        lines.extend(
            [
                f"- `{item.get('file')}:{span[0]}-{span[1]}` `{symbol}` ({kind})",
                "  [EXACT SYMBOL SLICE COMPLETE]",
                f"  SOURCE COVERAGE: {item.get('file')}:{span[0]}-{span[1]} only",
                f"  signature: `{signature}`",
                f"  decorators: {', '.join(decorators) or 'none'}",
                f"  returns: {returns}",
                f"  source_code:",
                f"```python\n{code}\n```",
            ]
        )
    return "\n".join(lines)


def _prune_contained_anchors(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the widest anchor when one source span fully contains another."""
    kept: list[dict[str, Any]] = []
    ordered = sorted(
        items,
        key=lambda item: (
            str(item.get("file") or "").replace("\\", "/").lstrip("./"),
            -(
                int((item.get("span") or [0, 0])[1])
                - int((item.get("span") or [0, 0])[0])
            ),
        ),
    )
    for item in ordered:
        file = str(item.get("file") or "").replace("\\", "/").lstrip("./")
        span = item.get("span") or []
        if len(span) != 2:
            kept.append(item)
            continue
        contained = any(
            file
            == str(prior.get("file") or "").replace("\\", "/").lstrip("./")
            and len(prior.get("span") or []) == 2
            and int(prior["span"][0]) <= int(span[0])
            and int(span[1]) <= int(prior["span"][1])
            for prior in kept
        )
        if not contained:
            kept.append(item)
    return kept


def _response_evidence_summary(state: AssembledState) -> str:
    anchors = _prune_contained_anchors(list(state.context_anchors.code))
    sections = [
        "### GROUNDED EVIDENCE SUMMARY ###",
        _duplicate_anchor_replay(anchors) if anchors else "No live code anchors remain.",
    ]
    imports_by_file: dict[str, list[str]] = {}
    for raw_file, contract in state.context_anchors.file_contracts.items():
        file = str(raw_file).replace("\\", "/").lstrip("./")
        values = imports_by_file.setdefault(file, [])
        for imported in contract.get("imports") or []:
            if imported not in values:
                values.append(str(imported))
    if imports_by_file:
        sections.append("### FILE IMPORTS (ONCE PER FILE) ###")
        for file, imports in sorted(imports_by_file.items()):
            sections.append(f"- `{file}`: " + ("; ".join(imports) or "none"))
    if state.context_anchors.schema_contracts:
        useful_schema = [
            (key, " ".join(str(value).split()))
            for key, value in sorted(state.context_anchors.schema_contracts.items())
            if len(" ".join(str(value).split())) >= 20
        ][:12]
        if useful_schema:
            sections.append("### SCHEMA FACTS ###")
            sections.extend(f"- `{key}`: {value}" for key, value in useful_schema)
    return "\n".join(sections)


def _search_cache_context_projection(
    search_cache: dict[str, Any],
    *,
    current_step: int,
) -> str:
    raw_anchors = search_cache.get("raw_evidence_store") or []
    projections = search_cache.get("symbol_projections") or []
    projected_symbols = {p.get("symbol") for p in projections if isinstance(p, dict) and p.get("symbol")}

    filtered_anchors = []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if not isinstance(item, dict):
                continue
            sym = item.get("symbol")
            if sym and sym in projected_symbols:
                continue
            filtered_anchors.append(item)

    if filtered_anchors:
        return json.dumps(filtered_anchors, ensure_ascii=False, indent=2)
    summaries = search_cache.get("summary_anchors") or {}
    search_output = search_cache.get("search_output")
    if not isinstance(search_output, str) or not search_output.strip():
        return ""

    last_step = search_cache.get("last_retrieval_step")
    if not isinstance(last_step, int):
        last_step = current_step
    age = max(0, current_step - last_step)

    try:
        payload = json.loads(search_output)
    except Exception:
        return search_output[:4000] if age <= 1 else ""
    if not isinstance(payload, list):
        return ""
    if age == 0:
        return json.dumps(payload, ensure_ascii=False, indent=2)
    return "\n\n".join(summaries.values()) if isinstance(summaries, dict) else ""


def _search_cache_for_context(
    search_cache: dict[str, Any],
    *,
    current_step: int,
) -> dict[str, Any]:
    if not search_cache:
        return search_cache
    projection = _search_cache_context_projection(
        search_cache,
        current_step=current_step,
    )
    if not projection:
        compacted = dict(search_cache)
        compacted.pop("search_output", None)
        compacted.pop("context_projection", None)
        return compacted
    compacted = dict(search_cache)
    compacted["context_projection"] = projection
    return compacted


def _format_symbol_slice(item: dict[str, Any]) -> str | None:
    projection_code = item.get("projection_code") or item.get("code")
    file_path = item.get("file")
    span = item.get("span") or []
    if not projection_code or not file_path or len(span) < 2:
        return None
    truncated_attr = ' truncated="true"' if item.get("truncated") else ""
    symbol_attr = f' symbol="{item.get("symbol")}"' if item.get("symbol") else ""
    return (
        f'<symbol_slice file="{file_path}"{symbol_attr} '
        f'span="{span[0]}-{span[1]}"{truncated_attr}>\n'
        f"{projection_code}\n"
        "</symbol_slice>"
    )


def _format_raw_anchor(item: dict[str, Any]) -> str:
    file_path = item.get("file", "?")
    span = item.get("span") or ["?", "?"]
    symbol = item.get("symbol")
    code = item.get("code") or item.get("verbatim_code") or ""
    header = f"// {file_path}:{span[0]}-{span[1]}"
    if symbol:
        header += f" {symbol}"
    return f"{header}\n```python\n{code}\n```"


def _norm_anchor_file(file: str | None) -> str:
    return str(file or "").replace("\\", "/").lstrip("./")


def _priority_anchor_files(
    search_cache: dict[str, Any],
    *,
    task_text: str = "",
    manifest: Any = None,
) -> frozenset[str]:
    """Files whose anchors must stay visible when edit_ready (not just last projections)."""
    priority: set[str] = set()

    if manifest is not None:
        for item in getattr(manifest, "required_items", ()) or ():
            if getattr(item, "status", None) in {"SATISFIED", "STALE"} and getattr(item, "file", None):
                priority.add(_norm_anchor_file(item.file))

    for match in re.finditer(r"[\w./-]+\.py", task_text):
        priority.add(_norm_anchor_file(match.group(0)))

    raw_anchors = search_cache.get("raw_evidence_store") or []
    if isinstance(raw_anchors, list):
        evidence_files = {
            _norm_anchor_file(str(item.get("file")))
            for item in raw_anchors
            if isinstance(item, dict) and item.get("file")
        }
        if len(evidence_files) >= 2:
            priority.update(evidence_files)

    return frozenset(priority)


def _build_deduped_loaded_anchors_block(
    search_cache: dict[str, Any],
    *,
    edit_ready: bool = False,
    task_text: str = "",
    manifest: Any = None,
) -> str:
    """Build one deduplicated loaded-code section for the Core LLM prompt."""
    seen: set[tuple[Any, ...]] = set()
    blocks: list[str] = []
    projected_symbols: set[str] = set()
    priority_files = _priority_anchor_files(
        search_cache,
        task_text=task_text,
        manifest=manifest,
    )

    projections = search_cache.get("symbol_projections") or []
    if isinstance(projections, list):
        if edit_ready and priority_files:
            selected = [
                item
                for item in projections
                if isinstance(item, dict)
                and _norm_anchor_file(str(item.get("file") or "")) in priority_files
            ]
            if not selected:
                selected = list(projections)
        else:
            selected = list(projections[-2:])

        for item in selected:
            key = _anchor_key(item)
            if key in seen:
                continue
            formatted = _format_symbol_slice(item)
            if not formatted:
                continue
            seen.add(key)
            sym = item.get("symbol")
            if sym:
                projected_symbols.add(str(sym))
            blocks.append(formatted)

    raw_anchors = search_cache.get("raw_evidence_store") or []
    if isinstance(raw_anchors, list):
        for item in raw_anchors:
            if not isinstance(item, dict):
                continue
            file_path = _norm_anchor_file(str(item.get("file") or ""))
            if edit_ready and priority_files and file_path not in priority_files:
                continue
            sym = item.get("symbol")
            if sym and str(sym) in projected_symbols:
                if not (edit_ready and priority_files and file_path in priority_files):
                    continue
            key = _anchor_key(item)
            if key in seen:
                continue
            code = str(item.get("code") or item.get("verbatim_code") or "").strip()
            if not code:
                continue
            seen.add(key)
            blocks.append(_format_raw_anchor(item))

    return "\n\n".join(blocks)


def _latest_symbol_slice_projection(search_cache: dict[str, Any]) -> str:
    """Legacy wrapper; prefer _build_deduped_loaded_anchors_block."""
    block = _build_deduped_loaded_anchors_block(search_cache)
    if not block:
        return ""
    return "### ACTIVE TEMPORARY CODE SLICES (from view_symbol_code) ###\n" + block


def _retrieval_snapshot_from_output(
    query: str,
    search_output: str,
    *,
    step: int,
) -> dict[str, Any] | None:
    try:
        payload = json.loads(search_output)
    except Exception:
        return None
    if isinstance(payload, list):
        files = {
            str(item.get("file"))
            for item in payload
            if isinstance(item, dict) and item.get("file")
        }
        symbols = {
            str(related.get("name"))
            for item in payload if isinstance(item, dict)
            for related in (item.get("related_functions") or [])
            if isinstance(related, dict) and related.get("name")
        }
        symbols.update(
            str(item["symbol"])
            for item in payload
            if isinstance(item, dict) and item.get("symbol")
        )
        list_anchors = []
        for item in payload:
            if not isinstance(item, dict) or not item.get("file"):
                continue
            span = item.get("span") or []
            list_anchors.append({
                "file": str(item["file"]),
                "span": list(span[:2]) if len(span) >= 2 else [],
                "symbol": str(item.get("symbol") or ""),
            })
        precise_keys = {
            (item["file"], item["symbol"])
            for item in list_anchors if len(item["span"]) >= 2
        }
        list_anchors = [
            item for item in list_anchors
            if len(item["span"]) >= 2 or (item["file"], item["symbol"]) not in precise_keys
        ]
        list_anchors = list({
            (item["file"], tuple(item["span"]), item["symbol"]): item
            for item in list_anchors
        }.values())
        list_anchors.sort(key=lambda item: (item["file"], item["span"], item["symbol"]))
        signature_src = json.dumps(
            {"files": sorted(files), "symbols": sorted(symbols), "anchors": list_anchors},
            ensure_ascii=False, sort_keys=True,
        )
        return {
            "step": step, "query": query, "files": sorted(files), "symbols": sorted(symbols),
            "anchors": list_anchors,
            "signature": hashlib.sha256(signature_src.encode("utf-8")).hexdigest()[:16],
        }
    if not isinstance(payload, dict):
        return None

    grounding = payload.get("grounding") or {}
    evidence = payload.get("evidence") or []
    files = set()
    symbols = set()
    anchors: list[dict[str, Any]] = []
    for item in grounding.get("files", []):
        if isinstance(item, dict) and item.get("path"):
            files.add(str(item["path"]))
    for item in grounding.get("symbols", []):
        if isinstance(item, dict):
            if item.get("file"):
                files.add(str(item["file"]))
            if item.get("name"):
                symbols.add(str(item["name"]))
            if item.get("file"):
                span = item.get("span") or []
                anchors.append({
                    "file": str(item["file"]),
                    "span": list(span[:2]) if len(span) >= 2 else [],
                    "symbol": str(item.get("name") or ""),
                })
    for item in evidence:
        if isinstance(item, dict):
            if item.get("file"):
                files.add(str(item["file"]))
            if item.get("symbol"):
                symbols.add(str(item["symbol"]))
            if item.get("file"):
                span = item.get("span") or []
                anchors.append({
                    "file": str(item["file"]),
                    "span": list(span[:2]) if len(span) >= 2 else [],
                    "symbol": str(item.get("symbol") or ""),
                })

    if not anchors:
        anchors = [
            {"file": file_path, "span": [], "symbol": symbol}
            for file_path in sorted(files)
            for symbol in (sorted(symbols) or [""])
        ]
    precise_keys = {
        (item["file"], item["symbol"])
        for item in anchors if len(item["span"]) >= 2
    }
    anchors = [
        item for item in anchors
        if len(item["span"]) >= 2 or (item["file"], item["symbol"]) not in precise_keys
    ]
    anchors = list({
        (item["file"], tuple(item["span"]), item["symbol"]): item
        for item in anchors
    }.values())
    anchors.sort(key=lambda item: (item["file"], item["span"], item["symbol"]))

    signature_src = json.dumps(
        {
            "files": sorted(files),
            "symbols": sorted(symbols),
            "anchors": anchors,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "step": step,
        "query": query,
        "files": sorted(files),
        "symbols": sorted(symbols),
        "anchors": anchors,
        "signature": hashlib.sha256(signature_src.encode("utf-8")).hexdigest()[:16],
    }


def _anchor_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    if left.get("file") != right.get("file"):
        return 0.0
    left_symbol = str(left.get("symbol") or "")
    right_symbol = str(right.get("symbol") or "")
    if left_symbol and right_symbol and left_symbol != right_symbol:
        return 0.0
    left_span = left.get("span") or []
    right_span = right.get("span") or []
    if len(left_span) < 2 or len(right_span) < 2:
        return 1.0 if left_symbol == right_symbol or not left_symbol or not right_symbol else 0.0
    left_start, left_end = int(left_span[0]), int(left_span[1])
    right_start, right_end = int(right_span[0]), int(right_span[1])
    intersection = max(0, min(left_end, right_end) - max(left_start, right_start) + 1)
    union = max(left_end, right_end) - min(left_start, right_start) + 1
    return intersection / union if union else 0.0


def _snapshot_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    left_anchors = left.get("anchors") or []
    right_anchors = right.get("anchors") or []
    if left_anchors and right_anchors:
        scores = [
            max((_anchor_similarity(anchor, other) for other in right_anchors), default=0.0)
            for anchor in left_anchors
        ]
        return sum(score >= 0.8 for score in scores) / max(len(left_anchors), len(right_anchors))
    left_facts = set(left.get("files") or ()) | set(left.get("symbols") or ())
    right_facts = set(right.get("files") or ()) | set(right.get("symbols") or ())
    union = left_facts | right_facts
    return len(left_facts & right_facts) / len(union) if union else 0.0





#
# NOTE: Retrieval repetition signalling / outcome ledgers were intentionally
# removed. The Core LLM prompt should contain concrete code context and actionable
# failures only (tool errors / validator errors). Any redundancy prevention is
# enforced via tool availability and preflight blockers.


def _run_state_projection(state: RunState) -> str:
    evidence_lines = [
        f"- {name}: {'satisfied' if name in state.evidence.grounded else 'missing'}"
        for name in sorted(state.evidence.required)
    ]
    next_action = (
        "Return the final answer without tools."
        if state.phase == RunPhase.RESPONDING
        else "Continue with decision_edit or return the final validated answer."
        if state.can_answer
        else " | ".join(_missing_evidence_actions(sorted(state.evidence.missing)))
    )
    return "\n".join([
        "### RUN STATE ###",
        f"- phase: {state.phase.value}",
        f"- task_mode: {state.task_mode}",
        f"- step: {state.step}",
        f"- transition_reason: {state.transition_reason}",
        f"- candidates: {', '.join(state.evidence.candidates) or 'none'}",
        "- evidence_requirements:",
        *evidence_lines,
        f"- allowed_tools: {', '.join(sorted(state.allowed_tools)) or 'none'}",
        f"- next_action: {next_action}",
    ])


def _missing_evidence_actions(missing: list[str]) -> list[str]:
    actions = {
        "target_implementation": (
            'target_implementation → grep_search(pattern="<target route or symbol>", '
            'include="*.py", max_results=20), then view_symbol_code on the best hit'
        ),
        "endpoint_implementation": (
            'endpoint_implementation → view_symbol_code(target_file="<endpoint file>", '
            'symbol="<endpoint function>")'
        ),
        "integration_or_mount_point": (
            'integration_or_mount_point → grep_search(pattern="build_router|include_router", '
            'include="*.py", max_results=20), then view_symbol_code on build_router'
        ),
        "authentication_context": (
            'authentication_context → grep_search(pattern="get_current_user|bearer|token", '
            'include="*.py", max_results=20), then view_symbol_code on the auth dependency'
        ),
        "authorization_policy": (
            'authorization_policy → grep_search(pattern="role|permission|authorize", '
            'include="*.py", max_results=20), then view_symbol_code on the policy implementation'
        ),
        "ownership_relation": (
            'ownership_relation → grep_search(pattern="owner_id|user_id|tenant_id|p_id", '
            'include="*.py", max_results=20), then inspect the query or model symbol'
        ),
        "relevant_schema": (
            'relevant_schema → grep_search(pattern="CREATE TABLE|<relevant table>", '
            'include="*.sql", max_results=20)'
        ),
        "test_or_validation_path": (
            'test_or_validation_path → grep_search(pattern="def test_|<target symbol>", '
            'include="test*.py", max_results=20)'
        ),
    }
    return [actions[name] for name in missing if name in actions] or [
        "No additional retrieval; proceed to edit or summarize."
    ]


async def _get_git_state(cwd: Path) -> str:
    """Return file-level working-tree state only; never inject patch contents."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "git", "status", "--short", "--untracked-files=normal",
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
        self.context_builder = context
        self.permissions = permissions
        self.settings = settings

        import uuid
        self.session_id = f"session_{uuid.uuid4().hex[:12]}"

        self.context_assembly = ContextAssembly(harness.project_root)
        self.stateLayer = StateLayer(harness.project_root, self.context_assembly)
        self._grep_search_history: list[dict[str, Any]] = []
        self._last_novelty_value: float = 1.0
        self._embeddings_cache: dict[str, list[float]] = {}
        self._current_step_tools: frozenset[str] = ASSEMBLED_TOOL_NAMES

        self.state = AssembledState(
            run_state=start_run("", edit_mode=False, max_steps=settings.max_turns)
        )
        self._approval_futures: dict[str, asyncio.Future[bool]] = {}
        self._summary_llm: LLMClient | None = None
        self._core_context_records: list[dict[str, Any]] = []
        self._run_events: list[RunEvent] = []
        self.agent_telemetry = AgentState()
        self._retrieval_history: tuple[dict[str, Any], ...] = ()
        self._task_text: str = ""

    async def _run_tool_with_sentinel(
        self, name: str, args: dict[str, Any], queue: asyncio.Queue[Any]
    ) -> ToolResult:
        try:
            res = await self.tools.call(name, args)
            return res
        finally:
            await queue.put(None)

    def _repo_map_snapshot(self) -> Any:
        context_builder = getattr(self, "context_builder", None)
        if context_builder is None:
            return None
        service = getattr(context_builder, "repo_map_service", None)
        if service is None:
            return None
        try:
            return getattr(service, "map", None)
        except Exception:
            return None

    def _inject_grep_execute_args(self, tool_args: dict[str, Any]) -> None:
        tool_args["_project_root"] = self.harness.project_root
        repo_map = self._repo_map_snapshot()
        if repo_map is not None:
            tool_args["_repo_map"] = repo_map

    def get_context_records(self) -> list[dict[str, Any]]:
        return copy.deepcopy(self._core_context_records)

    def _dispatch_run_event(self, event: RunEvent) -> tuple[Any, ...]:
        self._run_events.append(event)
        run_state, effects = reduce_run_state(self.state.run_state, event)
        self.state = replace(self.state, run_state=run_state)
        return effects

    def _last_tool_result(self) -> str | None:
        event = next(
            (item for item in reversed(self._run_events) if item.kind == "tool_round_observed"),
            None,
        )
        return event.reason if event and event.reason else None

    def _last_error(self) -> dict[str, Any] | None:
        event = next(
            (
                item for item in reversed(self._run_events)
                if item.kind == "tool_round_observed"
            ),
            None,
        )
        return (
            self._parse_structured_error(event.issues[0])
            if event and event.issues else None
        )

    def _validation_error(self) -> str | None:
        issues = self.state.run_state.validation.issues
        return "\n".join(issues) if issues else None

    def _normalize_anchor_memory(self) -> None:
        """Migrate legacy summaries and remove redundant durable anchors."""
        summaries = dict(self.state.context_anchors.summaries)
        file_contracts = copy.deepcopy(self.state.context_anchors.file_contracts)
        file_facts = {
            path: list(facts) for path, facts in self.state.context_anchors.file_facts.items()
        }
        schemas = dict(self.state.context_anchors.schema_contracts)
        for anchor_id, text in list(summaries.items()):
            item = _anchor_item_from_id(anchor_id)
            if item is None:
                continue
            file_path = str(item["file"])
            if file_path not in file_contracts:
                enriched = _enrich_anchor_contract(
                    self.harness.project_root,
                    {"file": file_path, "span": item["span"], "code": ""},
                )
                if enriched.get("file_hash"):
                    file_contracts[file_path] = {
                        "hash": enriched["file_hash"],
                        "imports": list(enriched.get("top_level_imports") or []),
                    }
            contract = file_contracts.get(file_path)
            if contract and "顶层物理进口：" in text:
                reference = f"文件契约引用：`{file_path}@{contract['hash']}`\n"
                text = re.sub(
                    r"顶层物理进口：.*?(?=\n\s*读取目的：|\n\s*物理特征：|\Z)",
                    reference.rstrip(),
                    text,
                    flags=re.DOTALL,
                )
                summaries[anchor_id] = text
            if "表名称：" in text or "CREATE TABLE" in text.upper():
                schemas[anchor_id] = " ".join(text.split())[:500]
                summaries.pop(anchor_id, None)
            elif any(marker in text for marker in (
                "函数名称：无", "字段名称：", "未解析到声明签名", "为import语句", "为常量定义"
            )):
                facts = file_facts.setdefault(file_path, [])
                fact = f"`{anchor_id}` " + " ".join(text.split())[:360]
                if fact not in facts:
                    facts.append(fact)
                    del facts[:-20]
                summaries.pop(anchor_id, None)

        def summary_symbol(text: str) -> str:
            match = re.search(r"(?:Symbol|函数名称)：\s*`?([^`\n（(]+)", text)
            return match.group(1).strip() if match else ""

        kept: dict[str, str] = {}
        for anchor_id, text in sorted(
            summaries.items(),
            key=lambda pair: -(
                ((_anchor_item_from_id(pair[0]) or {}).get("span") or [0, -1])[1]
                - ((_anchor_item_from_id(pair[0]) or {}).get("span") or [0, -1])[0]
            ),
        ):
            item = _anchor_item_from_id(anchor_id)
            if item is None:
                kept[anchor_id] = text
                continue
            duplicate = False
            for kept_id, kept_text in kept.items():
                kept_item = _anchor_item_from_id(kept_id)
                if kept_item and _span_overlap_ratio(item, kept_item) > 0.90:
                    left_symbol = summary_symbol(text)
                    right_symbol = summary_symbol(kept_text)
                    if not left_symbol or not right_symbol or left_symbol == right_symbol:
                        duplicate = True
                        break
            if not duplicate:
                kept[anchor_id] = text
        sanitized_messages = []
        for msg in self.state.messages_history:
            anchor_ids = list(dict.fromkeys(re.findall(
                r"\[CONTEXT COLLAPSE - ([^\]]+)\]", msg.content
            )))
            if anchor_ids and msg.role == "tool":
                sanitized_messages.append(replace(
                    msg,
                    content="\n".join(
                        f"[ANCHOR MOVED TO MEMORY: {anchor_id}]" for anchor_id in anchor_ids
                    ),
                ))
            elif "STRUCTURED CONVERSATION SUMMARY" in msg.content:
                sanitized_messages.append(replace(
                    msg,
                    content=(
                        "### TURN SUMMARY\n"
                        "- 决策：旧摘要仅保留已验证事实\n"
                        f"- 已读取：{', '.join(anchor_ids) if anchor_ids else '见 ContextAnchors'}\n"
                        "- 编辑：见工具完成记录\n"
                        "- 验证/错误：见当前状态\n"
                        "- 下一步：依据 RUN STATE 的 phase 与 evidence 行动"
                    ),
                ))
            else:
                sanitized_messages.append(msg)
        self.state = replace(
            self.state,
            context_anchors=replace(
                self.state.context_anchors,
                summaries=kept,
                file_contracts=file_contracts,
                file_facts={path: tuple(facts) for path, facts in file_facts.items()},
                schema_contracts=schemas,
            ),
            messages_history=tuple(sanitized_messages),
        )

    def _apply_context_update(self, result: ToolResult) -> dict[str, Any] | None:
        metadata = result.metadata or {}
        run_event = _run_evidence_event(result)
        if run_event is not None:
            self._dispatch_run_event(run_event)
        update = metadata.get("artifact_update")
        if not isinstance(update, dict):
            completion = metadata.get("task_completion")
            return completion if isinstance(completion, dict) else None
        invalidated = {
            str(path) for path in update.get("invalidate_code_files") or [] if path
        }
        if invalidated:
            removed_ids = {
                _anchor_id(item) for item in self.state.context_anchors.code
                if str(item.get("file") or "") in invalidated
            }
            removed_ids.update(
                key for key in self.state.context_anchors.summaries
                if any(key.startswith(f"{path}:") for path in invalidated)
            )
            removed_ids.update(
                key for key in self.state.context_anchors.schema_contracts
                if any(key.startswith(f"{path}:") for path in invalidated)
            )
            code = tuple(
                item for item in self.state.context_anchors.code
                if str(item.get("file") or "") not in invalidated
            )
            summaries = {
                key: value for key, value in self.state.context_anchors.summaries.items()
                if not any(key.startswith(f"{path}:") for path in invalidated)
            }
            purposes = {
                key: value for key, value in self.state.context_anchors.purposes.items()
                if not any(key.startswith(f"{path}:") for path in invalidated)
            }
            created_steps = {
                key: value for key, value in self.state.context_anchors.created_steps.items()
                if not any(key.startswith(f"{path}:") for path in invalidated)
            }
            file_contracts = {
                key: value for key, value in self.state.context_anchors.file_contracts.items()
                if key not in invalidated
            }
            file_facts = {
                key: value for key, value in self.state.context_anchors.file_facts.items()
                if key not in invalidated
            }
            schemas = {
                key: value for key, value in self.state.context_anchors.schema_contracts.items()
                if not any(key.startswith(f"{path}:") for path in invalidated)
            }
            cache = dict(self.state.search_cache)
            cache["symbol_projections"] = [
                item for item in cache.get("symbol_projections") or []
                if str(item.get("file") or "") not in invalidated
            ]
            encoded = cache.get("search_output")
            if isinstance(encoded, str):
                try:
                    payload = json.loads(encoded)
                    if isinstance(payload, list):
                        payload = [
                            item for item in payload
                            if not isinstance(item, dict)
                            or str(item.get("file") or "") not in invalidated
                        ]
                        cache["search_output"] = json.dumps(payload, ensure_ascii=False, indent=2)
                except (TypeError, json.JSONDecodeError):
                    pass
            self.state = replace(
                self.state,
                context_anchors=replace(
                    self.state.context_anchors,
                    code=code,
                    summaries=summaries,
                    purposes=purposes,
                    created_steps=created_steps,
                    file_contracts=file_contracts,
                    file_facts=file_facts,
                    schema_contracts=schemas,
                    last_updated_step=max(created_steps.values()) if created_steps else None,
                ),
                search_cache=cache,
            )
            self._dispatch_run_event(RunEvent(
                "artifacts_invalidated",
                artifact_refs=ArtifactRefs(
                    code=tuple(removed_ids),
                    schemas=tuple(removed_ids),
                    facts=tuple(removed_ids),
                    summaries=tuple(removed_ids),
                ),
                reason=f"files changed: {', '.join(sorted(invalidated))}",
            ))
        completion = metadata.get("task_completion")
        return completion if isinstance(completion, dict) else None

    def _dedupe_result_anchors(
        self,
        result: ToolResult,
        *,
        existing_anchors: list[dict[str, Any]] | None = None,
    ) -> ToolResult:
        metadata = dict(result.metadata or {})
        if metadata.get("is_mock_success"):
            return result
        raw = metadata.get("raw_evidence_store")
        if not isinstance(raw, list) or not raw:
            return result
        existing = (
            list(existing_anchors)
            if existing_anchors is not None
            else list(self.state.context_anchors.code)
        )
        accepted: list[dict[str, Any]] = []
        replayed: list[dict[str, Any]] = []
        for item in (entry for entry in raw if isinstance(entry, dict)):
            duplicates = [
                prior for prior in [*existing, *accepted]
                if _same_evidence_identity(item, prior)
                and _anchor_is_redundant(item, prior)
            ]
            if duplicates and not all(
                _anchor_quality(item) > _anchor_quality(prior) for prior in duplicates
            ):
                replayed.extend(duplicates)
                continue
            accepted.append(item)
        if len(accepted) == len(raw):
            return result
        metadata["raw_evidence_store"] = accepted
        if replayed:
            metadata["refresh_evidence_store"] = replayed
        accepted_ids = {_anchor_id(item) for item in accepted}

        def filter_json_list(value: Any) -> Any:
            if not isinstance(value, str):
                return value
            try:
                payload = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                return value
            if not isinstance(payload, list):
                return value
            filtered = [
                item for item in payload
                if not isinstance(item, dict) or _anchor_id(item) in accepted_ids
            ]
            return json.dumps(filtered, ensure_ascii=False, indent=2)

        if "search_output" in metadata:
            metadata["search_output"] = filter_json_list(metadata["search_output"])
        if "llm_observation" in metadata:
            metadata["llm_observation"] = filter_json_list(metadata["llm_observation"])
        output = filter_json_list(result.output)
        replay = _duplicate_anchor_replay(replayed) if replayed else ""
        if replay:
            metadata["duplicate_anchor_replay"] = replay
        if not accepted:
            output = replay
            metadata["llm_observation"] = output
        return ToolResult(
            success=result.success,
            output=output,
            error=result.error,
            metadata=metadata,
        )

    def _dedupe_parallel_anchor_results(
        self,
        results: list[tuple[ToolCall, ToolResult]],
    ) -> list[tuple[ToolCall, ToolResult]]:
        """Deduplicate one parallel batch against a stable pre-batch snapshot."""
        snapshot = list(self.state.context_anchors.code)
        accepted_in_batch: list[dict[str, Any]] = []
        deduped: list[tuple[ToolCall, ToolResult]] = []
        for tc, result in results:
            filtered = self._dedupe_result_anchors(
                result,
                existing_anchors=[*snapshot, *accepted_in_batch],
            )
            accepted_in_batch.extend(
                item
                for item in (filtered.metadata or {}).get("raw_evidence_store") or []
                if isinstance(item, dict)
            )
            deduped.append((tc, filtered))
        return deduped

    def _ingest_code_artifacts(
        self,
        tool_call: ToolCall,
        raw_store: list[dict[str, Any]],
    ) -> None:
        enriched = [
            _enrich_anchor_contract(self.harness.project_root, item)
            for item in raw_store if isinstance(item, dict)
        ]
        file_contracts = copy.deepcopy(self.state.context_anchors.file_contracts)
        file_facts = {
            path: list(facts) for path, facts in self.state.context_anchors.file_facts.items()
        }
        schemas = dict(self.state.context_anchors.schema_contracts)
        symbol_anchors: list[dict[str, Any]] = []
        for item in enriched:
            file_path = str(item.get("file") or "")
            if file_path:
                file_contracts[file_path] = {
                    "hash": item.get("file_hash") or item.get("content_hash"),
                    "imports": list(item.get("top_level_imports") or []),
                }
            kind = _anchor_memory_kind(item)
            if kind == "symbol":
                symbol_anchors.append(item)
            elif kind == "schema":
                schemas[_anchor_id(item)] = _fact_text(item)
            else:
                facts = file_facts.setdefault(file_path, [])
                fact = _fact_text(item)
                if fact not in facts:
                    facts.append(fact)
                    del facts[:-20]

        existing_code = list(self.state.context_anchors.code)
        summaries = dict(self.state.context_anchors.summaries)
        purposes = dict(self.state.context_anchors.purposes)
        created_steps = dict(self.state.context_anchors.created_steps)
        superseded: set[str] = set()
        for new_item in symbol_anchors:
            for old_item in existing_code:
                new_symbol = _anchor_symbol_name(new_item)
                old_symbol = _anchor_symbol_name(old_item)
                if (
                    _span_overlap_ratio(new_item, old_item) > 0.90
                    and (not new_symbol or not old_symbol or new_symbol == old_symbol)
                ):
                    superseded.add(_anchor_id(old_item))
            for anchor_id, summary_text in summaries.items():
                old_item = _anchor_item_from_id(anchor_id)
                summary_match = re.search(
                    r"(?:Symbol|函数名称)：\s*`?([^`\n（(]+)", summary_text
                )
                old_symbol = summary_match.group(1).strip() if summary_match else ""
                new_symbol = _anchor_symbol_name(new_item)
                if (
                    old_item
                    and _span_overlap_ratio(new_item, old_item) > 0.90
                    and (not new_symbol or not old_symbol or new_symbol == old_symbol)
                ):
                    superseded.add(anchor_id)
        existing_code = [item for item in existing_code if _anchor_id(item) not in superseded]
        summaries = {key: value for key, value in summaries.items() if key not in superseded}
        purposes = {key: value for key, value in purposes.items() if key not in superseded}
        created_steps = {key: value for key, value in created_steps.items() if key not in superseded}
        merged = self._merge_raw_evidence(self.harness.project_root, [*existing_code, *symbol_anchors])
        purpose = _tool_read_purpose(tool_call)
        for item in symbol_anchors:
            anchor_id = _anchor_id(item)
            purposes[anchor_id] = purpose
            created_steps[anchor_id] = self.state.run_state.step

        messages = self.state.messages_history
        if superseded:
            messages = tuple(
                replace(
                    msg,
                    content="\n".join(
                        line for line in msg.content.splitlines()
                        if not any(anchor_id in line for anchor_id in superseded)
                    ),
                )
                for msg in messages
            )
        self.state = replace(
            self.state,
            context_anchors=replace(
                self.state.context_anchors,
                code=tuple(merged),
                summaries=summaries,
                purposes=purposes,
                created_steps=created_steps,
                file_contracts=file_contracts,
                file_facts={path: tuple(facts) for path, facts in file_facts.items()},
                schema_contracts=schemas,
                last_updated_step=(
                    self.state.run_state.step
                    if symbol_anchors else self.state.context_anchors.last_updated_step
                ),
            ),
            messages_history=messages,
        )

    def _ingest_post_edit_observations(self, tool_call: ToolCall, target_file: str) -> None:
        """Register new DDL blocks from a validated SQL edit into manifest anchors."""
        norm = str(target_file).replace("\\", "/").lstrip("./")
        abs_path = (self.harness.project_root / norm).resolve()
        if not abs_path.is_file():
            return
        try:
            content = abs_path.read_text(encoding="utf-8")
        except OSError:
            return
        observations = observations_from_edited_file(norm, content)
        if not observations:
            return
        raw_store = [dict(item) for item in observations]
        self._ingest_code_artifacts(tool_call, raw_store)
        merged_cache = dict(self.state.search_cache)
        current_raw = list(merged_cache.get("raw_evidence_store") or [])
        merged_cache["raw_evidence_store"] = _touch_raw_evidence_lru(
            current_raw,
            raw_store,
        )
        self.state = replace(self.state, search_cache=merged_cache)
        self._dispatch_run_event(
            RunEvent(
                "evidence_stored",
                observations=tuple(raw_store),
                reason="post_edit_schema_observed",
            )
        )

    async def _inspect_tool_preflight_async(
        self,
        tool_name: str,
        tool_args: dict[str, Any],
    ) -> str | None:
        """Run static + fact-locking preflight for a tool call."""
        embedder_instance = getattr(self, "_embedder", None)
        if embedder_instance is None:
            from src.indexer.embedder import Embedder

            self._embedder = Embedder(
                model=self.settings.embedding_model,
                provider=self.settings.embedding_provider,
            )

        context_builder = getattr(self, "context_builder", None)
        repo_map = None
        if context_builder is not None:
            service = getattr(context_builder, "repo_map_service", None)
            if service is not None:
                try:
                    repo_map = getattr(service, "map", None)
                except Exception:
                    pass

        from src.hooks.before_tool import inspect_tool_request_async

        return await inspect_tool_request_async(
            tool_name,
            tool_args,
            allowed_tools=self._current_step_tools,
            has_compile_error=bool(self._validation_error()),
            search_history=self._grep_search_history,
            repo_map=repo_map,
            embedder=self._embedder,
            embeddings_cache=self._embeddings_cache,
            gravity_controller=None,
            checklist=list(self.state.checklist),
            context_anchors_code=list(self.state.context_anchors.code),
            raw_evidence_store=list(self.state.search_cache.get("raw_evidence_store", [])),
            git_diff=self.state.git_diff,
            modified_files=list(self.state.run_state.changes.files),
            manifest=self.state.run_state.manifest,
            project_root=self.harness.project_root,
            edit_recovery=bool(self.state.run_state.edit_patch_failed),
        )

    def _record_preflight_block(self, tc: ToolCall, err: str) -> ToolResult:
        """Persist a preflight rejection and reopen retrieval when edit was misused."""
        if tc.name == "decision_edit":
            self._dispatch_run_event(
                RunEvent(
                    "preflight_blocked",
                    tool_name="decision_edit",
                    reason=err,
                )
            )
        return ToolResult(
            success=False,
            output=f"Error: {err}",
            error=err,
        )



    async def _collapse_retrieval_before_core_llm(
        self,
        system_prompt: str | None = None,
        user_context: str | None = None,
    ) -> None:
        if system_prompt is None:
            system_prompt = self.context_assembly.load_system_prompt()
        if user_context is None:
            user_context = "test case query"
        exempted_files = set(self.state.run_state.active_files)

        # Helper to estimate current tokens
        def estimate_prompt_tokens() -> int:
            projected_search_cache = _search_cache_for_context(
                _search_cache_view(self.state),
                current_step=self.state.run_state.step,
            )
            shaped_active_files = RuntimeShaper().shape(
                self.state, self.state.run_state.active_files
            )
            shaped_checklist = MemoryShaper().shape(self.state, self.state.checklist)
            assembled_messages = self.context_assembly.assemble(
                user_query=user_context,
                active_files=list(shaped_active_files),
                checklist=list(shaped_checklist),
                git_diff=self.state.git_diff,
                validation_error=self._validation_error(),
                messages_history=list(self.state.messages_history),
                search_cache=projected_search_cache,
                last_tool_result=self._last_tool_result(),
                last_error=self._last_error(),
                modified_files=list(self.state.run_state.changes.files),
            )
            if hasattr(self.llm, "count_messages_tokens") and not hasattr(self.llm.count_messages_tokens, "assert_called"):
                try:
                    res = self.llm.count_messages_tokens(assembled_messages)
                    if isinstance(res, int):
                        return res
                except Exception:
                    pass
            from src.context.window import count_tokens
            total = 0
            for m in assembled_messages:
                total += 4
                content = m.get("content") or ""
                total += count_tokens(content)
            return total + 2

        max_context = getattr(self.settings, "max_context_tokens", 128000)
        token_threshold = int(max_context * 0.70)

        # 1. Below threshold check
        if estimate_prompt_tokens() <= token_threshold:
            return

        # 2. Identify eligible anchors (non-exempted)
        eligible_anchors = [
            item for item in self.state.context_anchors.code
            if item.get("file") not in exempted_files
        ]
        if not eligible_anchors:
            return

        # Group by file
        from collections import defaultdict
        anchors_by_file = defaultdict(list)
        for item in eligible_anchors:
            anchors_by_file[item.get("file")].append(item)

        # Calculate file ages based on oldest anchor step
        file_ages = {}
        for file_path, file_anchors in anchors_by_file.items():
            min_step = min(
                self.state.context_anchors.created_steps.get(
                    _anchor_id(item), self.state.run_state.step
                )
                for item in file_anchors
            )
            file_ages[file_path] = min_step

        # Sort files by age (oldest step first)
        sorted_files = sorted(file_ages.keys(), key=lambda f: file_ages[f])

        folded_eligible_ids = set()
        new_summaries = {}

        for file_path in sorted_files:
            file_anchors = anchors_by_file[file_path]
            summary_parts = []
            for item in file_anchors:
                aid = _anchor_id(item)
                folded_eligible_ids.add(aid)
                summary_parts.append(
                    _collapse_summary(
                        item,
                        dict(self.state.context_anchors.purposes).get(aid, "为当前任务建立代码事实")
                    )
                )
            summary_text = "\n\n".join(summary_parts)
            new_summaries[file_path] = summary_text

            # Apply temporary fold to state copy to check tokens
            temp_summaries = dict(self.state.context_anchors.summaries)
            temp_summaries[file_path] = summary_text
            temp_remaining_code = tuple(
                item for item in self.state.context_anchors.code if _anchor_id(item) not in folded_eligible_ids
            )
            temp_remaining_steps = {
                key: val for key, val in self.state.context_anchors.created_steps.items()
                if key not in folded_eligible_ids
            }
            temp_context_anchors = replace(
                self.state.context_anchors,
                code=temp_remaining_code,
                summaries=temp_summaries,
                created_steps=temp_remaining_steps,
            )
            temp_state = replace(self.state, context_anchors=temp_context_anchors)

            # Re-estimate with temp_state
            def estimate_temp_tokens() -> int:
                projected_search_cache = _search_cache_for_context(
                    _search_cache_view(temp_state),
                    current_step=temp_state.run_state.step,
                )
                shaped_active_files = RuntimeShaper().shape(
                    temp_state, temp_state.run_state.active_files
                )
                shaped_checklist = MemoryShaper().shape(temp_state, temp_state.checklist)
                assembled_messages = self.context_assembly.assemble(
                    user_query=user_context,
                    active_files=list(shaped_active_files),
                    checklist=list(shaped_checklist),
                    git_diff=temp_state.git_diff,
                    validation_error=(
                        "\n".join(temp_state.run_state.validation.issues) or None
                    ),
                    messages_history=list(temp_state.messages_history),
                    search_cache=projected_search_cache,
                    last_tool_result=self._last_tool_result(),
                    last_error=self._last_error(),
                    modified_files=list(temp_state.run_state.changes.files),
                )
                if hasattr(self.llm, "count_messages_tokens") and not hasattr(self.llm.count_messages_tokens, "assert_called"):
                    try:
                        res = self.llm.count_messages_tokens(assembled_messages)
                        if isinstance(res, int):
                            return res
                    except Exception:
                        pass
                from src.context.window import count_tokens
                total = 0
                for m in assembled_messages:
                    total += 4
                    content = m.get("content") or ""
                    total += count_tokens(content)
                return total + 2

            if estimate_temp_tokens() <= token_threshold:
                break

        # Finally, perform actual state update for all folded_eligible_ids
        if folded_eligible_ids:
            self.state = _collapse_retrieval_turn(self.state, folded_eligible_ids, new_summaries)

    async def run(self, user_msg: str) -> AsyncIterator[AgentEvent]:
        from src.harness.checkpoint.session_storage import SessionStorage
        SessionStorage.append_global_history(user_msg)

        self.harness.session_id = self.session_id

        system_prompt = self.context_assembly.load_system_prompt()
        user_context = user_msg
        edit_mode = detect_edit_mode(user_msg)
        self._run_events = []
        self._retrieval_history = ()
        self._grep_search_history = []
        self._last_novelty_value = 1.0
        self._embeddings_cache = {}
        self.stateLayer.clear_cache()
        self.state = replace(
            self.state,
            retrieval_outcome={},
            run_state=start_run(
                user_msg,
                edit_mode=edit_mode,
                max_steps=self.settings.max_turns,
            ),
        )
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
            if hasattr(self.harness, "session_storage"):
                self.harness.session_storage.append_event(self.session_id, event.as_dict())
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

        initial_run_state = self.state.run_state
        # Mutable state initialization (single State object)
        self.state = AssembledState(
            messages_history=(),
            checklist=(),
            git_diff="",
            search_cache={},
            core_context_history=tuple(self._core_context_records),
            run_state=initial_run_state,
        )

        yield AgentEvent(type=EventType.STREAM_START)
        self.harness.phase_metrics.reset_turn()
        self._task_text = user_context.strip()

        step = 0
        protocol_failures = 0
        state_violations = 0
        while self.state.run_state.step < self.state.run_state.max_steps:
            self._dispatch_run_event(RunEvent("step_started"))
            step = self.state.run_state.step
            self.harness.current_step = step
            self._normalize_anchor_memory()
            exempted_files = set(self.state.run_state.active_files)
            collapsible_count = sum(
                1 for item in self.state.context_anchors.code
                if (
                    item.get("file") not in exempted_files
                    and self.state.context_anchors.created_steps.get(
                        _anchor_id(item), self.state.context_anchors.last_updated_step or step
                    ) < step - 1
                )
            )
            if collapsible_count:
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=(
                        "正在压缩已读代码上下文… "
                        f"({collapsible_count} 个代码锚点)"
                    ),
                    data={
                        "spinner_only": True,
                        "phase": "summary",
                        "anchor_count": collapsible_count,
                    },
                )
            await self._collapse_retrieval_before_core_llm(system_prompt, user_context)

            # Update git diff
            git_state = await _get_git_state(self.harness.project_root)
            self.state = replace(self.state, git_diff=git_state)

            # Update decision_edit tool with current active files
            edit_tool = self.tools.get("decision_edit")
            if edit_tool and hasattr(edit_tool, "set_active_files"):
                edit_tool.set_active_files(list(self.state.run_state.active_files))

            self.state = _microcompact_context_state(self.state)
            ready_final = (
                self.state.run_state.phase == RunPhase.RESPONDING
                and self.state.run_state.task_mode == "diagnose"
            )

            # --- RUN PRE-MODEL CONTEXT SHAPERS IN SEQUENCE ---
            # Shaper 4: Conversation compaction
            self.state = conv_shaper.shape(self.state)

            # Shaper 5: Runtime Shaper
            shaped_active_files = run_shaper.shape(
                self.state, self.state.run_state.active_files
            )

            # Shaper 1: System Layer Shaper (injects history summaries if any)
            shaped_sys_prompt = sys_shaper.shape(self.state, system_prompt)

            # Shaper 2: Project Config Shaper
            context_cache = _search_cache_view(self.state)
            rules_text = self.stateLayer.get_user_context(
                list(shaped_active_files),
                context_cache,
            )
            shaped_rules_text = cfg_shaper.shape(self.state, rules_text)
            if ready_final:
                shaped_sys_prompt = system_prompt
                shaped_rules_text = ""

            # Shaper 3: Memory Shaper
            shaped_checklist = mem_shaper.shape(self.state, self.state.checklist)
            projected_search_cache = _search_cache_for_context(
                context_cache,
                current_step=self.state.run_state.step,
            )

            # Project the Step Evidence Manifest from durable verbatim anchors so
            # tool allocation, fact-locking and the execution card all read the
            # same computed sufficiency for this turn.
            anchor_pool = [
                *self.state.context_anchors.code,
                *(self.state.search_cache.get("raw_evidence_store") or []),
            ]
            projected_manifest = project_manifest(
                self.state.run_state.manifest,
                anchor_pool,
                step=self.state.run_state.step,
                task_mode=self.state.run_state.task_mode,
            )
            self.state = replace(
                self.state,
                run_state=replace(self.state.run_state, manifest=projected_manifest),
            )

            self._current_step_tools = determine_allowed_tools(
                self.state,
                None,
                default_tools=ASSEMBLED_TOOL_NAMES,
                has_compile_error=bool(self._validation_error()),
                validation_error=self._validation_error(),
            )
            self.state = replace(
                self.state,
                last_core_tools=self._current_step_tools,
            )
            self._trace_manifest(
                self.state.run_state.manifest,
                allowed_tools=self._current_step_tools,
                retrieval_no_gain_rounds=(
                    self.state.run_state.retrieval_no_gain_rounds
                ),
            )

            # --- CONTEXT ASSEMBLY ---
            # Slice messages history after the compact boundary
            sliced_messages = list(self.state.getMessagesAfterCompactBoundary())
            if ready_final:
                sliced_messages = []

            # Now build the messages payload for LiteLLM/OpenAI
            checklist_str = "\n".join(f"- {item}" for item in shaped_checklist) or "- No checklist items"
            if ready_final:
                context_block = _response_evidence_summary(self.state)
            else:
                manifest = self.state.run_state.manifest
                edit_ready = manifest.sufficiency in {
                    Sufficiency.SUFFICIENT_FOR_EDIT,
                    Sufficiency.SUFFICIENT_FOR_VERIFY,
                }
                context_block = _build_deduped_loaded_anchors_block(
                    projected_search_cache,
                    edit_ready=edit_ready,
                    task_text=self._task_text,
                    manifest=manifest,
                )

            rules_block = (
                f"\n\n### PROJECT RULES & USER CONTEXT ###\n{shaped_rules_text}\n"
                if shaped_rules_text
                else ""
            )

            runtime_state_block = _build_runtime_state_block(
                active_files=list(shaped_active_files),
                checklist_str=checklist_str,
                git_diff=self.state.git_diff,
                validation_error=self._validation_error(),
                last_tool_result=self._last_tool_result(),
                last_error=self._last_error(),
            )

            turn_context_block = ""
            if not ready_final:
                verification_active = post_edit_verification_ready(
                    self.state.run_state,
                    checklist=self.state.checklist,
                )
                card = execution_card(
                    self.state.run_state.manifest,
                    sorted(self._current_step_tools),
                    retrieval_no_gain_rounds=(
                        self.state.run_state.retrieval_no_gain_rounds
                    ),
                    task_slots=sorted(self.state.run_state.evidence.required),
                    task_text=self._task_text,
                    last_grep_error=self.state.run_state.last_grep_error,
                    last_view_error=self.state.run_state.last_view_error,
                    grep_suggested_views=self.state.run_state.grep_suggested_views,
                    edit_burst=bool(self.state.run_state.changes.files)
                    and self.state.run_state.validation.status == "passed"
                    and "decision_edit" in self._current_step_tools
                    and "grep_search" not in self._current_step_tools
                    and not self.state.run_state.edit_patch_failed
                    and not verification_active,
                    edited_files=self.state.run_state.changes.files,
                    edit_patch_failed=self.state.run_state.edit_patch_failed,
                    view_last_round_all_duplicate=(
                        self.state.run_state.view_last_round_all_duplicate
                    ),
                    verification_mode=verification_active,
                )
                turn_context_block = _build_turn_context_block(
                    loaded_anchors=context_block,
                    execution_card_text=card,
                )
            elif context_block.strip():
                turn_context_block = f"\n\n{context_block}"

            # System prompt is static policy; volatile state lives in the user message.
            assembled_sys_content = shaped_sys_prompt
            user_instruction_block = (
                f"Original Request: {user_context}"
                f"{rules_block}\n\n{runtime_state_block}"
                f"{turn_context_block}"
            )

            assembled_messages = []
            assembled_messages.append({"role": "system", "content": assembled_sys_content})
            for msg in sliced_messages:
                assembled_messages.append(msg.to_dict())
            assembled_messages.append({"role": "user", "content": user_instruction_block})

            available_tool_names = self._current_step_tools
            tool_schemas = self.tools.get_schemas(include=available_tool_names)
            context_record = {
                "call": len(self._core_context_records) + 1,
                "step": step,
                "messages": copy.deepcopy(assembled_messages),
                "tools": copy.deepcopy(tool_schemas),
            }
            self._core_context_records.append(context_record)
            self.state = replace(
                self.state,
                core_context_history=tuple(self._core_context_records),
            )

            # Stream thinking start status
            yield AgentEvent(
                type=EventType.STATUS,
                content=f"Agent · step {step}/{self.state.run_state.max_steps} · thinking…",
                data={
                    "spinner_only": True,
                    "llm_loading": True,
                    "phase": "agent",
                },
            )

            # Stream LLM coordinator thoughts & decisions
            response_text = ""
            response: LLMResponse | None = None
            self.harness.phase_metrics.start("assembled_llm", subtask_id=str(step))
            try:
                final_max_tokens = (
                    getattr(self.settings, "ready_final_max_tokens", 1024)
                    if ready_final
                    else None
                )
                if not isinstance(final_max_tokens, int):
                    final_max_tokens = 1024 if ready_final else None
                async for chunk in self._stream_llm(
                    assembled_messages,
                    tool_schemas,
                    max_tokens=final_max_tokens,
                ):
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
                protocol_failures += 1
                self.state = replace(
                    self.state,
                    messages_history=self.state.messages_history + (
                        assistant_message(f"Error recovery triggered: {err_msg}"),
                    )
                )
                yield error_event(f"LLM call failed: {err_msg}. Retrying step {step}...")
                if protocol_failures > 3:
                    self._dispatch_run_event(RunEvent(
                        "run_failed", reason="too many consecutive LLM call failures"
                    ))
                    yield error_event("Too many consecutive LLM call failures. Aborting.")
                    yield AgentEvent(type=EventType.STREAM_END)
                    return
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
                    messages=list(self.agent_telemetry.messages),
                    file_changes=list(self.agent_telemetry.file_changes),
                    current_plan=self.agent_telemetry.current_plan,
                    turn_count=self.agent_telemetry.turn_count,
                    total_tokens_used=self.agent_telemetry.total_tokens_used,
                    total_cost=self.agent_telemetry.total_cost,
                )
                new_agent_state.record_usage(response.usage, cost)
                self.agent_telemetry = new_agent_state
                yield cost_event(
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    cost,
                )

            await self.harness.after_llm_call(response, response.usage)

            if not response.tool_calls and contains_tool_call_markup(
                response.content,
                known_tool_names=ASSEMBLED_TOOL_NAMES,
            ):
                notice = (
                    "Malformed text tool call detected. Do not treat it as a final answer. "
                    "Retry using the native tool-calling API only; do not emit XML/HTML tool tags."
                )
                protocol_failures += 1
                self.state = replace(
                    self.state,
                    messages_history=self.state.messages_history + (
                        assistant_message(response.content or ""),
                        system_message(notice),
                    ),
                )
                yield AgentEvent(type=EventType.STATUS, content=notice)
                if protocol_failures > 3:
                    self._dispatch_run_event(RunEvent(
                        "run_failed", reason="too many malformed text tool calls"
                    ))
                    yield error_event("Too many malformed text tool-call responses. Aborting.")
                    yield AgentEvent(type=EventType.STREAM_END)
                    return
                step += 1
                continue

            # Recovery limits are consecutive: one valid protocol response
            # proves the transport/model pair recovered.
            protocol_failures = 0

            # Update checklist from thoughts/response dynamically
            from src.agent.checklist import parse_checklist_lines

            checklist_items = parse_checklist_lines(response_text)
            if checklist_items:
                self.state = replace(self.state, checklist=checklist_items)

            # Process actions (tool calls)
            if response.tool_calls:
                state_violations = 0
                async for event in self._process_tool_calls(response):
                    yield event

                # Save checkpoint after executing step
                cp_id = await self.harness.save_checkpoint(
                    f"assembled_step_{step}", self.agent_telemetry
                )
                if cp_id:
                    yield AgentEvent(
                        type=EventType.CHECKPOINT_SAVED,
                        data={"checkpoint_id": cp_id},
                    )
                if self.state.run_state.phase == RunPhase.TERMINAL:
                    reason = (
                        self.state.run_state.terminal.reason
                        if self.state.run_state.terminal else "run terminated"
                    )
                    yield error_event(reason)
                    yield AgentEvent(type=EventType.STREAM_END)
                    return

            else:
                # No tool calls means task is complete or clarification requested
                answer = response.content or response_text
                is_clarify = "clarification" in response_text.lower() or "clarify" in response_text.lower() or "?" in answer
                
                # -------------------------------------------------------------
                # Anchor 5: Clarification Request Anchor
                # -------------------------------------------------------------
                if is_clarify:
                    self._dispatch_run_event(RunEvent(
                        "clarification_requested",
                        reason="model requested user clarification",
                    ))
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            assistant_message(answer),
                        )
                    )
                    yield final_answer_event(answer)
                    yield AgentEvent(type=EventType.STREAM_END)
                    return



                self._dispatch_run_event(RunEvent("answer_proposed", answer=answer))
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

        max_steps_reason = f"Reached maximum steps ({self.state.run_state.max_steps})"
        self._dispatch_run_event(RunEvent("run_failed", reason=max_steps_reason))
        yield error_event(
            max_steps_reason,
            {"step_count": self.state.run_state.step},
        )
        yield AgentEvent(type=EventType.STREAM_END)

    def _post_process_tool_result(self, name: str, arguments: dict[str, Any], result: ToolResult) -> ToolResult:
        from dataclasses import is_dataclass

        run_state = getattr(getattr(self, "state", None), "run_state", None)
        if (
            result.success
            and name in {"grep_search", "view_symbol_code"}
            and is_dataclass(run_state)
            and run_state.edit_patch_failed
        ):
            self.state = replace(
                self.state,
                run_state=replace(run_state, edit_patch_failed=False),
            )
        if name == "grep_search" and result.success:
            path = str(arguments.get("path") or ".").strip() or "."
            include = arguments.get("include")
            mode = str(arguments.get("mode") or "default").strip() or "default"
            has_compile_error = bool(self._validation_error())
            searched = list(
                dict.fromkeys(
                    (result.metadata or {}).get("searched_patterns")
                    or arguments.get("patterns")
                    or ([arguments.get("pattern")] if arguments.get("pattern") else [])
                )
            )
            metadata = result.metadata or {}
            is_empty = bool(metadata.get("empty_result"))
            if not is_empty:
                try:
                    payload = json.loads(result.output)
                    if isinstance(payload, dict):
                        is_empty = (
                            int(payload.get("returned_matches") or 0) == 0
                            and int(payload.get("total_matches") or 0) == 0
                        )
                except Exception:
                    is_empty = False
            fingerprint = str(metadata.get("search_fingerprint") or "")
            if not fingerprint and searched:
                fingerprint = grep_search_fingerprint(
                    searched,
                    path=path,
                    include=str(include) if include else None,
                    mode=mode,
                )
            novelty = self._last_novelty_value
            for pattern in searched:
                if not str(pattern or "").strip():
                    continue
                evaluation = evaluate_search_intent(
                    str(pattern), path, self._grep_search_history, has_compile_error
                )
                novelty = arguments.get("_novelty_score", evaluation["novelty"])
                entry: dict[str, Any] = {
                    "file": path,
                    "pattern": str(pattern),
                    "patterns": searched,
                    "include": include,
                    "mode": mode,
                    "fingerprint": fingerprint,
                }
                if is_empty:
                    entry["empty"] = True
                self._grep_search_history.append(entry)
                self._last_novelty_value = novelty
            uncertainty = 1.0 if has_compile_error else 0.2
            retrieval_weight = 0.3 * novelty + 0.7 * uncertainty
            action = "PROCEED_WITH_TRUNCATED_DATA" if retrieval_weight <= 0.35 else "PROCEED_WITH_FULL_DATA"
            
            if action == "PROCEED_WITH_TRUNCATED_DATA":
                try:
                    import json
                    payload = json.loads(result.output)
                    if isinstance(payload, dict) and "matches" in payload:
                        matches = payload["matches"]
                        total_matches = len(matches)
                        if total_matches > 3:
                            truncated_matches = matches[:3]
                            payload["matches"] = truncated_matches
                            payload["returned_matches"] = 3
                            payload["truncated"] = True
                            new_output = json.dumps(payload, ensure_ascii=False, indent=2)
                            disclaimer = f"\n[... {total_matches - 3} matches hidden to reduce context noise. Please narrow your search pattern or use specific symbol view if you are looking for details.]"
                            new_output_with_disclaimer = new_output + disclaimer
                            new_metadata = dict(result.metadata or {})
                            new_metadata["raw_evidence_store"] = truncated_matches
                            new_metadata["returned_matches"] = 3
                            new_metadata["truncated"] = True
                            return ToolResult(
                                success=result.success,
                                output=new_output_with_disclaimer,
                                error=result.error,
                                metadata=new_metadata
                            )
                except Exception as e:
                    log.warning("Failed to truncate grep_search result: %s", e)
        return result

    async def _process_tool_calls(
        self, response: LLMResponse
    ) -> AsyncIterator[AgentEvent]:
        if not response.tool_calls:
            return



        self.state = replace(
            self.state,
            messages_history=self.state.messages_history + (
                assistant_message(response.content or "", response.tool_calls),
            )
        )

        executed_tool_signals = []
        retrieval_round_results: list[tuple[str, ToolResult]] = []
        first_error_to_structure = None
        round_grep_error = ""
        round_view_error = ""
        round_suggested_views: list[dict[str, Any]] = []
        edit_applied_this_round = False

        # Classify and batch tools: parallelizable tools run in parallel, sequential tools run one-by-one
        batches: list[list[ToolCall]] = []
        current_batch: list[ToolCall] = []
        for tc in response.tool_calls:


            if tc.name in PARALLEL_RETRIEVAL_TOOLS:
                current_batch.append(tc)
            else:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                batches.append([tc])
        if current_batch:
            batches.append(current_batch)

        preflight_results: dict[str, ToolResult] = {}
        for tc in response.tool_calls:
            effects = self._dispatch_run_event(RunEvent(
                "tool_requested",
                tool_name=tc.name,
            ))

        for batch in batches:
            if len(batch) > 1:
                # Parallel execution of safe retrieval tools
                for tc in batch:
                    yield tool_call_event(tc.name, tc.arguments)
                    yield AgentEvent(
                        type=EventType.STATUS,
                        content=get_tool_status_text(tc.name, tc.arguments),
                        data={"spinner_only": True, "phase": "executor"},
                    )

                async def run_one(tc: ToolCall) -> tuple[ToolCall, ToolResult]:
                    if tc.id in preflight_results:
                        return tc, preflight_results[tc.id]
                    tool_args = _prepare_tool_arguments(
                        tc.name,
                        dict(tc.arguments),
                        hint_text=self._task_text,
                    )
                    tool_cache = _search_cache_view(self.state)
                    if tool_cache:
                        tool_args["_search_cache"] = tool_cache
                    if tc.name == "decision_edit":
                        tool_args["_manifest"] = self.state.run_state.manifest
                    if tc.name == "grep_search":
                        self._inject_grep_execute_args(tool_args)

                    # Run preflight constraints and convergence checks
                    embedder_instance = getattr(self, "_embedder", None)
                    if embedder_instance is None:
                        from src.indexer.embedder import Embedder
                        self._embedder = Embedder(
                            model=self.settings.embedding_model,
                            provider=self.settings.embedding_provider,
                        )
                    
                    context_builder = getattr(self, "context_builder", None)
                    repo_map = None
                    if context_builder is not None:
                        service = getattr(context_builder, "repo_map_service", None)
                        if service is not None:
                            try:
                                repo_map = getattr(service, "map", None)
                            except Exception:
                                pass

                    from src.hooks.before_tool import inspect_tool_request_async
                    err = await inspect_tool_request_async(
                        tc.name,
                        tool_args,
                        allowed_tools=self._current_step_tools,
                        has_compile_error=bool(self._validation_error()),
                        search_history=self._grep_search_history,
                        repo_map=repo_map,
                        embedder=self._embedder,
                        embeddings_cache=self._embeddings_cache,
                        gravity_controller=None,
                        checklist=list(self.state.checklist),
                        context_anchors_code=list(self.state.context_anchors.code),
                        raw_evidence_store=list(self.state.search_cache.get("raw_evidence_store", [])),
                        git_diff=self.state.git_diff,
                        modified_files=list(self.state.run_state.changes.files),
                        manifest=self.state.run_state.manifest,
                        project_root=self.harness.project_root,
                        edit_recovery=bool(self.state.run_state.edit_patch_failed),
                    )
                    warning_feedback = None
                    is_hard_block = False
                    is_mock_success = False
                    if err:
                        if err.startswith("SUCCESS:"):
                            is_mock_success = True
                        else:
                            is_hard_block = True
                            warning_feedback = err

                    self.harness.phase_metrics.start(
                        f"tool_{tc.name}", subtask_id=str(self.state.run_state.step)
                    )
                    success = False
                    try:
                        if is_hard_block:
                            result = ToolResult(
                                success=False,
                                output=f"Error: {err}",
                                error=err,
                            )
                        elif is_mock_success:
                            output_msg = err[len("SUCCESS:"):].strip()
                            result = ToolResult(
                                success=True,
                                output=output_msg,
                                metadata={"is_mock_success": True},
                            )
                            success = True
                        else:
                            if repo_map is not None and tc.name in {"decision_edit", "grep_search"}:
                                tool_args["_repo_map"] = repo_map
                            result = await self.tools.call(tc.name, tool_args)
                            result = apply_after_tool_output_limit(tc.name, result)
                            result = apply_post_tool_context_hook(tc.name, tc.arguments, result)
                            result = self._post_process_tool_result(tc.name, tc.arguments, result)
                            success = result.success
                            if warning_feedback:
                                result.output = (
                                    f"Error: {warning_feedback}\n"
                                    f"[Note: This search was executed, but convergence feedback was triggered.]\n\n"
                                    f"[TOOL OUTPUT]\n"
                                    f"{result.output}"
                                )
                                result.error = warning_feedback
                                result.success = False
                                success = False
                    except Exception as exc:
                        result = ToolResult(
                            success=False,
                            output="",
                            error=f"Tool task failed: {type(exc).__name__}: {exc}",
                        )
                    finally:
                        self.harness.phase_metrics.end(
                            f"tool_{tc.name}",
                            subtask_id=str(self.state.run_state.step),
                            verdict="success" if success else "fail",
                        )
                    return tc, result

                results = await asyncio.gather(*[run_one(tc) for tc in batch])
                results = self._dedupe_parallel_anchor_results(results)

                for tc, result in results:
                    if tc.name in RETRIEVAL_TOOLS:
                        result = self._dedupe_result_anchors(result)
                        retrieval_round_results.append((tc.name, result))
                    task_completion = self._apply_context_update(result)
                    display_output = result.output if result.success else f"Error: {result.error}"
                    self._trace_tool_result(tc.name, display_output, success=result.success)

                    norm_args = _prepare_tool_arguments(
                        tc.name,
                        dict(tc.arguments),
                        hint_text=self._task_text,
                    )
                    executed_tool_signals.append(
                        _retrieval_tool_signal(tc, result, arguments=norm_args)
                    )
                    round_grep_error, round_suggested_views = _observe_grep_round(
                        tc.name,
                        result,
                        round_grep_error=round_grep_error,
                        round_suggested_views=round_suggested_views,
                    )

                    if not result.success:
                        if not first_error_to_structure:
                            first_error_to_structure = result.error or result.output

                    if result.success:
                        if result.metadata:
                            merged_cache = dict(self.state.search_cache)
                            if "search_output" in result.metadata:
                                new_output = result.metadata["search_output"]
                                retrieval_query = str(tc.arguments.get("query", ""))
                                existing_output = merged_cache.get("search_output")
                                if existing_output:
                                    try:
                                        pack1 = json.loads(existing_output)
                                        pack2 = json.loads(new_output)
                                        merged_pack = self._merge_evidence_packs(pack1, pack2)
                                        merged_cache["search_output"] = json.dumps(merged_pack, ensure_ascii=False, indent=2)
                                    except Exception as exc:
                                        log.warning("Failed to merge evidence packs in parallel retrieve: %s", exc)
                                        merged_cache["search_output"] = new_output
                                else:
                                    merged_cache["search_output"] = new_output
                                self.state = replace(
                                    self.state,
                                    context_anchors=replace(
                                        self.state.context_anchors,
                                        last_updated_step=self.state.run_state.step,
                                    ),
                                )
                            if "edit_context" in result.metadata:
                                merged_cache["edit_context"] = result.metadata["edit_context"]
                            if "snippets" in result.metadata:
                                merged_cache["snippets"] = result.metadata["snippets"]
                            symbol_projection = result.metadata.get("symbol_projection")
                            if symbol_projection:
                                existing_projections = list(merged_cache.get("symbol_projections", []))
                                existing_projections.append(symbol_projection)
                                merged_cache["symbol_projections"] = existing_projections[-8:]
                            
                            raw_store = result.metadata.get("raw_evidence_store")
                            refresh_store = result.metadata.get("refresh_evidence_store")
                            if raw_store:
                                self._ingest_code_artifacts(tc, raw_store)
                            if raw_store or refresh_store:
                                current_raw = list(merged_cache.get("raw_evidence_store") or [])
                                merged_cache["raw_evidence_store"] = _touch_raw_evidence_lru(
                                    current_raw,
                                    [
                                        *(
                                            item for item in (refresh_store or [])
                                            if isinstance(item, dict)
                                        ),
                                        *(
                                            item for item in (raw_store or [])
                                            if isinstance(item, dict)
                                        ),
                                    ],
                                )

                            self.state = replace(self.state, search_cache=merged_cache)

                        if tc.name == "codebase_retrieve":
                            retrieved = result.metadata.get("retrieved_files") or []
                            
                            self.state = replace(
                                self.state,
                                messages_history=self.state.messages_history + (
                                    tool_message(tc.id, _tool_history_receipt(tc, result)),
                                )
                            )
                            display_output = f"Retrieved codebase context. Loaded {len(retrieved)} files."
                        else:
                            self.state = replace(
                                self.state,
                                messages_history=self.state.messages_history + (
                                    tool_message(tc.id, _tool_history_receipt(tc, result)),
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

            else:
                # Sequential execution
                tc = batch[0]
                yield tool_call_event(tc.name, tc.arguments)

                if tc.id in preflight_results:
                    result = preflight_results[tc.id]
                    display_output = result.output if result.success else f"Error: {result.error}"
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, display_output),
                        ),
                    )
                    self._trace_tool_result(tc.name, display_output, success=result.success)
                    yield tool_result_event(tc.name, display_output, success=result.success)
                    signal = "retrieval_gate" if result.success else "run_state_rejected"
                    executed_tool_signals.append(f"{tc.name}(...) -> {signal}")
                    if not result.success and not first_error_to_structure:
                        first_error_to_structure = result.error
                    continue

                if tc.name not in self._current_step_tools:
                    denied_msg = (
                        f"Tool '{tc.name}' is not available in assembled mode. "
                        f"Allowed tools: {', '.join(sorted(self._current_step_tools))}."
                    )
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, denied_msg),
                        )
                    )
                    self._trace_tool_result(tc.name, denied_msg, success=False)
                    yield tool_result_event(tc.name, denied_msg, success=False)
                    executed_tool_signals.append(f"{tc.name}(...) -> not_allowed")
                    if not first_error_to_structure:
                        first_error_to_structure = denied_msg
                    continue

                tool_args = _prepare_tool_arguments(
                    tc.name,
                    dict(tc.arguments),
                    hint_text=self._task_text,
                )
                tool_cache = _search_cache_view(self.state)
                if tool_cache:
                    tool_args["_search_cache"] = tool_cache
                if tc.name == "grep_search":
                    self._inject_grep_execute_args(tool_args)

                preflight_err = await self._inspect_tool_preflight_async(tc.name, tool_args)
                if preflight_err:
                    if preflight_err.startswith("SUCCESS:"):
                        output_msg = preflight_err[len("SUCCESS:"):].strip()
                        result = ToolResult(
                            success=True,
                            output=output_msg,
                            metadata={"is_mock_success": True},
                        )
                        display_output = output_msg
                        self.state = replace(
                            self.state,
                            messages_history=self.state.messages_history + (
                                tool_message(tc.id, display_output),
                            ),
                        )
                        self._trace_tool_result(tc.name, display_output, success=True)
                        yield tool_result_event(tc.name, display_output, success=True)
                        executed_tool_signals.append(f"{tc.name}(...) -> preflight_mock")
                        continue

                    result = self._record_preflight_block(tc, preflight_err)
                    if tc.name in RETRIEVAL_TOOLS:
                        retrieval_round_results.append((tc.name, result))
                    display_output = (
                        _tool_history_receipt(tc, result)
                        if tc.name in RETRIEVAL_TOOLS
                        and is_duplicate_retrieval_result(tc.name, result)
                        else f"Error: {preflight_err}"
                    )
                    self.state = replace(
                        self.state,
                        messages_history=self.state.messages_history + (
                            tool_message(tc.id, display_output),
                        ),
                    )
                    self._trace_tool_result(tc.name, display_output, success=False)
                    yield tool_result_event(tc.name, display_output, success=False)
                    executed_tool_signals.append(
                        _retrieval_tool_signal(tc, result, arguments=dict(tc.arguments))
                        if tc.name in RETRIEVAL_TOOLS
                        else f"{tc.name}(...) -> preflight_blocked"
                    )
                    if not first_error_to_structure:
                        first_error_to_structure = preflight_err
                    continue

                # Check permission (after preflight — invalid args never prompt)
                approved = True
                permission_event_started = False
                tool = self.tools.get(tc.name)
                if tool is not None:
                    check = self.permissions.check(tc.name, tool.risk_level)
                    if check.allowed:
                        approved = True
                    elif not check.needs_prompt:
                        self._dispatch_run_event(RunEvent(
                            "permission_required",
                            reason=f"Permission required for {tc.name}",
                        ))
                        permission_event_started = True
                        approved = False
                    else:
                        self._dispatch_run_event(RunEvent(
                            "permission_required",
                            reason=f"Permission required for {tc.name}",
                        ))
                        permission_event_started = True
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
                    self._dispatch_run_event(RunEvent(
                        "permission_denied",
                        reason=denied_msg,
                    ))
                    executed_tool_signals.append(f"{tc.name}(...) -> denied")
                    if not first_error_to_structure:
                        first_error_to_structure = denied_msg
                    break

                if permission_event_started:
                    self._dispatch_run_event(RunEvent("permission_granted"))

                # Yield dynamic thinking / loading status for this tool
                yield AgentEvent(
                    type=EventType.STATUS,
                    content=get_tool_status_text(tc.name, tc.arguments),
                    data={"spinner_only": True, "phase": "executor"},
                )

                self.harness.phase_metrics.start(
                    f"tool_{tc.name}", subtask_id=str(self.state.run_state.step)
                )
                success = False
                try:
                    progress_queue = asyncio.Queue()
                    self.harness.progress_callback = progress_queue.put_nowait

                    tool_task = asyncio.create_task(
                        self._run_tool_with_sentinel(tc.name, tool_args, progress_queue),
                        name=f"action-layer:{tc.name}:{tc.id}",
                    )
                    while True:
                        progress_text = await progress_queue.get()
                        if progress_text is None:
                            break
                        yield AgentEvent(
                            type=EventType.STATUS,
                            content=progress_text,
                            data={"phase": "executor", "spinner_only": True},
                        )
                    result = await tool_task
                    if hasattr(self.harness, "progress_callback"):
                        delattr(self.harness, "progress_callback")
                    result = apply_after_tool_output_limit(tc.name, result)
                    result = apply_post_tool_context_hook(tc.name, tc.arguments, result)
                    result = self._post_process_tool_result(tc.name, tc.arguments, result)
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
                        subtask_id=str(self.state.run_state.step),
                        verdict="success" if success else "fail",
                    )

                result = self._dedupe_result_anchors(result)
                if tc.name in RETRIEVAL_TOOLS:
                    retrieval_round_results.append((tc.name, result))
                display_output = result.output if result.success else f"Error: {result.error}"
                task_completion = self._apply_context_update(result)
                self._trace_tool_result(tc.name, display_output, success=result.success)

                norm_args = _prepare_tool_arguments(
                    tc.name,
                    dict(tc.arguments),
                    hint_text=self._task_text,
                )
                executed_tool_signals.append(
                    _retrieval_tool_signal(tc, result, arguments=norm_args)
                )
                round_grep_error, round_suggested_views = _observe_grep_round(
                    tc.name,
                    result,
                    round_grep_error=round_grep_error,
                    round_suggested_views=round_suggested_views,
                )

                if not result.success:
                    if not first_error_to_structure:
                        first_error_to_structure = result.error or result.output
                    if tc.name == "view_symbol_code":
                        round_view_error = str(result.error or result.output or "")

                if result.success:
                    if result.metadata:
                        merged_cache = dict(self.state.search_cache)
                        if "search_output" in result.metadata:
                            new_output = result.metadata["search_output"]
                            existing_output = merged_cache.get("search_output")
                            if existing_output:
                                try:
                                    pack1 = json.loads(existing_output)
                                    pack2 = json.loads(new_output)
                                    merged_pack = self._merge_evidence_packs(pack1, pack2)
                                    merged_cache["search_output"] = json.dumps(merged_pack, ensure_ascii=False, indent=2)
                                except Exception as exc:
                                    log.warning("Failed to merge evidence packs in sequential retrieve: %s", exc)
                                    merged_cache["search_output"] = new_output
                            else:
                                merged_cache["search_output"] = new_output
                            self.state = replace(
                                self.state,
                                context_anchors=replace(
                                    self.state.context_anchors,
                                    last_updated_step=self.state.run_state.step,
                                ),
                            )
                        if "edit_context" in result.metadata:
                            merged_cache["edit_context"] = result.metadata["edit_context"]
                        if "snippets" in result.metadata:
                            merged_cache["snippets"] = result.metadata["snippets"]
                        symbol_projection = result.metadata.get("symbol_projection")
                        if symbol_projection:
                            existing_projections = list(merged_cache.get("symbol_projections", []))
                            existing_projections.append(symbol_projection)
                            merged_cache["symbol_projections"] = existing_projections[-8:]
                        
                        raw_store = result.metadata.get("raw_evidence_store")
                        refresh_store = result.metadata.get("refresh_evidence_store")
                        if raw_store:
                            self._ingest_code_artifacts(tc, raw_store)
                        if raw_store or refresh_store:
                            current_raw = list(merged_cache.get("raw_evidence_store") or [])
                            merged_cache["raw_evidence_store"] = _touch_raw_evidence_lru(
                                current_raw,
                                [
                                    *(
                                        item for item in (refresh_store or [])
                                        if isinstance(item, dict)
                                    ),
                                    *(
                                        item for item in (raw_store or [])
                                        if isinstance(item, dict)
                                    ),
                                ],
                            )

                        self.state = replace(self.state, search_cache=merged_cache)

                    if tc.name == "codebase_retrieve":
                        retrieved = result.metadata.get("retrieved_files") or []
                        
                        self.state = replace(
                            self.state,
                            messages_history=self.state.messages_history + (
                                tool_message(tc.id, _tool_history_receipt(tc, result)),
                            )
                        )
                        display_output = f"Retrieved codebase context. Loaded {len(retrieved)} files."
                    elif tc.name == "decision_edit":
                        target = tc.arguments.get("target_file", "unknown")
                        self._dispatch_run_event(RunEvent(
                            "edit_applied",
                            file=str(target),
                        ))
                        self._dispatch_run_event(RunEvent("validation_finished"))
                        self._ingest_post_edit_observations(tc, str(target))
                        import difflib
                        diff_text = ""
                        # Default: treat a validated edit as a real change. Only a
                        # positively-detected empty diff (the LLM abusing edit as a
                        # pseudo-viewer) is excluded so rounds_since_last_edit can
                        # advance and reopen real verification tools next round.
                        edit_produced_change = True
                        exec_res = result.metadata.get("execution") if result.metadata else None
                        if exec_res:
                            orig = getattr(exec_res, "original_content", "") or (exec_res.get("original_content", "") if isinstance(exec_res, dict) else "")
                            attempted = getattr(exec_res, "attempted_content", "") or (exec_res.get("attempted_content", "") if isinstance(exec_res, dict) else "")
                            if orig and attempted:
                                orig_lines = orig.splitlines(keepends=True)
                                att_lines = attempted.splitlines(keepends=True)
                                diff_lines = list(difflib.unified_diff(
                                    orig_lines,
                                    att_lines,
                                    fromfile=f"a/{target}",
                                    tofile=f"b/{target}",
                                ))
                                if diff_lines:
                                    diff_text = "\n\nApplied Diff:\n```diff\n" + "".join(diff_lines) + "\n```"
                                else:
                                    edit_produced_change = False
                        edit_applied_this_round = edit_applied_this_round or edit_produced_change

                        diff_suffix = f"\n\n{diff_text}" if diff_text else ""
                        if task_completion:
                            summary = json.dumps(task_completion, ensure_ascii=False) + diff_suffix
                        else:
                            summary = f"decision_edit applied to {target}: validation passed.{diff_suffix}"
                        self.state = replace(
                            self.state,
                            messages_history=self.state.messages_history + (
                                tool_message(tc.id, summary),
                            )
                        )
                        display_output = "decision_edit validation passed."
                    else:
                        self.state = replace(
                            self.state,
                            messages_history=self.state.messages_history + (
                                tool_message(tc.id, _tool_history_receipt(tc, result)),
                            )
                        )
                else:
                    if tc.name == "decision_edit":
                        exec_res = result.metadata.get("execution") if result.metadata else None
                        val_res = result.metadata.get("validation") if result.metadata else None
                        is_validation_failure = (exec_res and exec_res.success) and (val_res and not val_res.success)

                        if is_validation_failure:
                            target = str(tc.arguments.get("target_file") or "unknown")
                            self._dispatch_run_event(RunEvent("edit_applied", file=target))
                            self._dispatch_run_event(RunEvent(
                                "validation_finished",
                                issues=(str(val_res.error or result.error or "validation failed"),),
                                reason="decision_edit validation failed",
                                fingerprint=str(val_res.error or result.error or "validation failed"),
                            ))
                            val_err = val_res.error or result.error or "unknown validation error"
                            summary = (
                                f"❌ 【自动化代码验证失败】：文件 `{tc.arguments.get('target_file')}` 补丁应用成功但验证未通过，修改已回滚。\n"
                                f"👉 具体的验证器报错如下，请根据此错误重新构思修改指令：\n"
                                f"```\n{val_err}\n```"
                            )
                            self.state = replace(
                                self.state,
                                messages_history=self.state.messages_history + (
                                    tool_message(tc.id, summary),
                                )
                            )
                        else:
                            self._dispatch_run_event(RunEvent(
                                "tool_failed",
                                tool_name="decision_edit",
                                reason=str(result.error or result.output or "patch failed"),
                            ))
                            err_msg = result.error or result.output
                            summary = f"❌ 【补丁生成失败】：无法匹配或应用补丁到 `{tc.arguments.get('target_file')}`。错误：{err_msg}"
                            self.state = replace(
                                self.state,
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

        # Keep display projections while RunState remains the sole flow controller.
        last_tool_result = "; ".join(executed_tool_signals) if executed_tool_signals else None
        self._dispatch_run_event(RunEvent(
            "tool_round_observed",
            reason=last_tool_result or "",
            issues=((str(first_error_to_structure),) if first_error_to_structure else ()),
            retrieval_attempted=any(
                signal.startswith(("grep_search(", "view_symbol_code(", "codebase_retrieve("))
                for signal in executed_tool_signals
            ),
            view_all_duplicate=view_round_all_duplicate(retrieval_round_results),
            grep_error=round_grep_error,
            view_error=round_view_error,
            grep_suggested_views=tuple(round_suggested_views),
            edit_applied_this_round=edit_applied_this_round,
        ))

    def _merge_raw_evidence(self, project_root: Path, raw_evidence_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
        raw_evidence_list = [
            {
                **item,
                "code": item.get("code") or item.get("match_line") or "",
                "locator_only": bool(item.get("locator_only") or "match_line" in item),
            }
            for item in raw_evidence_list
            if isinstance(item, dict) and item.get("file") and item.get("span")
        ]
        raw_evidence_list = _dedupe_code_anchors(raw_evidence_list)
        # Retrieval anchors are atomic: (file, span) is their identity.  Combining
        # nearby spans would destroy that identity and its first-hop relations.
        if any("related_functions" in item for item in raw_evidence_list):
            merged: dict[tuple[Any, ...], dict[str, Any]] = {}
            for item in raw_evidence_list:
                key = _anchor_key(item)
                if key[0]:
                    merged[key] = item
            return list(merged.values())

        by_file: dict[str, list[dict[str, Any]]] = {}
        for item in raw_evidence_list:
            by_file.setdefault(item["file"], []).append(item)

        merged_evidence = []
        for file_path, items in by_file.items():
            if not items:
                continue
            # Sort by span start line
            items.sort(key=lambda x: x["span"][0])
            
            merged_spans: list[dict[str, Any]] = []
            for item in items:
                if not merged_spans:
                    merged_spans.append({
                        "file": item["file"],
                        "span": list(item["span"]),
                        "code": item["code"],
                        "locator_only": bool(item.get("locator_only")),
                        "hash": item.get("hash") or item.get("content_hash") or ""
                    })
                else:
                    last = merged_spans[-1]
                    s1, e1 = last["span"]
                    s2, e2 = item["span"]
                    # Check overlap or gap <= 5 lines (adjacent within 5 lines threshold)
                    if s2 <= e1 + 6:
                        last["span"] = [s1, max(e1, e2)]
                    else:
                        merged_spans.append({
                            "file": item["file"],
                            "span": list(item["span"]),
                            "code": item["code"],
                            "locator_only": bool(item.get("locator_only")),
                            "hash": item.get("hash") or item.get("content_hash") or ""
                        })

            abs_path = (project_root / file_path).resolve()
            for m_item in merged_spans:
                start_line, end_line = m_item["span"]
                content = None
                if abs_path.is_file():
                    try:
                        lines = abs_path.read_text(encoding="utf-8").splitlines()
                        start = max(1, start_line)
                        end = min(len(lines), end_line)
                        content = "\n".join(lines[start - 1 : end])
                    except Exception as exc:
                        log.warning("Failed to read merged span from file %s: %s", file_path, exc)

                if content is not None:
                    m_item["code"] = content
                else:
                    # Fallback reconstruction logic if file is not on disk
                    fallback_parts = []
                    last_end = start_line - 1
                    for it in sorted(items, key=lambda x: x["span"][0]):
                        it_start, it_end = it["span"]
                        if it_start <= end_line and it_end >= start_line:
                            it_lines = it["code"].splitlines()
                            overlap_start = max(it_start, last_end + 1)
                            overlap_end = min(it_end, end_line)
                            if overlap_start <= overlap_end:
                                idx_start = overlap_start - it_start
                                idx_end = overlap_end - it_start + 1
                                fallback_parts.extend(it_lines[idx_start:idx_end])
                                last_end = overlap_end
                    m_item["code"] = "\n".join(fallback_parts)

                import hashlib
                m_item["hash"] = hashlib.md5(m_item["code"].encode("utf-8")).hexdigest()[:6]
                merged_evidence.append(m_item)

        return merged_evidence

    def _merge_evidence_packs(self, pack1: Any, pack2: Any) -> Any:
        if isinstance(pack1, list) and isinstance(pack2, list):
            return _dedupe_code_anchors(
                [item for item in [*pack1, *pack2] if isinstance(item, dict)]
            )
        # 1. Query
        q1 = pack1.get("query", "")
        q2 = pack2.get("query", "")
        merged_query = q1
        if q2 and q2 != q1:
            if q1:
                merged_query = f"{q1}; {q2}"
            else:
                merged_query = q2

        # 2. Intent
        intent1 = pack1.get("intent", {})
        intent2 = pack2.get("intent", {})
        primary = intent2.get("primary") or intent1.get("primary") or "explain"
        domains = list(set(intent1.get("domain", [])) | set(intent2.get("domain", [])))

        # 3. Grounding
        g1 = pack1.get("grounding", {})
        g2 = pack2.get("grounding", {})
        
        # Merge files
        files_map = {}
        for f in g1.get("files", []):
            files_map[f["path"]] = f.get("score", 0.5)
        for f in g2.get("files", []):
            path = f["path"]
            files_map[path] = max(files_map.get(path, 0.0), f.get("score", 0.5))
        merged_files = [{"path": k, "score": round(v, 2)} for k, v in files_map.items()]

        # Merge symbols
        symbols_map = {}
        for s in g1.get("symbols", []):
            key = (s["name"], s["file"])
            symbols_map[key] = s
        for s in g2.get("symbols", []):
            key = (s["name"], s["file"])
            if key in symbols_map:
                if s.get("score", 0.0) > symbols_map[key].get("score", 0.0):
                    symbols_map[key] = s
            else:
                symbols_map[key] = s
        merged_symbols = list(symbols_map.values())

        # 4. Evidence
        evidence_map = {}
        for ev in pack1.get("evidence", []):
            key = (ev["symbol"], ev["file"])
            evidence_map[key] = ev
        for ev in pack2.get("evidence", []):
            key = (ev["symbol"], ev["file"])
            if key in evidence_map:
                existing = evidence_map[key]
                what_it_does = ev.get("what_it_does") or ""
                if len(what_it_does) < len(existing.get("what_it_does", "")):
                    what_it_does = existing.get("what_it_does", "")
                
                why_relevant = ev.get("why_relevant") or ""
                if len(why_relevant) < len(existing.get("why_relevant", "")):
                    why_relevant = existing.get("why_relevant", "")
                
                risk_flags = list(set(existing.get("risk_flags", [])) | set(ev.get("risk_flags", [])))
                
                evidence_map[key] = {
                    "type": ev.get("type") or existing.get("type"),
                    "file": ev["file"],
                    "symbol": ev["symbol"],
                    "span": ev.get("span") or existing.get("span"),
                    "what_it_does": what_it_does,
                    "why_relevant": why_relevant,
                    "how_it_answers_query": (
                        ev.get("how_it_answers_query")
                        or existing.get("how_it_answers_query")
                        or why_relevant
                    ),
                    "risk_flags": risk_flags
                }
            else:
                evidence_map[key] = ev
        merged_evidence = list(evidence_map.values())

        # 5. Dependencies
        dep_map = {}
        for d in pack1.get("dependencies", []):
            key = (d.get("from"), d.get("to"), d.get("type"))
            dep_map[key] = d
        for d in pack2.get("dependencies", []):
            key = (d.get("from"), d.get("to"), d.get("type"))
            dep_map[key] = d
        merged_dependencies = list(dep_map.values())

        # 6. Graph
        graph1 = pack1.get("retrieval_graph", {})
        graph2 = pack2.get("retrieval_graph", {})
        merged_nodes = list(set(graph1.get("nodes", [])) | set(graph2.get("nodes", [])))
        
        edges_set = set()
        for e in graph1.get("edges", []):
            if isinstance(e, list) and len(e) >= 2:
                edges_set.add((e[0], e[1]))
        for e in graph2.get("edges", []):
            if isinstance(e, list) and len(e) >= 2:
                edges_set.add((e[0], e[1]))
        merged_edges = [[u, v] for u, v in edges_set]

        # 7. Coverage
        cov1 = pack1.get("coverage", {})
        cov2 = pack2.get("coverage", {})
        merged_coverage = {}
        for k in set(cov1.keys()) | set(cov2.keys()):
            merged_coverage[k] = max(cov1.get(k, 0.0), cov2.get(k, 0.0))

        return {
            "query": merged_query,
            "intent": {
                "primary": primary,
                "domain": domains
            },
            "grounding": {
                "files": merged_files,
                "symbols": merged_symbols
            },
            "evidence": merged_evidence,
            "dependencies": merged_dependencies,
            "retrieval_graph": {
                "nodes": merged_nodes,
                "edges": merged_edges
            },
            "coverage": merged_coverage
        }

    def _parse_structured_error(self, error_str: str | None) -> dict[str, Any]:
        structured = {
            "error_type": "ValidationError",
            "file": "unknown",
            "line": 0,
            "message": "Validation failed"
        }
        if not error_str:
            return structured
        
        error_str = str(error_str)
        if "BLOCK:" in error_str or "BLOCK_SEARCH_FORCE_EDIT:" in error_str:
            structured["error_type"] = "ConvergenceGateWarning"
            structured["message"] = error_str.strip()
            return structured
        
        # 1. Try to find DEAD_SQL_ALIAS
        if "DEAD_SQL_ALIAS" in error_str:
            structured["error_type"] = "SchemaValidationError"
            structured["message"] = "DEAD_SQL_ALIAS: Schema validation failed due to invalid SQL alias."
            file_match = re.search(r'([\w\-./]+\.py)', error_str)
            if file_match:
                structured["file"] = file_match.group(1).split("/")[-1]
            return structured

        # 2. Try to find standard Python traceback patterns: File "...", line \d+
        tb_match = re.search(r'File\s+["\']([^"\']+)["\'],\s+line\s+(\d+)', error_str)
        if tb_match:
            file_path = tb_match.group(1)
            if "/" in file_path:
                file_path = file_path.split("/")[-1]
            structured["file"] = file_path
            structured["line"] = int(tb_match.group(2))
            
            lines = [line.strip() for line in error_str.splitlines() if line.strip()]
            if lines:
                last_line = lines[-1]
                exc_match = re.match(r'^(\w+Error|Exception):\s*(.*)$', last_line)
                if exc_match:
                    structured["error_type"] = exc_match.group(1)
                    structured["message"] = exc_match.group(2)
                else:
                    structured["message"] = last_line
            return structured

        # 3. General linter / compiler error patterns: e.g. list.py:32:15: SyntaxError: ...
        colon_match = re.search(r'([\w\-./]+\.py):(\d+):(?:\d+:)?\s*(\w+Error|\w+)?\s*(.*)', error_str)
        if colon_match:
            structured["file"] = colon_match.group(1).split("/")[-1]
            structured["line"] = int(colon_match.group(2))
            if colon_match.group(3) and colon_match.group(3).strip():
                structured["error_type"] = colon_match.group(3).strip()
            if colon_match.group(4) and colon_match.group(4).strip():
                structured["message"] = colon_match.group(4).strip()
            return structured

        # Fallback to first non-empty line
        lines = [l.strip() for l in error_str.splitlines() if l.strip()]
        if lines:
            structured["message"] = lines[0][:150]
        return structured

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
        *,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        try:
            async for chunk in self.llm.chat_stream(
                messages,
                tools=tools,
                max_tokens=max_tokens,
            ):
                if isinstance(chunk, dict):
                    yield chunk
                    continue
                content_chunk, final_response = chunk
                clean_chunk = strip_dsml_text(
                    content_chunk,
                    known_tool_names=ASSEMBLED_TOOL_NAMES,
                )
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
    def _trace_manifest(
        manifest: Any,
        *,
        allowed_tools: frozenset[str],
        retrieval_no_gain_rounds: int = 0,
    ) -> None:
        """Print the projected manifest that governs this step's tool access."""
        metrics = manifest_metrics(manifest)
        snapshot = {
            "step": getattr(manifest, "updated_at_step", None),
            "step_id": getattr(manifest, "step_id", ""),
            "step_kind": getattr(manifest, "step_kind", ""),
            "sufficiency": getattr(manifest, "sufficiency", "INSUFFICIENT"),
            "required_coverage": round(metrics.coverage, 3),
            "missing_ratio": round(metrics.missing_ratio, 3),
            "stale_ratio": round(metrics.stale_ratio, 3),
            "retrieval_no_gain_rounds": retrieval_no_gain_rounds,
            "items": [
                {
                    "id": getattr(item, "id", ""),
                    "need": getattr(item, "need", ""),
                    "type": getattr(item, "type", ""),
                    "role": getattr(item, "role", "required"),
                    "status": getattr(item, "status", "MISSING"),
                    "file": getattr(item, "file", None),
                    "span": getattr(item, "span", None),
                    "symbol": getattr(item, "symbol", None),
                    "stale_reason": getattr(item, "stale_reason", None),
                }
                for item in getattr(manifest, "required_items", ())
            ],
            "allowed_tools": sorted(allowed_tools),
        }
        print(
            "[debug][manifest][projection-json]\n"
            + json.dumps(snapshot, ensure_ascii=False, indent=2, default=str),
            flush=True,
        )

    @staticmethod
    def _trace_tool_result(name: str, output: str, *, success: bool) -> None:
        if name in {"grep_search", "view_symbol_code"}:
            return
        state = "ok" if success else "error"
        print(
            f"[debug][action-layer][tool-result][{state}] {name}:\n{output}",
            flush=True,
        )
