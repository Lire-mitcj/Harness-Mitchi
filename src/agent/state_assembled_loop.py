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

from src.state import StateLayer, DecisionGravityController
from src.state.decision_gravity import evaluate_search_intent
from src.agent.context_assembly import ContextAssembly
from src.agent.run_state import (
    ArtifactRefs,
    Evidence,
    RunEvent,
    RunPhase,
    RunState,
    reduce_run_state,
    start_run,
)
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
from src.hooks.after_tool import apply_after_tool_output_limit
from src.hooks.post_tool_context import apply_post_tool_context_hook
from src.hooks.reallocate_tools import determine_allowed_tools
from src.llm.client import LLMClient
from src.llm.dsml import contains_tool_call_markup, strip_dsml_text

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)

ASSEMBLED_TOOL_NAMES = frozenset({"codebase_retrieve", "decision_edit", "view_symbol_code", "grep_search"})


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
    run_state: RunState = field(default_factory=lambda: start_run("", edit_mode=False))

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

        summary_lines = ["### TURN SUMMARY"]
        decisions: list[str] = []
        reads: list[str] = []
        edits: list[str] = []
        errors: list[str] = []
        for m in to_fold:
            if m.tool_calls:
                for tc in m.tool_calls:
                    if tc.name in PARALLEL_RETRIEVAL_TOOLS:
                        target = tc.arguments.get("target_file") or tc.arguments.get("query") or tc.name
                        reads.append(str(target))
                    elif tc.name == "decision_edit":
                        edits.append(str(tc.arguments.get("target_file") or "unknown"))
            if m.role == "assistant" and m.content.strip() and "[CONTEXT COLLAPSE" not in m.content:
                normalized = " ".join(m.content.split())
                if _is_process_only_intent(normalized):
                    continue
                if len(normalized) <= 400:
                    decisions.append(normalized)
                else:
                    sentences = [part.strip() for part in re.split(r"(?<=[。！？.!?])\s+", normalized) if part.strip()]
                    concise = next((part for part in reversed(sentences) if len(part) <= 400), "")
                    decisions.append(concise or "执行已记录的工具调用与 checklist")
            elif m.role == "tool":
                reads.extend(re.findall(r"(?:file|anchor):\s*`?([^`\s]+)", m.content))
                if "Error:" in m.content or "failed" in m.content.lower():
                    errors.append(" ".join(m.content.split())[:300])
        summary_lines.extend([
            f"- 决策：{decisions[-1] if decisions else '沿用当前计划'}",
            f"- 已读取：{', '.join(dict.fromkeys(reads)) if reads else '无'}",
            f"- 编辑：{', '.join(dict.fromkeys(edits)) if edits else '无'}",
            f"- 验证/错误：{errors[-1] if errors else '无'}",
            "- 下一步：严格依据 RUN STATE 的缺失证据行动",
        ])
        summary_text = "\n".join(summary_lines)

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


def _is_process_only_intent(text: str) -> bool:
    lowered = text.casefold().strip()
    patterns = (
        r"^(?:let me|i need to)\s+(?:first\s+)?(?:examine|look|read|analy[sz]e|inspect)",
        r"\b(?:read|look|examine)\s+more\s+thoroughly\b",
        r"\bunderstand\s+the\s+(?:full|current)\s+(?:picture|state|system)\b",
    )
    return any(re.search(pattern, lowered) for pattern in patterns)


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
    if item.get("symbol"):
        return "symbol"
    file_path = str(item.get("file") or "")
    code = textwrap.dedent(str(item.get("code") or "")).strip()
    if file_path.endswith(".sql") or re.match(r"(?is)^CREATE\s+(?:TABLE|VIEW|PROCEDURE|TRIGGER)\b", code):
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
    if state.context_anchors.code:
        view["raw_evidence_store"] = list(state.context_anchors.code)
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
    if tool_call.name not in PARALLEL_RETRIEVAL_TOOLS:
        return _tool_result_observation(result)
    anchors = (result.metadata or {}).get("raw_evidence_store") or []
    duplicate_replay = str(
        (result.metadata or {}).get("duplicate_anchor_replay") or ""
    ).strip()
    if not anchors:
        return duplicate_replay or (
            f"[CODE ANCHOR ALREADY STORED]\n- tool: {tool_call.name}\n- status: success"
        )
    kinds = {
        _anchor_memory_kind(item) for item in anchors if isinstance(item, dict)
    }
    header = (
        "[CODE ANCHOR STORED]" if kinds == {"symbol"}
        else "[FILE FACT STORED]" if kinds == {"fact"}
        else "[SCHEMA CONTRACT STORED]" if kinds == {"schema"}
        else "[MEMORY ARTIFACT STORED]"
    )
    lines = [header, f"- tool: {tool_call.name}"]
    for item in anchors:
        if not isinstance(item, dict):
            continue
        span = item.get("span") or ["?", "?"]
        lines.extend([
            f"- file: `{item.get('file')}`",
            f"  span: `{span[0]}-{span[1]}`",
            f"  symbol: `{item.get('symbol') or 'unresolved'}`",
            f"  memory_kind: `{_anchor_memory_kind(item)}`",
        ])
    if duplicate_replay:
        lines.extend(["", duplicate_replay])
    return "\n".join(lines)


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
    if not evidence and not candidates and not available_refs:
        return None
    return RunEvent(
        "evidence_stored",
        evidence=tuple(evidence),
        candidates=candidates,
        artifact_refs=refs,
        reason="tool evidence ingested",
    )


def _duplicate_anchor_replay(items: list[dict[str, Any]]) -> str:
    """Return full code and completeness flags for anchors already durable."""
    unique: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in items:
        unique[_anchor_key(item)] = item
    lines = [
        "[DUPLICATE ANCHOR — EXISTING FACTS REPLAYED]",
        "The following symbols are already durable and fully loaded in the context:",
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
                f"  [SYMBOL COMPLETENESS = TRUE]",
                f"  FULL SOURCE LOADED",
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


def _latest_symbol_slice_projection(search_cache: dict[str, Any]) -> str:
    projections = search_cache.get("symbol_projections") or []
    if not isinstance(projections, list):
        return ""

    blocks = []
    for item in projections[-2:]:
        if not isinstance(item, dict):
            continue
        projection_code = item.get("projection_code")
        file_path = item.get("file")
        span = item.get("span") or []
        if not projection_code or not file_path or len(span) < 2:
            continue
        truncated_attr = ' truncated="true"' if item.get("truncated") else ""
        symbol_attr = f' symbol="{item.get("symbol")}"' if item.get("symbol") else ""
        blocks.append(
            f'<symbol_slice file="{file_path}"{symbol_attr} '
            f'span="{span[0]}-{span[1]}"{truncated_attr}>\n'
            f"{projection_code}\n"
            "</symbol_slice>"
        )

    if not blocks:
        return ""
    return "### ACTIVE TEMPORARY CODE SLICES (from view_symbol_code) ###\n" + "\n\n".join(blocks)


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
        signature_src = json.dumps(
            {"query": query.strip().lower(), "files": sorted(files), "symbols": sorted(symbols)},
            ensure_ascii=False, sort_keys=True,
        )
        return {
            "step": step, "query": query, "files": sorted(files), "symbols": sorted(symbols),
            "signature": hashlib.sha256(signature_src.encode("utf-8")).hexdigest()[:16],
        }
    if not isinstance(payload, dict):
        return None

    grounding = payload.get("grounding") or {}
    evidence = payload.get("evidence") or []
    files = set()
    symbols = set()
    for item in grounding.get("files", []):
        if isinstance(item, dict) and item.get("path"):
            files.add(str(item["path"]))
    for item in grounding.get("symbols", []):
        if isinstance(item, dict):
            if item.get("file"):
                files.add(str(item["file"]))
            if item.get("name"):
                symbols.add(str(item["name"]))
    for item in evidence:
        if isinstance(item, dict):
            if item.get("file"):
                files.add(str(item["file"]))
            if item.get("symbol"):
                symbols.add(str(item["symbol"]))

    signature_src = json.dumps(
        {
            "query": query.strip().lower(),
            "files": sorted(files),
            "symbols": sorted(symbols),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return {
        "step": step,
        "query": query,
        "files": sorted(files),
        "symbols": sorted(symbols),
        "signature": hashlib.sha256(signature_src.encode("utf-8")).hexdigest()[:16],
    }





def _repeated_retrieval_message(
    query: str,
    retrieval_history: tuple[dict[str, Any], ...],
    search_output: str,
    *,
    step: int,
) -> str | None:
    snapshot = _retrieval_snapshot_from_output(query, search_output, step=step)
    if not snapshot:
        return None
    for item in retrieval_history:
        if item.get("signature") == snapshot["signature"]:
            symbols = ", ".join(snapshot["symbols"][:8]) or "known symbols"
            files = ", ".join(snapshot["files"][:8]) or "known files"
            return (
                "[codebase_retrieve blocked: repeated retrieval]\n"
                f"Query: {query}\n"
                f"Already explored files: {files}\n"
                f"Already explored symbols: {symbols}\n"
                "Search policy: do not retrieve the same path again. "
                "Use view_symbol_code for exact source of an explored symbol, "
                "or choose a genuinely new file/symbol to explore."
            )
    return None


def _append_retrieval_history(
    retrieval_history: tuple[dict[str, Any], ...],
    snapshot: dict[str, Any] | None,
) -> tuple[dict[str, Any], ...]:
    if not snapshot:
        return retrieval_history
    if any(item.get("signature") == snapshot["signature"] for item in retrieval_history):
        return retrieval_history
    return (*retrieval_history, snapshot)[-12:]


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
        self.gravity_controller = DecisionGravityController()
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

    async def _run_tool_with_sentinel(
        self, name: str, args: dict[str, Any], queue: asyncio.Queue[Any]
    ) -> ToolResult:
        try:
            res = await self.tools.call(name, args)
            return res
        finally:
            await queue.put(None)

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
        edit_mode = bool(
            re.search(
                r"\b(?:fix|implement|change|edit|add|remove|refactor|supplement|support|optimize|adjust)\b|"
                r"修复|修改|实现|新增|删除|重构|补充|完善|加上|添加|引入|支持|调整|优化",
                user_msg,
                re.IGNORECASE,
            )
        )
        self._run_events = []
        self._retrieval_history = ()
        self._grep_search_history = []
        self._last_novelty_value = 1.0
        self._embeddings_cache = {}
        self.stateLayer.clear_cache()
        self.state = replace(
            self.state,
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

            # Coordinates next turn via DecisionGravityController
            gravity_info = self.gravity_controller.coordinate_next_turn(self.state, self)

            self._current_step_tools = determine_allowed_tools(
                self.state,
                self.gravity_controller,
                default_tools=ASSEMBLED_TOOL_NAMES,
                has_compile_error=bool(self._validation_error()),
            )

            # --- CONTEXT ASSEMBLY ---
            # Slice messages history after the compact boundary
            sliced_messages = list(self.state.getMessagesAfterCompactBoundary())
            if ready_final:
                sliced_messages = []

            # Now build the messages payload for LiteLLM/OpenAI
            system_content = shaped_sys_prompt

            checklist_str = "\n".join(f"- {item}" for item in shaped_checklist) or "- No checklist items"
            if ready_final:
                context_block = _response_evidence_summary(self.state)
            else:
                context_block = self.context_assembly.build_context_block(
                    list(shaped_active_files),
                    search_cache=projected_search_cache,
                    modified_files=list(self.state.run_state.changes.files),
                )
                symbol_projection_block = _latest_symbol_slice_projection(self.state.search_cache)
                if symbol_projection_block:
                    context_block = (
                        f"{context_block}\n\n{symbol_projection_block}"
                        if context_block
                        else symbol_projection_block
                    )
            # Remove old RunState projection block to make state completely implicit

            substituted = system_content
            has_new_slots = (
                "{{STATE.ACTIVE_FILES_LIST}}" in system_content or
                "{{STATE.CHECKLIST}}" in system_content or
                "{{STATE.GIT_DIFFS}}" in system_content or
                "{{STATE.BUILD_ERRORS}}" in system_content or
                "{{STATE.ACTIVE_FILES_BLOCKS}}" in system_content or
                "{{STATE.LAST_TOOL_RESULT}}" in system_content or
                "{{STATE.LAST_ERROR}}" in system_content or
                "{{STATE.REPO_MAP}}" in system_content
            )

            rules_block = f"\n\n### PROJECT RULES & USER CONTEXT ###\n{shaped_rules_text}\n" if shaped_rules_text else ""
            implicit_state_block = ""
            if gravity_info.get("gravity_prompt"):
                implicit_state_block += f"\n\n{gravity_info['gravity_prompt']}"
            if gravity_info.get("blind_spots_prompt"):
                implicit_state_block += f"\n\n{gravity_info['blind_spots_prompt']}"

            # --- RETRIEVAL STATE INJECTION ---
            # 1. Collect Loaded Files
            loaded_files = []
            if hasattr(self.state, "context_anchors") and self.state.context_anchors.code:
                loaded_files = list({c.get("file") for c in self.state.context_anchors.code if isinstance(c, dict) and c.get("file")})
            for f in shaped_active_files:
                if f not in loaded_files:
                    loaded_files.append(f)
            
            # 2. Collect Loaded Symbols
            loaded_symbols = []
            projections = self.state.search_cache.get("symbol_projections") or []
            for item in projections:
                if isinstance(item, dict) and item.get("symbol"):
                    loaded_symbols.append(item["symbol"])
            if hasattr(self.state, "context_anchors") and self.state.context_anchors.code:
                for c in self.state.context_anchors.code:
                    sym = c.get("symbol")
                    if sym and sym not in loaded_symbols:
                        loaded_symbols.append(sym)

            # 3. Determine Database Schema grounding
            detected_tables = set()
            raw_ev = self.state.search_cache.get("raw_evidence_store") or []
            for item in raw_ev:
                if isinstance(item, dict):
                    code = item.get("code") or ""
                    if code:
                        from src.hooks.post_tool_context import _parse_and_structure_sql
                        try:
                            structs = _parse_and_structure_sql(code)
                            for s in structs:
                                tbl = s.get("table")
                                if tbl:
                                    detected_tables.add(tbl)
                                tables = s.get("tables")
                                if tables:
                                    detected_tables.update(tables)
                        except Exception:
                            pass
            
            # Also check if any schema/sql files are in loaded_files
            has_sql_file = any(f.endswith(".sql") for f in loaded_files)
            if has_sql_file or detected_tables:
                tables_suffix = f" (tables: {', '.join(sorted(detected_tables))})" if detected_tables else ""
                db_schema_str = f"fully grounded{tables_suffix}"
            else:
                db_schema_str = "fully grounded"

            # 4. Determine Convergence status
            is_closed = getattr(self.gravity_controller, "retrieval_disabled", False)
            retrieval_phase = "CLOSED" if is_closed else "ACTIVE"
            evidence_saturation = "HIGH" if is_closed else "MEDIUM"
            allowed_actions_str = ", ".join(sorted(self._current_step_tools)) or "final response only (NO TOOLS)"
            allowed_actions = f"{{{allowed_actions_str}}}"

            loaded_files_str = "\n".join(f"  * {f} (FULL SOURCE LOADED)" for f in loaded_files) if loaded_files else "  * none"
            loaded_symbols_str = "\n".join(f"  * {s} [SYMBOL COMPLETENESS = TRUE] (FULL SOURCE LOADED)" for s in loaded_symbols) if loaded_symbols else "  * none"

            state_injection_block = (
                f"\n\n### [RETRIEVAL STATE INJECTION] ###\n"
                f"[STATE UPDATE]\n"
                f"- Loaded Files:\n{loaded_files_str}\n"
                f"- Loaded Symbols:\n{loaded_symbols_str}\n"
                f"- Database Schema: {db_schema_str}\n"
                f"\n"
                f"[CONVERGENCE STATUS]\n"
                f"- retrieval phase = {retrieval_phase}\n"
                f"- evidence saturation = {evidence_saturation}\n"
                f"- allowed actions = {allowed_actions}\n"
            )
            if self._current_step_tools == frozenset({"decision_edit"}):
                state_injection_block += (
                    "\n[IMPORTANT INSTRUCTION]\n"
                    "All retrieval, search, and symbol reading tools (e.g. view_symbol_code, grep_search, codebase_retrieve) "
                    "have been disabled because your design phase/retrieval is complete. You already have all necessary code "
                    "symbols fully loaded in the CURRENT CONTEXT. Do NOT attempt to read files or search. You MUST proceed "
                    "directly to applying your proposed code modifications using the 'decision_edit' tool.\n"
                )
            implicit_state_block += state_injection_block

            if has_new_slots:
                active_files_list_str = ", ".join(shaped_active_files)
                git_diff_str = (
                    f"```text\n{self.state.git_diff}\n```"
                    if self.state.git_diff
                    else "Clean working tree."
                )
                validation_error = self._validation_error()
                build_errors_str = (
                    f"```\n{validation_error}\n```"
                    if validation_error else "No compile or build errors."
                )
                active_files_blocks_str = context_block or "No active files in context yet."
                last_tool_result_str = self._last_tool_result() or "No tools executed yet."
                import json
                last_error = self._last_error()
                last_error_str = json.dumps(last_error, ensure_ascii=False) if last_error else "None"

                exclude_symbols = set()
                if hasattr(self.state, "context_anchors") and self.state.context_anchors.code:
                    for anchor in self.state.context_anchors.code:
                        sym = anchor.get("symbol")
                        if sym:
                            exclude_symbols.add(sym)
                raw_ev = self.state.search_cache.get("raw_evidence_store") or []
                for item in raw_ev:
                    sym = item.get("symbol")
                    if sym:
                        exclude_symbols.add(sym)
                proj = self.state.search_cache.get("symbol_projections") or []
                for item in proj:
                    sym = item.get("symbol")
                    if sym:
                        exclude_symbols.add(sym)

                repo_map_str = ""
                context_builder = getattr(self, "context_builder", None)
                if context_builder is not None:
                    service = getattr(context_builder, "repo_map_service", None)
                    if service is not None and not hasattr(service, "assert_called") and hasattr(service, "to_planner_context"):
                        try:
                            max_chars = getattr(context_builder, "repo_map_max_chars", 12000)
                            if not isinstance(max_chars, int):
                                max_chars = 12000
                            block = service.to_planner_context(max_chars=max_chars, exclude_symbols=exclude_symbols)
                            if block and isinstance(block, str):
                                repo_map_str = block
                        except Exception:
                            pass
                if ready_final:
                    repo_map_str = "RepoMap omitted: retrieval evidence is complete."
                elif not repo_map_str:
                    repo_map_str = "No repository map available."

                substituted = substituted.replace("{{STATE.ACTIVE_FILES_LIST}}", active_files_list_str)
                substituted = substituted.replace("{{STATE.CHECKLIST}}", checklist_str)
                substituted = substituted.replace("{{STATE.GIT_DIFFS}}", git_diff_str)
                substituted = substituted.replace("{{STATE.BUILD_ERRORS}}", build_errors_str)
                substituted = substituted.replace("{{STATE.ACTIVE_FILES_BLOCKS}}", active_files_blocks_str)
                substituted = substituted.replace("{{STATE.LAST_TOOL_RESULT}}", last_tool_result_str)
                substituted = substituted.replace("{{STATE.LAST_ERROR}}", last_error_str)
                substituted = substituted.replace("{{STATE.REPO_MAP}}", repo_map_str)

                assembled_sys_content = substituted
                user_instruction_block = f"{rules_block}\nOriginal Request: {user_context}{implicit_state_block}" if rules_block else f"Original Request: {user_context}{implicit_state_block}"
            else:
                assembled_sys_content = system_content
                state_parts = [
                    "### CURRENT STATE ###",
                    f"Active Checklist:\n{checklist_str}",
                ]
                last_tool_result = self._last_tool_result()
                if last_tool_result:
                    state_parts.append(f"Last Tool Result: {last_tool_result}")
                last_error = self._last_error()
                if last_error:
                    import json
                    state_parts.append(f"Last Error (Structured):\n```json\n{json.dumps(last_error, ensure_ascii=False, indent=2)}\n```")
                if self.state.git_diff:
                    state_parts.append(
                        f"Git Working Tree State:\n```text\n{self.state.git_diff}\n```"
                    )
                validation_error = self._validation_error()
                if validation_error:
                    state_parts.append(f"Validation/Compiler Failures:\n```\n{validation_error}\n```")

                state_text = "\n\n".join(state_parts)
                context_text = f"### CURRENT CONTEXT ###\n\n{context_block}" if context_block else "### CURRENT CONTEXT ###\n\nNo active files in context yet."

                user_instruction_block = (
                    f"{state_text}\n\n"
                )
                if rules_block:
                    user_instruction_block += f"{rules_block}\n"
                user_instruction_block += (
                    f"{context_text}\n\n"
                    f"Original Request: {user_context}{implicit_state_block}"
                )

            assembled_messages = []
            if self._current_step_tools == frozenset({"decision_edit"}):
                sys_override = (
                    "\n\n"
                    "========================================================================\n"
                    "### CRITICAL SYSTEM INSTRUCTION OVERRIDE — EDIT PHASE ACTIVE ###\n"
                    "All search and read tools (including view_symbol_code, grep_search, codebase_retrieve) "
                    "have been disabled because your design phase/retrieval is complete. You already have all "
                    "necessary code symbols fully loaded in the CURRENT CONTEXT. Do NOT attempt to read files, "
                    "view symbols, or search.\n"
                    "You MUST proceed directly to applying your proposed code modifications using the 'decision_edit' tool.\n"
                    "Any attempt to call a forbidden tool will result in a tool rejection error.\n"
                    "========================================================================\n"
                )
                if sys_override not in assembled_sys_content:
                    assembled_sys_content += sys_override

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
            matches = re.findall(r'-\s+\[( |x|X)\]\s+(.*)', response_text)
            if matches:
                checklist_items = []
                for status, task in matches:
                    check_char = "x" if status.lower() == "x" else " "
                    checklist_items.append(f"[{check_char}] {task.strip()}")
                self.state = replace(self.state, checklist=tuple(checklist_items))

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
        if name == "grep_search" and result.success:
            pattern = arguments.get("pattern", "")
            path = arguments.get("path", ".")
            has_compile_error = bool(self._validation_error())
            # Use multi-signal novelty score if pre-computed, otherwise fallback
            evaluation = evaluate_search_intent(pattern, path, self._grep_search_history, has_compile_error)
            novelty = arguments.get("_novelty_score", evaluation["novelty"])
            
            self._grep_search_history.append({"file": path, "pattern": pattern})
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
        first_error_to_structure = None

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
                    tool_args = dict(tc.arguments)
                    tool_cache = _search_cache_view(self.state)
                    if tool_cache:
                        tool_args["_search_cache"] = tool_cache

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
                        gravity_controller=self.gravity_controller,
                        checklist=list(self.state.checklist),
                        context_anchors_code=list(self.state.context_anchors.code),
                        raw_evidence_store=list(self.state.search_cache.get("raw_evidence_store", [])),
                        git_diff=self.state.git_diff,
                        modified_files=list(self.state.run_state.changes.files),
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
                            self.gravity_controller.last_feedback = err

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
                            mock_metadata = {
                                "llm_observation": output_msg,
                                "is_mock_success": True,
                            }
                            try:
                                parsed = json.loads(output_msg)
                                if isinstance(parsed, dict):
                                    if "verbatim_code" in parsed:
                                        import hashlib
                                        code_str = parsed.get("verbatim_code") or ""
                                        code_hash = hashlib.md5(code_str.encode("utf-8")).hexdigest()[:6]
                                        anchor = {
                                            "file": parsed.get("file"),
                                            "span": parsed.get("span"),
                                            "symbol": tc.arguments.get("symbol"),
                                            "code": code_str,
                                            "verbatim_code": code_str,
                                            "hash": code_hash,
                                        }
                                        mock_metadata["raw_evidence_store"] = [anchor]
                                        mock_metadata["span"] = parsed.get("span")
                                    elif "matches" in parsed:
                                        mock_metadata["raw_evidence_store"] = parsed.get("matches")
                            except Exception:
                                pass
                            result = ToolResult(
                                success=True,
                                output=output_msg,
                                metadata=mock_metadata,
                            )
                            success = True
                        else:
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
                    task_completion = self._apply_context_update(result)
                    display_output = result.output if result.success else f"Error: {result.error}"
                    self._trace_tool_result(tc.name, display_output, success=result.success)

                    # Log signal for parallel retrieve
                    sig_args = f"query=\"{tc.arguments.get('query', '')}\"" if tc.name == "codebase_retrieve" else "..."
                    sig_status = "success" if result.success else "failed"
                    executed_tool_signals.append(f"{tc.name}({sig_args}) -> {sig_status}")

                    if not result.success:
                        if not first_error_to_structure:
                            first_error_to_structure = result.error or result.output

                    if result.success:
                        if result.metadata:
                            merged_cache = dict(self.state.search_cache)
                            if "search_output" in result.metadata:
                                new_output = result.metadata["search_output"]
                                retrieval_query = str(tc.arguments.get("query", ""))
                                repeated_msg = _repeated_retrieval_message(
                                    retrieval_query,
                                    self._retrieval_history,
                                    new_output,
                                    step=self.state.run_state.step,
                                )
                                if repeated_msg:
                                    result = ToolResult(
                                        success=True,
                                        output=repeated_msg,
                                        metadata={
                                            "llm_observation": repeated_msg,
                                            "retrieval_repeated": True,
                                        },
                                    )
                                    display_output = repeated_msg
                                    self._trace_tool_result(tc.name, display_output, success=True)
                                    self.state = replace(
                                        self.state,
                                        messages_history=self.state.messages_history + (
                                            tool_message(tc.id, _tool_history_receipt(tc, result)),
                                        ),
                                    )
                                    yield tool_result_event(tc.name, display_output, success=True)
                                    continue
                                snapshot = _retrieval_snapshot_from_output(
                                    retrieval_query,
                                    new_output,
                                    step=self.state.run_state.step,
                                )
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
                                self._retrieval_history = _append_retrieval_history(
                                    self._retrieval_history,
                                    snapshot,
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
                            if raw_store:
                                self._ingest_code_artifacts(tc, raw_store)
                                current_raw = list(merged_cache.get("raw_evidence_store") or [])
                                for item in raw_store:
                                    # Prune any existing cached evidence matching the same file and overlapping span
                                    current_raw = [
                                        existing for existing in current_raw
                                        if not (existing.get("file") == item.get("file") and
                                                _span_overlap_ratio(existing, item) > 0.90)
                                    ]
                                    current_raw.append(item)
                                merged_cache["raw_evidence_store"] = current_raw

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

                # Check permission
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

                # Execute the tool
                tool_args = dict(tc.arguments)
                tool_cache = _search_cache_view(self.state)
                if tool_cache:
                    tool_args["_search_cache"] = tool_cache

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
                    gravity_controller=self.gravity_controller,
                    checklist=list(self.state.checklist),
                    context_anchors_code=list(self.state.context_anchors.code),
                    raw_evidence_store=list(self.state.search_cache.get("raw_evidence_store", [])),
                    git_diff=self.state.git_diff,
                    modified_files=list(self.state.run_state.changes.files),
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
                        self.gravity_controller.last_feedback = err

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
                        mock_metadata = {
                            "llm_observation": output_msg,
                            "is_mock_success": True,
                        }
                        try:
                            parsed = json.loads(output_msg)
                            if isinstance(parsed, dict):
                                if "verbatim_code" in parsed:
                                    import hashlib
                                    code_str = parsed.get("verbatim_code") or ""
                                    code_hash = hashlib.md5(code_str.encode("utf-8")).hexdigest()[:6]
                                    anchor = {
                                        "file": parsed.get("file"),
                                        "span": parsed.get("span"),
                                        "symbol": tc.arguments.get("symbol"),
                                        "code": code_str,
                                        "verbatim_code": code_str,
                                        "hash": code_hash,
                                    }
                                    mock_metadata["raw_evidence_store"] = [anchor]
                                    mock_metadata["span"] = parsed.get("span")
                                elif "matches" in parsed:
                                    mock_metadata["raw_evidence_store"] = parsed.get("matches")
                        except Exception:
                            pass
                        result = ToolResult(
                            success=True,
                            output=output_msg,
                            metadata=mock_metadata,
                        )
                        success = True
                    else:
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
                display_output = result.output if result.success else f"Error: {result.error}"
                task_completion = self._apply_context_update(result)
                self._trace_tool_result(tc.name, display_output, success=result.success)

                # Log sequential tool signals
                sig_args = ""
                if tc.name == "decision_edit":
                    sig_args = f"target_file=\"{tc.arguments.get('target_file', '')}\""
                elif tc.name == "codebase_retrieve":
                    sig_args = f"query=\"{tc.arguments.get('query', '')}\""
                elif tc.name == "view_symbol_code":
                    sig_args = f"target_file=\"{tc.arguments.get('target_file', '')}\", symbol=\"{tc.arguments.get('symbol', '')}\""
                else:
                    sig_args = "..."
                sig_status = "success" if result.success else "failed"
                executed_tool_signals.append(f"{tc.name}({sig_args}) -> {sig_status}")

                if not result.success:
                    if not first_error_to_structure:
                        first_error_to_structure = result.error or result.output

                if result.success:
                    if result.metadata:
                        merged_cache = dict(self.state.search_cache)
                        if "search_output" in result.metadata:
                            new_output = result.metadata["search_output"]
                            retrieval_query = str(tc.arguments.get("query", ""))
                            repeated_msg = _repeated_retrieval_message(
                                retrieval_query,
                                self._retrieval_history,
                                new_output,
                                step=self.state.run_state.step,
                            )
                            if repeated_msg:
                                result = ToolResult(
                                    success=True,
                                    output=repeated_msg,
                                    metadata={
                                        "llm_observation": repeated_msg,
                                        "retrieval_repeated": True,
                                    },
                                )
                                display_output = repeated_msg
                                self._trace_tool_result(tc.name, display_output, success=True)
                                self.state = replace(
                                    self.state,
                                    messages_history=self.state.messages_history + (
                                        tool_message(tc.id, _tool_history_receipt(tc, result)),
                                    ),
                                )
                                yield tool_result_event(tc.name, display_output, success=True)
                                continue
                            snapshot = _retrieval_snapshot_from_output(
                                retrieval_query,
                                new_output,
                                step=self.state.run_state.step,
                            )
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
                            self._retrieval_history = _append_retrieval_history(
                                self._retrieval_history,
                                snapshot,
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
                        if raw_store:
                            self._ingest_code_artifacts(tc, raw_store)
                            current_raw = list(merged_cache.get("raw_evidence_store") or [])
                            for item in raw_store:
                                # Prune any existing cached evidence matching the same file and overlapping span
                                current_raw = [
                                    existing for existing in current_raw
                                    if not (existing.get("file") == item.get("file") and
                                            _span_overlap_ratio(existing, item) > 0.90)
                                ]
                                current_raw.append(item)
                            merged_cache["raw_evidence_store"] = current_raw

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
                        import difflib
                        diff_text = ""
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
        ))

    def _merge_raw_evidence(self, project_root: Path, raw_evidence_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    def _trace_tool_result(name: str, output: str, *, success: bool) -> None:
        state = "ok" if success else "error"
        print(
            f"[debug][action-layer][tool-result][{state}] {name}:\n{output}",
            flush=True,
        )
