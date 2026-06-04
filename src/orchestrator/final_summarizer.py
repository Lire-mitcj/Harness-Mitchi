from __future__ import annotations

import json
import re
from typing import Any, Protocol

from src.agent.types import LLMResponse
from src.planner.task_tree import SubTaskStatus, TaskTree


class FinalSummaryLLM(Protocol):
    async def chat_stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ): ...


class FinalSummarizer:
    """Final user-facing summary after all plan subtasks finish."""

    def __init__(
        self,
        llm: FinalSummaryLLM,
        *,
        max_tokens: int = 1024,
        timeout: float = 45.0,
    ) -> None:
        self.llm = llm
        self.max_tokens = max_tokens
        self.timeout = timeout

    async def summarize(
        self,
        *,
        user_request: str,
        task_tree: TaskTree,
        subtask_summaries: dict[str, str],
    ) -> str:
        payload = _summary_payload(user_request, task_tree, subtask_summaries)
        messages = [
            {
                "role": "system",
                "content": (
                    "你是最终结果总结器。你只基于给定 JSON 总结本次执行结果，"
                    "不调用工具、不编造未提供的事实、不输出内部日志。"
                    "输出必须短、清楚、稳定，不要复制大段路径或代码，不要续写无意义符号。"
                ),
            },
            {
                "role": "user",
                "content": (
                    "请用简洁中文面向用户总结本次任务结果。\n"
                    "格式要求：\n"
                    "1. 第一行说明是否完成。\n"
                    "2. 查询/诊断任务列出最多 5 个关键发现，格式为 `路径:行号 - 说明`。\n"
                    "3. 修改任务说明 changed files 和 validation。\n"
                    "4. 如果存在风险或未完成，最后一行说明。\n"
                    "5. 不要输出工具日志、不要输出 JSON、不要输出连续引号或重复字符。\n\n"
                    "RUN_RESULT_JSON\n"
                    + json.dumps(payload, ensure_ascii=False, indent=2)
                ),
            },
        ]
        content_parts: list[str] = []
        final_response: LLMResponse | None = None
        async for chunk, response in self.llm.chat_stream(
            messages,
            tools=[],
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        ):
            if chunk:
                content_parts.append(str(chunk))
            if response is not None:
                final_response = response
        text = "".join(content_parts).strip()
        if not text and final_response is not None:
            text = (final_response.content or "").strip()
        return _clean_summary(text)


def _summary_payload(
    user_request: str,
    task_tree: TaskTree,
    subtask_summaries: dict[str, str],
) -> dict[str, Any]:
    return {
        "user_request": user_request,
        "steps": [
            {
                "id": node.id,
                "kind": node.kind.value,
                "description": node.description,
                "status": node.status.value,
                "summary": _compact_step_summary(subtask_summaries.get(node.id, "")),
            }
            for node in task_tree.nodes
        ],
    }


def build_deterministic_user_summary(
    *,
    user_request: str,
    task_tree: TaskTree,
    subtask_summaries: dict[str, str],
) -> str:
    """Stable non-LLM fallback for terminal-facing final summaries."""
    completed = all(node.status == SubTaskStatus.SUCCESS for node in task_tree.nodes)
    title = "已完成。" if completed else "未完全完成。"
    lines = [title]
    findings = _extract_evidence_findings(subtask_summaries)
    changed_files = _extract_changed_files(subtask_summaries)

    if findings:
        lines.extend(["", "关键发现："])
        for item in findings[:8]:
            lines.append(f"- {item}")
    elif changed_files:
        lines.extend(["", "修改文件："])
        for path in changed_files[:8]:
            lines.append(f"- {path}")
    else:
        lines.extend(["", f"任务：{user_request}"])

    if completed:
        if changed_files:
            lines.extend(["", "验证/风险：已完成计划步骤，请以项目测试结果为准。"])
        else:
            lines.extend(["", "验证/风险：本次为只读诊断，没有修改文件。"])
    else:
        failed = [
            f"{node.id}: {node.description}"
            for node in task_tree.nodes
            if node.status != SubTaskStatus.SUCCESS
        ]
        lines.extend(["", "未完成步骤："])
        lines.extend(f"- {item}" for item in failed[:5])
    return "\n".join(lines)


def _compact_step_summary(text: str, *, max_lines: int = 16, max_chars: int = 2400) -> str:
    kept: list[str] = []
    skip_digest = False
    for raw in (text or "").strip().splitlines():
        line = raw.rstrip()
        if line.startswith("Executor evidence digest:"):
            skip_digest = True
            continue
        if skip_digest:
            continue
        if not line.strip():
            if kept and kept[-1] != "":
                kept.append("")
            continue
        kept.append(line)
        if len([item for item in kept if item.strip()]) >= max_lines:
            break
    cleaned = "\n".join(kept).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 24] + "\n...[summary truncated]"


def _clean_summary(text: str) -> str:
    cleaned = text.strip()
    if not cleaned:
        return ""
    if len(cleaned) > 2000:
        return ""
    if _looks_corrupted(cleaned):
        return ""
    return cleaned


def _looks_corrupted(text: str) -> bool:
    if "�" in text:
        return True
    quote_noise = sum(text.count(ch) for ch in ('"', "'", "“", "”", "「", "」"))
    if quote_noise > 40:
        return True
    for ch in ('"', "'", "”", "0", "\\"):
        if ch * 12 in text:
            return True
    return False


def _extract_evidence_findings(subtask_summaries: dict[str, str]) -> list[str]:
    findings: list[str] = []
    for summary in subtask_summaries.values():
        in_evidence = False
        for raw in summary.splitlines():
            line = raw.strip()
            if line == "Evidence:":
                in_evidence = True
                continue
            if in_evidence and line.startswith("Conclusion:"):
                in_evidence = False
                continue
            if not in_evidence or not line.startswith("- "):
                continue
            item = _format_evidence_item(line[2:].strip())
            if item and item not in findings:
                findings.append(item)
    return findings


def _format_evidence_item(item: str) -> str:
    parts = [part.strip() for part in item.split(" | ", 2)]
    if len(parts) == 3:
        location, symbol, snippet = parts
        return f"{location} - {symbol}: {snippet}"
    return item


def _extract_changed_files(subtask_summaries: dict[str, str]) -> list[str]:
    changed: list[str] = []
    pattern = re.compile(r"Changed files:\s*(?P<files>.+)")
    for summary in subtask_summaries.values():
        for match in pattern.finditer(summary):
            for raw in match.group("files").split(","):
                path = raw.strip()
                if path and path != "(none)" and path not in changed:
                    changed.append(path)
    return changed
