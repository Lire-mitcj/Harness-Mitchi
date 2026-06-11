from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.executor.final_output import parse_executor_final
from src.harness.gates.types import GateResult
from src.planner.task_tree import SubTaskKind, SubTaskNode

_MIN_SUMMARY_CHARS = 20
_REQUIRED_FINAL_JSON_KEYS = frozenset({
    "result",
    "acceptance_met",
    "evidence",
    "blocker",
})
_REQUIRED_AGENT_OUTPUT_KEYS = frozenset({
    "status",
    "changed_files",
    "validation",
    "risks",
    "handoff",
})

_ACCEPTANCE_FAILURE_PHRASES = (
    "not met",
    "not yet met",
    "criteria not met",
    "acceptance criteria not",
    "acceptance not",
    "could not find",
    "couldn't find",
    "unable to locate",
    "unable to find",
    "did not find",
    "have not been identified",
    "has not been identified",
    "cannot identify",
    "unidentified",
    "not located",
    "not identified",
    "no evidence",
    "none found",
    "no relevant",
    "target not found",
    "未定位到可交接",
    "证据不足",
    "不能作为后续",
    "llm_call failed",
    "midstreamfallbackerror",
    "apiconnectionerror",
    "llm request timed out",
    "litellm.",
)
_REQ_FILE_LINE = re.compile(
    r"(file\s*:?\s*line|path\s*:?\s*line|file:line|line\s+range|行号|行范围|路径.*行|文件.*行)",
    re.I,
)
_REQ_SYMBOL = re.compile(
    r"\b(symbol|function|class|method|view|query|sql)\b|符号|函数|方法|类|视图|查询",
    re.I,
)
_REQ_SNIPPET = re.compile(
    r"\b(snippet|excerpt|decision|code|sql)\b|片段|代码|决策|结论",
    re.I,
)
_HAS_FILE_LINE = re.compile(
    r"[\w./-]+\.(?:py|sql|md|tsx?|jsx?|ya?ml|json|toml):\d+(?:-\d+)?"
)
_HAS_SYMBOL = re.compile(
    r"\b(def|class|function|method|view|query|sql|symbol)\b|函数|方法|类|视图|查询",
    re.I,
)
_HAS_SNIPPET = re.compile(
    r"\b(snippet|excerpt|decision|code|sql|select|from|join|where)\b|结论|片段|代码|决策",
    re.I,
)


@dataclass
class ExitCheckInput:
    subtask: SubTaskNode
    final_message: str | None
    error_trace: list[str]
    changed_files: list[str]
    turns_used: int = 0
    tool_failure_count: int = 0
    final_data: dict[str, Any] | None = None


def validate_exit(data: ExitCheckInput) -> GateResult:
    """Rule-based Executor Exit Gate (E0) — no LLM."""
    blocks: list[str] = []
    warns: list[str] = []
    kind = data.subtask.kind
    message = (data.final_message or "").strip()
    structured = parse_executor_final(message)
    if structured is None and data.final_data:
        import json

        structured = parse_executor_final(json.dumps(data.final_data, ensure_ascii=False))

    if not message:
        blocks.append("Executor finished with an empty final answer.")

    if structured is not None and structured.raw is not None:
        raw_keys = set(structured.raw)
        required = (
            _REQUIRED_AGENT_OUTPUT_KEYS
            if ("status" in raw_keys or "handoff" in raw_keys)
            else _REQUIRED_FINAL_JSON_KEYS
        )
        missing_keys = sorted(required - raw_keys)
        if missing_keys:
            blocks.append(
                "Executor final JSON is missing required key(s): "
                + ", ".join(missing_keys)
                + "."
            )

    if len(message) < _MIN_SUMMARY_CHARS and kind in {
        SubTaskKind.DIAGNOSE,
        SubTaskKind.VERIFY,
        SubTaskKind.SHELL,
    }:
        blocks.append(
            f"Subtask kind={kind.value} requires a substantive summary "
            f"(>= {_MIN_SUMMARY_CHARS} chars)."
        )

    if data.tool_failure_count >= 2 and not _message_acknowledges_failure(message):
        warns.append(
            f"{data.tool_failure_count} tool failures occurred; final answer should "
            "state what failed and what was learned."
        )

    if kind == SubTaskKind.EDIT and not data.changed_files:
        blocks.append("Edit subtask completed without modifying any files.")

    if kind == SubTaskKind.EDIT and data.subtask.context_files and data.changed_files:
        whitelist = {f.replace("\\", "/").lstrip("./") for f in data.subtask.context_files}
        changed = {f.replace("\\", "/").lstrip("./") for f in data.changed_files}
        if whitelist and not changed.intersection(whitelist):
            warns.append(
                "Edited files are outside subtask context_files whitelist — "
                "verify scope is correct."
            )

    if data.error_trace and kind == SubTaskKind.EDIT and not data.changed_files:
        blocks.append("Unresolved tool errors and no successful file edits.")

    if kind == SubTaskKind.DIAGNOSE and _message_has_transport_failure(message):
        blocks.append(
            "Diagnose summary reports an LLM/tool transport failure; retry before "
            "using it as evidence."
        )

    if (
        kind == SubTaskKind.DIAGNOSE
        and structured is not None
        and structured.acceptance_met is False
    ):
        warns.append(
            "Diagnose summary indicates acceptance_criteria was not met; treating "
            "output as partial evidence for downstream verification."
        )
    elif kind == SubTaskKind.DIAGNOSE and message and diagnose_acceptance_unmet(message):
        warns.append(
            "Diagnose summary indicates acceptance_criteria was not met; treating "
            "output as partial evidence for downstream verification."
        )
    if kind == SubTaskKind.DIAGNOSE and message:
        missing = diagnose_missing_required_outputs(
            data.subtask.acceptance_criteria,
            message,
            final_data=structured.raw if structured is not None else data.final_data,
        )
        if missing:
            warns.append(
                "Diagnose summary is missing required handoff evidence: "
                + ", ".join(missing)
                + "."
            )

    if blocks:
        return GateResult.block("exit_gate", blocks, actions=["re_plan"])

    if warns:
        return GateResult.warn("exit_gate", warns, kind=kind.value)

    return GateResult.pass_("exit_gate", kind=kind.value, turns_used=data.turns_used)


def diagnose_acceptance_unmet(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in _ACCEPTANCE_FAILURE_PHRASES)


def _message_has_transport_failure(message: str) -> bool:
    lower = message.lower()
    return any(
        phrase in lower
        for phrase in (
            "llm_call failed",
            "midstreamfallbackerror",
            "apiconnectionerror",
            "llm request timed out",
            "litellm.",
        )
    )


def diagnose_missing_required_outputs(
    criteria: str,
    message: str,
    *,
    final_data: dict[str, Any] | None = None,
) -> list[str]:
    missing: list[str] = []
    has_file_line, has_symbol, has_snippet = _structured_evidence_flags(final_data)
    if _REQ_FILE_LINE.search(criteria) and not (has_file_line or _HAS_FILE_LINE.search(message)):
        missing.append("file:line")
    if _REQ_SYMBOL.search(criteria) and not (has_symbol or _HAS_SYMBOL.search(message)):
        missing.append("symbol")
    if _REQ_SNIPPET.search(criteria) and not (has_snippet or _HAS_SNIPPET.search(message)):
        missing.append("snippet/decision")
    return missing


def _structured_evidence_flags(final_data: dict[str, Any] | None) -> tuple[bool, bool, bool]:
    if not isinstance(final_data, dict):
        return False, False, False
    evidence = final_data.get("evidence")
    if not isinstance(evidence, list):
        handoff = final_data.get("handoff")
        if isinstance(handoff, dict):
            evidence = handoff.get("evidence")
    if not isinstance(evidence, list):
        return False, False, False
    has_file_line = False
    has_symbol = False
    has_snippet = False
    for item in evidence:
        if isinstance(item, str):
            has_snippet = has_snippet or bool(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("file")
        line = item.get("line") or item.get("line_range") or item.get("lines")
        location = item.get("location")
        if (path and line) or (isinstance(location, str) and _HAS_FILE_LINE.search(location)):
            has_file_line = True
        if item.get("symbol"):
            has_symbol = True
        if item.get("snippet") or item.get("decision") or item.get("reason"):
            has_snippet = True
    return has_file_line, has_symbol, has_snippet


def _message_acknowledges_failure(message: str) -> bool:
    lower = message.lower()
    hints = ("fail", "error", "block", "unable", "could not", "cannot", "issue")
    return any(h in lower for h in hints)
