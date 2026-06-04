from __future__ import annotations

import re
from dataclasses import dataclass

from src.harness.gates.types import GateResult
from src.planner.task_tree import SubTaskKind, SubTaskNode

_MIN_SUMMARY_CHARS = 20

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


def validate_exit(data: ExitCheckInput) -> GateResult:
    """Rule-based Executor Exit Gate (E0) — no LLM."""
    blocks: list[str] = []
    warns: list[str] = []
    kind = data.subtask.kind
    message = (data.final_message or "").strip()

    if not message:
        blocks.append("Executor finished with an empty final answer.")

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

    if kind == SubTaskKind.DIAGNOSE and message and diagnose_acceptance_unmet(message):
        blocks.append(
            "Diagnose summary indicates acceptance_criteria was not met — "
            "revise the plan or search strategy before edit."
        )
    if kind == SubTaskKind.DIAGNOSE and message:
        missing = diagnose_missing_required_outputs(
            data.subtask.acceptance_criteria,
            message,
        )
        if missing:
            blocks.append(
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


def diagnose_missing_required_outputs(criteria: str, message: str) -> list[str]:
    missing: list[str] = []
    if _REQ_FILE_LINE.search(criteria) and not _HAS_FILE_LINE.search(message):
        missing.append("file:line")
    if _REQ_SYMBOL.search(criteria) and not _HAS_SYMBOL.search(message):
        missing.append("symbol")
    if _REQ_SNIPPET.search(criteria) and not _HAS_SNIPPET.search(message):
        missing.append("snippet/decision")
    return missing


def _message_acknowledges_failure(message: str) -> bool:
    lower = message.lower()
    hints = ("fail", "error", "block", "unable", "could not", "cannot", "issue")
    return any(h in lower for h in hints)
