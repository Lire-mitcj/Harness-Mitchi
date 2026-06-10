from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any


_FUNCTION_REFACTOR_RE = re.compile(
    r"\b(refactor|replace\s+function|replace\s+\w+\s+with\s+\w+|reuse\s+existing\s+function|normalize|masking)\b"
    r"|重构|复用|替换函数|替换方法|统一逻辑|脱敏|多个接口|多个调用点",
    re.IGNORECASE,
)
_SQL_VIEW_REWRITE_RE = re.compile(
    r"(?=.*(?:视图|view))(?=.*(?:替换|使用|改成|改为|replace|use|switch))",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class HarnessTaskAnalysis:
    intent: str
    confidence: float
    complexity: str
    edit_ready: bool
    edit_strategy: str
    readiness_checks: dict[str, bool] = field(default_factory=dict)
    resolved_dependencies: tuple[dict[str, Any], ...] = ()
    editable_targets: tuple[dict[str, Any], ...] = ()
    acceptance_contract: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "mitkii.harness_task_analysis.v1",
            "intent": self.intent,
            "confidence": self.confidence,
            "complexity": self.complexity,
            "edit_ready": self.edit_ready,
            "edit_strategy": self.edit_strategy,
            "readiness_checks": dict(self.readiness_checks),
            "resolved_dependencies": list(self.resolved_dependencies),
            "editable_targets": list(self.editable_targets),
            "acceptance_contract": dict(self.acceptance_contract),
        }

    def to_planner_block(self) -> str:
        return (
            "HARNESS_TASK_ANALYSIS_JSON\n"
            + json.dumps(self.to_dict(), ensure_ascii=False, indent=2)
        )


def analyze_task(user_request: str, context_pack: Any | None = None) -> HarnessTaskAnalysis:
    intent, confidence = classify_intent(user_request)
    edit_strategy = intent if intent in {
        "function_refactor",
        "sql_view_rewrite",
        "api_contract_change",
        "bug_fix",
        "rename_symbol",
    } else "general_edit"

    file_count = _safe_len(getattr(context_pack, "relevant_files", ()) if context_pack else ())
    symbol_count = _safe_len(getattr(context_pack, "symbols", ()) if context_pack else ())
    metadata = getattr(context_pack, "metadata", {}) if context_pack else {}
    resolved_dependencies = _extract_resolved_dependencies(metadata)
    editable_targets = _extract_editable_targets(metadata)

    unresolved_dependencies = 0
    if intent == "sql_view_rewrite" and not resolved_dependencies:
        unresolved_dependencies = 1
    if intent == "function_refactor" and not resolved_dependencies:
        unresolved_dependencies = 1

    edit_scope_bounded = bool(file_count and file_count <= 2 and (symbol_count or editable_targets))
    intent_resolved = intent != "unknown"
    targets_resolved = bool(symbol_count or editable_targets)
    dependencies_resolved = unresolved_dependencies == 0
    acceptance_resolved = True

    complexity = assess_complexity(
        file_count=file_count,
        symbol_count=symbol_count,
        dependency_count=len(resolved_dependencies),
        unresolved_dependencies=unresolved_dependencies,
        intent_resolved=intent_resolved,
        dependencies_resolved=dependencies_resolved,
        edit_scope_bounded=edit_scope_bounded,
        acceptance_resolved=acceptance_resolved,
    )

    checks = {
        "intent_resolved": intent_resolved,
        "targets_resolved": targets_resolved,
        "dependencies_resolved": dependencies_resolved,
        "acceptance_resolved": acceptance_resolved,
        "edit_scope_bounded": edit_scope_bounded,
    }
    edit_ready = all(checks.values())

    return HarnessTaskAnalysis(
        intent=intent,
        confidence=confidence,
        complexity=complexity,
        edit_ready=edit_ready,
        edit_strategy=edit_strategy,
        readiness_checks=checks,
        resolved_dependencies=tuple(resolved_dependencies),
        editable_targets=tuple(editable_targets),
        acceptance_contract={
            "intent": intent,
            "edit_strategy": edit_strategy,
            "must_modify_target_symbols": True,
            "must_remove_old_logic": intent in {"function_refactor", "sql_view_rewrite"},
            "must_use_resolved_dependencies": bool(resolved_dependencies),
        },
    )


def classify_intent(text: str) -> tuple[str, float]:
    if _FUNCTION_REFACTOR_RE.search(text):
        return "function_refactor", 0.91
    if _SQL_VIEW_REWRITE_RE.search(text):
        return "sql_view_rewrite", 0.9
    lowered = text.lower()
    if any(token in lowered for token in ("rename", "重命名")):
        return "rename_symbol", 0.82
    if any(token in lowered for token in ("fix", "bug", "修复")):
        return "bug_fix", 0.78
    if any(token in lowered for token in ("api", "接口", "contract")):
        return "api_contract_change", 0.76
    return "unknown", 0.45


def assess_complexity(
    *,
    file_count: int,
    symbol_count: int,
    dependency_count: int,
    unresolved_dependencies: int,
    intent_resolved: bool,
    dependencies_resolved: bool,
    edit_scope_bounded: bool,
    acceptance_resolved: bool,
) -> str:
    if (
        not intent_resolved
        or not dependencies_resolved
        or not edit_scope_bounded
        or not acceptance_resolved
        or unresolved_dependencies > 0
    ):
        return "high"
    if (
        file_count <= 1
        and symbol_count <= 1
        and unresolved_dependencies == 0
        and acceptance_resolved
        and edit_scope_bounded
    ):
        return "low"
    return "medium"


def is_edit_ready(analysis: HarnessTaskAnalysis) -> bool:
    return analysis.edit_ready and all(analysis.readiness_checks.values())


def _safe_len(value: Any) -> int:
    try:
        return len(value)
    except TypeError:
        return 0


def _extract_resolved_dependencies(metadata: Any) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("resolved_dependencies_json") or metadata.get("resolved_dependencies")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [item for item in raw or [] if isinstance(item, dict)]


def _extract_editable_targets(metadata: Any) -> list[dict[str, Any]]:
    if not isinstance(metadata, dict):
        return []
    raw = metadata.get("editable_targets_json") or metadata.get("editable_targets")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    return [item for item in raw or [] if isinstance(item, dict)]
