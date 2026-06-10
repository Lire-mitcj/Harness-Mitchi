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

    if (
        kind == SubTaskKind.DIAGNOSE
        and structured is not None
        and structured.acceptance_met is False
    ):
        blocks.append(
            "Diagnose summary indicates acceptance_criteria was not met — "
            "revise the plan or search strategy before edit."
        )
    elif kind == SubTaskKind.DIAGNOSE and message and diagnose_acceptance_unmet(message):
        blocks.append(
            "Diagnose summary indicates acceptance_criteria was not met — "
            "revise the plan or search strategy before edit."
        )
    if kind == SubTaskKind.DIAGNOSE and message:
        contract_errors = _validate_handoff_contract(message, data.subtask.acceptance_criteria)
        blocks.extend(contract_errors)
        missing = diagnose_missing_required_outputs(
            data.subtask.acceptance_criteria,
            message,
            final_data=structured.raw if structured is not None else data.final_data,
        )
        if missing:
            blocks.append(
                "Diagnose summary is missing required handoff evidence: "
                + ", ".join(missing)
                + "."
            )

    if kind == SubTaskKind.DESIGN and message:
        design_errors = _validate_patch_intent_json(message, data.subtask.acceptance_criteria)
        blocks.extend(design_errors)

    if blocks:
        return GateResult.block("exit_gate", blocks, actions=["re_plan"])

    if warns:
        return GateResult.warn("exit_gate", warns, kind=kind.value)

    return GateResult.pass_("exit_gate", kind=kind.value, turns_used=data.turns_used)


def diagnose_acceptance_unmet(message: str) -> bool:
    lower = message.lower()
    return any(phrase in lower for phrase in _ACCEPTANCE_FAILURE_PHRASES)


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


def _validate_handoff_contract(message: str, criteria: str) -> list[str]:
    import json
    errors: list[str] = []
    marker = "HANDOFF_CONTRACT_JSON"
    
    is_requested = "HANDOFF_CONTRACT" in criteria or "handoff" in criteria.lower()
    is_present = marker in message
    
    if not is_present:
        if is_requested:
            errors.append("Diagnose step must output HANDOFF_CONTRACT_JSON block.")
        return errors

    payload = message.split(marker, 1)[1].strip()
    start = payload.find("{")
    if start < 0:
        errors.append("Diagnose step output has an invalid HANDOFF_CONTRACT_JSON structure.")
        return errors

    depth = 0
    end_idx = -1
    for idx, ch in enumerate(payload[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break

    if end_idx < start:
        errors.append("Diagnose step output has an invalid/unclosed HANDOFF_CONTRACT_JSON structure.")
        return errors

    try:
        contract = json.loads(payload[start : end_idx + 1])
    except Exception as e:
        errors.append(f"Failed to parse HANDOFF_CONTRACT_JSON: {e}")
        return errors

    if not isinstance(contract, dict):
        errors.append("HANDOFF_CONTRACT_JSON must be a JSON object.")
        return errors

    contract = _normalize_contract_keys(contract)

    required_contract_keys = ["must_modify", "available_views", "evidence"]
    missing_contract_keys = [k for k in required_contract_keys if k not in contract]
    if missing_contract_keys:
        errors.append(
            f"HANDOFF_CONTRACT_JSON is missing required key(s): {', '.join(missing_contract_keys)}."
        )
    else:
        must_modify = contract.get("must_modify")
        if not isinstance(must_modify, list):
            errors.append("HANDOFF_CONTRACT_JSON 'must_modify' must be a list.")
        else:
            for idx, item in enumerate(must_modify):
                if not isinstance(item, dict):
                    errors.append(f"HANDOFF_CONTRACT_JSON 'must_modify[{idx}]' must be a dictionary.")
                    continue
                req_target_keys = ["file", "line", "symbol_or_api", "should_change_to"]
                missing_tgt_keys = [k for k in req_target_keys if k not in item]
                if missing_tgt_keys:
                    errors.append(
                        f"HANDOFF_CONTRACT_JSON 'must_modify[{idx}]' is missing required key(s): {', '.join(missing_tgt_keys)}."
                    )
    return errors


def _validate_patch_intent_json(message: str, criteria: str) -> list[str]:
    import json
    errors: list[str] = []
    marker = "PATCH_INTENT_JSON"
    
    is_requested = "PATCH_INTENT" in criteria or "design" in criteria.lower() or "patch" in criteria.lower()
    is_present = marker in message
    
    if not is_present:
        if is_requested:
            errors.append("Design step must output PATCH_INTENT_JSON block.")
        return errors

    payload = message.split(marker, 1)[1].strip()
    start = payload.find("{")
    if start < 0:
        errors.append("Design step output has an invalid PATCH_INTENT_JSON structure.")
        return errors

    depth = 0
    end_idx = -1
    for idx, ch in enumerate(payload[start:], start=start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end_idx = idx
                break
    
    if end_idx < start:
        errors.append("Design step output has an invalid/unclosed PATCH_INTENT_JSON structure.")
        return errors

    block = payload[start : end_idx + 1]
    try:
        patch_intent = json.loads(block)
    except Exception as e:
        errors.append(f"Failed to parse PATCH_INTENT_JSON: {e}")
        return errors

    if not isinstance(patch_intent, dict):
        errors.append("PATCH_INTENT_JSON must be a JSON object.")
        return errors

    required_design_keys = [
        "edit_ready",
        "edit_strategy",
        "edit_targets",
        "dependencies",
        "acceptance_criteria",
        "target_view",
    ]
    missing_design_keys = [k for k in required_design_keys if k not in patch_intent]
    if missing_design_keys:
        errors.append(
            f"PATCH_INTENT_JSON is missing required key(s): {', '.join(missing_design_keys)}."
        )
    else:
        if patch_intent.get("edit_ready") is not True:
            errors.append("PATCH_INTENT_JSON 'edit_ready' must be true.")
        if not str(patch_intent.get("edit_strategy") or "").strip():
            errors.append("PATCH_INTENT_JSON 'edit_strategy' must be a non-empty string.")
        if (
            str(patch_intent.get("edit_strategy") or "").strip() == "sql_view_rewrite"
            and not str(patch_intent.get("target_view") or "").strip()
        ):
            errors.append("PATCH_INTENT_JSON 'target_view' must be a non-empty string.")
        edit_targets = patch_intent.get("edit_targets")
        if not isinstance(edit_targets, list):
            errors.append("PATCH_INTENT_JSON 'edit_targets' must be a list.")
        else:
            for idx, item in enumerate(edit_targets):
                if not isinstance(item, dict):
                    errors.append(f"PATCH_INTENT_JSON 'edit_targets[{idx}]' must be a dictionary.")
                    continue
                req_target_keys = ["file", "symbol", "line_start", "line_end", "snippet", "decision"]
                missing_tgt_keys = [k for k in req_target_keys if k not in item]
                if missing_tgt_keys:
                    errors.append(
                        f"PATCH_INTENT_JSON 'edit_targets[{idx}]' is missing required key(s): {', '.join(missing_tgt_keys)}."
                    )
        
        dependencies = patch_intent.get("dependencies")
        if not isinstance(dependencies, list):
            errors.append("PATCH_INTENT_JSON 'dependencies' must be a list.")
        
        acceptance_criteria = patch_intent.get("acceptance_criteria")
        if not isinstance(acceptance_criteria, list):
            errors.append("PATCH_INTENT_JSON 'acceptance_criteria' must be a list.")
    return errors


def _normalize_contract_keys(contract: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(contract, dict):
        return contract
    
    # Map must_modify
    if "edit_targets" in contract and "must_modify" not in contract:
        contract["must_modify"] = contract["edit_targets"]
    elif "targets" in contract and "must_modify" not in contract:
        contract["must_modify"] = contract["targets"]
        
    # Map available_views
    if "resolved_dependencies" in contract and "available_views" not in contract:
        deps = contract["resolved_dependencies"]
        if isinstance(deps, list):
            views = []
            for dep in deps:
                if isinstance(dep, dict) and dep.get("kind") == "database_view":
                    views.append({
                        "name": dep.get("name") or "",
                        "fields": dep.get("fields") or []
                    })
            contract["available_views"] = views
    elif "dependencies" in contract and "available_views" not in contract:
        deps = contract["dependencies"]
        if isinstance(deps, list):
            views = []
            for dep in deps:
                if isinstance(dep, dict) and dep.get("kind") == "database_view":
                    views.append({
                        "name": dep.get("name") or "",
                        "fields": dep.get("fields") or []
                    })
            contract["available_views"] = views
            
    # Map evidence
    if "facts" in contract and "evidence" not in contract:
        contract["evidence"] = contract["facts"]
        
    # Map back must_modify to edit_targets / targets if missing
    if "must_modify" in contract and ("edit_targets" not in contract or not contract["edit_targets"]):
        mm = contract["must_modify"]
        if isinstance(mm, list):
            targets = []
            for item in mm:
                if isinstance(item, dict):
                    line_val = item.get("line") or 0
                    line_start = item.get("line_start") or line_val
                    line_end = item.get("line_end") or line_val
                    targets.append({
                        "file": str(item.get("file") or "").strip() or "unknown",
                        "symbol": str(item.get("symbol_or_api") or item.get("symbol") or "unknown").strip(),
                        "line_start": int(line_start) if line_start is not None else 0,
                        "line_end": int(line_end) if line_end is not None else 0,
                        "snippet": str(item.get("snippet") or item.get("current_code") or "").strip(),
                        "decision": str(item.get("decision") or item.get("should_change_to") or "").strip(),
                    })
            contract["edit_targets"] = targets
            if "targets" not in contract or not contract["targets"]:
                contract["targets"] = targets

    # Map back available_views to dependencies / resolved_dependencies if missing
    if "available_views" in contract and ("dependencies" not in contract or not contract["dependencies"]):
        views = contract["available_views"]
        if isinstance(views, list):
            deps = []
            for v in views:
                if isinstance(v, dict):
                    deps.append({
                        "role": "replacement_source",
                        "kind": "database_view",
                        "name": v.get("name") or "",
                        "file": v.get("file") or "",
                        "line_start": v.get("line_start") or 0,
                        "line_end": v.get("line_end") or 0,
                        "columns": v.get("columns") or v.get("fields") or [],
                        "fields": v.get("fields") or v.get("columns") or [],
                        "replaces_objects": v.get("replaces_objects") or []
                    })
            contract["dependencies"] = deps
            if "resolved_dependencies" not in contract or not contract["resolved_dependencies"]:
                contract["resolved_dependencies"] = deps

    for key in ("dependencies", "dependencies_to_use", "resolved_dependencies"):
        deps = contract.get(key)
        if isinstance(deps, list):
            for dep in deps:
                if isinstance(dep, dict) and dep.get("kind") == "database_view":
                    if not dep.get("role"):
                        dep["role"] = "replacement_source"

    # Ensure all required keys exist
    if "must_modify" not in contract:
        contract["must_modify"] = []
    if "available_views" not in contract:
        contract["available_views"] = []
    if "evidence" not in contract:
        contract["evidence"] = []
        
    # Normalize must_modify items
    must_modify = contract.get("must_modify")
    if isinstance(must_modify, list):
        for item in must_modify:
            if isinstance(item, dict):
                if "symbol" in item and "symbol_or_api" not in item:
                    item["symbol_or_api"] = item["symbol"]
                if "line_start" in item and "line" not in item:
                    item["line"] = item["line_start"]
                elif "line" not in item:
                    item["line"] = 0
                if "decision" in item and "should_change_to" not in item:
                    item["should_change_to"] = item["decision"]
                elif "should_change_to" not in item:
                    item["should_change_to"] = "apply change"
                    
    return contract
