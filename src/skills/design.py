from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

from src.skills.base import SkillContext, SkillResult


class DesignSkill:
    name = "design"

    async def run(self, context: SkillContext, **kwargs: object) -> SkillResult:
        task_analysis = kwargs.get("task_analysis")
        handoff_contract = kwargs.get("handoff_contract")
        search_output = str(kwargs.get("search_output") or "")

        analysis = task_analysis if isinstance(task_analysis, dict) else {}
        contract = handoff_contract if isinstance(handoff_contract, dict) else {}
        strategy = str(analysis.get("edit_strategy") or analysis.get("intent") or "general_edit")
        targets = _targets_from_contract(contract)
        if not targets:
            for item in analysis.get("editable_targets") or []:
                if isinstance(item, dict) and item.get("file"):
                    targets.append({
                        "file": str(item.get("file") or "").strip() or "unknown",
                        "symbol": str(item.get("symbol") or item.get("symbol_or_api") or "unknown").strip(),
                        "line_start": int(item.get("start_line") or item.get("line_start") or 0),
                        "line_end": int(item.get("end_line") or item.get("line_end") or 0),
                        "snippet": str(item.get("current_code") or item.get("snippet") or "").strip(),
                        "decision": str(item.get("decision") or ""),
                    })
        dependencies = (
            analysis.get("resolved_dependencies")
            or analysis.get("dependencies")
            or analysis.get("dependencies_to_use")
            or contract.get("dependencies")
            or contract.get("dependencies_to_use")
            or contract.get("resolved_dependencies")
            or []
        )
        dep_list = list(dependencies) if isinstance(dependencies, (list, tuple)) else []
        old_logic = _old_logic_for_strategy(strategy)

        # Determine view name if sql_view_rewrite
        view_name = ""
        for dep in dep_list:
            if isinstance(dep, dict) and dep.get("kind") == "database_view" and dep.get("name"):
                view_name = str(dep.get("name") or "")
                break
        if not view_name:
            view_name = str(contract.get("target_view") or analysis.get("target_view") or "")

        # Safely determine project root
        project_root = Path.cwd()
        if context and getattr(context, "project_root", None):
            project_root = Path(context.project_root)
        elif kwargs.get("project_root"):
            project_root = Path(kwargs.get("project_root"))

        # Refine targets using local workspace file reading
        refined_targets = []
        for tgt in targets:
            refined_targets.append(_refine_target(tgt, project_root, view_name))
        targets = refined_targets

        # Build acceptance_criteria list
        acceptance_criteria = []
        if strategy == "sql_view_rewrite":
            tgt_sym = targets[0]["symbol"] if (targets and targets[0].get("symbol")) else "SQL query"
            view_str = f" uses {view_name}" if view_name else " uses view"
            acceptance_criteria = [
                f"{tgt_sym}{view_str}",
                "old SQL tables removed",
                "py_compile passes",
                "tests pass"
            ]
        elif strategy == "function_refactor":
            tgt_sym = targets[0]["symbol"] if (targets and targets[0].get("symbol")) else "helper"
            acceptance_criteria = [
                f"refactored {tgt_sym} to reuse helper function",
                "duplicated logic removed",
                "tests pass"
            ]
        else:
            acceptance_criteria = [
                "Apply requested changes successfully",
                "tests pass"
            ]

        # Build must_modify list
        must_modify = []
        for tgt in targets:
            must_modify.append({
                "file": tgt.get("file") or "",
                "symbol_or_api": tgt.get("symbol") or "",
                "line_start": tgt.get("line_start") or 0,
                "line_end": tgt.get("line_end") or 0,
                "current_code": tgt.get("snippet") or "",
                "should_change_to": tgt.get("decision") or (f"use {view_name}" if view_name else "apply the requested code change"),
            })

        # Build available_views list
        available_views = []
        for dep in dep_list:
            if dep.get("kind") == "database_view":
                available_views.append({
                    "name": dep.get("name") or "",
                    "file": dep.get("file") or "",
                    "line_start": dep.get("line_start") or 0,
                    "line_end": dep.get("line_end") or 0,
                    "columns": dep.get("fields") or []
                })

        # Build evidence list
        evidence = []
        for tgt in targets:
            l_start = tgt.get("line_start") or 0
            l_end = tgt.get("line_end") or 0
            sym = tgt.get("symbol") or ""
            file_name = tgt.get("file") or ""
            if l_start != l_end:
                evidence.append(f"{file_name}:{l_start}-{l_end} {sym}".strip())
            else:
                evidence.append(f"{file_name}:{l_start} {sym}".strip())
        for view in available_views:
            evidence.append(f"{view.get('file')}:{view.get('line_start')} CREATE VIEW {view.get('name')}".strip())
        # Determine target_view
        target_view = contract.get("target_view") or view_name or ""
        if strategy == "sql_view_rewrite" and target_view:
            for tgt in targets:
                if not str(tgt.get("decision") or "").strip():
                    tgt["decision"] = f"use view {target_view}"

        patch_intent = {
            "schema": "mitkii.handoff.v1",
            "edit_strategy": strategy,
            "edit_ready": True,
            "edit_targets": targets,
            "must_modify": must_modify,
            "available_views": available_views,
            "evidence": evidence,
            "target_view": target_view,
            "patch_intent": _patch_intent_summary(strategy, context.user_request),
            "dependencies": dep_list,
            "dependencies_to_use": dep_list,
            "resolved_dependencies": dep_list,
            "old_logic_to_remove": old_logic,
            "acceptance_criteria": acceptance_criteria,
            "acceptance_contract": analysis.get("acceptance_contract") or {},
        }
        text = "PATCH_INTENT_JSON\n" + json.dumps(patch_intent, ensure_ascii=False, indent=2)
        return SkillResult(
            success=True,
            summary=f"Design produced PATCH_INTENT_JSON for strategy={strategy}.",
            metadata={
                "patch_intent_json": json.dumps(patch_intent, ensure_ascii=False),
                "final_message": text,
                "search_output_preview": search_output[:1000],
            },
        )


def _targets_from_contract(contract: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for item in contract.get("must_modify") or []:
        if not isinstance(item, dict):
            continue
        line_val = item.get("line") or 0
        line_start = item.get("line_start") or line_val
        line_end = item.get("line_end") or line_val
        targets.append({
            "file": str(item.get("file") or "").strip() or "unknown",
            "symbol": str(item.get("symbol_or_api") or item.get("symbol") or "unknown").strip(),
            "line_start": int(line_start) if line_start is not None else 0,
            "line_end": int(line_end) if line_end is not None else 0,
            "snippet": str(item.get("snippet") or item.get("current_sql") or "").strip(),
            "decision": str(item.get("decision") or item.get("should_change_to") or "").strip(),
        })
    return targets


def _old_logic_for_strategy(strategy: str) -> list[str]:
    if strategy == "function_refactor":
        return ["manual duplicate logic", "manual masking", "duplicated call-site transformations"]
    if strategy == "sql_view_rewrite":
        return ["replaced table joins", "legacy SQL source references"]
    return []


def _patch_intent_summary(strategy: str, request: str) -> str:
    if strategy == "function_refactor":
        return "Refactor implementation to reuse the resolved helper/function and remove duplicated logic."
    if strategy == "sql_view_rewrite":
        return "Rewrite the approved SQL target to use the resolved database view dependency."
    return request.strip()[:300] or "Apply requested code change."


def _refine_target(target: dict[str, Any], project_root: Path, view_name: str) -> dict[str, Any]:
    file_path = target.get("file")
    if not file_path or file_path == "unknown":
        return target
    
    # Try resolving path
    resolved_path = project_root / file_path
    if not resolved_path.is_file():
        resolved_path = Path(file_path)
        if not resolved_path.is_file():
            return target
            
    try:
        lines = resolved_path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return target
        
    line_start = target.get("line_start", 0)
    line_end = target.get("line_end", 0)
    symbol = target.get("symbol", "unknown")
    snippet = target.get("snippet", "")
    decision = target.get("decision", "")
    
    # 1. If line_start/line_end are 0 but symbol is known and not placeholder
    if line_start == 0 and symbol not in ("unknown", "目标代码", ""):
        # Search for symbol definition
        # e.g., def symbol, async def symbol, class symbol, or symbol = ...
        found_line = -1
        for idx, line in enumerate(lines):
            if re.search(rf"\b(?:def|class)\s+{re.escape(symbol)}\b", line):
                found_line = idx + 1
                break
        if found_line == -1:
            for idx, line in enumerate(lines):
                if re.search(rf"\b{re.escape(symbol)}\b", line):
                    found_line = idx + 1
                    break
        if found_line != -1:
            line_start = found_line
            # Find enclosing range for python files
            if file_path.endswith(".py"):
                try:
                    base_indent = len(lines[line_start - 1]) - len(lines[line_start - 1].lstrip(" "))
                    end_idx = line_start - 1
                    for idx in range(line_start, len(lines)):
                        candidate = lines[idx]
                        if not candidate.strip():
                            end_idx = idx
                            continue
                        indent = len(candidate) - len(candidate.lstrip(" "))
                        if indent < base_indent:
                            break
                        if indent == base_indent:
                            stripped = candidate.strip()
                            if stripped.startswith(("'''", '"""', ")", "]", "}")):
                                end_idx = idx
                            break
                        end_idx = idx
                    line_end = end_idx + 1
                except Exception:
                    line_end = line_start
            else:
                line_end = line_start

    # 2. If line_start > 0 but symbol is placeholder/missing
    if line_start > 0 and symbol in ("unknown", "目标代码", ""):
        if file_path.endswith(".py"):
            try:
                # Find definition upward
                def_idx = -1
                for idx in range(line_start - 1, -1, -1):
                    line = lines[idx]
                    if re.match(r"^\s*(?:async\s+def|def|class)\s+\w+", line):
                        def_idx = idx
                        break
                    # Also support variable/assignment definition
                    if re.match(r"^\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*=", line):
                        def_idx = idx
                        break
                if def_idx != -1:
                    def_line = lines[def_idx]
                    match = re.search(r"^\s*(?:async\s+def|def|class)?\s*([a-zA-Z_][a-zA-Z0-9_]*)", def_line)
                    if match:
                        symbol = match.group(1)
                        if line_end == 0 or line_end == line_start:
                            line_start = def_idx + 1
                            # Find enclosing end
                            base_indent = len(def_line) - len(def_line.lstrip(" "))
                            end_idx = def_idx
                            for idx in range(def_idx + 1, len(lines)):
                                candidate = lines[idx]
                                if not candidate.strip():
                                    end_idx = idx
                                    continue
                                indent = len(candidate) - len(candidate.lstrip(" "))
                                if indent < base_indent:
                                    break
                                if indent == base_indent:
                                    stripped = candidate.strip()
                                    if stripped.startswith(("'''", '"""', ")", "]", "}")):
                                        end_idx = idx
                                    break
                                end_idx = idx
                            line_end = end_idx + 1
            except Exception:
                pass

    # 3. Read snippet from file if line_start is positive and snippet is empty
    if line_start > 0 and not snippet:
        end_idx = min(len(lines), line_end if line_end > 0 else line_start)
        snippet = "\n".join(lines[line_start - 1 : end_idx])
        
    # 4. Generate decision if empty
    if not decision:
        if view_name:
            decision = f"replace old SQL with {view_name}"
        else:
            decision = "replace SQL with view"
            
    target["line_start"] = line_start
    target["line_end"] = line_end if line_end > 0 else line_start
    target["symbol"] = symbol
    target["snippet"] = snippet
    target["decision"] = decision
    return target
