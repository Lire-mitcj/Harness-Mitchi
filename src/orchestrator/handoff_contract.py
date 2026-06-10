from __future__ import annotations

import json
import re
from typing import Any

_LOCATION_RE = re.compile(
    r"(?P<path>[\w./-]+\.(?:py|sql|tsx?|jsx?|json|ya?ml|toml)):"
    r"(?P<line>\d+(?:-\d+)?)\s*(?:\|\s*(?P<symbol>[^|]+)\|\s*(?P<snippet>.*))?"
)
_VIEW_RE = re.compile(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:FORCE\s+)?(?:VIEW|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'\`\[]?(?P<name>[\w.]+)", re.I)
_FROM_RE = re.compile(r"\bFROM\s+(?P<table>[\w.]+)", re.I)
_REPLACEMENT_SOURCE_RE = re.compile(
    r'["\']?replacement_source["\']?\s*:\s*\{[^{}]*["\']?kind["\']?\s*:\s*["\']?database_view["\']?[^{}]*["\']?name["\']?\s*:\s*["\']?(?P<name>[\w.]+)["\']?',
    re.I | re.S,
)


def _format_evidence_strings(locations: list[dict[str, Any]], available_views: list[dict[str, Any]]) -> list[str]:
    evidence = []
    for loc in locations:
        path = loc.get("file") or loc.get("path") or ""
        line_start = loc.get("line_start") or loc.get("line") or 0
        line_end = loc.get("line_end") or loc.get("line") or 0
        symbol = loc.get("symbol") or loc.get("symbol_or_api") or ""
        if line_start != line_end:
            evidence.append(f"{path}:{line_start}-{line_end} {symbol}".strip())
        else:
            evidence.append(f"{path}:{line_start} {symbol}".strip())
    for view in available_views:
        path = view.get("file") or ""
        line = view.get("line_start") or 0
        name = view.get("name") or ""
        if path and line:
            evidence.append(f"{path}:{line} CREATE VIEW {name}")
        elif name:
            evidence.append(f"CREATE VIEW {name}")
    return evidence


def build_handoff_contract(
    *,
    user_request: str,
    subtask_id: str,
    summary: str,
    search_output: str = "",
) -> dict[str, Any]:
    text = "\n".join(part for part in (summary, search_output) if part)
    locations = _extract_locations(text)
    available_views = _extract_available_views(text, locations)
    target_locations = _rank_target_locations(locations, user_request)
    current_sql = _extract_current_sql(locations)
    explicit_view = _is_explicit_view_replacement(user_request)
    replacement_source = _extract_replacement_source(text)
    target_view = replacement_source or (
        _infer_target_view(user_request, available_views) if explicit_view else ""
    )
    should_change_to = (
        f"use view {target_view}" if target_view else "apply the requested code change"
    )
    must_modify = [
        {
            "file": item["file"],
            "line": item["line"],
            "line_start": item.get("line_start") or item["line"],
            "line_end": item.get("line_end") or item["line"],
            "symbol_or_api": item.get("symbol") or "目标代码",
            "current_sql": current_sql,
            "current_code": item.get("snippet") or "",
            "should_change_to": should_change_to,
            "snippet": item.get("snippet") or "",
            "decision": should_change_to,
        }
        for item in target_locations[:3]
        if item["file"].endswith((".py", ".ts", ".tsx", ".js", ".jsx"))
    ]
    contract = {
        "schema": "mitkii.handoff.v1",
        "source_subtask": subtask_id,
        "must_modify": must_modify,
        "available_views": available_views,
        "evidence": _format_evidence_strings(locations[:12], available_views),
        "target_view": target_view,
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "edit_rules": [
                "old_string must copy the complete existing target code block exactly.",
                "new_string must differ from old_string.",
                "If the target function/API cannot be identified, do not edit; return need_more_context.",
            ],
        },
    }
    return contract


def merge_handoff_contracts(
    *,
    user_request: str,
    prior_summaries: dict[str, str],
    current_search_output: str = "",
    global_summaries: dict[str, str] | None = None,
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    seen_subtasks = set()
    for sid, text in prior_summaries.items():
        seen_subtasks.add(sid)
        parsed = extract_handoff_contract(text)
        if parsed:
            contracts.append(parsed)
        else:
            contracts.append(
                _normalize_contract_keys(build_handoff_contract(
                    user_request=user_request,
                    subtask_id=sid,
                    summary=text,
                ))
            )
            
    if global_summaries:
        for sid, text in global_summaries.items():
            if sid in seen_subtasks:
                continue
            parsed = extract_handoff_contract(text)
            if parsed:
                contracts.append(parsed)

    all = contracts
    if not any(c.get("source_subtask") == "current_search" for c in contracts):
        current = _normalize_contract_keys(build_handoff_contract(
            user_request=user_request,
            subtask_id="current_search",
            summary="",
            search_output=current_search_output,
        ))
        all = contracts + [current]

    all_views = _dedupe_views(
        [
            view
            for contract in all
            for view in contract.get("available_views", [])
            if isinstance(view, dict)
        ]
    )
    evidence = _dedupe_locations(
        [
            item
            for contract in all
            for item in contract.get("evidence", [])
            if isinstance(item, dict)
        ]
    )
    must_modify = [
        item
        for contract in all
        for item in contract.get("must_modify", [])
        if isinstance(item, dict)
    ]
    dependencies = [
        item
        for contract in all
        for item in (
            contract.get("dependencies")
            or contract.get("dependencies_to_use")
            or contract.get("resolved_dependencies")
            or []
        )
        if isinstance(item, dict)
    ]
    
    # Fallback to raw text extraction if no contract-specific items found
    if not all_views or not evidence:
        all_texts = [current_search_output]
        for text in prior_summaries.values():
            all_texts.append(text)
        if global_summaries:
            for text in global_summaries.values():
                all_texts.append(text)
        combined_text = "\n".join(all_texts)
        
        fallback_locations = _extract_locations(combined_text)
        fallback_views = _extract_available_views(combined_text, fallback_locations)
        
        if not all_views:
            all_views = _dedupe_views(fallback_views)
        if not evidence:
            evidence = _dedupe_locations(fallback_locations)
            
    explicit_view = _is_explicit_view_replacement(user_request)
    replacement_source = _extract_replacement_source(current_search_output)
    if not replacement_source and prior_summaries:
        for text in prior_summaries.values():
            replacement_source = _extract_replacement_source(text)
            if replacement_source:
                break
    if not replacement_source and global_summaries:
        for text in global_summaries.values():
            replacement_source = _extract_replacement_source(text)
            if replacement_source:
                break
    target_view = replacement_source or (
        _infer_target_view(user_request, all_views) if explicit_view else ""
    )
    if not target_view:
        for contract in contracts:
            if contract.get("target_view"):
                target_view = str(contract["target_view"]).strip()
                break
    fallback_change = (
        f"use view {target_view}" if target_view else "apply the requested code change"
    )
    if not must_modify:
        must_modify = [
            {
                "file": item["file"],
                "line": item["line"],
                "line_start": item.get("line_start") or item["line"],
                "line_end": item.get("line_end") or item["line"],
                "symbol_or_api": item.get("symbol") or "目标代码",
                "current_sql": _extract_current_sql(evidence),
                "current_code": item.get("snippet") or "",
                "should_change_to": fallback_change,
                "snippet": item.get("snippet") or "",
                "decision": fallback_change,
            }
            for item in _rank_target_locations(evidence, user_request)[:2]
            if str(item.get("file", "")).endswith((".py", ".ts", ".tsx", ".js", ".jsx"))
        ]
    
    full_must_modify = []
    for item in must_modify[:3]:
        line_val = item.get("line") or 0
        full_must_modify.append({
            "file": item.get("file"),
            "line": line_val,
            "line_start": item.get("line_start") or line_val,
            "line_end": item.get("line_end") or line_val,
            "symbol_or_api": item.get("symbol_or_api") or item.get("symbol") or "目标代码",
            "current_sql": item.get("current_sql") or _extract_current_sql(evidence),
            "current_code": item.get("current_code") or item.get("snippet") or "",
            "should_change_to": item.get("should_change_to") or fallback_change,
            "snippet": item.get("snippet") or "",
            "decision": item.get("decision") or item.get("should_change_to") or fallback_change,
        })

    full_dependencies = _dedupe_dependencies(dependencies)
    if not full_dependencies:
        full_dependencies = [
            {
                "role": "replacement_source",
                "kind": "database_view",
                "name": view.get("name") or "",
                "file": view.get("file") or "",
                "line_start": view.get("line_start") or 0,
                "line_end": view.get("line_end") or 0,
                "columns": view.get("columns") or view.get("fields") or [],
                "replaces_objects": view.get("replaces_objects") or [],
            }
            for view in all_views
            if isinstance(view, dict) and view.get("name")
        ]

    return _normalize_contract_keys({
        "schema": "mitkii.handoff.v1",
        "source_subtask": "merged",
        "must_modify": full_must_modify,
        "edit_targets": [
            {
                "file": item.get("file") or "",
                "symbol": item.get("symbol_or_api") or "目标代码",
                "line_start": item.get("line_start") or item.get("line") or 0,
                "line_end": item.get("line_end") or item.get("line") or 0,
                "snippet": item.get("snippet") or item.get("current_code") or "",
                "decision": item.get("decision") or item.get("should_change_to") or fallback_change,
            }
            for item in full_must_modify
        ],
        "available_views": all_views[:10],
        "dependencies": full_dependencies[:10],
        "dependencies_to_use": full_dependencies[:10],
        "resolved_dependencies": full_dependencies[:10],
        "evidence": _format_evidence_strings(evidence[:16], all_views[:10]),
        "target_view": target_view,
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "edit_rules": [
                "old_string must copy the complete existing target code block exactly.",
                "new_string must differ from old_string.",
                "If the target function/API cannot be identified, do not edit; return need_more_context.",
            ],
        },
    })


def _dedupe_dependencies(dependencies: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for dep in dependencies:
        kind = str(dep.get("kind") or "")
        name = str(dep.get("name") or "")
        key = (kind, name.lower())
        if key in seen:
            continue
        seen.add(key)
        if kind == "database_view":
            dep = dict(dep)
            dep.setdefault("role", "replacement_source")
            dep.setdefault("columns", dep.get("fields") or [])
            dep.setdefault("fields", dep.get("columns") or [])
            dep.setdefault("replaces_objects", [])
        out.append(dep)
    return out


def extract_handoff_contract(text: str) -> dict[str, Any] | None:
    marker = "HANDOFF_CONTRACT_JSON"
    if marker not in text:
        marker = "PATCH_INTENT_JSON"
        if marker not in text:
            return None
    payload = text.split(marker, 1)[1].strip()
    start = payload.find("{")
    if start < 0:
        return None

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
        return None

    try:
        data = json.loads(payload[start : end_idx + 1])
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return _normalize_contract_keys(data)
    return None


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
                        "file": dep.get("file") or "",
                        "line_start": dep.get("line_start") or 0,
                        "line_end": dep.get("line_end") or 0,
                        "fields": dep.get("fields") or dep.get("columns") or [],
                        "columns": dep.get("columns") or dep.get("fields") or [],
                        "replaces_objects": dep.get("replaces_objects") or [],
                    })
            contract["available_views"] = views
    if "dependencies" not in contract and "dependencies_to_use" in contract:
        contract["dependencies"] = contract["dependencies_to_use"]
    elif "dependencies" not in contract and "resolved_dependencies" in contract:
        contract["dependencies"] = contract["resolved_dependencies"]

    for key in ("dependencies", "dependencies_to_use", "resolved_dependencies"):
        deps = contract.get(key)
        if isinstance(deps, list):
            for dep in deps:
                if isinstance(dep, dict) and dep.get("kind") == "database_view":
                    if not dep.get("role"):
                        dep["role"] = "replacement_source"

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

    # Map back available_views to dependencies / resolved_dependencies / dependencies_to_use if missing
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
            if "dependencies_to_use" not in contract or not contract["dependencies_to_use"]:
                contract["dependencies_to_use"] = deps
            
    # Map evidence
    if "facts" in contract and "evidence" not in contract:
        contract["evidence"] = contract["facts"]
        
    # Ensure all required keys exist
    if "must_modify" not in contract:
        contract["must_modify"] = []
    if "available_views" not in contract:
        contract["available_views"] = []
    if "evidence" not in contract:
        contract["evidence"] = []
    if "target_view" not in contract:
        contract["target_view"] = ""
    if "available_columns" not in contract:
        # Try to find columns of the target_view
        cols = []
        target_view_name = contract.get("target_view") or ""
        for key in ("dependencies", "resolved_dependencies", "available_views"):
            deps = contract.get(key)
            if isinstance(deps, list):
                for dep in deps:
                    if isinstance(dep, dict) and (not target_view_name or str(dep.get("name")).lower() == target_view_name.lower()):
                        cols = list(dep.get("columns") or dep.get("fields") or [])
                        if cols:
                            break
                if cols:
                    break
        contract["available_columns"] = cols
        
    # Normalize must_modify items
    must_modify = contract.get("must_modify")
    if isinstance(must_modify, list):
        for item in must_modify:
            if isinstance(item, dict):
                if "symbol" in item and "symbol_or_api" not in item:
                    item["symbol_or_api"] = item["symbol"]
                elif "symbol_or_api" in item and "symbol" not in item:
                    item["symbol"] = item["symbol_or_api"]
                if "line_start" in item and "line" not in item:
                    item["line"] = item["line_start"]
                elif "line" in item and "line_start" not in item:
                    item["line_start"] = item["line"]
                elif "line" not in item:
                    item["line"] = 0
                if "line_end" not in item:
                    item["line_end"] = item.get("line_start") or item.get("line") or 0
                if "decision" in item and "should_change_to" not in item:
                    item["should_change_to"] = item["decision"]
                elif "should_change_to" in item and "decision" not in item:
                    item["decision"] = item["should_change_to"]
                elif "should_change_to" not in item:
                    item["should_change_to"] = "apply change"
                if "current_code" in item and "snippet" not in item:
                    item["snippet"] = item["current_code"]
                elif "snippet" in item and "current_code" not in item:
                    item["current_code"] = item["snippet"]
                elif "snippet" not in item:
                    item["snippet"] = item.get("current_sql") or ""
                    item["current_code"] = item["snippet"]

    # Normalize available_views items
    available_views = contract.get("available_views")
    if isinstance(available_views, list):
        for item in available_views:
            if isinstance(item, dict):
                if "columns" in item and "fields" not in item:
                    item["fields"] = item["columns"]
                elif "fields" in item and "columns" not in item:
                    item["columns"] = item["fields"]

    # Normalize evidence items
    evidence = contract.get("evidence")
    if isinstance(evidence, list):
        normalized_evidence = []
        for item in evidence:
            if isinstance(item, str):
                match = _LOCATION_RE.search(item)
                if match:
                    path = match.group("path")
                    line_str = match.group("line")
                    if "-" in line_str:
                        line_no = int(line_str.split("-")[0])
                    else:
                        line_no = int(line_str)
                    symbol = (match.group("symbol") or "").strip()
                    snippet = (match.group("snippet") or "").strip()
                    normalized_evidence.append({
                        "file": path,
                        "line": line_no,
                        "line_start": line_no,
                        "line_end": line_no,
                        "symbol": symbol,
                        "snippet": snippet,
                    })
                else:
                    normalized_evidence.append({
                        "file": item,
                        "line": 0,
                        "line_start": 0,
                        "line_end": 0,
                        "symbol": "",
                        "snippet": "",
                    })
            elif isinstance(item, dict):
                normalized_evidence.append(item)
        contract["evidence"] = normalized_evidence
        
    # Extract target_view if missing
    if "target_view" not in contract or not contract["target_view"]:
        target_view = ""
        must_modify = contract.get("must_modify") or []
        for item in must_modify:
            if isinstance(item, dict):
                decision = item.get("decision") or item.get("should_change_to") or ""
                m = re.search(r"view\s+(\w+)", decision, re.I)
                if m:
                    target_view = m.group(1)
                    break
        if not target_view and contract.get("available_views"):
            views = contract.get("available_views")
            if views and isinstance(views[0], dict):
                target_view = views[0].get("name") or ""
        contract["target_view"] = target_view
                    
    return contract


def format_handoff_contract(contract: dict[str, Any]) -> str:
    return "HANDOFF_CONTRACT_JSON\n" + json.dumps(contract, ensure_ascii=False, indent=2)


def _extract_locations(text: str) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int, str]] = set()
    for raw in text.splitlines():
        line = raw.strip().lstrip("- ").strip()
        match = _LOCATION_RE.search(line)
        if not match:
            continue
        path = match.group("path")
        line_str = match.group("line")
        if "-" in line_str:
            parts = line_str.split("-")
            line_start = int(parts[0])
            line_end = int(parts[1])
            line_no = line_start
        else:
            line_no = int(line_str)
            line_start = line_no
            line_end = line_no
        symbol = (match.group("symbol") or "").strip()
        snippet = (match.group("snippet") or "").strip()
        key = (path, line_no, snippet)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "file": path,
            "line": line_no,
            "line_start": line_start,
            "line_end": line_end,
            "symbol": symbol,
            "snippet": snippet,
        })
    return out


def _extract_available_views(text: str, locations: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _VIEW_RE.finditer(text):
        name = match.group("name")
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        
        file_path = ""
        line_start = 0
        line_end = 0
        if locations:
            for loc in locations:
                loc_file = loc.get("file", "")
                loc_snippet = loc.get("snippet", "")
                if loc_file.endswith(".sql") and name in loc_snippet:
                    file_path = loc_file
                    line_start = loc.get("line_start") or loc.get("line") or 0
                    line_end = loc.get("line_end") or loc.get("line") or line_start
                    break
        views.append({
            "name": name,
            "fields": [],
            "file": file_path,
            "line_start": line_start,
            "line_end": line_end,
            "columns": []
        })
    return views


def _is_explicit_view_replacement(text: str) -> bool:
    lowered = text.lower()
    has_view = "视图" in text or "view" in lowered
    has_replace = any(
        marker in lowered
        for marker in ("replace", "use", "switch", "using")
    ) or any(marker in text for marker in ("替换", "使用", "改成", "改为", "用已有视图", "用现有视图"))
    return has_view and has_replace


def _extract_replacement_source(text: str) -> str:
    tv_match = re.search(r'["\']?target_view["\']?\s*:\s*["\']?(?P<name>[\w.]+)["\']?', text, re.I)
    if tv_match:
        return tv_match.group("name")
    match = _REPLACEMENT_SOURCE_RE.search(text)
    return match.group("name") if match else ""


def _rank_target_locations(locations: list[dict[str, Any]], user_request: str) -> list[dict[str, Any]]:
    intent = user_request.lower()
    terms = [term for term in ("登机牌", "boarding", "ticket", "接口", "api", "query", "查询") if term in intent]

    def score(item: dict[str, Any]) -> int:
        text = f"{item.get('file', '')} {item.get('symbol', '')} {item.get('snippet', '')}".lower()
        value = 0
        if str(item.get("file", "")).endswith(".py"):
            value += 5
        if any(term.lower() in text for term in terms):
            value += 6
        if any(token in text for token in ("select", "from", "route", "api", "query", "登机牌")):
            value += 3
        return value

    return sorted(locations, key=score, reverse=True)


def _extract_current_sql(locations: list[dict[str, Any]]) -> str:
    snippets = [str(item.get("snippet") or "") for item in locations]
    sqlish = [snippet for snippet in snippets if "SELECT" in snippet.upper() or " FROM " in snippet.upper()]
    return " ".join(sqlish[:3])[:500]


def _infer_target_view(user_request: str, views: list[dict[str, Any]]) -> str:
    if not views:
        return ""
    
    view_names = []
    for v in views:
        if isinstance(v, dict):
            name = str(v.get("name") or "")
            if name:
                view_names.append(name)
        elif isinstance(v, str):
            view_names.append(v)
            
    if not view_names:
        return ""
        
    lowered = user_request.lower()
    best_view = ""
    best_score = -1
    
    for view_name in view_names:
        name_lower = view_name.lower()
        score = 0
        
        if name_lower in lowered:
            score += 100
            
        # Enhanced matching with Chinese/pinyin/English combinations
        groups = [
            ({"登机牌", "boarding", "dengjipai", "dengji", "pass"}, {"boarding", "pass", "ticket", "dengji", "dengjipai"}),
            ({"订单", "order", "dingdan"}, {"order", "report", "detail", "dingdan"}),
            ({"机票", "ticket", "jipiao"}, {"ticket", "report", "detail", "jipiao", "passenger"}),
            ({"监控", "monitor", "flight", "jiankong"}, {"monitor", "flight", "jiankong", "load"}),
            ({"详情", "detail", "xiangqing"}, {"detail", "xiangqing"}),
            ({"报告", "report", "baogao"}, {"report", "baogao"}),
        ]
        for query_set, view_set in groups:
            if any(q_term in lowered for q_term in query_set):
                if any(v_term in name_lower for v_term in view_set):
                    score += 20
                    
        # Substring/word matching
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", user_request.replace("_", " ").lower())
        for w in words:
            if len(w) > 2 and w not in {"use", "view", "sql", "query", "replace", "change"}:
                if w in name_lower:
                    score += 5
                    
        if score > best_score:
            best_score = score
            best_view = view_name
            
    if best_score <= 0 or not best_view:
        return view_names[0]
    return best_view


def _dedupe_views(views: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for view in views:
        name = str(view.get("name") or "")
        if not name or name.lower() in seen:
            continue
        seen.add(name.lower())
        out.append(view)
    return out


def _dedupe_locations(locations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, int]] = set()
    for item in locations:
        path = str(item.get("file") or "")
        line = int(item.get("line") or 0)
        key = (path, line)
        if not path or not line or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
