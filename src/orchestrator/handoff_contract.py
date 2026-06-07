from __future__ import annotations

import json
import re
from typing import Any

_LOCATION_RE = re.compile(
    r"(?P<path>[\w./-]+\.(?:py|sql|tsx?|jsx?|json|ya?ml|toml)):"
    r"(?P<line>\d+)\s*(?:\|\s*(?P<symbol>[^|]+)\|\s*(?P<snippet>.*))?"
)
_VIEW_RE = re.compile(r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:FORCE\s+)?(?:VIEW|TABLE)\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"'\`\[]?(?P<name>[\w.]+)", re.I)
_FROM_RE = re.compile(r"\bFROM\s+(?P<table>[\w.]+)", re.I)


def build_handoff_contract(
    *,
    user_request: str,
    subtask_id: str,
    summary: str,
    search_output: str = "",
) -> dict[str, Any]:
    text = "\n".join(part for part in (summary, search_output) if part)
    locations = _extract_locations(text)
    available_views = _extract_available_views(text)
    target_locations = _rank_target_locations(locations, user_request)
    current_sql = _extract_current_sql(locations)
    target_view = _infer_target_view(user_request, available_views)
    contract = {
        "schema": "mitkii.handoff_contract.v1",
        "source_subtask": subtask_id,
        "must_modify": [
            {
                "file": item["file"],
                "line": item["line"],
                "symbol_or_api": item.get("symbol") or "目标代码",
                "current_sql": current_sql,
                "should_change_to": (
                    f"use view {target_view}" if target_view else "use the matching database view"
                ),
            }
            for item in target_locations[:3]
            if item["file"].endswith((".py", ".ts", ".tsx", ".js", ".jsx"))
        ],
        "available_views": available_views,
        "evidence": locations[:12],
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
) -> dict[str, Any]:
    contracts: list[dict[str, Any]] = []
    for sid, text in prior_summaries.items():
        parsed = extract_handoff_contract(text)
        if parsed:
            contracts.append(parsed)
        else:
            contracts.append(
                build_handoff_contract(
                    user_request=user_request,
                    subtask_id=sid,
                    summary=text,
                )
            )
    current = build_handoff_contract(
        user_request=user_request,
        subtask_id="current_search",
        summary="",
        search_output=current_search_output,
    )
    all_views = _dedupe_views(
        [
            view
            for contract in contracts + [current]
            for view in contract.get("available_views", [])
            if isinstance(view, dict)
        ]
    )
    evidence = _dedupe_locations(
        [
            item
            for contract in contracts + [current]
            for item in contract.get("evidence", [])
            if isinstance(item, dict)
        ]
    )
    must_modify = [
        item
        for contract in contracts + [current]
        for item in contract.get("must_modify", [])
        if isinstance(item, dict)
    ]
    if not must_modify:
        must_modify = [
            {
                "file": item["file"],
                "line": item["line"],
                "symbol_or_api": item.get("symbol") or "目标代码",
                "current_sql": _extract_current_sql(evidence),
                "should_change_to": "use the matching database view",
            }
            for item in _rank_target_locations(evidence, user_request)[:2]
            if str(item.get("file", "")).endswith((".py", ".ts", ".tsx", ".js", ".jsx"))
        ]
    return {
        "schema": "mitkii.handoff_contract.v1",
        "source_subtask": "merged",
        "must_modify": must_modify[:3],
        "available_views": all_views[:10],
        "evidence": evidence[:16],
        "tool_policy": {
            "allowed_tools": ["edit_file"],
            "edit_rules": [
                "old_string must copy the complete existing target code block exactly.",
                "new_string must differ from old_string.",
                "If the target function/API cannot be identified, do not edit; return need_more_context.",
            ],
        },
    }


def extract_handoff_contract(text: str) -> dict[str, Any] | None:
    marker = "HANDOFF_CONTRACT_JSON"
    if marker not in text:
        return None
    payload = text.split(marker, 1)[1].strip()
    start = payload.find("{")
    end = payload.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(payload[start : end + 1])
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


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
        line_no = int(match.group("line"))
        symbol = (match.group("symbol") or "").strip()
        snippet = (match.group("snippet") or "").strip()
        key = (path, line_no, snippet)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "file": path,
            "line": line_no,
            "symbol": symbol,
            "snippet": snippet,
        })
    return out


def _extract_available_views(text: str) -> list[dict[str, Any]]:
    views: list[dict[str, Any]] = []
    seen: set[str] = set()
    for match in _VIEW_RE.finditer(text):
        name = match.group("name")
        if name.lower() in seen:
            continue
        seen.add(name.lower())
        views.append({"name": name, "fields": []})
    return views


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
    lowered = user_request.lower()
    
    best_view = ""
    best_score = -1
    for view in views:
        name = str(view.get("name") or "")
        name_lower = name.lower()
        score = 0
        
        # Keyword matching
        if "登机牌" in user_request or "boarding" in lowered:
            if "boarding" in name_lower or "ticket" in name_lower:
                score += 20
        if "订单" in user_request or "order" in lowered:
            if "order" in name_lower or "report" in name_lower or "detail" in name_lower:
                score += 20
        if "机票" in user_request or "ticket" in lowered:
            if "ticket" in name_lower or "report" in name_lower or "detail" in name_lower:
                score += 20
        if "监控" in user_request or "monitor" in lowered or "flight" in lowered:
            if "monitor" in name_lower or "flight" in name_lower:
                score += 20
                
        # Basic substring matches from user_request words
        words = re.findall(r"\b[a-zA-Z0-9_]+\b", user_request.replace("_", " ").lower())
        for w in words:
            if len(w) > 2 and w not in {"use", "view", "sql", "query", "replace", "change"}:
                if w in name_lower:
                    score += 5
                    
        if score > best_score:
            best_score = score
            best_view = name
            
    if best_score <= 0:
        return str(views[0].get("name") or "")
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
