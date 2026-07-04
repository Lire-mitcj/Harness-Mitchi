from __future__ import annotations

import re
from typing import Any

_SQL_DEFINITION_KINDS = "TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER|EVENT"


def extract_symbol_from_match_line(content: str) -> str:
    """Derive a view_symbol_code target from a single grep match line."""
    py_match = re.search(
        r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)",
        content,
    )
    if py_match:
        return py_match.group(1)

    sql_match = re.search(
        r"(?is)\bCREATE\s+(?:OR\s+REPLACE\s+)?"
        r"(?:DEFINER\s*=\s*\S+\s+)?(?:TEMP\s+|TEMPORARY\s+)?"
        rf"(?:{_SQL_DEFINITION_KINDS})\s+(?:IF\s+NOT\s+EXISTS\s+)?"
        r"(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))",
        content,
    )
    if sql_match:
        for group in sql_match.groups():
            if group:
                return group

    router_match = re.search(
        r"@(?:app|router)\.\w+\([^)]*\).*?\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)",
        content,
    )
    if router_match:
        return router_match.group(1)

    return ""


def suggested_views_from_matches(
    matches: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Rank unique file+symbol pairs for the next view_symbol_code step."""
    views: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in matches:
        file_path = str(item.get("file") or "").strip()
        symbol = str(item.get("symbol") or "").strip()
        span = item.get("span")
        if not file_path or not symbol:
            continue
        key = (file_path, symbol)
        if key in seen:
            continue
        seen.add(key)
        view: dict[str, Any] = {"file": file_path, "symbol": symbol}
        if isinstance(span, list) and len(span) >= 2:
            view["span"] = [int(span[0]), int(span[1])]
        views.append(view)
        if len(views) >= limit:
            break
    return views
