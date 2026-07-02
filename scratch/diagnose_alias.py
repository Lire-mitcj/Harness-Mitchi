import re
from pathlib import Path

_SQL_RESERVED_ALIASES = {
    "select",
    "from",
    "join",
    "where",
    "and",
    "or",
    "on",
    "as",
    "order",
    "group",
    "by",
    "limit",
    "offset",
}

def _sql_alias_safety(content: str) -> dict[str, object]:
    aliases = {
        alias.casefold()
        for alias in re.findall(
            r"\b(?:from|join)\s+[`\"\[]?[\w.]+[`\"\]]?(?:\s+(?:as\s+)?)"
            r"([A-Za-z_][A-Za-z0-9_]*)\b",
            content,
            re.IGNORECASE,
        )
        if alias.casefold() not in _SQL_RESERVED_ALIASES
    }
    table_names = {
        name.split(".")[-1].strip("`\"[]").casefold()
        for name in re.findall(
            r"\b(?:from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)",
            content,
            re.IGNORECASE,
        )
    }
    used_aliases = {
        alias.casefold()
        for alias in re.findall(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*\.", content)
    }
    used_aliases -= table_names
    used_aliases -= _SQL_RESERVED_ALIASES
    if not used_aliases:
        return {"checked": False, "pass": True, "aliases": sorted(aliases), "dead_aliases": []}
    dead_aliases = sorted(used_aliases - aliases)
    return {
        "checked": True,
        "pass": not dead_aliases,
        "aliases": sorted(aliases),
        "used_aliases": sorted(used_aliases),
        "dead_aliases": dead_aliases,
    }

content = Path("db/init/init.sql").read_text(encoding="utf-8")
res = _sql_alias_safety(content)
print("Result:", res)
