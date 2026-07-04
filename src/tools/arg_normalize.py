from __future__ import annotations

import re
from pathlib import Path
from typing import Any

MAX_BATCH_PATTERNS = 8

_DEFAULT_FALLBACK_PATTERNS = [
    "CREATE TABLE",
    "build_router",
    "@router\\.",
    "APIRouter",
    "include_router",
]

_TASK_IDENTIFIER = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b")
_STORED_PROC = re.compile(r"\b(sp_[A-Za-z0-9_]+)\b", re.IGNORECASE)


def _has_pattern(args: dict[str, Any]) -> bool:
    pattern = str(args.get("pattern") or "").strip()
    if pattern:
        return True
    raw = args.get("patterns")
    if isinstance(raw, list):
        return any(str(item or "").strip() for item in raw)
    return False


def _grep_patterns_from_hint(hint_text: str) -> list[str]:
    blob = hint_text.strip()
    if not blob:
        return []

    lowered = blob.casefold()
    patterns: list[str] = []

    def add(value: str) -> None:
        text = value.strip()
        if text and text not in patterns and len(patterns) < MAX_BATCH_PATTERNS:
            patterns.append(text)

    for match in _STORED_PROC.finditer(blob):
        add(match.group(1))

    for match in _TASK_IDENTIFIER.finditer(blob):
        token = match.group(1)
        if token.casefold() in {
            "the", "and", "for", "with", "from", "that", "this", "task", "file", "code",
            "api", "sql", "def", "class", "true", "false", "none", "return", "import",
        }:
            continue
        if "_" in token or any(ch.isupper() for ch in token[1:]):
            add(token)

    if any(word in lowered for word in (
        "schema", "table", "database", "sql", "建表", "数据库", "表结构", "新建表",
    )):
        add("CREATE TABLE")
        add("CREATE VIEW")

    if any(word in lowered for word in (
        "endpoint", "route", "router", "接口", "路由", "timeline", "时间线",
    )):
        add("@router\\.")
        add("build_router")
        add("APIRouter")
        add("include_router")

    if "order" in lowered:
        add("order")
    if "ticket" in lowered:
        add("ticket")
    if "timeline" in lowered:
        add("timeline")

    if any(word in lowered for word in ("auth", "login", "token", "认证", "登录")):
        add("get_current_user")
        add("Bearer")

    return patterns


def normalize_grep_search_args(
    arguments: dict[str, Any],
    *,
    subtask: Any = None,
    hint_text: str = "",
) -> dict[str, Any]:
    """Fill missing grep_search.pattern/patterns from task text when the model omits them."""
    args = dict(arguments)
    if _has_pattern(args):
        return args

    blob = hint_text
    if subtask is not None:
        blob = " ".join(
            part
            for part in (
                getattr(subtask, "description", "") or "",
                getattr(subtask, "acceptance_criteria", "") or "",
                " ".join(getattr(subtask, "context_files", ()) or ()),
                hint_text,
            )
            if part
        )

    patterns = _grep_patterns_from_hint(blob)
    if not patterns:
        for word in (
            "register", "signup", "transaction", "commit", "rollback",
            "session", "passenger", "order", "ticket", "timeline",
        ):
            if word in blob.casefold():
                patterns.append(word)
        patterns = list(dict.fromkeys(patterns))

    if subtask is not None and getattr(subtask, "context_files", None) and "path" not in args:
        args["path"] = subtask.context_files[0]

    args.setdefault("path", ".")

    if not patterns:
        if subtask is not None:
            patterns = list(_DEFAULT_FALLBACK_PATTERNS)
        else:
            return args

    if len(patterns) == 1:
        args["pattern"] = patterns[0]
    else:
        args["patterns"] = patterns[:MAX_BATCH_PATTERNS]
        args.pop("pattern", None)
    return args


def normalize_shell_exec_args(
    arguments: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Default shell cwd to project_root; reject hallucinated paths like /workspace."""
    args = dict(arguments)
    root = project_root.resolve()
    wd = args.get("working_dir")
    if not isinstance(wd, str) or not wd.strip():
        args["working_dir"] = str(root)
        return args
    candidate = Path(wd).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_dir():
        args["working_dir"] = str(candidate)
    else:
        args["working_dir"] = str(root)
    return args
