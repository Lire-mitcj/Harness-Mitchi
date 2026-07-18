from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from src.agent.grep_discovery import grep_patterns_for_task, grep_scope_for_task

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
_RAW_STRING_FIELD = re.compile(r'"(pattern|include|path|query|mode)"\s*:\s*"([^"]*)')


def unwrap_raw_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Salvage fields from streamed/truncated tool-call JSON stored under ``_raw``."""
    args = dict(arguments)
    raw = args.pop("_raw", None)
    if not isinstance(raw, str) or not raw.strip():
        return args

    parsed: dict[str, Any] | None = None
    try:
        loaded = json.loads(raw)
        if isinstance(loaded, dict):
            parsed = loaded
    except json.JSONDecodeError:
        parsed = None

    if parsed is not None:
        for key, value in parsed.items():
            if key not in args or args.get(key) in (None, "", [], ()):
                args[key] = value
        return args

    for match in _RAW_STRING_FIELD.finditer(raw):
        key, value = match.group(1), match.group(2)
        if key not in args or not str(args.get(key) or "").strip():
            args[key] = value
    return args


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

    if any(word in lowered for word in (
        "exception", "handler", "异常", "错误处理", "logging", "日志", "统一",
    )):
        add("@app\\.exception_handler")
        add("add_exception_handler")
        add("async def handle_")
        add("def _handle_")
        add("logger\\.exception")

    return patterns


def normalize_grep_search_args(
    arguments: dict[str, Any],
    *,
    subtask: Any = None,
    hint_text: str = "",
) -> dict[str, Any]:
    """Fill missing grep_search.pattern/patterns from task text when the model omits them."""
    args = unwrap_raw_tool_arguments(arguments)
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
    if not patterns and hint_text.strip():
        patterns = grep_patterns_for_task(hint_text)
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

    if hint_text.strip() and "include" not in args:
        include, path_hint = grep_scope_for_task(hint_text)
        if include:
            args.setdefault("include", include)
        if path_hint and args.get("path", ".") == ".":
            args["path"] = path_hint

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
