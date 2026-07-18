"""Task-derived grep pattern batches and view-symbol anti-patterns for discovery."""

from __future__ import annotations

import re
from collections.abc import Sequence
from pathlib import Path

from src.agent.evidence_slots import (
    MAX_BATCH_PATTERNS,
    slot_grep_hint,
    themes_for_task,
)
from src.indexer.project_stack import default_include_for_stack, detect_project_stack

# view_symbol_code targets that are usually imports/decorators, not definitions.
_VIEW_SYMBOL_BLOCKLIST = frozenset({
    "exception_handler",
    "SQLAlchemyError",
    "RequestValidationError",
    "JSONResponse",
    "HTTPException",
    "logging",
    "logger",
    "app",
})


def _lowered(task_text: str) -> str:
    return task_text.casefold()


def grep_patterns_for_task(
    task_text: str,
    *,
    project_root: Path | None = None,
) -> list[str]:
    """Concrete grep_search patterns derived from the user task (max 8)."""
    lowered = _lowered(task_text)
    themes = themes_for_task(task_text)
    patterns: list[str] = []

    def add(*values: str) -> None:
        for value in values:
            text = value.strip()
            if text and text not in patterns and len(patterns) < MAX_BATCH_PATTERNS:
                patterns.append(text)

    stack = detect_project_stack(project_root) if project_root is not None else None
    if stack is not None:
        for pattern in stack.discovery_patterns:
            add(pattern)

    if stack is None or stack.primary in {"python", "mixed"}:
        if "exception_handler" in themes or (
            "db_integration" in themes and any(w in lowered for w in ("异常", "exception", "日志", "logging", "统一"))
        ):
            add(
                "@app\\.exception_handler",
                "add_exception_handler",
                "async def handle_",
                "def _handle_",
                "logger\\.exception",
            )

        if "endpoint" in themes or "db_integration" in themes:
            add("@router\\.", "build_router", "engine\\.(connect|begin)")

    if stack is not None and stack.primary in {"go", "mixed"}:
        if any(w in lowered for w in ("grpc", "proto", "rpc", "handler", "http")):
            add("func.*Handler", "grpc\\.", "http\\.Handle")

    if stack is not None and stack.primary in {"java", "mixed"}:
        if any(w in lowered for w in ("controller", "service", "spring", "rest", "api")):
            add("@RestController", "@Service", "@GetMapping", "@Autowired")

    if "schema" in themes:
        add("CREATE TABLE", "CREATE VIEW")

    if "auth" in themes:
        add("get_current_user", "Bearer")

    if any(w in lowered for w in ("proto", "protobuf", "grpc")):
        add("service ", "rpc ", "message ")

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]{2,})\b", task_text):
        token = match.group(1)
        if token.casefold() in {
            "the", "and", "for", "with", "from", "that", "this", "task", "file", "code",
            "api", "sql", "def", "class", "true", "false", "none", "return", "import",
            "main", "list", "python",
        }:
            continue
        if "_" in token or any(ch.isupper() for ch in token[1:]):
            add(token)

    return patterns[:MAX_BATCH_PATTERNS]


def grep_scope_for_task(
    task_text: str,
    *,
    project_root: Path | None = None,
) -> tuple[str | None, str | None]:
    """Return (include_glob, path) hints for the primary grep batch."""
    themes = themes_for_task(task_text)
    lowered = _lowered(task_text)
    stack = detect_project_stack(project_root) if project_root is not None else None

    if stack is not None and stack.primary == "go":
        return "*.go", "."
    if stack is not None and stack.maven_modules:
        return "*.{java,xml}", stack.maven_modules[0]
    if stack is not None and stack.primary == "java":
        return "*.{java,xml}", "."
    if any(w in lowered for w in ("proto", "protobuf", "grpc")):
        return "*.proto", "."

    if "exception_handler" in themes and any(
        word in lowered for word in ("main.py", "统一", "全局", "global", "unified")
    ):
        return "main.py", "main.py"
    if "exception_handler" in themes:
        return "main.py", "main.py"
    if "schema" in themes and "endpoint" not in themes:
        return "*.sql", "db"
    if stack is not None:
        include = default_include_for_stack(stack)
        if include:
            return include, "."
    return "*.py", "."


def view_symbol_avoid(task_text: str) -> tuple[str, ...]:
    """Symbols that look relevant in imports but are not view_symbol_code targets."""
    themes = themes_for_task(task_text)
    if "exception_handler" not in themes and "db_integration" not in themes:
        return ()
    return tuple(sorted(_VIEW_SYMBOL_BLOCKLIST))


def discovery_hint_lines(
    task_text: str,
    task_slots: Sequence[str],
    *,
    project_root: Path | None = None,
) -> list[str]:
    """Render STEP EVIDENCE discovery hints from task text and evidence slots."""
    lines: list[str] = []
    themes = themes_for_task(task_text)
    stack = detect_project_stack(project_root) if project_root is not None else None
    if stack is not None:
        lines.append(
            f"- project_stack: {stack.primary} ({', '.join(stack.languages)}) "
            f"validator={list(stack.validator_command)!r}"
        )
        if stack.maven_modules:
            preview = ", ".join(stack.maven_modules[:8])
            extra = "..." if len(stack.maven_modules) > 8 else ""
            lines.append(f"- maven_modules: {preview}{extra}")

    patterns = grep_patterns_for_task(task_text, project_root=project_root)
    include, path = grep_scope_for_task(task_text, project_root=project_root)
    if patterns:
        batch = ", ".join(repr(p) for p in patterns[:6])
        extra = ""
        if len(patterns) > 6:
            extra = ", ..."
        scope = f"include={include!r}, path={path!r}" if include else f"path={path!r}"
        lines.append(
            f"- task_batch: grep_search(patterns=[{batch}{extra}], {scope}) "
            "→ then view_symbol_code on suggested_views (def names, not imports)"
        )

    if "exception_handler" in themes:
        lines.append(
            "- exception_handler: grep @app.exception_handler / def handle_* in main.py; "
            "view the async def name (e.g. handle_db_error), NOT "
            "SQLAlchemyError/RequestValidationError/exception_handler/logger"
        )
    if "db_integration" in themes and "endpoint" in themes:
        lines.append(
            "- db_routes: list.py build_router uses engine.connect/begin — "
            "grep engine\\.(connect|begin) in list.py after loading handler pattern from main.py"
        )

    for slot in task_slots[:6]:
        hint = slot_grep_hint(slot)
        if hint is None:
            continue
        include_glob, path_hint, pattern_tuple = hint
        if slot == "exception_handler_context" and "exception_handler" not in themes:
            continue
        pat_hint = "|".join(pattern_tuple[:4])
        lines.append(
            f"- {slot}: grep_search(patterns=[{pat_hint!r}, ...], "
            f"include={include_glob!r}, path={path_hint!r})"
        )

    avoid = view_symbol_avoid(task_text)
    if avoid:
        preview = ", ".join(repr(name) for name in avoid[:6])
        lines.append(
            f"- avoid_view_symbols: {preview} — imports/decorators/module vars; "
            "use suggested_views or def handle_* names from grep"
        )

    return lines
