from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.indexer.language_profiles import (
    SQL,
    classify_line_with_profiles,
    default_profiles,
    extract_symbol_with_profiles,
    profile_for_path,
    profiles_from_include_glob,
)
from src.indexer.project_stack import detect_project_stack

_SQL_DEFINITION_KINDS = "TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER|EVENT"
_DECORATOR_LINE_RE = re.compile(
    r"@(?:app|router)\.(?:exception_handler|middleware|(?:get|post|put|delete|patch)\()",
)
_MOUNT_LINE_RE = re.compile(
    r"\binclude_router\s*\(|\bFastAPI\s*\(|\bcreate_app\s*\(|\badd_exception_handler\s*\(",
)
_DEFINITION_LINE_RE = re.compile(r"\b(?:async\s+def|def|class)\s+")
_SCHEMA_LINE_RE = re.compile(r"\bCREATE\s+(?:TABLE|VIEW)\b", re.IGNORECASE)


def _active_profiles(
    *,
    file_path: str = "",
    include: str | None = None,
    project_root: Path | None = None,
) -> tuple[Any, ...]:
    narrowed = profiles_from_include_glob(include)
    if narrowed:
        return narrowed
    profile = profile_for_path(file_path)
    if profile is not None:
        return (profile,)
    if project_root is not None:
        return detect_project_stack(project_root).profiles()
    return default_profiles()


def extract_symbol_from_match_line(
    content: str,
    *,
    file_path: str = "",
    project_root: Path | None = None,
) -> str:
    """Derive a view_symbol_code target from a single grep match line."""
    profiles = _active_profiles(file_path=file_path, project_root=project_root)
    sql_symbol = extract_symbol_with_profiles(content, (SQL,))
    if sql_symbol:
        return sql_symbol
    symbol = extract_symbol_with_profiles(content, profiles)
    if symbol:
        return symbol
    return ""


def fill_symbols_from_adjacent_lines(
    matches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Resolve decorator-only grep hits to the following def/class symbol."""
    by_file: dict[str, list[dict[str, Any]]] = {}
    for item in matches:
        file_path = str(item.get("file") or "").strip()
        if file_path:
            by_file.setdefault(file_path, []).append(item)

    decorator_re = re.compile(
        r"@(?:app|router)\.(?:exception_handler|middleware|(?:get|post|put|delete|patch)\()",
    )
    for file_matches in by_file.values():
        file_matches.sort(key=lambda row: int((row.get("span") or [0])[0]))
        for index, item in enumerate(file_matches):
            if str(item.get("symbol") or "").strip():
                continue
            line = str(item.get("match_line") or "")
            if not decorator_re.search(line):
                continue
            start_line = int((item.get("span") or [0])[0])
            for nxt in file_matches[index + 1 : index + 4]:
                next_line = int((nxt.get("span") or [0])[0])
                if next_line - start_line > 3:
                    break
                symbol = extract_symbol_from_match_line(str(nxt.get("match_line") or ""))
                if symbol:
                    item["symbol"] = symbol
                    break
    return matches


def resolve_decorator_symbols_from_files(
    matches: list[dict[str, Any]],
    *,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """When a decorator line matched grep but the def is not in the batch, peek the file."""
    from pathlib import Path as _Path

    root = project_root or _Path.cwd()
    decorator_re = re.compile(
        r"@(?:app|router)\.(?:exception_handler|middleware)|"
        r"\badd_exception_handler\s*\(",
    )
    for item in matches:
        if str(item.get("symbol") or "").strip():
            continue
        line = str(item.get("match_line") or "")
        if not decorator_re.search(line):
            continue
        file_path = str(item.get("file") or "").strip().lstrip("./")
        if not file_path:
            continue
        abs_path = (root / file_path).resolve()
        if not abs_path.is_file():
            continue
        try:
            file_lines = abs_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        start_line = int((item.get("span") or [0])[0])
        if start_line <= 0:
            continue
        for offset in range(1, 5):
            index = start_line - 1 + offset
            if index >= len(file_lines):
                break
            symbol = extract_symbol_from_match_line(file_lines[index])
            if symbol:
                item["symbol"] = symbol
                item["span"] = [start_line, index + 1]
                item["resolved_from"] = "decorator_context"
                break
    return matches


def resolve_mount_line_symbols(
    matches: list[dict[str, Any]],
    *,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Resolve include_router / mount grep hits to an enclosing def span for viewing."""
    root = project_root or Path.cwd()
    mount_re = re.compile(r"\binclude_router\s*\(")

    for item in matches:
        if str(item.get("symbol") or "").strip():
            continue
        line = str(item.get("match_line") or "")
        if not mount_re.search(line):
            continue
        file_path = str(item.get("file") or "").strip().lstrip("./")
        if not file_path:
            continue
        abs_path = (root / file_path).resolve()
        if not abs_path.is_file():
            continue
        try:
            file_lines = abs_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        start_line = int((item.get("span") or [0])[0])
        if start_line <= 0:
            continue

        symbol = ""
        context_start = start_line
        for index in range(start_line - 1, max(0, start_line - 25), -1):
            match = re.search(r"\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)", file_lines[index])
            if match:
                symbol = match.group(1)
                context_start = index + 1
                break
        if not symbol:
            symbol, context_start, end_line = infer_caller_setup_symbol(
                file_lines,
                mount_line=start_line,
            )
        else:
            end_line = min(len(file_lines), start_line + 2)

        item["symbol"] = symbol
        item["span"] = [context_start, end_line]
        item["resolved_from"] = "mount_context"
    return matches


def infer_caller_setup_symbol(
    file_lines: Sequence[str],
    *,
    mount_line: int | None = None,
) -> tuple[str, int, int]:
    """Infer (symbol, start_line, end_line) for a FastAPI caller / mount setup region."""
    line_count = len(file_lines)
    scan_limit = line_count
    if mount_line is not None and mount_line > 0:
        scan_limit = min(line_count, mount_line)

    for index in range(scan_limit):
        line = file_lines[index]
        factory = re.search(
            r"\bdef\s+(create_app|wire_routes|make_app|build_app|init_app)\s*\(",
            line,
        )
        if factory:
            start = index + 1
            end = min(line_count, (mount_line or start) + 12)
            return factory.group(1), start, end

    for index in range(scan_limit):
        line = file_lines[index]
        assign = re.search(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*FastAPI\s*\(",
            line,
        )
        if assign:
            name = assign.group(1)
            start = max(1, index + 1 - 2)
            end = min(line_count, (mount_line or index + 1) + 12)
            return name, start, end

    end = min(line_count, max((mount_line or 1) + 10, 50))
    return "create_app", 1, end


def normalize_match_path(file_path: str, project_root: Path | None = None) -> str:
    """Return a stable repo-relative path for match records."""
    text = str(file_path or "").replace("\\", "/").strip()
    if not text:
        return text
    path = Path(text)
    if project_root is not None:
        root = project_root.resolve()
        if path.is_absolute():
            try:
                return path.relative_to(root).as_posix()
            except ValueError:
                pass
        else:
            candidate = (root / path).resolve()
            if candidate.is_file():
                try:
                    return candidate.relative_to(root).as_posix()
                except ValueError:
                    pass
    return text.lstrip("./")


def classify_match_line(
    match_line: str,
    *,
    pattern: str = "",
    file_path: str = "",
    project_root: Path | None = None,
) -> str:
    """Classify a grep hit for ranking (definition > mount > call-site noise)."""
    line = str(match_line or "").strip()
    if not line:
        return "usage"
    profiles = _active_profiles(file_path=file_path, project_root=project_root)
    if _SCHEMA_LINE_RE.search(line):
        return "schema"
    if _DECORATOR_LINE_RE.search(line):
        return "decorator"
    classified = classify_line_with_profiles(line, pattern=pattern, profiles=profiles)
    if classified != "usage":
        return classified
    if _DEFINITION_LINE_RE.search(line):
        return "definition"
    if _MOUNT_LINE_RE.search(line):
        return "mount"
    return classified


_MATCH_KIND_PRIORITY: dict[str, int] = {
    "definition": 0,
    "schema": 1,
    "decorator": 2,
    "mount": 3,
    "import": 4,
    "usage": 5,
    "call_site": 6,
}


def rank_matches(
    matches: list[dict[str, Any]],
    *,
    searched_patterns: Sequence[str] = (),
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Sort matches so definitions and wiring hits surface before call-site noise."""
    primary_pattern = str(searched_patterns[0] or "") if searched_patterns else ""
    definition_symbols: set[str] = set()
    definition_files: set[str] = set()

    for item in matches:
        pattern = str(item.get("matched_pattern") or primary_pattern)
        file_path = normalize_match_path(str(item.get("file") or ""))
        kind = classify_match_line(
            str(item.get("match_line") or ""),
            pattern=pattern,
            file_path=file_path,
            project_root=project_root,
        )
        item["match_kind"] = kind
        symbol = str(item.get("symbol") or "").strip()
        if not symbol and kind == "call_site":
            leaf = pattern.split(".")[-1].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf):
                symbol = leaf
                item["symbol"] = symbol
        if kind == "definition":
            symbol = str(item.get("symbol") or "").strip()
            if not symbol:
                symbol = extract_symbol_from_match_line(
                    str(item.get("match_line") or ""),
                    file_path=file_path,
                    project_root=project_root,
                )
                if symbol:
                    item["symbol"] = symbol
            if symbol:
                definition_symbols.add(symbol)
            file_path = normalize_match_path(str(item.get("file") or ""))
            if file_path:
                definition_files.add(file_path)

    ranked: list[dict[str, Any]] = []
    for item in matches:
        kind = str(item.get("match_kind") or "usage")
        line = str(item.get("match_line") or "")
        symbol = str(item.get("symbol") or "").strip()
        file_path = normalize_match_path(str(item.get("file") or ""))

        if kind == "call_site" and symbol in definition_symbols and file_path not in definition_files:
            if not _MOUNT_LINE_RE.search(line):
                continue
            item["match_kind"] = "mount"

        ranked.append(item)

    def _score(item: dict[str, Any]) -> tuple[int, int, str]:
        kind = str(item.get("match_kind") or "usage")
        priority = _MATCH_KIND_PRIORITY.get(kind, 9)
        span = item.get("span") or [0, 0]
        width = int(span[1]) - int(span[0]) + 1 if isinstance(span, list) and len(span) >= 2 else 1
        return (priority, -width, normalize_match_path(str(item.get("file") or "")))

    return sorted(ranked, key=_score)


def wiring_symbol_suggestions(file_lines: Sequence[str]) -> list[str]:
    """Concrete setup symbols present in a Python module header."""
    names: list[str] = []
    for line in file_lines[:120]:
        factory = re.search(
            r"\bdef\s+(create_app|wire_routes|make_app|build_app|init_app)\s*\(",
            line,
        )
        if factory and factory.group(1) not in names:
            names.append(factory.group(1))
        assign = re.search(
            r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*(?::[^=]+)?=\s*FastAPI\s*\(",
            line,
        )
        if assign and assign.group(1) not in names:
            names.append(assign.group(1))
    return names


_TRIVIAL_ASSIGNMENT_SYMBOLS = frozenset({"logger", "app", "router"})


def is_actionable_suggested_view(view: Mapping[str, Any]) -> bool:
    """True when a grep-suggested view is worth a view_symbol_code hop (not a one-liner)."""
    symbol = str(view.get("symbol") or "").strip()
    if not symbol:
        return False
    span = view.get("span")
    if isinstance(span, list) and len(span) >= 2:
        line_count = int(span[1]) - int(span[0]) + 1
        if line_count >= 3:
            return True
        if symbol in _TRIVIAL_ASSIGNMENT_SYMBOLS and line_count <= 1:
            return False
        if view.get("resolved_from") in {"decorator_context", "mount_context"}:
            return True
    return symbol not in _TRIVIAL_ASSIGNMENT_SYMBOLS


def has_actionable_suggested_views(
    views: Sequence[Mapping[str, Any]] | None,
) -> bool:
    return any(is_actionable_suggested_view(view) for view in (views or ()))


def rank_suggested_views(
    views: list[dict[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    """Prefer handler/route defs over module-level one-line assignments."""

    def _score(view: dict[str, Any]) -> tuple[int, int]:
        symbol = str(view.get("symbol") or "")
        span = view.get("span") or [0, 0]
        line_count = int(span[1]) - int(span[0]) + 1 if len(span) >= 2 else 0
        priority = 0
        if view.get("resolved_from") in {"decorator_context", "mount_context"}:
            priority -= 100
        if view.get("resolved_from") == "repo_reference":
            priority -= 80
        if "handler" in symbol.casefold() or "exception" in symbol.casefold():
            priority -= 50
        if symbol in _TRIVIAL_ASSIGNMENT_SYMBOLS and line_count <= 1:
            priority += 50
        if line_count >= 8:
            priority -= 10
        return (priority, -line_count)

    ranked = sorted(views, key=_score)
    return ranked[:limit]


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
        if item.get("resolved_from"):
            view["resolved_from"] = item["resolved_from"]
        if item.get("reference_from"):
            view["reference_from"] = item["reference_from"]
        if item.get("match_line"):
            view["match_line"] = item["match_line"]
        views.append(view)
    return rank_suggested_views(views, limit=limit)


def _find_sql_definition_end(lines: list[str], start_index: int) -> int:
    """Locate a SQL object's end while respecting MySQL DELIMITER blocks."""
    delimiter = ";"
    for line in lines[:start_index]:
        match = re.match(r"\s*DELIMITER\s+(\S+)", line, re.IGNORECASE)
        if match:
            delimiter = match.group(1)

    if delimiter != ";":
        for index in range(start_index, len(lines)):
            if delimiter in lines[index]:
                return index + 1

    declaration = lines[start_index].casefold()
    is_programmable = any(
        f" {kind} " in f" {declaration} "
        for kind in ("procedure", "function", "trigger", "event")
    )
    if is_programmable:
        for index in range(start_index, len(lines)):
            if re.search(r"\bEND\s*;", lines[index], re.IGNORECASE):
                return index + 1

    for index in range(start_index, len(lines)):
        if ";" in lines[index]:
            return index + 1
    return len(lines)


def expand_schema_definition_spans(
    matches: list[dict[str, Any]],
    *,
    project_root: Path | None = None,
) -> list[dict[str, Any]]:
    """Expand CREATE TABLE/VIEW grep hits to full DDL block spans."""
    root = project_root or Path.cwd()
    cache: dict[str, list[str]] = {}
    for item in matches:
        if str(item.get("match_kind") or "") != "schema":
            continue
        file_path = str(item.get("file") or "").strip().lstrip("./")
        span = item.get("span")
        if not file_path or not isinstance(span, list) or len(span) < 1:
            continue
        if file_path not in cache:
            abs_path = (root / file_path).resolve()
            if not abs_path.is_file():
                continue
            try:
                cache[file_path] = abs_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
        lines = cache.get(file_path)
        if not lines:
            continue
        start_line = int(span[0])
        if start_line < 1 or start_line > len(lines):
            continue
        end_line = _find_sql_definition_end(lines, start_line - 1)
        item["span"] = [start_line, end_line]
        if not item.get("symbol"):
            symbol = extract_symbol_from_match_line(lines[start_line - 1])
            if symbol:
                item["symbol"] = symbol
    return matches


def reference_views_from_repo_map(
    matches: list[dict[str, Any]],
    repo_map: Any,
    *,
    limit: int = 6,
) -> list[dict[str, Any]]:
    """Add cross-file caller/callee views using repo_map reference edges."""
    if repo_map is None:
        return []
    edges = list(getattr(repo_map, "reference_edges", ()) or ())
    symbols_by_file = getattr(repo_map, "symbols_by_file", {})
    symbols_by_id = getattr(repo_map, "symbols_by_id", {})
    if not edges or not symbols_by_file:
        return []

    views: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()

    def _sym_to_view(sym: Any, *, reference_from: str, resolved_from: str) -> dict[str, Any] | None:
        file_path = normalize_match_path(str(getattr(sym, "file_path", "") or ""))
        name = str(getattr(sym, "name", "") or "")
        if not file_path or not name:
            return None
        key = (file_path, name)
        if key in seen:
            return None
        seen.add(key)
        return {
            "file": file_path,
            "symbol": name,
            "span": [
                int(getattr(sym, "start_line", 1) or 1),
                int(getattr(sym, "end_line", getattr(sym, "start_line", 1)) or 1),
            ],
            "resolved_from": resolved_from,
            "reference_from": reference_from,
        }

    for item in matches:
        if str(item.get("match_kind") or "") not in {"definition", "schema"}:
            continue
        symbol = str(item.get("symbol") or "").strip()
        file_path = normalize_match_path(str(item.get("file") or ""))
        if not symbol or not file_path:
            continue
        symbol_ids: list[str] = []
        for sym in symbols_by_file.get(file_path, []):
            if sym.name == symbol:
                symbol_ids.append(str(sym.symbol_id))
        for sid in symbol_ids:
            for src_id, dst_id in edges:
                other_id = None
                relation = ""
                if dst_id == sid:
                    other_id = src_id
                    relation = "referenced_by"
                elif src_id == sid:
                    other_id = dst_id
                    relation = "references"
                if other_id is None:
                    continue
                other = symbols_by_id.get(other_id)
                if other is None:
                    continue
                other_file = normalize_match_path(str(getattr(other, "file_path", "") or ""))
                if other_file == file_path:
                    continue
                view = _sym_to_view(
                    other,
                    reference_from=f"{file_path}::{symbol}",
                    resolved_from="repo_reference",
                )
                if view is None:
                    continue
                view["reference_relation"] = relation
                views.append(view)
                if len(views) >= limit:
                    return views
    return views


def merge_suggested_views(
    primary: Sequence[Mapping[str, Any]],
    extra: Sequence[Mapping[str, Any]],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = [dict(item) for item in primary]
    seen = {(str(v.get("file") or ""), str(v.get("symbol") or "")) for v in merged}
    for item in extra:
        view = dict(item)
        key = (str(view.get("file") or ""), str(view.get("symbol") or ""))
        if key in seen:
            continue
        seen.add(key)
        merged.append(view)
    return rank_suggested_views(merged, limit=limit)


def parse_rg_hit_line(
    line: str,
    matched_pattern: str,
    *,
    project_root: Path | None = None,
) -> dict[str, Any] | None:
    parts = line.split(":", 2)
    if len(parts) < 3:
        return None
    file_path = normalize_match_path(parts[0], project_root)
    try:
        line_num = int(parts[1])
    except ValueError:
        return None
    content = parts[2]
    symbol_name = extract_symbol_from_match_line(
        content,
        file_path=file_path,
        project_root=project_root,
    )
    return {
        "file": file_path,
        "symbol": symbol_name,
        "span": [line_num, line_num],
        "match_line": content.strip(),
        "matched_pattern": matched_pattern,
    }


def enrich_grep_matches(
    matches: list[dict[str, Any]],
    *,
    searched_patterns: Sequence[str] = (),
    project_root: Path | None = None,
    repo_map: Any = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run ranking, span expansion, wiring resolution, and suggested view assembly."""
    if not matches:
        return [], []
    matches = rank_matches(matches, searched_patterns=searched_patterns, project_root=project_root)
    matches = expand_schema_definition_spans(matches, project_root=project_root)
    matches = fill_symbols_from_adjacent_lines(matches)
    matches = resolve_decorator_symbols_from_files(matches, project_root=project_root)
    matches = resolve_mount_line_symbols(matches, project_root=project_root)
    suggested = suggested_views_from_matches(matches)
    ref_views = reference_views_from_repo_map(matches, repo_map)
    if ref_views:
        suggested = merge_suggested_views(suggested, ref_views)
    return matches, suggested


def grep_search_fingerprint(
    patterns: Sequence[str],
    *,
    path: str = ".",
    include: str | None = None,
    mode: str = "default",
) -> str:
    """Stable key for empty-result deduplication."""
    norm_path = str(path or ".").replace("\\", "/").strip().rstrip("/") or "."
    norm_include = str(include or "").strip()
    norm_mode = str(mode or "default").strip()
    norm_patterns = "|".join(sorted(str(p).strip() for p in patterns if str(p).strip()))
    return f"{norm_path}::{norm_include}::{norm_mode}::{norm_patterns}"


def history_entry_matches_fingerprint(entry: Mapping[str, Any], fingerprint: str) -> bool:
    if str(entry.get("fingerprint") or "") == fingerprint:
        return True
    entry_patterns = entry.get("patterns")
    if isinstance(entry_patterns, list):
        entry_fp = grep_search_fingerprint(
            entry_patterns,
            path=str(entry.get("path") or "."),
            include=entry.get("include"),
            mode=str(entry.get("mode") or "default"),
        )
        return entry_fp == fingerprint
    pattern = str(entry.get("pattern") or "").strip()
    if not pattern:
        return False
    entry_fp = grep_search_fingerprint(
        [pattern],
        path=str(entry.get("path") or "."),
        include=entry.get("include"),
        mode=str(entry.get("mode") or "default"),
    )
    return entry_fp == fingerprint
