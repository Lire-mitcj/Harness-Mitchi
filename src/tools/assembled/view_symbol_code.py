from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool
from src.indexer.language_profiles import (
    profile_for_path,
    searchable_extensions,
)
from src.indexer.project_stack import detect_project_stack
from src.tools.grep_match_symbols import wiring_symbol_suggestions

log = logging.getLogger(__name__)

MAX_SYMBOL_OUTPUT_LINES = 120
MAX_SYMBOL_OUTPUT_CHARS = 8_000
_SKIPPED_SEARCH_DIRS = {".git", ".venv", "venv", "node_modules", "dist", "build"}
_SQL_DEFINITION_KINDS = "TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER|EVENT"
_WIRING_PROBE_SYMBOLS = frozenset({"app", "application", "router", "logger", "create_app"})


def _captured_symbol_matches(
    pattern: re.Pattern[str],
    line: str,
    symbol_name: str,
) -> bool:
    """True when a profile regex match names the requested symbol."""
    leaf = symbol_name.split(".")[-1]
    match = pattern.search(line)
    if match is None:
        return False
    if match.lastindex and match.lastindex >= 1:
        captured = match.group(1)
        if captured is not None:
            return captured == leaf
    return bool(re.search(r"\b" + re.escape(leaf) + r"\b", line))


def _find_symbol_span_in_python_file(content: str, symbol_name: str) -> tuple[int, int] | None:
    import ast
    try:
        tree = ast.parse(content)

        # Build parent map to construct full dot-paths
        parent_map = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        def get_full_name(node: Any) -> str:
            name_parts = []
            curr = node
            while curr is not None:
                if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name_parts.append(curr.name)
                curr = parent_map.get(curr)
            return ".".join(reversed(name_parts))

        # First pass: try matching Function, AsyncFunction, and Class (including nested)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                full_name = get_full_name(node)
                if node.name == symbol_name or full_name == symbol_name:
                    decorator_lines = [
                        getattr(decorator, "lineno", node.lineno)
                        for decorator in getattr(node, "decorator_list", [])
                    ]
                    start = min([getattr(node, "lineno", 1), *decorator_lines])
                    end = getattr(node, "end_lineno", start)
                    return start, end

        # Second pass: module-level assignments (e.g. app = FastAPI()).
        # Nested assignments inside create_app() must not satisfy a module-level probe.
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol_name:
                        start = getattr(node, "lineno", 1)
                        end = getattr(node, "end_lineno", start)
                        return start, end
            if isinstance(node, ast.AnnAssign):
                target = node.target
                if isinstance(target, ast.Name) and target.id == symbol_name:
                    start = getattr(node, "lineno", 1)
                    end = getattr(node, "end_lineno", start)
                    return start, end
    except Exception as exc:
        log.warning("view_symbol_code AST parsing failed: %s", exc)
    return None


def _find_symbol_span_by_regex(
    content: str,
    symbol_name: str,
    *,
    file_path: str = "",
) -> tuple[int, int] | None:
    lines = content.splitlines()
    last_part = symbol_name.split(".")[-1]
    profile = profile_for_path(file_path)

    block_patterns: list[re.Pattern[str]] = []
    if profile is not None:
        block_patterns.extend(profile.block_start_res)
    else:
        block_patterns.append(
            re.compile(
                r"\b(?:def|class|function|func|type|interface|enum)\s+"
                + re.escape(last_part)
                + r"\b"
            )
        )

    for idx, line in enumerate(lines):
        matched = False
        for pattern in block_patterns:
            if _captured_symbol_matches(pattern, line, symbol_name):
                matched = True
                break
        assign_match = re.search(r'\b' + re.escape(last_part) + r'\s*=\s*', line)
        # 3. matches SQL view/table definitions with optional backticks/quotes
        escaped_name = re.escape(last_part)
        sql_match = re.search(
            r'\bCREATE\s+(?:OR\s+REPLACE\s+)?'
            r'(?:DEFINER\s*=\s*\S+\s+)?(?:TEMP\s+|TEMPORARY\s+)?'
            rf'(?:{_SQL_DEFINITION_KINDS})\s+(?:IF\s+NOT\s+EXISTS\s+)?'
            rf'(?:`{escaped_name}`|"{escaped_name}"|\'{escaped_name}\'|\b{escaped_name}\b)',
            line,
            re.IGNORECASE
        )

        if matched or assign_match or sql_match:
            start_line = idx + 1
            if sql_match:
                return start_line, _find_sql_definition_end(lines, idx)

            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ""
            # Go/Java/Proto blocks often use braces — scan for closing brace at base indent
            if "{" in line and profile is not None and profile.id in {"go", "java", "proto"}:
                depth = line.count("{") - line.count("}")
                end_line = start_line
                for j in range(idx + 1, len(lines)):
                    depth += lines[j].count("{") - lines[j].count("}")
                    end_line = j + 1
                    if depth <= 0:
                        break
                return start_line, end_line

            end_line = start_line
            for j in range(idx + 1, len(lines)):
                if not lines[j].strip():
                    continue
                curr_indent_match = re.match(r'^(\s*)', lines[j])
                curr_indent = curr_indent_match.group(1) if curr_indent_match else ""
                if len(curr_indent) <= len(indent) and lines[j].strip():
                    end_line = j
                    break
                end_line = j + 1
            return start_line, end_line
    return None


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


def _find_symbol_definitions(
    project_root: Path,
    symbol_name: str,
    *,
    exclude_file: str,
) -> list[tuple[str, str, tuple[int, int]]]:
    """Find project-local definitions for a symbol after a file mismatch."""
    candidates: list[tuple[str, str, tuple[int, int]]] = []
    leaf_name = symbol_name.split(".")[-1]
    stack = detect_project_stack(project_root)
    # Union (not fallback) with the common definition-bearing suffixes: a project
    # detected as python-only still holds symbols in .sql schema files, .proto
    # contracts, etc. Definition lookup is a miss-only fallback, so searching the
    # extra suffixes is cheap and restores cross-language resolution.
    extensions = searchable_extensions(stack.languages) | frozenset(
        {".py", ".sql", ".go", ".java", ".proto"}
    )

    py_declaration = re.compile(
        rf"\b(?:async\s+def|def|class)\s+{re.escape(leaf_name)}\b"
    )
    go_declaration = re.compile(
        rf"\b(?:func|type)\s+(?:\([^)]*\)\s+)?{re.escape(leaf_name)}\b"
    )
    java_declaration = re.compile(
        rf"\b(?:class|interface|enum)\s+{re.escape(leaf_name)}\b"
    )
    proto_declaration = re.compile(
        rf"\b(?:service|message|enum|rpc)\s+{re.escape(leaf_name)}\b"
    )
    
    # SQL view/table declaration pattern
    escaped_name = re.escape(leaf_name)
    sql_declaration = re.compile(
        r'\bCREATE\s+(?:OR\s+REPLACE\s+)?'
        r'(?:DEFINER\s*=\s*\S+\s+)?(?:TEMP\s+|TEMPORARY\s+)?'
        rf'(?:{_SQL_DEFINITION_KINDS})\s+(?:IF\s+NOT\s+EXISTS\s+)?'
        rf'(?:`{escaped_name}`|"{escaped_name}"|\'{escaped_name}\'|\b{escaped_name}\b)',
        re.IGNORECASE
    )

    for path in project_root.rglob("*"):
        if path.is_dir() or any(part in _SKIPPED_SEARCH_DIRS for part in path.parts):
            continue
        relative = path.relative_to(project_root).as_posix()
        if relative == exclude_file:
            continue
            
        if path.suffix not in extensions:
            continue

        try:
            candidate_content = path.read_text(encoding="utf-8")
        except OSError:
            continue

        matched = False
        if path.suffix == ".py" and py_declaration.search(candidate_content):
            matched = True
        elif path.suffix == ".go" and go_declaration.search(candidate_content):
            matched = True
        elif path.suffix == ".java" and java_declaration.search(candidate_content):
            matched = True
        elif path.suffix == ".proto" and proto_declaration.search(candidate_content):
            matched = True
        elif path.suffix == ".sql" and sql_declaration.search(candidate_content):
            matched = True
        if not matched:
            continue

        span = None
        if path.suffix == ".py":
            span = _find_symbol_span_in_python_file(candidate_content, symbol_name)
        if span is None:
            span = _find_symbol_span_by_regex(
                candidate_content,
                symbol_name,
                file_path=relative,
            )
        if span is not None:
            candidates.append((relative, candidate_content, span))

    return candidates


def _find_caller_setup_fallback(
    content: str,
    requested_symbol: str,
) -> tuple[tuple[int, int], str] | None:
    """Resolve generic wiring probes (app/create_app) to real setup spans."""
    lines = content.splitlines()
    suggestions = wiring_symbol_suggestions(lines)
    probe_symbols = {requested_symbol, *suggestions}
    for candidate in suggestions:
        if candidate in probe_symbols or requested_symbol in _WIRING_PROBE_SYMBOLS:
            span = _find_symbol_span_in_python_file(content, candidate)
            if span is None:
                span = _find_symbol_span_by_regex(content, candidate)
            if span is not None:
                return span, candidate
    if requested_symbol in _WIRING_PROBE_SYMBOLS and lines:
        end = min(len(lines), 80)
        if end >= 3:
            return (1, end), requested_symbol
    return None


def _format_missing_symbol_error(
    requested_file: str,
    symbol: str,
    content: str,
) -> str:
    suggestions = wiring_symbol_suggestions(content.splitlines())
    base = (
        f"Symbol '{symbol}' not found in {requested_file} "
        "via AST, regex, or retrieval cache."
    )
    if not suggestions:
        return (
            f"{base} Try grep_search(pattern='include_router|create_app|FastAPI\\('', "
            f"path={requested_file!r}) or view_symbol_code with symbol='*' for the module header."
        )
    alts = ", ".join(f"{requested_file}::{name}" for name in suggestions[:3])
    return f"{base} Symbols present in file: {alts}."


class ViewSymbolCodeTool(Tool):
    name = "view_symbol_code"
    description = (
        "Primary load tool: fetch the verbatim source slice of one known symbol "
        "(function/class) or an identified DDL block from a specific file. Use after "
        "grep_search or STEP EVIDENCE names a concrete file+symbol that is missing or "
        "stale. Provides durable code anchors for reasoning and edit context_window "
        "spans. Do NOT use for fuzzy discovery or whole-file overview. Do NOT re-fetch "
        "symbols already listed under loaded in STEP EVIDENCE or LOADED CODE ANCHORS."
    )
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "target_file": {
                "type": "string",
                "description": "Relative path to the target file."
            },
            "symbol": {
                "type": "string",
                "description": "Name of the symbol (class or function) to inspect."
            }
        },
        "required": ["target_file", "symbol"]
    }

    def __init__(
        self,
        *,
        project_root: Path,
        settings: Any,
    ) -> None:
        self.project_root = project_root.resolve()
        self.settings = settings

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        target_file = validated["target_file"]
        requested_file = target_file
        symbol = validated["symbol"]
        search_cache = params.get("_search_cache") or {}

        # 1. Try to read from disk
        abs_path = (self.project_root / target_file).resolve()
        content = ""
        if abs_path.is_file():
            try:
                content = abs_path.read_text(encoding="utf-8")
            except OSError as exc:
                log.warning("view_symbol_code: failed to read %s: %s", target_file, exc)

        # 2. Try to get content from raw_evidence_store if disk is empty
        if not content and search_cache:
            raw_store = search_cache.get("raw_evidence_store") or []
            parts = []
            for item in raw_store:
                if item.get("file") == target_file:
                    parts.append(item)
            if parts:
                parts.sort(key=lambda x: x.get("span", [0, 0])[0])
                content = "\n".join(p.get("code", "") for p in parts)

        if not content:
            return ToolResult(
                success=False,
                output="",
                error=f"File {target_file} not found on disk and not present in retrieval cache."
            )

        # 3. Locate the symbol span
        from src.hooks.tool_gate import normalize_view_symbol

        span = None
        basename = Path(target_file).name
        stem = Path(target_file).stem
        symbol = normalize_view_symbol(symbol)
        if symbol.lower() in ("*", "all", "file"):
            span = (1, len(content.splitlines()))
        elif symbol == basename or symbol == stem:
            return ToolResult(
                success=False,
                output="",
                error=(
                    f"Symbol '{symbol}' is the filename, not a code symbol. "
                    "Use grep_search to locate a CREATE TABLE / def / @router target, "
                    "then view_symbol_code with that concrete symbol name."
                ),
            )

        if not span and target_file.endswith(".py"):
            span = _find_symbol_span_in_python_file(content, symbol)

        resolved_symbol = symbol
        if not span and symbol in _WIRING_PROBE_SYMBOLS:
            fallback = _find_caller_setup_fallback(content, symbol)
            if fallback is not None:
                span, resolved_symbol = fallback
                symbol = resolved_symbol

        if not span:
            span = _find_symbol_span_by_regex(content, symbol, file_path=target_file)

        if not span and search_cache:
            search_output_str = search_cache.get("search_output")
            if search_output_str:
                try:
                    search_output = json.loads(search_output_str)
                    evidence = search_output.get("evidence") or []
                    for item in evidence:
                        if item.get("symbol") == symbol and item.get("file") == target_file:
                            span = item.get("span")
                            break
                    if not span:
                        grounding = search_output.get("grounding") or {}
                        for item in grounding.get("symbols", []):
                            if item.get("name") == symbol and item.get("file") == target_file:
                                span = item.get("span")
                                break
                except Exception:
                    pass

        if not span:
            candidates = _find_symbol_definitions(
                self.project_root,
                symbol,
                exclude_file=target_file,
            )
            if len(candidates) == 1:
                target_file, content, span = candidates[0]
            elif candidates:
                candidate_files = ", ".join(item[0] for item in candidates)
                return ToolResult(
                    success=False,
                    output="",
                    error=(
                        f"Symbol '{symbol}' is not defined in {requested_file}. "
                        f"Multiple definitions found in: {candidate_files}."
                    ),
                    metadata={"symbol_candidates": [item[0] for item in candidates]},
                )

        if not span:
            fallback = _find_caller_setup_fallback(content, symbol)
            if fallback is not None:
                span, resolved_symbol = fallback
                symbol = resolved_symbol

        if not span:
            return ToolResult(
                success=False,
                output="",
                error=_format_missing_symbol_error(requested_file, symbol, content),
                metadata={
                    "wiring_symbol_suggestions": wiring_symbol_suggestions(
                        content.splitlines()
                    ),
                },
            )

        # 4. Extract verbatim slice
        lines = content.splitlines()
        start, end = span
        start = max(1, start)
        end = min(len(lines), end)
        if start > end:
            return ToolResult(
                success=False,
                output="",
                error=f"Invalid span [{start}, {end}] located for symbol '{symbol}'."
            )

        slice_lines = lines[start - 1:end]
        projection_lines: list[str] = []
        projection_chars = 0
        for i, line in enumerate(slice_lines[:MAX_SYMBOL_OUTPUT_LINES]):
            formatted = f"{start + i}| {line}"
            next_chars = projection_chars + len(formatted) + (1 if projection_lines else 0)
            if next_chars > MAX_SYMBOL_OUTPUT_CHARS:
                break
            projection_lines.append(formatted)
            projection_chars = next_chars
        output_code_slice = "\n".join(projection_lines)
        verbatim_code = "\n".join(slice_lines)
        code_hash = hashlib.md5(verbatim_code.encode("utf-8")).hexdigest()[:8]
        projection_truncated = len(projection_lines) < len(slice_lines)

        output_data = {
            "file": target_file,
            "span": [start, end],
            "observation_code": output_code_slice,
            "verbatim_code": verbatim_code,
        }
        output_json = json.dumps(output_data, ensure_ascii=False, indent=2)

        metadata = {
            "llm_observation": output_code_slice,
            "file": target_file,
            "symbol": symbol,
            "requested_file": requested_file,
            "resolved_file": target_file,
            "span": [start, end],
            "verbatim_code": verbatim_code,
            "projection_truncated": projection_truncated,
            "raw_evidence_store": [
                {
                    "file": target_file,
                    "span": [start, end],
                    "code": verbatim_code,
                    "symbol": symbol,
                    "related_functions": [],
                    "hash": code_hash,
                }
            ],
        }

        return ToolResult(
            success=True,
            output=output_json,
            metadata=metadata
        )
