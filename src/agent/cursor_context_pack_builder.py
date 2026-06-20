from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from src.agent.cursor_contracts import (
    ContextPack,
    ContextWindow,
    RetrievalResult,
    RetrievalSymbol,
    SemanticAnnotations,
)


class CursorContextPackBuilder:
    """Compile retrieval output into a three-layer, evidence-labelled context IR."""

    def __init__(
        self,
        project_root: Path,
        *,
        max_files: int = 3,
        max_chars_per_file: int = 12_000,
        padding_lines: int = 20,
        semantic_padding_lines: int = 50,
        dependency_affinity_threshold: float = 0.65,
    ) -> None:
        self.project_root = project_root.resolve()
        self.max_files = max_files
        self.max_chars_per_file = max_chars_per_file
        self.padding_lines = padding_lines
        self.semantic_padding_lines = semantic_padding_lines
        self.dependency_affinity_threshold = dependency_affinity_threshold

    def adjust_budget(self, intent: str | None) -> None:
        if intent == "explain":
            self.max_chars_per_file = 64 * 1024
        else:
            self.max_chars_per_file = 16 * 1024

    def build_context(
        self,
        result: RetrievalResult,
        annotations: SemanticAnnotations | None = None,
        final_context: tuple[str, ...] | None = None,
    ) -> ContextPack:
        annotations = annotations or SemanticAnnotations(tags_by_file={})
        routes = final_context or _routes_from_result(result)
        return self._build_coordinate_context(result, annotations, routes)

    def _build_coordinate_context(
        self,
        result: RetrievalResult,
        annotations: SemanticAnnotations,
        final_context: tuple[str, ...],
    ) -> ContextPack:
        symbol_index = {
            (symbol.file, symbol.name, symbol.start_line, symbol.end_line): symbol
            for symbol in result.symbols
        }
        routes = tuple(
            route for item in final_context
            if (route := _parse_context_route(item)) is not None
        )
        core_routes = _select_core_routes(routes, symbol_index, result.symbols)
        # Retrieval can occasionally provide only file paths or DDL.  Keep a
        # bounded fallback rather than forcing Decision into ask_clarify, but
        # do not manufacture an arbitrary header/line slice.
        if not core_routes:
            for route in routes:
                if route.is_file:
                    path = self._safe_file(route.file)
                    if path is None:
                        continue
                    line_count = len(
                        path.read_text(encoding="utf-8", errors="replace").splitlines()
                    )
                    core_routes.append((
                        _ContextRoute(
                            route.file, "FILE_FALLBACK", 1, max(1, line_count), False, False,
                        ),
                        None,
                    ))
                else:
                    core_routes.append((
                        route,
                        symbol_index.get(
                            (route.file, route.name, route.start_line, route.end_line)
                        ),
                    ))
                if len(core_routes) >= self.max_files:
                    break
        windows: list[ContextWindow] = []
        for route, core_symbol in core_routes[: self.max_files]:
            path = self._safe_file(route.file)
            if path is None:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if not lines:
                continue
            start = max(1, min(route.start_line, len(lines)))
            end = max(start, min(route.end_line, len(lines)))
            core = _numbered_lines(lines, start, end)
            header = _semantic_header(path, lines, start, end)
            skeleton = _global_skeleton(path, lines)
            soft = self._soft_dependencies(
                core_symbol, result.symbols, lines[start - 1:end], symbol_index,
            )
            content = "\n\n".join(part for part in (
                "[LAYER_1_CORE_SYMBOL]\n"
                f"file: {route.file}\nlines: {start}-{end}\n"
                f"symbol: {route.name}\n{core}",
                "[LAYER_2_DEPENDENCY_EVIDENCE]\n"
                + (header or "[SEMANTIC_HEADER] none")
                + ("\n\n" + soft if soft else ""),
                "[LAYER_3_GLOBAL_SKELETON]\n" + skeleton,
            ) if part)
            content = _clip(content, self.max_chars_per_file)
            windows.append(ContextWindow(
                file=route.file,
                start_line=start,
                end_line=end,
                content=content,
                symbols=(route.name,),
                semantic_tags=annotations.tags_by_file.get(route.file, ()),
            ))
        return ContextPack(windows=tuple(windows))

    def _soft_dependencies(
        self,
        core: RetrievalSymbol | None,
        symbols: tuple[RetrievalSymbol, ...],
        core_lines: list[str],
        symbol_index: dict[tuple[str, str, int, int], RetrievalSymbol],
    ) -> str:
        if core is None:
            return ""
        core_text = "\n".join(core_lines)
        core_tables = set(core.tables_referenced) | _sql_tables(core_text)
        candidates: list[tuple[float, RetrievalSymbol]] = []
        for candidate in symbols:
            if candidate == core or not candidate.kind.startswith("ddl_"):
                continue
            score = _ddl_affinity(core_tables, core_text, candidate)
            if score >= self.dependency_affinity_threshold:
                candidates.append((score, candidate))
        candidates.sort(key=lambda item: (-item[0], item[1].file, item[1].start_line))
        evidence: list[str] = []
        for score, candidate in candidates[:2]:
            path = self._safe_file(candidate.file)
            if path is None:
                continue
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            body = _numbered_lines(lines, candidate.start_line, candidate.end_line)
            evidence.append(
                "[SOFT_DEPENDENCY]\n"
                f"kind: ddl_affinity\naffinity_score: {score:.2f}\n"
                f"file: {candidate.file}\nlines: {candidate.start_line}-{candidate.end_line}\n"
                f"symbol: {candidate.name}\n{body}"
            )
        return "\n\n".join(evidence)

    def _file_placeholder_window(
        self,
        route: _ContextRoute,
        lines: list[str],
        annotations: SemanticAnnotations,
    ) -> ContextWindow:
        start = 1
        end = min(len(lines), max(1, self.padding_lines * 2))
        content = "\n".join(
            f"{index}: {lines[index - 1]}" for index in range(start, end + 1)
        )
        if len(content) > self.max_chars_per_file:
            content = content[: self.max_chars_per_file] + "\n...[truncated]"
        return ContextWindow(
            file=route.file,
            start_line=start,
            end_line=end,
            content=content,
            semantic_tags=annotations.tags_by_file.get(route.file, ()),
        )

    def _semantic_window(
        self,
        route: _ContextRoute,
        lines: list[str],
        symbols: list[RetrievalSymbol],
        all_symbols: tuple[RetrievalSymbol, ...],
        annotations: SemanticAnnotations,
    ) -> ContextWindow:
        anchor = symbols[0] if symbols else None
        neighborhood = _expand_ast_neighborhood(anchor, all_symbols)
        block_symbols = tuple(dict.fromkeys((
            *(symbols or ()),
            *(
                dep
                for dep in neighborhood.all_symbols
                if dep.file == route.file
            ),
        )))
        start, end = self._merged_semantic_range(route, lines, block_symbols)
        content_limit = self.max_chars_per_file

        body = "\n".join(
            f"{index}: {lines[index - 1]}" for index in range(start, end + 1)
        )
        if route.file.casefold().endswith(".sql") and (start > 1 or end < len(lines)):
            collapsed = "-- [Context Collapsed]"
            if start > 1:
                body = f"1: {collapsed}\n{body}"
            if end < len(lines):
                body = f"{body}\n{len(lines)}: {collapsed}"

        header = _semantic_block_header(
            route,
            anchor,
            block_symbols,
            neighborhood,
            all_symbols,
        )
        content = f"{header}\n{body}" if header else body
        if route.evidence:
            content = f"{route.evidence}\n{content}"
        if len(content) > content_limit:
            suffix = "\n...[truncated]"
            content = content[: max(0, content_limit - len(suffix))] + suffix
        return ContextWindow(
            file=route.file,
            start_line=start,
            end_line=end,
            content=content,
            symbols=tuple(item.name for item in block_symbols) or (route.name,),
            semantic_tags=annotations.tags_by_file.get(route.file, ()),
        )

    def _merged_semantic_range(
        self,
        route: _ContextRoute,
        lines: list[str],
        symbols: tuple[RetrievalSymbol, ...],
    ) -> tuple[int, int]:
        anchor_padding = max(self.padding_lines, self.semantic_padding_lines)
        windows = [(
            max(1, route.start_line - anchor_padding),
            min(len(lines), route.end_line + anchor_padding),
        )]
        for symbol in symbols:
            if symbol.file != route.file:
                continue
            windows.append((
                max(1, symbol.start_line - self.padding_lines),
                min(len(lines), symbol.end_line + self.padding_lines),
            ))
        return _merge_line_ranges(windows)

    def _skeleton_window(
        self,
        route: _ContextRoute,
        lines: list[str],
        symbols: list[RetrievalSymbol],
        annotations: SemanticAnnotations,
    ) -> ContextWindow:
        start = min(max(1, route.start_line), len(lines))
        signature = lines[start - 1]
        indentation = len(signature) - len(signature.lstrip())
        prefix = _comment_prefix(route.file)
        folded = (
            " " * indentation
            + signature.lstrip()
            + f" {prefix} ... [code body folded] ..."
        )
        body = f"{start}: {folded}"
        header = _window_header(symbols)
        content = f"{header}\n{body}" if header else body
        if len(content) > self.max_chars_per_file:
            content = content[: self.max_chars_per_file] + "\n...[truncated]"
        return ContextWindow(
            file=route.file,
            start_line=start,
            end_line=start,
            content=content,
            symbols=tuple(item.name for item in symbols) or (route.name,),
            semantic_tags=annotations.tags_by_file.get(route.file, ()),
        )

    def with_annotations(
        self,
        pack: ContextPack,
        annotations: SemanticAnnotations,
    ) -> ContextPack:
        return ContextPack(windows=tuple(
            ContextWindow(
                file=window.file,
                start_line=window.start_line,
                end_line=window.end_line,
                content=window.content,
                symbols=window.symbols,
                semantic_tags=annotations.tags_by_file.get(window.file, ()),
            )
            for window in pack.windows
        ))

    def _safe_file(self, file_rel: str) -> Path | None:
        try:
            path = (self.project_root / file_rel).resolve()
            path.relative_to(self.project_root)
        except (OSError, ValueError):
            return None
        return path if path.is_file() else None


def _window_header(symbols: list[RetrievalSymbol]) -> str:
    if not symbols:
        return ""
    functions = ", ".join(symbol.name for symbol in symbols)
    calls = tuple(dict.fromkeys(call for symbol in symbols for call in symbol.calls))
    tables = tuple(
        dict.fromkeys(
            table
            for symbol in symbols
            for table in symbol.tables_referenced
        )
    )
    lines = [f"[functions] {functions}"]
    dependencies = tuple(dict.fromkeys((*calls, *tables)))
    if dependencies:
        lines.append(f"[dependencies] {', '.join(dependencies)}")
        graph: list[str] = []
        for symbol in symbols:
            graph.extend(f"{symbol.name} -> {call}" for call in symbol.calls)
            graph.extend(
                f"{symbol.name} -> table:{table}"
                for table in symbol.tables_referenced
            )
        if graph:
            lines.append(f"[call_graph] {'; '.join(graph)}")
    return "\n".join(lines)


def _semantic_block_header(
    route: _ContextRoute,
    anchor: RetrievalSymbol | None,
    block_symbols: tuple[RetrievalSymbol, ...],
    neighborhood: _SymbolNeighborhood,
    all_symbols: tuple[RetrievalSymbol, ...],
) -> str:
    focus_mode = (
        "highlight_with_context_expansion_and_dependency_trace"
        if route.exclusive
        else "context_expansion_with_dependency_trace"
    )
    role = _classify_role(anchor, route)
    schema = _bind_schema(anchor)
    relation = _schema_relation(anchor, all_symbols)
    lines = [
        "SEMANTIC BLOCK:",
        f"anchor: {route.name}",
        f"file: {route.file}",
        f"role: {role}",
        f"focus_mode: {focus_mode}",
        f"confidence: {getattr(anchor, 'score', 0.0):.4f}" if anchor else "confidence: 0.0000",
    ]
    if schema:
        lines.append("schema_refs:")
        for key, value in schema.items():
            lines.append(f"  {key}: {value}")
    deps = {
        "parents": neighborhood.parents,
        "children": neighborhood.children,
        "siblings": neighborhood.siblings,
    }
    lines.append("dependencies:")
    for label, values in deps.items():
        rendered = ", ".join(_symbol_ref(symbol) for symbol in values) or "-"
        lines.append(f"  {label}: {rendered}")
    if block_symbols:
        lines.append(
            "code_window_symbols: "
            + ", ".join(_symbol_ref(symbol) for symbol in block_symbols)
        )
    if relation:
        lines.append("relation:")
        for item in relation:
            lines.append(f"  - {item}")
    legacy = _window_header(list(block_symbols))
    if legacy:
        lines.append(legacy)
    return "\n".join(lines)


def _expand_ast_neighborhood(
    symbol: RetrievalSymbol | None,
    all_symbols: tuple[RetrievalSymbol, ...],
) -> _SymbolNeighborhood:
    if symbol is None:
        return _SymbolNeighborhood()
    target_names = _symbol_aliases(symbol)
    parents = tuple(
        candidate
        for candidate in all_symbols
        if candidate != symbol
        and any(call.casefold() in target_names for call in candidate.calls)
    )
    child_names = {call.casefold() for call in symbol.calls}
    children = tuple(
        candidate
        for candidate in all_symbols
        if candidate != symbol
        and _symbol_aliases(candidate) & child_names
    )
    same_file = sorted(
        (candidate for candidate in all_symbols if candidate.file == symbol.file),
        key=lambda item: item.start_line,
    )
    siblings: list[RetrievalSymbol] = []
    for index, candidate in enumerate(same_file):
        if candidate != symbol:
            continue
        if index > 0:
            siblings.append(same_file[index - 1])
        if index + 1 < len(same_file):
            siblings.append(same_file[index + 1])
        break
    return _SymbolNeighborhood(
        parents=tuple(dict.fromkeys(parents[:4])),
        children=tuple(dict.fromkeys(children[:4])),
        siblings=tuple(dict.fromkeys(siblings[:4])),
    )


def _bind_schema(symbol: RetrievalSymbol | None) -> dict[str, str]:
    if symbol is None:
        return {}
    tables = ", ".join(symbol.tables_referenced) or "-"
    views = ", ".join(
        table for table in symbol.tables_referenced if _is_view_name(table)
    ) or "-"
    return {
        "tables": tables,
        "views": views,
        "is_view": str(symbol.kind == "ddl_view").lower(),
    }


def _classify_role(symbol: RetrievalSymbol | None, route: _ContextRoute) -> str:
    kind = (symbol.kind if symbol else "").casefold()
    name = route.name.casefold()
    if kind == "ddl_view" or _is_view_name(name):
        return "sql_view"
    if kind.startswith("dml_") or "sql" in name or "query" in name:
        return "dml_query"
    if kind in {"class", "interface"}:
        return "type_definition"
    if "test" in route.file.casefold() or name.startswith("test_"):
        return "test"
    return kind or "code_symbol"


def _schema_relation(
    anchor: RetrievalSymbol | None,
    all_symbols: tuple[RetrievalSymbol, ...],
) -> tuple[str, ...]:
    if anchor is None or not anchor.tables_referenced:
        return ()
    anchor_tables = {table.casefold() for table in anchor.tables_referenced}
    relations: list[str] = []
    for candidate in all_symbols:
        if candidate == anchor or not candidate.tables_referenced:
            continue
        shared = sorted(anchor_tables & {table.casefold() for table in candidate.tables_referenced})
        if not shared:
            continue
        if candidate.kind == "ddl_view" and anchor.kind.startswith("dml_"):
            label = "view may replace query join chain"
        elif anchor.kind == "ddl_view" and candidate.kind.startswith("dml_"):
            label = "view supports query source replacement"
        else:
            label = "schema overlap"
        relations.append(
            f"{label}: {anchor.name} <-> {candidate.name}; overlap={', '.join(shared)}"
        )
    return tuple(relations[:4])


def _symbol_ref(symbol: RetrievalSymbol) -> str:
    return f"{symbol.file}:{symbol.name}"


def _symbol_aliases(symbol: RetrievalSymbol) -> set[str]:
    name = symbol.name.casefold()
    aliases = {name, name.split(":")[0]}
    aliases.add(name.rsplit(".", 1)[-1])
    return aliases


def _is_view_name(value: str) -> bool:
    lowered = value.casefold()
    return "view" in lowered or lowered.startswith("v_")


def _merge_line_ranges(ranges: list[tuple[int, int]]) -> tuple[int, int]:
    if not ranges:
        return 1, 1
    start = min(item[0] for item in ranges)
    end = max(item[1] for item in ranges)
    return start, max(start, end)


def _global_anchor(
    result: RetrievalResult,
    final_context: tuple[str, ...],
) -> str:
    routes = tuple(
        route
        for route in (_parse_context_route(item) for item in final_context)
        if route is not None
    )
    focused = tuple(route for route in routes if route.exclusive)
    selected_files = tuple(dict.fromkeys(route.file for route in routes))
    top_symbols = result.symbols[:8]
    dependencies = tuple(dict.fromkeys(
        dep
        for symbol in top_symbols
        for dep in (*symbol.calls, *symbol.tables_referenced)
    ))
    lines = [
        "PROJECT GLOBAL SUMMARY:",
        "context_policy: focus means highlight, not isolation",
    ]
    if selected_files:
        lines.append(f"selected_files: {', '.join(selected_files)}")
    if focused:
        lines.append(
            "focused_symbols: "
            + ", ".join(f"{route.file}:{route.name}" for route in focused)
        )
    if top_symbols:
        lines.append(
            "retrieval_symbols: "
            + ", ".join(
                f"{symbol.file}:{symbol.name}"
                for symbol in top_symbols
            )
        )
    if dependencies:
        lines.append(f"depends_on: {', '.join(dependencies)}")
    schema_symbols = [
        symbol
        for symbol in top_symbols
        if symbol.kind in {"ddl_view", "dml_select"} or symbol.tables_referenced
    ]
    if schema_symbols:
        lines.append(
            "schema_relation: "
            + "; ".join(
                f"{symbol.name} -> {', '.join(symbol.tables_referenced) or 'unknown'}"
                for symbol in schema_symbols
            )
        )
    return "\n".join(lines)


def _inject_global_anchor(
    windows: tuple[ContextWindow, ...],
    anchor: str,
    max_chars_per_file: int,
) -> tuple[ContextWindow, ...]:
    if not windows or not anchor:
        return windows
    first, *rest = windows
    content = f"{anchor}\n\n{first.content}"
    if len(content) > max_chars_per_file:
        content = content[:max_chars_per_file] + "\n...[truncated]"
    return (
        ContextWindow(
            file=first.file,
            start_line=first.start_line,
            end_line=first.end_line,
            content=content,
            symbols=first.symbols,
            semantic_tags=first.semantic_tags,
        ),
        *rest,
    )


def _comment_prefix(file_path: str) -> str:
    ext = file_path.casefold().split(".")[-1]
    if ext == "sql":
        return "--"
    if ext in ("java", "go", "js", "ts", "cpp", "c", "h", "rs"):
        return "//"
    return "#"


def _routes_from_result(result: RetrievalResult) -> tuple[str, ...]:
    """Compatibility entry point: convert raw retrieval to explicit IR anchors."""
    if result.symbols:
        return tuple(
            f"{'FOCUS:' if index < 2 else ''}{symbol.file}:{symbol.name}:"
            f"{symbol.start_line}-{symbol.end_line}"
            for index, symbol in enumerate(result.symbols)
        )
    return tuple(f"{file}:FILE:0-0" for file in result.files)


def _select_core_routes(
    routes: tuple[_ContextRoute, ...],
    index: dict[tuple[str, str, int, int], RetrievalSymbol],
    symbols: tuple[RetrievalSymbol, ...],
) -> list[tuple[_ContextRoute, RetrievalSymbol | None]]:
    """Pick at most two editable code anchors; DDL remains Layer-2 evidence."""
    selected: list[tuple[_ContextRoute, RetrievalSymbol | None]] = []
    seen_files: set[str] = set()
    focus_files = {route.file for route in routes if route.exclusive and not route.is_file}

    def add(route: _ContextRoute, symbol: RetrievalSymbol | None) -> None:
        if route.file in seen_files or route.is_file or _is_soft_only(symbol, route):
            return
        seen_files.add(route.file)
        selected.append((route, symbol))

    # A focused embedded SQL statement should promote its enclosing function,
    # not turn the statement itself into the editable core block.
    for file in focus_files:
        focused = [route for route in routes if route.file == file and route.exclusive]
        for route in focused:
            enclosing = next((
                symbol for symbol in symbols
                if symbol.file == file
                and not _is_soft_only(symbol, None)
                and symbol.start_line <= route.start_line <= symbol.end_line
            ), None)
            if enclosing is not None:
                add(_ContextRoute(
                    file=file,
                    name=enclosing.name,
                    start_line=enclosing.start_line,
                    end_line=enclosing.end_line,
                    exclusive=True,
                    is_file=False,
                ), enclosing)
                break

    for route in routes:
        symbol = index.get((route.file, route.name, route.start_line, route.end_line))
        add(route, symbol)
    return selected


def _is_soft_only(symbol: RetrievalSymbol | None, route: _ContextRoute | None) -> bool:
    kind = (symbol.kind if symbol is not None else "").casefold()
    name = (symbol.name if symbol is not None else (route.name if route else "")).casefold()
    return kind.startswith(("ddl_", "dml_")) or name.startswith(
        ("select:", "insert:", "update:", "delete:")
    )


def _numbered_lines(lines: list[str], start: int, end: int) -> str:
    start = max(1, start)
    end = min(len(lines), max(start, end))
    return "\n".join(f"{number}: {lines[number - 1]}" for number in range(start, end + 1))


def _semantic_header(path: Path, lines: list[str], start: int, end: int) -> str:
    """Return only imports whose bound names occur in the core AST node."""
    if path.suffix.casefold() != ".py":
        return "[SEMANTIC_HEADER] unavailable (non-Python source)"
    try:
        tree = ast.parse("\n".join(lines))
    except SyntaxError:
        return "[SEMANTIC_HEADER] unavailable (parse error)"
    target = next((
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef)
        and node.lineno == start
        and getattr(node, "end_lineno", end) >= end
    ), None)
    if target is None:
        return "[SEMANTIC_HEADER] none"
    used = {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}
    imports: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            bound = {alias.asname or alias.name.split(".", 1)[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom):
            bound = {alias.asname or alias.name for alias in node.names}
        else:
            continue
        if bound & used:
            imports.append(f"{node.lineno}: {lines[node.lineno - 1]}")
    return "[SEMANTIC_HEADER]\n" + ("\n".join(imports) or "none")


def _global_skeleton(path: Path, lines: list[str]) -> str:
    """Low-resolution topology: declarations only, never implementation bodies."""
    declarations: list[str] = []
    if path.suffix.casefold() == ".py":
        try:
            tree = ast.parse("\n".join(lines))
            nodes = sorted(
                (
                    node for node in ast.walk(tree)
                    if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
                ),
                key=lambda node: node.lineno,
            )
            declarations = [f"{node.lineno}: {lines[node.lineno - 1].strip()}" for node in nodes]
        except SyntaxError:
            declarations = []
    if not declarations:
        pattern = re.compile(r"^\s*(?:async\s+)?(?:def|class|function|func)\s+.+")
        declarations = [
            f"{number}: {line.strip()}"
            for number, line in enumerate(lines, 1)
            if pattern.match(line)
        ]
    return "\n".join(declarations[:80]) or "(no declarations detected)"


def _sql_tables(text: str) -> set[str]:
    return {
        match.group(1).strip("`\"[]").casefold()
        for match in re.finditer(r"\b(?:from|join|update|into)\s+([\w.`\"\[\]]+)", text, re.I)
    }


def _ddl_affinity(
    core_tables: set[str],
    core_text: str,
    candidate: RetrievalSymbol,
) -> float:
    """Score DDL as optional evidence; it never becomes a hard focus target."""
    ddl_tables = {table.casefold() for table in candidate.tables_referenced}
    if not core_tables or not ddl_tables:
        return 0.0
    overlap = len(core_tables & ddl_tables) / len(core_tables | ddl_tables)
    identifier = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
    core_columns = {token.casefold() for token in identifier.findall(core_text)} - _SQL_KEYWORDS
    ddl_columns = {token.casefold() for token in identifier.findall(candidate.name)} - _SQL_KEYWORDS
    column_match = len(core_columns & ddl_columns) / max(1, len(core_columns))
    usage = min(1.0, sum(core_text.casefold().count(table) for table in ddl_tables) / 2.0)
    return round(0.55 * overlap + 0.30 * column_match + 0.15 * usage, 4)


_SQL_KEYWORDS = frozenset({
    "select", "from", "where", "join", "left", "right", "inner", "outer", "on",
    "as", "and", "or", "count", "group", "by", "order", "limit", "offset", "into",
    "update", "insert", "delete", "create", "view", "table", "return", "for", "in",
})


def _clip(content: str, limit: int) -> str:
    if len(content) <= limit:
        return content
    suffix = "\n...[truncated]"
    return content[: max(0, limit - len(suffix))] + suffix


@dataclass(frozen=True, slots=True)
class _ContextRoute:
    file: str
    name: str
    start_line: int
    end_line: int
    exclusive: bool
    is_file: bool
    evidence: str = ""


@dataclass(frozen=True, slots=True)
class _SymbolNeighborhood:
    parents: tuple[RetrievalSymbol, ...] = ()
    children: tuple[RetrievalSymbol, ...] = ()
    siblings: tuple[RetrievalSymbol, ...] = ()

    @property
    def all_symbols(self) -> tuple[RetrievalSymbol, ...]:
        return tuple(dict.fromkeys((*self.parents, *self.children, *self.siblings)))


def _parse_context_route(item: str) -> _ContextRoute | None:
    exclusive = item.startswith(("FOCUS:", "EXCLUSIVE_FOCUS:"))
    if item.startswith("FOCUS:"):
        item = item.removeprefix("FOCUS:")
    elif item.startswith("EXCLUSIVE_FOCUS:"):
        item = item.removeprefix("EXCLUSIVE_FOCUS:")
    item, evidence = _split_evidence(item)
    try:
        file_and_name, line_range = item.rsplit(":", 1)
        file_rel, name = file_and_name.rsplit(":", 1)
        start_text, end_text = line_range.split("-", 1)
        start_line = int(start_text)
        end_line = int(end_text)
    except ValueError:
        return None
    return _ContextRoute(
        file=file_rel,
        name=name,
        start_line=start_line,
        end_line=end_line,
        exclusive=exclusive,
        is_file=name == "FILE" and start_line == 0 and end_line == 0,
        evidence=evidence,
    )


def _split_evidence(item: str) -> tuple[str, str]:
    for marker in ("\n[SCHEMA_EVIDENCE]", "\n[SCHEMA_GRAPH_AFFINITY]:"):
        if marker in item:
            coordinate, evidence = item.split(marker, 1)
            return coordinate, f"{marker.strip()}{evidence}"
    return item, ""
