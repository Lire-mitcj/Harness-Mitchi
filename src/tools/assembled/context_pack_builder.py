from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from pathlib import Path

from src.agent.contracts import (
    ContextPack,
    ContextWindow,
    RetrievalResult,
    RetrievalSymbol,
    SemanticAnnotations,
)


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]
    for current in intervals[1:]:
        prev_start, prev_end = merged[-1]
        curr_start, curr_end = current
        if curr_start <= prev_end + 1:
            merged[-1] = (prev_start, max(prev_end, curr_end))
        else:
            merged.append(current)
    return merged


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

        # Hash GroupBy File (Vulnerability 2 Fix: prevent multi-file quota preemption)
        from collections import defaultdict
        routes_by_file = defaultdict(list)
        file_order = []
        for route in routes:
            if route.file not in routes_by_file:
                file_order.append(route.file)
            routes_by_file[route.file].append(route)

        resolved_by_file = {}
        evidence_by_file = defaultdict(list)
        for file_rel in file_order:
            file_routes = routes_by_file[file_rel]
            core_res, evidence_res = _select_file_core_routes(
                self.project_root, file_rel, file_routes, symbol_index, result.symbols
            )
            if core_res:
                resolved_by_file[file_rel] = core_res
            if evidence_res:
                evidence_by_file[file_rel].extend(evidence_res)

        # Fallback check globally across all files if no core routes are resolved anywhere
        if not resolved_by_file:
            for file_rel in file_order:
                file_routes = routes_by_file[file_rel]
                fallback_res = []
                for r in file_routes:
                    if r.is_file:
                        path = self._safe_file(r.file)
                        if path is not None:
                            try:
                                line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
                                fallback_res.append((
                                    _ContextRoute(
                                        r.file, "FILE_FALLBACK", 1, max(1, line_count), False, False
                                    ),
                                    None
                                ))
                            except Exception:
                                pass
                    else:
                        fallback_res.append((
                            r,
                            symbol_index.get((r.file, r.name, r.start_line, r.end_line))
                        ))
                if fallback_res:
                    resolved_by_file[file_rel] = fallback_res
                    if len(resolved_by_file) >= self.max_files:
                        break

        active_files = [f for f in file_order if f in resolved_by_file]
        selected_files = active_files[: self.max_files]

        windows: list[ContextWindow] = []
        for file_rel in selected_files:
            routes_with_symbols = resolved_by_file[file_rel]
            path = self._safe_file(file_rel)
            if path is None:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            if not lines:
                continue

            # Singleton AST parsing per file (Vulnerability 3 Fix: avoid AST re-parsing cost)
            tree = _parse_file_ast(path, lines)

            # Collect ranges to merge
            intervals = []
            for route, _ in routes_with_symbols:
                start = max(1, min(route.start_line, len(lines)))
                end = max(start, min(route.end_line, len(lines)))
                intervals.append((start, end))

            merged_ranges = merge_intervals(intervals)
            if not merged_ranges:
                continue

            # Layer 1 (Vulnerability 1 Fix: physical line interval collapse marker)
            spans = []
            symbols_set = set()
            for route, _ in routes_with_symbols:
                symbols_set.add(route.name)
            symbols_str = ", ".join(sorted(list(symbols_set)))

            for i, (start, end) in enumerate(merged_ranges):
                if i > 0:
                    prev_end = merged_ranges[i - 1][1]
                    collapsed_start = prev_end + 1
                    collapsed_end = start - 1
                    if collapsed_start <= collapsed_end:
                        spans.append(
                            f"... 🚨 [PHYSICAL LINE INTERVAL {collapsed_start}-{collapsed_end} COLLAPSED DUE TO PRUNING POLICY] ..."
                        )
                core = _numbered_lines(lines, start, end)
                spans.append(
                    f"[INTERVAL_CHUNK RANGE {start}-{end}]\n{core}"
                )

            layer1_content = (
                "[LAYER_1_CORE_SYMBOL]\n"
                f"file: {file_rel}\n"
                f"symbol: {symbols_str}\n\n"
                + "\n".join(spans)
            )

            # Layer 2: semantic header & soft dependencies
            header = _merged_semantic_header(path, tree, lines, merged_ranges)

            soft_blocks = []
            for route, core_symbol in routes_with_symbols:
                if core_symbol is not None:
                    r_start = max(1, min(route.start_line, len(lines)))
                    r_end = max(r_start, min(route.end_line, len(lines)))
                    soft_str = self._soft_dependencies(
                        core_symbol, result.symbols, lines[r_start - 1 : r_end], symbol_index
                    )
                    if soft_str:
                        soft_blocks.append(soft_str)

            # Format and merge resolved evidence routes from other files under Layer 2
            for ev_file, ev_list in evidence_by_file.items():
                for ev_route, ev_symbol in ev_list:
                    ev_path = self._safe_file(ev_route.file)
                    if ev_path is not None:
                        try:
                            ev_lines = ev_path.read_text(encoding="utf-8", errors="replace").splitlines()
                            ev_start = max(1, min(ev_route.start_line, len(ev_lines)))
                            ev_end = max(ev_start, min(ev_route.end_line, len(ev_lines)))
                            ev_body = _numbered_lines(ev_lines, ev_start, ev_end)
                            ev_str = (
                                "[SOFT_DEPENDENCY]\n"
                                "kind: direct_evidence\n"
                                f"file: {ev_route.file}\nlines: {ev_start}-{ev_end}\n"
                                f"symbol: {ev_route.name}\n{ev_body}"
                            )
                            soft_blocks.append(ev_str)
                        except Exception:
                            pass

            unique_soft_blocks = []
            for block in soft_blocks:
                for sub_block in block.split("\n\n"):
                    if sub_block.strip() and sub_block not in unique_soft_blocks:
                        unique_soft_blocks.append(sub_block)
            soft_content = "\n\n".join(unique_soft_blocks)

            layer2_content = (
                "[LAYER_2_DEPENDENCY_EVIDENCE]\n"
                + (header or "[SEMANTIC_HEADER] none")
                + ("\n\n" + soft_content if soft_content else "")
            )

            # Layer 3: global skeleton
            skeleton = _global_skeleton(path, lines)
            layer3_content = "[LAYER_3_GLOBAL_SKELETON]\n" + skeleton

            # Combined content
            content = "\n\n".join(
                part for part in (layer1_content, layer2_content, layer3_content) if part
            )
            content = _clip(content, self.max_chars_per_file)

            windows.append(ContextWindow(
                file=file_rel,
                start_line=min(start for start, end in merged_ranges),
                end_line=max(end for start, end in merged_ranges),
                content=content,
                symbols=tuple(sorted(list(symbols_set))),
                semantic_tags=annotations.tags_by_file.get(file_rel, ()),
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

    def merge_interval_subgraph(
        self,
        context_pack: ContextPack,
        raw_retrieval: RetrievalResult,
    ) -> ContextPack:
        from collections import defaultdict
        file_intervals = defaultdict(list)
        file_symbols = defaultdict(set)
        file_tags = defaultdict(set)
        
        for w in context_pack.windows:
            file_intervals[w.file].append((w.start_line, w.end_line))
            file_symbols[w.file].update(w.symbols)
            file_tags[w.file].update(w.semantic_tags)
            
        for sym in raw_retrieval.symbols:
            file_intervals[sym.file].append((sym.start_line, sym.end_line))
            file_symbols[sym.file].add(sym.name)
            
        for f in raw_retrieval.files:
            if f not in file_intervals:
                path = self._safe_file(f)
                if path is not None:
                    try:
                        line_count = len(path.read_text(encoding="utf-8", errors="replace").splitlines())
                        file_intervals[f].append((1, max(1, line_count)))
                    except Exception:
                        pass

        merged_windows = []
        for file_rel, intervals in file_intervals.items():
            if not intervals:
                continue
            
            path = self._safe_file(file_rel)
            if path is None:
                continue
            try:
                lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            except Exception:
                continue
            if not lines:
                continue
                
            merged_ranges = merge_intervals(intervals)
            if not merged_ranges:
                continue

            # Singleton AST parsing per file (Vulnerability 3 Fix: avoid AST re-parsing cost)
            tree = _parse_file_ast(path, lines)

            # Layer 1 (Vulnerability 1 Fix: physical line interval collapse marker)
            spans = []
            symbols_str = ", ".join(sorted(file_symbols[file_rel])) or "merged"
            for i, (start, end) in enumerate(merged_ranges):
                start = max(1, min(start, len(lines)))
                end = max(start, min(end, len(lines)))
                
                if i > 0:
                    prev_end = merged_ranges[i - 1][1]
                    collapsed_start = prev_end + 1
                    collapsed_end = start - 1
                    if collapsed_start <= collapsed_end:
                        spans.append(
                            f"... 🚨 [PHYSICAL LINE INTERVAL {collapsed_start}-{collapsed_end} COLLAPSED DUE TO PRUNING POLICY] ..."
                        )
                core = _numbered_lines(lines, start, end)
                spans.append(
                    f"[INTERVAL_CHUNK RANGE {start}-{end}]\n{core}"
                )

            layer1_content = (
                "[LAYER_1_CORE_SYMBOL]\n"
                f"file: {file_rel}\n"
                f"symbol: {symbols_str}\n\n"
                + "\n".join(spans)
            )

            # Layer 2: semantic header
            header = _merged_semantic_header(path, tree, lines, merged_ranges)
            layer2_content = (
                "[LAYER_2_DEPENDENCY_EVIDENCE]\n"
                + (header or "[SEMANTIC_HEADER] none")
            )

            # Layer 3: global skeleton
            skeleton = _global_skeleton(path, lines)
            layer3_content = "[LAYER_3_GLOBAL_SKELETON]\n" + skeleton

            # Combined content
            content = "\n\n".join(
                part for part in (layer1_content, layer2_content, layer3_content) if part
            )
            content = _clip(content, self.max_chars_per_file)

            merged_windows.append(ContextWindow(
                file=file_rel,
                start_line=min(start for start, end in merged_ranges),
                end_line=max(end for start, end in merged_ranges),
                content=content,
                symbols=tuple(sorted(file_symbols[file_rel])),
                semantic_tags=tuple(sorted(file_tags[file_rel])),
            ))
        return ContextPack(windows=tuple(merged_windows))

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


def _find_enclosing_range_in_file(project_root: Path, file_rel: str, line_no: int) -> tuple[str, int, int] | None:
    try:
        path = (project_root / file_rel).resolve()
        path.relative_to(project_root.resolve())
        if not path.is_file() or path.suffix.casefold() != ".py":
            return None
        content = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content)
        enclosing_node = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = node.lineno
                end = getattr(node, "end_lineno", start)
                if start <= line_no <= end:
                    if enclosing_node is None:
                        enclosing_node = node
                    else:
                        curr_start = enclosing_node.lineno
                        curr_end = getattr(enclosing_node, "end_lineno", curr_start)
                        if (end - start) < (curr_end - curr_start):
                            enclosing_node = node
        if enclosing_node:
            return enclosing_node.name, enclosing_node.lineno, getattr(enclosing_node, "end_lineno", enclosing_node.lineno)
    except Exception:
        pass
    return None


def _select_core_routes(
    project_root: Path,
    routes: tuple[_ContextRoute, ...],
    index: dict[tuple[str, str, int, int], RetrievalSymbol],
    symbols: tuple[RetrievalSymbol, ...],
) -> list[tuple[_ContextRoute, RetrievalSymbol | None]]:
    """Pick at most two editable code anchors; DDL remains Layer-2 evidence."""
    selected: list[tuple[_ContextRoute, RetrievalSymbol | None]] = []
    seen_files: set[str] = set()
    focus_files = dict.fromkeys(route.file for route in routes if route.exclusive and not route.is_file)

    def add(route: _ContextRoute, symbol: RetrievalSymbol | None) -> None:
        if route.file in seen_files or route.is_file or _is_soft_only(symbol, route):
            return
        seen_files.add(route.file)
        selected.append((route, symbol))

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
            else:
                res = _find_enclosing_range_in_file(project_root, route.file, route.start_line)
                if res is not None:
                    enc_name, enc_start, enc_end = res
                    add(_ContextRoute(
                        file=file,
                        name=enc_name,
                        start_line=enc_start,
                        end_line=enc_end,
                        exclusive=True,
                        is_file=False,
                    ), None)
                    break

    for route in routes:
        symbol = index.get((route.file, route.name, route.start_line, route.end_line))
        if symbol is None or _is_soft_only(symbol, route):
            res = _find_enclosing_range_in_file(project_root, route.file, route.start_line)
            if res is not None:
                enc_name, enc_start, enc_end = res
                promoted_route = _ContextRoute(
                    file=route.file,
                    name=enc_name,
                    start_line=enc_start,
                    end_line=enc_end,
                    exclusive=route.exclusive,
                    is_file=False,
                )
                add(promoted_route, None)
                continue
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
        file_rel, name = file_and_name.split(":", 1)
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


def _parse_file_ast(path: Path, lines: list[str]) -> ast.AST | None:
    if path.suffix.casefold() != ".py":
        return None
    try:
        return ast.parse("\n".join(lines))
    except SyntaxError:
        return None


def _merged_semantic_header(
    path: Path,
    tree: ast.AST | None,
    lines: list[str],
    ranges: list[tuple[int, int]],
) -> str:
    """Return only imports whose bound names occur in target functions/classes spanning the merged ranges."""
    if path.suffix.casefold() != ".py":
        return "[SEMANTIC_HEADER] unavailable (non-Python source)"
    if tree is None:
        return "[SEMANTIC_HEADER] unavailable (parse error)"

    target_starts = {start for start, end in ranges}
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if node.lineno in target_starts:
                targets.append(node)

    if not targets:
        return "[SEMANTIC_HEADER] none"

    used = set()
    for target in targets:
        used.update(node.id for node in ast.walk(target) if isinstance(node, ast.Name))

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


def _resolve_route(
    project_root: Path,
    file_rel: str,
    r: _ContextRoute,
    index: dict[tuple[str, str, int, int], RetrievalSymbol],
    symbols: tuple[RetrievalSymbol, ...],
) -> tuple[str, _ContextRoute, RetrievalSymbol | None] | None:
    if r.is_file:
        return None

    if r.exclusive:
        # Check if enclosing symbol in symbols (not soft-only)
        enclosing = next((
            symbol for symbol in symbols
            if symbol.file == file_rel
            and not _is_soft_only(symbol, None)
            and symbol.start_line <= r.start_line <= symbol.end_line
        ), None)
        if enclosing is not None:
            return (
                "core",
                _ContextRoute(
                    file=file_rel,
                    name=enclosing.name,
                    start_line=enclosing.start_line,
                    end_line=enclosing.end_line,
                    exclusive=True,
                    is_file=False,
                ),
                enclosing
            )
        else:
            # Check AST
            res = _find_enclosing_range_in_file(project_root, file_rel, r.start_line)
            if res is not None:
                enc_name, enc_start, enc_end = res
                return (
                    "core",
                    _ContextRoute(
                        file=file_rel,
                        name=enc_name,
                        start_line=enc_start,
                        end_line=enc_end,
                        exclusive=True,
                        is_file=False,
                    ),
                    None
                )
            else:
                # Exclusive route that is soft-only or not in a Python file -> evidence
                return ("evidence", r, None)
    else:
        # Non-exclusive route
        sym = index.get((r.file, r.name, r.start_line, r.end_line))
        if sym is None or _is_soft_only(sym, r):
            res = _find_enclosing_range_in_file(project_root, r.file, r.start_line)
            if res is not None:
                enc_name, enc_start, enc_end = res
                return (
                    "core",
                    _ContextRoute(
                        file=r.file,
                        name=enc_name,
                        start_line=enc_start,
                        end_line=enc_end,
                        exclusive=r.exclusive,
                        is_file=False,
                    ),
                    None
                )
            else:
                # Soft-only or query route not in a Python file -> evidence
                return ("evidence", r, sym)
        else:
            return ("core", r, sym)


def _select_file_core_routes(
    project_root: Path,
    file_rel: str,
    routes: list[_ContextRoute],
    index: dict[tuple[str, str, int, int], RetrievalSymbol],
    symbols: tuple[RetrievalSymbol, ...],
) -> tuple[list[tuple[_ContextRoute, RetrievalSymbol | None]], list[tuple[_ContextRoute, RetrievalSymbol | None]]]:
    """Pick core and evidence routes and symbols within a single file, processing all routes sequentially."""
    core_resolved: list[tuple[_ContextRoute, RetrievalSymbol | None]] = []
    evidence_resolved: list[tuple[_ContextRoute, RetrievalSymbol | None]] = []
    seen_keys: set[tuple[str, str, int, int]] = set()

    for r in routes:
        res = _resolve_route(project_root, file_rel, r, index, symbols)
        if res is not None:
            kind, resolved_route, resolved_symbol = res
            key = (resolved_route.file, resolved_route.name, resolved_route.start_line, resolved_route.end_line)
            if key not in seen_keys:
                seen_keys.add(key)
                if kind == "core":
                    core_resolved.append((resolved_route, resolved_symbol))
                else:
                    evidence_resolved.append((resolved_route, resolved_symbol))
                
    return core_resolved, evidence_resolved
