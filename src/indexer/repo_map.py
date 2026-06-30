from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.agent.sql_parser import UniversalSqlParser
from src.indexer.ctags import CtagsIndexResult, CtagsSymbol, index_project
from src.indexer.graph import build_reference_edges
from src.indexer.pagerank import pagerank
from src.indexer.scanner import ProjectScanner


@dataclass(frozen=True)
class RankedSymbol:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    score: float
    symbol_id: str
    tables_referenced: tuple[str, ...] = ()
    parent_symbol: str = ""
    parent_symbol_id: str = ""

    @property
    def location(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class SymbolCandidate:
    symbol: RankedSymbol
    score_hint: float
    reasons: tuple[str, ...] = ()


@dataclass
class RepoMap:
    """Compressed project skeleton for Planner (symbols + PageRank hubs)."""

    project_root: Path
    symbols: list[RankedSymbol] = field(default_factory=list)
    all_symbols: list[RankedSymbol] = field(default_factory=list)
    symbols_by_file: dict[str, list[RankedSymbol]] = field(default_factory=dict)
    symbols_by_id: dict[str, RankedSymbol] = field(default_factory=dict)
    file_scores: dict[str, float] = field(default_factory=dict)
    reference_edges: list[tuple[str, str]] = field(default_factory=list)
    source: str = "parser"
    build_ms: int = 0
    symbol_count: int = 0

    def search(self, query: str, *, limit: int = 20) -> list[RankedSymbol]:
        q = query.lower().strip()
        if not q:
            pool = self.all_symbols or self.symbols
            return pool[:limit]
        pool = self.all_symbols or self.symbols
        hits = [
            s
            for s in pool
            if q in s.name.lower()
            or q in s.file_path.lower()
            or q in s.signature.lower()
        ]
        hits.sort(key=lambda s: s.score, reverse=True)
        return hits[:limit]

    def lookup_candidates(
        self,
        expanded_terms: list[str] | tuple[str, ...],
        *,
        domain: str = "",
        constraints: dict[str, list[str]] | None = None,
        limit: int = 40,
    ) -> list[SymbolCandidate]:
        """Locate symbol candidates from already-expanded query terms."""
        pool = self.all_symbols or self.symbols
        if not pool:
            return []
        constraints = constraints or {}
        terms = _lookup_terms(expanded_terms, domain, constraints)
        candidates: list[SymbolCandidate] = []
        by_id = {sym.symbol_id: sym for sym in pool}
        for sym in pool:
            haystacks = {
                "name": sym.name.casefold(),
                "file": sym.file_path.casefold(),
                "signature": sym.signature.casefold(),
                "kind": sym.kind.casefold(),
            }
            score = 0.0
            reasons: set[str] = set()
            name_tokens = set(_split_identifier(sym.name))
            file_tokens = set(_split_identifier(sym.file_path))
            sig_tokens = set(_split_identifier(sym.signature))
            for term in terms:
                term_low = term.casefold()
                term_tokens = set(_split_identifier(term))
                if not term_low or not term_tokens:
                    continue
                if term_low == haystacks["name"]:
                    score += 1.0
                    reasons.add("name_exact")
                elif term_low in haystacks["name"]:
                    score += 0.72
                    reasons.add("name_match")
                elif term_tokens & name_tokens:
                    score += 0.58
                    reasons.add("name_token_match")
                if term_low in haystacks["file"] or term_tokens & file_tokens:
                    score += 0.34
                    reasons.add("file_match")
                if term_low in haystacks["signature"] or term_tokens & sig_tokens:
                    score += 0.22
                    reasons.add("signature_match")
                if term_low == haystacks["kind"]:
                    score += 0.18
                    reasons.add("kind_match")
            if score <= 0.0:
                continue
            if sym.kind.startswith("dml_"):
                score *= 0.35
            score += min(sym.score, 0.15)
            candidates.append(
                SymbolCandidate(
                    symbol=sym,
                    score_hint=round(min(score / 2.2, 1.0), 4),
                    reasons=tuple(sorted(reasons)),
                )
            )
        candidates.extend(_embedded_sql_parent_candidates(candidates, by_id))
        candidates.sort(
            key=lambda item: (
                -item.score_hint,
                -item.symbol.score,
                item.symbol.file_path,
                item.symbol.start_line,
                item.symbol.name,
            )
        )
        return candidates[:limit]

    def expand_symbol_edges(
        self,
        symbol_ids: list[str],
        *,
        depth: int = 2,
        limit: int = 20,
    ) -> list[tuple[RankedSymbol, RankedSymbol]]:
        """Expand high-signal symbol references around focused symbols."""
        if not symbol_ids or not self.reference_edges:
            return []
        adjacency: dict[str, list[str]] = {}
        for src, dst in self.reference_edges:
            adjacency.setdefault(src, []).append(dst)

        out: list[tuple[RankedSymbol, RankedSymbol]] = []
        seen_edges: set[tuple[str, str]] = set()
        seen_nodes = set(symbol_ids)
        frontier = list(symbol_ids)
        for _level in range(max(1, depth)):
            next_frontier: list[str] = []
            for src in frontier:
                src_sym = self.symbols_by_id.get(src)
                for dst in adjacency.get(src, []):
                    dst_sym = self.symbols_by_id.get(dst)
                    if src_sym is not None and dst_sym is not None:
                        key = (src, dst)
                        if key not in seen_edges:
                            seen_edges.add(key)
                            out.append((src_sym, dst_sym))
                            if len(out) >= limit:
                                return out
                    if dst not in seen_nodes:
                        seen_nodes.add(dst)
                        next_frontier.append(dst)
            frontier = next_frontier
            if not frontier:
                break
        return out

    def to_skeleton_block(
        self,
        *,
        top_symbols: int = 40,
        top_files: int = 15,
        max_chars: int = 12_000,
        exclude_symbols: set[str] | None = None,
    ) -> str:
        exclude = exclude_symbols or set()
        header = [
            f'<repo_map source="{self.source}" symbols="{self.symbol_count}" '
            f'build_ms="{self.build_ms}">',
            "Use this skeleton to set context_files and narrow edit targets — "
            "executors read/grep within each subtask.",
        ]
        ranked_files = sorted(
            self.file_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:top_files]
        file_lines = []
        for path, score in ranked_files:
            syms = [s for s in self.symbols_by_file.get(path, []) if s.name not in exclude][:5]
            preview = ", ".join(s.name for s in syms) if syms else "(no symbols)"
            file_lines.append(f"- {path}  score={score:.4f}  — {preview}")

        symbol_lines = []
        filtered_top_symbols = [sym for sym in self.symbols if sym.name not in exclude]
        for sym in filtered_top_symbols[:top_symbols]:
            sig = sym.signature.replace("\n", " ")[:100]
            sig_part = f"  {sig}" if sig else ""
            symbol_lines.append(
                f"- {sym.location}  {sym.kind} {sym.name}{sig_part}  "
                f"score={sym.score:.5f}"
            )

        skeleton_lines: list[str] = []
        for path, _ in ranked_files:
            file_syms = self.symbols_by_file.get(path, [])
            filtered_file_syms = [s for s in file_syms if s.name not in exclude]
            if not filtered_file_syms:
                continue
            skeleton_lines.append(f"{path}")
            for s in filtered_file_syms[:12]:
                sig = s.signature[:80] if s.signature else s.kind
                skeleton_lines.append(f"  L{s.start_line}-{s.end_line}  {s.name}  {sig}")

        # Construct filtered symbols map for search modules building
        filtered_symbols_by_file = {}
        for path, syms in self.symbols_by_file.items():
            filtered_symbols_by_file[path] = [s for s in syms if s.name not in exclude]

        search_modules = _build_search_modules(
            ranked_files,
            filtered_symbols_by_file,
        )

        return _truncate_skeleton_sections(
            header=header,
            top_files=file_lines,
            search_modules=search_modules,
            top_symbols=symbol_lines,
            file_skeleton=skeleton_lines,
            max_chars=max_chars,
        )

    def to_planner_context(self, *, max_chars: int = 12_000, exclude_symbols: set[str] | None = None) -> str:
        return self.to_skeleton_block(max_chars=max_chars, exclude_symbols=exclude_symbols)


def _truncate_skeleton_sections(
    *,
    header: list[str],
    top_files: list[str],
    search_modules: list[str],
    top_symbols: list[str],
    file_skeleton: list[str],
    max_chars: int,
) -> str:
    """Keep header + modules + hubs; drop file skeleton first, then trim symbols."""
    footer = ["</repo_map>"]
    suffix = "\n…[repo_map truncated]\n</repo_map>"
    fixed = "\n".join(header + footer)
    budget = max_chars - len(suffix)
    if budget <= len(fixed):
        return fixed[: max(0, max_chars - len(suffix))] + suffix

    def join(*sections: list[str]) -> str:
        body: list[str] = []
        for section in sections:
            body.extend(section)
        return "\n".join(header + body + footer)

    # Full content fits
    full = join(
        ["", "## Top files (PageRank hubs)", *top_files],
        ["", "## Search modules (one module per step; combine patterns with |)", *search_modules],
        ["", "## Top symbols (PageRank)", *top_symbols],
        ["", "## File skeleton (signatures only)", *file_skeleton],
    )
    if len(full) <= max_chars:
        return full

    # Drop file skeleton entirely
    without_skeleton = join(
        ["", "## Top files (PageRank hubs)", *top_files],
        ["", "## Search modules (one module per step; combine patterns with |)", *search_modules],
        ["", "## Top symbols (PageRank)", *top_symbols],
        ["", "## File skeleton (signatures only)", "  …[file skeleton omitted for size]"],
    )
    if len(without_skeleton) <= max_chars:
        return without_skeleton

    # Trim top symbol lines from the bottom
    sym_header = ["", "## Top symbols (PageRank)"]
    files_block = ["", "## Top files (PageRank hubs)", *top_files]
    modules_block = [
        "",
        "## Search modules (one module per step; combine patterns with |)",
        *search_modules,
    ]
    kept_syms = list(top_symbols)
    while kept_syms:
        candidate = join(
            files_block,
            modules_block,
            sym_header + kept_syms,
            ["", "## File skeleton (signatures only)", "  …[omitted]"],
        )
        if len(candidate) <= max_chars:
            return candidate
        kept_syms.pop()

    # Trim top files from the bottom
    kept_files = list(top_files)
    while kept_files:
        candidate = join(
            ["", "## Top files (PageRank hubs)", *kept_files],
            modules_block,
            sym_header + ["  …[top symbols omitted for size]"],
            ["", "## File skeleton (signatures only)", "  …[omitted]"],
        )
        if len(candidate) <= max_chars:
            return candidate
        kept_files.pop()

    return join(
        ["", "## Top files (PageRank hubs)", "  …[truncated]"],
        modules_block,
        sym_header + ["  …[omitted]"],
        ["", "## File skeleton (signatures only)", "  …[omitted]"],
    )


def _build_search_modules(
    ranked_files: list[tuple[str, float]],
    symbols_by_file: dict[str, list[RankedSymbol]],
    *,
    max_modules: int = 8,
    max_files_per_module: int = 5,
    max_patterns: int = 10,
) -> list[str]:
    grouped: dict[str, list[str]] = {}
    for path, _score in ranked_files:
        module = _module_name(path)
        grouped.setdefault(module, []).append(path)

    lines: list[str] = [
        "Rule: pick one module, then run grep_search with a single OR regex "
        "over that module's files/glob. Do not probe one keyword per turn.",
    ]
    for module, files in list(grouped.items())[:max_modules]:
        kept_files = files[:max_files_per_module]
        patterns = _module_patterns(module, kept_files, symbols_by_file, limit=max_patterns)
        if not kept_files or not patterns:
            continue
        glob_hint = _module_glob(kept_files)
        lines.append(
            f"- module={module} files={kept_files!r} glob={glob_hint!r} "
            f"patterns={'|'.join(patterns)!r}"
        )
    return lines


def _module_name(path: str) -> str:
    p = Path(path)
    if len(p.parts) > 1:
        return str(Path(*p.parts[:-1])).replace("\\", "/")
    return p.stem


def _module_glob(files: list[str]) -> str:
    suffixes = {Path(f).suffix for f in files if Path(f).suffix}
    if len(suffixes) == 1:
        return f"*{next(iter(suffixes))}"
    return "*"


def _module_patterns(
    module: str,
    files: list[str],
    symbols_by_file: dict[str, list[RankedSymbol]],
    *,
    limit: int,
) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    def add(token: str) -> None:
        token = token.strip("_").lower()
        if len(token) < 3 or token in seen:
            return
        seen.add(token)
        out.append(re.escape(token))

    for token in _split_identifier(module):
        add(token)
    for path in files:
        stem = Path(path).stem
        for token in _split_identifier(stem):
            add(token)
        for sym in symbols_by_file.get(path, [])[:12]:
            add(sym.name)
            for token in _split_identifier(sym.name):
                add(token)
            if sym.kind.lower() in {"function", "method"}:
                add("def")
            elif "view" in sym.kind.lower() or sym.name.lower().startswith("v_"):
                add("create view")
        if len(out) >= limit:
            break
    return out[:limit]


def _split_identifier(text: str) -> list[str]:
    text = text.replace("-", "_").replace("/", "_").replace(".", "_")
    parts = re.findall(r"[A-Za-z][A-Za-z0-9]*|[0-9]+", text)
    split: list[str] = []
    for part in parts:
        split.extend(re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", part))
    return split or parts


def _lookup_terms(
    expanded_terms: list[str] | tuple[str, ...],
    domain: str,
    constraints: dict[str, list[str]],
) -> tuple[str, ...]:
    terms: list[str] = []
    seen: set[str] = set()
    for raw in (
        list(expanded_terms)
        + ([domain] if domain else [])
        + list(constraints.get("layer_hint", ()))
    ):
        term = raw.strip()
        key = term.casefold()
        if len(term) < 2 or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return tuple(terms)


def _embedded_sql_parent_candidates(
    candidates: list[SymbolCandidate],
    by_id: dict[str, RankedSymbol],
) -> list[SymbolCandidate]:
    existing = {candidate.symbol.symbol_id for candidate in candidates}
    parents: list[SymbolCandidate] = []
    for candidate in candidates:
        parent_id = candidate.symbol.parent_symbol_id
        if not parent_id or parent_id in existing:
            continue
        parent = by_id.get(parent_id)
        if parent is None:
            continue
        existing.add(parent_id)
        parents.append(
            SymbolCandidate(
                symbol=parent,
                score_hint=max(candidate.score_hint, 0.72),
                reasons=tuple(sorted((*candidate.reasons, "embedded_sql_parent"))),
            )
        )
    return parents


def build_repo_map(
    project_root: Path,
    *,
    top_k: int = 200,
    indexed: CtagsIndexResult | None = None,
) -> RepoMap:
    """Build RepoMap: static index → reference graph → PageRank → skeleton."""
    t0 = time.perf_counter()
    root = project_root.resolve()
    if indexed is None:
        indexed = index_project(root)
    indexed_symbols = _merge_sql_structural_symbols(root, indexed.symbols)

    name_to_ids: dict[str, list[str]] = {}
    file_nodes: dict[str, str] = {}
    symbol_nodes: dict[tuple[str, str, int], str] = {}
    edges: list[tuple[str, str]] = []

    for sym in indexed_symbols:
        symbol_nodes[(sym.file_path, sym.name, sym.start_line)] = _symbol_id(sym)

    for sym in indexed_symbols:
        sid = symbol_nodes[(sym.file_path, sym.name, sym.start_line)]
        name_to_ids.setdefault(sym.name, []).append(sid)
        if sym.file_path not in file_nodes:
            file_nodes[sym.file_path] = f"file:{sym.file_path}"
        edges.append((file_nodes[sym.file_path], sid))

    edges.extend(
        build_reference_edges(
            indexed,
            file_nodes=file_nodes,
            symbol_nodes=symbol_nodes,
            name_to_ids=name_to_ids,
        )
    )

    all_nodes = list({n for pair in edges for n in pair})
    all_nodes.extend(symbol_nodes.values())
    scores = pagerank(edges, nodes=list(set(all_nodes)))

    ranked: list[RankedSymbol] = []
    file_score_acc: dict[str, float] = {}
    for sym in indexed_symbols:
        sid = symbol_nodes[(sym.file_path, sym.name, sym.start_line)]
        score = scores.get(sid, 0.0)
        ranked.append(
            RankedSymbol(
                file_path=sym.file_path,
                name=sym.name,
                kind=sym.kind,
                start_line=sym.start_line,
                end_line=sym.end_line,
                signature=sym.signature,
                score=score,
                symbol_id=sid,
                tables_referenced=tuple(getattr(sym, "tables_referenced", ())),
                parent_symbol=str(getattr(sym, "parent_symbol", "")),
                parent_symbol_id=str(getattr(sym, "parent_symbol_id", "")),
            )
        )
        file_score_acc[sym.file_path] = file_score_acc.get(sym.file_path, 0.0) + score

    for ranked_symbol in ranked:
        if ranked_symbol.parent_symbol_id and ranked_symbol.parent_symbol_id in scores:
            edges.append((ranked_symbol.parent_symbol_id, ranked_symbol.symbol_id))

    ranked.sort(key=lambda s: s.score, reverse=True)
    symbols_by_file: dict[str, list[RankedSymbol]] = {}
    for ranked_symbol in ranked:
        symbols_by_file.setdefault(ranked_symbol.file_path, []).append(ranked_symbol)
    for path in symbols_by_file:
        symbols_by_file[path].sort(key=lambda s: s.start_line)

    top_symbols = ranked[:top_k] if top_k > 0 else ranked

    file_scores = {
        path: file_score_acc.get(path, 0.0)
        for path in file_nodes
    }
    if file_scores:
        total = sum(file_scores.values()) or 1.0
        file_scores = {k: v / total for k, v in file_scores.items()}

    elapsed = int((time.perf_counter() - t0) * 1000)
    return RepoMap(
        project_root=root,
        symbols=top_symbols,
        all_symbols=ranked,
        symbols_by_file=symbols_by_file,
        symbols_by_id={sym.symbol_id: sym for sym in ranked},
        file_scores=file_scores,
        reference_edges=edges,
        source=indexed.source,
        build_ms=elapsed,
        symbol_count=len(indexed_symbols),
    )


def _merge_sql_structural_symbols(
    root: Path,
    indexed_symbols: list[CtagsSymbol],
) -> list[Any]:
    merged: dict[tuple[str, str, int], Any] = {
        (sym.file_path, sym.name, sym.start_line): sym
        for sym in indexed_symbols
    }
    for symbol in _collect_sql_structural_symbols(root):
        merged[(symbol.file_path, symbol.name, symbol.start_line)] = symbol
    return list(merged.values())


def _collect_sql_structural_symbols(root: Path) -> tuple[Any, ...]:
    parser = UniversalSqlParser()
    symbols: list[Any] = []
    for path in ProjectScanner(root).scan(max_files=5000).files:
        suffix = path.suffix.casefold()
        if suffix not in {".sql", ".py"}:
            continue
        try:
            rel = path.relative_to(root).as_posix()
            text = path.read_text(encoding="utf-8", errors="replace")
        except (OSError, ValueError):
            continue
        if suffix == ".sql":
            symbols.extend(parser.parse_text_block(text, rel, 1))
        else:
            symbols.extend(_parse_python_sql_strings(parser, text, rel))
    return tuple(symbols)


def _parse_python_sql_strings(
    parser: UniversalSqlParser,
    text: str,
    rel: str,
) -> tuple[Any, ...]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return ()
    symbols: list[Any] = []
    joined_children = _joined_string_child_ids(tree)
    for node in ast.walk(tree):
        if id(node) in joined_children:
            continue
        sql_text = _python_sql_literal_text(node)
        if sql_text is None:
            continue
        if not _looks_like_sql(sql_text):
            continue
        base_line = int(getattr(node, "lineno", 1))
        parent = _enclosing_function(tree, node)
        parsed = parser.parse_text_block(sql_text, rel, base_line)
        if parent is None:
            symbols.extend(parsed)
            continue
        parent_id = f"{rel}:{parent.name}:{parent.lineno}"
        for symbol in parsed:
            symbols.append(
                type(symbol)(
                    file_path=symbol.file_path,
                    name=f"{parent.name}:{symbol.name}",
                    kind=symbol.kind,
                    start_line=symbol.start_line,
                    end_line=symbol.end_line,
                    signature=f"{parent.name} embeds {symbol.signature}",
                    tables_referenced=symbol.tables_referenced,
                    parent_symbol=parent.name,
                    parent_symbol_id=parent_id,
                )
            )
    return tuple(symbols)


def _joined_string_child_ids(tree: ast.AST) -> set[int]:
    child_ids: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.JoinedStr):
            continue
        for child in ast.walk(node):
            if child is not node:
                child_ids.add(id(child))
    return child_ids


def _python_sql_literal_text(node: ast.AST) -> str | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue):
                expr = ast.unparse(value.value) if hasattr(ast, "unparse") else "expr"
                parts.append("{" + expr + "}")
        return "".join(parts)
    return None


def _enclosing_function(
    tree: ast.AST,
    target: ast.AST,
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    target_line = int(getattr(target, "lineno", 0))
    best: ast.FunctionDef | ast.AsyncFunctionDef | None = None
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        start = int(getattr(node, "lineno", 0))
        end = int(getattr(node, "end_lineno", start))
        if start <= target_line <= end and (best is None or start >= best.lineno):
            best = node
    return best


def _looks_like_sql(text: str) -> bool:
    lowered = text.casefold()
    return any(
        token in lowered
        for token in (
            "select ",
            "insert into ",
            "update ",
            "create view ",
            "create or replace view ",
            "create table ",
        )
    )


def _symbol_id(sym: Any) -> str:
    return f"{sym.file_path}:{sym.name}:{sym.start_line}"
