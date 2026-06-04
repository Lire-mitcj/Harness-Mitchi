from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.indexer.ctags import CtagsIndexResult, CtagsSymbol, index_project
from src.indexer.graph import build_reference_edges
from src.indexer.pagerank import pagerank


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

    @property
    def location(self) -> str:
        if self.start_line == self.end_line:
            return f"{self.file_path}:{self.start_line}"
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


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
    ) -> str:
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
            syms = self.symbols_by_file.get(path, [])[:5]
            preview = ", ".join(s.name for s in syms) if syms else "(no symbols)"
            file_lines.append(f"- {path}  score={score:.4f}  — {preview}")

        symbol_lines = []
        for sym in self.symbols[:top_symbols]:
            sig = sym.signature.replace("\n", " ")[:100]
            sig_part = f"  {sig}" if sig else ""
            symbol_lines.append(
                f"- {sym.location}  {sym.kind} {sym.name}{sig_part}  "
                f"score={sym.score:.5f}"
            )

        skeleton_lines: list[str] = []
        for path, _ in ranked_files:
            file_syms = self.symbols_by_file.get(path, [])
            if not file_syms:
                continue
            skeleton_lines.append(f"{path}")
            for s in file_syms[:12]:
                sig = s.signature[:80] if s.signature else s.kind
                skeleton_lines.append(f"  L{s.start_line}-{s.end_line}  {s.name}  {sig}")

        search_modules = _build_search_modules(
            ranked_files,
            self.symbols_by_file,
        )

        return _truncate_skeleton_sections(
            header=header,
            top_files=file_lines,
            search_modules=search_modules,
            top_symbols=symbol_lines,
            file_skeleton=skeleton_lines,
            max_chars=max_chars,
        )

    def to_planner_context(self, *, max_chars: int = 12_000) -> str:
        return self.to_skeleton_block(max_chars=max_chars)


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

    name_to_ids: dict[str, list[str]] = {}
    file_nodes: dict[str, str] = {}
    symbol_nodes: dict[tuple[str, str, int], str] = {}
    edges: list[tuple[str, str]] = []

    for sym in indexed.symbols:
        symbol_nodes[(sym.file_path, sym.name, sym.start_line)] = _symbol_id(sym)

    for sym in indexed.symbols:
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
    for sym in indexed.symbols:
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
            )
        )
        file_score_acc[sym.file_path] = file_score_acc.get(sym.file_path, 0.0) + score

    ranked.sort(key=lambda s: s.score, reverse=True)
    symbols_by_file: dict[str, list[RankedSymbol]] = {}
    for sym in ranked:
        symbols_by_file.setdefault(sym.file_path, []).append(sym)
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
        symbol_count=len(indexed.symbols),
    )


def _symbol_id(sym: CtagsSymbol) -> str:
    return f"{sym.file_path}:{sym.name}:{sym.start_line}"
