from __future__ import annotations

import re
from pathlib import Path
from typing import Protocol

from src.context.pack import ContextPack, ContextSnippet, ContextSymbol, SearchPlan


class RepoMapSearcher(Protocol):
    def search(self, query: str, *, limit: int = 20) -> list[object]: ...


class ContextRetriever:
    """Deterministic first-pass context gathering before planner/executor LLM calls."""

    def __init__(
        self,
        *,
        project_root: Path,
        repo_map: RepoMapSearcher | None = None,
        max_queries: int = 10,
        max_symbols: int = 12,
        max_snippets: int = 8,
    ) -> None:
        self.project_root = project_root
        self.repo_map = repo_map
        self.max_queries = max_queries
        self.max_symbols = max_symbols
        self.max_snippets = max_snippets

    def retrieve(self, user_request: str, *, task_template: str = "") -> ContextPack:
        queries = build_context_queries(user_request, limit=self.max_queries)
        missing: list[str] = []
        constraints: list[str] = [
            "Use ContextPack evidence before exploratory ReAct.",
            "Run extra search only for explicit missing_info.",
        ]

        raw_symbols: list[object] = []
        if self.repo_map is None:
            missing.append("repo_map unavailable")
        else:
            for query in queries:
                raw_symbols.extend(self.repo_map.search(query, limit=20))

        symbols = _dedupe_symbols(raw_symbols)[: self.max_symbols]
        relevant_files = _rank_files(symbols)
        snippets = [
            snippet
            for symbol in symbols[: self.max_snippets]
            if (snippet := _read_symbol_snippet(self.project_root, symbol)) is not None
        ]
        if not symbols:
            missing.append("no relevant symbols found")
        if symbols and not snippets:
            missing.append("no source snippets available")

        confidence = _confidence(symbols=symbols, snippets=snippets, missing=missing)
        return ContextPack(
            user_request=user_request,
            relevant_files=tuple(relevant_files),
            symbols=tuple(_to_context_symbol(symbol) for symbol in symbols),
            snippets=tuple(snippets),
            constraints=tuple(constraints),
            confidence=confidence,
            missing_info=tuple(missing),
            search_plan=tuple(_search_plans(relevant_files, queries)),
            metadata={
                "retriever": "repo_map_v1",
                "task_template": task_template,
                "queries": "|".join(queries),
            },
        )


def build_context_queries(text: str, *, limit: int = 10) -> list[str]:
    """Build a bounded batch of retrieval queries from user intent."""
    terms: list[str] = []

    def add(term: str) -> None:
        term = term.strip()
        if len(term) >= 2 and term not in terms:
            terms.append(term)

    domain_terms = {
        "登机牌": ("登机牌", "boarding", "boarding_pass"),
        "视图": ("视图", "view"),
        "查询": ("查询", "query"),
        "接口": ("接口", "api"),
        "订单": ("订单", "orders", "order"),
    }
    for trigger, expanded in domain_terms.items():
        if trigger in text:
            for term in expanded:
                add(term)

    for endpoint in re.findall(r"/[A-Za-z0-9_./{}-]+", text):
        add(endpoint.strip("/"))
        add(endpoint)

    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        add(token)
        for part in token.split("_"):
            add(part)

    for phrase in re.findall(r"[\u4e00-\u9fff]{2,}", text):
        if len(phrase) <= 8:
            add(phrase)

    return terms[:limit]


def _dedupe_symbols(raw_symbols: list[object]) -> list[object]:
    seen: set[tuple[str, str, int, int]] = set()
    out: list[object] = []
    for symbol in sorted(raw_symbols, key=lambda s: getattr(s, "score", 0.0), reverse=True):
        key = (
            str(getattr(symbol, "file_path", "")),
            str(getattr(symbol, "name", "")),
            int(getattr(symbol, "start_line", 0)),
            int(getattr(symbol, "end_line", 0)),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(symbol)
    return out


def _rank_files(symbols: list[object]) -> list[str]:
    scores: dict[str, float] = {}
    for symbol in symbols:
        file_path = str(getattr(symbol, "file_path", ""))
        if not file_path:
            continue
        scores[file_path] = scores.get(file_path, 0.0) + float(getattr(symbol, "score", 0.0))
    return [
        path
        for path, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
    ]


def _read_symbol_snippet(project_root: Path, symbol: object) -> ContextSnippet | None:
    rel = str(getattr(symbol, "file_path", ""))
    if not rel:
        return None
    start = max(1, int(getattr(symbol, "start_line", 1)) - 2)
    end = max(start, int(getattr(symbol, "end_line", start)) + 2)
    try:
        path = (project_root / rel).resolve()
        path.relative_to(project_root.resolve())
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return None
    if not lines:
        return None
    start = min(start, len(lines))
    end = min(end, len(lines))
    text = "\n".join(
        f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1)
    )
    return ContextSnippet(file_path=rel, start_line=start, end_line=end, text=text)


def _to_context_symbol(symbol: object) -> ContextSymbol:
    return ContextSymbol(
        file_path=str(getattr(symbol, "file_path", "")),
        name=str(getattr(symbol, "name", "")),
        kind=str(getattr(symbol, "kind", "")),
        start_line=int(getattr(symbol, "start_line", 0)),
        end_line=int(getattr(symbol, "end_line", 0)),
        signature=str(getattr(symbol, "signature", "")),
        score=float(getattr(symbol, "score", 0.0)),
    )


def _search_plans(files: list[str], queries: list[str]) -> list[SearchPlan]:
    if not files or not queries:
        return []
    grouped: dict[str, list[str]] = {}
    for file_path in files:
        module = _module_name(file_path)
        grouped.setdefault(module, []).append(file_path)
    return [
        SearchPlan(
            module=module,
            files=tuple(module_files[:5]),
            patterns=tuple(queries[:10]),
            globs=tuple(
                sorted({
                    f"*{Path(path).suffix}"
                    for path in module_files
                    if Path(path).suffix
                })
            ),
        )
        for module, module_files in list(grouped.items())[:4]
    ]


def _module_name(file_path: str) -> str:
    path = Path(file_path)
    if len(path.parts) > 1:
        return str(Path(*path.parts[:-1])).replace("\\", "/")
    return path.stem


def _confidence(
    *,
    symbols: list[object],
    snippets: list[ContextSnippet],
    missing: list[str],
) -> float:
    if missing and not symbols:
        return 0.15
    score = 0.25
    if symbols:
        score += 0.35
    if snippets:
        score += 0.30
    if len({getattr(symbol, "file_path", "") for symbol in symbols}) == 1:
        score += 0.10
    if missing:
        score -= 0.20
    return max(0.0, min(score, 1.0))
