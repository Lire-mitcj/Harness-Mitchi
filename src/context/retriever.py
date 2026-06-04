from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Protocol

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
        return self.build(
            user_request=user_request,
            task_template=task_template,
        )

    def build(
        self,
        *,
        user_request: str,
        task_template: str = "",
        current_files: tuple[str, ...] = (),
        open_files: tuple[str, ...] = (),
        recent_files: tuple[str, ...] = (),
        previous_handoff: dict[str, Any] | None = None,
        mode: str | None = None,
        max_input_tokens: int = 24_000,
        reserved_output_tokens: int = 4_000,
    ) -> ContextPack:
        queries = build_context_queries(user_request, limit=self.max_queries)
        missing: list[str] = []
        constraints: list[str] = [
            "Start from ContextPack evidence and focused_snippets.",
            "Do not edit files outside focused_snippets unless explicitly authorized by tool_policy.",
            "Run extra search only for explicit missing_info or low confidence.",
            "Do not repeat known_negatives.",
            "Prefer minimal patches in existing project style.",
        ]

        raw_symbols: list[object] = []
        if self.repo_map is None:
            missing.append("repo_map unavailable")
        else:
            for query in queries:
                raw_symbols.extend(self.repo_map.search(query, limit=20))

        focused_symbols = _dedupe_symbols(raw_symbols)[: self.max_symbols]
        graph_edges = _graph_call_edges(self.repo_map, focused_symbols)
        symbols = _dedupe_symbols(
            focused_symbols + [dst for _src, dst in graph_edges]
        )[: self.max_symbols]
        explicit_files = _explicit_files(user_request, self.project_root)
        file_scores = _score_files(
            symbols=symbols,
            queries=queries,
            explicit_files=explicit_files,
            current_files=tuple(_norm_path(p) for p in current_files + open_files),
            recent_files=tuple(_norm_path(p) for p in recent_files),
            repo_map=self.repo_map,
            project_root=self.project_root,
        )
        relevant_files = [path for path, _score in file_scores[:12]]
        test_config_files = _related_test_config_files(self.project_root, relevant_files)
        for path in test_config_files:
            if path not in relevant_files:
                relevant_files.append(path)
        snippets = _build_focused_snippets(
            project_root=self.project_root,
            symbols=symbols,
            files=relevant_files,
            queries=queries,
            max_snippets=self.max_snippets,
            max_chars=_token_chars(max_input_tokens - reserved_output_tokens),
        )
        evidence = _build_evidence(
            symbols=symbols,
            snippets=snippets,
            file_scores=file_scores,
            explicit_files=explicit_files,
            queries=queries,
        )
        evidence = _merge_prior_evidence(evidence, previous_handoff)
        known_negatives = _known_negatives(
            queries=queries,
            symbols=symbols,
            snippets=snippets,
            previous_handoff=previous_handoff,
        )
        call_chain = _call_chain(symbols, snippets, graph_edges=graph_edges)
        if not symbols:
            missing.append("no relevant symbols found")
        if symbols and not snippets:
            missing.append("no source snippets available")

        confidence = _confidence(symbols=symbols, snippets=snippets, missing=missing)
        inferred_mode = mode or _infer_mode(user_request, task_template)
        return ContextPack(
            user_request=user_request,
            task={
                "objective": user_request,
                "mode": inferred_mode,
                "risk": _risk_for_mode(inferred_mode),
            },
            candidate_files=tuple(
                {
                    "file": path,
                    "score": round(score, 4),
                    "reasons": _file_reasons(
                        path,
                        symbols=symbols,
                        explicit_files=explicit_files,
                        current_files=tuple(_norm_path(p) for p in current_files + open_files),
                        recent_files=tuple(_norm_path(p) for p in recent_files),
                        repo_map=self.repo_map,
                    ),
                }
                for path, score in file_scores[:12]
            ),
            candidate_symbols=tuple(
                {
                    "file": str(getattr(symbol, "file_path", "")),
                    "symbol": str(getattr(symbol, "name", "")),
                    "kind": str(getattr(symbol, "kind", "")),
                    "start_line": int(getattr(symbol, "start_line", 0)),
                    "end_line": int(getattr(symbol, "end_line", 0)),
                    "score": float(getattr(symbol, "score", 0.0)),
                }
                for symbol in symbols
            ),
            relevant_files=tuple(relevant_files),
            symbols=tuple(_to_context_symbol(symbol) for symbol in symbols),
            repo_map=tuple(_repo_map_entries(symbols, relevant_files)),
            snippets=tuple(snippets),
            focused_snippets=tuple(snippets),
            evidence=tuple(evidence),
            known_negatives=tuple(known_negatives),
            call_chain=tuple(call_chain),
            constraints=tuple(constraints),
            tool_policy={
                "allowed_tools": _allowed_tools_for_mode(inferred_mode),
                "denied_tools": ["delete_file"],
                "max_tool_calls": 6 if inferred_mode == "edit" else 2,
            },
            budget={
                "max_input_tokens": max_input_tokens,
                "reserved_output_tokens": reserved_output_tokens,
                "snippet_chars": _token_chars(max_input_tokens - reserved_output_tokens),
                "context_search_max_results": 80,
            },
            confidence=confidence,
            missing_info=tuple(missing),
            search_plan=tuple(_search_plans(relevant_files, queries)),
            metadata={
                "retriever": "context_builder_v1",
                "task_template": task_template,
                "queries": "|".join(queries),
                "previous_next_focus": "|".join(
                    _prior_text_items(previous_handoff, "next_focus")
                ),
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


def _score_files(
    *,
    symbols: list[object],
    queries: list[str],
    explicit_files: tuple[str, ...],
    current_files: tuple[str, ...],
    recent_files: tuple[str, ...],
    repo_map: RepoMapSearcher | None,
    project_root: Path,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    reasons: dict[str, set[str]] = {}

    def add(path: str, value: float, reason: str) -> None:
        rel = _norm_path(path)
        if not rel:
            return
        scores[rel] = scores.get(rel, 0.0) + value
        reasons.setdefault(rel, set()).add(reason)

    for symbol in symbols:
        path = str(getattr(symbol, "file_path", ""))
        add(path, 0.45 + float(getattr(symbol, "score", 0.0)), "symbol_match")

    for path in explicit_files:
        add(path, 1.5, "explicit_file")
    for path in current_files:
        add(path, 0.4, "open_file")
    for path in recent_files:
        add(path, 0.55, "recent_edit")

    file_scores = getattr(repo_map, "file_scores", {}) if repo_map is not None else {}
    for path, score in getattr(file_scores, "items", lambda: [])():
        add(str(path), min(float(score) * 5.0, 0.5), "repo_rank")

    for path in list(scores):
        p = project_root / path
        if Path(path).name in {"main.py", "app.py", "server.py"}:
            add(path, 0.25, "entry_file")
        if path.startswith("tests/") or Path(path).name.startswith("test_"):
            add(path, -0.15, "test_file")
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        if size > 80_000:
            add(path, -0.25, "large_file")
        if any(term.lower() in path.lower() for term in queries):
            add(path, 0.35, "filename_match")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [(path, max(0.0, min(score, 1.0))) for path, score in ranked if score > 0]


def _explicit_files(text: str, project_root: Path) -> tuple[str, ...]:
    hits: list[str] = []
    for raw in re.findall(r"[\w./-]+\.(?:py|sql|js|ts|tsx|jsx|json|toml|yaml|yml|md)", text):
        rel = _norm_path(raw)
        try:
            path = (project_root / rel).resolve()
            path.relative_to(project_root.resolve())
        except (OSError, ValueError):
            continue
        if path.exists() and rel not in hits:
            hits.append(rel)
    return tuple(hits)


def _build_focused_snippets(
    *,
    project_root: Path,
    symbols: list[object],
    files: list[str],
    queries: list[str],
    max_snippets: int,
    max_chars: int,
) -> list[ContextSnippet]:
    snippets: list[ContextSnippet] = []
    seen: set[tuple[str, int, int]] = set()
    used_chars = 0

    def add(snippet: ContextSnippet | None) -> None:
        nonlocal used_chars
        if snippet is None:
            return
        key = (snippet.file_path, snippet.start_line, snippet.end_line)
        if key in seen:
            return
        if used_chars + len(snippet.text) > max_chars:
            return
        seen.add(key)
        used_chars += len(snippet.text)
        snippets.append(snippet)

    for symbol in symbols:
        if len(snippets) >= max_snippets:
            break
        add(_read_symbol_snippet(project_root, symbol))

    pattern = _context_pattern(queries)
    for rel in files:
        if len(snippets) >= max_snippets:
            break
        for snippet in _read_line_hit_snippets(project_root, rel, pattern):
            if len(snippets) >= max_snippets:
                break
            add(snippet)

    for rel in files:
        if len(snippets) >= max_snippets:
            break
        if _is_config_file(rel):
            add(_read_file_snippet(project_root, rel, max_lines=120))
    return snippets


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


def _read_line_hit_snippets(
    project_root: Path,
    rel: str,
    pattern: re.Pattern[str],
    *,
    padding: int = 60,
    max_hits: int = 2,
) -> list[ContextSnippet]:
    try:
        path = (project_root / rel).resolve()
        path.relative_to(project_root.resolve())
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return []
    out: list[ContextSnippet] = []
    for idx, line in enumerate(lines, start=1):
        if not pattern.search(line):
            continue
        start = max(1, idx - padding)
        end = min(len(lines), idx + padding)
        text = "\n".join(
            f"{line_no}: {lines[line_no - 1]}" for line_no in range(start, end + 1)
        )
        out.append(ContextSnippet(file_path=rel, start_line=start, end_line=end, text=text, source="line_hit"))
        if len(out) >= max_hits:
            break
    return out


def _read_file_snippet(project_root: Path, rel: str, *, max_lines: int) -> ContextSnippet | None:
    try:
        path = (project_root / rel).resolve()
        path.relative_to(project_root.resolve())
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except (OSError, ValueError):
        return None
    if not lines:
        return None
    end = min(len(lines), max_lines)
    text = "\n".join(f"{line_no}: {lines[line_no - 1]}" for line_no in range(1, end + 1))
    return ContextSnippet(file_path=rel, start_line=1, end_line=end, text=text, source="config")


def _context_pattern(queries: list[str]) -> re.Pattern[str]:
    terms = [
        re.escape(term)
        for term in queries
        if len(term.strip()) >= 2
        and term.lower() not in {"query", "api", "接口", "查询"}
    ]
    if not terms:
        terms = [re.escape(term) for term in queries if term.strip()]
    if not terms:
        terms = [r"$^"]
    return re.compile("|".join(terms), re.IGNORECASE)


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


def _build_evidence(
    *,
    symbols: list[object],
    snippets: list[ContextSnippet],
    file_scores: list[tuple[str, float]],
    explicit_files: tuple[str, ...],
    queries: list[str],
) -> list[dict[str, Any]]:
    evidence: list[dict[str, Any]] = []
    for symbol in symbols[:12]:
        evidence.append({
            "type": "symbol_match",
            "file": str(getattr(symbol, "file_path", "")),
            "symbol": str(getattr(symbol, "name", "")),
            "start_line": int(getattr(symbol, "start_line", 0)),
            "end_line": int(getattr(symbol, "end_line", 0)),
            "reason": "matches user request or expanded query",
        })
    for snippet in snippets[:12]:
        evidence.append({
            "type": "snippet",
            "file": snippet.file_path,
            "start_line": snippet.start_line,
            "end_line": snippet.end_line,
            "reason": snippet.source,
        })
    for path in explicit_files:
        evidence.append({"type": "explicit_file", "file": path, "reason": "mentioned by user"})
    for path, score in file_scores[:8]:
        evidence.append({"type": "file_score", "file": path, "score": round(score, 4)})
    if queries:
        evidence.append({"type": "query_expansion", "queries": list(queries)})
    return _dedupe_dicts(evidence, limit=30)


def _known_negatives(
    *,
    queries: list[str],
    symbols: list[object],
    snippets: list[ContextSnippet],
    previous_handoff: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    negatives: list[dict[str, Any]] = []
    matched = {
        str(getattr(symbol, "name", "")).lower()
        for symbol in symbols
    } | {snippet.text.lower() for snippet in snippets}
    for query in queries:
        if not any(query.lower() in item for item in matched):
            negatives.append({
                "query": query,
                "scope": "repo_map/focused_snippets",
                "result": "no direct focused match",
            })
    prior = previous_handoff or {}
    for item in prior.get("known_negatives", []) if isinstance(prior, dict) else []:
        if isinstance(item, dict):
            negatives.append(dict(item))
    return _dedupe_dicts(negatives, limit=12)


def _merge_prior_evidence(
    evidence: list[dict[str, Any]],
    previous_handoff: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    merged = list(evidence)
    prior = previous_handoff or {}
    for item in prior.get("facts", []) if isinstance(prior, dict) else []:
        if isinstance(item, dict):
            merged.append({"type": "prior_fact", **item})
        elif str(item).strip():
            merged.append({"type": "prior_fact", "fact": str(item)})
    for item in prior.get("evidence", []) if isinstance(prior, dict) else []:
        if isinstance(item, dict):
            merged.append({"type": "prior_handoff", **item})
    return _dedupe_dicts(merged, limit=40)


def _prior_text_items(previous_handoff: dict[str, Any] | None, key: str) -> list[str]:
    prior = previous_handoff or {}
    if not isinstance(prior, dict):
        return []
    out: list[str] = []
    for item in prior.get(key, []) or []:
        if isinstance(item, dict):
            value = item.get("focus") or item.get("fact") or item.get("path") or item.get("file")
            if value:
                out.append(str(value))
        elif str(item).strip():
            out.append(str(item))
    return out[:20]


def _graph_call_edges(
    repo_map: RepoMapSearcher | None,
    symbols: list[object],
) -> list[tuple[object, object]]:
    if repo_map is None:
        return []
    expand = getattr(repo_map, "expand_symbol_edges", None)
    if not callable(expand):
        return []
    symbol_ids = [
        str(getattr(symbol, "symbol_id", ""))
        for symbol in symbols
        if str(getattr(symbol, "symbol_id", ""))
    ]
    if not symbol_ids:
        return []
    try:
        return list(expand(symbol_ids, depth=2, limit=20))
    except TypeError:
        return []


def _call_chain(
    symbols: list[object],
    snippets: list[ContextSnippet],
    *,
    graph_edges: list[tuple[object, object]] = (),
) -> list[str]:
    edges: list[str] = []
    for src, dst in graph_edges:
        src_name = str(getattr(src, "name", ""))
        dst_name = str(getattr(dst, "name", ""))
        if not src_name or not dst_name:
            continue
        edge = f"{src_name} -> {dst_name}"
        if edge not in edges:
            edges.append(edge)
        if len(edges) >= 12:
            return edges
    if not symbols or not snippets:
        return edges
    symbol_names = [
        str(getattr(symbol, "name", ""))
        for symbol in symbols
        if str(getattr(symbol, "name", ""))
    ]
    for symbol in symbols[:8]:
        src = str(getattr(symbol, "name", ""))
        if not src:
            continue
        rel = str(getattr(symbol, "file_path", ""))
        text = "\n".join(snippet.text for snippet in snippets if snippet.file_path == rel)
        for dst in symbol_names:
            if dst == src:
                continue
            if re.search(rf"\b{re.escape(dst)}\b", text):
                edge = f"{src} -> {dst}"
                if edge not in edges:
                    edges.append(edge)
            if len(edges) >= 12:
                return edges
    return edges


def _repo_map_entries(symbols: list[object], files: list[str]) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    by_file: dict[str, list[str]] = {}
    for symbol in symbols:
        by_file.setdefault(str(getattr(symbol, "file_path", "")), []).append(
            str(getattr(symbol, "name", ""))
        )
    for rel in files[:12]:
        entries.append({
            "file": rel,
            "symbols": [name for name in by_file.get(rel, []) if name][:12],
        })
    return entries


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


def _related_test_config_files(project_root: Path, relevant_files: list[str]) -> list[str]:
    out: list[str] = []
    for name in ("pyproject.toml", "requirements.txt", "package.json", "pytest.ini"):
        if (project_root / name).is_file():
            out.append(name)
    for rel in relevant_files[:5]:
        path = Path(rel)
        if path.suffix != ".py":
            continue
        stem = path.stem
        candidates = [
            Path("tests") / f"test_{stem}.py",
            Path("tests") / path.parent / f"test_{stem}.py",
            path.parent / f"test_{stem}.py",
        ]
        for cand in candidates:
            norm = _norm_path(str(cand))
            if (project_root / norm).is_file() and norm not in out:
                out.append(norm)
    return out[:6]


def _file_reasons(
    path: str,
    *,
    symbols: list[object],
    explicit_files: tuple[str, ...],
    current_files: tuple[str, ...],
    recent_files: tuple[str, ...],
    repo_map: RepoMapSearcher | None,
) -> list[str]:
    reasons: list[str] = []
    if path in explicit_files:
        reasons.append("explicit_file")
    if path in current_files:
        reasons.append("open_file")
    if path in recent_files:
        reasons.append("recent_edit")
    if any(str(getattr(symbol, "file_path", "")) == path for symbol in symbols):
        reasons.append("symbol_match")
    file_scores = getattr(repo_map, "file_scores", {}) if repo_map is not None else {}
    if path in file_scores:
        reasons.append("repo_rank")
    if Path(path).name in {"main.py", "app.py", "server.py"}:
        reasons.append("entry_file")
    if path.startswith("tests/") or Path(path).name.startswith("test_"):
        reasons.append("test_file")
    return reasons or ["candidate"]


def _infer_mode(user_request: str, task_template: str) -> str:
    text = f"{user_request} {task_template}".lower()
    if any(term in text for term in ("修改", "改成", "fix", "edit", "change", "patch")):
        return "edit"
    if any(term in text for term in ("测试", "verify", "run", "验证")):
        return "verify"
    return "diagnose"


def _risk_for_mode(mode: str) -> str:
    if mode == "edit":
        return "medium"
    if mode == "verify":
        return "low"
    return "low"


def _allowed_tools_for_mode(mode: str) -> list[str]:
    if mode == "edit":
        return ["context_search", "edit_file", "write_file", "shell_exec"]
    if mode == "verify":
        return ["shell_exec", "context_search"]
    return ["context_search"]


def _token_chars(tokens: int) -> int:
    return max(2_000, min(tokens * 4, 60_000))


def _is_config_file(path: str) -> bool:
    return Path(path).name in {
        "pyproject.toml",
        "requirements.txt",
        "package.json",
        "pytest.ini",
    } or Path(path).suffix in {".json", ".toml", ".yaml", ".yml"}


def _norm_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def _dedupe_dicts(items: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for item in items:
        key = str(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
        if len(out) >= limit:
            break
    return out


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
