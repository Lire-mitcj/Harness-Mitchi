from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Protocol

from src.agent.cursor_contracts import RetrievalResult, RetrievalSymbol
from src.agent.cursor_query_bridge import QueryBridgeResult


@dataclass(frozen=True, slots=True)
class FusionResult:
    final_context: tuple[str, ...]
    confidence: float
    retrieval: RetrievalResult
    rerank: dict[str, object] = field(default_factory=dict)

    @property
    def final_files(self) -> tuple[str, ...]:
        """Actual file set retained by Fusion after scoring and caps."""
        return self.retrieval.candidate_files


class CursorReranker(Protocol):
    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int,
    ) -> dict[int, float]: ...


@dataclass(frozen=True, slots=True)
class _FusionCandidate:
    kind: str
    document: str
    symbol: RetrievalSymbol | None = None
    file: str = ""


class CursorFusionEngine:
    """Deduplicate, rank, and cap raw retrieval candidates."""

    def __init__(
        self,
        *,
        reranker: CursorReranker | None = None,
        rerank_enabled: bool = False,
        rerank_top_n: int = 50,
    ) -> None:
        self.reranker = reranker
        self.rerank_enabled = rerank_enabled
        self.rerank_top_n = rerank_top_n

    @staticmethod
    def fuse(
        retrieval: RetrievalResult,
        bridge: QueryBridgeResult,
        *,
        max_files: int,
        max_symbols: int,
    ) -> RetrievalResult:
        terms = bridge.search_terms()
        file_hints = tuple(term.casefold() for term in bridge.file_hints)
        layer_hints = tuple(
            term.casefold()
            for term in (bridge.constraints or {}).get("layer_hint", ())
        )
        file_symbol_scores: dict[str, float] = {}
        for symbol in retrieval.symbols:
            file_symbol_scores[symbol.file] = max(
                file_symbol_scores.get(symbol.file, 0.0),
                symbol.score,
            )

        def file_score(path: str) -> tuple[float, str]:
            normalized = path.casefold()
            score = file_symbol_scores.get(path, 0.0) * 10
            score += 6 * sum(hint in normalized for hint in file_hints)
            score += 4 * sum(hint in normalized for hint in layer_hints)
            score += 3 * sum(term.casefold() in normalized for term in terms)
            return (-score, normalized)

        def symbol_score(symbol: RetrievalSymbol) -> tuple[float, str, int]:
            name = symbol.name.casefold()
            score = symbol.score * 100
            framework_tokens = {
                "query",
                "api",
                "get",
                "post",
                "put",
                "delete",
                "route",
                "endpoint",
                "sql",
                "db",
            }
            if name in framework_tokens:
                score *= 0.05
            score += 4 * sum(term.casefold() in name for term in terms)
            score += 2 * sum(hint in symbol.file.casefold() for hint in file_hints)
            score += 2 * sum(hint in symbol.file.casefold() for hint in layer_hints)
            return (-score, symbol.file.casefold(), symbol.start_line)

        files = sorted(dict.fromkeys(retrieval.files), key=file_score)
        symbols = sorted(dict.fromkeys(retrieval.symbols), key=symbol_score)
        return RetrievalResult(
            files=tuple(files[:max_files]),
            symbols=tuple(symbols[:max_symbols]),
        )

    @classmethod
    def decide(
        cls,
        retrieval: RetrievalResult,
        bridge: QueryBridgeResult,
        *,
        max_files: int,
        max_symbols: int,
        top_k: int = 8,
    ) -> FusionResult:
        fused = cls.fuse(
            retrieval,
            bridge,
            max_files=max_files,
            max_symbols=max_symbols,
        )
        return cls._finalize(fused, bridge, top_k=top_k)

    async def decide_async(
        self,
        retrieval: RetrievalResult,
        bridge: QueryBridgeResult,
        *,
        max_files: int,
        max_symbols: int,
        top_k: int = 8,
        user_intent: str = "",
    ) -> FusionResult:
        fused = await self._rerank_or_fuse(
            retrieval,
            bridge,
            max_files=max_files,
            max_symbols=max_symbols,
            user_intent=user_intent,
        )
        return self._finalize(fused[0], bridge, top_k=top_k, rerank=fused[1])

    async def _rerank_or_fuse(
        self,
        retrieval: RetrievalResult,
        bridge: QueryBridgeResult,
        *,
        max_files: int,
        max_symbols: int,
        user_intent: str,
    ) -> tuple[RetrievalResult, dict[str, object]]:
        local = self.fuse(
            retrieval,
            bridge,
            max_files=max_files,
            max_symbols=max_symbols,
        )
        if not self.rerank_enabled or self.reranker is None:
            return local, {"enabled": False, "status": "disabled"}

        saturated = self.fuse(
            retrieval,
            bridge,
            max_files=min(max(max_files * 3, max_files), 50),
            max_symbols=min(max(max_symbols * 4, max_symbols), 50),
        )
        candidates = _rerank_candidates(saturated)
        if not candidates:
            return local, {"enabled": True, "status": "skipped", "reason": "no_candidates"}

        documents = [candidate.document for candidate in candidates]
        query = _rerank_query(bridge, user_intent)
        started = time.perf_counter()
        scores = await self.reranker.rerank(
            query,
            documents,
            top_n=min(self.rerank_top_n, len(documents)),
        )
        duration_ms = round((time.perf_counter() - started) * 1000, 1)
        summary: dict[str, object] = {
            "enabled": True,
            "status": "applied" if scores else "fallback",
            "model": str(getattr(self.reranker, "model", "custom")),
            "candidate_count": len(candidates),
            "scored_count": len(scores),
            "duration_ms": duration_ms,
        }
        if not scores:
            return local, summary

        symbol_scores: dict[RetrievalSymbol, float] = {}
        file_scores: dict[str, float] = {}
        for index, candidate in enumerate(candidates):
            if index not in scores:
                continue
            if candidate.symbol is not None:
                symbol_scores[candidate.symbol] = scores[index]
            else:
                file_scores[candidate.file] = scores[index]

        symbols = sorted(
            saturated.symbols,
            key=lambda symbol: (
                -symbol_scores.get(symbol, -1.0),
                -symbol.score,
                symbol.file.casefold(),
                symbol.start_line,
            ),
        )
        symbols = _apply_slot_defense(symbols, bridge, user_intent, max_symbols)
        files = _reranked_files(saturated, symbols, file_scores, max_files)
        selected = tuple(symbols[:max_symbols])
        summary["top_symbols"] = [
            {
                "file": symbol.file,
                "symbol": symbol.name,
                "score": round(symbol_scores.get(symbol, 0.0), 4),
            }
            for symbol in selected[:5]
        ]
        return RetrievalResult(files=files, symbols=selected), summary

    @classmethod
    def _finalize(
        cls,
        fused: RetrievalResult,
        bridge: QueryBridgeResult,
        *,
        top_k: int,
        rerank: dict[str, object] | None = None,
    ) -> FusionResult:
        contexts: list[str] = []
        seen: set[str] = set()
        is_destructive = bridge.intent in ("modify", "debug")
        if is_destructive:
            for symbol in fused.symbols:
                if str(getattr(symbol, "kind", "")).startswith("ddl_"):
                    continue
                item = f"{symbol.file}:{symbol.name}:{symbol.start_line}-{symbol.end_line}"
                if item in seen:
                    continue
                contexts.append(f"FOCUS:{item}")
                seen.add(item)
                if len(contexts) >= 2:
                    break

        for symbol in fused.symbols:
            item = f"{symbol.file}:{symbol.name}:{symbol.start_line}-{symbol.end_line}"
            if item not in seen:
                seen.add(item)
                # DDL is probabilistic Layer-2 evidence.  Never promote it to
                # an edit anchor merely because it scored highly in retrieval.
                contexts.append(item)
            if len(contexts) >= top_k:
                break

        for path in fused.files:
            if len(contexts) >= top_k:
                break
            if path in seen:
                continue
            seen.add(path)
            item = f"{path}:FILE:0-0"
            contexts.append(item)

        confidence = _confidence(fused)
        return FusionResult(
            final_context=tuple(contexts[:top_k]),
            confidence=confidence,
            retrieval=fused,
            rerank=dict(rerank or {}),
        )


def _rerank_query(bridge: QueryBridgeResult, user_intent: str) -> str:
    signals = list(bridge.search_terms(limit=24))
    if user_intent:
        return f"{user_intent}\nIntent terms: {', '.join(signals)}"
    return ", ".join(signals)


def _rerank_candidates(retrieval: RetrievalResult) -> list[_FusionCandidate]:
    candidates: list[_FusionCandidate] = []
    for symbol in retrieval.symbols:
        doc = (
            f"file: {symbol.file}\n"
            f"symbol: {symbol.name}\n"
            f"kind: {symbol.kind or '-'}\n"
            f"lines: {symbol.start_line}-{symbol.end_line}\n"
            f"tables: {', '.join(symbol.tables_referenced) or '-'}\n"
            f"calls: {', '.join(symbol.calls) or '-'}\n"
            f"reasons: {', '.join(symbol.reasons) or '-'}"
        )
        candidates.append(_FusionCandidate("symbol", doc, symbol=symbol))
    for file in retrieval.files:
        candidates.append(_FusionCandidate("file", f"file: {file}", file=file))
    return candidates


def _reranked_files(
    retrieval: RetrievalResult,
    symbols: list[RetrievalSymbol],
    file_scores: dict[str, float],
    max_files: int,
) -> tuple[str, ...]:
    ordered: list[str] = []
    for symbol in symbols:
        if symbol.file not in ordered:
            ordered.append(symbol.file)
    for file in sorted(
        retrieval.files,
        key=lambda path: (-file_scores.get(path, -1.0), path.casefold()),
    ):
        if file not in ordered:
            ordered.append(file)
    return tuple(ordered[:max_files])


def _apply_slot_defense(
    symbols: list[RetrievalSymbol],
    bridge: QueryBridgeResult,
    user_intent: str,
    max_symbols: int,
) -> list[RetrievalSymbol]:
    if max_symbols <= 0:
        return []

    protected: list[RetrievalSymbol] = []
    if _is_sql_view_task(bridge, user_intent):
        query_logic = next(
            (symbol for symbol in symbols if _is_query_logic_symbol(symbol)),
            None,
        )
        if query_logic is not None:
            protected.append(query_logic)
        ddl_view = next(
            (
                symbol
                for symbol in symbols
                if str(getattr(symbol, "kind", "")) == "ddl_view"
                and symbol not in protected
            ),
            None,
        )
        if ddl_view is not None:
            protected.append(ddl_view)

    selected: list[RetrievalSymbol] = []
    per_file: dict[str, int] = {}
    cap_per_file = max(2, max_symbols // 2)

    def add(symbol: RetrievalSymbol, *, ignore_cap: bool = False) -> None:
        if symbol in selected or len(selected) >= max_symbols:
            return
        count = per_file.get(symbol.file, 0)
        if not ignore_cap and count >= cap_per_file and not _is_query_logic_symbol(symbol):
            return
        selected.append(symbol)
        per_file[symbol.file] = count + 1

    for symbol in protected:
        add(symbol, ignore_cap=True)
    for symbol in symbols:
        add(symbol)
    for symbol in symbols:
        add(symbol, ignore_cap=True)
    return selected


def _is_sql_view_task(bridge: QueryBridgeResult, user_intent: str) -> bool:
    text = " ".join((*bridge.search_terms(limit=32), user_intent)).casefold()
    wants_sql_or_view = any(
        token in text for token in ("sql", "view", "视图", "查询", "数据库")
    )
    wants_logic = any(
        token in text for token in ("order", "订单", "query", "select", "查询")
    )
    return wants_sql_or_view and wants_logic


def _is_query_logic_symbol(symbol: RetrievalSymbol) -> bool:
    name = symbol.name.casefold()
    kind = str(getattr(symbol, "kind", "")).casefold()
    return (
        kind.startswith("dml_")
        or bool(symbol.tables_referenced)
        or "sql" in name
        or "query" in name
        or "select" in name
        or "订单" in name
    )


def _confidence(retrieval: RetrievalResult) -> float:
    if not retrieval.symbols and not retrieval.files:
        return 0.0
    if retrieval.symbols:
        best = max(symbol.score for symbol in retrieval.symbols)
        if best > 0.0:
            return round(min(0.99, max(0.35, best)), 4)
    return 0.55 if retrieval.files else 0.0


def _alignment_signal(retrieval: RetrievalResult) -> str:
    sql_view = next(
        (
            symbol
            for symbol in retrieval.symbols
            if str(getattr(symbol, "kind", "")) == "ddl_view"
        ),
        None,
    )
    py_query = next(
        (
            symbol
            for symbol in retrieval.symbols
            if str(getattr(symbol, "kind", "")) == "dml_select"
        ),
        None,
    )
    if sql_view is None or py_query is None:
        return ""

    view_tables = {table.casefold() for table in sql_view.tables_referenced}
    query_tables = {table.casefold() for table in py_query.tables_referenced}
    shared_tables = sorted(view_tables & query_tables)
    if not shared_tables:
        return ""

    all_tables = view_tables | query_tables
    overlap_score = len(shared_tables) / len(all_tables) if all_tables else 0.0
    if overlap_score >= 0.95:
        confidence = "high"
        reason = "all referenced tables overlap"
    elif overlap_score >= 0.50:
        confidence = "medium"
        reason = "partial overlap"
    else:
        confidence = "low"
        reason = "limited overlap"

    return (
        "\n[SCHEMA_EVIDENCE]\n"
        f"python_logic: {py_query.name}\n"
        f"sql_view: {sql_view.name}\n"
        f"shared_tables: {shared_tables}\n"
        f"overlap_score: {overlap_score:.2f}\n"
        f"confidence: {confidence}\n"
        f"reason: {reason}"
    )
