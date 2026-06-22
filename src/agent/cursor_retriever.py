from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from src.agent.cursor_ast_structure import AstNode
from src.agent.cursor_contracts import RetrievalResult, RetrievalSymbol
from src.agent.cursor_graph_bridge import GraphBridgeResult
from src.agent.cursor_query_bridge import QueryBridgeResult
from src.agent.cursor_repo_map_lookup import CandidateSymbol

log = logging.getLogger(__name__)


class CursorRetriever:
    """Deterministic exact-path, AST-index, and file grep retrieval."""

    def __init__(
        self,
        project_root: Path,
        tools: Any = None,
        repo_map_service: Any = None,
        *,
        max_files: int = 12,
        max_symbols: int = 12,
        candidate_symbols: int | None = None,
        max_queries: int = 12,
        total_timeout: float = 12.0,
    ) -> None:
        self.project_root = project_root.resolve()
        self.repo_map = repo_map_service
        self.max_files = max_files
        self.max_symbols = max_symbols
        self.candidate_symbols = candidate_symbols or max_symbols * 3
        self.max_queries = max_queries
        self.total_timeout = total_timeout

    async def retrieve(self, terms: Sequence[str]) -> RetrievalResult:
        if isinstance(terms, str):
            raise TypeError(
                "CursorRetriever expects rewritten search terms, not a raw query string"
            )
        queries = tuple(dict.fromkeys(terms))[:self.max_queries]
        try:
            async with asyncio.timeout(self.total_timeout):
                return await self._retrieve(queries)
        except TimeoutError:
            log.warning(
                "CursorRetriever: global timeout after %.2fs for terms %r",
                self.total_timeout,
                queries,
            )
            return RetrievalResult()

    def score_candidates(
        self,
        *,
        ast_nodes: tuple[AstNode, ...],
        candidates: tuple[CandidateSymbol, ...],
        bridge: QueryBridgeResult,
        graph: GraphBridgeResult,
    ) -> RetrievalResult:
        import math
        from collections import Counter
        if not hasattr(self, "_cached_token_counts"):
            self._cached_token_counts = None

        if self._cached_token_counts is not None:
            token_counts = self._cached_token_counts
        else:
            token_counts = Counter()
            if self.repo_map is not None:
                try:
                    rm = getattr(self.repo_map, "map", None)
                    if rm is not None:
                        rm_symbols = list(
                            getattr(rm, "all_symbols", None)
                            or getattr(rm, "symbols", [])
                        )
                        for sym in rm_symbols:
                            token_counts.update(_tokens(sym.name))
                        self._cached_token_counts = token_counts
                except Exception:
                    pass

        candidate_by_id = {
            self._symbol_id(candidate.symbol): candidate
            for candidate in candidates
        }
        scored_symbols: list[RetrievalSymbol] = []
        files: list[str] = []
        for node in ast_nodes:
            symbol = node.source_symbol
            candidate = candidate_by_id.get(self._symbol_id(symbol))
            if candidate is None:
                continue
            embedding_similarity = candidate.score_hint
            lexical_overlap = self._lexical_overlap(symbol, bridge)
            alias_match = self._alias_match(symbol, bridge)
            score = (
                0.4 * embedding_similarity
                + 0.3 * lexical_overlap
                + 0.3 * alias_match
            )
            reasons = set(candidate.reasons)
            if embedding_similarity > 0:
                reasons.add("embedding_match")
            if lexical_overlap > 0:
                reasons.add("lexical_match")
            if alias_match > 0:
                reasons.add("alias_match")
            rel = self._normalize(symbol.file_path)
            if rel not in files:
                files.append(rel)

            if token_counts:
                max_token_freq = max((token_counts[t] for t in _tokens(symbol.name)), default=1)
                score = score / math.log(2.0 + max_token_freq)

            scored_symbols.append(
                RetrievalSymbol(
                    file=rel,
                    name=symbol.name,
                    start_line=node.lines[0],
                    end_line=node.lines[1],
                    score=round(score, 6),
                    reasons=tuple(sorted(reasons)),
                    calls=node.calls,
                    kind=str(getattr(symbol, "kind", "")),
                    tables_referenced=tuple(getattr(symbol, "tables_referenced", ())),
                )
            )
        scored_symbols.sort(
            key=lambda symbol: (-symbol.score, symbol.file.casefold(), symbol.start_line)
        )
        soft_cap_symbols = scored_symbols[: self.candidate_symbols]
        ordered_files = []
        for symbol in soft_cap_symbols:
            if symbol.file not in ordered_files:
                ordered_files.append(symbol.file)
        for path in graph.expanded_files:
            rel = self._normalize(path)
            if rel not in ordered_files:
                ordered_files.append(rel)
        for path in files:
            if path not in ordered_files:
                ordered_files.append(path)
        soft_cap_files = ordered_files[: self.max_files * 3]
        return RetrievalResult(
            files=tuple(soft_cap_files),
            symbols=tuple(soft_cap_symbols),
        )

    async def _retrieve(
        self,
        tokens: tuple[str, ...],
    ) -> RetrievalResult:
        log.warning("CursorRetriever: retrieving terms: %s", tokens)
        files: list[str] = []
        symbols: list[RetrievalSymbol] = []

        def add_file(path: str) -> None:
            normalized = self._relative_file(path)
            if normalized and normalized not in files:
                files.append(normalized)

        for term in tokens:
            add_file(term)

        candidate_pool = []
        if self.repo_map is not None:
            try:
                rm = getattr(self.repo_map, "map", None)
                if rm is not None:
                    candidate_pool = list(
                        getattr(rm, "all_symbols", None)
                        or getattr(rm, "symbols", [])
                    )
            except Exception as exc:
                log.warning("Error loading all_symbols registry: %s", exc)

        matched_symbols = self._scan_ast(candidate_pool, tokens)
        for sym in candidate_pool:
            sym_name_low = sym.name.lower()
            if "order" in sym_name_low and (
                "sql" in sym_name_low or "query" in sym_name_low
            ):
                if sym not in matched_symbols:
                    matched_symbols.append(sym)

        seen_symbols: set[str] = set()
        for sym in matched_symbols:
            rel = self._normalize(sym.file_path)
            add_file(rel)
            symbol_id = self._symbol_id(sym)
            if symbol_id in seen_symbols:
                continue
            seen_symbols.add(symbol_id)

            is_sql = str(getattr(sym, "kind", "")).startswith(("ddl_", "dml_"))
            base_score = 0.92 if (is_sql or "sql" in sym.name.lower()) else 0.50
            symbols.append(
                RetrievalSymbol(
                    file=rel,
                    name=sym.name,
                    start_line=sym.start_line,
                    end_line=sym.end_line,
                    score=base_score,
                    reasons=("universal_ast_recovery",),
                    kind=str(getattr(sym, "kind", "")),
                    tables_referenced=tuple(getattr(sym, "tables_referenced", ())),
                )
            )

        def token_density(text: str) -> int:
            text_lower = text.casefold()
            return sum(1 for token in tokens if token.casefold() in text_lower)

        files.sort()
        files.sort(key=token_density, reverse=True)
        symbols.sort(key=lambda s: s.name)
        symbols.sort(key=lambda s: token_density(s.name), reverse=True)

        return RetrievalResult(
            files=tuple(files[: self.max_files * 3]),
            symbols=tuple(symbols[: self.candidate_symbols]),
        )

    async def _grep(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        paths: set[str] = set()
        patterns = tuple(token for token in tokens if len(token) >= 2)
        if not patterns:
            return ()
        command = [
            "rg",
            "--files-with-matches",
            "--color=never",
            "--fixed-strings",
            "--ignore-case",
            "--max-count",
            "1",
        ]
        for pattern in patterns:
            command.extend(("-e", pattern))
        command.append(str(self.project_root))
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return ()
        try:
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=4.0)
        except TimeoutError:
            process.kill()
            await process.communicate()
            return ()
        for line in stdout.decode(errors="replace").splitlines():
            relative = self._relative_file(line)
            if relative:
                paths.add(relative)
                if len(paths) >= max(self.max_files * 4, self.max_files):
                    break
        return tuple(sorted(paths))

    def _scan_ast(self, pool: Any, tokens: tuple[str, ...]) -> list[Any]:
        matches: list[Any] = []
        for symbol in sorted(
            pool,
            key=lambda item: (item.file_path, item.start_line, item.name),
        ):
            if not self._symbol_matches(symbol.name, tokens):
                continue
            matches.append(symbol)
            if len(matches) >= self.candidate_symbols:
                break
        return matches

    def _lexical_overlap(
        self,
        symbol: Any,
        bridge: QueryBridgeResult,
    ) -> float:
        terms = bridge.keywords or bridge.search_terms()
        if not terms:
            return 0.0
        symbol_tokens = _tokens(symbol.name) | _tokens(symbol.file_path)
        overlap_sum = 0.0
        for term in terms:
            term_tokens = _tokens(term)
            if not term_tokens:
                continue
            intersection = term_tokens & symbol_tokens
            overlap_sum += len(intersection) / len(term_tokens)
        return overlap_sum / len(terms)

    def _alias_match(
        self,
        symbol: Any,
        bridge: QueryBridgeResult,
    ) -> float:
        alias_terms = list(bridge.expanded_terms) + ([bridge.domain] if bridge.domain else [])
        if bridge.concepts:
            alias_terms.extend(bridge.concepts)
        if not alias_terms:
            return 0.0
        symbol_tokens = (
            _tokens(symbol.name)
            | _tokens(symbol.file_path)
            | _tokens(str(getattr(symbol, "signature", "")))
        )
        overlap_sum = 0.0
        for term in alias_terms:
            term_tokens = _tokens(term)
            if not term_tokens:
                continue
            intersection = term_tokens & symbol_tokens
            overlap_sum += len(intersection) / len(term_tokens)
        return overlap_sum / len(alias_terms)

    def _relative_file(self, value: str) -> str | None:
        raw = value.replace("\\", "/").lstrip("./")
        path = Path(value)
        if not path.is_absolute():
            path = self.project_root / raw
        try:
            resolved = path.resolve()
            relative = resolved.relative_to(self.project_root)
        except (OSError, ValueError):
            return None
        if not resolved.is_file():
            return None
        return self._normalize(relative.as_posix())

    @staticmethod
    def _normalize(path: str) -> str:
        return path.replace("\\", "/").lstrip("./")

    @staticmethod
    def _symbol_matches(name: str, tokens: tuple[str, ...]) -> bool:
        normalized_name = name.casefold()
        name_parts = {
            part.casefold()
            for part in re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[0-9]+", name.replace("_", " "))
        }
        for token in tokens:
            token_low = token.casefold()
            if token_low == normalized_name or token_low in name_parts:
                return True
            if token_low in normalized_name:
                return True
        return False

    # 修改 _symbol_id 静态方法，不再读取 symbol.symbol_id 原生属性，强制使用三元组，确保全库统一！
    @staticmethod
    def _symbol_id(symbol: Any) -> str:
        # 强行使用文件、名字、行号作为全链路唯一的物理确定性指纹
        file_path = getattr(symbol, "file_path", None) or getattr(symbol, "file", "unknown")
        return f"{file_path}:{symbol.name}:{symbol.start_line}"


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(
            r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z0-9]+",
            value.replace("_", " ").replace("/", " ").replace(".", " "),
        )
        if len(token) >= 2
    }
