from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from src.agent.cursor_query_bridge import QueryBridgeResult


@dataclass(frozen=True, slots=True)
class CandidateSymbol:
    symbol: Any
    score_hint: float
    reasons: tuple[str, ...] = ()


class CursorRepoMapLookup:
    """Locate symbol candidates from QueryBridge terms using repo-map metadata only."""

    def __init__(self, repo_map_service: Any = None, *, ready_timeout: float = 2.0) -> None:
        self.repo_map_service = repo_map_service
        self.ready_timeout = ready_timeout

    def lookup(
        self,
        bridge: QueryBridgeResult,
        *,
        limit: int = 40,
    ) -> tuple[CandidateSymbol, ...]:
        repo_map = self._repo_map()
        if repo_map is None:
            return ()
        terms = bridge.expanded_terms + (bridge.concepts or [])
        if bridge.domain:
            terms.append(bridge.domain)
        terms.extend((bridge.constraints or {}).get("layer_hint", ()))
        if hasattr(repo_map, "lookup_candidates"):
            return tuple(
                CandidateSymbol(
                    symbol=candidate.symbol,
                    score_hint=float(candidate.score_hint),
                    reasons=tuple(candidate.reasons),
                )
                for candidate in repo_map.lookup_candidates(
                    tuple(dict.fromkeys(terms)),
                    domain="",
                    constraints={"layer_hint": [], "exclude": []},
                    limit=limit,
                )
            )
        return self._lookup_sync(repo_map, tuple(dict.fromkeys(terms)), limit=limit)

    def merge_by_ids(
        self,
        candidates: tuple[CandidateSymbol, ...],
        symbol_ids: tuple[str, ...],
    ) -> tuple[CandidateSymbol, ...]:
        repo_map = self._repo_map()
        if repo_map is None or not symbol_ids:
            return candidates
        out = list(candidates)
        seen = {self._symbol_id(candidate.symbol) for candidate in candidates}
        for symbol_id in symbol_ids:
            if symbol_id in seen:
                continue
            symbol = getattr(repo_map, "symbols_by_id", {}).get(symbol_id)
            if symbol is None:
                continue
            seen.add(symbol_id)
            out.append(CandidateSymbol(symbol=symbol, score_hint=0.0, reasons=("graph_expanded",)))
        return tuple(out)

    def _repo_map(self) -> Any | None:
        service = self.repo_map_service
        if service is None:
            return None
        try:
            service.wait_until_ready(timeout=self.ready_timeout)
        except Exception:
            return None
        return getattr(service, "map", None)

    # 修改 _symbol_id 静态方法，不再读取 symbol.symbol_id 原生属性，强制使用三元组，确保全库统一！
    @staticmethod
    def _symbol_id(symbol: Any) -> str:
        # 强行使用文件、名字、行号作为全链路唯一的物理确定性指纹
        file_path = getattr(symbol, "file_path", None) or getattr(symbol, "file", "unknown")
        return f"{file_path}:{symbol.name}:{symbol.start_line}"

    @staticmethod
    def _lookup_sync(
        repo_map: Any,
        terms: tuple[str, ...],
        *,
        limit: int,
    ) -> tuple[CandidateSymbol, ...]:
        pool = list(getattr(repo_map, "all_symbols", None) or repo_map.symbols)
        candidates: list[CandidateSymbol] = []
        for symbol in pool:
            score = 0.0
            reasons: set[str] = set()
            haystacks = {
                "name": str(symbol.name).casefold(),
                "file": str(symbol.file_path).casefold(),
                "signature": str(getattr(symbol, "signature", "")).casefold(),
                "kind": str(getattr(symbol, "kind", "")).casefold(),
            }
            name_tokens = _tokens(symbol.name)
            file_tokens = _tokens(symbol.file_path)
            sig_tokens = _tokens(str(getattr(symbol, "signature", "")))
            for term in terms:
                term_low = term.casefold()
                term_tokens = _tokens(term)
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
            score += min(float(getattr(symbol, "score", 0.0)), 0.15)
            candidates.append(
                CandidateSymbol(
                    symbol=symbol,
                    score_hint=round(min(score / 2.2, 1.0), 4),
                    reasons=tuple(sorted(reasons)),
                )
            )
        candidates.sort(
            key=lambda item: (
                -item.score_hint,
                -float(getattr(item.symbol, "score", 0.0)),
                item.symbol.file_path,
                item.symbol.start_line,
                item.symbol.name,
            )
        )
        return tuple(candidates[:limit])


def _tokens(value: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(
            r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z0-9]+",
            value.replace("_", " ").replace("/", " ").replace(".", " "),
        )
        if len(token) >= 2
    }


# 修改 _symbol_id 静态方法，不再读取 symbol.symbol_id 原生属性，强制使用三元组，确保全库统一！
def _symbol_id(symbol: Any) -> str:
    # 强行使用文件、名字、行号作为全链路唯一的物理确定性指纹
    file_path = getattr(symbol, "file_path", None) or getattr(symbol, "file", "unknown")
    return f"{file_path}:{symbol.name}:{symbol.start_line}"
