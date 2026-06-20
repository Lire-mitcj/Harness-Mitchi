from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

from src.agent.cursor_repo_map_lookup import CandidateSymbol

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class GraphNode:
    id: str
    name: str
    type: str
    file: str | None
    score: float
    distance: int


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: str
    target: str
    weight: float
    type: str


@dataclass(frozen=True, slots=True)
class GraphBridgeResult:
    expanded_symbols: tuple[str, ...] = ()
    expanded_symbol_ids: tuple[str, ...] = ()
    expanded_files: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    graph_nodes: tuple[GraphNode, ...] = ()
    graph_edges: tuple[GraphEdge, ...] = ()
    meta: dict[str, Any] | None = None

    def retrieval_terms(self) -> tuple[str, ...]:
        return self.expanded_symbols + self.expanded_files


class CursorGraphQueryBridge:
    """Expand retrieval concepts through the repository AST symbol graph."""

    def __init__(
        self,
        repo_map_service: Any = None,
        *,
        depth: int = 2,
        top_symbols: int = 10,
        top_files: int = 5,
        max_seeds: int = 8,
        cooccurrence_per_file: int = 8,
        ready_timeout: float = 2.0,
    ) -> None:
        self.repo_map_service = repo_map_service
        self.depth = depth
        self.top_symbols = top_symbols
        self.top_files = top_files
        self.max_seeds = max_seeds
        self.cooccurrence_per_file = cooccurrence_per_file
        self.ready_timeout = ready_timeout
        self.k1 = 1.2
        self.b = 0.75
        self.FRAMEWORK_TOKENS = {
            "query", "api", "get", "post", "put", "delete", "route", "endpoint",
            "sql", "db",
        }
        self._scc_map: dict[str, set[str]] | None = None
        self._node_degrees: dict[str, int] | None = None

    async def expand(self, bridge: Any) -> GraphBridgeResult:
        """Deprecated compatibility path; graph expansion requires explicit candidates."""
        _ = bridge
        return GraphBridgeResult()

    async def expand_candidates(
        self,
        candidates: tuple[CandidateSymbol, ...],
        *,
        bridge: Any = None,
    ) -> GraphBridgeResult:
        repo_map = await self._repo_map()
        if repo_map is None or not candidates:
            return GraphBridgeResult()
        # 🔥 统一修复：全部强制通过 self._symbol_id 统一方法取值，消除原生 symbol_id 带来的格式断层
        candidate_ids = tuple(self._symbol_id(candidate.symbol) for candidate in candidates)
        candidate_scores = {
            self._symbol_id(candidate.symbol): float(candidate.score_hint)
            for candidate in candidates
        }
        concepts = ()
        if bridge is not None:
            if hasattr(bridge, "concepts") and bridge.concepts:
                concepts = tuple(bridge.concepts)
            elif hasattr(bridge, "search_terms"):
                concepts = tuple(bridge.search_terms())

        return await asyncio.to_thread(
            self._expand_sync,
            repo_map,
            candidate_ids,
            candidate_scores,
            concepts,
        )

    async def _repo_map(self) -> Any | None:
        service = self.repo_map_service
        if service is None:
            return None
        try:
            await asyncio.wait_for(
                asyncio.to_thread(
                    service.wait_until_ready,
                    timeout=self.ready_timeout,
                ),
                timeout=self.ready_timeout + 0.25,
            )
        except Exception:
            return None
        return getattr(service, "map", None)

    def _prepare_bm25_stats(self, symbols: list[Any]) -> dict[str, Any]:
        """
        性能核心防御：在进入检索循环前，一次性线性扫描全库符号，计算 BM25 所需的全局元数据。
        时间复杂度：O(N)
        """
        total_symbols_count = len(symbols)
        if total_symbols_count == 0:
            return {"doc_freq": {}, "avgdl": 0.0, "total_docs": 0}

        doc_freq: Counter[str] = Counter()
        total_tokens_length = 0

        for symbol in symbols:
            name_tokens = self._tokens(symbol.name)
            file_tokens = self._tokens(symbol.file_path)
            combined_tokens = name_tokens | file_tokens

            doc_freq.update(combined_tokens)
            total_tokens_length += len(name_tokens) + len(file_tokens)

        return {
            "doc_freq": doc_freq,
            "avgdl": total_tokens_length / total_symbols_count,
            "total_docs": total_symbols_count
        }

    def _calculate_bm25_score(
        self,
        concept: str,
        symbol: Any,
        stats: dict[str, Any]
    ) -> float:
        """
        使用标准的 BM25 算法替代原有的硬 overlap 匹配
        """
        concept_tokens = self._tokens(concept)
        if not concept_tokens:
            return 0.0

        symbol_name_tokens = list(self._tokens(symbol.name))
        symbol_file_tokens = list(self._tokens(symbol.file_path))

        name_counter = Counter(symbol_name_tokens)
        file_counter = Counter(symbol_file_tokens)

        if concept.casefold() == symbol.name.casefold():
            return 1.0

        score = 0.0
        doc_len = len(symbol_name_tokens) + len(symbol_file_tokens)
        avgdl = stats["avgdl"]
        total_docs = stats["total_docs"]
        doc_freq = stats["doc_freq"]

        for q_token in concept_tokens:
            tf = (name_counter[q_token] * 2.0) + (file_counter[q_token] * 0.5)
            if tf == 0:
                continue

            df = doc_freq.get(q_token, 0)

            if q_token not in self.FRAMEWORK_TOKENS:
                df = min(df, 2)

            idf = math.log((total_docs - df + 0.5) / (df + 0.5) + 1.0)
            idf = max(0.0001, idf)

            denominator = tf + self.k1 * (
                1.0 - self.b + self.b * (doc_len / (avgdl if avgdl > 0 else 1.0))
            )
            token_bm25 = idf * (tf * (self.k1 + 1.0)) / denominator

            score += token_bm25

        return min(0.95, score / 5.0)

    def _assign_domain(self, file_path: str | None) -> str:
        if not file_path:
            return "backend"
        lowered = file_path.casefold()
        if any(part in lowered for part in ("view", "page", "component", "ui")):
            return "ui"
        if any(part in lowered for part in ("dao", "db", "sql", "repository", "schema")):
            return "db"
        return "backend"

    def _find_sccs(self, adjacency: dict[str, list[tuple[str, float, str]]]) -> list[set[str]]:
        index_counter = [0]
        stack: list[str] = []
        lowlink: dict[str, int] = {}
        index: dict[str, int] = {}
        on_stack: set[str] = set()
        sccs: list[set[str]] = []

        def strongconnect(node: str) -> None:
            index[node] = index_counter[0]
            lowlink[node] = index_counter[0]
            index_counter[0] += 1
            stack.append(node)
            on_stack.add(node)

            for target, _, _ in adjacency.get(node, []):
                if target not in index:
                    strongconnect(target)
                    lowlink[node] = min(lowlink[node], lowlink[target])
                elif target in on_stack:
                    lowlink[node] = min(lowlink[node], index[target])

            if lowlink[node] == index[node]:
                scc: set[str] = set()
                while True:
                    w = stack.pop()
                    on_stack.remove(w)
                    scc.add(w)
                    if w == node:
                        break
                sccs.append(scc)

        for node in adjacency:
            if node not in index:
                strongconnect(node)
        return sccs

    def _expand_sync(
        self,
        repo_map: Any,
        candidate_ids: tuple[str, ...] = (),
        candidate_scores: dict[str, float] | None = None,
        concepts: tuple[str, ...] = (),
    ) -> GraphBridgeResult:
        symbols = list(getattr(repo_map, "all_symbols", None) or repo_map.symbols)
        if not symbols:
            return GraphBridgeResult()

        token_counts: Counter[str] = Counter()
        for s in symbols:
            token_counts.update(self._tokens(s.name))

        bm25_stats = self._prepare_bm25_stats(symbols)
        lexical_dict = {}
        for symbol in symbols:
            symbol_id = self._symbol_id(symbol)
            lex_scores = [
                self._calculate_bm25_score(concept, symbol, bm25_stats)
                for concept in concepts
            ]
            lexical_dict[symbol_id] = max(lex_scores) if lex_scores else 0.0

        by_id = {
            self._symbol_id(symbol): symbol
            for symbol in symbols
        }
        by_file: dict[str, list[Any]] = defaultdict(list)
        for symbol in symbols:
            by_file[symbol.file_path].append(symbol)
        for file_symbols in by_file.values():
            file_symbols.sort(key=lambda item: (item.start_line, item.name))

        adjacency: dict[str, list[tuple[str, float, str]]] = defaultdict(list)
        graph_edges: dict[tuple[str, str, str], GraphEdge] = {}

        def add_edge(source: str, target: str, weight: float, edge_type: str) -> None:
            if source == target or source not in by_id or target not in by_id:
                return
            key = (source, target, edge_type)
            if key in graph_edges:
                return
            graph_edges[key] = GraphEdge(source, target, weight, edge_type)
            adjacency[source].append((target, weight, edge_type))

        for source, target in getattr(repo_map, "reference_edges", ()):
            if source in by_id and target in by_id:
                add_edge(source, target, 1.0, "reference")
                add_edge(target, source, 0.75, "reference_reverse")
            elif source.startswith("file:") and target.startswith("file:"):
                source_file = source.removeprefix("file:")
                target_file = target.removeprefix("file:")
                for source_symbol in by_file.get(source_file, ())[:4]:
                    for target_symbol in by_file.get(target_file, ())[:4]:
                        add_edge(
                            self._symbol_id(source_symbol),
                            self._symbol_id(target_symbol),
                            0.55,
                            "import",
                        )

        for file_symbols in by_file.values():
            bounded = file_symbols[:self.cooccurrence_per_file]
            file_normalizer = max(1.0, math.log1p(len(file_symbols)))
            for index, left in enumerate(bounded):
                left_id = self._symbol_id(left)
                for right in bounded[index + 1:]:
                    right_id = self._symbol_id(right)
                    line_gap = abs(left.start_line - right.start_line)
                    weight = (
                        0.45 + (0.35 / (1.0 + line_gap / 50.0))
                    ) / file_normalizer
                    add_edge(left_id, right_id, weight, "co_occurrence")
                    add_edge(right_id, left_id, weight, "co_occurrence")

        for symbol in symbols:
            parent_id = str(getattr(symbol, "parent_symbol_id", ""))
            if parent_id:
                add_edge(parent_id, self._symbol_id(symbol), 1.0, "embedded_sql")
                add_edge(self._symbol_id(symbol), parent_id, 0.85, "embedded_sql")

        self._add_db_schema_affinity_edges(symbols, add_edge)

        token_index: dict[str, list[str]] = defaultdict(list)
        for symbol in symbols:
            symbol_id = self._symbol_id(symbol)
            for token in sorted(self._tokens(symbol.name)):
                token_index[token].append(symbol_id)
        for token_ids in token_index.values():
            bounded_ids = sorted(set(token_ids))[:12]
            for index, source in enumerate(bounded_ids):
                for target in bounded_ids[index + 1:index + 4]:
                    add_edge(source, target, 0.3, "naming_similarity")
                    add_edge(target, source, 0.3, "naming_similarity")

        # Static pre-computation caching of SCCs and Degrees
        if self._scc_map is None or self._node_degrees is None:
            node_degrees: dict[str, int] = {
                node_id: len(adjacency[node_id])
                for node_id in by_id
            }
            sccs = self._find_sccs(adjacency)
            scc_map: dict[str, set[str]] = {}
            for scc in sccs:
                for node_id in scc:
                    scc_map[node_id] = scc
            self._node_degrees = node_degrees
            self._scc_map = scc_map
        node_degrees = self._node_degrees or {}
        scc_map = self._scc_map or {}

        graph_dict: dict[str, float] = defaultdict(float)

        seed_scores: dict[str, float] = {}
        candidate_scores = candidate_scores or {}
        allowed_seed_ids = set(candidate_ids)
        for symbol_id in allowed_seed_ids:
            if symbol_id not in by_id:
                continue
            base_score = max(candidate_scores.get(symbol_id, 0.0), 0.01)
            bm25_score = lexical_dict.get(symbol_id, 0.0)
            score = max(base_score, bm25_score)
            sym_name = by_id[symbol_id].name
            if any(t in self.FRAMEWORK_TOKENS for t in self._tokens(sym_name)):
                score *= 0.01
            seed_scores[symbol_id] = score

        ranked_seeds = sorted(
            seed_scores,
            key=lambda symbol_id: (
                -seed_scores[symbol_id],
                -float(getattr(by_id[symbol_id], "score", 0.0)),
                symbol_id,
            ),
        )[:self.max_seeds]
        if not ranked_seeds:
            return GraphBridgeResult()

        scores = {
            symbol_id: seed_scores[symbol_id]
            for symbol_id in ranked_seeds
        }
        distances = {symbol_id: 0 for symbol_id in ranked_seeds}

        # Determine dominant seed domain
        if ranked_seeds:
            top_seed_id = ranked_seeds[0]
            top_seed_domain = self._assign_domain(by_id[top_seed_id].file_path)
        else:
            top_seed_domain = "backend"

        # BFS queue holds: (node_id, seed_domain, distance)
        queue: deque[tuple[str, str, int]] = deque()
        for seed_id in ranked_seeds:
            s_domain = self._assign_domain(by_id[seed_id].file_path)
            queue.append((seed_id, s_domain, 0))

        visited = set(ranked_seeds)
        traversed: dict[tuple[str, str, str], GraphEdge] = {}

        while queue:
            source, seed_domain, distance = queue.popleft()
            if distance >= self.depth:
                continue

            # BFS branching limit: max 8 neighbors sorted by weight descending
            neighbors = sorted(adjacency[source], key=lambda item: -item[1])[:8]

            for target, weight, edge_type in neighbors:
                next_distance = distance + 1

                source_domain = self._assign_domain(by_id[source].file_path)
                target_domain = self._assign_domain(by_id[target].file_path)

                # Check domain firewalls at distance 0
                if distance == 0:
                    # Caller exemption: always allow reference_reverse
                    if edge_type != "reference_reverse":
                        if seed_domain == "ui":
                            if target_domain != "ui":
                                continue
                        elif seed_domain == "db":
                            if target_domain != "db":
                                continue
                        elif seed_domain == "backend":
                            if target_domain == "ui":
                                continue

                domain_match = (source_domain == target_domain)
                decay_val = self._decay(edge_type, next_distance, domain_match)
                propagated = scores[source] * weight * decay_val

                # Apply SCC & degree cluster penalty
                target_degree = node_degrees.get(target, 0)
                cluster_penalty = max(1.0, math.log(target_degree + 1.0))
                scc = scc_map.get(target, set())
                if len(scc) >= 4:
                    cluster_penalty *= 2.0
                propagated /= cluster_penalty

                proximity = 0.08 if by_id[source].file_path == by_id[target].file_path else 0.0
                rank_bonus = min(float(getattr(by_id[target], "score", 0.0)), 0.2)
                candidate_score = propagated + proximity + rank_bonus

                # Smoothed lexical cap
                target_lexical = lexical_dict.get(target, 0.0)
                candidate_score = min(candidate_score, max(0.12, target_lexical * 1.2))

                edge = graph_edges[(source, target, edge_type)]
                traversed[(source, target, edge_type)] = edge

                graph_dict[target] = max(graph_dict[target], candidate_score)
                target_bm25 = lexical_dict.get(target, 0.0)
                new_score = max(graph_dict[target], target_bm25)
                scores[target] = max(scores.get(target, 0.0), new_score)

                if target in visited:
                    continue
                visited.add(target)
                distances[target] = next_distance
                queue.append((target, seed_domain, next_distance))

        # Keep high-confidence seed symbols anchored, then fill with expanded neighbors.
        seed_set = set(ranked_seeds)
        remaining_ids = sorted(
            (symbol_id for symbol_id in visited if symbol_id not in seed_set),
            key=lambda symbol_id: (
                -scores[symbol_id],
                distances.get(symbol_id, self.depth + 1),
                by_id[symbol_id].file_path,
                by_id[symbol_id].name,
            ),
        )
        top_ids = (ranked_seeds + remaining_ids)[:self.top_symbols]

        # NO ranking in output order: Sort stably by file path and name
        stable_ids = sorted(
            top_ids,
            key=lambda symbol_id: (
                by_id[symbol_id].file_path,
                by_id[symbol_id].name,
            ),
        )
        ranked_ids = stable_ids

        nodes = [
            GraphNode(
                id=symbol_id,
                name=by_id[symbol_id].name,
                type=str(getattr(by_id[symbol_id], "kind", "symbol")),
                file=by_id[symbol_id].file_path,
                score=round(scores[symbol_id], 6),
                distance=distances.get(symbol_id, 0),
            )
            for symbol_id in ranked_ids
        ]

        expanded_files = sorted(list({by_id[sid].file_path for sid in ranked_ids}))[:self.top_files]

        relevant_edges = [
            edge
            for edge in traversed.values()
            if edge.source in by_id and edge.target in by_id
        ]
        relevant_edges.sort(key=lambda edge: (edge.type, edge.source, edge.target))
        paths = tuple(
            _edge_path(edge, by_id)
            for edge in relevant_edges
            if edge.source in by_id and edge.target in by_id
        )

        meta_dict = {
            "domain": top_seed_domain,
            "scc_penalty_applied": True
        }

        return GraphBridgeResult(
            expanded_symbols=tuple(by_id[symbol_id].name for symbol_id in ranked_ids),
            expanded_symbol_ids=tuple(ranked_ids),
            expanded_files=tuple(expanded_files),
            paths=tuple(dict.fromkeys(paths)),
            graph_nodes=tuple(nodes),
            graph_edges=tuple(relevant_edges),
            meta=meta_dict,
        )

    def _decay(self, edge_type: str, distance: int, domain_match: bool = True) -> float:
        """Calculate decay factor based on edge type, distance, and domain matching."""
        decay_map = {
            "reference": 0.3,
            "reference_reverse": 0.4,
            "embedded_sql": 0.2,
            "db_schema_affinity": 0.25,
            "import": 0.55,
            "co_occurrence": 0.65,
            "naming_similarity": 0.75,
        }
        base = decay_map.get(edge_type, 0.4)
        exponent = base * (1.0 + 0.15 * (distance - 1))
        val = math.exp(-exponent)
        if not domain_match:
            val *= 0.3
        return val

    # 修改 _symbol_id 静态方法，不再读取 symbol.symbol_id 原生属性，强制使用三元组，确保全库统一！
    @staticmethod
    def _symbol_id(symbol: Any) -> str:
        # 强行使用文件、名字、行号作为全链路唯一的物理确定性指纹
        file_path = getattr(symbol, "file_path", None) or getattr(symbol, "file", "unknown")
        return f"{file_path}:{symbol.name}:{symbol.start_line}"

    def _add_db_schema_affinity_edges(
        self,
        symbols: list[Any],
        add_edge: Any,
    ) -> None:
        dml_symbols = [
            symbol
            for symbol in symbols
            if str(getattr(symbol, "kind", "")).startswith("dml_")
        ]
        ddl_symbols = [
            symbol
            for symbol in symbols
            if str(getattr(symbol, "kind", "")).startswith("ddl_")
        ]
        for dml_symbol in dml_symbols:
            dml_refs = self._schema_tokens(dml_symbol)
            if not dml_refs:
                continue
            for ddl_symbol in ddl_symbols:
                ddl_refs = self._schema_tokens(ddl_symbol) | self._tokens(ddl_symbol.name)
                if not dml_refs & ddl_refs:
                    continue
                dml_id = self._symbol_id(dml_symbol)
                ddl_id = self._symbol_id(ddl_symbol)
                add_edge(dml_id, ddl_id, 0.95, "db_schema_affinity")
                add_edge(ddl_id, dml_id, 0.75, "db_schema_affinity")

    def _schema_tokens(self, symbol: Any) -> set[str]:
        tokens: set[str] = set()
        for table in getattr(symbol, "tables_referenced", ()) or ():
            tokens.update(self._tokens(str(table)))
        if not tokens:
            signature = str(getattr(symbol, "signature", ""))
            for keyword in ("from", "join", "into", "update"):
                pattern = rf"\b{keyword}\s+([`\"\[]?[\w$.]+[`\"\]]?)"
                for match in re.finditer(pattern, signature, flags=re.IGNORECASE):
                    tokens.update(self._tokens(match.group(1).strip("`\"[]")))
        return tokens

    @staticmethod
    def _tokens(value: str) -> set[str]:
        return {
            token.casefold()
            for token in re.findall(
                r"[A-Z]?[a-z]+|[A-Z]+(?=[A-Z]|$)|[A-Za-z0-9]+",
                value.replace("_", " ").replace("/", " ").replace(".", " "),
            )
            if len(token) >= 2
        }


def _edge_path(edge: GraphEdge, by_id: dict[str, Any]) -> str:
    source = by_id[edge.source]
    target = by_id[edge.target]
    return (
        f"{_layer_name(source.file_path)}:{source.name}"
        f" -> {_layer_name(target.file_path)}:{target.name}"
    )


def _layer_name(path: str) -> str:
    lowered = path.casefold()
    if any(part in lowered for part in ("api", "route", "controller", "handler")):
        return "api"
    if "service" in lowered:
        return "service"
    if any(part in lowered for part in ("dao", "db", "sql", "repository", "schema")):
        return "dao"
    if any(part in lowered for part in ("view", "page", "component", "ui")):
        return "ui"
    return "code"
