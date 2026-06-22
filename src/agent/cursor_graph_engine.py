from __future__ import annotations

import asyncio
from typing import Any
from src.agent.cursor_fusion import FusionResult
from src.agent.cursor_query_bridge import QueryBridgeResult


class CursorGraphEngine:

    def __init__(
        self,
        repo_map_lookup: Any,
        graph_bridge: Any,
        ast_structure: Any,
        retriever: Any,
        fusion: Any,
        settings: Any,
        harness: Any,
    ) -> None:
        self.repo_map_lookup = repo_map_lookup
        self.graph_bridge = graph_bridge
        self.ast_structure = ast_structure
        self.retriever = retriever
        self.fusion = fusion
        self.settings = settings
        self.harness = harness

    async def compile_dependency_subgraph(self, seed: str, depth: int = 2) -> FusionResult:
        # Mock bridge result targeting the seed
        bridge_result = QueryBridgeResult(
            intent="modify",
            expanded_terms=[seed],
            keywords=[seed],
            symbols=[],
            file_hints=[],
            domain="codebase",
            concepts=[seed],
            constraints={"layer_hint": [], "exclude": []},
        )

        candidate_symbols = await asyncio.to_thread(
            self.repo_map_lookup.lookup,
            bridge_result,
            limit=max(self.settings.cursor_retrieval_candidate_symbols, 40),
        )
        graph_result = await self.graph_bridge.expand_candidates(
            candidate_symbols,
            bridge=bridge_result,
        )
        ast_candidates = await asyncio.to_thread(
            self.repo_map_lookup.merge_by_ids,
            candidate_symbols,
            graph_result.expanded_symbol_ids,
        )
        ast_nodes = await asyncio.to_thread(
            self.ast_structure.ground,
            ast_candidates,
            limit=self.settings.cursor_retrieval_candidate_symbols,
        )
        if self.retriever.repo_map is not None:
            raw_retrieval = await asyncio.to_thread(
                self.retriever.score_candidates,
                ast_nodes=ast_nodes,
                candidates=ast_candidates,
                bridge=bridge_result,
                graph=graph_result,
            )
        else:
            guarded = await self.harness.cursor_retrieval_guardrail.run(
                bridge_result.search_terms(limit=32),
                self.retriever.retrieve,
            )
            raw_retrieval = guarded.retrieval

        fusion_result = await self.fusion.decide_async(
            raw_retrieval,
            bridge_result,
            max_files=self.settings.cursor_retrieval_max_files,
            max_symbols=self.settings.cursor_retrieval_max_symbols,
            top_k=min(8, self.settings.cursor_retrieval_max_symbols),
            user_intent=seed,
        )
        return fusion_result
