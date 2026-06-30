from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from src.tools.assembled.ast_structure import CursorAstStructureLayer
from src.tools.assembled.context_pack_builder import CursorContextPackBuilder
from src.tools.assembled.fusion import CursorFusionEngine
from src.tools.assembled.graph_bridge import CursorGraphQueryBridge
from src.tools.assembled.query_bridge import CursorQueryBridge
from src.tools.assembled.repo_map_lookup import CursorRepoMapLookup
from src.tools.assembled.reranker import SiliconFlowReranker
from src.tools.assembled.retriever import CursorRetriever
from src.tools.assembled.semantic_tagger import CursorSemanticTagger
from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool
from src.llm.client import LLMClient

log = logging.getLogger(__name__)

BUILTIN_NOISE_METHODS = {
    "all",
    "append",
    "bool",
    "dict",
    "enumerate",
    "execute",
    "first",
    "get",
    "join",
    "lower",
    "mappings",
    "pop",
    "split",
    "strip",
    "text",
    "upper",
}

MAX_EVIDENCE_ITEMS = 4


class CodebaseRetrieveTool(Tool):
    name = "codebase_retrieve"
    description = (
        "Search and retrieve relevant codebase context (files, symbols, code blocks, relation graphs) "
        "for a given query. This returns structured code snippets and summaries, and automatically "
        "loads them into CURRENT_CONTEXT for the next turn."
    )
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "高度提炼的搜索关键词、符号名(如类名/函数名)或报错特征。严禁包含任何礼貌用语、命令、或多余的解释。"
            }
        },
        "required": ["query"]
    }

    def __init__(
        self,
        *,
        project_root: Path,
        settings: Any,
        repo_map: Any,
        decision_llm: Any,
        inter_llm: Any = None,
        harness: Any,
        tools: Any,
    ) -> None:
        self.project_root = project_root.resolve()
        self.settings = settings
        self.repo_map = repo_map
        self.decision_llm = decision_llm
        self.inter_llm = inter_llm
        self.harness = harness
        self.tools = tools

        self.retriever = CursorRetriever(
            self.project_root,
            tools,
            repo_map,
            max_files=settings.cursor_retrieval_max_files,
            max_symbols=settings.cursor_retrieval_max_symbols,
            candidate_symbols=settings.cursor_retrieval_candidate_symbols,
            max_queries=settings.cursor_retrieval_max_queries,
            total_timeout=settings.cursor_retrieval_timeout,
        )
        self.query_bridge_llm = LLMClient(
            model=settings.cursor_query_bridge_model,
            request_timeout=float(settings.cursor_query_bridge_timeout),
            prompt_cache_enabled=settings.prompt_cache_enabled,
            prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
            prompt_cache_ttl="5m",
        )
        self.query_bridge = CursorQueryBridge(
            self.query_bridge_llm,
            timeout=settings.cursor_query_bridge_timeout,
        )
        self.repo_map_lookup = CursorRepoMapLookup(repo_map)
        self.graph_bridge = CursorGraphQueryBridge(
            repo_map,
            depth=settings.cursor_graph_bridge_depth,
            top_symbols=settings.cursor_graph_bridge_max_symbols,
            top_files=settings.cursor_graph_bridge_max_files,
            max_seeds=settings.cursor_graph_bridge_max_seeds,
        )
        reranker = (
            SiliconFlowReranker(
                model=settings.cursor_reranker_model,
                api_base=settings.cursor_reranker_api_base,
                timeout=settings.cursor_reranker_timeout,
            )
            if settings.cursor_reranker_enabled
            else None
        )
        self.fusion = CursorFusionEngine(
            reranker=reranker,
            rerank_enabled=settings.cursor_reranker_enabled,
            rerank_top_n=settings.cursor_reranker_top_n,
        )
        self.context_builder = CursorContextPackBuilder(
            self.project_root,
            max_files=settings.cursor_max_context_files,
            max_chars_per_file=settings.cursor_context_chars_per_file,
            dependency_affinity_threshold=settings.cursor_dependency_affinity_threshold,
        )
        self.ast_structure = CursorAstStructureLayer(self.project_root)
        self.semantic_tagger = CursorSemanticTagger()

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        user_msg = validated["query"]
        paths = validated.get("paths") or []


        try:
            bridge_content = await self.query_bridge.generate_raw(user_msg)
            session_id = getattr(self.harness, "session_id", "default_session")
            subtask_id = f"step_{getattr(self.harness, 'current_step', 1)}_retrieve_bridge"
            if hasattr(self.harness, "session_storage"):
                self.harness.session_storage.append_sidechain_message(
                    session_id, subtask_id, "user", f"Generate query bridge for: {user_msg}"
                )
                self.harness.session_storage.append_sidechain_message(
                    session_id, subtask_id, "assistant", bridge_content
                )
        except Exception as exc:
            log.warning("Cursor Query Bridge call failed inside codebase_retrieve: %s", exc)
            bridge_content = "{}"

        guarded_bridge = self.harness.cursor_retrieval_guardrail.validate_bridge_json(
            bridge_content,
            self.query_bridge.fallback(user_msg),
        )
        bridge_result = guarded_bridge.bridge

        if paths:
            bridge_result.file_hints = [str(p) for p in paths]

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

        ast_nodes_list = []
        seen_symbols = set()
        limit = self.settings.cursor_retrieval_candidate_symbols
        for candidate in ast_candidates:
            symbol = candidate.symbol
            symbol_id = str(
                getattr(
                    symbol,
                    "symbol_id",
                    f"{getattr(symbol, 'file_path', None) or getattr(symbol, 'file', 'unknown')}:{symbol.name}:{symbol.start_line}",
                )
            )
            if symbol_id in seen_symbols:
                continue
            seen_symbols.add(symbol_id)
            node = await asyncio.to_thread(self.ast_structure._node, symbol)
            if node is not None:
                ast_nodes_list.append(node)
            if len(ast_nodes_list) >= limit:
                break
        ast_nodes = tuple(ast_nodes_list)

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
            user_intent=user_msg,
        )
        retrieval = fusion_result.retrieval

        context_pack = self.context_builder.build_context(
            retrieval,
            final_context=fusion_result.final_context,
        )
        if self.settings.cursor_semantic_tags_enabled:
            annotations = self.semantic_tagger.annotate(context_pack)
            context_pack = self.context_builder.with_annotations(
                context_pack,
                annotations,
            )

        top_symbols = list(retrieval.symbols[:MAX_EVIDENCE_ITEMS])
        # The coordinator receives one deliberately small schema.  A code anchor is
        # identified by (file, span); only first-hop function anchors accompany it.
        # RetrievalSymbol carries normalized call edges. AstNode adds grounded
        # first-hop symbols. Repo-map RankedSymbol is intentionally excluded here:
        # it has ranking metadata but no ``calls`` field.
        related_symbol_pool = [*retrieval.symbols, *ast_nodes]

        def symbol_name(value: Any) -> str:
            return str(getattr(value, "name", None) or getattr(value, "symbol", ""))

        def symbol_file(value: Any) -> str:
            return str(getattr(value, "file", None) or getattr(value, "file_path", ""))

        def symbol_span(value: Any) -> list[int]:
            lines = getattr(value, "lines", None)
            if lines and len(lines) >= 2:
                return [int(lines[0]), int(lines[1])]
            return [int(getattr(value, "start_line", 0)), int(getattr(value, "end_line", 0))]

        symbol_by_name = {
            symbol_name(symbol): symbol
            for symbol in related_symbol_pool
            if symbol_name(symbol) and symbol_name(symbol) not in BUILTIN_NOISE_METHODS
        }
        inbound: dict[str, set[str]] = {}
        for candidate in related_symbol_pool:
            for called in getattr(candidate, "calls", ()) or ():
                inbound.setdefault(str(called), set()).add(symbol_name(candidate))

        code_anchors = []
        for symbol in top_symbols:
            start_line = max(1, int(getattr(symbol, "start_line", 1)))
            end_line = max(start_line, int(getattr(symbol, "end_line", start_line)))
            code_anchors.append({
                "file": symbol.file,
                "symbol": symbol.name,
                "span": [start_line, end_line],
            })

        compact_json_output = json.dumps(code_anchors, ensure_ascii=False, indent=2)

        return ToolResult(
            success=True,
            output=compact_json_output,
            metadata={
                "llm_observation": compact_json_output,
                "retrieved_files": list(context_pack.candidate_files),
                "context_pack": context_pack,
                "search_output": compact_json_output,
                "raw_evidence_store": code_anchors,
            }
        )

    def _symbol_code_slice(self, symbol: Any, context_pack: Any) -> tuple[str, int, int, bool]:
        abs_path = (self.project_root / symbol.file).resolve()
        start_line = max(1, int(symbol.start_line))
        end_line = max(start_line, int(symbol.end_line))
        lines: list[str] | None = None
        if abs_path.is_file():
            try:
                lines = abs_path.read_text(encoding="utf-8").splitlines()
            except OSError as exc:
                log.warning("Failed to read symbol slice from %s: %s", symbol.file, exc)
        if lines is None:
            for window in context_pack.windows:
                if window.file != symbol.file:
                    continue
                lines = window.content.splitlines()
                start_line = max(1, int(window.start_line))
                end_line = start_line + len(lines) - 1
                break
        if not lines:
            return "", start_line, end_line, False

        if abs_path.is_file():
            slice_start = max(0, start_line - 1)
            slice_end = min(len(lines), end_line)
            sliced = lines[slice_start:slice_end]
            actual_start = start_line
        else:
            sliced = lines
            actual_start = start_line

        actual_end = actual_start + len(sliced) - 1 if sliced else actual_start
        return "\n".join(sliced), actual_start, actual_end, False
