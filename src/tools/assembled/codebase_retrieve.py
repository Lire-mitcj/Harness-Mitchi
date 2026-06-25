from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from src.agent.cursor_ast_structure import CursorAstStructureLayer
from src.agent.cursor_context_pack_builder import CursorContextPackBuilder
from src.agent.cursor_fusion import CursorFusionEngine
from src.agent.cursor_graph_bridge import CursorGraphQueryBridge
from src.agent.cursor_inter_llm import CursorInterLLM
from src.agent.cursor_query_bridge import CursorQueryBridge
from src.agent.cursor_repo_map_lookup import CursorRepoMapLookup
from src.agent.cursor_reranker import SiliconFlowReranker
from src.agent.cursor_retriever import CursorRetriever
from src.agent.cursor_semantic_tagger import CursorSemanticTagger
from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool

log = logging.getLogger(__name__)


class CodebaseRetrieveTool(Tool):
    name = "codebase_retrieve"
    description = (
        "Search and retrieve relevant codebase context (files, symbols, code blocks) "
        "for a given query. This will automatically load retrieved files into the active context."
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
        inter_llm: Any,
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
        self.query_bridge = CursorQueryBridge(
            self.decision_llm,
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
        self.inter = CursorInterLLM(self.inter_llm)

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        user_msg = validated["query"]
        paths = validated.get("paths") or []

        hint = None
        if self.settings.cursor_inter_enabled:
            try:
                inter_messages = await self.harness.before_llm_call(
                    self.inter.build_messages(user_msg)
                )
                response = await self.inter_llm.chat(
                    inter_messages,
                    tools=None,
                    stream=False,
                )
                hint = self.inter.parse(response.content or "")
                await self.harness.after_llm_call(response, response.usage)
            except Exception as exc:
                log.warning("Cursor Inter call failed inside codebase_retrieve: %s", exc)

        if hint is None:
            hint = CursorInterLLM.fallback(user_msg)

        try:
            bridge_content = await self.query_bridge.generate_raw(user_msg)
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

        summary_blocks = []
        for window in context_pack.windows:
            tags = ", ".join(window.semantic_tags) or "none"
            summary_blocks.append(
                f"- 找到关联文件: `{window.file}` (行号: {window.start_line}-{window.end_line}, 语义标签: [{tags}])"
            )

        output_text = (
            f"检索成功。已经自动将以下相关文件加载进你的活动上下文空间（Active Files）中。\n"
            f"你可以直接阅读你当前 Prompt 中对应的 `<file>` 块，无需重新检索：\n"
            + "\n".join(summary_blocks)
        )

        return ToolResult(
            success=True,
            output=output_text,
            metadata={
                "retrieved_files": list(context_pack.candidate_files),  # 供 Loop 捕获并推进 State 的核心元数据
                "context_pack": context_pack,
                "hint": hint,
            }
        )
