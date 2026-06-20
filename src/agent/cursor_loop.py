from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast

from src.agent.cursor_ast_structure import CursorAstStructureLayer
from src.agent.cursor_context_pack_builder import CursorContextPackBuilder
from src.agent.cursor_contracts import InterHint
from src.agent.cursor_decision import CursorDecisionLLM, DecisionError
from src.agent.cursor_evaluator import (
    CursorEvaluator,
    compute_layer1_metrics,
    format_bi_report,
)
from src.agent.cursor_executor import CursorExecutor
from src.agent.cursor_fusion import CursorFusionEngine
from src.agent.cursor_graph_bridge import CursorGraphQueryBridge
from src.agent.cursor_inter_llm import CursorInterLLM
from src.agent.cursor_patch_applier import CursorPatchApplier
from src.agent.cursor_query_bridge import CursorQueryBridge
from src.agent.cursor_repo_map_lookup import CursorRepoMapLookup
from src.agent.cursor_reranker import SiliconFlowReranker
from src.agent.cursor_retriever import CursorRetriever
from src.agent.cursor_semantic_tagger import CursorSemanticTagger
from src.agent.cursor_state import CursorState
from src.agent.cursor_validator import CursorValidator
from src.agent.events import (
    AgentEvent,
    EventType,
    cost_event,
    error_event,
    final_answer_event,
)
from src.agent.types import LLMResponse
from src.harness.cursor.manager import CursorStateManager
from src.llm.client import LLMClient

if TYPE_CHECKING:
    from src.config.permissions import PermissionManager
    from src.config.settings import MitKIISettings
    from src.harness.engine import HarnessEngine
    from src.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


class CursorLoop:
    """Deterministic Cursor runtime with bounded LLM decision points."""

    def __init__(
        self,
        llm: LLMClient,
        tools: ToolRegistry,
        harness: HarnessEngine,
        context: Any,
        permissions: PermissionManager,
        settings: MitKIISettings,
        *,
        inter_llm: LLMClient | None = None,
        decision_llm: LLMClient | None = None,
    ) -> None:
        self.inter_llm = inter_llm or llm
        self.decision_llm = decision_llm or llm
        self.llm = self.decision_llm
        self.tools = tools
        self.harness = harness
        self.permissions = permissions
        self.settings = settings
        self.file_tracker = getattr(context, "file_tracker", None)
        self.evaluation_expected_targets = tuple(
            getattr(context, "evaluation_expected_targets", ()) or ()
        )

        repo_map = getattr(context, "repo_map_service", None)
        self.state_manager = CursorStateManager(max_bytes=settings.cursor_state_max_bytes)
        self.state: CursorState = self.state_manager.initial("", max_steps=settings.cursor_max_steps)
        self.retriever = CursorRetriever(
            harness.project_root,
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
            harness.project_root,
            max_files=settings.cursor_max_context_files,
            max_chars_per_file=settings.cursor_context_chars_per_file,
            dependency_affinity_threshold=settings.cursor_dependency_affinity_threshold,
        )
        self.ast_structure = CursorAstStructureLayer(harness.project_root)
        self.patch_applier = CursorPatchApplier(harness.project_root)
        self.executor = CursorExecutor(harness.project_root, self.patch_applier)
        self.evaluator = CursorEvaluator(settings.cursor_evaluation_dir)
        validator_llm = self.decision_llm
        if getattr(validator_llm, "model", "") != settings.cursor_validator_model:
            validator_llm = LLMClient(
                model=settings.cursor_validator_model,
                request_timeout=float(settings.llm_request_timeout),
                prompt_cache_enabled=settings.prompt_cache_enabled,
                prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
                prompt_cache_ttl=(
                    "1h"
                    if settings.prompt_cache_ttl.strip().lower()
                    in {"1h", "hour", "60m"}
                    else "5m"
                ),
            )
        self.validator = CursorValidator(
            harness.project_root,
            command=tuple(settings.cursor_validator_command),
            timeout=settings.cursor_validator_timeout,
            max_error_chars=settings.cursor_observation_max_chars,
            semantic_llm=validator_llm,
            semantic_timeout=settings.cursor_validator_semantic_timeout,
        )
        self.decision = CursorDecisionLLM(self.decision_llm)
        self.inter = CursorInterLLM(self.inter_llm)
        self.semantic_tagger = CursorSemanticTagger()

    async def run(self, user_msg: str) -> AsyncIterator[AgentEvent]:
        yield AgentEvent(type=EventType.STREAM_START)
        self.harness.phase_metrics.reset_turn()
        self.state = self.state_manager.initial(user_msg, max_steps=self.settings.cursor_max_steps)

        hint = None
        inter_response = None
        bridge_content = None
        bridge_result = None
        guarded_bridge = None
        pending: dict[asyncio.Task[Any], str] = {}
        pm = self.harness.phase_metrics

        if self.settings.cursor_inter_enabled:
            yield self._parallel_status(
                "Understanding request...",
                phase="inter",
                task_id="inter",
                state="running",
            )
            pm.start("cursor_inter")

            async def run_inter() -> tuple[InterHint | None, LLMResponse | None]:
                inter_messages = await self.harness.before_llm_call(
                    self.inter.build_messages(user_msg)
                )
                response = cast(
                    LLMResponse,
                    await self.inter_llm.chat(
                        inter_messages,
                        tools=None,
                        stream=False,
                    ),
                )
                parsed_hint = self.inter.parse(response.content or "")
                await self.harness.after_llm_call(
                    response,
                    response.usage,
                )
                return parsed_hint, response

            pending[asyncio.create_task(run_inter())] = "inter"

        yield self._parallel_status(
            "Rewriting retrieval query...",
            phase="query_bridge",
            task_id="query_bridge",
            state="running",
        )
        yield self._log_status(
            "Query Bridge: Rewriting query to formulate structured semantic and lexical search terms...",
            phase="query_bridge",
        )
        pm.start("cursor_query_bridge")
        pending[asyncio.create_task(self.query_bridge.generate_raw(user_msg))] = (
            "query_bridge"
        )

        while pending:
            done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task_name = pending.pop(task)
                if task_name == "inter":
                    try:
                        hint, inter_response = task.result()
                    except Exception as exc:
                        log.warning("Cursor Inter call failed: %s", exc)
                    self._end_phase(
                        "cursor_inter",
                        verdict="valid" if hint is not None else "unavailable",
                    )
                    yield self._parallel_status(
                        "Request understood",
                        phase="inter",
                        task_id="inter",
                        state="done",
                    )
                    if hint:
                        yield self._log_status(
                            f"[Inter Hint] intent: {hint.intent}, domains: {hint.domains}, concepts: {hint.concepts}, ambiguity: {hint.ambiguity}, confidence: {hint.confidence}",
                            phase="inter",
                        )
                    if inter_response and inter_response.usage:
                        yield self._cost_event(inter_response)
                else:
                    try:
                        bridge_content = task.result()
                    except Exception as exc:
                        log.warning("Cursor Query Bridge call failed: %s", exc)
                    guarded_bridge = (
                        self.harness.cursor_retrieval_guardrail.validate_bridge_json(
                            bridge_content,
                            self.query_bridge.fallback(user_msg),
                        )
                    )
                    bridge_result = guarded_bridge.bridge
                    self._end_phase(
                        "cursor_query_bridge",
                        verdict="ok",
                        metadata={
                            "queries": len(bridge_result.search_terms()),
                            "json_repaired": guarded_bridge.repaired,
                            "used_fallback": guarded_bridge.used_fallback,
                            "missing_keys": list(guarded_bridge.missing_keys),
                        },
                    )
                    yield self._parallel_status(
                        "Retrieval query rewritten",
                        phase="query_bridge",
                        task_id="query_bridge",
                        state="done",
                    )
                    yield self._log_status(
                        f"[Query Bridge] Repaired and validated search criteria. Raw JSON:\n{bridge_content}",
                        phase="query_bridge",
                    )

        if hint is None:
            hint = CursorInterLLM.fallback(user_msg)

        self.context_builder.adjust_budget(hint.intent)

        if bridge_result is None:
            guarded_bridge = (
                self.harness.cursor_retrieval_guardrail.validate_bridge_json(
                    bridge_content,
                    self.query_bridge.fallback(user_msg),
                )
            )
            bridge_result = guarded_bridge.bridge
        yield self._status("Looking up repo map candidates...", phase="repo_map_lookup")
        pm.start("cursor_repo_map_lookup")
        candidate_symbols = await asyncio.to_thread(
            self.repo_map_lookup.lookup,
            bridge_result,
            limit=max(self.settings.cursor_retrieval_candidate_symbols, 40),
        )
        self._end_phase(
            "cursor_repo_map_lookup",
            verdict="ok" if candidate_symbols else "empty",
            metadata={"candidate_symbols": len(candidate_symbols)},
        )

        yield self._status("Expanding symbol graph...", phase="graph_bridge")
        yield self._log_status(
            "Graph Bridge: Expanding candidate nodes in symbol graph...",
            phase="graph_bridge",
        )
        pm.start("cursor_graph_bridge")
        graph_result = await self.graph_bridge.expand_candidates(
            candidate_symbols,
            bridge=bridge_result,
        )
        self._end_phase(
            "cursor_graph_bridge",
            verdict="ok" if graph_result.graph_nodes else "empty",
            metadata={
                "symbols": len(graph_result.expanded_symbols),
                "files": len(graph_result.expanded_files),
                "nodes": len(graph_result.graph_nodes),
                "edges": len(graph_result.graph_edges),
                "paths_top10": list(graph_result.paths[:10]),
            },
        )
        top_paths = graph_result.paths[:10]
        top_nodes = graph_result.graph_nodes[:10]
        paths_str = "\n".join(f" - {path}" for path in top_paths)
        nodes_str = "\n".join(
            " - "
            f"{node.name} "
            f"(file={node.file}, score={node.score:.4f}, distance={node.distance})"
            for node in top_nodes
        )
        yield self._log_status(
            f"[Graph Bridge] Top paths ({len(top_paths)}/{len(graph_result.paths)}):\n"
            f"{paths_str or 'None'}\n"
            f"[Graph Bridge] Top nodes ({len(top_nodes)}/{len(graph_result.graph_nodes)}):\n"
            f"{nodes_str or 'None'}",
            phase="graph_bridge",
        )

        yield self._status("Grounding AST structure...", phase="ast_structure")
        pm.start("cursor_ast_structure")
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
        self._end_phase(
            "cursor_ast_structure",
            verdict="ok" if ast_nodes else "empty",
            metadata={"ast_nodes": len(ast_nodes)},
        )

        yield self._status("Retrieving code candidates...", phase="retrieval")
        yield self._log_status(
            "Retriever: Running local similarity scoring on candidates...",
            phase="retrieval",
        )
        pm = self.harness.phase_metrics
        pm.start("cursor_retrieval")
        if self.retriever.repo_map is not None:
            raw_retrieval = self.retriever.score_candidates(
                ast_nodes=ast_nodes,
                candidates=ast_candidates,
                bridge=bridge_result,
                graph=graph_result,
            )
            guarded = None
        else:
            guarded = await self.harness.cursor_retrieval_guardrail.run(
                bridge_result.search_terms(limit=32),
                self.retriever.retrieve,
            )
            raw_retrieval = guarded.retrieval
            log.warning("FINAL QUERY: %s", guarded.queries)
        self._end_phase(
            "cursor_retrieval",
            verdict="timeout" if guarded is not None and guarded.timed_out else "ok",
            metadata={
                "files": len(raw_retrieval.files),
                "symbols": len(raw_retrieval.symbols),
                "queries": len(guarded.queries) if guarded is not None else 0,
                "batches": guarded.batches_started if guarded is not None else 0,
                "early_stop": guarded.stopped_early if guarded is not None else False,
                "timed_out": guarded.timed_out if guarded is not None else False,
            },
        )
        retrieval_files_str = "\n".join(f" - {f}" for f in raw_retrieval.files)
        retrieval_syms_str = "\n".join(
            f" - {sym.name} ({sym.file}:{sym.start_line}-{sym.end_line}) [score={sym.score:.4f}, reasons={list(sym.reasons)}]"
            for sym in raw_retrieval.symbols
        )
        yield self._log_status(
            f"[Retriever v2] Scored files:\n{retrieval_files_str or 'None'}\n[Retriever v2] Scored symbols:\n{retrieval_syms_str or 'None'}",
            phase="retrieval",
        )

        yield self._status("Fusing and ranking context...", phase="fusion")
        yield self._log_status(
            "Fusion Engine: Ranking, capping, and selecting final context slices...",
            phase="fusion",
        )
        pm.start("cursor_fusion")
        pm.start("cursor_rerank")
        fusion_result = await self.fusion.decide_async(
            raw_retrieval,
            bridge_result,
            max_files=self.settings.cursor_retrieval_max_files,
            max_symbols=self.settings.cursor_retrieval_max_symbols,
            top_k=min(8, self.settings.cursor_retrieval_max_symbols),
            user_intent=user_msg,
        )
        rerank_summary = fusion_result.rerank
        self._end_phase(
            "cursor_rerank",
            verdict=str(rerank_summary.get("status", "unknown")),
            metadata={
                "enabled": bool(rerank_summary.get("enabled", False)),
                "model": rerank_summary.get("model"),
                "candidates": rerank_summary.get("candidate_count", 0),
                "scored": rerank_summary.get("scored_count", 0),
                "rerank_duration_ms": rerank_summary.get("duration_ms", 0.0),
            },
        )
        retrieval = fusion_result.retrieval
        self._end_phase(
            "cursor_fusion",
            verdict="ok",
            metadata={
                "files": len(retrieval.files),
                "symbols": len(retrieval.symbols),
                "confidence": fusion_result.confidence,
                "final_context": list(fusion_result.final_context),
                "rerank_status": rerank_summary.get("status"),
            },
        )
        yield self._log_status(
            _format_rerank_summary(rerank_summary),
            phase="fusion",
        )
        yield self._log_status(
            f"[Fusion Engine] Confidence: {fusion_result.confidence:.4f}\n[Fusion Engine] Selected final context items:\n" +
            "\n".join(f" - {item}" for item in fusion_result.final_context),
            phase="fusion",
        )
        layer1_metrics = compute_layer1_metrics(
            tuple(fusion_result.final_context),
            self.evaluation_expected_targets,
        )
        layer1_note = (
            "oracle unavailable"
            if not self.evaluation_expected_targets
            else f"{len(layer1_metrics.hits)}/{len(layer1_metrics.expected)}"
        )
        yield self._log_status(
            "[Evaluation] Layer 1 retrieval metrics:\n"
            f" - Precision: {layer1_metrics.precision:.4f} "
            f"({len(layer1_metrics.hits)}/{len(layer1_metrics.retrieved)})\n"
            f" - Recall   : {layer1_metrics.recall:.4f} ({layer1_note})\n"
            f" - F1-Score : {layer1_metrics.f1_score:.4f}\n"
            f" - Hits     : {list(layer1_metrics.hits)}\n"
            f" - Misses   : {list(layer1_metrics.misses)}",
            phase="fusion",
        )
        if self.evaluation_expected_targets and layer1_metrics.recall < 1.0:
            self.state = self.state_manager.failed(
                self.state,
                "Layer 1 retrieval recall below CI threshold",
            )
            yield self._log_status(
                "[Evaluation Gate] Layer 1 recall below 1.0; blocking DecisionLLM.",
                phase="fusion",
            )
            yield final_answer_event("Layer 1 retrieval recall below CI threshold")
            yield AgentEvent(type=EventType.STREAM_END)
            return

        if not retrieval.files and not retrieval.symbols:
            self.state = self.state_manager.failed(
                self.state,
                "No matching code found",
            )
            yield final_answer_event("No matching code found")
            yield AgentEvent(type=EventType.STREAM_END)
            return
        pm.start("cursor_context")
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
        self._end_phase(
            "cursor_context",
            verdict="ok",
            metadata={"context_files": len(context_pack.windows)},
        )

        yield AgentEvent(
            type=EventType.STATUS,
            content=f"Retrieved {len(context_pack.windows)} candidate file(s).",
            data={
                "phase": "retrieval",
                "spinner_only": False,
                "files": list(context_pack.candidate_files),
            },
        )

        for step in range(1, self.settings.cursor_max_steps + 1):
            yield self._status(
                f"Decision step {step}/{self.settings.cursor_max_steps}...",
                phase="decision",
            )
            yield self._log_status(
                f"Decision: Formatting current Kanban board state and context pack for LLM inference (step {step}/{self.settings.cursor_max_steps})...",
                phase="decision",
            )
            kanban_state = self.state_manager.format_for_prompt(self.state)
            yield self._log_status(
                f"[Decision] Injected Kanban State:\n{kanban_state}",
                phase="decision",
            )

            try:
                pm.start("cursor_decision", subtask_id=str(step))
                decision_messages = self.decision.build_messages(
                    state_text=kanban_state,
                    context_pack=context_pack,
                    hint=hint,
                )
                trimmed_messages = await self.harness.before_llm_call(decision_messages)
                response = cast(
                    LLMResponse,
                    await self.decision_llm.chat(
                        trimmed_messages,
                        tools=None,
                        stream=False,
                    ),
                )
                await self.harness.after_llm_call(response, response.usage)
                decision = self.decision.parse(
                    response.content or "",
                    context_pack.candidate_files,
                )
                self._end_phase(
                    "cursor_decision",
                    subtask_id=str(step),
                    verdict=decision.action,
                )
                yield self._log_status(
                    f"[Decision] Raw JSON Output:\n{response.content}",
                    phase="decision",
                )
                yield self._log_status(
                    f"[Decision] Parsed action: {decision.action}, target_file: {decision.target_file or 'None'}, suggested_completion: {decision.suggested_completion}",
                    phase="decision",
                )
            except DecisionError as exc:
                self._end_phase(
                    "cursor_decision",
                    subtask_id=str(step),
                    verdict="schema_error",
                )
                yield self._log_status(
                    f"[Decision] Failed to parse response due to schema error: {exc}",
                    phase="decision",
                )
                self.state = self.state_manager.observe(self.state, str(exc))
                continue
            except Exception as exc:
                self._end_phase(
                    "cursor_decision",
                    subtask_id=str(step),
                    verdict="error",
                )
                yield self._log_status(
                    f"[Decision] Exception encountered: {exc}",
                    phase="decision",
                )
                log.warning("Cursor decision call failed: %s", exc)
                self.state = self.state_manager.observe(
                    self.state,
                    f"decision_error: {exc}",
                )
                continue
            if response.usage:
                yield self._cost_event(response)

            can_answer = len(context_pack.windows) > 0
            self.state = self.state_manager.record_decision(
                self.state,
                action=decision.action,
                target_file=decision.target_file,
                can_answer=can_answer,
            )

            if decision.action == "ask_clarify":
                if can_answer:
                    self.state = self.state_manager.mark_retry(self.state)
                    yield self._log_status(
                        "[Decision] ask_clarify recorded as a loop retry because "
                        "grounded context is available.",
                        phase="decision",
                    )
                    continue
                self.state = self.state_manager.failed(
                    self.state,
                    decision.clarification,
                )
                yield final_answer_event(decision.clarification)
                yield AgentEvent(type=EventType.STREAM_END)
                return

            if decision.action == "answer":
                self.state = self.state_manager.succeeded(self.state)
                yield final_answer_event(decision.answer)
                yield AgentEvent(type=EventType.STREAM_END)
                return

            yield self._status(
                f"Applying patch to {decision.target_file}...",
                phase="executor",
            )
            yield self._log_status(
                f"[Sandbox] Entering transaction context for target file '{decision.target_file}'...",
                phase="executor",
            )
            pm.start("cursor_executor", subtask_id=str(step))
            execution, validation, pipeline_metrics = await self.executor.execute_transaction(
                decision.target_file,
                decision.patch,
                self.validator,
                step=step,
                layer1=layer1_metrics,
                user_intent=user_msg,
            )
            self.evaluator.record(pipeline_metrics)
            yield self._log_status(
                format_bi_report(pipeline_metrics),
                phase="evaluation",
            )
            self._end_phase(
                "cursor_executor",
                subtask_id=str(step),
                verdict="pass" if execution.success else "fail",
                metadata={
                    "file": decision.target_file,
                    "patch_correctness": pipeline_metrics.layer2.patch_correctness,
                    "execution_success": pipeline_metrics.layer2.execution_success,
                    "code_diff_correctness": (
                        pipeline_metrics.layer2.code_diff_correctness
                    ),
                },
            )
            if not execution.success:
                yield self._log_status(
                    f"[Sandbox] Patch application failed: {execution.error or 'Unknown error'}",
                    phase="executor",
                )
                yield self._log_status(
                    f"[Sandbox] Transaction rolled back for target file '{decision.target_file}'.",
                    phase="executor",
                )
                self.state = self.state_manager.after_execution(
                    self.state,
                    decision.target_file,
                    decision.patch,
                    execution,
                )
                continue

            yield self._log_status(
                f"[Sandbox] Patch successfully applied to '{decision.target_file}'. Proceeding to validation check...",
                phase="executor",
            )

            self._end_phase(
                "cursor_validator",
                subtask_id=str(step),
                verdict="pass" if validation.success else "fail",
            )
            if validation.success:
                yield self._log_status(
                    f"[Sandbox] Validation passed. Transaction committed for target file '{decision.target_file}'.",
                    phase="executor",
                )
            else:
                yield self._log_status(
                    f"[Sandbox] Validation failed! Error:\n{validation.error or 'Unknown validation error'}",
                    phase="executor",
                )
                yield self._log_status(
                    f"[Sandbox] Transaction rolled back for target file '{decision.target_file}'.",
                    phase="executor",
                )

            self.state = self.state_manager.after_execution(
                self.state,
                decision.target_file,
                decision.patch,
                execution,
            )
            self.state = self.state_manager.after_validation(
                self.state,
                validation,
                suggested_completion=decision.suggested_completion,
                patch=decision.patch,
                execution=execution,
            )
            if not validation.success:
                continue

            if self.file_tracker is not None:
                self.file_tracker.record_edit(decision.target_file)

            if self.state.stage_completion >= 1.0 or validation.success:
                yield final_answer_event(
                    f"Modified and validated {decision.target_file}.",
                    changed_file=decision.target_file,
                )
                yield AgentEvent(type=EventType.STREAM_END)
                return

        self.state = self.state_manager.failed(
            self.state,
            self.state.last_observation or "Cursor loop exhausted its step limit.",
        )
        yield error_event(self.state.last_observation)
        yield AgentEvent(type=EventType.STREAM_END)

    def _cost_event(self, response: Any) -> AgentEvent:
        record = self.harness.probe.metrics.last_record
        cost = float(record.cost) if record is not None else 0.0
        return cost_event(
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            cost,
        )

    def _end_phase(
        self,
        phase: str,
        *,
        subtask_id: str | None = None,
        verdict: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        record = self.harness.phase_metrics.end(
            phase,
            subtask_id=subtask_id,
            verdict=verdict,
            metadata=metadata,
        )
        log.info(
            "Cursor phase %s completed in %.1fms verdict=%s metadata=%s",
            phase,
            record.duration_ms,
            verdict,
            metadata or {},
        )

    @staticmethod
    def _status(content: str, *, phase: str) -> AgentEvent:
        return AgentEvent(
            type=EventType.STATUS,
            content=content,
            data={"spinner_only": True, "phase": phase},
        )

    @staticmethod
    def _log_status(content: str, *, phase: str) -> AgentEvent:
        return AgentEvent(
            type=EventType.STATUS,
            content=f"\033[90m{content}\033[0m",
            data={"spinner_only": False, "phase": phase},
        )

    @staticmethod
    def _parallel_status(
        content: str,
        *,
        phase: str,
        task_id: str,
        state: str,
    ) -> AgentEvent:
        return AgentEvent(
            type=EventType.STATUS,
            content=content,
            data={
                "spinner_only": True,
                "llm_loading": state == "running",
                "phase": phase,
                "parallel_task_id": task_id,
                "parallel_state": state,
            },
        )

    async def resolve_approval(self, action: str, approved: bool) -> None:
        return None

    async def list_checkpoints(self) -> list[dict[str, Any]]:
        return []

    def get_probe_metrics(self) -> dict[str, Any]:
        usage = self.harness.probe.metrics.get_summary()
        phases = self.harness.phase_metrics.get_summary()
        return {**usage, **phases}

    async def run_score_now(self) -> dict[str, Any] | None:
        return None


def _format_rerank_summary(summary: dict[str, object]) -> str:
    status = str(summary.get("status", "unknown"))
    if status == "disabled":
        return "[Reranker] disabled; using local fusion order"
    if status == "skipped":
        return f"[Reranker] skipped: {summary.get('reason', 'no candidates')}"

    model = str(summary.get("model") or "custom")
    duration = summary.get("duration_ms", 0.0)
    candidates = summary.get("candidate_count", 0)
    scored = summary.get("scored_count", 0)
    top_items = summary.get("top_symbols") or []
    top = []
    if isinstance(top_items, list):
        for item in top_items[:3]:
            if not isinstance(item, dict):
                continue
            file = str(item.get("file", "?"))
            symbol = str(item.get("symbol", "?")).split(":")[0]
            score = item.get("score", 0.0)
            top.append(f"{file}:{symbol}({score})")
    top_text = ", ".join(top) if top else "-"
    return (
        f"[Reranker] {status} model={model} "
        f"scored={scored}/{candidates} duration={duration}ms top=[{top_text}]"
    )
