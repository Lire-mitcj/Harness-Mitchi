from __future__ import annotations

import re
from typing import Any
from src.agent.cursor_contracts import ControlEvent


class ReactiveController:

    def __init__(self, graph_engine: Any, context_builder: Any, state_manager: Any) -> None:
        self.graph_engine = graph_engine
        self.context_builder = context_builder
        self.state_manager = state_manager
        self._handlers = {
            "missing_info": self._handle_missing_info,
            "low_confidence": self._handle_low_confidence,
            "runtime_error": self._handle_runtime_error,
            "clarify_escape": self._handle_clarify_escape,
            "execution_success": self._handle_success,
        }

    async def handle(self, event: ControlEvent, context_pack: Any, state: Any) -> tuple[Any, Any, bool]:
        handler = self._handlers.get(event.type)
        if not handler:
            return context_pack, state, False
        return await handler(event.payload, event.severity, context_pack, state)

    async def _handle_missing_info(
        self,
        payload: dict,
        severity: float,
        context_pack: Any,
        state: Any,
    ) -> tuple[Any, Any, bool]:
        seed = payload.get("symbol_name")
        fusion_result = await self.graph_engine.compile_dependency_subgraph(seed, depth=2)
        # 🚀 显式解包 `.retrieval` 契约，并交付给 merge_interval_subgraph
        updated_context = self.context_builder.merge_interval_subgraph(context_pack, fusion_result.retrieval)
        next_state = self.state_manager.observe_failure_signature(
            state, action="missing_info", file=seed, error_type="missing_info_signal"
        )
        next_state = self.state_manager.apply_time_decay(next_state)
        return updated_context, next_state, True

    async def _handle_low_confidence(
        self,
        payload: dict,
        severity: float,
        context_pack: Any,
        state: Any,
    ) -> tuple[Any, Any, bool]:
        patch = payload.get("patch", "")
        extracted_seed = self._parse_missing_span(patch)
        if extracted_seed:
            payload["symbol_name"] = extracted_seed
            return await self._handle_missing_info(payload, severity, context_pack, state)
        
        target_file = payload.get("file")
        if target_file:
            fusion_result = await self.graph_engine.compile_dependency_subgraph(target_file, depth=2)
            context_pack = self.context_builder.merge_interval_subgraph(context_pack, fusion_result.retrieval)
            
        return context_pack, state, True

    async def _handle_runtime_error(
        self,
        payload: dict,
        severity: float,
        context_pack: Any,
        state: Any,
    ) -> tuple[Any, Any, bool]:
        error_trace = payload.get("error", "validator_rejected")
        target_file = payload.get("file", "unknown")
        
        fusion_result = await self.graph_engine.compile_dependency_subgraph(target_file, depth=2)
        updated_context = self.context_builder.merge_interval_subgraph(context_pack, fusion_result.retrieval)
        
        next_state = self.state_manager.observe_failure_signature(
            state,
            action="edit",
            file=target_file,
            error_type=str(error_trace),
        )
        next_state = self.state_manager.apply_time_decay(next_state)
        return updated_context, next_state, True

    async def _handle_success(
        self,
        payload: dict,
        severity: float,
        context_pack: Any,
        state: Any,
    ) -> tuple[Any, Any, bool]:
        next_state = self.state_manager.apply_time_decay(state)
        return context_pack, next_state, False

    async def _handle_clarify_escape(
        self,
        payload: dict,
        severity: float,
        context_pack: Any,
        state: Any,
    ) -> tuple[Any, Any, bool]:
        clarification_text = payload.get("clarification", "")

        candidates = []
        filter_words = {
            "select", "from", "where", "join", "left", "right", "inner", "outer", "on",
            "as", "and", "or", "count", "group", "by", "order", "limit", "offset", "into",
            "update", "insert", "delete", "create", "view", "table", "return", "for", "in",
            "file", "code", "path", "info", "text", "ask", "clarify", "interface", "api",
            "dialog", "class", "def", "async", "await", "import", "from", "none", "true",
            "false", "null", "passenger", "ticket", "order", "user", "admin", "get", "post",
            "put", "patch", "route", "controller", "service", "repository", "model",
            "database", "db", "sql", "query", "connection", "conn", "execute", "cursor",
            "result", "response", "request", "param", "params", "argument", "args", "kwargs",
            "function", "method", "variable", "var", "object", "obj", "exception", "error",
            "failed", "success", "fail", "pass", "run", "start", "stop", "close", "open",
            "read", "write", "print", "log", "logger", "debug", "warning", "warn", "critical",
            "fatal", "exception", "please", "provide", "complete", "the", "and", "but", "has",
            "had", "have", "this", "that", "with", "you", "your", "they", "them", "his", "her",
            "its", "our", "are", "was", "were", "been", "does", "did", "can", "could", "will",
            "would", "should", "may", "might", "must", "shall", "about", "above", "after", "again",
            "against", "all", "am", "any", "at", "be", "because", "before", "being", "below",
            "between", "both", "by", "down", "during", "each", "few", "further", "having", "he",
            "her", "here", "hers", "herself", "him", "himself", "his", "how", "if", "is", "it",
            "itself", "more", "most", "my", "myself", "no", "nor", "not", "of", "off", "once",
            "only", "other", "ought", "ours", "ourselves", "out", "over", "own", "same", "she",
            "so", "some", "such", "than", "their", "theirs", "themselves", "then", "there",
            "these", "those", "through", "to", "too", "under", "until", "up", "very", "we",
            "what", "when", "where", "which", "while", "who", "whom", "why", "yourself", "yourselves"
        }

        for text in (clarification_text, getattr(state, "task", "")):
            for word in re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]{2,})\b", text):
                if word.lower() not in filter_words:
                    candidates.append(word)

        unique_candidates = []
        for c in candidates:
            if c not in unique_candidates:
                unique_candidates.append(c)

        if unique_candidates:
            inferred_seed = unique_candidates[0]
        else:
            inferred_seed = "passenger_snapshot" if "旅客" in clarification_text or "搜索" in clarification_text else "passenger"

        fusion_result = await self.graph_engine.compile_dependency_subgraph(inferred_seed, depth=2)
        updated_context = self.context_builder.merge_interval_subgraph(context_pack, fusion_result.retrieval)
        next_state = self.state_manager.observe_failure_signature(
            state, action="ask_clarify", file=inferred_seed, error_type="unjustified_clarify_escape"
        )
        next_state = self.state_manager.mark_retry(next_state, clarification_text)
        next_state = self.state_manager.apply_time_decay(next_state)
        return updated_context, next_state, True

    @staticmethod
    def _parse_missing_span(patch: str) -> str | None:
        match = re.search(r"<<<<<<< SEARCH\n.*?([\w_]+).*?\n=======", patch, re.DOTALL)
        if match:
            seed = match.group(1).strip()
            if len(seed) >= 3 and seed.lower() not in {"select", "from", "join", "where"}:
                return seed
        return None
