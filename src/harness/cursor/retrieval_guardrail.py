from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Literal, cast

from src.agent.contracts import RetrievalResult, RetrievalSymbol
from src.tools.assembled.query_bridge import QueryBridgeResult

log = logging.getLogger(__name__)

RetrieveBatch = Callable[[Sequence[str]], Awaitable[RetrievalResult]]


@dataclass(frozen=True, slots=True)
class GuardedRetrievalResult:
    retrieval: RetrievalResult
    queries: tuple[str, ...]
    batches_started: int
    stopped_early: bool
    timed_out: bool


@dataclass(frozen=True, slots=True)
class GuardedBridgeResult:
    bridge: QueryBridgeResult
    repaired: bool
    used_fallback: bool
    missing_keys: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RetrievalGuardrailPolicy:
    """Immutable mechanical limits; contains no semantic retrieval signals."""

    query_cap: int = 12
    fan_out: int = 4
    timeout: float = 12.0
    early_stop_candidates: int = 8

    def __post_init__(self) -> None:
        if self.query_cap < 1:
            raise ValueError("query_cap must be positive")
        if self.fan_out < 1:
            raise ValueError("fan_out must be positive")
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        if self.early_stop_candidates < 1:
            raise ValueError("early_stop_candidates must be positive")


class CursorRetrievalGuardrail:
    """Apply deterministic mechanical limits after Query Bridge.

    This layer must not inspect intent, domain, meaning, similarity, file hints,
    or source content. It preserves input order and only applies whitespace
    trimming, exact case-insensitive deduplication, fixed batching, a wall-clock
    deadline, and a unique-candidate-count stop threshold.
    """

    REQUIRED_BRIDGE_KEYS = (
        "intent",
        "expanded_terms",
        "keywords",
        "symbols",
        "file_hints",
    )
    VALID_INTENTS = frozenset({"list", "query", "modify", "debug", "explain"})
    SAFE_DEFAULT_TERMS = ("code", "module", "class", "function")

    def __init__(
        self,
        *,
        query_cap: int = 12,
        fan_out: int = 4,
        timeout: float = 12.0,
        early_stop_candidates: int = 8,
    ) -> None:
        self.policy = RetrievalGuardrailPolicy(
            query_cap=query_cap,
            fan_out=fan_out,
            timeout=timeout,
            early_stop_candidates=early_stop_candidates,
        )

    def normalize(self, terms: Sequence[str]) -> tuple[str, ...]:
        normalized: list[str] = []
        seen: set[str] = set()
        for raw in terms:
            term = raw.strip()
            key = term.casefold()
            if len(term) < 2 or key in seen:
                continue
            seen.add(key)
            normalized.append(term)
            if len(normalized) >= self.policy.query_cap:
                break
        return tuple(normalized)

    def validate_bridge_json(
        self,
        content: str | None,
        fallback: QueryBridgeResult,
    ) -> GuardedBridgeResult:
        data, repaired = self._repair_json_object(content or "")
        if data is None:
            return self._fallback_bridge(fallback, repaired=True)

        missing = tuple(key for key in self.REQUIRED_BRIDGE_KEYS if key not in data)
        if len(missing) == len(self.REQUIRED_BRIDGE_KEYS):
            return self._fallback_bridge(
                fallback,
                repaired=True,
                missing_keys=missing,
            )
        schema_repaired = bool(missing)
        raw_intent = data.get("intent")
        if raw_intent not in self.VALID_INTENTS:
            intent = fallback.intent
            schema_repaired = True
        else:
            intent = cast(
                Literal["list", "query", "modify", "debug", "explain"],
                raw_intent,
            )

        fields: dict[str, list[str]] = {}
        for key in self.REQUIRED_BRIDGE_KEYS[1:]:
            value = data.get(key)
            if key not in data or not isinstance(value, list):
                value = getattr(fallback, key)
                schema_repaired = True
            fields[key] = self._string_list(value)

        bridge = QueryBridgeResult(
            intent=intent,
            expanded_terms=fields["expanded_terms"],
            keywords=fields["keywords"],
            symbols=[],
            file_hints=fields["file_hints"],
            domain=str(data.get("domain") or fallback.domain or ""),
            concepts=self._string_list(data.get("concepts", fallback.concepts or [])),
            constraints=self._constraints(data.get("constraints"), fallback.constraints),
        )
        if not self.normalize(bridge.search_terms(limit=self.policy.query_cap)):
            return self._fallback_bridge(
                fallback,
                repaired=repaired or schema_repaired,
                missing_keys=missing,
            )
        return GuardedBridgeResult(
            bridge=bridge,
            repaired=repaired or schema_repaired,
            used_fallback=False,
            missing_keys=missing,
        )

    def _fallback_bridge(
        self,
        fallback: QueryBridgeResult,
        *,
        repaired: bool,
        missing_keys: tuple[str, ...] = (),
    ) -> GuardedBridgeResult:
        log.warning("CursorQueryBridge falling back to local heuristic: %s", fallback)
        if self.normalize(fallback.search_terms(limit=self.policy.query_cap)):
            bridge = fallback
        else:
            bridge = QueryBridgeResult(
                intent=fallback.intent,
                expanded_terms=list(self.SAFE_DEFAULT_TERMS),
                keywords=[],
                symbols=[],
                file_hints=[],
                domain=fallback.domain,
                concepts=fallback.concepts,
                constraints=fallback.constraints,
            )
        return GuardedBridgeResult(
            bridge=bridge,
            repaired=repaired,
            used_fallback=True,
            missing_keys=missing_keys,
        )

    def _string_list(self, value: object) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(self.normalize(tuple(item for item in value if isinstance(item, str))))

    def _constraints(
        self,
        value: object,
        fallback: dict[str, list[str]] | None,
    ) -> dict[str, list[str]]:
        base = fallback or {"layer_hint": [], "exclude": []}
        if not isinstance(value, dict):
            return base
        return {
            "layer_hint": self._string_list(value.get("layer_hint", base.get("layer_hint", []))),
            "exclude": self._string_list(value.get("exclude", base.get("exclude", []))),
        }

    @staticmethod
    def _repair_json_object(content: str) -> tuple[dict[str, object] | None, bool]:
        text = content.strip()
        if not text:
            return None, False
        repaired = False
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            extracted = text[start:end + 1]
            repaired = extracted != text
            text = extracted
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            repaired = True
            text = re.sub(
                r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:",
                r'\1"\2":',
                text,
            )
            # Repair mismatched single/double quotes around strings
            text = re.sub(r'["\']([^"\']*)["\']', r'"\1"', text)
            text = re.sub(r"'([^']*)'", r'"\1"', text)
            text = re.sub(r",\s*([}\]])", r"\1", text)
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                return None, repaired
        if not isinstance(value, dict):
            return None, repaired
        return value, repaired

    async def run(
        self,
        terms: Sequence[str],
        retrieve_batch: RetrieveBatch,
    ) -> GuardedRetrievalResult:
        queries = self.normalize(terms)
        if not queries:
            queries = self.normalize(self.SAFE_DEFAULT_TERMS)

        files: list[str] = []
        symbols: list[RetrievalSymbol] = []
        batches_started = 0
        stopped_early = False
        timed_out = False

        async def retrieve_all() -> None:
            nonlocal batches_started, stopped_early
            for offset in range(0, len(queries), self.policy.fan_out):
                if (
                    self._candidate_count(files, symbols)
                    >= self.policy.early_stop_candidates
                ):
                    stopped_early = True
                    break
                batch = queries[offset:offset + self.policy.fan_out]
                batches_started += 1
                result = await retrieve_batch(batch)
                self._extend_unique(files, result.files)
                self._extend_unique(symbols, result.symbols)

        try:
            async with asyncio.timeout(self.policy.timeout):
                await retrieve_all()
        except TimeoutError:
            timed_out = True
            log.warning(
                "Cursor retrieval guardrail timed out after %.2fs; returning partial results",
                self.policy.timeout,
            )

        return GuardedRetrievalResult(
            retrieval=RetrievalResult(files=tuple(files), symbols=tuple(symbols)),
            queries=queries,
            batches_started=batches_started,
            stopped_early=stopped_early,
            timed_out=timed_out,
        )

    @staticmethod
    def _candidate_count(files: Sequence[str], symbols: Sequence[RetrievalSymbol]) -> int:
        candidate_files = set(files)
        candidate_files.update(symbol.file for symbol in symbols)
        return len(candidate_files)

    @staticmethod
    def _extend_unique[T](target: list[T], values: Sequence[T]) -> None:
        seen = set(target)
        for value in values:
            if value not in seen:
                seen.add(value)
                target.append(value)
