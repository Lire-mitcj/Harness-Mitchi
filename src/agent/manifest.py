"""Step Evidence Manifest: verifiable evidence tracking for the current step.

This module is the pure data-model layer of the manifest architecture. It has
no dependency on hooks, the loop, or the reducer -- it only defines the manifest
schema and the pure functions that:

  * project item status (MISSING | SATISFIED | STALE) from durable code anchors,
  * compute step sufficiency from item status,
  * render the short "execution card" that is injected into the Core LLM context.

Status is *computed from verifiable evidence* (verbatim code anchors), never from
inference tags. A grep hit alone never satisfies an item; only a durable anchor
that carries verbatim code covering the target (span/symbol/schema) does.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from src.agent.evidence_slots import rule_matches, slot_def, slot_satisfy_rule
from src.agent.grep_discovery import discovery_hint_lines
from src.agent.wiring import (
    caller_has_setup_evidence,
    default_wiring_probe_symbol,
    grep_patterns_for_wiring,
    integration_layer,
    mount_setup_confirmed,
    peer_layers_skip_wiring,
    task_needs_integration_wiring,
    wiring_gap_message,
    wiring_probe_symbols,
)
from src.indexer.language_profiles import ALL_PROFILES, SQL, extract_symbol_with_profiles


class Sufficiency(StrEnum):
    INSUFFICIENT = "INSUFFICIENT"
    SUFFICIENT_FOR_EDIT = "SUFFICIENT_FOR_EDIT"
    SUFFICIENT_FOR_VERIFY = "SUFFICIENT_FOR_VERIFY"
    SUFFICIENT_FOR_ANSWER = "SUFFICIENT_FOR_ANSWER"


ItemStatus = Literal["MISSING", "SATISFIED", "STALE"]
ItemType = Literal["span", "symbol", "schema", "mount", "test_failure", "tool_error"]
ItemRole = Literal["required", "observed"]
StepKind = Literal["retrieval", "edit", "verify", "answer"]

FAILURE_TYPES: frozenset[str] = frozenset({"test_failure", "tool_error"})

# Max symbols listed in the STEP EVIDENCE "loaded" inventory. Must comfortably
# exceed the number of slices a multi-step edit accumulates; otherwise the Core
# LLM cannot see everything it already holds and re-fetches the hidden overflow.
_LOADED_INVENTORY_CAP = 30


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """A single verifiable evidence requirement for the current step."""

    id: str
    need: str
    type: ItemType = "symbol"
    role: ItemRole = "required"
    file: str | None = None
    span: tuple[int, int] | None = None
    symbol: str | None = None
    status: ItemStatus = "MISSING"
    provenance: str | None = None
    content_hash: str | None = None
    stale_reason: str | None = None
    # Internal-only matching keywords used by the projection to decide coverage
    # for need-level items whose concrete target is not yet known.
    keywords: tuple[str, ...] = ()

    @property
    def is_failure(self) -> bool:
        return self.type in FAILURE_TYPES


@dataclass(frozen=True, slots=True)
class StepManifest:
    """System-maintained checklist of evidence for the current step."""

    step_id: str = "task.default"
    step_kind: StepKind = "retrieval"
    required_items: tuple[EvidenceItem, ...] = ()
    sufficiency: Sufficiency = Sufficiency.INSUFFICIENT
    updated_at_step: int = 0
    notes: tuple[str, ...] = ()

    @property
    def core_items(self) -> tuple[EvidenceItem, ...]:
        """Required obligations only; observations never affect sufficiency."""
        return tuple(
            item for item in self.required_items
            if not item.is_failure and item.role == "required"
        )

    @property
    def observed_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in self.required_items if item.role == "observed")

    @property
    def missing_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in self.core_items if item.status == "MISSING")

    @property
    def stale_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in self.required_items if item.status == "STALE")

    @property
    def failure_items(self) -> tuple[EvidenceItem, ...]:
        return tuple(item for item in self.required_items if item.is_failure)

    @property
    def has_missing(self) -> bool:
        return bool(self.missing_items)

    @property
    def has_stale(self) -> bool:
        return bool(self.stale_items)


_ITEM_WEIGHTS: dict[str, float] = {
    "schema": 3.0,
    "symbol": 2.0,
    "mount": 2.0,
    "span": 1.0,
}


@dataclass(frozen=True, slots=True)
class ManifestMetrics:
    coverage: float
    missing_ratio: float
    stale_ratio: float
    required_count: int
    critical_missing: bool


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    """Manifest-derived retrieval needs for the current step."""

    needs_grep: bool
    needs_view: bool
    needs_heavy: bool
    bootstrap: bool


def _item_has_concrete_target(item: EvidenceItem) -> bool:
    return bool(item.file) and bool(item.symbol or item.span)


def stale_needing_refresh(manifest: StepManifest) -> tuple[EvidenceItem, ...]:
    """STALE items that lack a fresh SATISFIED anchor for the same file+symbol."""
    fresh = {
        (_norm_file(item.file), item.symbol)
        for item in manifest.required_items
        if item.status == "SATISFIED" and item.file and item.symbol
    }
    needing: list[EvidenceItem] = []
    for item in manifest.stale_items:
        if not item.file:
            continue
        if not item.symbol:
            needing.append(item)
            continue
        if (_norm_file(item.file), item.symbol) not in fresh:
            needing.append(item)
    return tuple(needing)


def wiring_gap_lines(
    manifest: StepManifest,
    *,
    task_text: str = "",
) -> tuple[str, ...]:
    """Caller/support files that lack entry/integration spans block cross-file bootstrap."""
    if not task_needs_integration_wiring(task_text):
        return ()

    grounded = _grounded_observed_items(manifest)
    if not grounded:
        return ()

    symbol_files: dict[str, list[EvidenceItem]] = {}
    for item in grounded:
        if item.type != "symbol" or not item.file:
            continue
        norm = _norm_file(item.file)
        symbol_files.setdefault(norm, []).append(item)

    if len(symbol_files) < 2:
        return ()

    edit_target = _likely_edit_target_file(
        grounded,
        task_text=task_text,
        manifest=manifest,
    )
    edit_norm = _norm_file(edit_target) if edit_target else None
    edit_peer_skip = peer_layers_skip_wiring(edit_target or "")

    gaps: list[str] = []
    for norm, items in symbol_files.items():
        if edit_norm and norm == edit_norm:
            continue
        file_label = items[0].file or norm
        if edit_peer_skip and peer_layers_skip_wiring(file_label):
            continue
        if caller_has_setup_evidence(items, file_path=file_label):
            continue
        gaps.append(wiring_gap_message(file_label))
    return tuple(gaps)


def _observed_line_count(item: EvidenceItem) -> int:
    if item.span is None:
        return 0
    return item.span[1] - item.span[0] + 1


def _grounded_observed_items(manifest: StepManifest) -> tuple[EvidenceItem, ...]:
    return tuple(
        item
        for item in manifest.observed_items
        if item.status in {"SATISFIED", "STALE"}
    )


def _likely_edit_target_file(
    grounded: Sequence[EvidenceItem],
    *,
    task_text: str = "",
    manifest: StepManifest | None = None,
) -> str | None:
    best_file: str | None = None
    best_lines = 0
    for item in grounded:
        if item.type != "symbol" or not item.file:
            continue
        lines = _observed_line_count(item)
        if lines > best_lines:
            best_lines = lines
            best_file = item.file
    if best_file:
        return best_file
    if manifest is not None and task_text.strip():
        targets = primary_edit_target_files(manifest, task_text)
        if targets:
            return targets[0]
    return None


def wiring_gap_pending_loads(
    manifest: StepManifest,
    *,
    task_text: str = "",
) -> tuple[dict[str, object], ...]:
    """Concrete retrieval actions for caller files that still lack mount/setup spans."""
    if not wiring_gap_lines(manifest, task_text=task_text):
        return ()

    grounded = _grounded_observed_items(manifest)
    symbol_files: dict[str, list[EvidenceItem]] = {}
    for item in grounded:
        if item.type != "symbol" or not item.file:
            continue
        norm = _norm_file(item.file)
        symbol_files.setdefault(norm, []).append(item)

    edit_target = _likely_edit_target_file(
        grounded,
        task_text=task_text,
        manifest=manifest,
    )
    edit_norm = _norm_file(edit_target) if edit_target else None

    loads: list[dict[str, object]] = []
    edit_peer_skip = peer_layers_skip_wiring(edit_target or "")
    for norm, items in symbol_files.items():
        if edit_norm and norm == edit_norm:
            continue
        file_label = str(items[0].file or norm)
        if edit_peer_skip and peer_layers_skip_wiring(file_label):
            continue
        if caller_has_setup_evidence(items, file_path=file_label):
            continue
        loaded_syms = {
            str(item.symbol)
            for item in items
            if item.symbol and item.status in {"SATISFIED", "STALE"}
        }
        probes = wiring_probe_symbols(file_label)
        hint = next((sym for sym in probes if sym not in loaded_syms), None)
        if hint is None:
            hint = default_wiring_probe_symbol(file_label)
        patterns = grep_patterns_for_wiring(file_label)
        loads.append(
            {
                "file": file_label,
                "symbol": hint,
                "span": [1, 80],
                "grep_patterns": patterns or grep_patterns_for_wiring(file_label),
                "reason": "caller integration/bootstrap region not loaded",
            }
        )
    return tuple(loads)


def grep_pending_caller_loads(
    manifest: StepManifest,
    grep_suggested_views: Sequence[Mapping[str, object]],
) -> tuple[dict[str, object], ...]:
    """Grep mount/setup hits on files not yet grounded in the manifest."""
    grounded_files = {
        _norm_file(item.file)
        for item in _grounded_observed_items(manifest)
        if item.file
    }
    loads: list[dict[str, object]] = []
    seen: set[str] = set()
    for view in grep_suggested_views:
        file_label = str(view.get("file") or "").strip()
        norm = _norm_file(file_label)
        if not norm or norm in grounded_files or norm in seen:
            continue
        if view.get("resolved_from") not in {"mount_context", "decorator_context"}:
            continue
        symbol = str(view.get("symbol") or default_wiring_probe_symbol(file_label))
        span = view.get("span")
        load: dict[str, object] = {
            "file": file_label,
            "symbol": symbol,
            "reason": "grep mount/setup hit — load caller wiring before decision_edit",
        }
        if isinstance(span, list) and len(span) >= 2:
            load["span"] = [int(span[0]), int(span[1])]
        loads.append(load)
        seen.add(norm)
    return tuple(loads)


def cross_file_partner_files(manifest: StepManifest, edited_file: str) -> frozenset[str]:
    """Files to additionally mark STALE after editing ``edited_file``.

    Always empty: blanket partner invalidation caused yaml/config edits to mark
    every observed ``.py`` anchor STALE and reopen VIEW while ``edit_ready: yes``.
    Invalidate scope is owned by ``src.hooks.tool_gate.files_to_invalidate``
    (edited file only). Cross-file refresh uses wiring_gap / verification mode.
    """
    _ = manifest, edited_file
    return frozenset()


def _cross_file_observed_ready(
    manifest: StepManifest,
    *,
    task_text: str = "",
) -> bool:
    grounded = _grounded_observed_items(manifest)
    symbol_files = {
        _norm_file(item.file)
        for item in grounded
        if item.type == "symbol" and item.file and _observed_line_count(item) >= 3
    }
    if len(symbol_files) < 2:
        return False
    return not wiring_gap_lines(manifest, task_text=task_text)


def retrieval_profile(
    manifest: StepManifest,
    *,
    task_text: str = "",
) -> RetrievalProfile:
    """Classify which retrieval modalities the manifest still requires."""
    missing = manifest.missing_items
    stale = stale_needing_refresh(manifest)
    core = manifest.core_items
    has_satisfied = any(item.status == "SATISFIED" for item in core)
    bootstrap = not core or not has_satisfied

    needs_view = bool(stale)
    needs_grep = False
    needs_heavy = False
    unlocated_missing = False

    for item in missing:
        if _item_has_concrete_target(item):
            needs_view = True
        else:
            unlocated_missing = True
            needs_grep = True

    if bootstrap and not missing and not needs_view and not _observed_ready_for_task(
        manifest, "edit", task_text=task_text
    ):
        needs_grep = True
    if wiring_gap_lines(manifest, task_text=task_text):
        needs_view = True
    if bootstrap and unlocated_missing:
        needs_heavy = True
    elif unlocated_missing and any(
        item.type in {"schema", "symbol", "mount"} for item in missing
    ):
        needs_heavy = True

    return RetrievalProfile(
        needs_grep=needs_grep,
        needs_view=needs_view,
        needs_heavy=needs_heavy,
        bootstrap=bootstrap,
    )


def manifest_metrics(manifest: StepManifest) -> ManifestMetrics:
    """Compute weighted readiness from required obligations only."""
    items = manifest.core_items
    total = sum(_ITEM_WEIGHTS.get(item.type, 1.0) for item in items)
    if total <= 0:
        return ManifestMetrics(0.0, 0.0, 0.0, 0, False)
    satisfied = sum(
        _ITEM_WEIGHTS.get(item.type, 1.0)
        for item in items if item.status == "SATISFIED"
    )
    missing = sum(
        _ITEM_WEIGHTS.get(item.type, 1.0)
        for item in items if item.status == "MISSING"
    )
    stale = sum(
        _ITEM_WEIGHTS.get(item.type, 1.0)
        for item in items if item.status == "STALE"
    )
    return ManifestMetrics(
        coverage=satisfied / total,
        missing_ratio=missing / total,
        stale_ratio=stale / total,
        required_count=len(items),
        critical_missing=any(
            item.status == "MISSING" and item.type in {"schema", "symbol", "mount"}
            for item in items
        ),
    )


# ---------------------------------------------------------------------------
# Template construction (need-level items keyed by task requirement slots)
# ---------------------------------------------------------------------------


def manifest_from_slots(
    slots: Sequence[str],
    *,
    step_id: str = "task.default",
    step_kind: StepKind = "retrieval",
) -> StepManifest:
    """Build an initial need-level manifest from requirement slots.

    Slot descriptions/types/keywords come from the shared evidence-slot registry.
    Targets (file/span/symbol) are intentionally left empty; they are backfilled
    from observations as retrieval discovers concrete evidence.
    """
    items: list[EvidenceItem] = []
    for slot in slots:
        definition = slot_def(slot)
        if definition is None:
            items.append(EvidenceItem(id=slot, need=slot, type="symbol"))
            continue
        items.append(
            EvidenceItem(
                id=slot,
                need=definition.need,
                type=definition.item_type,  # type: ignore[arg-type]
            )
        )
    return StepManifest(
        step_id=step_id,
        step_kind=step_kind,
        required_items=tuple(items),
        sufficiency=Sufficiency.INSUFFICIENT,
    )


# ---------------------------------------------------------------------------
# Coverage / satisfaction rules (computed from durable verbatim anchors)
# ---------------------------------------------------------------------------


def _anchor_code(anchor: Mapping[str, object]) -> str:
    for key in ("code", "verbatim_code", "observation_code"):
        value = anchor.get(key)
        if value:
            return str(value)
    return ""


def _anchor_file(anchor: Mapping[str, object]) -> str:
    return str(anchor.get("file") or "").replace("\\", "/").lstrip("./")


def _anchor_span(anchor: Mapping[str, object]) -> tuple[int, int] | None:
    span = anchor.get("span") or []
    if isinstance(span, (list, tuple)) and len(span) == 2:
        try:
            return int(span[0]), int(span[1])
        except (TypeError, ValueError):
            return None
    return None


def _anchor_symbol(anchor: Mapping[str, object]) -> str:
    return str(anchor.get("symbol") or "").split(".")[-1].casefold()


def _norm_file(file: str | None) -> str:
    return str(file or "").replace("\\", "/").lstrip("./")


def loaded_symbols_for_file(manifest: StepManifest, file_path: str) -> frozenset[str]:
    """Symbols with SATISFIED/STALE anchors on the given file."""
    norm = _norm_file(file_path)
    return frozenset(
        str(item.symbol)
        for item in manifest.required_items
        if item.symbol
        and item.status in {"SATISFIED", "STALE"}
        and _norm_file(item.file) == norm
    )


def missing_focus_symbols(
    manifest: StepManifest,
    file_path: str,
    focus_symbols: Sequence[str],
) -> tuple[str, ...]:
    """Return focus symbol names not yet loaded for file_path."""
    loaded = loaded_symbols_for_file(manifest, file_path)
    return tuple(
        symbol
        for symbol in focus_symbols
        if symbol.strip() and symbol not in loaded
    )


def _span_covers(outer: tuple[int, int], inner: tuple[int, int]) -> bool:
    return outer[0] <= inner[0] and inner[1] <= outer[1]


def _verbatim_anchors(
    anchors: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    return [anchor for anchor in anchors if _anchor_code(anchor).strip()]


def _item_covered(
    item: EvidenceItem,
    anchors: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return the covering verbatim anchor for an item, or None.

    Priority: explicit span coverage > symbol coverage > schema/keyword match.
    """
    verbatim = _verbatim_anchors(anchors)
    item_file = _norm_file(item.file)

    # 1. Span coverage (strongest): same file + anchor span covers item span.
    if item.span is not None and item_file:
        for anchor in verbatim:
            span = _anchor_span(anchor)
            if (
                _anchor_file(anchor) == item_file
                and span is not None
                and _span_covers(span, item.span)
            ):
                return anchor

    # 2. Symbol coverage: full verbatim slice of the target symbol.
    if item.symbol:
        target = item.symbol.split(".")[-1].casefold()
        for anchor in verbatim:
            if _anchor_symbol(anchor) != target:
                continue
            if item_file and _anchor_file(anchor) != item_file:
                continue
            return anchor

    # Structural predicate for registry slots (schema + need-level items). When
    # present it supersedes flat keyword matching so single-line assignments /
    # import lines can't false-ground a slot merely by containing a substring.
    rule = slot_satisfy_rule(item.id)

    # 3. Schema evidence: verbatim DDL block.
    if item.type == "schema":
        for anchor in verbatim:
            code = _anchor_code(anchor)
            if rule is not None:
                if rule_matches(rule, code, str(anchor.get("symbol") or "")):
                    return anchor
                continue
            if re.search(
                r"create\s+(?:or\s+replace\s+)?(?:temp\w*\s+)?"
                r"(?:table|view|trigger|procedure)",
                code.lower(),
            ) and _keywords_match(item, anchor):
                return anchor
        return None

    # 4. Need-level match against durable verbatim anchors.
    if item.span is None and not item.symbol:
        if rule is not None:
            for anchor in verbatim:
                if rule_matches(
                    rule, _anchor_code(anchor), str(anchor.get("symbol") or "")
                ):
                    return anchor
            return None
        if not item.keywords:
            for anchor in verbatim:
                code = _anchor_code(anchor)
                if re.search(r"\b(?:async\s+def|def|class)\b", code):
                    return anchor
            return None
        for anchor in verbatim:
            if _keywords_match(item, anchor):
                return anchor

    return None


def _keywords_match(item: EvidenceItem, anchor: Mapping[str, object]) -> bool:
    if not item.keywords:
        return True
    haystack = " ".join(
        (
            _anchor_code(anchor),
            str(anchor.get("symbol") or ""),
            _anchor_file(anchor),
        )
    ).casefold()
    return any(keyword.casefold() in haystack for keyword in item.keywords)


def _anchor_hash(anchor: Mapping[str, object]) -> str | None:
    value = anchor.get("content_hash") or anchor.get("hash")
    return str(value) if value else None


# ---------------------------------------------------------------------------
# Projection + sufficiency
# ---------------------------------------------------------------------------


def project_manifest(
    manifest: StepManifest,
    anchors: Sequence[Mapping[str, object]],
    *,
    step: int,
    task_mode: str = "diagnose",
) -> StepManifest:
    """Recompute item status and sufficiency from durable code anchors.

    Rules:
      * A covering verbatim anchor -> SATISFIED (records provenance + hash).
      * No coverage but item was flagged STALE by the reducer -> stays STALE
        (needs a fresh read; retrieval is allowed once).
      * Otherwise -> MISSING.

    Failure items (test_failure / tool_error) are passed through unchanged; they
    are cleared by the reducer when the failure is resolved.
    """
    updated: list[EvidenceItem] = []
    for item in manifest.required_items:
        if item.is_failure:
            updated.append(item)
            continue
        if item.status == "STALE":
            updated.append(item)
            continue
        covering = _item_covered(item, anchors)
        if covering is not None:
            updated.append(
                replace(
                    item,
                    status="SATISFIED",
                    file=item.file or (_anchor_file(covering) or None),
                    span=item.span or _anchor_span(covering),
                    symbol=item.symbol or (str(covering.get("symbol")) or None),
                    content_hash=_anchor_hash(covering),
                    stale_reason=None,
                )
            )
        else:
            updated.append(replace(item, status="MISSING"))

    projected = replace(
        manifest,
        required_items=tuple(updated),
        updated_at_step=step,
    )
    return replace(
        projected,
        sufficiency=compute_sufficiency(projected, task_mode),
    )


def _observed_ready_for_task(
    manifest: StepManifest,
    task_mode: str,
    *,
    task_text: str = "",
) -> bool:
    """Bootstrap edit/answer gate from grounded observed anchors (no abstract slots).

    STALE observed items still count — they were loaded before an edit invalidated
    the on-disk copy; refresh is optional and must not drop edit readiness.

    Edit bootstrap paths (any one is enough):
      1. schema DDL (3+ lines) + any symbol anchor — table/view change tasks.
      2. large symbol anchor (50+ lines) — single-file route/handler edits.
      3. cross-file symbols with wiring — integration/wiring (caller setup/mount
         region loaded on support files, not only mid-file handler slices).
    """
    grounded = list(_grounded_observed_items(manifest))
    if not grounded:
        return False
    if task_mode == "diagnose":
        return True

    def _line_count(item: EvidenceItem) -> int:
        return _observed_line_count(item)

    has_schema = any(
        item.type == "schema" and _line_count(item) >= 3 for item in grounded
    )
    has_symbol = any(item.type == "symbol" for item in grounded)
    if has_schema and has_symbol:
        return True

    if any(item.type == "symbol" and _line_count(item) >= 50 for item in grounded):
        return True

    if _cross_file_observed_ready(manifest, task_text=task_text):
        return True

    return False


def compute_sufficiency(manifest: StepManifest, task_mode: str = "diagnose") -> Sufficiency:
    """Derive sufficiency from item status.

    * Any MISSING core (required) item -> INSUFFICIENT.
    * All core items SATISFIED/STALE -> edit/answer ready.
    * Empty core (bootstrap): require grounded observed schema+symbol for edit.
    STALE items do not block sufficiency; they reopen retrieval for one refresh.
    """
    if manifest.core_items:
        if manifest.has_missing:
            return Sufficiency.INSUFFICIENT
        if task_mode == "edit":
            return Sufficiency.SUFFICIENT_FOR_EDIT
        return Sufficiency.SUFFICIENT_FOR_ANSWER

    if _observed_ready_for_task(manifest, task_mode):
        if task_mode == "edit":
            return Sufficiency.SUFFICIENT_FOR_EDIT
        return Sufficiency.SUFFICIENT_FOR_ANSWER
    return Sufficiency.INSUFFICIENT


def backfill_targets(
    manifest: StepManifest,
    observations: Sequence[Mapping[str, object]],
) -> StepManifest:
    """Attach concrete file/span/symbol targets from observations to items.

    Never sets SATISFIED (that is the projection's job); only records the most
    specific target so later projection can verify span/symbol coverage.
    """
    if not observations:
        return manifest
    updated: list[EvidenceItem] = []
    for item in manifest.required_items:
        if item.is_failure or item.span is not None:
            updated.append(item)
            continue
        match = _best_observation(item, observations)
        if match is None:
            updated.append(item)
            continue
        span = _anchor_span(match)
        updated.append(
            replace(
                item,
                file=item.file or (_anchor_file(match) or None),
                span=item.span or span,
                symbol=item.symbol or (str(match.get("symbol")) or None),
            )
        )
    return replace(manifest, required_items=tuple(updated))


MAX_OBSERVED_ITEMS = 32


def supersede_stale_with_observations(
    manifest: StepManifest,
    observations: Sequence[Mapping[str, object]],
) -> StepManifest:
    """Promote STALE manifest rows superseded by fresh post-edit/load observations."""
    if not observations:
        return manifest
    fresh_by_symbol: dict[tuple[str, str], Mapping[str, object]] = {}
    for observation in observations:
        file = _anchor_file(observation)
        symbol = str(observation.get("symbol") or "").strip()
        if not file or not symbol or not _anchor_code(observation).strip():
            continue
        fresh_by_symbol[(_norm_file(file), symbol)] = observation
    if not fresh_by_symbol:
        return manifest

    updated: list[EvidenceItem] = []
    for item in manifest.required_items:
        if item.status != "STALE" or not item.file or not item.symbol:
            updated.append(item)
            continue
        observation = fresh_by_symbol.get((_norm_file(item.file), item.symbol))
        if observation is None:
            updated.append(item)
            continue
        span = _anchor_span(observation)
        updated.append(
            replace(
                item,
                status="SATISFIED",
                file=_norm_file(_anchor_file(observation) or item.file),
                span=span or item.span,
                content_hash=_anchor_hash(observation),
                stale_reason=None,
            )
        )
    return replace(manifest, required_items=tuple(updated))


def reconcile_observations(
    manifest: StepManifest,
    observations: Sequence[Mapping[str, object]],
) -> StepManifest:
    """Reconcile successful retrieval observations into the manifest.

    Grep locators never become manifest items (hints only). Full symbol/DDL
    loads become ``observed`` memory — they do not inflate required coverage.
    Need-level ``required`` slots (if any) only receive backfilled targets.
    """
    reconciled = supersede_stale_with_observations(manifest, observations)
    reconciled = backfill_targets(reconciled, observations)
    existing_ids = {item.id for item in reconciled.required_items}
    observed_count = sum(1 for item in reconciled.required_items if item.role == "observed")
    additions: list[EvidenceItem] = []

    for observation in observations:
        code = _anchor_code(observation).strip()
        file = _anchor_file(observation)
        span = _anchor_span(observation)
        symbol = str(observation.get("symbol") or "").strip()
        locator_only = bool(observation.get("locator_only") or observation.get("match_line"))

        if locator_only or not code or not file or span is None:
            continue

        is_schema = bool(
            re.search(
                r"(?is)\bcreate\s+(?:or\s+replace\s+)?(?:temp\w*\s+)?"
                r"(?:table|view|trigger|procedure)\b",
                code,
            )
        )
        if not (symbol or is_schema):
            continue

        target_kind = "schema" if is_schema else "symbol"
        target_name = symbol or f"ddl@{span[0]}-{span[1]}"
        already_tracked = any(
            _norm_file(item.file or "") == file
            and (
                (item.symbol and item.symbol == target_name)
                or (
                    item.span is not None
                    and span is not None
                    and (
                        _span_covers(span, item.span)
                        or _span_covers(item.span, span)
                    )
                )
            )
            for item in (*reconciled.required_items, *additions)
        )
        if already_tracked:
            continue
        if observed_count + len(additions) >= MAX_OBSERVED_ITEMS:
            continue

        observed_id = f"observed.{target_kind}:{file}:{target_name}"
        if observed_id in existing_ids:
            continue
        existing_ids.add(observed_id)
        keywords: tuple[str, ...] = ()
        if target_kind == "symbol" and mount_setup_confirmed(code, file_path=file):
            keywords = ("mount_confirmed",)
        additions.append(
            EvidenceItem(
                id=observed_id,
                need=f"已加载观察：{file}::{target_name}",
                type=target_kind,  # type: ignore[arg-type]
                role="observed",
                file=file,
                span=span,
                symbol=symbol or None,
                status="MISSING",
                provenance="retrieval_request",
                keywords=keywords,
            )
        )

    if not additions:
        return reconciled
    return replace(
        reconciled,
        required_items=(*reconciled.required_items, *tuple(additions)),
    )


_DDL_HEAD = re.compile(
    r"(?is)^\s*CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\w*\s+)?"
    r"(?:TABLE|VIEW|TRIGGER|PROCEDURE)\s+(?:IF\s+NOT\s+EXISTS\s+)?"
    r"(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))"
)


def observations_from_edited_file(file_path: str, content: str) -> list[dict[str, object]]:
    """Build observation payloads from a post-edit file for manifest reconciliation."""
    norm_file = _norm_file(file_path)
    if not norm_file.endswith(".sql"):
        return []

    lines = content.splitlines()
    observations: list[dict[str, object]] = []
    index = 0
    while index < len(lines):
        match = _DDL_HEAD.match(lines[index])
        if not match:
            index += 1
            continue

        symbol = next(group for group in match.groups() if group)
        start_line = index + 1
        block_lines = [lines[index]]
        paren_depth = lines[index].count("(") - lines[index].count(")")
        end_index = index
        while end_index + 1 < len(lines):
            end_index += 1
            block_lines.append(lines[end_index])
            paren_depth += lines[end_index].count("(") - lines[end_index].count(")")
            if ";" in lines[end_index] and paren_depth <= 0:
                break

        observations.append(
            {
                "file": norm_file,
                "span": [start_line, end_index + 1],
                "code": "\n".join(block_lines),
                "symbol": symbol,
            }
        )
        index = end_index + 1

    return observations


def _best_observation(
    item: EvidenceItem,
    observations: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    for observation in observations:
        if observation.get("locator_only") or observation.get("match_line"):
            continue
        if _keywords_match(item, observation) and _anchor_span(observation) is not None:
            return observation
    return None


def _structural_locator(text: str) -> tuple[str | None, str | None]:
    ddl = re.search(
        r"(?i)\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\w*\s+)?"
        r"(?:TABLE|VIEW|TRIGGER|PROCEDURE)\s+(?:IF\s+NOT\s+EXISTS\s+)?`?([\w$]+)`?",
        text,
    )
    if ddl:
        return "schema", ddl.group(1)
    symbol = extract_symbol_with_profiles(text, ALL_PROFILES)
    if symbol:
        return "symbol", symbol
    return None, None


def _item_locator(item: EvidenceItem) -> str:
    file = item.file or "?"
    if item.span is not None:
        return f"{file}:{item.span[0]}-{item.span[1]}"
    if item.symbol:
        return f"{file}::{item.symbol}"
    return file


def _discovery_hints(
    task_slots: Sequence[str],
    *,
    task_text: str = "",
    project_root: Path | None = None,
) -> list[str]:
    if task_text.strip():
        return discovery_hint_lines(task_text, task_slots, project_root=project_root)
    hints: list[str] = []
    for slot_id in task_slots[:5]:
        slot_meta = slot_def(slot_id)
        if slot_meta is None or not slot_meta.grep_patterns:
            continue
        include = slot_meta.grep_include or "*.*"
        pattern_hint = "|".join(slot_meta.grep_patterns[:5])
        hints.append(
            f"- {slot_id}: grep_search(patterns=[{pattern_hint!r}, ...], include={include!r})"
        )
    return hints


def _norm_manifest_file(file: str | None) -> str:
    return str(file or "").replace("\\", "/").lstrip("./")


def primary_edit_target_files(
    manifest: StepManifest,
    task_text: str = "",
    *,
    limit: int = 3,
) -> tuple[str, ...]:
    """Files the Core LLM should treat as in-scope for decision_edit when retrieval is closed."""
    ranked: list[str] = []
    seen: set[str] = set()

    def add(file: str | None) -> None:
        norm = _norm_manifest_file(file)
        if not norm or norm in seen:
            return
        seen.add(norm)
        ranked.append(norm)

    likely = _likely_edit_target_file(
        _grounded_observed_items(manifest),
        task_text=task_text,
        manifest=manifest,
    )
    if likely:
        add(likely)

    for item in manifest.required_items:
        if item.status not in {"SATISFIED", "STALE"} or not item.file:
            continue
        if item.role == "required" or item.id == "target_implementation":
            add(item.file)

    for item in manifest.required_items:
        if item.status not in {"SATISFIED", "STALE"} or not item.file:
            continue
        if item.role == "observed":
            add(item.file)

    lowered = task_text.casefold()
    for item in manifest.required_items:
        if not item.file:
            continue
        stem = _norm_manifest_file(item.file).split("/")[-1].casefold()
        if stem and stem in lowered:
            add(item.file)

    return tuple(ranked[:limit])


def execution_card(
    manifest: StepManifest,
    tools_available: Sequence[str],
    *,
    retrieval_no_gain_rounds: int = 0,
    task_slots: Sequence[str] = (),
    task_text: str = "",
    last_grep_error: str = "",
    last_view_error: str = "",
    grep_suggested_views: Sequence[Mapping[str, object]] = (),
    edit_burst: bool = False,
    edit_patch_failed: bool = False,
    view_last_round_all_duplicate: bool = False,
    verification_mode: bool = False,
    edited_files: Sequence[str] = (),
    preferred_tool: str | None = None,
    tool_probabilities: Mapping[str, float] | None = None,
    project_root: Path | None = None,
    edit_plan_view: Any = None,
) -> str:
    """Render the STEP EVIDENCE card for the Core LLM."""
    edit_ready = manifest.sufficiency in {
        Sufficiency.SUFFICIENT_FOR_EDIT,
        Sufficiency.SUFFICIENT_FOR_VERIFY,
    }
    metrics = manifest_metrics(manifest)
    lines = [
        "### STEP EVIDENCE (current step)",
        f"step: {manifest.step_id}",
        f"edit_ready: {'yes' if edit_ready else 'no'}",
        f"required_coverage: {metrics.coverage:.2f}",
        f"retrieval_no_gain_rounds: {retrieval_no_gain_rounds}",
    ]
    if tool_probabilities:
        ranked = sorted(
            tool_probabilities.items(), key=lambda kv: kv[1], reverse=True
        )
        ranking = " > ".join(f"{tool}({prob:.2f})" for tool, prob in ranked)
        lines.append(f"tool_ranking: {ranking}")
    if preferred_tool:
        lines.append(
            f"preferred_tool: {preferred_tool} — start here unless the guidance "
            "below justifies another allowed tool"
        )
    if not manifest.core_items:
        satisfied_observed = [
            item
            for item in manifest.observed_items
            if item.status in {"SATISFIED", "STALE"}
        ]
        if not satisfied_observed:
            lines.append("bootstrap: no verified target loaded yet")
        elif not edit_ready:
            if wiring_gap_lines(manifest, task_text=task_text):
                lines.append(
                    "bootstrap_gate: wiring_gap — support file missing entry/integration "
                    "region; load language-specific bootstrap wiring before decision_edit"
                )
            else:
                lines.append(
                    "bootstrap_gate: observed anchors present but edit_ready=no — "
                    "need schema+symbol, or a 50+ line symbol, or wired cross-file symbols"
                )
        if satisfied_observed:
            lines.append(
                f"grounded_observations: {len(satisfied_observed)} "
                f"(schema={'yes' if any(i.type == 'schema' for i in satisfied_observed) else 'no'}, "
                f"symbol={'yes' if any(i.type == 'symbol' for i in satisfied_observed) else 'no'})"
            )
    wiring_gaps = wiring_gap_lines(manifest, task_text=task_text)
    for gap in wiring_gaps:
        lines.append(f"wiring_gap: {gap}")
    pending_wiring = wiring_gap_pending_loads(manifest, task_text=task_text)
    for load in pending_wiring:
        file_label = str(load.get("file") or "?")
        symbol = str(load.get("symbol") or default_wiring_probe_symbol(file_label))
        patterns = load.get("grep_patterns")
        if isinstance(patterns, tuple):
            grep_hint = "|".join(str(p) for p in patterns)
        else:
            grep_hint = "|".join(grep_patterns_for_wiring(file_label)[:4]) or "main"
        lines.append(
            f"pending_wiring: view_symbol_code({file_label!r}, symbol={symbol!r}) "
            f"or grep {grep_hint!r} on {file_label!r} — {load.get('reason', 'mount/setup')}"
        )
    grep_pending = grep_pending_caller_loads(manifest, grep_suggested_views)
    for load in grep_pending:
        file_label = str(load.get("file") or "?")
        symbol = str(load.get("symbol") or default_wiring_probe_symbol(file_label))
        lines.append(
            f"pending_wiring: view_symbol_code({file_label!r}, symbol={symbol!r}) "
            f"— {load.get('reason', 'grep mount/setup not loaded')}"
        )
    if edit_ready and wiring_gaps:
        targets = primary_edit_target_files(manifest, task_text)
        if targets:
            lines.append(
                f"edit_target: {targets[0]} — decision_edit applies here; "
                "wiring_gap lines above are advisory (use LOADED CODE ANCHORS); "
                "do not decision_edit caller files for inspection"
            )
    if edit_ready and project_root is not None:
        from src.agent.edit_brief import available_symbols_lines

        targets = primary_edit_target_files(manifest, task_text)
        if not targets:
            # Fall back to files that already have loaded symbols.
            targets = sorted(
                {
                    str(item.file)
                    for item in manifest.required_items
                    if item.file
                    and item.symbol
                    and item.status in {"SATISFIED", "STALE"}
                }
            )[:3]
        lines.extend(available_symbols_lines(project_root, targets))
    if edit_burst:
        lines.append(
            "edit_burst: the frozen edit_plan is draining — do not re-fetch edited "
            "files; verify only after the plan drains."
        )
    elif edit_patch_failed:
        lines.append(
            "edit_recovery: last decision_edit patch failed — verify focus_symbols and "
            "context_window still match disk, then retry the same plan step"
        )
    elif view_last_round_all_duplicate and edit_ready and not wiring_gaps:
        lines.append(
            "retrieval_converged: last read was duplicate (already in LOADED CODE ANCHORS); "
            "use decision_edit only — do not re-fetch loaded symbols"
        )
    elif view_last_round_all_duplicate and edit_ready and wiring_gaps:
        lines.append(
            "wiring_duplicate: last view replayed a loaded symbol on the wrong file — "
            "use pending_wiring above (different file/symbol), not decision_edit to read"
        )
    if edit_plan_view is not None:
        from src.agent.edit_plan import format_edit_plan_evidence_lines

        lines.extend(format_edit_plan_evidence_lines(edit_plan_view))
    # Event-gated mechanical rules (plan contract / drain / ErrorClass) — not in system.
    from src.agent.edit_plan import mechanical_rule_lines_for_turn

    lines.extend(
        mechanical_rule_lines_for_turn(
            edit_ready=edit_ready,
            edit_plan_view=edit_plan_view,
            edit_burst=edit_burst,
        )
    )
    tools_set = frozenset(tools_available)
    retrieval_closed = edit_ready and tools_set == frozenset({"decision_edit"})
    if retrieval_closed:
        targets = primary_edit_target_files(manifest, task_text)
        target_hint = targets[0] if targets else "the task target file"
        lines.append(
            f"edit_target: {target_hint} — decision_edit applies here only; "
            "caller files (e.g. main.py) are reference context, not edit targets"
        )
        lines.append(
            "retrieval_closed: use LOADED CODE ANCHORS for wiring context; "
            f"decision_edit({target_hint}) for patches — do not decision_edit other files to read"
        )
        same_file_stale = [
            item
            for item in stale_needing_refresh(manifest)
            if item.file and _norm_file(item.file) == _norm_file(target_hint)
        ]
        if same_file_stale:
            lines.append(
                "context_window: use stale anchors below for spans on the edit target; "
                "end line may exceed current file length — harness clips to EOF"
            )
    elif edit_ready and "decision_edit" in tools_available:
        if "grep_search" not in tools_available and "view_symbol_code" not in tools_available:
            lines.append(
                "edit_ready: prefer decision_edit; retrieval is closed — "
                "one call per active plan step (intent + focus_symbols + context_window); "
                "split heavy work across multiple Core decision_edit calls"
            )
        else:
            lines.append(
                "edit_ready: prefer decision_edit; retrieval remains open only for "
                "missing/stale refresh or verification — wiring_gap is advisory"
            )
    if verification_mode:
        lines.append(
            "verification: plan edits complete — use view_symbol_code (or grep_search) "
            "to read disk; do not decision_edit for inspection"
        )
    unique_loaded: list[EvidenceItem] = []
    seen_loaded: set[tuple[str | None, tuple[int, int] | None, str | None]] = set()
    load_candidates = [
        item for item in manifest.required_items if item.status == "SATISFIED"
    ]
    load_candidates.sort(
        key=lambda item: (
            0 if item.role == "required" else 1,
            -(item.span[1] - item.span[0]) if item.span else 0,
        )
    )
    for item in load_candidates:
        identity = (item.file, item.span, item.symbol)
        if identity in seen_loaded:
            continue
        seen_loaded.add(identity)
        unique_loaded.append(item)
    # Cap must exceed the typical loaded-symbol count: an under-cap hides slices
    # the agent already holds, so the Core LLM re-views/greps them (duplicate
    # anchors) and burns rounds. Keep the full inventory visible.
    loaded = unique_loaded[:_LOADED_INVENTORY_CAP]
    if loaded:
        lines.append(
            "loaded (already in LOADED CODE ANCHORS — reuse; do NOT "
            "view_symbol_code/grep_search these again):"
        )
        for item in loaded:
            symbol = f"  symbol={item.symbol}" if item.symbol else ""
            lines.append(f"- {_item_locator(item)}{symbol}")
        remaining = len(unique_loaded) - len(loaded)
        if remaining > 0:
            lines.append(f"- … (+{remaining} more already loaded — also do not re-fetch)")
    missing = manifest.missing_items[:5]
    if missing:
        lines.append("missing:")
        for item in missing:
            lines.append(f"- {_item_locator(item)}  ({item.need})")
    hints = _discovery_hints(task_slots, task_text=task_text, project_root=project_root)
    if hints and "grep_search" in tools_available and not edit_ready:
        lines.append("discovery_hints (task-derived grep batch; not manifest obligations):")
        lines.extend(hints)
    if last_grep_error.strip() and "grep_search" in tools_available and not edit_ready:
        lines.append(f"last_grep_error: {last_grep_error.strip()[:240]}")
        if hints:
            lines.append(
                "next_grep: copy patterns from discovery_hints above; "
                "never retry include/mode-only calls."
            )
    if last_view_error.strip() and "view_symbol_code" in tools_available:
        lines.append(f"last_view_error: {last_view_error.strip()[:280]}")
        lines.append(
            "next_view: use symbols from last_view_error or pending_wiring — "
            "do not repeat the same failed file+symbol pair."
        )
    loaded_symbol_by_name: dict[str, set[str]] = {}
    for item in loaded:
        if not item.symbol or not item.file:
            continue
        loaded_symbol_by_name.setdefault(str(item.symbol), set()).add(_norm_file(item.file))

    def _suggested_view_actionable(view: Mapping[str, object]) -> bool:
        symbol = str(view.get("symbol") or "")
        file_label = str(view.get("file") or "")
        if not symbol or not file_label:
            return True
        loaded_files = loaded_symbol_by_name.get(symbol)
        if not loaded_files:
            return True
        norm_view = _norm_file(file_label)
        if norm_view in loaded_files:
            return True
        # Same symbol name on a different file is usually a grep mount-line false positive.
        return False

    filtered_suggested_views = tuple(
        item
        for item in grep_suggested_views
        if _suggested_view_actionable(item)
    )
    if filtered_suggested_views and (
        "view_symbol_code" in tools_available or "grep_search" in tools_available
    ):
        lines.append("suggested_views (from last successful grep; use view_symbol_code next):")
        for item in list(filtered_suggested_views)[:6]:
            file = str(item.get("file") or "?")
            symbol = str(item.get("symbol") or "?")
            span = item.get("span")
            match_line = str(item.get("match_line") or "").strip()
            if isinstance(span, list) and len(span) >= 2:
                line = (
                    f"- view_symbol_code(target_file={file!r}, symbol={symbol!r})  "
                    f"# grep hit ~L{span[0]}"
                )
            else:
                line = f"- view_symbol_code(target_file={file!r}, symbol={symbol!r})"
            if match_line:
                line += f"  ({match_line[:80]})"
            lines.append(line)
    stale = list(stale_needing_refresh(manifest))[:3]
    if stale:
        if edit_burst:
            lines.append(
                "stale anchors (use spans in decision_edit context_window; "
                "end clipped to EOF if file shrank/grew):"
            )
        else:
            lines.append("stale (needs refresh):")
        for item in stale:
            symbol = f"  symbol={item.symbol}" if item.symbol else ""
            lines.append(f"- {_item_locator(item)}{symbol}  ({item.need})")
    failures = manifest.failure_items[:3]
    if failures:
        lines.append("fix:")
        for item in failures:
            lines.append(f"- {item.need}")
    tools = ", ".join(dict.fromkeys(tools_available)) or "(none)"
    lines.append(f"tools_available: {tools}")
    return "\n".join(lines)
