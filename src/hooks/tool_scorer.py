"""Probabilistic tool scoring layer for the assembled loop.

This module turns the manifest/runtime signals that ``reallocate_tools`` already
computes into a per-tool score, then a probability distribution via a
temperature-controlled softmax. It is deliberately dependency-light (stdlib +
dataclasses only) so it stays a pure math layer with no import cycle against
``reallocate_tools`` or ``manifest``.

Formula
-------
For every tool ``t``::

    score(t) = Σ_l  w_l · f_l(t | state)

    P(t) = exp(score(t) / τ) / Σ_{t' ∈ allowed} exp(score(t') / τ)

* ``w_l`` are per-layer weights (``LAYER_WEIGHTS``).
* ``f_l`` are the per-layer logit contributions in roughly ``[-1, 1]``.
* ``τ`` (temperature) controls sharpness: ``τ → 0`` approaches argmax, larger
  ``τ`` flattens toward uniform.

Hard gates are **not** applied here. The caller passes the already-gated
``allowed`` set (the ``determine_allowed_tools`` waterfall encodes the safety
constraints); the softmax is computed over that set only. The scores/probabilities
then rank the allowed tools so the Core LLM is steered toward the right one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from src.hooks.tool_gate import scorer_should_boost_view_for_stale

GREP = "grep_search"
VIEW = "view_symbol_code"
EDIT = "decision_edit"
RETRIEVE = "codebase_retrieve"

TOOLS: tuple[str, ...] = (GREP, VIEW, EDIT, RETRIEVE)

LAYER_WEIGHTS: dict[str, float] = {
    "phase": 1.0,
    "manifest": 1.2,
    "runtime": 1.5,
    "affordance": 1.0,
}

DEFAULT_TEMPERATURE = 0.6


@dataclass(frozen=True, slots=True)
class ToolScoringContext:
    """Flattened, tool-agnostic signals used to score tools for one turn."""

    task_mode: str = "diagnose"
    is_responding: bool = False
    sufficiency: str = "INSUFFICIENT"
    coverage: float = 0.0
    missing_ratio: float = 0.0
    stale_ratio: float = 0.0
    critical_missing: bool = False
    has_missing: bool = False
    has_stale: bool = False
    wiring_gap: bool = False
    grep_pending: bool = False
    validation_failed: bool = False
    edit_burst: bool = False
    verification_mode: bool = False
    no_gain_rounds: int = 0
    view_last_round_all_duplicate: bool = False
    edit_patch_failed: bool = False
    has_failures: bool = False
    actionable_suggested_views: bool = False
    bootstrap: bool = False


@dataclass(frozen=True, slots=True)
class ToolAllocation:
    """Result of allocation: gated ``allowed`` set plus ranked probabilities."""

    allowed: frozenset[str] = frozenset()
    scores: dict[str, float] = field(default_factory=dict)
    probabilities: dict[str, float] = field(default_factory=dict)
    preferred: str | None = None
    temperature: float = DEFAULT_TEMPERATURE


def _zero_logits() -> dict[str, float]:
    return {tool: 0.0 for tool in TOOLS}


def _phase_layer(ctx: ToolScoringContext) -> dict[str, float]:
    logits = _zero_logits()
    if ctx.is_responding:
        logits[VIEW] += 0.4
        logits[GREP] += 0.3
        logits[EDIT] -= 0.3
        logits[RETRIEVE] -= 0.3
    if ctx.task_mode == "diagnose":
        logits[EDIT] -= 1.0
        logits[RETRIEVE] += 0.2
    return logits


def _manifest_layer(ctx: ToolScoringContext) -> dict[str, float]:
    logits = _zero_logits()
    if ctx.missing_ratio > 0.0:
        logits[GREP] += 0.9 * ctx.missing_ratio
        logits[RETRIEVE] += (0.5 if ctx.bootstrap else 0.2) * ctx.missing_ratio
        logits[EDIT] -= 0.8
    if ctx.stale_ratio > 0.0 and scorer_should_boost_view_for_stale(ctx):
        logits[VIEW] += 0.9 * ctx.stale_ratio
        logits[GREP] += 0.1 * ctx.stale_ratio
    if ctx.wiring_gap:
        logits[VIEW] += 0.95
        logits[GREP] += 0.3
    if (
        ctx.coverage >= 0.95
        and not ctx.wiring_gap
        and not ctx.has_missing
    ):
        logits[EDIT] += 0.9
        logits[GREP] -= 0.5
        logits[VIEW] -= 0.3
        logits[RETRIEVE] -= 0.6
    if ctx.critical_missing:
        logits[EDIT] -= 1.0
        logits[GREP] += 0.3
    return logits


def _runtime_layer(ctx: ToolScoringContext) -> dict[str, float]:
    logits = _zero_logits()
    if ctx.verification_mode:
        # Verification reads back what was just edited: view is primary, grep is
        # secondary discovery, decision_edit must not be abused for inspection.
        logits[VIEW] += 0.95
        logits[GREP] += 0.4
        logits[EDIT] += 0.1
        logits[RETRIEVE] -= 0.5
    if ctx.edit_burst:
        logits[EDIT] += 0.95
        logits[GREP] -= 0.7
        logits[VIEW] -= 0.8
        logits[RETRIEVE] -= 0.9
    if ctx.edit_patch_failed:
        logits[EDIT] += 0.7
        logits[GREP] += 0.5
        logits[VIEW] += 0.5
    if ctx.has_failures or ctx.validation_failed:
        logits[EDIT] += 0.6
        logits[GREP] += 0.3
        logits[VIEW] += 0.3
    if ctx.no_gain_rounds >= 2:
        logits[GREP] -= 0.6
        logits[VIEW] -= 0.8
        logits[EDIT] += 0.4
        logits[RETRIEVE] -= 0.7
    elif ctx.no_gain_rounds >= 1:
        logits[VIEW] -= 0.3
        logits[GREP] += 0.1
    if ctx.view_last_round_all_duplicate:
        logits[VIEW] -= 0.9
        logits[EDIT] += 0.6
        logits[GREP] += 0.2
    return logits


def _affordance_layer(ctx: ToolScoringContext) -> dict[str, float]:
    logits = _zero_logits()
    if ctx.actionable_suggested_views:
        logits[VIEW] += 0.8
        logits[GREP] -= 0.3
    if ctx.grep_pending:
        logits[VIEW] += 0.9
        logits[GREP] += 0.2
    return logits


_LAYERS = {
    "phase": _phase_layer,
    "manifest": _manifest_layer,
    "runtime": _runtime_layer,
    "affordance": _affordance_layer,
}


def compute_scores(ctx: ToolScoringContext) -> dict[str, float]:
    """Weighted sum of every layer's per-tool logit contribution."""
    totals = _zero_logits()
    for name, layer in _LAYERS.items():
        weight = LAYER_WEIGHTS[name]
        for tool, value in layer(ctx).items():
            totals[tool] += weight * value
    return totals


def softmax(scores: Mapping[str, float], temperature: float) -> dict[str, float]:
    """Numerically stable softmax over ``scores`` with the given temperature."""
    if not scores:
        return {}
    tau = temperature if temperature and temperature > 1e-6 else 1e-6
    peak = max(scores.values())
    exps = {tool: math.exp((value - peak) / tau) for tool, value in scores.items()}
    total = sum(exps.values())
    if total <= 0.0:
        uniform = 1.0 / len(scores)
        return {tool: uniform for tool in scores}
    return {tool: value / total for tool, value in exps.items()}


def allocate(
    ctx: ToolScoringContext,
    allowed: frozenset[str],
    *,
    temperature: float = DEFAULT_TEMPERATURE,
) -> ToolAllocation:
    """Score all tools, then softmax the probabilities over the gated ``allowed`` set."""
    allowed = frozenset(allowed)
    scores = compute_scores(ctx)
    masked = {tool: scores.get(tool, 0.0) for tool in allowed}
    probabilities = softmax(masked, temperature)
    preferred: str | None = None
    if probabilities:
        preferred = max(
            probabilities,
            key=lambda tool: (probabilities[tool], scores.get(tool, 0.0)),
        )
    return ToolAllocation(
        allowed=allowed,
        scores=scores,
        probabilities=probabilities,
        preferred=preferred,
        temperature=temperature,
    )
