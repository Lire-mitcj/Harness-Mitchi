from __future__ import annotations

import math
import logging
from collections import defaultdict
from typing import Any

log = logging.getLogger(__name__)


def calculate_retrieval_gravity(coverage_score: float, step_count: int) -> float:
    """With code coverage increase and step count increase, retrieval gravity exponentially decays."""
    k = 1.5
    base_gravity = math.exp(-k * coverage_score)

    # Step count penalty: prevents getting stuck in the same place
    time_penalty = 1.0 / (1.0 + math.log(step_count + 1))

    # Retain at least 0.1 gravity even at full saturation to allow LLM intuitive searches
    return max(0.1, base_gravity * time_penalty)


def evaluate_search_intent(
    new_pattern: str,
    target_file: str,
    execution_history: list[dict[str, Any]],
    has_compile_error: bool,
) -> dict[str, Any]:
    """Evaluates search intent to distinguish loop/repetition from reasonable review."""
    # 1. Calculate Novelty: check if keyword contains tokens not seen in previous searches of target_file
    words = set(new_pattern.split("|"))
    historical_words = set().union(*[
        set(h["pattern"].split("|"))
        for h in execution_history
        if h.get("file") == target_file
    ])

    novelty_score = len(words - historical_words) / len(words) if words else 0.0

    # 2. Calculate Uncertainty: compilation/test failure status
    # If build fails, uncertainty goes up, allowing the model to search again
    uncertainty_score = 1.0 if has_compile_error else 0.2

    # 3. Combine: approved if novelty OR uncertainty is high
    retrieval_weight = 0.3 * novelty_score + 0.7 * uncertainty_score

    action = "PROCEED_WITH_FULL_DATA" if retrieval_weight > 0.35 else "PROCEED_WITH_TRUNCATED_DATA"
    return {
        "action": action,
        "weight": retrieval_weight,
        "novelty": novelty_score,
        "uncertainty": uncertainty_score,
    }


def detect_structural_blind_spots(viewed_symbol_ids: set[str], repo_map: Any) -> list[dict[str, Any]]:
    """Find neighbor nodes in the repo map graph that are linked to viewed symbol IDs,

    have PageRank/importance > 0.4, but have not been viewed yet.
    """
    if not repo_map:
        return []

    symbols = list(getattr(repo_map, "all_symbols", None) or getattr(repo_map, "symbols", []))
    by_id = {}
    for sym in symbols:
        file_path = getattr(sym, "file_path", None) or getattr(sym, "file", "unknown")
        # Ensure ID format matches viewed symbol ID: file_path:name:start_line
        symbol_id = f"{file_path}:{sym.name}:{sym.start_line}"
        by_id[symbol_id] = sym

    adjacency = defaultdict(list)
    for source, target in getattr(repo_map, "reference_edges", ()):
        adjacency[source].append(target)
        adjacency[target].append(source)

    blind_spots = []
    for symbol_id in viewed_symbol_ids:
        for neighbor_id in adjacency.get(symbol_id, []):
            if neighbor_id not in viewed_symbol_ids and neighbor_id in by_id:
                neighbor_sym = by_id[neighbor_id]
                score = getattr(neighbor_sym, "score", 0.0)
                if score > 0.4:
                    blind_spots.append({
                        "id": neighbor_id,
                        "name": neighbor_sym.name,
                        "file": getattr(neighbor_sym, "file_path", None) or getattr(neighbor_sym, "file", "unknown"),
                        "score": score,
                    })

    # De-duplicate
    unique_blind_spots = {}
    for bs in blind_spots:
        unique_blind_spots[bs["id"]] = bs

    return list(unique_blind_spots.values())


class DecisionGravityController:
    """Coordinates next turn implicit state gravity and search intents."""

    def __init__(self) -> None:
        self.strict_truncation_enabled = False
        self.retrieval_disabled = False
        self.last_gravity = 1.0
        self.last_coverage = 0.0
        self.last_feedback = None

    def coordinate_next_turn(self, context_store: Any, env_state: Any) -> dict[str, Any]:
        # 1. coverage
        coverage_dict = context_store.search_cache.get("coverage", {})
        coverage = sum(coverage_dict.values()) / len(coverage_dict) if coverage_dict else 0.0
        self.last_coverage = coverage

        # 2. novelty (average novelty of past searches, or 1.0 if empty)
        novelty = getattr(env_state, "_last_novelty_value", 1.0)

        # 3. uncertainty
        has_compile_error = bool(env_state._validation_error())
        uncertainty = 1.0 if has_compile_error else 0.2

        # 4. gravity & search intensity
        gravity = calculate_retrieval_gravity(coverage, context_store.run_state.step)
        self.last_gravity = gravity
        search_intensity = (1.0 - coverage) * 0.4 + novelty * 0.2 + uncertainty * 0.4

        last_feedback = getattr(self, "last_feedback", None)
        self.last_feedback = None # Clear after reading

        feedback_prompt = ""
        if last_feedback:
            feedback_prompt = (
                f"\n\n### [CONVERGENCE FEEDBACK WARNING] ###\n"
                f"⚠️ Your previous search was flagged by the convergence manager:\n"
                f"\"{last_feedback}\"\n"
                f"Please synthesize the existing viewed symbols, call view_symbol_code to inspect implementation details, "
                f"or proceed to edit/answer instead of repeating search patterns."
            )

        override_layer = ""
        if gravity < 0.35 or search_intensity < 0.3 or last_feedback is not None:
            self.retrieval_disabled = True
            override_layer = (
                f"\n\n### [DECISION OVERRIDE LAYER] ###\n"
                f"- Status: CONVERGENCE MODE ACTIVE (Saturated Code Evidence)\n"
                f"- Retrieval: DISABLED\n"
                f"- Reasoning mode: SYNTHESIS ONLY\n"
                f"- Allowed actions: {{edit (`decision_edit`), answer (final response)}}\n"
                f"- Forbidden actions: {{all search/retrieval tools (grep_search, codebase_retrieve)}}\n"
                f"- Instruction: You MUST synthesize from CURRENT_CONTEXT only. DO NOT search."
            )
        else:
            self.retrieval_disabled = False

        if search_intensity < 0.3:
            self.strict_truncation_enabled = True
            gravity_prompt = (
                f"[DECISION_GRAVITY]: System entropy is low. Code evidence is saturated. "
                f"(Gravity: {gravity:.2f}). "
                f"Further searching is highly likely to waste tokens. "
                f"Please initiate decision_edit or enter final report formulation."
            )
        else:
            self.strict_truncation_enabled = False
            gravity_prompt = f"[DECISION_GRAVITY]: Active exploration is permitted. (Gravity: {gravity:.2f})."

        if feedback_prompt:
            gravity_prompt += feedback_prompt
        if override_layer:
            gravity_prompt += override_layer

        # 5. Sniff blind spots
        viewed_symbol_ids = set()
        for item in context_store.context_anchors.code:
            file_path = item.get("file")
            name = item.get("symbol")
            span = item.get("span") or [0, 0]
            if name and file_path:
                viewed_symbol_ids.add(f"{file_path}:{name}:{span[0]}")

        # Also get from summaries if possible
        for anchor_id in context_store.context_anchors.summaries:
            # Anchor IDs are in format file_path:start-end
            # Let's see if we can resolve the symbol name
            pass

        context_builder = getattr(env_state, "context_builder", None)
        repo_map = None
        if context_builder is not None:
            service = getattr(context_builder, "repo_map_service", None)
            if service is not None:
                try:
                    repo_map = getattr(service, "map", None)
                except Exception:
                    pass

        blind_spots = detect_structural_blind_spots(viewed_symbol_ids, repo_map)

        blind_spots_prompt = ""
        if blind_spots:
            blind_spots_parts = ["### [LOGICAL_BLIND_SPOTS]"]
            for bs in blind_spots[:3]:
                blind_spots_parts.append(
                    f"- Warning: You have inspected viewed code, but you have NOT yet looked at '{bs['name']}' in file '{bs['file']}' (PageRank: {bs['score']:.2f}), which is statistically linked to this data flow."
                )
            blind_spots_parts.append("- Uncertainty Level: HIGH (Potential bypass vulnerability unchecked).")
            blind_spots_prompt = "\n".join(blind_spots_parts)

        return {
            "gravity": gravity,
            "search_intensity": search_intensity,
            "gravity_prompt": gravity_prompt,
            "blind_spots_prompt": blind_spots_prompt,
            "strict_truncation": self.strict_truncation_enabled,
        }
