from __future__ import annotations

import math
import logging
from collections import defaultdict
from collections.abc import Mapping, Set
from typing import Any

log = logging.getLogger(__name__)


def inspect_tool_request(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
) -> str | None:
    """Read-only request validation and safe argument pruning used before dispatching a tool event.

    === HOOK RESPONSIBILITY BOUNDARIES ===
    🟢 ALLOWED:
      - Physical Constraints: Validation of argument presence, type-safety, and range checks.
      - Information Pruning (信息修剪): In-place trimming of parameters, path normalization,
        and capping excessive result limits (e.g., max_results) to control context blowout.
    
    ❌ STRICTLY PROHIBITED (DO NOT DO):
      1. No Decisions: Cannot choose/switch between editing or answering, cannot decide retries,
         and cannot evaluate or trigger phase transitions.
      2. No Planning: Cannot rewrite the search intent, alter queries semantically, or expand query tokens.
      3. No State Mutation: Cannot touch, update, or mutate any RunState/phase attributes.

    Returns:
        str | None: An error message if physical constraints are violated, otherwise None.
    """
    # 1. Allowed tools boundary check
    if tool_name not in allowed_tools:
        return f"Tool {tool_name!r} is not in the reducer-provided allow list."

    # In-place pruning / information pruning if mutable dictionary
    is_mutable = isinstance(arguments, dict)

    # 2. Grep search validation & pruning
    if tool_name == "grep_search":
        pattern = arguments.get("pattern")
        if pattern is None:
            return "grep_search requires a 'pattern' parameter."
        
        pattern_str = str(pattern).strip()
        if not pattern_str:
            return "grep_search requires a non-empty pattern."
            
        if is_mutable:
            # Pruning redundant quotes, backticks, or trailing newlines in search patterns
            if (pattern_str.startswith("`") and pattern_str.endswith("`")) or (pattern_str.startswith('"') and pattern_str.endswith('"')):
                pattern_str = pattern_str[1:-1].strip()
            arguments["pattern"] = pattern_str

            # Prune and normalize path
            if "path" in arguments:
                arguments["path"] = str(arguments["path"]).strip().rstrip("/")
            
            # Prune and cap excessive limits (max_results) to prevent token blowout
            if "max_results" in arguments:
                try:
                    val = int(arguments["max_results"])
                    if val > 100:
                        arguments["max_results"] = 50
                except (ValueError, TypeError):
                    pass

    # 3. View symbol validation & pruning
    elif tool_name == "view_symbol_code":
        target_file = arguments.get("target_file")
        symbol = arguments.get("symbol")
        
        if not target_file or not str(target_file).strip():
            return "view_symbol_code requires target_file."
        if not symbol or not str(symbol).strip():
            return "view_symbol_code requires symbol."

        if is_mutable:
            arguments["target_file"] = str(target_file).strip()
            arguments["symbol"] = str(symbol).strip()

    # 4. Decision edit validation & pruning
    elif tool_name == "decision_edit":
        target_file = arguments.get("target_file")
        intent = arguments.get("intent")
        if not target_file or not str(target_file).strip():
            return "SIGNAL: Invalid Task Packet. Missing required 'target_file' field."
        if not intent or not str(intent).strip():
            return "SIGNAL: Invalid Task Packet. Missing required 'intent' description."

        # Verify focus_symbols format
        focus_symbols = arguments.get("focus_symbols")
        if focus_symbols is not None:
            if not isinstance(focus_symbols, list):
                return "SIGNAL: Invalid Task Packet. 'focus_symbols' must be a JSON array of strings."
            for idx, item in enumerate(focus_symbols):
                if not isinstance(item, str):
                    return f"SIGNAL: Invalid Task Packet. 'focus_symbols' index {idx} must be a string."

        # Verify context_window format
        context_window = arguments.get("context_window")
        if context_window is not None:
            if not isinstance(context_window, list):
                return "SIGNAL: Invalid Task Packet. 'context_window' must be a JSON array of objects."
            for idx, item in enumerate(context_window):
                if not isinstance(item, dict):
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} must be a JSON object."
                
                ref_file = item.get("file")
                ref_span = item.get("span")
                if not ref_file or not isinstance(ref_file, str) or not ref_file.strip():
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} is missing a valid 'file' string."
                if ref_span is None:
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} is missing a 'span' range."
                if not isinstance(ref_span, list) or len(ref_span) < 2:
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} 'span' must be a list containing [start_line, end_line]."
                try:
                    start_line = int(ref_span[0])
                    end_line = int(ref_span[1])
                    if start_line > end_line:
                        return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} 'span' start_line {start_line} cannot be greater than end_line {end_line}."
                except (ValueError, TypeError):
                    return f"SIGNAL: Invalid Task Packet. 'context_window' index {idx} 'span' elements must be valid integers."

        # Verify constraints format
        constraints = arguments.get("constraints")
        if constraints is not None:
            if not isinstance(constraints, dict):
                return "SIGNAL: Invalid Task Packet. 'constraints' must be a JSON object."

        if is_mutable:
            arguments["target_file"] = str(target_file).strip()
            arguments["intent"] = str(intent).strip()

    return None


def get_jaccard_similarity(s1: str, s2: str) -> float:
    """Computes Jaccard similarity of split tokens between two search patterns."""
    w1 = set(s1.lower().split("|"))
    w2 = set(s2.lower().split("|"))
    if not w1 or not w2:
        return 0.0
    return len(w1 & w2) / len(w1 | w2)


def cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Computes cosine similarity between two float vectors."""
    if not v1 or not v2:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def get_structural_connection(
    target_path: str,
    search_history: list[dict[str, Any]],
    repo_map: Any,
) -> float:
    """Traces if the target_path is within graph distance 2 of any previously searched files."""
    if not repo_map or not search_history:
        return 0.0

    # Build file-level adjacency list from repository reference edges
    symbols = list(getattr(repo_map, "all_symbols", None) or getattr(repo_map, "symbols", []))
    adjacency = defaultdict(list)
    for source, target in getattr(repo_map, "reference_edges", ()):
        src_file = source.split(":")[0]
        tgt_file = target.split(":")[0]
        if src_file != tgt_file:
            adjacency[src_file].append(tgt_file)
            adjacency[tgt_file].append(src_file)

    history_files = {h["file"] for h in search_history if "file" in h}
    t_file = target_path.strip().rstrip("/")
    if t_file in history_files:
        return 1.0

    # BFS to check if target file is within graph distance 2 of any file in history_files
    visited = {t_file}
    queue = [(t_file, 0)]
    while queue:
        curr, dist = queue.pop(0)
        if curr in history_files:
            return 1.0 if dist <= 2 else 0.0
        if dist < 2:
            for neighbor in adjacency.get(curr, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, dist + 1))

    return 0.0


async def inspect_tool_request_async(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
    allowed_tools: Set[str],
    has_compile_error: bool = False,
    search_history: list[dict[str, Any]] | None = None,
    repo_map: Any = None,
    embedder: Any = None,
    embeddings_cache: dict[str, list[float]] | None = None,
    gravity_controller: Any = None,
    checklist: list[str] | None = None,
    context_anchors_code: list[dict[str, Any]] | None = None,
    raw_evidence_store: list[dict[str, Any]] | None = None,
    git_diff: str | None = None,
    modified_files: list[str] | None = None,
) -> str | None:
    """Runs async verification checks, pruning parameters, and computing convergence signals

    (Lexical Jaccard, Semantic Similarity, and Structural graph connection distance).
    """
    # Normalize modified_files and target file paths for clean checks
    normalized_modified_files = set()
    if modified_files:
        for f in modified_files:
            normalized_modified_files.add(str(f).replace("\\", "/").strip().lstrip("./"))

    target_file = arguments.get("target_file") or arguments.get("path")
    normalized_target_file = ""
    if target_file:
        normalized_target_file = str(target_file).replace("\\", "/").strip().lstrip("./")

    is_target_modified = normalized_target_file in normalized_modified_files

    # 0.0 Verification Search Gate (Rely on git diff / validator instead of re-searching modified files)
    if git_diff and (tool_name == "grep_search" or tool_name == "view_symbol_code"):
        # Exempt view_symbol_code on modified files
        if not (tool_name == "view_symbol_code" and is_target_modified):
            pattern = arguments.get("pattern") or arguments.get("symbol")
            if pattern:
                pattern_str = str(pattern).strip()
                normalized_pattern = pattern_str.strip('"`\'')
                
                # Check if we are searching modified code
                if is_target_modified or not target_file:
                    diff_lines = []
                    for line in git_diff.splitlines():
                        if line.startswith('+') or line.startswith('-'):
                            if not line.startswith('+++') and not line.startswith('---'):
                                diff_lines.append(line[1:])
                    
                    for diff_line in diff_lines:
                        if normalized_pattern in diff_line:
                            modified_desc = ", ".join(modified_files) if modified_files else "recently modified files"
                            return (
                                f"BLOCK: Verification search detected. The pattern '{pattern_str}' was found in the "
                                f"active git diff of {modified_desc}. Verification queries on modified code are forbidden "
                                f"since the edit tool already returned the unified diff. Rely on validator results."
                            )

    # 0.1 RULE 10 — PLAN LOCK (CRITICAL)
    if checklist and len(checklist) > 0:
        # EXECUTION MODE is active
        if tool_name == "codebase_retrieve":
            return (
                "BLOCK: codebase_retrieve is DISABLED in EXECUTION MODE (Plan Lock active). "
                "You have already produced a plan/checklist, so you must execute the steps using "
                "decision_edit, view_symbol_code, or final response."
            )
        elif tool_name == "grep_search":
            if not has_compile_error:
                mode = arguments.get("mode", "default")
                pattern = arguments.get("pattern", "")
                if mode not in ("symbol", "import"):
                    return (
                        f"BLOCK: grep_search is DISABLED in EXECUTION MODE (Plan Lock active) "
                        f"unless performing explicit symbol or import scanning for missing symbols. "
                        f"Your query '{pattern}' (mode: {mode}) is rejected."
                    )
                # Check if search targets a symbol already in context
                if context_anchors_code:
                    for anchor in context_anchors_code:
                        sym = anchor.get("symbol")
                        if sym and sym == pattern:
                            return f"BLOCK: Redundant search. Symbol '{pattern}' is already present in CURRENT_CONTEXT."
    # 1. Sync validations & parameter trimming
    err = inspect_tool_request(tool_name, arguments, allowed_tools=allowed_tools)
    if err:
        return err

    # 1.5 Global Redundant Read Prevention for view_symbol_code (stops loop query issues)
    if tool_name == "view_symbol_code":
        requested_symbol = arguments.get("symbol")
        if requested_symbol:
            # Bypass blocker if the target file or the symbol's file is modified (allowing verification)
            bypass_blocker = is_target_modified
            if not bypass_blocker and normalized_modified_files:
                matching_files = set()
                if context_anchors_code:
                    for anchor in context_anchors_code:
                        if anchor.get("symbol") == requested_symbol and anchor.get("file"):
                            matching_files.add(str(anchor["file"]).replace("\\", "/").strip().lstrip("./"))
                if raw_evidence_store:
                    for anchor in raw_evidence_store:
                        if anchor.get("symbol") == requested_symbol and anchor.get("file"):
                            matching_files.add(str(anchor["file"]).replace("\\", "/").strip().lstrip("./"))
                if matching_files & normalized_modified_files:
                    bypass_blocker = True

            if not bypass_blocker:
                # Condition A: Symbol in context
                if context_anchors_code:
                    for anchor in context_anchors_code:
                        sym = anchor.get("symbol")
                        if sym and sym == requested_symbol:
                            cached_code = anchor.get("code") or anchor.get("verbatim_code") or ""
                            file_name = anchor.get("file") or "unknown"
                            span = anchor.get("span") or [1, 1]
                            import json
                            res_payload = {
                                "info": f"Redundant read. Symbol '{requested_symbol}' is already present in CURRENT_CONTEXT (Fact Locking active).",
                                "file": file_name,
                                "span": span,
                                "observation_code": cached_code,
                                "verbatim_code": cached_code,
                            }
                            return "SUCCESS: " + json.dumps(res_payload, ensure_ascii=False)
                # Condition B: Symbol already loaded via previous step
                if raw_evidence_store:
                    for anchor in raw_evidence_store:
                        sym = anchor.get("symbol")
                        if sym and sym == requested_symbol:
                            cached_code = anchor.get("code") or anchor.get("verbatim_code") or ""
                            file_name = anchor.get("file") or "unknown"
                            span = anchor.get("span") or [1, 1]
                            import json
                            res_payload = {
                                "info": f"Redundant read. Symbol '{requested_symbol}' was already loaded in a previous step and is cached. Synthesize from existing cache instead of re-fetching.",
                                "file": file_name,
                                "span": span,
                                "observation_code": cached_code,
                                "verbatim_code": cached_code,
                            }
                            return "SUCCESS: " + json.dumps(res_payload, ensure_ascii=False)

    # 1.6 Global Redundant Search Prevention for grep_search (prevents redundant search loops)
    if tool_name == "grep_search":
        pattern = arguments.get("pattern", "")
        if pattern:
            if context_anchors_code:
                for anchor in context_anchors_code:
                    sym = anchor.get("symbol")
                    if sym and sym == pattern:
                        is_modified = False
                        anchor_file = anchor.get("file")
                        if anchor_file:
                            normalized_anchor_file = str(anchor_file).replace("\\", "/").strip().lstrip("./")
                            if normalized_anchor_file in normalized_modified_files:
                                is_modified = True
                        if not is_modified:
                            cached_code = anchor.get("code") or anchor.get("verbatim_code") or ""
                            file_name = anchor.get("file") or "unknown"
                            span = anchor.get("span") or [1, 1]
                            import json
                            res_payload = {
                                "info": f"Redundant search. Symbol '{pattern}' is already present in CURRENT_CONTEXT.",
                                "matches": [
                                    {
                                        "file": file_name,
                                        "span": span,
                                        "match_line": cached_code.splitlines()[0] if cached_code else "",
                                    }
                                ],
                                "returned_matches": 1,
                                "total_matches": 1,
                            }
                            return "SUCCESS: " + json.dumps(res_payload, ensure_ascii=False)

    # 2. Convergence signal compilation
    if tool_name == "grep_search" and search_history:
        pattern = arguments.get("pattern", "")
        path = arguments.get("path", ".")

        # Lexical signal check
        lexical_similarity = 0.0
        for h in search_history:
            sim = get_jaccard_similarity(pattern, h.get("pattern", ""))
            if sim > lexical_similarity:
                lexical_similarity = sim

        # Semantic signal check
        semantic_similarity = 0.0
        if embedder:
            try:
                # Retrieve or calculate embedding of current pattern
                if embeddings_cache is not None and pattern in embeddings_cache:
                    new_emb = embeddings_cache[pattern]
                else:
                    new_emb = await embedder.embed(pattern)
                    if embeddings_cache is not None:
                        embeddings_cache[pattern] = new_emb

                # Retrieve or calculate embedding for previous patterns
                for h in search_history:
                    hist_pattern = h.get("pattern", "")
                    if hist_pattern:
                        if embeddings_cache is not None and hist_pattern in embeddings_cache:
                            hist_emb = embeddings_cache[hist_pattern]
                        else:
                            hist_emb = await embedder.embed(hist_pattern)
                            if embeddings_cache is not None:
                                embeddings_cache[hist_pattern] = hist_emb

                        sim = cosine_similarity(new_emb, hist_emb)
                        if sim > semantic_similarity:
                            semantic_similarity = sim
            except Exception as e:
                log.warning("Embedding similarity evaluation failed: %s", e)

        # Structural signal check
        structural_connection = get_structural_connection(path, search_history, repo_map)

        if isinstance(arguments, dict):
            arguments["_lexical_similarity"] = lexical_similarity
            arguments["_semantic_similarity"] = semantic_similarity
            arguments["_structural_connection"] = structural_connection
            
            # Combine signals to a unified novelty score
            # Novelty drops if Jaccard Jaccard matches, embedding matches, or graph connectivity is adjacent
            max_sim = max(lexical_similarity, semantic_similarity, structural_connection)
            arguments["_novelty_score"] = 1.0 - max_sim

        # 3. Gating checks to enforce search default policies and drive LLM decisions
        if not has_compile_error:
            # A. Coverage Gate (DECISION_GRAVITY < 0.3 indicating evidence saturation)
            if gravity_controller is not None:
                last_gravity = getattr(gravity_controller, "last_gravity", 1.0)
                if last_gravity < 0.3:
                    return (
                        f"BLOCK_SEARCH_FORCE_EDIT: Retrieval is complete. All required code evidence has been saturated "
                        f"(DECISION_GRAVITY: {last_gravity:.2f} < 0.3). You MUST stop calling search/retrieval tools "
                        f"and immediately proceed to edit or formulate final response."
                    )

            # C. Symbol Dominance Gate (Exploring within the same structural area and repeating queries)
            if structural_connection == 1.0 and (lexical_similarity > 0.8 or semantic_similarity > 0.8):
                return (
                    f"BLOCK: Symbol dominance detected. You are repeatedly searching within the same structural "
                    f"dependency area ('{path}') that you have already explored. Please synthesize your findings, "
                    f"call view_symbol_code to inspect core implementation details, or proceed to edit/answer."
                )

            # B. Redundancy Gate (Lexical or Semantic similarity > 0.85 with recent search history)
            if lexical_similarity > 0.85 or semantic_similarity > 0.85:
                return (
                    f"BLOCK: Redundant search query. The pattern '{pattern}' shares high semantic overlap "
                    f"(Lexical: {lexical_similarity:.2f}, Semantic: {semantic_similarity:.2f} > 0.85) "
                    f"with your recent search queries in '{path}'. Please proceed to edit/answer or broaden your search."
                )

    return None
