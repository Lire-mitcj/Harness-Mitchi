from __future__ import annotations

import math
import logging
import json
from collections import defaultdict
from collections.abc import Mapping, Set
from pathlib import Path
from typing import Any

from src.tools.grep_match_symbols import (
    grep_search_fingerprint,
    history_entry_matches_fingerprint,
)

log = logging.getLogger(__name__)


def _find_symbol_span_in_python_file(content: str, symbol_name: str) -> tuple[int, int] | None:
    import ast
    try:
        tree = ast.parse(content)

        # Build parent map to construct full dot-paths
        parent_map = {}
        for parent in ast.walk(tree):
            for child in ast.iter_child_nodes(parent):
                parent_map[child] = parent

        def get_full_name(node: Any) -> str:
            name_parts = []
            curr = node
            while curr is not None:
                if isinstance(curr, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    name_parts.append(curr.name)
                curr = parent_map.get(curr)
            return ".".join(reversed(name_parts))

        # First pass: try matching Function, AsyncFunction, and Class (including nested)
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                full_name = get_full_name(node)
                if node.name == symbol_name or full_name == symbol_name:
                    decorator_lines = [
                        getattr(decorator, "lineno", node.lineno)
                        for decorator in getattr(node, "decorator_list", [])
                    ]
                    start = min([getattr(node, "lineno", 1), *decorator_lines])
                    end = getattr(node, "end_lineno", start)
                    return start, end

        # Second pass: try matching Variable/Constant assignments (e.g. app = FastAPI())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == symbol_name:
                        start = getattr(node, "lineno", 1)
                        end = getattr(node, "end_lineno", start)
                        return start, end
    except Exception as exc:
        log.warning("Fact locking AST parsing failed: %s", exc)
    return None


def _find_symbol_span_by_regex(content: str, symbol_name: str) -> tuple[int, int] | None:
    import re
    lines = content.splitlines()
    last_part = symbol_name.split(".")[-1]

    for idx, line in enumerate(lines):
        def_match = re.search(r'\b(def|class|function)\s+' + re.escape(last_part) + r'\b', line)
        assign_match = re.search(r'\b' + re.escape(last_part) + r'\s*=\s*', line)
        escaped_name = re.escape(last_part)
        sql_match = re.search(
            r'\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\s+|TEMPORARY\s+)?(?:TABLE|VIEW)\s+(?:IF\s+NOT\s+EXISTS\s+)?'
            rf'(?:`{escaped_name}`|"{escaped_name}"|\'{escaped_name}\'|\b{escaped_name}\b)',
            line,
            re.IGNORECASE
        )

        if def_match or assign_match or sql_match:
            start_line = idx + 1
            if sql_match:
                end_line = start_line
                for j in range(idx, len(lines)):
                    if ';' in lines[j]:
                        end_line = j + 1
                        break
                    end_line = j + 1
                return start_line, end_line

            indent_match = re.match(r'^(\s*)', line)
            indent = indent_match.group(1) if indent_match else ""

            end_line = start_line
            for j in range(idx + 1, len(lines)):
                if not lines[j].strip():
                    continue
                curr_indent_match = re.match(r'^(\s*)', lines[j])
                curr_indent = curr_indent_match.group(1) if curr_indent_match else ""
                if len(curr_indent) <= len(indent) and lines[j].strip():
                    end_line = j
                    break
                end_line = j + 1
            return start_line, end_line
    return None


def resolve_symbol_span(
    target_file: str,
    requested_symbol: str,
    repo_map: Any = None,
) -> tuple[int, int] | None:
    """High-performance symbol span resolution using repo_map cache and AST/regex fallback."""
    if not target_file or not requested_symbol:
        return None

    # Step 1: Query ctags cached symbol spans in repo_map (0ms)
    if repo_map:
        norm_target = str(target_file).replace("\\", "/").strip().lstrip("./")
        symbols_by_file = getattr(repo_map, "symbols_by_file", {})
        for file_path, syms in symbols_by_file.items():
            norm_path = str(file_path).replace("\\", "/").strip().lstrip("./")
            if norm_path == norm_target:
                for sym in syms:
                    if sym.name == requested_symbol or sym.name.split(".")[-1] == requested_symbol:
                        return sym.start_line, sym.end_line

    # Step 2: Fallback to reading file on disk and parsing AST/regex
    try:
        file_path = Path(target_file)
        if file_path.exists():
            content = file_path.read_text(encoding="utf-8", errors="replace")
            span = None
            if file_path.suffix == ".py":
                span = _find_symbol_span_in_python_file(content, requested_symbol)
            if span is None:
                span = _find_symbol_span_by_regex(content, requested_symbol)
            return span
    except Exception as e:
        log.warning("Fallback symbol span lookup failed for %s: %s", requested_symbol, e)
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


def _manifest_stale_targets(manifest: Any) -> tuple[set[str], set[str]]:
    """Return (stale_files, stale_symbols) from the projected manifest.

    STALE items were invalidated by a prior edit and are allowed to be re-read.
    """
    stale_files: set[str] = set()
    stale_symbols: set[str] = set()
    items = getattr(manifest, "stale_items", None)
    if not items:
        return stale_files, stale_symbols
    for item in items:
        file = getattr(item, "file", None)
        if file:
            stale_files.add(str(file).replace("\\", "/").strip().lstrip("./"))
        symbol = getattr(item, "symbol", None)
        if symbol:
            stale_symbols.add(str(symbol))
    return stale_files, stale_symbols


def _anchor_has_complete_source(anchor: Mapping[str, Any]) -> bool:
    code = str(anchor.get("code") or anchor.get("verbatim_code") or "").strip()
    if not code:
        return False
    span = anchor.get("span") or []
    if not isinstance(span, (list, tuple)) or len(span) < 2:
        return False
    try:
        width = int(span[1]) - int(span[0]) + 1
    except (TypeError, ValueError):
        return False
    return width > 0 and len(code.splitlines()) >= width


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


async def inspect_fact_locking_async(
    tool_name: str,
    arguments: Mapping[str, Any],
    *,
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
    manifest: Any = None,
    edit_recovery: bool = False,
    rounds_since_last_edit: int = 0,
) -> str | None:
    """Layer 2: Async Fact Locking, line span matching, and semantic query gating.

    Fact locking blocks re-reading evidence that is already SATISFIED (present as
    a durable verbatim anchor). STALE manifest targets are exempt so the agent can
    refresh code that a prior edit invalidated. Policy about *which* tools to open
    lives in reallocate_tools, not here.
    """
    # Normalize modified_files and target file paths for clean checks
    normalized_modified_files = set()
    if modified_files:
        for f in modified_files:
            normalized_modified_files.add(str(f).replace("\\", "/").strip().lstrip("./"))

    stale_files, stale_symbols = _manifest_stale_targets(manifest)

    target_file = arguments.get("target_file") or arguments.get("path")
    normalized_target_file = ""
    if target_file:
        normalized_target_file = str(target_file).replace("\\", "/").strip().lstrip("./")

    is_target_modified = normalized_target_file in normalized_modified_files

    # 1. Verification Search Gate (Rely on git diff / validator instead of re-searching modified files)
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

    # 2. Fact Locking / Cache Overlap checking for view_symbol_code.
    # A checklist is descriptive state, not retrieval authorization; the
    # relevance preflight and reducer-provided allowed tools own that decision.
    if tool_name == "view_symbol_code":
        requested_symbol = arguments.get("symbol")
        if requested_symbol:
            # Bypass blocker if the target file or the symbol's file is modified (allowing verification)
            # or if the manifest marks this target STALE (needs a fresh read).
            bypass_blocker = (
                is_target_modified
                or requested_symbol in stale_symbols
                or normalized_target_file in stale_files
            )
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
                # 3.1 Check file-span range coverage (resolving physical spans from repo_map or AST)
                span = resolve_symbol_span(target_file, requested_symbol, repo_map)
                if span:
                    start_line, end_line = span
                    norm_file = str(target_file).replace("\\", "/").strip().lstrip("./")

                    def check_coverage(anchors):
                        if not anchors:
                            return None
                        for anchor in anchors:
                            anchor_file = anchor.get("file")
                            if not anchor_file:
                                continue
                            norm_anchor_file = str(anchor_file).replace("\\", "/").strip().lstrip("./")
                            if norm_anchor_file == norm_file:
                                anchor_span = anchor.get("span")
                                if anchor_span and isinstance(anchor_span, list) and len(anchor_span) >= 2:
                                    cache_start, cache_end = anchor_span[0], anchor_span[1]
                                    if cache_start <= start_line and end_line <= cache_end:
                                        cached_code = anchor.get("code") or anchor.get("verbatim_code") or ""
                                        if cached_code:
                                            lines = cached_code.splitlines()
                                            s_idx = start_line - cache_start
                                            e_idx = end_line - cache_start + 1
                                            if 0 <= s_idx < len(lines) and 0 < e_idx <= len(lines) + 1:
                                                sliced_code = "\n".join(lines[s_idx:e_idx])
                                                return {
                                                    "code": sliced_code,
                                                    "file": anchor_file,
                                                    "span": [start_line, end_line]
                                                }
                        return None

                    coverage = check_coverage(context_anchors_code)
                    if not coverage:
                        coverage = check_coverage(raw_evidence_store)

                    if coverage:
                        return (
                            f"BLOCK: Symbol '{requested_symbol}' lines "
                            f"[{start_line}-{end_line}] are already present in CURRENT_CONTEXT. "
                            "Reuse the cached slice; do not re-fetch."
                        )

                # 3.2 Symbol Name matching fallback (Condition A: Symbol in context)
                if context_anchors_code:
                    for anchor in context_anchors_code:
                        sym = anchor.get("symbol")
                        if (
                            sym
                            and sym == requested_symbol
                            and _anchor_has_complete_source(anchor)
                        ):
                            anchor_file = str(anchor.get("file") or "").replace(
                                "\\", "/"
                            ).strip().lstrip("./")
                            if (
                                normalized_target_file
                                and anchor_file
                                and anchor_file != normalized_target_file
                            ):
                                continue
                            cached_code = anchor.get("code") or anchor.get("verbatim_code") or ""
                            file_name = anchor.get("file") or "unknown"
                            span = anchor.get("span") or [1, 1]
                            return (
                                f"BLOCK: Symbol '{requested_symbol}' is already present in CURRENT_CONTEXT "
                                f"({file_name}:{span[0]}-{span[1]}). Reuse it; do not re-fetch."
                            )

                # 3.3 Symbol Name matching fallback (Condition B: Symbol loaded via previous step)
                if raw_evidence_store:
                    for anchor in raw_evidence_store:
                        sym = anchor.get("symbol")
                        if (
                            sym
                            and sym == requested_symbol
                            and _anchor_has_complete_source(anchor)
                        ):
                            anchor_file = str(anchor.get("file") or "").replace(
                                "\\", "/"
                            ).strip().lstrip("./")
                            if (
                                normalized_target_file
                                and anchor_file
                                and anchor_file != normalized_target_file
                            ):
                                continue
                            cached_code = anchor.get("code") or anchor.get("verbatim_code") or ""
                            file_name = anchor.get("file") or "unknown"
                            span = anchor.get("span") or [1, 1]
                            return (
                                f"BLOCK: Symbol '{requested_symbol}' was already loaded previously "
                                f"({file_name}:{span[0]}-{span[1]}). Reuse it; do not re-fetch."
                            )

    # 4. Redundant / empty grep prevention
    if tool_name == "grep_search":
        pattern = str(arguments.get("pattern") or "").strip()
        patterns = arguments.get("patterns")
        pattern_items: list[str] = []
        if isinstance(patterns, list):
            for item in patterns:
                text = str(item or "").strip()
                if text and text not in pattern_items:
                    pattern_items.append(text)
        if pattern and pattern not in pattern_items:
            pattern_items.insert(0, pattern)
        path = str(arguments.get("path") or ".").strip() or "."
        include = arguments.get("include")
        mode = str(arguments.get("mode") or "default").strip() or "default"
        fingerprint = grep_search_fingerprint(
            pattern_items or ([pattern] if pattern else []),
            path=path,
            include=str(include) if include else None,
            mode=mode,
        )

        if search_history and fingerprint and pattern_items:
            for entry in reversed(search_history):
                if not entry.get("empty"):
                    continue
                if not history_entry_matches_fingerprint(entry, fingerprint):
                    continue
                prior_patterns = entry.get("patterns") or [entry.get("pattern")]
                return (
                    "BLOCK: Repeated empty grep_search. This exact query already returned "
                    f"0 matches for {path!r} (patterns={prior_patterns!r}). "
                    "Broaden patterns, change include/path, or use view_symbol_code on "
                    "suggested_views / LOADED CODE ANCHORS instead of retrying."
                )

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
                            return (
                                "BLOCK: Redundant search. The pattern is already present in CURRENT_CONTEXT. "
                                "Reuse the existing context; do not re-search for confirmation."
                            )

    # 5. Convergence similarity scoring and search gating for grep_search
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
            max_sim = max(lexical_similarity, semantic_similarity, structural_connection)
            arguments["_novelty_score"] = 1.0 - max_sim

        # Redundancy gating to prevent no-novelty grep loops. Tool-opening policy
        # (convergence / gravity) is owned by reallocate_tools, not fact locking.
        # After a blocked edit or post-edit refresh window, allow recovery grep/view.
        post_edit_refresh = bool(
            modified_files
            and rounds_since_last_edit <= 1
            and manifest is not None
            and getattr(manifest, "has_stale", False)
        )
        if not has_compile_error and not edit_recovery and not post_edit_refresh:
            # C. Symbol Dominance Gate
            if structural_connection == 1.0 and (lexical_similarity > 0.8 or semantic_similarity > 0.8):
                return (
                    f"BLOCK: Symbol dominance detected. You are repeatedly searching within the same structural "
                    f"dependency area ('{path}') that you have already explored. Please synthesize your findings, "
                    f"call view_symbol_code to inspect core implementation details, or proceed to edit/answer."
                )

            # B. Redundancy Gate
            if lexical_similarity > 0.85 or semantic_similarity > 0.85:
                return (
                    f"BLOCK: Redundant search query. The pattern '{pattern}' shares high semantic overlap "
                    f"(Lexical: {lexical_similarity:.2f}, Semantic: {semantic_similarity:.2f} > 0.85) "
                    f"with your recent search queries in '{path}'. Please proceed to edit/answer or broaden your search."
                )

    return None
