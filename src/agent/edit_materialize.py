"""Synthesize classic SEARCH/REPLACE patches from REPLACE-only edit sites.

EditLLM no longer emits SEARCH (verbatim copy is its weakest skill). It emits
locate keys (symbol / span) plus the new text; this module reads the on-disk
span and builds the SEARCH/REPLACE blocks ``CursorPatchApplier`` already knows.
"""

from __future__ import annotations

import re
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path
from typing import Any

from src.tools.assembled.view_symbol_code import (
    _find_symbol_span_by_regex,
    _find_symbol_span_in_python_file,
)

_DEF_OR_CLASS_NAME_RE = re.compile(
    r"(?m)^[ \t]*(?:async[ \t]+)?(?:def|class)[ \t]+([A-Za-z_][A-Za-z0-9_]*)"
)

_SEARCH_MARKER = "<<<<<<< SEARCH"
_REPLACE_OPEN = "<<<<<<< REPLACE"
_REPLACE_CLOSE = ">>>>>>> REPLACE"

_SITE_REPLACE_RE = re.compile(
    r"(?im)^[ \t]*SITE[ \t]*:[ \t]*(?P<header>[^\n]*)\n"
    r"(?:^[ \t]*MODE[ \t]*:[ \t]*(?P<mode_line>[^\n]*)\n)?"
    r"(?:^[ \t]*ANCHOR[ \t]*:[ \t]*(?P<anchor>[^\n]*)\n)?"
    r"<<<<<<< REPLACE[ \t]*\n"
    r"(?P<body>.*?)\n"
    r">>>>>>> REPLACE[ \t]*(?:\n|$)",
    re.DOTALL,
)
_BARE_REPLACE_RE = re.compile(
    r"<<<<<<< REPLACE[ \t]*\n"
    r"(?P<body>.*?)\n"
    r">>>>>>> REPLACE[ \t]*(?:\n|$)",
    re.DOTALL,
)
_HEADER_KV_RE = re.compile(
    r"(?i)\b(symbol|span|mode)\s*=\s*([^\s,]+)"
)
# CURRENT_CONTEXT uses ``{line}: {text}`` (see context_pack_builder._numbered_lines).
_CONTEXT_LINE_PREFIX_RE = re.compile(r"^[ \t]*(\d+):[ \t](.*)$")
_LEGACY_SPLIT_RE = re.compile(r"(?m)^=======[ \t]*$")

_DELTA_MODES = frozenset({
    "insert_after",
    "after",
    "insert_before",
    "before",
    "replace_anchor",
    "replace_line",
    "swap",
})
_FULL_REPLACE_MODES = frozenset({"replace", "full", ""})


class MaterializeError(ValueError):
    """Locate key missing or span/symbol could not be resolved on disk."""


def is_legacy_search_replace(patch: str) -> bool:
    return _SEARCH_MARKER in (patch or "")


def parse_replace_sites(patch: str) -> list[dict[str, Any]]:
    """Parse SITE + REPLACE-only blocks (or a single bare REPLACE)."""
    text = (patch or "").strip()
    if not text:
        return []
    sites: list[dict[str, Any]] = []
    for match in _SITE_REPLACE_RE.finditer(text):
        header = _parse_site_header(match.group("header") or "")
        mode = str(match.group("mode_line") or header.get("mode") or "").strip()
        anchor = match.group("anchor")
        sites.append({
            "symbol": header.get("symbol"),
            "span": header.get("span"),
            "mode": mode or None,
            "anchor": (anchor.strip() if isinstance(anchor, str) else None),
            "body": match.group("body"),
        })
    if sites:
        return sites
    for match in _BARE_REPLACE_RE.finditer(text):
        sites.append({
            "symbol": None,
            "span": None,
            "mode": None,
            "anchor": None,
            "body": match.group("body"),
        })
    return sites


def _parse_site_header(header: str) -> dict[str, Any]:
    symbol: str | None = None
    span: tuple[int, int] | None = None
    mode: str | None = None
    for match in _HEADER_KV_RE.finditer(header or ""):
        key = match.group(1).casefold()
        value = match.group(2).strip().strip("\"'")
        if key == "symbol":
            symbol = value
        elif key == "mode":
            mode = value
        elif key == "span":
            span_match = re.match(r"(\d+)\s*[-:]\s*(\d+)$", value)
            if span_match:
                start = int(span_match.group(1))
                end = int(span_match.group(2))
                if start > 0 and end >= start:
                    span = (start, end)
    return {"symbol": symbol, "span": span, "mode": mode}


def strip_context_line_prefixes(text: str) -> str:
    """Remove CURRENT_CONTEXT ``N: `` prefixes when most lines look numbered."""
    raw = (text or "").replace("\r\n", "\n")
    lines = raw.splitlines()
    if not lines:
        return raw
    matched = [_CONTEXT_LINE_PREFIX_RE.match(line) for line in lines]
    hit_count = sum(1 for item in matched if item is not None)
    if hit_count < max(1, int(len(lines) * 0.8)):
        return raw
    numbers = [int(item.group(1)) for item in matched if item is not None]
    if len(numbers) >= 2 and numbers[-1] < numbers[0]:
        return raw
    return "\n".join(
        item.group(2) if item is not None else line
        for item, line in zip(matched, lines)
    )


def split_legacy_replace_body(text: str) -> str:
    """If REPLACE still embeds ``old\\n=======\\nnew``, keep only the new half."""
    raw = (text or "").replace("\r\n", "\n")
    if not _LEGACY_SPLIT_RE.search(raw):
        return raw
    parts = _LEGACY_SPLIT_RE.split(raw, maxsplit=1)
    if len(parts) != 2:
        return raw
    return parts[1].lstrip("\n")


def normalize_replace_body(body: str) -> str:
    """Normalize EditLLM REPLACE text before comparing / applying."""
    return split_legacy_replace_body(strip_context_line_prefixes(body or ""))


def _fingerprint(text: str) -> str:
    return "\n".join(
        line.rstrip() for line in (text or "").replace("\r\n", "\n").splitlines()
    ).strip()


def _line_similarity(left: str, right: str) -> float:
    return SequenceMatcher(None, left, right).ratio()


def locate_surgical_hunk(
    old_code: str,
    new_code: str,
) -> tuple[str, str] | None:
    """If ``new_code`` is a near-edit of a contiguous window in ``old_code``, shrink.

    Models often emit only the changed lines instead of the whole SITE symbol.
    When that fragment uniquely aligns to a same-length window with high
    similarity (not identical), return ``(window, new_code)`` for SEARCH/REPLACE.
    """
    old_lines = (old_code or "").splitlines()
    new_lines = (new_code or "").splitlines()
    if not new_lines or not old_lines:
        return None
    if len(new_lines) >= len(old_lines):
        return None

    n = len(new_lines)
    best: tuple[float, int, int] | None = None
    for start in range(0, len(old_lines) - n + 1):
        window = old_lines[start : start + n]
        if window == new_lines:
            continue  # identical window → surgical no-op
        score = sum(
            _line_similarity(left, right)
            for left, right in zip(window, new_lines)
        ) / n
        # Single-line edits need a stronger match to avoid wrong anchors.
        min_score = 0.72 if n == 1 else 0.55
        if score < min_score:
            continue
        if best is None or score > best[0]:
            best = (score, start, start + n)

    if best is None:
        return None
    _, start, end = best
    # Require the best window to be a clear winner (avoid ambiguous ties).
    winners = 0
    min_score = 0.72 if n == 1 else 0.55
    for start_i in range(0, len(old_lines) - n + 1):
        window = old_lines[start_i : start_i + n]
        if window == new_lines:
            continue
        score = sum(
            _line_similarity(left, right)
            for left, right in zip(window, new_lines)
        ) / n
        if score < min_score:
            continue
        if score >= best[0] - 1e-9:
            winners += 1
            if winners > 1:
                return None
    search = "\n".join(old_lines[start:end])
    return search, new_code


def _find_anchor_line_index(lines: list[str], anchor: str) -> int:
    needle = strip_context_line_prefixes(anchor).rstrip("\n")
    needle_stripped = needle.strip()
    if not needle_stripped:
        raise MaterializeError("E1_FORMAT: ANCHOR is empty")

    exact = [i for i, line in enumerate(lines) if line == needle]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise MaterializeError(
            f"E2_LOCATE: ANCHOR matches {len(exact)} identical lines; "
            "narrow SITE span or make ANCHOR unique."
        )

    stripped = [i for i, line in enumerate(lines) if line.strip() == needle_stripped]
    if len(stripped) == 1:
        return stripped[0]
    if len(stripped) > 1:
        raise MaterializeError(
            f"E2_LOCATE: ANCHOR matches {len(stripped)} lines after strip; "
            "narrow SITE or use a longer unique ANCHOR."
        )

    contains = [i for i, line in enumerate(lines) if needle_stripped in line]
    if len(contains) == 1:
        return contains[0]
    if not contains:
        raise MaterializeError(
            f"E2_LOCATE: ANCHOR not found in SITE span: {needle_stripped!r}"
        )
    raise MaterializeError(
        f"E2_LOCATE: ANCHOR substring matches {len(contains)} lines; "
        "copy a more specific on-disk line into ANCHOR."
    )


def apply_delta_in_span(
    old_code: str,
    *,
    mode: str,
    anchor: str | None,
    new_body: str,
) -> tuple[str, str]:
    """Build a tiny SEARCH/REPLACE hunk inside ``old_code`` for delta modes.

    Modes:
      insert_after / after   — insert ``new_body`` after the ANCHOR line
      insert_before / before — insert ``new_body`` before the ANCHOR line
      replace_anchor / swap  — replace the ANCHOR line with ``new_body``
    """
    normalized_mode = str(mode or "").casefold().strip()
    if normalized_mode not in _DELTA_MODES:
        raise MaterializeError(
            f"E1_FORMAT: unknown MODE={mode!r}; use insert_after, insert_before, "
            "replace_anchor, or omit MODE for full-span replace."
        )
    if not (anchor or "").strip():
        raise MaterializeError(
            f"E1_FORMAT: MODE={normalized_mode} requires "
            "`ANCHOR: <exact on-disk line from CURRENT_CONTEXT>`."
        )
    body = str(new_body or "").rstrip("\n")
    if not body.strip():
        raise MaterializeError("E1_FORMAT: delta REPLACE body is empty")

    lines = (old_code or "").splitlines()
    if not lines:
        raise MaterializeError("E2_LOCATE: SITE span is empty on disk")
    idx = _find_anchor_line_index(lines, anchor or "")
    anchor_text = lines[idx]

    if normalized_mode in {"insert_after", "after"}:
        return anchor_text, f"{anchor_text}\n{body}"
    if normalized_mode in {"insert_before", "before"}:
        return anchor_text, f"{body}\n{anchor_text}"
    # replace_anchor / replace_line / swap
    if _fingerprint(body) == _fingerprint(anchor_text):
        raise MaterializeError(
            "E1_FORMAT: replace_anchor REPLACE equals ANCHOR; produce a real change."
        )
    return anchor_text, body


def list_file_symbol_names(project_root: Path, target_file: str) -> list[str]:
    """Best-effort top-level def/class names in ``target_file`` (for fuzzy SITE)."""
    path = (project_root / target_file).resolve()
    if not path.is_file():
        return []
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return []
    names: list[str] = []
    seen: set[str] = set()
    for match in _DEF_OR_CLASS_NAME_RE.finditer(content):
        name = match.group(1)
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def locate_symbol_span(
    project_root: Path,
    target_file: str,
    symbol: str,
) -> tuple[int, int] | None:
    """Resolve a symbol to 1-based inclusive line span on disk."""
    name = str(symbol or "").strip()
    if not name:
        return None
    path = (project_root / target_file).resolve()
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    span = None
    if path.suffix.casefold() == ".py":
        span = _find_symbol_span_in_python_file(content, name)
    if span is None:
        span = _find_symbol_span_by_regex(content, name, file_path=target_file)
    return span


def _close_disk_symbol_names(
    name: str,
    disk_names: list[str],
    *,
    cutoff: float = 0.55,
) -> list[str]:
    """Rank near on-disk names for a possibly-hallucinated symbol."""
    needle = str(name or "").strip()
    if not needle or not disk_names:
        return []
    close = get_close_matches(needle, disk_names, n=3, cutoff=cutoff)
    if close:
        return close
    # Token overlap fallback: load_noise_policy_from_dict ≈ load_noise_policy_from_path
    name_tokens = {t for t in re.split(r"[_\W]+", needle.casefold()) if t}
    scored: list[tuple[float, str]] = []
    for candidate in disk_names:
        cand_tokens = {t for t in re.split(r"[_\W]+", candidate.casefold()) if t}
        if not name_tokens or not cand_tokens:
            continue
        overlap = len(name_tokens & cand_tokens) / len(name_tokens | cand_tokens)
        if overlap >= 0.5:
            scored.append((overlap, candidate))
    scored.sort(key=lambda item: (-item[0], len(item[1])))
    return [item[1] for item in scored[:3]]


def sanitize_focus_symbols(
    project_root: Path,
    target_file: str,
    focus_symbols: list[str] | None,
    *,
    remap_cutoff: float = 0.72,
) -> tuple[list[str], list[str], dict[str, str]]:
    """Keep only on-disk focus symbols; high-confidence remap typos; drop the rest.

    Core often invents names for symbols it plans to *add* (not yet on disk) or
    mistypes loaders (``_load_…_from_dict`` vs ``load_…_from_path``). Passing those
    through teaches EditLLM to emit ``SITE: symbol=<hallucination>`` → E2_LOCATE.

    Returns ``(kept, dropped, remapped)`` where ``remapped`` maps Core name → disk.
    """
    disk_names = list_file_symbol_names(project_root, target_file)
    disk_set = set(disk_names)
    kept: list[str] = []
    dropped: list[str] = []
    remapped: dict[str, str] = {}
    seen: set[str] = set()

    for raw in focus_symbols or []:
        name = str(raw or "").strip()
        if not name:
            continue
        if name in disk_set:
            if name not in seen:
                seen.add(name)
                kept.append(name)
            continue
        close = _close_disk_symbol_names(name, disk_names, cutoff=remap_cutoff)
        # Require SequenceMatcher ratio ≥ remap_cutoff so weak hits like
        # strip_text_mention ≈ strip_chat_noise (~0.53) are dropped, not remapped.
        chosen: str | None = None
        for candidate in close:
            ratio = SequenceMatcher(None, name, candidate).ratio()
            if ratio >= remap_cutoff:
                chosen = candidate
                break
        if chosen is None:
            dropped.append(name)
            continue
        remapped[name] = chosen
        if chosen not in seen:
            seen.add(chosen)
            kept.append(chosen)
    return kept, dropped, remapped


def resolve_symbol_alias(
    project_root: Path,
    target_file: str,
    symbol: str,
    *,
    focus_symbols: list[str] | None = None,
) -> tuple[str, tuple[int, int]] | None:
    """Exact locate, else fuzzy-match a near name on disk / in focus_symbols."""
    name = str(symbol or "").strip()
    if not name:
        return None
    exact = locate_symbol_span(project_root, target_file, name)
    if exact is not None:
        return name, exact

    disk_names = list_file_symbol_names(project_root, target_file)
    focus = [str(s).strip() for s in (focus_symbols or []) if str(s).strip()]
    # Prefer focus_symbols that actually exist on disk, then remaining disk names.
    focus_on_disk = [s for s in focus if s in disk_names]
    pool: list[str] = []
    for candidate in focus_on_disk + disk_names:
        if candidate not in pool:
            pool.append(candidate)
    if not pool:
        return None

    # Underscore / from_dict vs from_path style drift is common with weak Edit models.
    close = _close_disk_symbol_names(name, pool, cutoff=0.55)
    for candidate in close:
        located = locate_symbol_span(project_root, target_file, candidate)
        if located is not None:
            return candidate, located
    return None


def _span_from_context_window(
    target_file: str,
    context_window: list[dict[str, Any]] | None,
    index: int,
) -> tuple[int, int] | None:
    if not context_window:
        return None
    norm_target = str(target_file).replace("\\", "/").strip().lstrip("./")
    target_spans: list[tuple[int, int]] = []
    for item in context_window:
        if not isinstance(item, dict):
            continue
        file_path = str(item.get("file") or "").replace("\\", "/").strip().lstrip("./")
        span = item.get("span")
        if file_path != norm_target:
            continue
        if not isinstance(span, (list, tuple)) or len(span) < 2:
            continue
        start, end = int(span[0]), int(span[1])
        if start > 0 and end >= start:
            target_spans.append((start, end))
    if not target_spans:
        return None
    if 0 <= index < len(target_spans):
        return target_spans[index]
    return target_spans[0]


def resolve_site_span(
    project_root: Path,
    target_file: str,
    site: dict[str, Any],
    *,
    focus_symbols: list[str] | None = None,
    context_window: list[dict[str, Any]] | None = None,
    site_index: int = 0,
) -> tuple[int, int]:
    """Pick a concrete on-disk span for one REPLACE site."""
    span = site.get("span")
    if isinstance(span, tuple) and len(span) == 2:
        return int(span[0]), int(span[1])

    symbol = str(site.get("symbol") or "").strip()
    symbols = [str(s).strip() for s in (focus_symbols or []) if str(s).strip()]
    if not symbol and site_index < len(symbols):
        symbol = symbols[site_index]
    if not symbol and len(symbols) == 1:
        symbol = symbols[0]
    if symbol:
        aliased = resolve_symbol_alias(
            project_root,
            target_file,
            symbol,
            focus_symbols=symbols,
        )
        if aliased is not None and aliased[0] == symbol:
            return aliased[1]
        # Do not silently remap a hallucinated name onto another symbol — REPLACE
        # was written for the wrong target. Surface close candidates for Edit retry.
        suggestions: list[str] = []
        if aliased is not None and aliased[0] != symbol:
            suggestions.append(aliased[0])
        disk_names = list_file_symbol_names(project_root, target_file)
        for name in disk_names:
            if name not in suggestions and name != symbol:
                suggestions.append(name)
            if len(suggestions) >= 8:
                break
        # Only echo focus symbols that exist on disk — hallucinated Core names
        # in the error text teach EditLLM to retry the same bad SITE.
        valid_focus = [s for s in symbols if s in set(disk_names)]
        focus_hint = ", ".join(valid_focus[:8]) if valid_focus else "(none on disk)"
        suggest_hint = ", ".join(suggestions[:6]) if suggestions else "(none)"
        raise MaterializeError(
            f"E2_LOCATE: SITE symbol={symbol!r} not found in {target_file}. "
            f"Did you mean one of: [{suggest_hint}]? "
            f"valid_focus_symbols=[{focus_hint}]. "
            "Use an on-disk symbol name only (ignore invented focus names)."
        )

    fallback = _span_from_context_window(target_file, context_window, site_index)
    if fallback is not None:
        return fallback

    raise MaterializeError(
        "E2_LOCATE: REPLACE site has no symbol= / span= and Core did not supply "
        "focus_symbols or a target context_window span. Add "
        "`SITE: symbol=<name>` (preferred) or `SITE: span=<start>-<end>`."
    )


def read_file_span(
    project_root: Path,
    target_file: str,
    start_line: int,
    end_line: int,
) -> str:
    path = (project_root / target_file).resolve()
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MaterializeError(f"E2_LOCATE: cannot read {target_file}: {exc}") from exc
    if start_line < 1 or end_line > len(lines) or end_line < start_line:
        raise MaterializeError(
            f"E2_LOCATE: span {start_line}-{end_line} out of range for "
            f"{target_file} ({len(lines)} lines)."
        )
    return "\n".join(lines[start_line - 1 : end_line])


def materialize_edit_patch(
    project_root: Path,
    target_file: str,
    patch: str,
    *,
    focus_symbols: list[str] | None = None,
    context_window: list[dict[str, Any]] | None = None,
) -> str:
    """Return a classic SEARCH/REPLACE patch ready for ``CursorPatchApplier``.

    Legacy patches that already contain ``<<<<<<< SEARCH`` are returned unchanged.
    New REPLACE-only / SITE patches are expanded by reading the located span
    from disk into SEARCH.
    """
    raw = (patch or "").strip()
    if not raw:
        raise MaterializeError("E1_FORMAT: empty patch")
    if is_legacy_search_replace(raw):
        return raw

    sites = parse_replace_sites(raw)
    if not sites:
        raise MaterializeError(
            "E1_FORMAT: expected SITE: + <<<<<<< REPLACE … >>>>>>> REPLACE "
            "(or legacy SEARCH/REPLACE). Do not emit SEARCH — harness fills it."
        )

    blocks: list[str] = []
    skipped_noops: list[str] = []
    for index, site in enumerate(sites):
        start, end = resolve_site_span(
            project_root,
            target_file,
            site,
            focus_symbols=focus_symbols,
            context_window=context_window,
            site_index=index,
        )
        old_code = read_file_span(project_root, target_file, start, end)
        new_code = normalize_replace_body(str(site.get("body") or ""))
        mode = str(site.get("mode") or "").casefold().strip()
        anchor = site.get("anchor")

        # Delta modes: only emit the new/changed snippet; harness merges at ANCHOR.
        if mode in _DELTA_MODES or (not mode and anchor):
            delta_mode = mode if mode in _DELTA_MODES else "insert_after"
            try:
                search_code, replace_code = apply_delta_in_span(
                    old_code,
                    mode=delta_mode,
                    anchor=str(anchor) if anchor is not None else None,
                    new_body=new_code,
                )
            except MaterializeError:
                raise
            blocks.append(
                "<<<<<<< SEARCH\n"
                f"{search_code}\n"
                "=======\n"
                f"{replace_code}\n"
                ">>>>>>> REPLACE"
            )
            continue

        if _fingerprint(new_code) == _fingerprint(old_code):
            # Multi-SITE: skip unchanged sites; only fail when nothing remains.
            label = str(site.get("symbol") or f"span={start}-{end}")
            skipped_noops.append(f"SITE {index + 1} ({label})")
            continue

        search_code = old_code
        replace_code = new_code
        if len(new_code.splitlines()) < len(old_code.splitlines()):
            surgical = locate_surgical_hunk(old_code, new_code)
            if surgical is not None:
                search_code, replace_code = surgical

        blocks.append(
            "<<<<<<< SEARCH\n"
            f"{search_code}\n"
            "=======\n"
            f"{replace_code}\n"
            ">>>>>>> REPLACE"
        )

    if not blocks:
        excerpt = ""
        if skipped_noops:
            excerpt = (
                " All SITE blocks echoed on-disk text (no real change): "
                + ", ".join(skipped_noops)
                + "."
            )
        raise MaterializeError(
            "E1_FORMAT: REPLACE equals on-disk for every SITE; produce a real change."
            f"{excerpt} REPLACE must be AFTER-edit text with CURRENT_STATE intent "
            "applied — not a copy of CURRENT_CONTEXT."
        )
    return "\n".join(blocks)
