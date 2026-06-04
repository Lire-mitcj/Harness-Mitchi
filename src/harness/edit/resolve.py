from __future__ import annotations

from pathlib import Path

from src.harness.edit.extract import anchor_hash, slice_file_lines
from src.harness.edit.target import EditTarget
from src.indexer.parser import CodeParser, Symbol
from src.planner.context_policy import _norm_path
from src.planner.prior_context import (
    extract_line_refs_from_summaries,
    extract_symbol_hits_from_text,
    rank_edit_relevant_symbol_hits,
)


def resolve_edit_targets(
    *,
    project_root: Path,
    prior_summaries: dict[str, str],
    whitelist_files: list[str],
    intent_text: str = "",
) -> list[EditTarget]:
    """Build EditTargets from diagnose summaries (symbol hits, then line slices)."""
    root = project_root.resolve()
    parser = CodeParser()
    targets: list[EditTarget] = []
    seen: set[tuple[str, str]] = set()

    for _sid, text in prior_summaries.items():
        hits = rank_edit_relevant_symbol_hits(
            text,
            extract_symbol_hits_from_text(text, root),
            intent_text=intent_text,
        )
        for rel, hint_line, symbol in hits:
            key = (rel, symbol)
            if key in seen:
                continue
            target = _target_from_symbol(
                parser,
                root=root,
                rel=rel,
                symbol=symbol,
                hint_line=hint_line,
            )
            if target is not None:
                seen.add(key)
                targets.append(target)

    if targets:
        return targets[:5]

    slices = extract_line_refs_from_summaries(prior_summaries, root)
    allowed = {_norm_path(p) or p for p in whitelist_files}
    for rel, (start, end) in slices.items():
        if allowed and rel not in allowed:
            continue
        key = (rel, f"lines_{start}_{end}")
        if key in seen:
            continue
        target = _target_from_span(root, rel=rel, start=start, end=end)
        if target is not None:
            seen.add(key)
            targets.append(target)

    return targets[:5]


def refresh_target_span(
    project_root: Path,
    target: EditTarget,
) -> EditTarget | None:
    """Re-resolve symbol boundaries on disk (handles line drift)."""
    root = project_root.resolve()
    parser = CodeParser()
    if target.symbol and not target.symbol.startswith("lines_"):
        sym = _find_symbol(
            parser,
            root=root,
            rel=target.path,
            symbol=target.symbol,
            hint_line=target.start_line,
        )
        if sym is None:
            return None
        return _build_target(root, rel=target.path, sym=sym, symbol_name=target.symbol)
    return _target_from_span(
        root,
        rel=target.path,
        start=target.start_line,
        end=target.end_line,
    )


def _target_from_symbol(
    parser: CodeParser,
    *,
    root: Path,
    rel: str,
    symbol: str,
    hint_line: int,
) -> EditTarget | None:
    sym = _find_symbol(parser, root=root, rel=rel, symbol=symbol, hint_line=hint_line)
    if sym is None:
        return None
    return _build_target(root, rel=rel, sym=sym, symbol_name=symbol)


def _find_symbol(
    parser: CodeParser,
    *,
    root: Path,
    rel: str,
    symbol: str,
    hint_line: int | None,
) -> Symbol | None:
    path = (root / rel).resolve()
    if not path.is_file():
        return None
    parsed = parser.parse_file(path)
    candidates = [s for s in parsed.all_symbols if s.name == symbol]
    if not candidates:
        return None
    if hint_line is not None and len(candidates) > 1:
        return min(candidates, key=lambda s: abs(s.start_line - hint_line))
    return candidates[0]


def _build_target(
    root: Path,
    *,
    rel: str,
    sym: Symbol,
    symbol_name: str,
) -> EditTarget | None:
    path = (root / rel).resolve()
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    original = slice_file_lines(content, sym.start_line, sym.end_line)
    if not original.strip():
        return None
    return EditTarget(
        path=rel.replace("\\", "/"),
        symbol=symbol_name,
        kind=sym.kind,
        start_line=sym.start_line,
        end_line=sym.end_line,
        original_source=original,
        anchor_hash=anchor_hash(original),
    )


def _target_from_span(
    root: Path,
    *,
    rel: str,
    start: int,
    end: int,
) -> EditTarget | None:
    path = (root / rel).resolve()
    if not path.is_file():
        return None
    try:
        content = path.read_text(encoding="utf-8")
    except OSError:
        return None
    original = slice_file_lines(content, start, end)
    if not original.strip():
        return None
    return EditTarget(
        path=rel.replace("\\", "/"),
        symbol=f"lines_{start}_{end}",
        kind="slice",
        start_line=start,
        end_line=end,
        original_source=original,
        anchor_hash=anchor_hash(original),
    )
