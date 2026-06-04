from __future__ import annotations

import logging
from pathlib import Path

from src.agent.types import ToolResult
from src.harness.edit.resolve import refresh_target_span
from src.harness.edit.splice import SpliceResult, apply_splice
from src.harness.edit.target import EditTarget

log = logging.getLogger(__name__)

MAX_SPLICE_ATTEMPTS = 2


def execute_replace_symbol(
    *,
    project_root: Path,
    targets: list[EditTarget],
    path: str,
    symbol: str,
    new_body: str,
    max_attempts: int = MAX_SPLICE_ATTEMPTS,
) -> ToolResult:
    """Splice new_body into a resolved EditTarget with anchor retry on drift."""
    rel = path.replace("\\", "/").lstrip("./")
    matched = _match_target(targets, rel=rel, symbol=symbol)
    if matched is None:
        available = ", ".join(f"{t.path}:{t.symbol}" for t in targets[:4]) or "(none)"
        return ToolResult(
            success=False,
            output="",
            error=(
                f"No EditTarget for path={rel!r} symbol={symbol!r}. "
                f"Available targets: {available}"
            ),
        )

    file_path = (project_root / rel).resolve()
    if not file_path.is_file():
        return ToolResult(success=False, output="", error=f"File not found: {file_path}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return ToolResult(success=False, output="", error=f"IO error reading {file_path}: {exc}")

    current = matched
    last_error = ""
    attempts_used = 0
    for attempt in range(1, max_attempts + 1):
        attempts_used = attempt
        result = apply_splice(
            content,
            start_line=current.start_line,
            end_line=current.end_line,
            new_body=new_body,
            expected_hash=matched.anchor_hash,
        )
        if result.success and result.new_content is not None:
            try:
                file_path.write_text(result.new_content, encoding="utf-8")
            except OSError as exc:
                return ToolResult(
                    success=False,
                    output="",
                    error=f"IO error writing {file_path}: {exc}",
                )
            old_lines = current.end_line - current.start_line + 1
            new_lines = new_body.count("\n") + (0 if new_body.endswith("\n") else 1)
            retry_note = (
                f" (anchor re-resolved on attempt {attempt})"
                if attempt > 1
                else ""
            )
            return ToolResult(
                success=True,
                output=(
                    f"Spliced {rel} [{current.symbol}] L{current.start_line}-"
                    f"{current.end_line}: replaced ~{old_lines} lines with ~{new_lines} lines"
                    f"{retry_note}."
                ),
                metadata={
                    "path": str(file_path),
                    "symbol": current.symbol,
                    "start_line": current.start_line,
                    "end_line": current.end_line,
                },
            )

        last_error = result.error or "splice failed"
        if attempt >= max_attempts:
            break
        if "hash mismatch" not in last_error.lower() and "anchor" not in last_error.lower():
            break

        refreshed = refresh_target_span(project_root, matched)
        if refreshed is None:
            last_error = (
                f"{last_error} Retried after anchor failure but could not re-resolve "
                f"symbol {symbol!r} in {rel}."
            )
            break

        log.info(
            "replace_symbol anchor retry: %s:%s L%d-%d → L%d-%d (attempt %d)",
            rel,
            symbol,
            current.start_line,
            current.end_line,
            refreshed.start_line,
            refreshed.end_line,
            attempt + 1,
        )
        current = refreshed
        try:
            content = file_path.read_text(encoding="utf-8")
        except OSError as exc:
            return ToolResult(success=False, output="", error=f"IO error re-reading: {exc}")

    return ToolResult(
        success=False,
        output="",
        error=(
            f"replace_symbol failed after {attempts_used} attempt(s): "
            f"{last_error}"
        ),
    )


def _match_target(
    targets: list[EditTarget],
    *,
    rel: str,
    symbol: str,
) -> EditTarget | None:
    for target in targets:
        if target.path == rel and target.symbol == symbol:
            return target
    for target in targets:
        if target.path == rel and len(targets) == 1:
            return target
    return None
