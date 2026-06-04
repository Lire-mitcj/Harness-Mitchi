from __future__ import annotations

from dataclasses import dataclass

from src.harness.edit.errors import AnchorError
from src.harness.edit.extract import anchor_hash, slice_file_lines


@dataclass(frozen=True)
class SpliceResult:
    success: bool
    new_content: str | None = None
    error: str | None = None
    applied_start: int | None = None
    applied_end: int | None = None


def verify_anchor(
    content: str,
    *,
    start_line: int,
    end_line: int,
    expected_hash: str,
) -> str:
    """Return the on-disk slice; raise AnchorError when hash mismatches."""
    current_slice = slice_file_lines(content, start_line, end_line)
    if not current_slice and expected_hash:
        raise AnchorError(
            "anchor_empty",
            f"Anchor slice L{start_line}-{end_line} is empty but hash was expected.",
        )
    current_hash = anchor_hash(current_slice)
    if current_hash != expected_hash:
        raise AnchorError(
            "anchor_drift",
            (
                f"Anchor hash mismatch at L{start_line}-{end_line} "
                f"(expected {expected_hash}, got {current_hash}). "
                "File content may have drifted since diagnose."
            ),
        )
    return current_slice


def apply_splice(
    file_content: str,
    *,
    start_line: int,
    end_line: int,
    new_body: str,
    expected_hash: str | None = None,
) -> SpliceResult:
    """Replace [start_line, end_line] with new_body; optional anchor hash check."""
    if expected_hash is not None:
        try:
            verify_anchor(
                file_content,
                start_line=start_line,
                end_line=end_line,
                expected_hash=expected_hash,
            )
        except AnchorError as exc:
            return SpliceResult(success=False, error=str(exc))

    if not new_body.strip():
        return SpliceResult(success=False, error="new_body is empty.")

    lines = file_content.splitlines(keepends=True)
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)

    old_slice = slice_file_lines(file_content, start_line, end_line)
    normalized_new = new_body if new_body.endswith("\n") else new_body + "\n"
    if old_slice == normalized_new or old_slice.rstrip("\n") == new_body.rstrip("\n"):
        return SpliceResult(
            success=False,
            error=(
                "new_body is identical to the current target span — "
                "provide the modified function/block."
            ),
        )

    new_lines = (
        lines[:start_idx]
        + normalized_new.splitlines(keepends=True)
        + lines[end_idx:]
    )
    return SpliceResult(
        success=True,
        new_content="".join(new_lines),
        applied_start=start_line,
        applied_end=end_line,
    )
