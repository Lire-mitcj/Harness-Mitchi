from __future__ import annotations

import hashlib


def anchor_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def slice_file_lines(content: str, start_line: int, end_line: int) -> str:
    """Extract 1-based inclusive line range, preserving line endings."""
    lines = content.splitlines(keepends=True)
    if not lines and start_line <= 1:
        return ""
    start_idx = max(0, start_line - 1)
    end_idx = min(len(lines), end_line)
    if start_idx >= len(lines):
        return ""
    return "".join(lines[start_idx:end_idx])
