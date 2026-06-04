from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.harness.gates.types import TruncationPolicy
from src.harness.subtask.preload import load_context_file_contents


def format_cached_read_from_policy(
    project_root: Path,
    rel: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
    policy: TruncationPolicy,
) -> str | None:
    """Return preloaded/sliced file text when a duplicate read should reuse context."""
    norm = rel.replace("\\", "/").lstrip("./")
    slice_policy = policy
    if start_line is not None and end_line is not None:
        slices = dict(policy.line_slices or {})
        slices[norm] = (start_line, end_line)
        slice_policy = replace(policy, line_slices=slices)
    loaded = load_context_file_contents(project_root, [norm], policy=slice_policy)
    for path, content in loaded:
        if not content or content.startswith(("[missing file:", "[read error:", "[blocked:")):
            continue
        return (
            "[Already in context — served from preload; do not read again. "
            "Use edit_file with a unique multi-line old_string.]\n"
            f"===== {path} =====\n{content}"
        )
    return None
