from __future__ import annotations

from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = _SRC_ROOT.parent


def load_internal_prompt(
    name: str,
    *,
    fallback: str,
    override: Path | None = None,
) -> str:
    """Load a Harness-owned prompt without consulting the target repository."""
    candidates = [
        _SRC_ROOT / "_prompts" / name,
        _REPO_ROOT / "prompts" / name,
    ]
    if override is not None:
        candidates.append(override)
    for path in candidates:
        if path.is_file():
            try:
                return path.read_text(encoding="utf-8").strip()
            except OSError:
                continue
    return fallback
