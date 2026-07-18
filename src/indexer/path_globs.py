"""Path glob helpers for repo_map indexing scope."""

from __future__ import annotations

import fnmatch
import re
from pathlib import Path


def path_matches_glob(path: str, pattern: str) -> bool:
    """Return True when a repo-relative path matches a glob pattern."""
    norm = path.replace("\\", "/").lstrip("./")
    name = Path(norm).name
    pat = pattern.replace("\\", "/").lstrip("./")
    if not pat:
        return False
    if "**" in pat:
        if pat.startswith("**/"):
            suffix = pat[3:]
            if fnmatch.fnmatch(norm, suffix) or fnmatch.fnmatch(name, suffix):
                return True
            return f"/{suffix}" in f"/{norm}/" or norm.endswith(f"/{suffix}")
        if pat.endswith("/**"):
            prefix = pat[:-3]
            return norm == prefix or norm.startswith(f"{prefix}/")
        regex = re.escape(pat).replace(r"\*\*", ".*")
        return re.fullmatch(regex, norm) is not None
    return fnmatch.fnmatch(norm, pat) or fnmatch.fnmatch(name, pat)


def path_matches_any(path: str, patterns: tuple[str, ...] | list[str]) -> bool:
    if not patterns:
        return False
    return any(path_matches_glob(path, pattern) for pattern in patterns)


def repo_map_path_allowed(
    path: str,
    *,
    include_globs: tuple[str, ...] | list[str] = (),
    exclude_globs: tuple[str, ...] | list[str] = (),
) -> bool:
    """True when a file path is inside the repo_map indexing scope."""
    if path_matches_any(path, exclude_globs):
        return False
    if include_globs:
        return path_matches_any(path, include_globs)
    return True


def ctags_exclude_args(patterns: tuple[str, ...] | list[str]) -> list[str]:
    """Map user exclude globs to universal-ctags --exclude flags."""
    args: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        pat = pattern.replace("\\", "/").lstrip("./")
        if not pat or "**" in pat:
            continue
        if pat not in seen:
            seen.add(pat)
            args.append(f"--exclude={pat}")
    return args
