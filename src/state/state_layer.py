from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)


class StateLayer:
    """Computes session-level system context, including git status and user context (rules/contracts).
    Cached for reuse.
    """

    def __init__(self, project_root: Path, context_assembly: Any) -> None:
        self.project_root = project_root
        self.context_assembly = context_assembly
        self._cached_git_diff: str | None = None
        self._cached_user_context: str | None = None
        self._last_cache_key: tuple[tuple[str, ...], int] | None = None

    def get_git_status(self, git_diff: str) -> str:
        """Returns the git diff/status, using cached value if unchanged."""
        if self._cached_git_diff is not None and self._cached_git_diff == git_diff:
            return self._cached_git_diff
        self._cached_git_diff = git_diff
        return git_diff

    def get_user_context(
        self,
        active_files: list[str],
        search_cache: dict[str, Any] | None = None,
    ) -> str:
        """Computes and caches the user context (rules, contracts, facts)."""
        # Create a stable hash for search_cache to check for changes
        cache_hash = hash(str(search_cache)) if search_cache else 0
        cache_key = (tuple(sorted(active_files)), cache_hash)

        if self._last_cache_key == cache_key and self._cached_user_context is not None:
            return self._cached_user_context

        user_context = self.context_assembly.get_user_context(active_files, search_cache)
        self._last_cache_key = cache_key
        self._cached_user_context = user_context
        return user_context

    def clear_cache(self) -> None:
        """Clears the cached system/user contexts."""
        self._cached_git_diff = None
        self._cached_user_context = None
        self._last_cache_key = None
