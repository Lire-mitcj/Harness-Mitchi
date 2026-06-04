from __future__ import annotations

import os
from pathlib import Path

PROTECTED_PATTERNS: list[str] = [
    ".env",
    ".env.local",
    ".env.production",
    "credentials.json",
    "service-account.json",
    "secrets.yaml",
    "secrets.yml",
    "id_rsa",
    "id_ed25519",
    ".ssh/",
    ".gnupg/",
    ".aws/credentials",
    ".npmrc",
    ".pypirc",
    "token.json",
]

NEVER_WRITE_DIRS: list[str] = [
    "/etc",
    "/usr",
    "/bin",
    "/sbin",
    "/boot",
    "/proc",
    "/sys",
    "/var/log",
]


class FileGuard:
    """Prevents writes outside the project root and to protected files."""

    def __init__(
        self,
        project_root: str | None = None,
        extra_protected: list[str] | None = None,
    ) -> None:
        self._root = Path(project_root).resolve() if project_root else Path.cwd().resolve()
        self._extra = extra_protected or []

    def is_safe_path(self, path: str) -> bool:
        """Return True if *path* is inside the project root and not in a
        system directory."""
        resolved = Path(path).resolve()

        try:
            resolved.relative_to(self._root)
        except ValueError:
            return False

        resolved_str = str(resolved)
        return not any(resolved_str.startswith(d) for d in NEVER_WRITE_DIRS)

    def is_protected(self, path: str) -> bool:
        """Return True if *path* matches a protected-file pattern."""
        resolved = Path(path).resolve()
        name = resolved.name
        rel = str(resolved)

        for pattern in (*PROTECTED_PATTERNS, *self._extra):
            if pattern.endswith("/"):
                if f"/{pattern}" in rel or rel.startswith(pattern):
                    return True
            elif name == pattern or rel.endswith(f"/{pattern}"):
                return True
        return False

    def validate_write(self, path: str) -> None:
        """Raise if writing to *path* should be blocked."""
        if not self.is_safe_path(path):
            raise PermissionError(
                f"Write blocked: {path} is outside the project root ({self._root})"
            )
        if self.is_protected(path):
            raise PermissionError(
                f"Write blocked: {path} matches a protected file pattern"
            )
