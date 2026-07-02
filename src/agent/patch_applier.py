from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


class CursorPatchApplier:
    """Deterministic, atomic SEARCH/REPLACE patch application."""

    _BLOCK = re.compile(
        r"<<<<<<< SEARCH[ \t]*\n(?:(.*?)\n)?=======[ \t]*\n(.*?)\n>>>>>>> REPLACE"
        r"[ \t]*(?:\n|$)",
        re.DOTALL,
    )

    def __init__(self, project_root: Path) -> None:
        self.project_root = project_root.resolve()

    def parse_blocks(self, patch: str) -> tuple[tuple[str, str], ...]:
        normalized = patch.replace("\r\n", "\n")
        blocks = tuple(
            (match.group(1) or "", match.group(2) or "")
            for match in self._BLOCK.finditer(normalized)
        )
        if not blocks or self._BLOCK.sub("", normalized).strip():
            return ()
        return blocks

    def apply_patch(self, file_rel: str, patch: str) -> tuple[bool, str]:
        path, error = self._safe_target(file_rel)
        if path is None:
            return False, error
        blocks = self.parse_blocks(patch)
        if not blocks:
            return False, "invalid_patch: expected SEARCH/REPLACE blocks only"
        try:
            if not path.exists():
                original = ""
            else:
                original = path.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"io: unable to read target: {exc}"

        trailing_newline = original.endswith("\n") if original else True
        lines = original.splitlines()
        planned: list[tuple[int, int, list[str]]] = []
        occupied: list[tuple[int, int]] = []
        for index, (old_code, new_code) in enumerate(blocks, start=1):
            start = self._find_match(lines, old_code)
            if start < 0:
                return False, f"mismatch: block {index} SEARCH code not found"
            old_len = len(old_code.splitlines())
            end = start + old_len
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                return False, f"invalid_patch: block {index} overlaps another block"
            occupied.append((start, end))
            planned.append((start, end, new_code.splitlines()))

        for start, end, replacement in sorted(planned, reverse=True):
            lines[start:end] = replacement
        updated = "\n".join(lines)
        if trailing_newline and updated:
            updated += "\n"
        if updated == original:
            return False, "invalid_patch: patch produces no change"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._atomic_write(path, updated)
        except OSError as exc:
            return False, f"io: unable to write target: {exc}"
        return True, ""

    def _safe_target(self, file_rel: str) -> tuple[Path | None, str]:
        try:
            path = (self.project_root / file_rel).resolve()
            path.relative_to(self.project_root)
        except (OSError, ValueError):
            return None, "scope: target is outside project root"
        if path.exists() and not path.is_file():
            return None, "scope: target exists but is not a file"
        return path, ""

    @staticmethod
    def _find_match(lines: list[str], old_code: str) -> int:
        old_lines = old_code.splitlines()
        if not old_lines:
            if not lines:
                return 0
            return -1
        width = len(old_lines)
        for start in range(len(lines) - width + 1):
            if lines[start : start + width] == old_lines:
                return start
        normalized_old = [CursorPatchApplier._normalize_line(line) for line in old_lines]
        for start in range(len(lines) - width + 1):
            candidate = [
                CursorPatchApplier._normalize_line(line)
                for line in lines[start : start + width]
            ]
            if candidate == normalized_old:
                return start
        return -1

    @staticmethod
    def _normalize_line(line: str) -> str:
        return "".join(line.split())

    @staticmethod
    def _atomic_write(path: Path, content: str) -> None:
        try:
            mode = path.stat().st_mode
        except OSError:
            mode = 0o644
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            text=True,
        )
        try:
            os.fchmod(descriptor, mode)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except BaseException:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
            raise
