from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

DEFAULT_MAX_PATCH_BLOCKS = 3

# Markers that must never appear inside SEARCH/REPLACE bodies.
_EMBEDDED_MARKER = re.compile(
    r"(?:^|\n)(?:<<<<<<< SEARCH|=======|>>>>>>> REPLACE)(?:[ \t]*$|[ \t]*<<<<<<<)",
    re.MULTILINE,
)
# Model sometimes glues block boundaries: ">>>>>>> REPLACE<<<<<<< SEARCH"
_GLUED_BLOCK_BOUNDARY = re.compile(
    r"(>>>>>>> REPLACE)[ \t]*(<<<<<<< SEARCH)",
)


class CursorPatchApplier:
    """Deterministic, atomic SEARCH/REPLACE patch application."""

    _BLOCK = re.compile(
        r"<<<<<<< SEARCH[ \t]*\n(?:(.*?)\n)?=======[ \t]*\n(.*?)\n>>>>>>> REPLACE"
        r"[ \t]*(?:\n|$)",
        re.DOTALL,
    )

    def __init__(
        self,
        project_root: Path,
        *,
        sequential: bool = False,
        max_blocks: int = DEFAULT_MAX_PATCH_BLOCKS,
    ) -> None:
        self.project_root = project_root.resolve()
        self.sequential = sequential
        self.max_blocks = max(1, int(max_blocks))

    @staticmethod
    def normalize_patch_text(patch: str) -> str:
        """Normalize newlines and unstick glued SEARCH/REPLACE boundaries."""
        normalized = patch.replace("\r\n", "\n")
        return _GLUED_BLOCK_BOUNDARY.sub(r"\1\n\2", normalized)

    def parse_blocks(self, patch: str) -> tuple[tuple[str, str], ...]:
        normalized = self.normalize_patch_text(patch)
        blocks = tuple(
            (match.group(1) or "", match.group(2) or "")
            for match in self._BLOCK.finditer(normalized)
        )
        if not blocks or self._BLOCK.sub("", normalized).strip():
            return ()
        return blocks

    @staticmethod
    def validate_block_bodies(blocks: tuple[tuple[str, str], ...]) -> str:
        """Reject SEARCH/REPLACE bodies that still contain conflict markers."""
        for index, (old_code, new_code) in enumerate(blocks, start=1):
            for label, body in (("SEARCH", old_code), ("REPLACE", new_code)):
                if "<<<<<<<" in body or ">>>>>>>" in body or _EMBEDDED_MARKER.search(
                    f"\n{body}\n"
                ):
                    return (
                        f"invalid_patch: block {index} {label} contains nested "
                        "SEARCH/REPLACE markers. Separate blocks with a newline "
                        "after >>>>>>> REPLACE; never embed markers in code bodies."
                    )
                # A lone ======= line inside body is almost always a broken split.
                if re.search(r"(?:^|\n)=======[ \t]*(?:\n|$)", body):
                    return (
                        f"invalid_patch: block {index} {label} contains a stray "
                        "======= marker. Emit clean SEARCH/REPLACE blocks only."
                    )
        return ""

    def apply_patch(self, file_rel: str, patch: str) -> tuple[bool, str]:
        path, error = self._safe_target(file_rel)
        if path is None:
            return False, error
        blocks = self.parse_blocks(patch)
        if not blocks:
            return False, "invalid_patch: expected SEARCH/REPLACE blocks only"
        body_error = self.validate_block_bodies(blocks)
        if body_error:
            return False, body_error
        if len(blocks) > self.max_blocks:
            return False, (
                f"invalid_patch: too many SEARCH/REPLACE blocks "
                f"({len(blocks)} > {self.max_blocks}). "
                f"Keep at most {self.max_blocks} blocks per decision_edit; "
                "prefer one block when possible. Split remaining sites into "
                "later Core decision_edit calls (top-to-bottom)."
            )
        try:
            if not path.exists():
                original = ""
            else:
                original = path.read_text(encoding="utf-8")
        except OSError as exc:
            return False, f"io: unable to read target: {exc}"

        trailing_newline = original.endswith("\n") if original else True
        if self.sequential:
            lines, apply_error = self._plan_sequential(original.splitlines(), blocks)
        else:
            lines, apply_error = self._plan_reverse(original.splitlines(), blocks)
        if apply_error:
            return False, apply_error

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

    def _plan_reverse(
        self,
        lines: list[str],
        blocks: tuple[tuple[str, str], ...],
    ) -> tuple[list[str] | None, str]:
        """Match all SEARCH against the original file, then apply bottom-up."""
        planned: list[tuple[int, int, list[str]]] = []
        occupied: list[tuple[int, int]] = []
        working = list(lines)
        for index, (old_code, new_code) in enumerate(blocks, start=1):
            start = self._find_match(working, old_code)
            if start < 0:
                diagnostic = self._mismatch_diagnostic(working, old_code, index)
                return None, f"mismatch: block {index} SEARCH code not found\n{diagnostic}"
            old_len = len(old_code.splitlines())
            end = start + old_len
            if any(start < used_end and end > used_start for used_start, used_end in occupied):
                return None, f"invalid_patch: block {index} overlaps another block"
            occupied.append((start, end))
            planned.append((start, end, new_code.splitlines()))

        for start, end, replacement in sorted(planned, reverse=True):
            working[start:end] = replacement
        return working, ""

    def _plan_sequential(
        self,
        lines: list[str],
        blocks: tuple[tuple[str, str], ...],
    ) -> tuple[list[str] | None, str]:
        """Apply blocks top-to-bottom against the evolving buffer (one write later)."""
        working = list(lines)
        for index, (old_code, new_code) in enumerate(blocks, start=1):
            start = self._find_match(working, old_code)
            if start < 0:
                diagnostic = self._mismatch_diagnostic(working, old_code, index)
                return None, (
                    f"mismatch: block {index} SEARCH code not found\n{diagnostic}"
                )
            old_len = len(old_code.splitlines())
            working[start : start + old_len] = new_code.splitlines()
        return working, ""

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
        # Last resort: fold literal \uXXXX escapes and their decoded glyphs to a
        # common form. The Edit model single-escapes backslashes over the JSON
        # transport, so on-disk source like r"[\u4e00-\u9fa5]" arrives in SEARCH
        # as the decoded glyphs r"[一-龥]"; without this the block never locates
        # even though it is byte-equivalent source. Only widens matches, so it is
        # safe as a final pass after exact/whitespace comparison already failed.
        folded_old = [CursorPatchApplier._escape_fold(line) for line in old_lines]
        for start in range(len(lines) - width + 1):
            candidate = [
                CursorPatchApplier._escape_fold(line)
                for line in lines[start : start + width]
            ]
            if candidate == folded_old:
                return start
        return -1

    @staticmethod
    def _phantom_search_lines(lines: list[str], old_lines: list[str]) -> list[str]:
        """SEARCH lines that do not exist anywhere in the file (normalized).

        These are almost always additions the model put into SEARCH instead of
        REPLACE (target-state SEARCH).
        """
        if not old_lines:
            return []
        disk_norms = {
            CursorPatchApplier._normalize_line(line) for line in lines
        }
        phantoms: list[str] = []
        seen: set[str] = set()
        for line in old_lines:
            norm = CursorPatchApplier._normalize_line(line)
            if not norm or norm in disk_norms or norm in seen:
                continue
            seen.add(norm)
            phantoms.append(line)
        return phantoms

    @staticmethod
    def _correct_search_candidate(
        lines: list[str],
        old_lines: list[str],
        *,
        phantoms: list[str],
    ) -> list[str]:
        """Paste-ready on-disk SEARCH built from non-phantom SEARCH lines.

        Drops phantom (target-state) lines, then finds the longest contiguous
        on-disk window that matches the remaining SEARCH lines in order. Falls
        back to the closest-match window when no contiguous non-phantom match
        exists.
        """
        if not lines or not old_lines:
            return []
        phantom_norms = {
            CursorPatchApplier._normalize_line(line) for line in phantoms
        }
        real_search = [
            line
            for line in old_lines
            if CursorPatchApplier._normalize_line(line) not in phantom_norms
            and CursorPatchApplier._normalize_line(line)
        ]
        if real_search:
            start = CursorPatchApplier._find_match(lines, "\n".join(real_search))
            if start >= 0:
                return lines[start : start + len(real_search)]
            # Non-contiguous after dropping phantoms: take the longest prefix of
            # real_search that still matches as a contiguous window.
            for width in range(len(real_search), 0, -1):
                start = CursorPatchApplier._find_match(
                    lines, "\n".join(real_search[:width])
                )
                if start >= 0:
                    return lines[start : start + width]
        best_start, _ = CursorPatchApplier._best_match_window(lines, old_lines)
        width = min(len(old_lines), max(1, len(lines) - best_start))
        return lines[best_start : best_start + width]

    @staticmethod
    def _mismatch_diagnostic(
        lines: list[str],
        old_code: str,
        block_index: int,
    ) -> str:
        """Human-readable hint when SEARCH text is absent from the target file."""
        old_lines = old_code.splitlines()
        preview_count = min(3, len(old_lines))
        search_preview = "\n".join(
            f"  | {line}" for line in old_lines[:preview_count]
        )
        if len(old_lines) > preview_count:
            search_preview += f"\n  | ... ({len(old_lines)} lines total in SEARCH)"

        if not old_lines:
            return (
                f"Diagnostic for block {block_index}:\n"
                "SEARCH block is empty but the target file is not."
            )

        phantoms = CursorPatchApplier._phantom_search_lines(lines, old_lines)
        best_start, best_score = CursorPatchApplier._best_match_window(
            lines, old_lines
        )
        width = len(old_lines)
        fragment = lines[best_start : best_start + width]
        if not fragment and lines:
            fragment = [lines[best_start]] if best_start < len(lines) else []

        # Show the FULL closest window verbatim with a per-line diff marker.
        # Previously only the first 3 lines were shown, which hid the exact line
        # that diverged — so the model kept regenerating the same near-miss
        # SEARCH and the inner retry was pointless. Bounded to keep prompt small.
        max_show = 60
        show = min(max(width, len(fragment)), max_show)
        aligned: list[str] = []
        for offset in range(show):
            disk_line = fragment[offset] if offset < len(fragment) else None
            search_line = old_lines[offset] if offset < len(old_lines) else None
            if disk_line is None:
                aligned.append(
                    f"  ✗ (no on-disk line here) ↳ your SEARCH had: {search_line}"
                )
                continue
            disk_no = best_start + offset + 1
            if search_line is None:
                aligned.append(f"    {disk_no}: {disk_line}")
                continue
            same = (
                CursorPatchApplier._normalize_line(disk_line)
                == CursorPatchApplier._normalize_line(search_line)
            )
            marker = "  " if same else "✗ "
            aligned.append(f"{marker}{disk_no}: {disk_line}")
            if not same:
                aligned.append(f"       ↳ your SEARCH had: {search_line}")
        file_preview = "\n".join(aligned)
        if width > max_show:
            file_preview += (
                f"\n  ... ({width} SEARCH lines total; showing first {max_show})"
            )

        parts = [
            f"Diagnostic for block {block_index}:",
            f"SEARCH (first lines, must exist verbatim in file):\n{search_preview}",
            (
                f"Closest file fragment (lines {best_start + 1}-"
                f"{best_start + len(fragment)}, score {best_score}/{width}). "
                "Lines marked ✗ differ from your SEARCH — copy the on-disk code below "
                "VERBATIM into SEARCH (change only the REPLACE block):\n"
                f"{file_preview}"
            ),
        ]
        if phantoms:
            phantom_list = "\n".join(f"  - {line}" for line in phantoms[:12])
            if len(phantoms) > 12:
                phantom_list += f"\n  - ... ({len(phantoms)} phantom lines total)"
            parts.append(
                "PHANTOM SEARCH lines (do not exist anywhere in the file — these "
                "are additions; move them to REPLACE only):\n"
                f"{phantom_list}"
            )
        candidate = CursorPatchApplier._correct_search_candidate(
            lines, old_lines, phantoms=phantoms
        )
        if candidate:
            candidate_preview = "\n".join(f"  | {line}" for line in candidate[:60])
            if len(candidate) > 60:
                candidate_preview += (
                    f"\n  | ... ({len(candidate)} lines; showing first 60)"
                )
            parts.append(
                "CORRECT SEARCH candidate (copy verbatim into SEARCH; put new "
                "lines only in REPLACE):\n"
                f"{candidate_preview}"
            )
        parts.append(
            "Hint: SEARCH must be the current on-disk code (before edit). "
            "Put new decorators or other additions only in REPLACE."
        )
        return "\n".join(parts)

    @staticmethod
    def _best_match_window(
        lines: list[str],
        old_lines: list[str],
    ) -> tuple[int, int]:
        """Return (start_line_index, matching_line_count) for the closest window."""
        if not lines:
            return 0, 0

        width = len(old_lines)
        normalized_old = [
            CursorPatchApplier._normalize_line(line) for line in old_lines
        ]
        best_start = 0
        best_score = -1

        if width <= len(lines):
            for start in range(len(lines) - width + 1):
                candidate = [
                    CursorPatchApplier._normalize_line(line)
                    for line in lines[start : start + width]
                ]
                score = sum(
                    left == right for left, right in zip(candidate, normalized_old)
                )
                if score > best_score:
                    best_score = score
                    best_start = start
        else:
            first = normalized_old[0]
            for start, line in enumerate(lines):
                score = 1 if CursorPatchApplier._normalize_line(line) == first else 0
                if score > best_score:
                    best_score = score
                    best_start = start

        return best_start, max(best_score, 0)

    @staticmethod
    def _normalize_line(line: str) -> str:
        return "".join(line.split())

    _UNICODE_ESCAPE_RE = re.compile(r"\\u([0-9A-Fa-f]{4})")

    @staticmethod
    def _escape_fold(line: str) -> str:
        """Whitespace fold that also equates a literal ``\\uXXXX`` escape with its
        decoded glyph, so a SEARCH mangled by JSON transport can still be located.

        Steps: drop whitespace, lowercase any literal ``\\uXXXX`` hex, then render
        every BMP non-ASCII char back to its ``\\uXXXX`` form. Both the on-disk
        ``\\u4e00`` and the decoded ``一`` collapse to ``\\u4e00``. Non-BMP chars are
        left untouched (would need ``\\U`` form) so they simply don't fold.
        """
        stripped = "".join(line.split())
        stripped = CursorPatchApplier._UNICODE_ESCAPE_RE.sub(
            lambda match: "\\u" + match.group(1).lower(), stripped
        )
        return "".join(
            char if ord(char) < 128 or ord(char) > 0xFFFF else f"\\u{ord(char):04x}"
            for char in stripped
        )

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
