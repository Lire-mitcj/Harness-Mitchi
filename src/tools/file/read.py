from __future__ import annotations

import time
from pathlib import Path
from typing import Any, ClassVar

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool

MAX_FILE_SIZE = 100 * 1024  # 100 KB
DEFAULT_PREVIEW_LINES = 200


class _FileReadCache:
    _cache: ClassVar[dict[str, tuple[int, int, str, list[str]]]] = {}

    @classmethod
    def load(cls, file_path: Path) -> tuple[str, list[str], int, bool, float]:
        """Return (raw, lines, total_lines, cache_hit, elapsed_ms)."""
        start = time.monotonic()
        size = file_path.stat().st_size
        mtime_ns = file_path.stat().st_mtime_ns
        cache_key = str(file_path)
        cached = cls._cache.get(cache_key)
        cache_hit = False

        if cached and cached[0] == mtime_ns and cached[1] == size:
            raw, lines = cached[2], cached[3]
            cache_hit = True
        else:
            raw = file_path.read_text(encoding="utf-8", errors="replace")
            lines = raw.splitlines(keepends=True)
            if len(cls._cache) >= 128:
                cls._cache.pop(next(iter(cls._cache)))
            cls._cache[cache_key] = (mtime_ns, size, raw, lines)

        elapsed_ms = round((time.monotonic() - start) * 1000, 1)
        return raw, lines, len(lines), cache_hit, elapsed_ms


def _format_file_content(
    file_path: Path,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> tuple[str, dict[str, Any]]:
    """Read and format one file. Returns (content, metadata)."""
    try:
        size = file_path.stat().st_size
        raw, lines, total_lines, cache_hit, elapsed_ms = _FileReadCache.load(file_path)
        truncated = size > MAX_FILE_SIZE

        if start_line is not None or end_line is not None:
            s = max((start_line or 1) - 1, 0)
            e = min(end_line or total_lines, total_lines)
            selected = lines[s:e]
            numbered = [f"{i + s + 1:>6}|{line}" for i, line in enumerate(selected)]
        elif truncated:
            char_limit = MAX_FILE_SIZE
            raw = raw[:char_limit]
            lines = raw.splitlines(keepends=True)
            numbered = [f"{i + 1:>6}|{line}" for i, line in enumerate(lines)]
        else:
            preview = lines[:DEFAULT_PREVIEW_LINES]
            numbered = [f"{i + 1:>6}|{line}" for i, line in enumerate(preview)]

        content = "".join(numbered)
        if truncated and start_line is None:
            content += (
                f"\n\n[Truncated — file is {size:,} bytes, "
                f"showing first {MAX_FILE_SIZE:,} bytes. "
                f"Use start_line/end_line for targeted reading.]"
            )
        elif start_line is None and end_line is None and total_lines > DEFAULT_PREVIEW_LINES:
            content += (
                f"\n\n[Preview mode — file has {total_lines:,} lines, "
                f"showing first {DEFAULT_PREVIEW_LINES}. "
                "Use start_line/end_line for targeted reading.]"
            )

        return content, {
            "path": str(file_path),
            "total_lines": total_lines,
            "elapsed_ms": elapsed_ms,
            "cache_hit": cache_hit,
        }
    except PermissionError:
        raise PermissionError(f"Permission denied: {file_path}") from None
    except OSError as exc:
        raise OSError(f"IO error reading {file_path}: {exc}") from exc


def _list_dir_result(directory: Path, *, limit: int = 80) -> ToolResult:
    """When read_file gets a directory, return a listing instead of hard-failing."""
    try:
        entries = sorted(directory.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except PermissionError:
        return ToolResult(success=False, output="", error=f"Permission denied: {directory}")
    except OSError as exc:
        return ToolResult(success=False, output="", error=f"Cannot list directory: {exc}")

    lines = [f"[Directory listing — use list_dir or read specific files, not the project root as a file]"]
    for entry in entries[:limit]:
        suffix = "/" if entry.is_dir() else ""
        lines.append(f"  {entry.name}{suffix}")
    if len(entries) > limit:
        lines.append(f"  ... ({len(entries) - limit} more entries)")
    return ToolResult(
        success=True,
        output="\n".join(lines),
        metadata={"path": str(directory), "entry_count": len(entries), "is_directory": True},
    )


class ReadFileTool(Tool):
    name = "read_file"
    description = "Read the contents of a file. Can optionally specify line range."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file to read"},
            "start_line": {
                "type": "integer",
                "description": "First line to read (1-indexed, inclusive)",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to read (1-indexed, inclusive)",
            },
        },
        "required": ["path"],
    }

    async def execute(self, **params: object) -> ToolResult:
        validated = self.validate_params(params)
        file_path = Path(validated["path"]).expanduser().resolve()
        start_line: int | None = validated.get("start_line")
        end_line: int | None = validated.get("end_line")

        if not file_path.exists():
            return ToolResult(success=False, output="", error=f"File not found: {file_path}")
        if file_path.is_dir():
            return _list_dir_result(file_path)

        try:
            content, meta = _format_file_content(
                file_path, start_line=start_line, end_line=end_line
            )
            return ToolResult(success=True, output=content, metadata=meta)
        except PermissionError as exc:
            return ToolResult(success=False, output="", error=str(exc))
        except OSError as exc:
            return ToolResult(success=False, output="", error=str(exc))


class ReadFilesTool(Tool):
    name = "read_files"
    description = (
        "Read multiple files in one call. Prefer this over many separate read_file calls "
        "when you need several files at once."
    )
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "paths": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of file paths to read",
            },
            "start_line": {
                "type": "integer",
                "description": "Optional first line (1-indexed) applied to every file",
            },
            "end_line": {
                "type": "integer",
                "description": "Optional last line (1-indexed) applied to every file",
            },
        },
        "required": ["paths"],
    }

    async def execute(self, **params: object) -> ToolResult:
        validated = self.validate_params(params)
        paths: list[str] = validated["paths"]
        start_line: int | None = validated.get("start_line")
        end_line: int | None = validated.get("end_line")

        if not paths:
            return ToolResult(success=False, output="", error="paths must not be empty")

        sections: list[str] = []
        file_meta: list[dict[str, Any]] = []
        per_file_outputs: list[dict[str, str]] = []
        errors: list[str] = []
        total_elapsed = 0.0

        for raw_path in paths:
            file_path = Path(raw_path).expanduser().resolve()
            if not file_path.exists():
                errors.append(f"File not found: {file_path}")
                continue
            if file_path.is_dir():
                listing = _list_dir_result(file_path)
                sections.append(
                    f"===== {file_path} (directory) =====\n{listing.output}"
                )
                per_file_outputs.append({"path": str(file_path), "output": listing.output})
                continue
            if not file_path.is_file():
                errors.append(f"Not a file: {file_path}")
                continue

            try:
                content, meta = _format_file_content(
                    file_path, start_line=start_line, end_line=end_line
                )
                rel = str(file_path)
                sections.append(
                    f"===== {rel} ({meta['total_lines']} lines) =====\n{content}"
                )
                file_meta.append(meta)
                per_file_outputs.append({"path": meta["path"], "output": content})
                total_elapsed += float(meta.get("elapsed_ms", 0))
            except (PermissionError, OSError) as exc:
                errors.append(str(exc))

        if not sections and errors:
            return ToolResult(
                success=False,
                output="",
                error="; ".join(errors),
            )

        output = "\n\n".join(sections)
        if errors:
            output += "\n\n[Skipped paths]\n" + "\n".join(f"- {e}" for e in errors)

        return ToolResult(
            success=True,
            output=output,
            metadata={
                "paths": [m["path"] for m in file_meta],
                "file_count": len(file_meta),
                "elapsed_ms": round(total_elapsed, 1),
                "files": file_meta,
                "per_file_outputs": per_file_outputs,
                "errors": errors,
            },
        )
