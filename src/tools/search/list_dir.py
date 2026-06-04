from __future__ import annotations

from pathlib import Path
from typing import Any

from src.agent.types import RiskLevel, ToolResult
from src.tools.base import Tool


def _format_size(size: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


class ListDirTool(Tool):
    name = "list_dir"
    description = "List directory contents with file types and sizes."
    risk_level = RiskLevel.SAFE
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list (default: current dir)",
                "default": ".",
            },
            "recursive": {
                "type": "boolean",
                "description": "List subdirectories recursively",
                "default": False,
            },
            "max_depth": {
                "type": "integer",
                "description": "Maximum directory depth when recursive (default: 2)",
                "default": 2,
            },
        },
        "required": [],
    }

    async def execute(self, **params: Any) -> ToolResult:
        validated = self.validate_params(params)
        root = Path(validated.get("path", ".")).expanduser().resolve()
        recursive: bool = validated.get("recursive", False)
        max_depth: int = validated.get("max_depth", 2)

        if not root.exists():
            return ToolResult(success=False, output="", error=f"Directory not found: {root}")
        if not root.is_dir():
            return ToolResult(success=False, output="", error=f"Not a directory: {root}")

        lines: list[str] = []
        self._collect(root, root, lines, recursive, max_depth, current_depth=0)

        if not lines:
            return ToolResult(success=True, output="(empty directory)")

        MAX_ENTRIES = 500
        truncated = len(lines) > MAX_ENTRIES
        output = "\n".join(lines[:MAX_ENTRIES])
        if truncated:
            output += f"\n\n[Showing first {MAX_ENTRIES} of {len(lines)} entries.]"

        return ToolResult(
            success=True,
            output=output,
            metadata={"entry_count": len(lines), "truncated": truncated},
        )

    def _collect(
        self,
        current: Path,
        root: Path,
        lines: list[str],
        recursive: bool,
        max_depth: int,
        current_depth: int,
    ) -> None:
        try:
            entries = sorted(current.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except PermissionError:
            lines.append(f"  {'  ' * current_depth}[permission denied]")
            return

        indent = "  " * current_depth
        for entry in entries:
            try:
                rel = entry.relative_to(root)
            except ValueError:
                rel = entry

            if entry.is_dir():
                child_count = sum(1 for _ in entry.iterdir()) if _can_list(entry) else "?"
                lines.append(f"{indent}📁 {rel.name}/ ({child_count} items)")
                if recursive and current_depth < max_depth:
                    self._collect(entry, root, lines, recursive, max_depth, current_depth + 1)
            elif entry.is_file():
                size = _format_size(entry.stat().st_size)
                lines.append(f"{indent}📄 {rel.name} ({size})")
            else:
                lines.append(f"{indent}   {rel.name}")


def _can_list(path: Path) -> bool:
    try:
        next(path.iterdir())
        return True
    except (PermissionError, StopIteration):
        return True
    except OSError:
        return False
