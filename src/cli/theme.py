from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rich.theme import Theme


@dataclass(frozen=True, slots=True)
class MitKIITheme:
    """Terminal colour palette for all MitKII UI elements."""

    thinking: str = "dim cyan"
    tool: str = "bold blue"
    tool_param: str = "dim blue"
    success: str = "bold green"
    error: str = "bold red"
    warning: str = "bold yellow"
    info: str = "bold white"
    accent: str = "bold magenta"
    muted: str = "dim"
    banner: str = "bold cyan"
    prompt: str = "bold green"
    file_path: str = "underline cyan"
    diff_add: str = "green"
    diff_remove: str = "red"
    cost: str = "dim yellow"
    score_pass: str = "bold green"
    score_fail: str = "bold red"

    def to_rich_theme(self) -> Theme:
        return Theme({
            "mitkii.thinking": self.thinking,
            "mitkii.tool": self.tool,
            "mitkii.tool_param": self.tool_param,
            "mitkii.success": self.success,
            "mitkii.error": self.error,
            "mitkii.warning": self.warning,
            "mitkii.info": self.info,
            "mitkii.accent": self.accent,
            "mitkii.muted": self.muted,
            "mitkii.banner": self.banner,
            "mitkii.prompt": self.prompt,
            "mitkii.file_path": self.file_path,
            "mitkii.diff_add": self.diff_add,
            "mitkii.diff_remove": self.diff_remove,
            "mitkii.cost": self.cost,
        })


@lru_cache(maxsize=1)
def get_theme() -> MitKIITheme:
    return MitKIITheme()
