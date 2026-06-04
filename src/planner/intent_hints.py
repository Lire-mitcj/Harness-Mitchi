from __future__ import annotations

import re

_EXPLORATION = re.compile(
    r"(哪些|什么|有没有|列出|介绍|说明|查看|找找|统计|架构|结构|视图|页面|路由|"
    r"what|which|list|show me|describe|inventory|overview|views?|pages?|routes?)",
    re.IGNORECASE,
)
_CODING_ACTION = re.compile(
    r"(fix|修复|bug|implement|实现|refactor|重构|add|添加|delete|删除|"
    r"patch|migrate|debug|优化|update|修改|改成|改为|使用|写入|创建)",
    re.IGNORECASE,
)


def is_exploration_request(text: str) -> bool:
    """True for inventory / Q&A tasks that should not start as edit subtasks."""
    stripped = text.strip()
    if not stripped:
        return False
    if _CODING_ACTION.search(stripped):
        return False
    if stripped.endswith("?"):
        return True
    return bool(_EXPLORATION.search(stripped))


def cap_structure_text(text: str, *, max_chars: int = 3500) -> str:
    """Trim project tree/manifest blocks to keep Scout/Planner prompts small."""
    if len(text) <= max_chars:
        return text
    suffix = "\n...[structure truncated]"
    keep = max(0, max_chars - len(suffix))
    return text[:keep] + suffix
