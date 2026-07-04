from __future__ import annotations

import re

# Completed: [√] [✓] [x] [X] [v] [V]   Open: [ ]
_CHECKLIST_DONE_INNER = r"x|X|✓|√|v|V"
_CHECKLIST_LINE = re.compile(
    rf"-\s+\[({_CHECKLIST_DONE_INNER}| )\]\s+(.*)",
)
_CHECKLIST_DONE_ITEM = re.compile(rf"^\[(?:{_CHECKLIST_DONE_INNER})\]")
_CHECKLIST_OPEN_ITEM = re.compile(r"^\[\s\]")


def checklist_item_done(item: str) -> bool:
    return bool(_CHECKLIST_DONE_ITEM.match(item.strip()))


def checklist_item_open(item: str) -> bool:
    return bool(_CHECKLIST_OPEN_ITEM.match(item.strip()))


def checklist_plan_complete(checklist: tuple[str, ...]) -> bool:
    if not checklist:
        return False
    return not any(checklist_item_open(item) for item in checklist)


def normalize_checklist_marker(raw: str) -> str:
    """Map parsed bracket content to stored done (√) or open (space) marker."""
    if raw.strip() and raw.strip() != " ":
        if raw in {"x", "X", "✓", "√", "v", "V"}:
            return "√"
    return " "


def parse_checklist_lines(text: str) -> tuple[str, ...]:
    items: list[str] = []
    for status, task in _CHECKLIST_LINE.findall(text):
        marker = normalize_checklist_marker(status)
        items.append(f"[{marker}] {task.strip()}")
    return tuple(items)
