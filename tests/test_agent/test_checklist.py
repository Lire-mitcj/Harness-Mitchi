from __future__ import annotations

from src.agent.checklist import (
    checklist_plan_complete,
    parse_checklist_lines,
)


def test_parse_checklist_normalizes_x_to_checkmark() -> None:
    text = "- [x] Done task\n- [ ] Open task\n"
    assert parse_checklist_lines(text) == (
        "[√] Done task",
        "[ ] Open task",
    )


def test_parse_checklist_accepts_checkmark_input() -> None:
    text = "- [√] Done\n- [✓] Also done\n"
    assert parse_checklist_lines(text) == (
        "[√] Done",
        "[√] Also done",
    )


def test_checklist_plan_complete_accepts_legacy_x_markers() -> None:
    assert checklist_plan_complete(("[x] a", "[X] b")) is True


def test_checklist_plan_complete_false_when_open_items() -> None:
    assert checklist_plan_complete(("[√] a", "[ ] b")) is False
