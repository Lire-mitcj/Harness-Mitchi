from __future__ import annotations

_DIRECT_PLAN_PREFIX = "/plan"


def parse_turn_input(raw: str) -> tuple[str, bool]:
    """Parse user input; return (task_text, skip_discovery).

    Prefix ``/plan `` skips the Scout discovery phase (direct Planner), similar
    to Cursor plan-direct mode for greenfield / unrelated new files.
    """
    text = raw.strip()
    lower = text.lower()
    if lower.startswith(f"{_DIRECT_PLAN_PREFIX} "):
        return text[len(_DIRECT_PLAN_PREFIX) + 1 :].strip(), True
    if lower.startswith(f"{_DIRECT_PLAN_PREFIX}:"):
        return text[len(_DIRECT_PLAN_PREFIX) + 1 :].strip(), True
    return text, False
