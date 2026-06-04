from __future__ import annotations

EDIT_AMBIGUOUS_HINT = (
    "edit_file failed because old_string matched multiple times. "
    "read_file the exact lines you need, then call edit_file with a UNIQUE "
    "old_string (include ≥5 lines of surrounding SQL/code). "
    "Do NOT retry the same short identifier alone."
)

EDIT_IDENTICAL_HINT = (
    "edit_file failed: old_string and new_string were identical. "
    "Copy the exact lines to change from the preloaded <file> block as old_string, "
    "then provide the modified code as new_string — they must differ."
)

EDIT_NOT_FOUND_HINT = (
    "edit_file failed: old_string not found in the file. "
    "Copy text exactly from the preloaded <file> slice (same whitespace and quotes). "
    "Use a multi-line old_string with surrounding code — not just a function signature."
)


def is_edit_ambiguous_error(message: str) -> bool:
    lower = message.lower()
    return (
        "old_string" in lower
        and "appears" in lower
        and "times" in lower
    )


def is_edit_identical_error(message: str) -> bool:
    lower = message.lower()
    return "old_string and new_string are identical" in lower


def is_edit_not_found_error(message: str) -> bool:
    lower = message.lower()
    return "old_string not found" in lower


def is_edit_recoverable_error(message: str) -> bool:
    return (
        is_edit_ambiguous_error(message)
        or is_edit_identical_error(message)
        or is_edit_not_found_error(message)
    )


def edit_failure_hint(error_trace: list[str], *, lookback: int = 6) -> str | None:
    for entry in reversed(error_trace[-lookback:]):
        if is_edit_identical_error(entry):
            return EDIT_IDENTICAL_HINT
        if is_edit_not_found_error(entry):
            return EDIT_NOT_FOUND_HINT
        if is_edit_ambiguous_error(entry):
            return EDIT_AMBIGUOUS_HINT
    return None


def recent_edit_ambiguous(error_trace: list[str], *, lookback: int = 6) -> bool:
    for entry in reversed(error_trace[-lookback:]):
        if is_edit_ambiguous_error(entry):
            return True
    return False


def recent_edit_recoverable_failure(error_trace: list[str], *, lookback: int = 6) -> bool:
    return edit_failure_hint(error_trace, lookback=lookback) is not None


def should_skip_explore_fold(error_trace: list[str]) -> bool:
    """Keep grep/read tool output in context after edit match failures."""
    return recent_edit_recoverable_failure(error_trace)
