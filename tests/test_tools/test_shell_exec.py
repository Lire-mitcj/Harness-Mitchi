from __future__ import annotations

from src.tools.shell.executor import _is_no_match_search_result


def test_shell_exec_classifies_grep_no_matches_as_non_failure() -> None:
    assert _is_no_match_search_result(
        "grep -n definitely_not_here pyproject.toml",
        1,
        "",
    )
    assert _is_no_match_search_result(
        "printf 'abc\\n' | grep -n definitely_not_here",
        1,
        "",
    )
    assert not _is_no_match_search_result("grep '[' pyproject.toml", 2, "bad regex")
