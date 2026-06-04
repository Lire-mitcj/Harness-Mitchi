from __future__ import annotations

from src.llm.dsml import split_dsml_tool_calls, strip_dsml_text


def test_split_dsml_tool_call_from_content() -> None:
    text = (
        '<｜DSML｜invoke name="grep_search">\n'
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="grep_search">\n'
        '<｜DSML｜parameter name="arguments" string="false">'
        '{"pattern":"@app\\\\.(get|post).*ticket"}'
        "</｜DSML｜parameter>\n"
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    cleaned, calls = split_dsml_tool_calls(text)

    assert cleaned is None
    assert len(calls) == 1
    assert calls[0].name == "grep_search"
    assert calls[0].arguments == {"pattern": "@app\\.(get|post).*ticket"}


def test_strip_dsml_keeps_plain_text() -> None:
    text = "Found target.\n<｜DSML｜tool_calls></｜DSML｜tool_calls>"

    assert strip_dsml_text(text) == "Found target."
