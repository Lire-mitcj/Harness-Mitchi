from __future__ import annotations

from src.llm.dsml import (
    contains_tool_call_markup,
    split_dsml_tool_calls,
    strip_dsml_text,
)


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


def test_plain_stream_chunks_preserve_boundary_whitespace() -> None:
    chunks = ["The", " user", " is", " asking", " about", " auth."]
    assert "".join(strip_dsml_text(chunk) for chunk in chunks) == (
        "The user is asking about auth."
    )
    assert strip_dsml_text("\n\n<thinking>\n") == "\n\n<thinking>\n"


def test_split_deepseek_xml_invoke_tool_calls() -> None:
    text = """
<thinking>Need source.</thinking>
<tool_calls>
  <invoke name="view_symbol_code">
    <parameter name="target_file" string="true">list.py</parameter>
    <parameter name="symbol" string="true">build_router.archive_passenger</parameter>
  </invoke>
  <invoke name="grep_search">
    <parameter name="pattern" string="true">auth|token</parameter>
    <parameter name="path" string="true">main.py</parameter>
    <parameter name="max_results" string="false">50</parameter>
  </invoke>
</tool_calls>
"""
    cleaned, calls = split_dsml_tool_calls(text)
    assert cleaned == "<thinking>Need source.</thinking>"
    assert [(call.name, call.arguments) for call in calls] == [
        (
            "view_symbol_code",
            {"target_file": "list.py", "symbol": "build_router.archive_passenger"},
        ),
        (
            "grep_search",
            {"pattern": "auth|token", "path": "main.py", "max_results": 50},
        ),
    ]


def test_split_deepseek_tool_results_wrapped_calls() -> None:
    text = """
<tool_results><tool_call><tool_name>grep_search</tool_name><parameters>
<pattern>role|admin</pattern><path>db/init</path><include>*.sql</include>
<max_results>20</max_results></parameters></tool_call></tool_results>
"""
    cleaned, calls = split_dsml_tool_calls(text)
    assert cleaned is None
    assert len(calls) == 1
    assert calls[0].name == "grep_search"
    assert calls[0].arguments == {
        "pattern": "role|admin",
        "path": "db/init",
        "include": "*.sql",
        "max_results": 20,
    }


def test_split_deepseek_standalone_tool_call_invoke() -> None:
    text = """
<tool_call>
<invoke name="grep_search">
<parameter name="pattern" string="true">archive</parameter>
<parameter name="path" string="true">.</parameter>
<parameter name="include" string="true">*.py</parameter>
</invoke>
</tool_call>
"""
    cleaned, calls = split_dsml_tool_calls(text)

    assert cleaned is None
    assert [(call.name, call.arguments) for call in calls] == [
        ("grep_search", {"pattern": "archive", "path": ".", "include": "*.py"})
    ]


def test_split_deepseek_function_tool_call() -> None:
    text = """
<function>
<tool_name>grep_search</tool_name>
<parameters>
<pattern>archive</pattern><path>.</path><include>*.py</include>
<case_insensitive>true</case_insensitive>
</parameters>
</function>
"""
    cleaned, calls = split_dsml_tool_calls(text)

    assert cleaned is None
    assert [(call.name, call.arguments) for call in calls] == [
        (
            "grep_search",
            {
                "pattern": "archive",
                "path": ".",
                "include": "*.py",
                "case_insensitive": True,
            },
        )
    ]


def test_malformed_tool_markup_is_preserved_for_retry_guard() -> None:
    text = "<tool_calls><invoke name=\"grep_search\">broken"
    cleaned, calls = split_dsml_tool_calls(text)
    assert cleaned == text
    assert calls == []
    assert contains_tool_call_markup(cleaned)


def test_split_deepseek_bare_xml_tool_calls() -> None:
    text = """
<thinking>
We need to search for archive.
</thinking>
<grep_search>
  <pattern>archive</pattern>
  <path>.</path>
</grep_search>
"""
    known_tools = {"grep_search", "view_symbol_code"}
    cleaned, calls = split_dsml_tool_calls(text, known_tool_names=known_tools)
    assert cleaned is None
    assert len(calls) == 1
    assert calls[0].name == "grep_search"
    assert calls[0].arguments == {"pattern": "archive", "path": "."}

    # Test markup detection
    assert contains_tool_call_markup(text, known_tool_names=known_tools)
