from __future__ import annotations

from src.executor.final_output import normalize_executor_final_text, parse_executor_final


def test_parse_executor_final_json_and_format_display_text() -> None:
    text = (
        '{"result":"已定位视图","acceptance_met":true,'
        '"evidence":[{"path":"app.py","line":42,"symbol":"query",'
        '"snippet":"CREATE VIEW v AS SELECT ...","reason":"目标 SQL"}],'
        '"blocker":""}'
    )

    parsed = parse_executor_final(text)
    display, raw = normalize_executor_final_text(text)

    assert parsed is not None
    assert parsed.acceptance_met is True
    assert raw is not None
    assert "Result: 已定位视图" in display
    assert "app.py:42" in display
    assert "CREATE VIEW" in display


def test_parse_executor_final_ignores_plain_text() -> None:
    assert parse_executor_final("Result: plain text") is None
