from __future__ import annotations

from src.cli.report_format import compact_diagnose_report, format_report_for_terminal, milestone_from_report


def test_format_report_flattens_markdown_table() -> None:
    raw = """## Diagnosis Summary

| File | Symbol | Kind |
|------|--------|------|
| db/init/init.sql (line 354) | v_flight_monitoring | CREATE VIEW |
| main.py (line 1803) | v_flight_monitoring | SELECT usage |
"""
    out = format_report_for_terminal(raw)
    assert "db/init/init.sql (line 354) — v_flight_monitoring (CREATE VIEW)" in out
    assert "|" not in out
    assert "Diagnosis Summary" in out


def test_milestone_from_report_prefers_bullet_line() -> None:
    raw = """## Diagnosis Summary

| File | Symbol | Kind |
|------|--------|------|
| db/init/init.sql | v_flight_monitoring | CREATE VIEW |
"""
    hint = milestone_from_report(raw)
    assert "v_flight_monitoring" in hint


def test_compact_diagnose_report_strips_meta_sections() -> None:
    raw = """## Diagnosis Summary — Subtask [st-1]

What I found

The view exists but export does not use it.

## Acceptance criteria status

✅ Met — listed paths

## Next subtask (for the Orchestrator)

Edit app.py

  • db/init/init.sql — view_ticket_report_detail (CREATE VIEW)
  • app.py — make_boarding_pass_pdf (hardcoded dict)
"""
    out = compact_diagnose_report(raw, max_lines=8)
    assert "Acceptance criteria" not in out
    assert "Next subtask" not in out
    assert "Met —" not in out
    assert "view_ticket_report_detail" in out or "app.py" in out


def test_milestone_strips_dsml_tool_protocol() -> None:
    raw = (
        '<｜DSML｜invoke name="grep_search">\n'
        "<｜DSML｜tool_calls>\n"
        '<｜DSML｜invoke name="grep_search">\n'
        '<｜DSML｜parameter name="arguments" string="false">{"pattern":"x"}</｜DSML｜parameter>\n'
        "</｜DSML｜invoke>\n"
        "</｜DSML｜tool_calls>"
    )

    assert "DSML" not in milestone_from_report(raw)
    assert compact_diagnose_report(raw) == ""


def test_chinese_diagnose_report_is_structured_for_terminal() -> None:
    raw = """诊断结果：登机牌信息查询接口未使用视图
发现总结
1. **接口定位**：在 `app.py:1584` 找到 `make_boarding_pass_pdf()`
2. **视图使用情况**：未发现 `FROM v_boarding`
完成条件
✅ 确认接口是否使用视图：否
"""

    out = compact_diagnose_report(raw, max_lines=8)

    assert "发现总结" not in out
    assert "完成条件" not in out
    assert "**" not in out
    assert "`" not in out
    assert "• 接口定位: 在 app.py:1584 找到 make_boarding_pass_pdf()" in out
    assert "✓ 确认接口是否使用视图：否" in out
