from __future__ import annotations

from src.harness.task_analysis import analyze_task, classify_intent


def test_intent_classification_function_refactor() -> None:
    intent, confidence = classify_intent("重构订单详情处理，统一脱敏")
    assert intent == "function_refactor"
    assert confidence >= 0.9

    intent, _ = classify_intent("Replace build_order_detail_sql with _fetch_order_detail")
    assert intent == "function_refactor"


def test_intent_classification_view_requires_explicit_view_replacement() -> None:
    intent, _ = classify_intent("查询订单 SQL")
    assert intent != "sql_view_rewrite"

    intent, _ = classify_intent("使用 view_ticket_report_detail 视图替换订单详情查询")
    assert intent == "sql_view_rewrite"


def test_task_analysis_marks_function_refactor_high_not_ready_without_context() -> None:
    analysis = analyze_task("重构订单详情处理，统一脱敏", None)
    assert analysis.intent == "function_refactor"
    assert analysis.complexity == "high"
    assert not analysis.edit_ready
    assert analysis.edit_strategy == "function_refactor"
