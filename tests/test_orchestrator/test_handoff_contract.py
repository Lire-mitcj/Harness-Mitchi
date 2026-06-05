from __future__ import annotations

from src.orchestrator.handoff_contract import (
    build_handoff_contract,
    extract_handoff_contract,
    format_handoff_contract,
    merge_handoff_contracts,
)


def test_handoff_contract_extracts_views_and_target_locations() -> None:
    summary = (
        "Result: 已定位 3 个相关代码位置。\n"
        "Evidence:\n"
        "- main.py:120 | query_boarding_pass | SELECT * FROM boarding_pass WHERE id=:id\n"
        "- db/init/init.sql:392 | 视图定义 | -- 视图：登机牌详情\n"
        "- db/init/init.sql:394 | 视图定义 | CREATE VIEW v_boarding_pass AS SELECT * FROM boarding_pass\n"
    )

    contract = build_handoff_contract(
        user_request="修改 main.py 中登机牌查询接口，使其从视图查询",
        subtask_id="st-1",
        summary=summary,
    )

    assert contract["must_modify"][0]["file"] == "main.py"
    assert contract["must_modify"][0]["symbol_or_api"] == "query_boarding_pass"
    assert contract["available_views"][0]["name"] == "v_boarding_pass"
    assert "boarding_pass" in contract["must_modify"][0]["current_sql"]


def test_handoff_contract_round_trip_and_merge() -> None:
    contract = {
        "schema": "mitkii.handoff_contract.v1",
        "source_subtask": "st-1",
        "must_modify": [{"file": "main.py", "line": 10}],
        "available_views": [{"name": "v_boarding_pass", "fields": []}],
        "evidence": [{"file": "main.py", "line": 10, "snippet": "SELECT"}],
        "tool_policy": {"allowed_tools": ["edit_file"]},
    }
    text = format_handoff_contract(contract)

    assert extract_handoff_contract(text)["source_subtask"] == "st-1"  # type: ignore[index]

    merged = merge_handoff_contracts(
        user_request="登机牌",
        prior_summaries={"st-1": text},
        current_search_output="",
    )

    assert merged["must_modify"][0]["file"] == "main.py"
    assert merged["available_views"][0]["name"] == "v_boarding_pass"
