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
    assert contract["must_modify"][0]["should_change_to"] == "apply the requested code change"


def test_handoff_contract_only_uses_view_when_explicit() -> None:
    summary = (
        "- main.py:120 | query_order | SELECT * FROM ticket_order\n"
        "- db/init/init.sql:394 | 视图定义 | CREATE VIEW view_ticket_report_detail AS SELECT 1\n"
    )

    generic = build_handoff_contract(
        user_request="重构订单详情处理，统一脱敏逻辑",
        subtask_id="st-1",
        summary=summary,
    )
    assert generic["must_modify"][0]["should_change_to"] == "apply the requested code change"

    explicit = build_handoff_contract(
        user_request="使用 view_ticket_report_detail 视图替换订单详情查询",
        subtask_id="st-1",
        summary=summary,
    )
    assert explicit["must_modify"][0]["should_change_to"] == "use view view_ticket_report_detail"


def test_handoff_contract_round_trip_and_merge() -> None:
    contract = {
        "schema": "mitkii.handoff.v1",
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


def test_handoff_contract_extract_with_trailing_content() -> None:
    text = (
        "HANDOFF_CONTRACT_JSON\n"
        "{\n"
        '  "source_subtask": "st-1",\n'
        '  "must_modify": []\n'
        "}\n"
        "\n"
        "Some extra notes with braces like {this} here."
    )
    contract = extract_handoff_contract(text)
    assert contract is not None
    assert contract["source_subtask"] == "st-1"


def test_handoff_contract_normalization() -> None:
    text = (
        "HANDOFF_CONTRACT_JSON\n"
        "{\n"
        '  "edit_targets": [\n'
        "    {\n"
        '      "file": "app.py",\n'
        '      "symbol": "build_order_detail_sql",\n'
        '      "line_start": 450,\n'
        '      "line_end": 485,\n'
        '      "snippet": "...",\n'
        '      "decision": "replace SQL with view"\n'
        "    }\n"
        "  ],\n"
        '  "resolved_dependencies": [\n'
        "    {\n"
        '      "kind": "database_view",\n'
        '      "name": "view_ticket_report_detail"\n'
        "    }\n"
        "  ],\n"
        '  "facts": [{"note": "order logic uses ticket table"}]\n'
        "}"
    )
    contract = extract_handoff_contract(text)
    assert contract is not None
    assert "must_modify" in contract
    assert contract["must_modify"][0]["file"] == "app.py"
    assert contract["must_modify"][0]["symbol_or_api"] == "build_order_detail_sql"
    assert contract["must_modify"][0]["line"] == 450
    assert contract["must_modify"][0]["should_change_to"] == "replace SQL with view"
    assert "available_views" in contract
    assert contract["available_views"][0]["name"] == "view_ticket_report_detail"
    assert "evidence" in contract
    assert len(contract["evidence"]) == 1


def test_handoff_contract_extract_from_patch_intent() -> None:
    text = (
        "PATCH_INTENT_JSON\n"
        "{\n"
        '  "edit_strategy": "sql_view_rewrite",\n'
        '  "edit_targets": [{"file": "main.py", "symbol": "foo"}],\n'
        '  "dependencies": [{"kind": "database_view", "name": "view_foo"}]\n'
        "}"
    )
    contract = extract_handoff_contract(text)
    assert contract is not None
    assert contract["must_modify"][0]["file"] == "main.py"
    assert contract["available_views"][0]["name"] == "view_foo"


def test_infer_target_view_pinyin_chinese_matching() -> None:
    from src.orchestrator.handoff_contract import _infer_target_view
    views = [
        {"name": "v_order_detail"},
        {"name": "v_boarding_pass"},
        {"name": "v_flight_load"},
    ]
    # Exact view name match in request
    assert _infer_target_view("Please use v_boarding_pass view", views) == "v_boarding_pass"
    # Chinese match
    assert _infer_target_view("使用登机牌视图替换", views) == "v_boarding_pass"
    # Pinyin match
    assert _infer_target_view("dingdan query rewrite using view", views) == "v_order_detail"
    # Fallback to first view name
    assert _infer_target_view("unrelated request", views) == "v_order_detail"


def test_merge_handoff_contracts_global_summaries() -> None:
    from src.orchestrator.handoff_contract import merge_handoff_contracts
    global_summaries = {
        "st-1": "HANDOFF_CONTRACT_JSON\n"
                "{\n"
                '  "must_modify": [{"file": "main.py", "line": 100}],\n'
                '  "available_views": [{"name": "v_global_view"}]\n'
                "}"
    }
    merged = merge_handoff_contracts(
        user_request="Please rewrite using view",
        prior_summaries={},
        current_search_output="",
        global_summaries=global_summaries,
    )
    assert merged["must_modify"][0]["file"] == "main.py"
    assert merged["available_views"][0]["name"] == "v_global_view"
    assert merged["target_view"] == "v_global_view"


def test_merge_handoff_contracts_global_patch_intent_backfills_dependencies() -> None:
    from src.orchestrator.handoff_contract import merge_handoff_contracts

    global_summaries = {
        "st-2": """PATCH_INTENT_JSON
{
  "edit_ready": true,
  "edit_strategy": "sql_view_rewrite",
  "edit_targets": [
    {
      "file": "main.py",
      "symbol": "query",
      "line_start": 1,
      "line_end": 3,
      "snippet": "def query(): pass",
      "decision": "use v_global_view"
    }
  ],
  "available_views": [
    {
      "name": "v_global_view",
      "file": "schema.sql",
      "line_start": 10,
      "line_end": 20,
      "columns": ["id"]
    }
  ],
  "dependencies": [],
  "acceptance_criteria": ["query uses view"],
  "target_view": "v_global_view"
}
"""
    }
    merged = merge_handoff_contracts(
        user_request="Please rewrite using v_global_view view",
        prior_summaries={},
        current_search_output="",
        global_summaries=global_summaries,
    )

    assert merged["target_view"] == "v_global_view"
    assert merged["dependencies"][0]["name"] == "v_global_view"
    assert merged["resolved_dependencies"][0]["columns"] == ["id"]


def test_merge_handoff_contracts_preserves_patch_intent_contract() -> None:
    patch = """PATCH_INTENT_JSON
{
  "edit_ready": true,
  "edit_strategy": "sql_view_rewrite",
  "edit_targets": [
    {
      "file": "main.py",
      "symbol": "build_order_detail_sql",
      "line_start": 10,
      "line_end": 40,
      "snippet": "SELECT o.id FROM ticket_order o",
      "decision": "use view_ticket_report_detail"
    }
  ],
  "dependencies": [
    {
      "role": "replacement_source",
      "kind": "database_view",
      "name": "view_ticket_report_detail",
      "columns": ["id"],
      "replaces_objects": ["ticket_order"]
    }
  ],
  "acceptance_criteria": ["target uses view"],
  "target_view": "view_ticket_report_detail"
}
"""
    merged = merge_handoff_contracts(
        user_request="使用 view_ticket_report_detail 视图替换订单详情查询",
        prior_summaries={"st-2": patch},
        current_search_output="",
    )

    assert merged["target_view"] == "view_ticket_report_detail"
    assert merged["edit_targets"][0]["symbol"] == "build_order_detail_sql"
    assert merged["dependencies"][0]["name"] == "view_ticket_report_detail"
    assert merged["resolved_dependencies"][0]["columns"] == ["id"]


def test_normalize_contract_keys_bidirectional() -> None:
    from src.orchestrator.handoff_contract import _normalize_contract_keys

    # Case 1: only has available_views & must_modify -> should map to dependencies & edit_targets
    contract_in1 = {
        "must_modify": [
            {
                "file": "app.py",
                "symbol_or_api": "query_method",
                "line_start": 45,
                "line_end": 55,
                "snippet": "SELECT 1",
                "decision": "use view_abc"
            }
        ],
        "available_views": [
            {
                "name": "view_abc",
                "columns": ["col1", "col2"]
            }
        ]
    }
    normalized1 = _normalize_contract_keys(contract_in1)
    assert "edit_targets" in normalized1
    assert normalized1["edit_targets"][0]["symbol"] == "query_method"
    assert "dependencies" in normalized1
    assert normalized1["dependencies"][0]["name"] == "view_abc"
    assert normalized1["dependencies"][0]["columns"] == ["col1", "col2"]


def test_normalize_contract_keys_ensures_required_fields() -> None:
    from src.orchestrator.handoff_contract import _normalize_contract_keys

    contract_in = {
        "target_view": "my_view",
        "resolved_dependencies": [
            {
                "kind": "database_view",
                "name": "my_view",
                "columns": ["col1"]
            }
        ]
    }
    normalized = _normalize_contract_keys(contract_in)
    assert normalized["target_view"] == "my_view"
    assert normalized["available_columns"] == ["col1"]
    assert "must_modify" in normalized
    assert "evidence" in normalized


def test_merge_handoff_contracts_scans_prior_summaries_target_view() -> None:
    from src.orchestrator.handoff_contract import merge_handoff_contracts

    prior_summaries = {
        "st-1": 'Handoff mentions we should use replacement_source: {"kind": "database_view", "name": "v_prior_view"} for replacement.'
    }
    merged = merge_handoff_contracts(
        user_request="Rewrite using view",
        prior_summaries=prior_summaries,
        current_search_output="",
    )
    assert merged["target_view"] == "v_prior_view"

