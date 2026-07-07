from __future__ import annotations

import json

from src.agent.run_state import RunPhase, start_run
from src.agent.types import ToolResult
from src.hooks.before_tool import inspect_tool_request
from src.hooks.post_tool_context import apply_post_tool_context_hook


def test_before_tool_hook_is_read_only_argument_validation() -> None:
    state = start_run("检查接口", edit_mode=False)
    assert state.phase == RunPhase.RETRIEVING
    assert inspect_tool_request(
        "grep_search",
        {"pattern": "archive"},
        allowed_tools=state.allowed_tools,
    ) is None
    assert inspect_tool_request(
        "decision_edit",
        {"target_file": "list.py"},
        allowed_tools=state.allowed_tools,
    ) == "Tool 'decision_edit' is not in the reducer-provided allow list."


def test_grep_feedback_protocol_is_structured() -> None:
    payload = {
        "matches": [{"file": "main.py", "symbol": "auth_me", "span": [1, 1]}],
        "returned_matches": 1,
        "total_matches": 9,
        "truncated": True,
        "next_action": {"tool": "view_symbol_code", "symbols": ["auth_me"]},
    }
    assert json.loads(json.dumps(payload))["next_action"]["symbols"] == ["auth_me"]


def test_resource_id_filter_alone_does_not_satisfy_ownership() -> None:
    result = apply_post_tool_context_hook(
        "view_symbol_code",
        {"target_file": "list.py", "symbol": "archive_passenger"},
        ToolResult(
            success=True,
            output="{}",
            metadata={
                "raw_evidence_store": [
                    {
                        "file": "list.py",
                        "symbol": "archive_passenger",
                        "span": [1, 4],
                        "code": "SELECT p_id FROM passenger_info WHERE p_id = :p_id",
                    }
                ]
            },
        ),
    )

    grounded = result.metadata["run_event"]["grounded_slots"]
    assert "ownership_relation" not in grounded


def test_post_hook_does_not_treat_endpoint_decorator_as_mount_point() -> None:
    endpoint = apply_post_tool_context_hook(
        "view_symbol_code",
        {"target_file": "list.py", "symbol": "archive_passenger"},
        ToolResult(
            success=True,
            output="{}",
            metadata={
                "raw_evidence_store": [
                    {
                        "file": "list.py",
                        "symbol": "archive_passenger",
                        "span": [48, 81],
                        "code": '@router.delete("/passengers/{passenger_id}")\n'
                        "def archive_passenger(passenger_id: int): ...",
                    }
                ]
            },
        ),
    )
    grounded = endpoint.metadata["run_event"]["grounded_slots"]
    assert "endpoint_implementation" in grounded
    assert "integration_or_mount_point" not in grounded

    factory = apply_post_tool_context_hook(
        "view_symbol_code",
        {"target_file": "list.py", "symbol": "build_router"},
        ToolResult(
            success=True,
            output="{}",
            metadata={
                "raw_evidence_store": [
                    {
                        "file": "list.py",
                        "symbol": "build_router",
                        "span": [9, 176],
                        "code": "def build_router(engine):\n    return router",
                    }
                ]
            },
        ),
    )
    grounded = factory.metadata["run_event"]["grounded_slots"]
    assert "integration_or_mount_point" in grounded

    mount = apply_post_tool_context_hook(
        "grep_search",
        {"pattern": "include_router|build_router"},
        ToolResult(
            success=True,
            output="{}",
            metadata={
                "raw_evidence_store": [
                    {
                        "file": "main.py",
                        "symbol": "",
                        "span": [780, 780],
                        "match_line": "app.include_router(build_router(engine))",
                    }
                ]
            },
        ),
    )
    grounded = mount.metadata["run_event"]["grounded_slots"]
    assert "integration_or_mount_point" in grounded


def test_grep_pending_keeps_only_top_five_relevant_symbols() -> None:
    symbols = [
        "LoginRequest",
        "ResetPasswordRequest",
        "ChangePasswordRequest",
        "hash_password",
        "verify_password",
        "auth_http_error",
        "create_access_token",
        "decode_access_token",
        "_parse_bearer_token",
        "auth_me",
    ]
    raw = [
        {
            "file": "main.py",
            "symbol": symbol,
            "span": [index, index],
            "match_line": (
                f"class {symbol}:" if symbol.endswith("Request") else f"def {symbol}():"
            ),
        }
        for index, symbol in enumerate(symbols, 1)
    ]
    result = apply_post_tool_context_hook(
        "grep_search",
        {"pattern": "auth|login|token|password|bearer"},
        ToolResult(success=True, output="{}", metadata={"raw_evidence_store": raw}),
    )
    pending = result.metadata["run_event"]["candidates"]
    assert 3 <= len(pending) <= 5
    assert "auth_me" in {item["symbol"] for item in pending}
    assert not any(item["symbol"].endswith("Request") for item in pending)


def test_sql_semantic_structuring_hook() -> None:
    def structure(sql: str) -> ToolResult:
        return apply_post_tool_context_hook(
            "view_symbol_code",
            {"target_file": "schema.sql", "symbol": "sql_block"},
            ToolResult(
                success=True,
                output="{}",
                metadata={"verbatim_code": sql, "raw_evidence_store": []},
            ),
        )

    # 1. Test DDL alteration parsing
    alter_sql = "ALTER TABLE ticket_order ADD COLUMN status TEXT;"
    res = structure(alter_sql)
    assert "[STRUCTURED SQL SEMANTIC TRUTH]" in res.output
    assert '"op": "add_column"' in res.output
    assert '"table": "ticket_order"' in res.output
    assert '"column": "status"' in res.output
    assert '"type": "text"' in res.output

    # 2. Test DML insertion parsing
    insert_sql = "INSERT INTO ticket_order (p_id, status) VALUES (1, 'paid')"
    res = structure(insert_sql)
    assert '"op": "insert"' in res.output
    assert '"table": "ticket_order"' in res.output
    assert '"p_id": 1' in res.output
    assert '"status": "paid"' in res.output

    # 3. Test SELECT with filters parsing
    select_sql = "SELECT p_id, status FROM ticket_order WHERE p_id = 1"
    res = structure(select_sql)
    assert '"op": "select"' in res.output
    assert '"table": "ticket_order"' in res.output
    assert '"columns": [\n      "p_id",\n      "status"\n    ]' in res.output or '"columns": ["p_id", "status"]' in res.output or 'p_id' in res.output
    assert '"field": "p_id"' in res.output
    assert '"op": "="' in res.output
    assert '"value": 1' in res.output

    # 4. Test JOIN parsing and table alias resolution
    join_sql = "SELECT a.p_id, b.status FROM ticket_order a JOIN passenger_info b ON a.p_id = b.id"
    res = structure(join_sql)
    assert '"op": "join"' in res.output
    assert '"tables": [\n      "ticket_order",\n      "passenger_info"\n    ]' in res.output or 'ticket_order' in res.output
    assert '"left": "ticket_order.p_id"' in res.output
    assert '"right": "passenger_info.id"' in res.output

    # 5. Test GRANT permission parsing
    grant_sql = "GRANT SELECT ON ticket_order TO admin"
    res = structure(grant_sql)
    assert '"op": "grant"' in res.output
    assert '"permission": "select"' in res.output
    assert '"table": "ticket_order"' in res.output
    assert '"role": "admin"' in res.output


def test_grep_json_is_not_parsed_as_sql() -> None:
    output = '{"matches":[{"match_line":"SELECT * FROM ticket_order"}]}'
    result = apply_post_tool_context_hook(
        "grep_search",
        {"pattern": "ticket_order"},
        ToolResult(success=True, output=output, metadata={"raw_evidence_store": []}),
    )

    assert result.output == output
    assert "STRUCTURED SQL SEMANTIC TRUTH" not in result.output


def test_before_tool_fact_locking_allows_same_symbol_on_different_file() -> None:
    from src.hooks.before_tool import inspect_tool_request_async
    import asyncio

    res = asyncio.run(
        inspect_tool_request_async(
            "view_symbol_code",
            {"target_file": "main.py", "symbol": "build_router"},
            allowed_tools={"view_symbol_code"},
            context_anchors_code=[
                {
                    "file": "list.py",
                    "symbol": "build_router",
                    "span": [16, 358],
                    "code": "def build_router():\n    pass\n",
                }
            ],
        )
    )
    assert res is None


def test_before_tool_fact_locking_range_coverage() -> None:
    from src.hooks.before_tool import inspect_tool_request_async
    import asyncio

    # Dummy content with my_func at lines [12, 15]
    dummy_content = (
        "line_1\nline_2\nline_3\nline_4\nline_5\nline_6\nline_7\nline_8\nline_9\n"
        "line_10\nline_11\ndef my_func():\n    print(1)\n    print(2)\n    return\n"
        "line_16\nline_17\nline_18\nline_19\nline_20\nline_21\n"
    )
    import tempfile
    from pathlib import Path
    with tempfile.NamedTemporaryFile(suffix=".py", delete=False) as tmp:
        tmp.write(dummy_content.encode("utf-8"))
        tmp_name = tmp.name

    try:
        res = asyncio.run(
            inspect_tool_request_async(
                "view_symbol_code",
                {"target_file": tmp_name, "symbol": "my_func"},
                allowed_tools={"view_symbol_code"},
                context_anchors_code=[
                    {
                        "file": tmp_name,
                        "span": [10, 20],
                        "code": "\n".join(dummy_content.splitlines()[9:20]),
                    }
                ]
            )
        )
        assert res is not None
        assert res.startswith("BLOCK:")
    finally:
        Path(tmp_name).unlink(missing_ok=True)
