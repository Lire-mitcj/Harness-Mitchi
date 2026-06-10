from __future__ import annotations

from src.harness.gates.exit_gate import ExitCheckInput, validate_exit
from src.harness.gates.types import GateVerdict
from src.planner.task_tree import SubTaskKind, SubTaskNode


def test_exit_gate_blocks_empty_answer() -> None:
    node = SubTaskNode(id="st-1", description="read api", kind=SubTaskKind.DIAGNOSE)
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="",
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_blocks_edit_without_changes() -> None:
    node = SubTaskNode(
        id="st-1",
        description="patch handler",
        kind=SubTaskKind.EDIT,
        context_files=["api.py"],
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="Done editing the handler.",
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_passes_diagnose_with_summary() -> None:
    node = SubTaskNode(id="st-1", description="inspect schema", kind=SubTaskKind.DIAGNOSE)
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="Table user_account has columns id, username, password_hash.",
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.PASS


def test_exit_gate_passes_edit_with_changes() -> None:
    node = SubTaskNode(
        id="st-1",
        description="fix api",
        kind=SubTaskKind.EDIT,
        context_files=["api.py"],
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message="Updated login handler to validate password hash.",
            error_trace=[],
            changed_files=["api.py"],
        )
    )
    assert result.verdict == GateVerdict.PASS


def test_exit_gate_blocks_incomplete_final_json_schema() -> None:
    node = SubTaskNode(
        id="st-1",
        description="fix api",
        kind=SubTaskKind.EDIT,
        context_files=["api.py"],
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message='{"result":"updated"}',
            error_trace=[],
            changed_files=["api.py"],
        )
    )

    assert result.verdict == GateVerdict.BLOCK
    assert "missing required key" in "; ".join(result.messages)


def test_exit_gate_passes_agent_output_schema() -> None:
    node = SubTaskNode(
        id="st-1",
        description="fix api",
        kind=SubTaskKind.EDIT,
        context_files=["api.py"],
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                '{"status":"success","changed_files":["api.py"],'
                '"validation":{"ran":["pytest"],"result":"passed","summary":"ok"},'
                '"risks":[],"handoff":{"facts":["fixed"],"evidence":[],'
                '"known_negatives":[],"next_focus":[]}}'
            ),
            error_trace=[],
            changed_files=["api.py"],
        )
    )

    assert result.verdict == GateVerdict.PASS


def test_exit_gate_blocks_incomplete_agent_output_schema() -> None:
    node = SubTaskNode(id="st-1", description="verify", kind=SubTaskKind.VERIFY)
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message='{"status":"success","handoff":{}}',
            error_trace=[],
            changed_files=[],
        )
    )

    assert result.verdict == GateVerdict.BLOCK
    assert "changed_files" in "; ".join(result.messages)


def test_exit_gate_blocks_diagnose_acceptance_unmet() -> None:
    node = SubTaskNode(
        id="st-1",
        description="find sql",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="List file paths and symbol names",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "Could not find the boarding pass SQL query in the project. "
                "Acceptance criteria not met."
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_blocks_diagnose_not_yet_met() -> None:
    node = SubTaskNode(
        id="st-2",
        description="locate query",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Identify file and line with current select query",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "Acceptance criteria not yet met — the exact file and line "
                "have not been identified due to file truncation."
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_blocks_diagnose_missing_required_file_line() -> None:
    node = SubTaskNode(
        id="st-1",
        description="locate query",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "Result: found the query handler.\n"
                "Evidence: query_orders function handles the endpoint.\n"
                "Conclusion: use this for edit."
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK
    assert "file:line" in "; ".join(result.messages)


def test_exit_gate_passes_diagnose_with_required_handoff_evidence() -> None:
    node = SubTaskNode(
        id="st-1",
        description="locate query",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Output file:line, symbol, and snippet/decision",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "Result: found target.\n"
                "Evidence: main.py:1626 function query_orders contains SQL snippet "
                "SELECT * FROM orders.\n"
                "Conclusion: decision is to edit query_orders."
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.PASS


def test_exit_gate_passes_chinese_diagnose_handoff_evidence() -> None:
    node = SubTaskNode(
        id="st-1",
        description="定位查询接口",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="输出文件:行号、符号和代码片段/决策",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "结果: 已定位目标。\n"
                "证据: main.py:1626 函数 query_orders 包含 SQL 代码片段 "
                "SELECT * FROM orders。\n"
                "结论: 决策是编辑 query_orders。"
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.PASS


def test_exit_gate_passes_structured_diagnose_handoff_evidence() -> None:
    node = SubTaskNode(
        id="st-1",
        description="定位查询接口",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="输出文件:行号、符号和代码片段/决策",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                '{"result":"已定位目标","acceptance_met":true,'
                '"evidence":[{"path":"main.py","line":1626,'
                '"symbol":"query_orders","snippet":"SELECT * FROM orders"}],'
                '"blocker":""}'
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.PASS


def test_exit_gate_blocks_structured_diagnose_acceptance_false() -> None:
    node = SubTaskNode(
        id="st-1",
        description="定位查询接口",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="输出文件:行号、符号和代码片段/决策",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                '{"result":"未定位目标","acceptance_met":false,'
                '"evidence":[],"blocker":"没有命中"}'
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_blocks_llm_transport_error_summary() -> None:
    node = SubTaskNode(
        id="st-1",
        description="diagnose",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Output file:line and decision",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "Tool 'llm_call' failed with MidStreamFallbackError: "
                "litellm.APIConnectionError"
            ),
            error_trace=[],
            changed_files=[],
            turns_used=1,
        )
    )
    assert result.verdict == GateVerdict.BLOCK


def test_exit_gate_blocks_diagnose_when_handoff_missing_required_fields() -> None:
    node = SubTaskNode(
        id="st-1",
        description="diagnose query location",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Output HANDOFF_CONTRACT_JSON with must_modify evidence.",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "HANDOFF_CONTRACT_JSON\n"
                "{\n"
                '  "must_modify": [\n'
                "    {\n"
                '      "file": "main.py",\n'
                '      "line": 10\n'
                "    }\n"
                "  ],\n"
                '  "available_views": [],\n'
                '  "evidence": []\n'
                "}"
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK
    assert "HANDOFF_CONTRACT_JSON 'must_modify[0]' is missing required key(s)" in "; ".join(result.messages)


def test_exit_gate_blocks_design_when_patch_intent_missing_required_fields() -> None:
    node = SubTaskNode(
        id="st-2",
        description="produce design",
        kind=SubTaskKind.DESIGN,
        acceptance_criteria="Output PATCH_INTENT_JSON detailing strategy and targets.",
    )
    result = validate_exit(
        ExitCheckInput(
            subtask=node,
            final_message=(
                "PATCH_INTENT_JSON\n"
                "{\n"
                '  "edit_strategy": "sql_view_rewrite",\n'
                '  "edit_ready": true,\n'
                '  "edit_targets": [\n'
                "    {\n"
                '      "file": "main.py"\n'
                "    }\n"
                "  ],\n"
                '  "dependencies": [],\n'
                '  "acceptance_criteria": [],\n'
                '  "target_view": "view_ticket_report_detail"\n'
                "}"
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert result.verdict == GateVerdict.BLOCK
    assert "PATCH_INTENT_JSON 'edit_targets[0]' is missing required key(s)" in "; ".join(result.messages)


def test_exit_gate_passes_diagnose_and_design_with_valid_handoff_and_patch_intent() -> None:
    node_diag = SubTaskNode(
        id="st-1",
        description="diagnose query location",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Output HANDOFF_CONTRACT_JSON with must_modify evidence.",
    )
    res_diag = validate_exit(
        ExitCheckInput(
            subtask=node_diag,
            final_message=(
                "HANDOFF_CONTRACT_JSON\n"
                "{\n"
                '  "must_modify": [\n'
                "    {\n"
                '      "file": "main.py",\n'
                '      "line": 10,\n'
                '      "symbol_or_api": "query_func",\n'
                '      "should_change_to": "use view"\n'
                "    }\n"
                "  ],\n"
                '  "available_views": [],\n'
                '  "evidence": []\n'
                "}"
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert res_diag.verdict == GateVerdict.PASS

    node_design = SubTaskNode(
        id="st-2",
        description="produce design",
        kind=SubTaskKind.DESIGN,
        acceptance_criteria="Output PATCH_INTENT_JSON detailing strategy and targets.",
    )
    res_design = validate_exit(
        ExitCheckInput(
            subtask=node_design,
            final_message=(
                "PATCH_INTENT_JSON\n"
                "{\n"
                '  "edit_strategy": "sql_view_rewrite",\n'
                '  "edit_targets": [\n'
                "    {\n"
                '      "file": "main.py",\n'
                '      "symbol": "query_func",\n'
                '      "line_start": 10,\n'
                '      "line_end": 15,\n'
                '      "snippet": "SELECT ...",\n'
                '      "decision": "use view"\n'
                "    }\n"
                "  ],\n"
                '  "edit_ready": true,\n'
                '  "dependencies": [],\n'
                '  "acceptance_criteria": ["test passes"],\n'
                '  "target_view": "view_ticket_report_detail"\n'
                "}"
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert res_design.verdict == GateVerdict.PASS


def test_exit_gate_passes_diagnose_with_trailing_content() -> None:
    from src.harness.gates.exit_gate import validate_exit, ExitCheckInput
    from src.harness.gates.types import GateVerdict
    
    node_diag = SubTaskNode(
        id="st-1",
        description="diagnose query location",
        kind=SubTaskKind.DIAGNOSE,
        acceptance_criteria="Output HANDOFF_CONTRACT_JSON with must_modify evidence.",
    )
    res_diag = validate_exit(
        ExitCheckInput(
            subtask=node_diag,
            final_message=(
                "HANDOFF_CONTRACT_JSON\n"
                "{\n"
                '  "must_modify": [\n'
                "    {\n"
                '      "file": "main.py",\n'
                '      "line": 10,\n'
                '      "symbol_or_api": "query_func",\n'
                '      "should_change_to": "use view"\n'
                "    }\n"
                "  ],\n"
                '  "available_views": [],\n'
                '  "evidence": []\n'
                "}\n"
                "\n"
                "Trailing notes or markdown comments with braces like {}."
            ),
            error_trace=[],
            changed_files=[],
        )
    )
    assert res_diag.verdict == GateVerdict.PASS


def test_normalize_contract_keys_bidirectional_exit_gate() -> None:
    from src.harness.gates.exit_gate import _normalize_contract_keys

    contract_in = {
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
    normalized = _normalize_contract_keys(contract_in)
    assert "edit_targets" in normalized
    assert normalized["edit_targets"][0]["symbol"] == "query_method"
    assert "dependencies" in normalized
    assert normalized["dependencies"][0]["name"] == "view_abc"
    assert normalized["dependencies"][0]["columns"] == ["col1", "col2"]

