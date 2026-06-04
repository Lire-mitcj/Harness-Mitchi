from __future__ import annotations

from src.harness.discovery.input_parser import parse_turn_input
from src.harness.discovery.manifest import DiagnosticsManifest, discovery_display_summary, parse_manifest_json
from src.harness.discovery.manifest_gate import validate_manifest
from src.harness.gates.types import GateVerdict


def test_parse_turn_input_direct_plan() -> None:
    task, skip = parse_turn_input("/plan create hello.py with add(a,b)")
    assert skip is True
    assert task == "create hello.py with add(a,b)"


def test_parse_turn_input_normal() -> None:
    task, skip = parse_turn_input("fix API 500 error")
    assert skip is False
    assert task == "fix API 500 error"


def test_parse_manifest_json_from_fence() -> None:
    raw = """```json
{"root_cause": "missing table", "error_evidence": ["1146"], "victim_files": []}
```"""
    m = parse_manifest_json(raw, user_request="fix db")
    assert m.root_cause == "missing table"
    assert "1146" in m.error_evidence[0]


def test_parse_manifest_json_after_discovery_trace() -> None:
    import json

    from src.harness.discovery.manifest import extract_discovery_trace

    payload = {
        "root_cause": "missing table",
        "error_evidence": ["1146"],
        "victim_files": [{"path": "api/routes.py", "lines": [88], "note": "handler"}],
    }
    raw = (
        "<discovery_trace>\n"
        "1. Task type: bug_fix — missing table\n"
        "2. Search strategy: pytest then grep\n"
        "3. Evidence collected: error 1146\n"
        "4. Victim candidates: api/routes.py:88\n"
        "5. Stop check: enough for Planner\n"
        "</discovery_trace>\n"
        f"{json.dumps(payload)}"
    )
    trace = extract_discovery_trace(raw)
    assert trace is not None
    assert "bug_fix" in trace
    m = parse_manifest_json(raw, user_request="fix db")
    assert m.root_cause == "missing table"
    assert m.victim_files[0].path == "api/routes.py"
    assert 88 in m.victim_files[0].lines


def test_manifest_gate_warns_on_empty() -> None:
    m = DiagnosticsManifest(user_request="x")
    result = validate_manifest(m)
    assert result.verdict == GateVerdict.WARN


def test_manifest_gate_passes_with_signal() -> None:
    m = DiagnosticsManifest(
        user_request="x",
        root_cause="syntax error",
        victim_files=[],
    )
    result = validate_manifest(m)
    assert result.verdict == GateVerdict.PASS


def test_parse_manifest_json_malformed_trace_does_not_pollute_root_cause() -> None:
    raw = (
        '<discovery_trace">\n'
        '1. Task type: bug_fix"\n'
        '2. Search strategy: grep"\n'
        '3. Evidence collected: "数据库事务异常" in "app.py"\n'
        "4. Victim candidates: app.py lines 11\n"
    )
    m = parse_manifest_json(raw, user_request="fix registration")
    assert m.root_cause is None
    assert any("parse failed" in u.lower() for u in m.uncertainties)


def test_discovery_display_summary_skips_trace_garbage() -> None:
    m = DiagnosticsManifest(
        user_request="x",
        root_cause='<discovery_trace">\n1. Task type: bug_fix"',
        victim_files=[],
        scout_turns_used=2,
    )
    summary = discovery_display_summary(m)
    assert "discovery_trace" not in summary
    assert "Discovery complete (2 turns)" in summary


def test_discovery_display_summary_uses_victim_files() -> None:
    from src.harness.discovery.manifest import VictimFile

    m = DiagnosticsManifest(
        user_request="x",
        root_cause=None,
        victim_files=[VictimFile(path="app.py", lines=[11])],
        scout_turns_used=1,
    )
    summary = discovery_display_summary(m)
    assert "app.py" in summary


def test_skipped_manifest() -> None:
    m = DiagnosticsManifest.skipped_manifest("new file", "proj/\n  foo.py")
    assert m.skipped is True
    block = m.to_planner_block()
    assert "SKIPPED" in block
