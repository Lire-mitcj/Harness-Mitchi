from __future__ import annotations

from src.agent.types import ToolResult
from src.hooks.after_tool import apply_after_tool_output_limit


def test_after_tool_output_limit_trims_output_and_observation_only() -> None:
    long_output = "A" * 80 + "B" * 80
    long_observation = "C" * 90 + "D" * 90
    raw_code = "def keep_me():\n    return 42\n"
    result = ToolResult(
        success=True,
        output=long_output,
        metadata={
            "llm_observation": long_observation,
            "verbatim_code": raw_code,
            "raw_evidence_store": [{"file": "x.py", "code": raw_code}],
        },
    )

    trimmed = apply_after_tool_output_limit("view_symbol_code", result, max_chars=100)

    assert len(trimmed.output) < len(long_output)
    assert "tool_output_trimmed" in trimmed.output
    assert len(trimmed.metadata["llm_observation"]) < len(long_observation)
    assert "tool_output_trimmed" in trimmed.metadata["llm_observation"]
    assert trimmed.metadata["verbatim_code"] == raw_code
    assert trimmed.metadata["raw_evidence_store"][0]["code"] == raw_code
    assert trimmed.metadata["tool_output_trim"]["tool"] == "view_symbol_code"


def test_after_tool_output_limit_returns_original_when_short() -> None:
    result = ToolResult(success=True, output="short", metadata={"llm_observation": "short"})

    trimmed = apply_after_tool_output_limit("codebase_retrieve", result, max_chars=100)

    assert trimmed is result
