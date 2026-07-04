from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.agent.decision import CursorDecisionLLM, DecisionError
from src.tools.assembled.decision_edit import DecisionEditTool


def test_decision_edit_only_mode_rejects_ask_clarify() -> None:
    with pytest.raises(DecisionError, match="edit-only mode forbids action 'ask_clarify'"):
        CursorDecisionLLM.parse(
            '{"action":"ask_clarify","answer":"","clarification":"need more context",'
            '"target_file":"","patch":"","suggested_completion":0}',
            ("db/init/init.sql",),
            edit_only=True,
        )


def test_decision_edit_only_prompt_mentions_edit_only_mode() -> None:
    messages = CursorDecisionLLM(MagicMock()).build_messages(
        state_text="apply change",
        context_pack=MagicMock(windows=(), candidate_files=()),
        hint=None,
        edit_only=True,
    )

    assert "EDIT_ONLY_MODE" in messages[0]["content"]
    assert '"action":"edit"' in messages[0]["content"]


def test_decision_edit_context_pack_includes_all_target_file_spans(tmp_path: Path) -> None:
    sql_path = tmp_path / "db" / "init" / "init.sql"
    sql_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"line {index}" for index in range(1, 401)]
    lines[142] = "CREATE TABLE order_timeline ("
    lines[327] = "CREATE TRIGGER tg_release_seat"
    sql_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    tool = DecisionEditTool(
        project_root=tmp_path,
        settings=MagicMock(),
        decision_llm=MagicMock(),
        harness=MagicMock(project_root=tmp_path),
    )

    context_pack = tool._build_context_pack(
        "./db/init/init.sql",
        context_window=[
            {"file": "./db/init/init.sql", "span": [143, 155]},
            {"file": "./db/init/init.sql", "span": [311, 330]},
        ],
    )

    target_windows = [window for window in context_pack.windows if window.file.endswith("init.sql")]
    assert len(target_windows) == 2
    assert target_windows[0].role == "target"
    assert target_windows[1].role == "reference"
    assert "order_timeline" in target_windows[0].content
    assert "tg_release_seat" in target_windows[1].content
