from __future__ import annotations

from pathlib import Path

from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, TaskTree


def test_rebuild_retry_uses_paths_only_not_full_files(tmp_path: Path) -> None:

    from src.orchestrator.isolation import rebuild_executor_retry_messages
    main = tmp_path / "main.py"
    main.write_text("x = 1\n" * 5000)
    tree = TaskTree(
        root_task="fix query",
        nodes=[
            SubTaskNode(
                id="st-2",
                kind=SubTaskKind.EDIT,
                description="update select",
                context_files=["main.py"],
            ),
        ],
    )
    messages = rebuild_executor_retry_messages(
        root_task="fix query",
        task_tree=tree,
        subtask=tree.nodes[0],
        project_root=tmp_path,
        prior_summaries=None,
        error_trace=["write_file blocked: /tmp/note.txt"],
        context_files=["main.py"],
    )
    system = "\n".join(m.content or "" for m in messages if m.role == "system")
    assert '"preload_mode": "paths_only"' in system
    assert '\n<file path="main.py">' not in system
    assert "write_file blocked" in (messages[-1].content or "")
