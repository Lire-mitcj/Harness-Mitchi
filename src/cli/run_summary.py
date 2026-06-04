from __future__ import annotations

from src.cli.report_format import milestone_from_report
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree

_STATUS_ICON = {
    SubTaskStatus.PENDING: "○",
    SubTaskStatus.RUNNING: "●",
    SubTaskStatus.SUCCESS: "✓",
    SubTaskStatus.FAILED: "✗",
}


def build_terminal_run_summary(
    task_tree: TaskTree,
    summaries: dict[str, str],
) -> str:
    """Compact end-of-run summary — no duplicate markdown reports."""
    done = sum(1 for n in task_tree.nodes if n.status == SubTaskStatus.SUCCESS)
    total = len(task_tree.nodes)
    lines = [
        f"Done ({done}/{total} steps)",
        "",
        f"Task: {task_tree.root_task}",
    ]
    if task_tree.has_pending():
        lines.extend(["", "Some steps did not complete."])
    lines.extend(["", "Steps:"])
    for node in task_tree.nodes:
        lines.extend(_format_step_line(node, summaries))
    return "\n".join(lines).rstrip()


def _format_step_line(node: SubTaskNode, summaries: dict[str, str]) -> list[str]:
    icon = _STATUS_ICON.get(node.status, "?")
    out = [f"  {icon} [{node.id}] {node.description}"]
    summary = summaries.get(node.id, "").strip()
    if summary and node.status == SubTaskStatus.SUCCESS:
        hint = milestone_from_report(summary, max_len=100)
        if hint:
            out.append(f"      → {hint}")
    return out
