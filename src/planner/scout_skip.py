from __future__ import annotations

from src.harness.discovery.manifest import DiagnosticsManifest, manifest_actionable
from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskStatus, TaskTree


def manifest_executor_summary(manifest: DiagnosticsManifest) -> str:
    """Text injected when diagnose subtasks are auto-skipped."""
    lines = ["## Scout discovery (auto-applied — do not re-explore)"]
    if manifest.root_cause:
        lines.append(f"Root cause: {manifest.root_cause}")
    if manifest.error_evidence:
        lines.append("Evidence:")
        lines.extend(f"- {e}" for e in manifest.error_evidence[:5])
    if manifest.victim_files:
        lines.append("Victim files:")
        for v in manifest.victim_files[:3]:
            loc = f" lines {v.lines}" if v.lines else ""
            note = f" — {v.note}" if v.note else ""
            lines.append(f"- {v.path}{loc}{note}")
    if manifest.file_snippets:
        for snip in manifest.file_snippets[:2]:
            if snip.path == "__project_structure__":
                continue
            body = snip.content
            if len(body) > 800:
                body = body[:800] + "\n[truncated]"
            lines.append(f"\n### {snip.path}\n{body}")
    return "\n".join(lines)


def apply_scout_discovery_to_plan(
    task_tree: TaskTree,
    manifest: DiagnosticsManifest,
) -> tuple[TaskTree, dict[str, str]]:
    """Mark diagnose subtasks SUCCESS when Scout already found root cause."""
    summaries: dict[str, str] = {}
    if not manifest_actionable(manifest):
        return task_tree, summaries

    summary = manifest_executor_summary(manifest)
    victim_paths = [v.path for v in manifest.victim_files if v.path]

    for node in task_tree.nodes:
        if node.kind != SubTaskKind.DIAGNOSE:
            continue
        if node.status != SubTaskStatus.PENDING:
            continue
        task_tree.mark_success(node.id)
        summaries[node.id] = summary

    if victim_paths:
        from src.planner.context_policy import enrich_task_tree_context_files

        enrich_task_tree_context_files(task_tree)

    return task_tree, summaries
