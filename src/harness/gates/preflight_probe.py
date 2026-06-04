from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from src.config.settings import MitKIISettings
from src.harness.gates.preflight_slices import resolve_repo_map_line_slices
from src.harness.gates.types import GateVerdict, PreflightResult, TruncationPolicy
from src.harness.probe.token_budget import TokenBudget
from src.harness.subtask.prompt_builder import estimate_executor_prompt_tokens
from src.planner.context_policy import effective_context_files
from src.planner.task_tree import SubTaskNode, TaskTree

if TYPE_CHECKING:
    from src.indexer.repo_map import RepoMap

_TOOLS_SCHEMA_TOKENS = 3_500


def assess_preflight(
    *,
    subtask: SubTaskNode,
    task_tree: TaskTree,
    project_root: Path,
    settings: MitKIISettings,
    tools_schema_tokens: int = _TOOLS_SCHEMA_TOKENS,
    repo_map: RepoMap | None = None,
) -> PreflightResult:
    """Estimate Executor prompt size and choose a truncation tier."""
    budget = settings.context_budget
    yellow_threshold = int(budget * settings.preflight_yellow_ratio)
    red_threshold = int(budget * settings.preflight_red_ratio)

    counter = TokenBudget()
    base_tokens = estimate_executor_prompt_tokens(
        counter,
        root_task=task_tree.root_task,
        task_tree=task_tree,
        subtask=subtask,
        project_root=project_root,
        policy=TruncationPolicy.green(settings.preflight_max_chars_per_file),
    )
    estimated = base_tokens + tools_schema_tokens + settings.preflight_turn_reserve_tokens

    messages: list[str] = []
    context_files = effective_context_files(task_tree, subtask)
    large_files = _large_file_paths(project_root, context_files, settings)

    if estimated <= yellow_threshold and not large_files:
        return PreflightResult(
            passed=True,
            verdict=GateVerdict.PASS,
            policy=TruncationPolicy.green(settings.preflight_max_chars_per_file),
            estimated_tokens=estimated,
            budget_tokens=budget,
        )

    if large_files or estimated > yellow_threshold:
        max_chars = min(6_000, settings.preflight_max_chars_per_file)
        slice_targets = large_files or context_files
        line_slices: dict[str, tuple[int, int]] = {}
        if repo_map is not None and slice_targets:
            line_slices = resolve_repo_map_line_slices(
                subtask=subtask,
                task_tree=task_tree,
                context_files=context_files,
                repo_map=repo_map,
                settings=settings,
                target_files=slice_targets,
            )

        if line_slices:
            yellow_policy = TruncationPolicy(
                tier="yellow",
                max_chars_per_file=max_chars,
                line_slices=line_slices,
            )
        else:
            yellow_policy = TruncationPolicy.yellow(max_chars=max_chars)

        yellow_est = (
            estimate_executor_prompt_tokens(
                counter,
                root_task=task_tree.root_task,
                task_tree=task_tree,
                subtask=subtask,
                project_root=project_root,
                policy=yellow_policy,
            )
            + tools_schema_tokens
            + settings.preflight_turn_reserve_tokens
        )
        if yellow_est <= red_threshold:
            if line_slices:
                preview = ", ".join(
                    f"{p}:L{start}-{end}" for p, (start, end) in line_slices.items()
                )
                messages.append(f"Repo map slice preload: {preview}")
            elif large_files:
                messages.append(f"Large context files truncated: {', '.join(large_files)}")
            if estimated > yellow_threshold and not line_slices:
                messages.append(
                    f"Estimated {estimated} tokens > yellow threshold {yellow_threshold}; "
                    "using head/tail truncation."
                )
            return PreflightResult(
                passed=True,
                verdict=GateVerdict.WARN,
                policy=yellow_policy,
                estimated_tokens=yellow_est,
                budget_tokens=budget,
                messages=messages,
            )

    return PreflightResult(
        passed=False,
        verdict=GateVerdict.BLOCK,
        policy=TruncationPolicy.red_fallback(),
        estimated_tokens=estimated,
        budget_tokens=budget,
        messages=[
            f"Estimated {estimated} tokens exceeds red threshold {red_threshold} "
            f"(budget {budget}).",
            "Planner should reduce context_files or split the subtask.",
        ],
        skip_preload=True,
    )


def _large_file_paths(
    project_root: Path,
    context_files: list[str],
    settings: MitKIISettings,
) -> list[str]:
    limit = settings.preflight_large_file_bytes
    large: list[str] = []
    root = project_root.resolve()
    for rel in context_files:
        try:
            path = (root / rel.replace("\\", "/").lstrip("./")).resolve()
            if path.is_file() and path.stat().st_size > limit:
                large.append(rel)
        except OSError:
            continue
    return large
