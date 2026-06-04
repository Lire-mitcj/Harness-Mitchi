from __future__ import annotations

from src.harness.discovery.manifest import DiagnosticsManifest, VictimFile, manifest_actionable
from src.planner.kinds import SubTaskKind
from src.planner.scout_skip import apply_scout_discovery_to_plan
from src.planner.task_tree import SubTaskNode, TaskTree


def test_manifest_actionable_with_root_cause() -> None:
    m = DiagnosticsManifest(user_request="x", root_cause="missing commit in register route")
    assert manifest_actionable(m)


def test_manifest_actionable_with_victim_files() -> None:
    m = DiagnosticsManifest(
        user_request="x",
        victim_files=[VictimFile(path="main.py", lines=[42])],
    )
    assert manifest_actionable(m)


def test_apply_scout_skips_diagnose() -> None:
    manifest = DiagnosticsManifest(
        user_request="fix register",
        root_cause="register route missing commit/rollback",
        error_evidence=["transaction error on POST /register"],
        victim_files=[VictimFile(path="main.py", lines=[120], note="register handler")],
        scout_turns_used=2,
    )
    tree = TaskTree(
        root_task="fix",
        nodes=[
            SubTaskNode(id="st-1", description="find bug", kind=SubTaskKind.DIAGNOSE),
            SubTaskNode(
                id="st-2",
                description="fix",
                kind=SubTaskKind.EDIT,
                depends_on=["st-1"],
                context_files=["main.py"],
            ),
        ],
    )
    updated, summaries = apply_scout_discovery_to_plan(tree, manifest)
    assert updated.get("st-1").status.value == "SUCCESS"
    assert "st-1" in summaries
    assert updated.get("st-2").status.value == "PENDING"
