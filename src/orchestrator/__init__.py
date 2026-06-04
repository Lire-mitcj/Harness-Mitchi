"""Planner-driven ReAct orchestration layer."""

__all__ = [
    "EvidencePack",
    "OrchestratorLoop",
    "OrchestratorState",
    "plan_update_event",
]


def __getattr__(name: str):
    if name == "EvidencePack":
        from src.orchestrator.evidence import EvidencePack

        return EvidencePack
    if name in {"OrchestratorLoop", "OrchestratorState", "plan_update_event"}:
        from src.orchestrator.orchestrator import (
            OrchestratorLoop,
            OrchestratorState,
            plan_update_event,
        )

        return {
            "OrchestratorLoop": OrchestratorLoop,
            "OrchestratorState": OrchestratorState,
            "plan_update_event": plan_update_event,
        }[name]
    raise AttributeError(name)
