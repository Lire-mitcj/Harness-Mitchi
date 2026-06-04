from __future__ import annotations

from src.harness.discovery.manifest import DiagnosticsManifest
from src.harness.gates.types import GateResult


def validate_manifest(manifest: DiagnosticsManifest) -> GateResult:
    """Lightweight gate for Scout output before waking Planner."""
    if manifest.skipped:
        return GateResult.pass_("manifest_gate", skipped=True)

    has_signal = bool(
        manifest.root_cause
        or manifest.error_evidence
        or manifest.victim_files
        or manifest.file_snippets
    )
    if not has_signal:
        return GateResult.warn(
            "manifest_gate",
            [
                "Scout produced no root cause, evidence, or file snippets — "
                "Planner will rely on project structure only."
            ],
        )
    return GateResult.pass_(
        "manifest_gate",
        victim_count=len(manifest.victim_files),
        evidence_count=len(manifest.error_evidence),
    )
