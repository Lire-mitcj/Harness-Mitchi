"""Harness discovery (Scout) phase — read-only project context before Planner."""

from src.harness.discovery.input_parser import parse_turn_input
from src.harness.discovery.manifest import DiagnosticsManifest, parse_manifest_json
from src.harness.discovery.manifest_gate import validate_manifest
from src.harness.discovery.scout_agent import ScoutAgent

__all__ = [
    "DiagnosticsManifest",
    "ScoutAgent",
    "parse_manifest_json",
    "parse_turn_input",
    "validate_manifest",
]
