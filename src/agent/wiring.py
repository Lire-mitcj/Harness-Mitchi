"""Language-aware integration / entry-point wiring heuristics for manifest gates."""

from __future__ import annotations

import re
from pathlib import Path

from src.agent.evidence_slots import slots_for_task, themes_for_task
from src.indexer.language_profiles import (
    GO,
    JAVA,
    PYTHON,
    LanguageProfile,
    profile_for_path,
)

_MIN_WIRING_HEADER_LINES = 12

_ENTRY_FILE_NAMES = frozenset({
    "main.py",
    "__main__.py",
    "main.go",
    "Main.java",
    "Application.java",
})

_ENTRY_PATH_SEGMENTS = frozenset({
    "cmd",
    "main",
    "app",
    "bootstrap",
    "entry",
})

_INFRA_PATH_SEGMENTS = frozenset({
    "infrastructure",
    "conf",
    "config",
    "internal",
    "domain",
    "pkg",
})

_HANDLER_PATH_SEGMENTS = frozenset({
    "interfaces",
    "handlers",
    "handler",
    "api",
    "router",
    "controller",
    "service",
    "server",
})

_WIRING_PROBE_SYMBOLS: dict[str, tuple[str, ...]] = {
    "python": ("create_app", "app", "wire_routes", "application"),
    "go": ("main", "Run", "NewServer", "Serve", "Start"),
    "java": ("Application", "main", "SpringBootApplication"),
    "proto": (),
    "sql": (),
}

_TASK_WIRING_OPT_OUT = (
    "noise_policy",
    "bot_nickname",
    "bot nickname",
    "mention",
    "艾特",
    "昵称",
    "yaml",
    ".yaml",
    ".yml",
    ".proto",
    "配置文件",
)

_TASK_WIRING_OPT_IN = (
    "include_router",
    "main.py",
    "挂载",
    "接入",
    "caller",
    "integration",
    "integrate",
    "mount point",
)


def wiring_probe_symbols(file_path: str) -> tuple[str, ...]:
    profile = profile_for_path(file_path)
    if profile is None:
        return ()
    if profile.wiring_probe_symbols:
        return profile.wiring_probe_symbols
    return _WIRING_PROBE_SYMBOLS.get(profile.id, ())


def grep_patterns_for_wiring(file_path: str) -> tuple[str, ...]:
    profile = profile_for_path(file_path)
    if profile is None:
        return ()
    return profile.discovery_patterns[:6]


def mount_setup_confirmed(code: str, *, file_path: str = "") -> bool:
    """True when loaded source shows an integration/entry registration region."""
    if not code.strip():
        return False
    profile = profile_for_path(file_path) if file_path else None
    if profile is not None and profile.mount_line_res:
        if any(pattern.search(code) for pattern in profile.mount_line_res):
            return True
    if profile is None or profile.id == "python":
        return _python_mount_fallback(code)
    return False


def _python_mount_fallback(code: str) -> bool:
    lowered = code.casefold()
    if any(marker in lowered for marker in ("include_router(", "app.include_router")):
        return True
    if any(
        ("build_router(engine" in line or "build_router(app" in line)
        and not line.strip().startswith(("def ", "async def "))
        for line in lowered.splitlines()
    ):
        return True
    return "fastapi(" in lowered and (
        "include_router" in lowered or re.search(r"\bapp\s*=", lowered) is not None
    )


def task_needs_integration_wiring(task_text: str) -> bool:
    """Whether cross-file entry/caller wiring checks apply to this task."""
    lowered = task_text.casefold().strip()
    if not lowered:
        return True
    if any(token in lowered for token in _TASK_WIRING_OPT_OUT):
        return False
    slots = slots_for_task(task_text)
    if "integration_or_mount_point" in slots:
        return True
    themes = themes_for_task(task_text)
    if themes & {"endpoint", "db_integration"}:
        return True
    if "exception_handler" in themes and any(
        token in lowered for token in _TASK_WIRING_OPT_IN
    ):
        return True
    if any(token in lowered for token in _TASK_WIRING_OPT_IN):
        return True
    return False


def integration_layer(file_path: str) -> str:
    """Rough file layer used to avoid infra↔infra false wiring gaps."""
    norm = file_path.replace("\\", "/").casefold().strip("./")
    name = norm.rsplit("/", 1)[-1]
    parts = set(norm.split("/"))
    if name in {n.casefold() for n in _ENTRY_FILE_NAMES}:
        return "entry"
    if parts & {seg.casefold() for seg in _HANDLER_PATH_SEGMENTS}:
        return "handler"
    if parts & {seg.casefold() for seg in _INFRA_PATH_SEGMENTS}:
        return "infrastructure"
    if parts & {seg.casefold() for seg in _ENTRY_PATH_SEGMENTS}:
        return "entry"
    return "other"


def is_entry_file(file_path: str) -> bool:
    return integration_layer(file_path) == "entry"


def peer_layers_skip_wiring(file_path: str) -> bool:
    """Infrastructure/config peers do not require FastAPI-style caller wiring."""
    return integration_layer(file_path) in {"infrastructure", "other"}


def caller_has_setup_evidence(
    items: tuple[object, ...],
    *,
    file_path: str = "",
) -> bool:
    """True when a support file already has verified entry/integration context."""
    for item in items:
        keywords = getattr(item, "keywords", ()) or ()
        if "mount_confirmed" in keywords:
            return True
    if file_path and is_entry_file(file_path):
        for item in items:
            span = getattr(item, "span", None)
            if span is not None and span[1] - span[0] + 1 >= _MIN_WIRING_HEADER_LINES:
                symbol = str(getattr(item, "symbol", "") or "")
                if symbol in wiring_probe_symbols(file_path):
                    return False
                return True
    return False


def wiring_gap_message(file_path: str) -> str:
    profile = profile_for_path(file_path)
    if profile is GO:
        return (
            f"{file_path}: entry/integration region not loaded "
            "(view package main, http.Handle, or grpc Register* wiring first)"
        )
    if profile is JAVA:
        return (
            f"{file_path}: application wiring not loaded "
            "(view @SpringBootApplication / @RestController / @Bean setup first)"
        )
    if profile is PYTHON:
        return (
            f"{file_path}: app/init/mount region not loaded "
            "(view include_router, app setup, or create_app first)"
        )
    return (
        f"{file_path}: integration/bootstrap region not loaded "
        "(view entry-point registration for this file's language first)"
    )


def default_wiring_probe_symbol(file_path: str) -> str:
    probes = wiring_probe_symbols(file_path)
    if probes:
        return probes[0]
    profile = profile_for_path(file_path)
    if profile is PYTHON:
        return "create_app"
    if profile is GO:
        return "main"
    if profile is JAVA:
        return "Application"
    return Path(file_path).stem or "main"
