from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.cli.commands.chat import AgentLoopAdapter
from src.config.settings import get_settings
from src.runtime.session_factory import create_mitkii_session
from src.server.handler import ServerHandler
from src.server.stdio_transport import StdioTransport

log = logging.getLogger(__name__)


def run_server(project_path: Path | None = None) -> None:
    """Start JSON-RPC server on stdio with the same bootstrap as ``mitkii chat``."""
    settings = get_settings()
    settings.ensure_dirs()
    session = create_mitkii_session(project_root=project_path)
    core_loop = session.create_core_loop()
    agent = AgentLoopAdapter(core_loop)
    handler = ServerHandler(agent)
    transport = StdioTransport(handler)
    log.info(
        "MitKII server starting (orchestrator=%s, repo_map=%s)",
        settings.orchestrator_mode,
        settings.repo_map_enabled,
    )
    log.info("MitKII project root: %s", session.project_root)
    try:
        asyncio.run(transport.start())
    except KeyboardInterrupt:
        log.info("Server stopped")
