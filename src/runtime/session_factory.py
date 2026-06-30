from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from src.config.permissions import PermissionManager
from src.config.settings import MitKIISettings, get_settings
from src.context.builder import ContextBuilder
from src.context.file_tracker import FileTracker
from src.harness.engine import HarnessEngine
from src.indexer.repo_map_service import RepoMapService
from src.llm.client import LLMClient
from src.tools.registry import ToolRegistry, create_default_registry

log = logging.getLogger(__name__)


@dataclass
class MitKIISession:
    """Shared runtime bundle for CLI, server, and SDK entry points."""

    project_root: Path
    settings: MitKIISettings
    file_tracker: FileTracker
    context_builder: ContextBuilder
    repo_map_service: RepoMapService | None
    tools: ToolRegistry
    harness: HarnessEngine
    permissions: PermissionManager
    llm: LLMClient
    cursor_inter_llm: LLMClient
    cursor_decision_llm: LLMClient

    def create_core_loop(self) -> Any:
        """Create the core loop driver."""
        from src.agent.state_assembled_loop import StateAssembledLoop
        return StateAssembledLoop(
            llm=self.llm,
            tools=self.tools,
            harness=self.harness,
            context=self.context_builder,
            permissions=self.permissions,
            settings=self.settings,
        )


def create_mitkii_session(
    project_root: Path | None = None,
    *,
    settings: MitKIISettings | None = None,
) -> MitKIISession:
    """Bootstrap MitKII with repo map background build and shared context."""
    settings = settings or get_settings()
    settings.ensure_dirs()
    root = (project_root or settings.project_root or Path.cwd()).resolve()

    file_tracker = FileTracker()
    repo_map_service: RepoMapService | None = None
    if settings.repo_map_enabled:
        repo_map_service = RepoMapService(
            root,
            enabled=True,
            top_k=settings.repo_map_top_k,
            settings=settings,
        )
        repo_map_service.start_background_build()
        log.info("Repo map building in background (thread: mitkii-repo-map)")

    context_builder = ContextBuilder(
        project_root=root,
        file_tracker=file_tracker,
        repo_map_service=repo_map_service,
        repo_map_max_chars=settings.repo_map_max_chars,
    )
    if repo_map_service is not None:
        def _on_file_change(path: str, _action: str) -> None:
            repo_map_service.mark_dirty(path)
            context_builder.invalidate_project_context()

        file_tracker.set_change_callback(_on_file_change)
    tools = create_default_registry(project_root=root, repo_map_service=repo_map_service)
    harness = HarnessEngine.create(settings, project_root=root)
    permissions = PermissionManager()
    cache_ttl = "1h" if settings.prompt_cache_ttl.strip().lower() in {"1h", "hour", "60m"} else "5m"
    def create_llm(model: str) -> LLMClient:
        return LLMClient(
            model=model,
            request_timeout=float(settings.llm_request_timeout),
            stream_idle_timeout=float(settings.llm_stream_idle_timeout),
            prompt_cache_enabled=settings.prompt_cache_enabled,
            prompt_cache_min_tokens=settings.prompt_cache_min_tokens,
            prompt_cache_ttl=cache_ttl,  # type: ignore[arg-type]
        )

    llm = create_llm(settings.model)
    cursor_inter_llm = create_llm(settings.cursor_inter_model)
    cursor_decision_llm = create_llm(settings.cursor_decision_model)

    from src.tools.assembled.codebase_retrieve import CodebaseRetrieveTool
    from src.tools.assembled.decision_edit import DecisionEditTool
    from src.tools.assembled.view_symbol_code import ViewSymbolCodeTool

    tools.register(
        CodebaseRetrieveTool(
            project_root=root,
            settings=settings,
            repo_map=repo_map_service,
            decision_llm=cursor_decision_llm,
            inter_llm=cursor_inter_llm,
            harness=harness,
            tools=tools,
        )
    )
    tools.register(
        DecisionEditTool(
            project_root=root,
            settings=settings,
            decision_llm=cursor_decision_llm,
            harness=harness,
        )
    )
    tools.register(
        ViewSymbolCodeTool(
            project_root=root,
            settings=settings,
        )
    )

    return MitKIISession(
        project_root=root,
        settings=settings,
        file_tracker=file_tracker,
        context_builder=context_builder,
        repo_map_service=repo_map_service,
        tools=tools,
        harness=harness,
        permissions=permissions,
        llm=llm,
        cursor_inter_llm=cursor_inter_llm,
        cursor_decision_llm=cursor_decision_llm,
    )
