from pathlib import Path

from src.agent.cursor_loop import CursorLoop
from src.config.settings import MitKIISettings
from src.harness.gates.preflight_probe import assess_preflight
from src.harness.gates.preflight_slices import resolve_repo_map_line_slices
from src.harness.gates.types import GateVerdict
from src.indexer.repo_map import build_repo_map
from src.indexer.repo_map_service import BuildState, RepoMapService
from src.orchestrator.isolation import load_context_file_contents
from src.planner.task_tree import SubTaskKind, SubTaskNode, TaskTree
from src.runtime.session_factory import create_mitkii_session

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "repo_map_sample"


def test_background_build_completes(tmp_path: Path) -> None:
    service = RepoMapService(
        FIXTURE,
        enabled=True,
        top_k=20,
        cache_path=tmp_path / "symbols.db",
    )
    service.start_background_build()
    assert service.wait_until_ready(timeout=30.0)
    assert service.build_state == BuildState.READY
    assert service.map is not None
    assert service.map.symbol_count > 0


def test_create_mitkii_session_starts_background_repo_map(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(FIXTURE)
    settings = MitKIISettings(data_dir=tmp_path / ".mitkii", repo_map_enabled=True)
    session = create_mitkii_session(settings=settings)
    assert session.repo_map_service is not None
    assert session.repo_map_service.build_state in {BuildState.BUILDING, BuildState.READY}
    assert session.repo_map_service.wait_until_ready(timeout=30.0)


def test_create_mitkii_session_uses_settings_project_root(tmp_path: Path) -> None:
    project = tmp_path / "target"
    project.mkdir()
    (project / "app.py").write_text("x = 1\n", encoding="utf-8")
    settings = MitKIISettings(
        data_dir=tmp_path / ".mitkii",
        repo_map_enabled=False,
        project_root=project,
    )

    session = create_mitkii_session(settings=settings)

    assert session.project_root == project.resolve()
    assert session.harness.project_root == project.resolve()


def test_create_mitkii_session_explicit_project_root_overrides_settings(tmp_path: Path) -> None:
    configured = tmp_path / "configured"
    explicit = tmp_path / "explicit"
    configured.mkdir()
    explicit.mkdir()
    settings = MitKIISettings(
        data_dir=tmp_path / ".mitkii",
        repo_map_enabled=False,
        project_root=configured,
    )

    session = create_mitkii_session(project_root=explicit, settings=settings)

    assert session.project_root == explicit.resolve()


def test_create_core_loop_uses_cursor_runtime(tmp_path: Path) -> None:
    settings = MitKIISettings(
        data_dir=tmp_path / ".mitkii",
        repo_map_enabled=False,
    )
    session = create_mitkii_session(project_root=tmp_path, settings=settings)

    loop = session.create_core_loop()

    assert isinstance(loop, CursorLoop)
    assert session.cursor_inter_llm.model == settings.cursor_inter_model
    assert session.cursor_decision_llm.model == settings.cursor_decision_model
    assert loop.inter_llm is session.cursor_inter_llm
    assert loop.decision_llm is session.cursor_decision_llm


def test_preflight_repo_map_slices(tmp_path: Path) -> None:
    big = tmp_path / "big.py"
    big.write_text("x = 1\n" + "def target_fn():\n    return 42\n" + ("pass\n" * 12000))
    indexed = __import__("src.indexer.ctags", fromlist=["index_project"]).index_project(tmp_path)
    repo_map = build_repo_map(tmp_path, indexed=indexed, top_k=50)
    settings = MitKIISettings(
        data_dir=tmp_path / ".mitkii",
        max_context_tokens=128_000,
        context_budget_ratio=0.75,
        preflight_large_file_bytes=10_000,
    )
    tree = TaskTree(
        root_task="fix target_fn",
        nodes=[
            SubTaskNode(
                id="st-1",
                kind=SubTaskKind.EDIT,
                description="Fix target_fn return value",
                context_files=["big.py"],
                acceptance_criteria="target_fn fixed",
            )
        ],
    )
    slices = resolve_repo_map_line_slices(
        subtask=tree.nodes[0],
        task_tree=tree,
        context_files=["big.py"],
        repo_map=repo_map,
        settings=settings,
        target_files=["big.py"],
    )
    assert "big.py" in slices

    result = assess_preflight(
        subtask=tree.nodes[0],
        task_tree=tree,
        project_root=tmp_path,
        settings=settings,
        repo_map=repo_map,
    )
    assert result.verdict in {GateVerdict.WARN, GateVerdict.PASS}
    if result.policy.line_slices:
        loaded = load_context_file_contents(
            tmp_path, ["big.py"], policy=result.policy
        )
        assert loaded
        assert "repo_map slice" in loaded[0][1]
        assert "target_fn" in loaded[0][1]
