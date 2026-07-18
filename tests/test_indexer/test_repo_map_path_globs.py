from __future__ import annotations

from pathlib import Path

from src.indexer.ctags import index_project
from src.indexer.path_globs import path_matches_glob, repo_map_path_allowed


def test_path_matches_glob_prefix_and_suffix() -> None:
    assert path_matches_glob("agentmesh_orchestrator/foo.py", "agentmesh_orchestrator/**")
    assert path_matches_glob("internal/ws/server.go", "internal/**")
    assert not path_matches_glob("cmd/main.go", "internal/**")


def test_repo_map_path_allowed_include_and_exclude() -> None:
    assert repo_map_path_allowed(
        "internal/ws/server.go",
        include_globs=("internal/**", "api/**"),
        exclude_globs=(),
    )
    assert not repo_map_path_allowed(
        "agentmesh_orchestrator/list.py",
        include_globs=("internal/**", "api/**"),
        exclude_globs=(),
    )
    assert not repo_map_path_allowed(
        "agentmesh_orchestrator/list.py",
        include_globs=(),
        exclude_globs=("agentmesh_orchestrator/**",),
    )


def test_index_project_honors_exclude_globs(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.22\n", encoding="utf-8")
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    noise = tmp_path / "agentmesh_orchestrator"
    noise.mkdir()
    (noise / "list.py").write_text("def noise():\n    pass\n", encoding="utf-8")

    result = index_project(
        tmp_path,
        exclude_globs=("agentmesh_orchestrator/**",),
    )
    files = {sym.file_path for sym in result.symbols}
    assert any(path.endswith("main.go") for path in files)
    assert not any(path.startswith("agentmesh_orchestrator/") for path in files)


def test_index_project_honors_include_globs(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.22\n", encoding="utf-8")
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    internal = tmp_path / "internal" / "ws"
    internal.mkdir(parents=True)
    (internal / "server.go").write_text(
        "package ws\n\nfunc Serve() {}\n",
        encoding="utf-8",
    )
    noise = tmp_path / "vendor_noise"
    noise.mkdir()
    (noise / "extra.go").write_text("package noise\n", encoding="utf-8")

    result = index_project(
        tmp_path,
        include_globs=("internal/**", "main.go"),
    )
    files = {sym.file_path for sym in result.symbols}
    assert "internal/ws/server.go" in files
    assert "main.go" in files
    assert "vendor_noise/extra.go" not in files
