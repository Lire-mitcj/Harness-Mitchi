from __future__ import annotations

from pathlib import Path

import pytest

from src.indexer.scanner import ProjectScanner


class TestProjectScanner:
    @pytest.fixture
    def project_dir(self, tmp_path: Path) -> Path:
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("def main(): pass\n")
        (tmp_path / "src" / "utils.py").write_text("def helper(): pass\n")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("def test_main(): pass\n")
        (tmp_path / "README.md").write_text("# Project\n")
        (tmp_path / ".gitignore").write_text("__pycache__/\n*.pyc\n")
        (tmp_path / "__pycache__").mkdir()
        (tmp_path / "__pycache__" / "main.cpython-312.pyc").write_bytes(b"")
        return tmp_path

    def test_scan_finds_files(self, project_dir: Path) -> None:
        scanner = ProjectScanner(project_dir)
        structure = scanner.scan()
        file_names = {f.name for f in structure.files}
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "README.md" in file_names

    def test_scan_ignores_pycache(self, project_dir: Path) -> None:
        scanner = ProjectScanner(project_dir)
        structure = scanner.scan()
        file_names = {f.name for f in structure.files}
        assert "main.cpython-312.pyc" not in file_names

    def test_scan_detects_languages(self, project_dir: Path) -> None:
        scanner = ProjectScanner(project_dir)
        structure = scanner.scan()
        assert "Python" in structure.languages

    def test_directory_tree(self, project_dir: Path) -> None:
        scanner = ProjectScanner(project_dir)
        tree = scanner.get_directory_tree(max_depth=2)
        assert "src/" in tree
        assert "main.py" in tree
