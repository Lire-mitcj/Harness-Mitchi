from pathlib import Path

from src.indexer.ctags import index_project


def test_index_project_filters_non_stack_extensions(tmp_path: Path) -> None:
    (tmp_path / "go.mod").write_text("module example.com/demo\n\ngo 1.22\n", encoding="utf-8")
    (tmp_path / "main.go").write_text("package main\n\nfunc main() {}\n", encoding="utf-8")
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "svc.proto").write_text("service Demo {}\n", encoding="utf-8")
    sub = tmp_path / "py_sub"
    sub.mkdir()
    (sub / "helper.py").write_text("def noise():\n    pass\n", encoding="utf-8")
    (sub / "data.json").write_text('{"x": 1}\n', encoding="utf-8")

    result = index_project(tmp_path)
    suffixes = {Path(sym.file_path).suffix for sym in result.symbols}
    assert ".go" in suffixes or any(sym.file_path.endswith(".go") for sym in result.symbols)
    assert ".py" not in suffixes
    assert ".json" not in suffixes
