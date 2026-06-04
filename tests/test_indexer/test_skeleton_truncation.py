from pathlib import Path

from src.indexer.repo_map import RepoMap, _truncate_skeleton_sections


def test_skeleton_truncation_keeps_top_files_over_skeleton() -> None:
    header = ["<repo_map>", "intro"]
    top_files = [f"- file{i}.py  score=0.1" for i in range(20)]
    top_symbols = [f"- file{i}.py:1  function fn{i}  score=0.01" for i in range(40)]
    skeleton = ["big.py", "  L1-2 foo  def foo():"] * 30
    text = _truncate_skeleton_sections(
        header=header,
        top_files=top_files,
        search_modules=[
            "Rule: pick one module.",
            "- module=src files=['src/app.py'] glob='*.py' patterns='app|query'",
        ],
        top_symbols=top_symbols,
        file_skeleton=skeleton,
        max_chars=2500,
    )
    assert text.endswith("</repo_map>")
    assert "## Top files" in text
    assert "## Search modules" in text
    assert "## Top symbols" in text
    assert "file skeleton omitted" in text or "…[omitted]" in text or "L1-2" not in text


def test_repo_map_skeleton_respects_max_chars() -> None:
    repo = RepoMap(
        project_root=Path("."),
        symbols=[],
        file_scores={f"src/f{i}.py": 0.01 * i for i in range(30, 0, -1)},
        symbol_count=100,
    )
    block = repo.to_skeleton_block(max_chars=1500, top_files=20, top_symbols=50)
    assert len(block) <= 1500
    assert block.startswith("<repo_map")
