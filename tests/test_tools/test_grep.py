from __future__ import annotations

from pathlib import Path
import pytest

from src.tools.assembled.grep_search import GrepSearchTool


@pytest.fixture
def grep_tool() -> GrepSearchTool:
    return GrepSearchTool()


@pytest.fixture
def sample_files(tmp_path: Path) -> Path:
    # Create files for testing
    file1 = tmp_path / "helper.py"
    file1.write_text(
        "def helper_func():\n"
        "    print('hello world')\n"
        "    return 42\n"
        "\n"
        "class HelperClass:\n"
        "    pass\n"
    )

    file2 = tmp_path / "test_api.py"
    file2.write_text(
        "import helper\n"
        "def test_helper():\n"
        "    assert helper.helper_func() == 42\n"
    )

    # Noise file to ignore (lock file)
    noise_file = tmp_path / "package-lock.json"
    noise_file.write_text("{\"helper_func\": true}")

    return tmp_path


@pytest.mark.asyncio
async def test_symbol_promotion_def(grep_tool: GrepSearchTool, sample_files: Path) -> None:
    import json
    # Searching for symbol "helper_func" should promote to definition first
    result = await grep_tool.execute(pattern="helper_func", path=str(sample_files))
    assert result.success
    payload = json.loads(result.output)
    matches = payload["matches"]
    assert payload["next_action"] == {"tool": "view_symbol_code", "symbols": ["helper_func"]}
    # Output should contain definition (line 1 of helper.py) and NOT the import/usage inside test_api.py
    assert any(m["file"].endswith("helper.py") and m["span"] == [1, 1] and m["symbol"] == "helper_func" for m in matches)
    assert not any("test_api.py" in m["file"] for m in matches)


@pytest.mark.asyncio
async def test_symbol_promotion_fallback_to_word_boundary(grep_tool: GrepSearchTool, sample_files: Path) -> None:
    import json
    # Searching for a symbol that has no "def" or "class" declaration (e.g. "print")
    result = await grep_tool.execute(pattern="print", path=str(sample_files))
    assert result.success
    matches = json.loads(result.output)["matches"]
    assert any(m["file"].endswith("helper.py") and m["span"] == [2, 2] and "print" in m["match_line"] for m in matches)


@pytest.mark.asyncio
async def test_multi_word_and_search(grep_tool: GrepSearchTool, sample_files: Path) -> None:
    import json
    # Searching for "helper assert 42" should match test_api.py line 3
    result = await grep_tool.execute(pattern="helper assert 42", path=str(sample_files))
    assert result.success
    matches = json.loads(result.output)["matches"]
    assert any(m["file"].endswith("test_api.py") and m["span"] == [3, 3] and "helper_func() == 42" in m["match_line"] for m in matches)
    # helper.py line 3 has "return 42" but not "assert", so it shouldn't match
    assert not any("helper.py" in m["file"] for m in matches)


@pytest.mark.asyncio
async def test_noise_excludes(grep_tool: GrepSearchTool, sample_files: Path) -> None:
    import json
    # Searching for "helper_func" in package-lock.json should be skipped because of excludes
    result = await grep_tool.execute(pattern="helper_func", path=str(sample_files))
    assert result.success
    matches = json.loads(result.output)["matches"]
    assert not any("package-lock.json" in m["file"] for m in matches)


@pytest.mark.asyncio
async def test_grep_search_modes(grep_tool: GrepSearchTool, sample_files: Path) -> None:
    import json

    # 1. Symbol mode: should only match definition lines
    res_sym = await grep_tool.execute(pattern="helper_func", path=str(sample_files), mode="symbol")
    assert res_sym.success
    payload_sym = json.loads(res_sym.output)
    matches_sym = payload_sym["matches"]
    assert len(matches_sym) == 1
    assert matches_sym[0]["file"].endswith("helper.py")
    assert matches_sym[0]["span"] == [1, 1]

    # 2. Import mode: should only match import lines
    res_imp = await grep_tool.execute(pattern="helper", path=str(sample_files), mode="import")
    assert res_imp.success
    payload_imp = json.loads(res_imp.output)
    matches_imp = payload_imp["matches"]
    assert len(matches_imp) == 1
    assert matches_imp[0]["file"].endswith("test_api.py")
    assert matches_imp[0]["span"] == [1, 1]
    assert "import helper" in matches_imp[0]["match_line"]

    # 3. Structure mode: should return file level existence summaries
    res_struct = await grep_tool.execute(pattern="helper_func", path=str(sample_files), mode="structure")
    assert res_struct.success
    payload_struct = json.loads(res_struct.output)
    summary = payload_struct["file_level_summary"]
    # helper_func exists in helper.py (def) and test_api.py (assert call)
    assert len(summary) == 2
    assert any(item["file"].endswith("helper.py") and item["exists"] is True and item["match_count"] == 1 for item in summary)
    assert any(item["file"].endswith("test_api.py") and item["exists"] is True and item["match_count"] == 1 for item in summary)
