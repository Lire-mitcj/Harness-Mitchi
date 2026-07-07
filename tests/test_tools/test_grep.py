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
    assert payload["next_action"]["tool"] == "view_symbol_code"
    assert payload["next_action"]["symbols"] == ["helper_func"]
    assert payload["suggested_views"][0]["symbol"] == "helper_func"
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
async def test_grep_search_batch_patterns(grep_tool: GrepSearchTool, sample_files: Path) -> None:
    import json

    result = await grep_tool.execute(
        patterns=["helper_func", "HelperClass", "missing_symbol"],
        path=str(sample_files),
    )
    assert result.success
    payload = json.loads(result.output)
    matches = payload["matches"]
    assert payload["searched_patterns"] == ["helper_func", "HelperClass", "missing_symbol"]
    assert any(m["symbol"] == "helper_func" for m in matches)
    assert any(m["symbol"] == "HelperClass" for m in matches)
    assert payload["returned_matches"] >= 2


@pytest.mark.asyncio
async def test_grep_search_sql_match_suggests_view(tmp_path: Path) -> None:
    import json

    sql_file = tmp_path / "schema.sql"
    sql_file.write_text(
        "CREATE TABLE IF NOT EXISTS `order_timeline` (\n"
        "  id INT PRIMARY KEY\n"
        ");\n",
        encoding="utf-8",
    )
    tool = GrepSearchTool()
    result = await tool.execute(pattern="CREATE TABLE", path=str(tmp_path), include="*.sql")
    assert result.success
    payload = json.loads(result.output)
    assert payload["suggested_views"]
    assert payload["suggested_views"][0]["symbol"] == "order_timeline"
    assert payload["next_action"]["suggested_views"][0]["file"].endswith("schema.sql")


@pytest.mark.asyncio
async def test_grep_search_ranks_definition_before_mount(tmp_path: Path) -> None:
    import json

    (tmp_path / "main.py").write_text(
        "from fastapi import FastAPI\n"
        "app = FastAPI()\n"
        "app.include_router(build_router(engine))\n",
        encoding="utf-8",
    )
    (tmp_path / "list.py").write_text(
        "def build_router(engine):\n    return engine\n",
        encoding="utf-8",
    )
    tool = GrepSearchTool()
    result = await tool.execute(
        pattern="build_router",
        path=str(tmp_path),
        _project_root=tmp_path,
    )
    assert result.success
    payload = json.loads(result.output)
    kinds = payload.get("match_kinds_top") or []
    assert kinds[0] == "definition"
    views = payload["suggested_views"]
    assert any(v["file"].endswith("list.py") and v["symbol"] == "build_router" for v in views)
    assert any(
        v["file"].endswith("main.py") and v.get("resolved_from") == "mount_context"
        for v in views
    )


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

    # 3. Structure mode: file summary plus suggested_views for representative hits
    res_struct = await grep_tool.execute(pattern="helper_func", path=str(sample_files), mode="structure")
    assert res_struct.success
    payload_struct = json.loads(res_struct.output)
    summary = payload_struct["file_level_summary"]
    assert len(summary) == 2
    assert payload_struct.get("suggested_views")
    assert payload_struct.get("next_action", {}).get("tool") == "view_symbol_code"
