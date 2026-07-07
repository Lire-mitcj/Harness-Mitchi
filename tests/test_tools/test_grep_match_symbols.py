from __future__ import annotations

from pathlib import Path

from src.tools.grep_match_symbols import (
    classify_match_line,
    expand_schema_definition_spans,
    extract_symbol_from_match_line,
    fill_symbols_from_adjacent_lines,
    rank_matches,
    reference_views_from_repo_map,
    resolve_decorator_symbols_from_files,
    resolve_mount_line_symbols,
    suggested_views_from_matches,
)


def test_extract_sql_create_table_symbol() -> None:
    line = "CREATE TABLE IF NOT EXISTS `ticket_order` ("
    assert extract_symbol_from_match_line(line) == "ticket_order"


def test_extract_python_def_symbol() -> None:
    line = "async def build_router(engine: Engine) -> APIRouter:"
    assert extract_symbol_from_match_line(line) == "build_router"


def test_suggested_views_deduplicates_file_symbol() -> None:
    matches = [
        {"file": "db/init/init.sql", "symbol": "ticket_order", "span": [10, 10]},
        {"file": "db/init/init.sql", "symbol": "ticket_order", "span": [11, 11]},
        {"file": "list.py", "symbol": "build_router", "span": [16, 16]},
    ]
    views = suggested_views_from_matches(matches)
    assert len(views) == 2
    assert views[0]["symbol"] == "ticket_order"


def test_fill_symbols_from_exception_handler_decorator() -> None:
    matches = [
        {
            "file": "main.py",
            "symbol": "",
            "span": [120, 120],
            "match_line": "@app.exception_handler(SQLAlchemyError)",
        },
        {
            "file": "main.py",
            "symbol": "",
            "span": [121, 121],
            "match_line": "async def handle_db_error(request, exc):",
        },
    ]
    filled = fill_symbols_from_adjacent_lines(matches)
    assert filled[0]["symbol"] == "handle_db_error"


def test_resolve_decorator_symbols_from_file(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "app = FastAPI()",
                "",
                "@app.exception_handler(SQLAlchemyError)",
                "async def handle_db_error(request, exc):",
                "    return JSONResponse(status_code=500, content={'detail': 'db'})",
            ]
        ),
        encoding="utf-8",
    )
    matches = [
        {
            "file": "main.py",
            "symbol": "",
            "span": [4, 4],
            "match_line": "@app.exception_handler(SQLAlchemyError)",
        },
    ]
    filled = resolve_decorator_symbols_from_files(
        matches,
        project_root=tmp_path,
    )
    assert filled[0]["symbol"] == "handle_db_error"
    assert filled[0]["resolved_from"] == "decorator_context"


def test_resolve_mount_line_symbols_from_include_router(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "app = FastAPI()",
                "",
                "def wire_routes():",
                "    app.include_router(build_router(engine))",
            ]
        ),
        encoding="utf-8",
    )
    matches = [
        {
            "file": "main.py",
            "symbol": "",
            "span": [5, 5],
            "match_line": "    app.include_router(build_router(engine))",
        },
    ]
    filled = resolve_mount_line_symbols(matches, project_root=tmp_path)
    views = suggested_views_from_matches(filled)
    assert filled[0]["symbol"] == "wire_routes"
    assert filled[0]["resolved_from"] == "mount_context"
    assert views
    assert views[0]["span"][1] - views[0]["span"][0] + 1 >= 2


def test_resolve_mount_line_without_enclosing_def_uses_app_not_callee(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "app = FastAPI()",
                "app.include_router(build_router(engine))",
            ]
        ),
        encoding="utf-8",
    )
    matches = [
        {
            "file": "main.py",
            "symbol": "",
            "span": [3, 3],
            "match_line": "app.include_router(build_router(engine))",
        },
    ]
    filled = resolve_mount_line_symbols(matches, project_root=tmp_path)
    assert filled[0]["symbol"] == "app"
    assert filled[0]["symbol"] != "build_router"


def test_resolve_mount_line_factory_pattern_uses_create_app(tmp_path: Path) -> None:
    main_py = tmp_path / "main.py"
    main_py.write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "",
                "def create_app():",
                "    app = FastAPI()",
                "    app.include_router(build_router(engine))",
                "    return app",
            ]
        ),
        encoding="utf-8",
    )
    matches = [
        {
            "file": "main.py",
            "symbol": "",
            "span": [5, 5],
            "match_line": "    app.include_router(build_router(engine))",
        },
    ]
    filled = resolve_mount_line_symbols(matches, project_root=tmp_path)
    assert filled[0]["symbol"] == "create_app"


def test_classify_match_line_kinds() -> None:
    assert classify_match_line("def build_router(engine):", pattern="build_router") == "definition"
    assert classify_match_line("app.include_router(build_router(engine))", pattern="build_router") == "mount"
    assert classify_match_line("@app.exception_handler(SQLAlchemyError)", pattern="handler") == "decorator"
    assert classify_match_line("CREATE TABLE IF NOT EXISTS `order_timeline` (", pattern="order") == "schema"


def test_rank_matches_prefers_definition_over_call_site() -> None:
    matches = [
        {
            "file": "main.py",
            "symbol": "build_router",
            "span": [10, 10],
            "match_line": "app.include_router(build_router(engine))",
            "matched_pattern": "build_router",
        },
        {
            "file": "list.py",
            "symbol": "build_router",
            "span": [16, 16],
            "match_line": "def build_router(engine: Engine) -> APIRouter:",
            "matched_pattern": "build_router",
        },
    ]
    ranked = rank_matches(matches, searched_patterns=["build_router"])
    assert ranked[0]["file"] == "list.py"
    assert ranked[0]["match_kind"] == "definition"
    assert any(item["file"] == "main.py" and item["match_kind"] == "mount" for item in ranked)


def test_rank_matches_drops_pure_call_site_when_definition_exists() -> None:
    matches = [
        {
            "file": "routes.py",
            "symbol": "helper_func",
            "span": [5, 5],
            "match_line": "    return helper_func(x)",
            "matched_pattern": "helper_func",
        },
        {
            "file": "helper.py",
            "symbol": "helper_func",
            "span": [1, 1],
            "match_line": "def helper_func():",
            "matched_pattern": "helper_func",
        },
    ]
    ranked = rank_matches(matches, searched_patterns=["helper_func"])
    assert len(ranked) == 1
    assert ranked[0]["file"] == "helper.py"


def test_expand_schema_definition_spans_multiline(tmp_path: Path) -> None:
    sql_path = tmp_path / "schema.sql"
    sql_path.write_text(
        "CREATE TABLE IF NOT EXISTS `order_timeline` (\n"
        "  id INT PRIMARY KEY,\n"
        "  order_id INT NOT NULL\n"
        ");\n",
        encoding="utf-8",
    )
    matches = [
        {
            "file": "schema.sql",
            "symbol": "order_timeline",
            "span": [1, 1],
            "match_line": "CREATE TABLE IF NOT EXISTS `order_timeline` (",
            "match_kind": "schema",
        }
    ]
    expanded = expand_schema_definition_spans(matches, project_root=tmp_path)
    assert expanded[0]["span"] == [1, 4]


def test_reference_views_from_repo_map_cross_file() -> None:
    from types import SimpleNamespace

    list_sym = SimpleNamespace(
        symbol_id="list.py:build_router:16",
        file_path="list.py",
        name="build_router",
        start_line=16,
        end_line=100,
    )
    main_sym = SimpleNamespace(
        symbol_id="main.py:create_app:3",
        file_path="main.py",
        name="create_app",
        start_line=3,
        end_line=20,
    )
    repo_map = SimpleNamespace(
        symbols_by_file={
            "list.py": [list_sym],
            "main.py": [main_sym],
        },
        symbols_by_id={
            list_sym.symbol_id: list_sym,
            main_sym.symbol_id: main_sym,
        },
        reference_edges=[(main_sym.symbol_id, list_sym.symbol_id)],
    )
    matches = [
        {
            "file": "list.py",
            "symbol": "build_router",
            "match_kind": "definition",
            "span": [16, 100],
        }
    ]
    views = reference_views_from_repo_map(matches, repo_map)
    assert len(views) == 1
    assert views[0]["file"] == "main.py"
    assert views[0]["symbol"] == "create_app"
    assert views[0]["resolved_from"] == "repo_reference"
