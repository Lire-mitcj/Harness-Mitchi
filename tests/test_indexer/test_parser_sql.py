from pathlib import Path

from src.indexer.parser import CodeParser


def test_parse_sql_views_and_procedures() -> None:
    sql_path = (
        Path(__file__).resolve().parents[1]
        / "fixtures"
        / "repo_map_sample"
        / "schema.sql"
    )
    result = CodeParser().parse_file(sql_path)
    names = {sym.name for sym in result.all_symbols}
    assert "v_boarding_pass" in names
    assert "sp_check_in" in names
    kinds = {sym.name: sym.kind for sym in result.all_symbols}
    assert kinds["v_boarding_pass"] == "view"
    assert kinds["sp_check_in"] == "procedure"


def test_parse_python_function_call_references(tmp_path: Path) -> None:
    src = tmp_path / "main.py"
    src.write_text(
        "def query_orders():\n"
        "    return build_order_query()\n"
        "\n"
        "def build_order_query():\n"
        "    return format_order_response()\n"
        "\n"
        "def format_order_response():\n"
        "    return {}\n",
        encoding="utf-8",
    )

    result = CodeParser().parse_file(src)

    assert (f"{src}:query_orders", "build_order_query") in result.references
    assert (f"{src}:build_order_query", "format_order_response") in result.references
