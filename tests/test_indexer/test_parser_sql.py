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
