from __future__ import annotations

from src.skills.sql_ast import (
    ProjectSqlAstCache,
    extract_table_sql_blocks,
    extract_view_sql_blocks,
    parse_python_sql_queries,
    extract_sql_literals_from_python,
    parse_query,
    parse_table_sql,
    parse_view_sql,
    rewrite_query_with_view,
)


def test_parse_view_sql_builds_column_source_mappings() -> None:
    model = parse_view_sql(
        "view_ticket_report_detail",
        """
        SELECT
            o.order_id,
            p.real_name AS passenger_name,
            COUNT(*) AS total_orders,
            '' AS airline
        FROM ticket_order o
        JOIN passenger_info p ON p.p_id = o.p_id
        GROUP BY o.order_id, p.real_name
        """,
    )

    assert model is not None
    assert model.columns == ["order_id", "passenger_name", "total_orders", "airline"]
    assert model.source_tables == ["ticket_order", "passenger_info"]
    assert model.aliases["o"] == "ticket_order"
    assert model.aliases["p"] == "passenger_info"
    assert model.source_to_view_column["ticket_order.order_id"] == "order_id"
    assert model.source_to_view_column["passenger_info.real_name"] == "passenger_name"
    assert model.view_column_to_source["passenger_name"] == "passenger_info.real_name"
    kinds = {source.name: source.kind for source in model.column_sources}
    assert kinds["total_orders"] == "aggregate"
    assert kinds["airline"] == "literal"
    assert model.column_defaults["airline"] == "''"


def test_parse_query_extracts_from_joins_where_group_order() -> None:
    model = parse_query(
        """
        SELECT o.order_id, p.real_name AS passenger_name
        FROM ticket_order o
        JOIN passenger_info p ON p.p_id = o.p_id
        WHERE p.real_name LIKE :keyword
        GROUP BY o.order_id, p.real_name
        ORDER BY o.order_id DESC
        """
    )

    assert model is not None
    assert [item.output_name for item in model.selects] == ["order_id", "passenger_name"]
    assert model.from_table == "ticket_order"
    assert model.from_alias == "o"
    assert model.joins[0].table == "passenger_info"
    assert model.joins[0].alias == "p"
    assert "real_name" in model.where
    assert model.group_by
    assert model.order_by


def test_parse_query_extracts_cte_select_model() -> None:
    model = parse_query(
        """
        WITH base AS (
          SELECT o.order_id, p.real_name AS passenger_name
          FROM ticket_order o
          JOIN passenger_info p ON p.p_id = o.p_id
        )
        SELECT COUNT(*) AS total FROM base
        """
    )

    assert model is not None
    assert model.selects[0].output_name == "total"
    assert model.selects[0].kind == "aggregate"
    assert model.from_table == "base"


def test_parse_python_sql_queries_extracts_embedded_select() -> None:
    models = parse_python_sql_queries(
        "def query():\n"
        "    sql = '''SELECT o.order_id FROM ticket_order o WHERE o.status = :status'''\n"
        "    return sql\n"
    )

    assert len(models) == 1
    assert models[0].from_table == "ticket_order"
    assert models[0].selects[0].output_name == "order_id"


def test_extract_sql_literals_from_python_requires_valid_sql() -> None:
    literals = extract_sql_literals_from_python(
        "text = 'SELECT words FROM a sentence, not valid sql !!!'\n"
        "sql = 'SELECT o.order_id FROM ticket_order o'\n"
    )

    assert [literal.sql for literal in literals] == [
        "SELECT o.order_id FROM ticket_order o"
    ]


def test_extract_sql_literals_from_python_records_assignment_variable() -> None:
    literals = extract_sql_literals_from_python(
        "def admin_list_orders():\n"
        "    list_sql = 'SELECT o.order_id FROM ticket_order o'\n"
        "    count_sql = 'SELECT COUNT(*) AS total FROM ticket_order o'\n"
    )

    assert [literal.variable for literal in literals] == ["list_sql", "count_sql"]


def test_rewrite_query_with_view_rewrites_cte_inner_select_only() -> None:
    rewritten = rewrite_query_with_view(
        """
        WITH base AS (
          SELECT o.order_id, p.real_name AS passenger_name
          FROM ticket_order o
          JOIN passenger_info p ON p.p_id = o.p_id
          WHERE p.real_name LIKE :keyword
        )
        SELECT COUNT(*) AS total FROM base
        """,
        target_source="view_ticket_report_detail",
        replaces_objects=["ticket_order", "passenger_info"],
        target_columns=["order_id", "passenger_name"],
        source_to_view_column={
            "ticket_order.order_id": "order_id",
            "passenger_info.real_name": "passenger_name",
        },
    )

    assert rewritten is not None
    assert "WITH base AS" in rewritten
    assert "FROM view_ticket_report_detail AS v" in rewritten
    assert "v.passenger_name LIKE :keyword" in rewritten
    assert "SELECT COUNT(*) AS total FROM base" in rewritten
    assert "ticket_order" not in rewritten
    assert "passenger_info" not in rewritten


def test_project_sql_ast_cache_scans_views(tmp_path) -> None:
    (tmp_path / "schema.sql").write_text(
        "CREATE VIEW view_ticket_report_detail AS\n"
        "SELECT o.order_id, p.real_name AS passenger_name\n"
        "FROM ticket_order o\n"
        "JOIN passenger_info p ON p.p_id = o.p_id;\n",
        encoding="utf-8",
    )

    cache = ProjectSqlAstCache(tmp_path)
    view = cache.get_view("view_ticket_report_detail")

    assert view is not None
    assert view.sql_range is not None
    assert view.sql_range.file == "schema.sql"
    assert view.sql_range.start_line == 1
    assert view.sql_range.end_line == 4
    assert view.to_dependency_dict()["columns"] == ["order_id", "passenger_name"]
    assert view.to_dependency_dict()["source_to_view_column"]["passenger_info.real_name"] == "passenger_name"


def test_project_sql_ast_cache_scans_tables(tmp_path) -> None:
    (tmp_path / "schema.sql").write_text(
        "CREATE TABLE IF NOT EXISTS passenger_info (\n"
        "  p_id INT PRIMARY KEY,\n"
        "  real_name VARCHAR(64),\n"
        "  phone VARCHAR(32)\n"
        ");\n",
        encoding="utf-8",
    )

    table = ProjectSqlAstCache(tmp_path).get_table("passenger_info")

    assert table is not None
    assert table.columns == ["p_id", "real_name", "phone"]
    assert table.sql_range is not None
    assert table.sql_range.file == "schema.sql"


def test_parse_table_sql_extracts_columns() -> None:
    table = parse_table_sql(
        "ticket_order",
        "CREATE TABLE ticket_order (order_id INT, status VARCHAR(16), amount DECIMAL(10,2))",
    )

    assert table is not None
    assert table.columns == ["order_id", "status", "amount"]


def test_extract_view_sql_blocks_tracks_line_range() -> None:
    blocks = extract_view_sql_blocks(
        "-- comment\n"
        "CREATE VIEW v_demo AS\n"
        "SELECT a.id\n"
        "FROM table_a a;\n"
    )

    assert blocks == [("v_demo", "SELECT a.id\nFROM table_a a", 2, 4)]


def test_extract_table_sql_blocks_tracks_line_range() -> None:
    blocks = extract_table_sql_blocks(
        "-- tables\n"
        "CREATE TABLE passenger_info (\n"
        "  p_id INT,\n"
        "  real_name VARCHAR(64)\n"
        ");\n"
    )

    assert blocks == [
        (
            "passenger_info",
            "CREATE TABLE passenger_info (\n  p_id INT,\n  real_name VARCHAR(64)\n)",
            2,
            5,
        )
    ]
