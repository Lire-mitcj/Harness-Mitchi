from pathlib import Path
import pytest
import asyncio
from unittest.mock import MagicMock
from src.indexer.ctags import CtagsSymbol
from src.indexer.repo_map import RepoMap, RankedSymbol
from src.tools.assembled.retriever import CursorRetriever
from src.tools.assembled.context_pack_builder import CursorContextPackBuilder, _comment_prefix
from src.agent.contracts import RetrievalResult, RetrievalSymbol
from src.agent.state import CursorState
from src.harness.cursor.manager import CursorStateManager
from src.tools.assembled.graph_bridge import CursorGraphQueryBridge, GraphNode, GraphEdge
from src.tools.assembled.repo_map_lookup import CandidateSymbol


class _RepoMapService:
    def __init__(self, repo_map):
        self.map = repo_map
    def wait_until_ready(self, timeout=None):
        return True


def _symbol(file_path: str, name: str, start_line: int, score: float = 0.0) -> RankedSymbol:
    return RankedSymbol(
        file_path=file_path,
        name=name,
        kind="function",
        start_line=start_line,
        end_line=start_line + 2,
        signature=f"def {name}()",
        score=score,
        symbol_id=f"{file_path}:{name}:{start_line}",
    )


def test_sql_anchor_integrity(tmp_path: Path) -> None:
    from src.agent.sql_parser import UniversalSqlParser

    sql_lines = [f"/* line {i} */" for i in range(1, 150)]
    sql_lines.append("CREATE VIEW v_order_detail AS SELECT * FROM orders;")
    sql_lines.extend(f"/* line {i} */" for i in range(152, 301))

    sql_file = tmp_path / "view.sql"
    sql_file.write_text("\n".join(sql_lines), encoding="utf-8")

    symbols = UniversalSqlParser().parse_text_block(
        sql_file.read_text(encoding="utf-8"),
        "view.sql",
        1,
    )

    assert len(symbols) == 1
    sym = symbols[0]
    assert sym.file_path == "view.sql"
    assert sym.name == "v_order_detail"
    assert sym.kind == "ddl_view"
    assert sym.start_line == 150
    assert sym.end_line == 150
    assert sym.tables_referenced == ("orders",)


def test_sql_view_signature_contains_clean_alias_outline() -> None:
    from src.agent.sql_parser import UniversalSqlParser

    sql = (
        "CREATE VIEW view_ticket_report_detail AS\n"
        "SELECT\n"
        "396 | o.order_id,\n"
        "398 | o.p_id,\n"
        "400 | f.flight_id,\n"
        "410 | s.seat_no,\n"
        "412 | p.real_name AS passenger_name,\n"
        "414 | p.id_card AS passenger_id_no\n"
        "FROM ticket_order o\n"
        "JOIN passenger_info p ON p.p_id = o.p_id\n"
        "JOIN flight_seat s ON s.seat_id = o.seat_id;\n"
    )

    symbols = UniversalSqlParser().parse_text_block(sql, "init.sql", 400)

    assert symbols[0].signature == (
        "CREATE VIEW `view_ticket_report_detail` AVAILABLE FIELDS: "
        "[order_id, p_id, flight_id, seat_no, passenger_name, passenger_id_no]"
    )
    assert "412 |" not in symbols[0].signature


def test_syntax_isolation_verification(tmp_path: Path) -> None:
    # Syntax Isolation Verification: Verify that a folded .sql node contains -- syntax comments, while a folded .py node contains # structures.
    # Test _comment_prefix mappings directly
    assert _comment_prefix("db.sql") == "--"
    assert _comment_prefix("main.py") == "#"
    assert _comment_prefix("helper.go") == "//"
    assert _comment_prefix("app.js") == "//"
    assert _comment_prefix("service.java") == "//"

    # Test actual folding in Context Builder
    py_file = tmp_path / "main.py"
    py_file.write_text("def top_1():\n    pass\n\ndef other_py():\n    x = 1\n    y = 2\n", encoding="utf-8")
    
    # We create a pseudo go file that we will fold
    go_file = tmp_path / "helper.go"
    go_file.write_text("package main\n\nfunc otherGo() {\n\tprintln(1)\n\tprintln(2)\n}\n", encoding="utf-8")

    retrieval = RetrievalResult(
        files=("main.py", "helper.go"),
        symbols=(
            RetrievalSymbol("main.py", "top_1", 1, 2),
            RetrievalSymbol("main.py", "other_py", 4, 6),
            RetrievalSymbol("helper.go", "otherGo", 3, 6),
        ),
    )

    builder = CursorContextPackBuilder(tmp_path)
    pack = builder.build_context(retrieval)

    py_win = next(w for w in pack.windows if w.file == "main.py")
    go_win = next(w for w in pack.windows if w.file == "helper.go")

    # Core blocks are exact; non-core declarations move to the skeleton.
    assert "[LAYER_1_CORE_SYMBOL]" in py_win.content
    assert "[LAYER_3_GLOBAL_SKELETON]" in py_win.content
    assert "[LAYER_3_GLOBAL_SKELETON]" in go_win.content


def test_state_serialization_boundary_check() -> None:
    # State Serialization Boundary Check: Assert that injecting a 10KB dummy failure traceback string into last_observation gets aggressively truncated under 500 characters by the CursorStateManager, preserving the core CursorState serialization profile below 2KB.
    manager = CursorStateManager(max_bytes=2048)
    
    dummy_traceback = "Traceback (most recent call last):\n" + "  File \"app.py\", line 42, in run\n    raise ValueError(\"Something went wrong\")\n" * 200
    state = CursorState(
        task="Test task",
        current_file="app.py",
        last_patch="patch content",
        last_observation=dummy_traceback,
        status="running",
        current_step=1,
        max_steps=10,
        stage_completion=0.0,
    )
    
    bounded_state = manager._bounded(state)
    formatted = manager.format_for_prompt(bounded_state)
    
    # Split formatted output to isolate observation part
    assert "--- LAST RUNTIME OBSERVATION ---" in formatted
    obs_part = formatted.split("--- LAST RUNTIME OBSERVATION ---\n")[1].split(
        "\n--- EXECUTION TRACE LAYER ---"
    )[0]
    
    assert len(obs_part) <= 500
    assert "... [TRUNCATED SYSTEM LOGS] ..." in obs_part
    assert manager.serialized_size(bounded_state) <= 2048


@pytest.mark.asyncio
async def test_graph_expansion_domain_isolation(tmp_path: Path) -> None:
    # DB seed must ONLY expand to DB domain at distance 0, EXCEPT for reference_reverse which can cross to backend
    db_sym = _symbol("dao/db.py", "db_query", 1, score=1.0)
    backend_sym = _symbol("service/api.py", "api_call", 1, score=0.0)
    ui_sym = _symbol("view/main.py", "render_ui", 1, score=0.0)
    
    symbols = [db_sym, backend_sym, ui_sym]
    
    # reference_reverse edge: api_call calls db_query, so db_query -> api_call is reference_reverse
    # reference edge: db_query -> render_ui
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={
            "dao/db.py": [db_sym],
            "service/api.py": [backend_sym],
            "view/main.py": [ui_sym],
        },
        symbols_by_id={s.symbol_id: s for s in symbols},
        reference_edges=[
            (backend_sym.symbol_id, db_sym.symbol_id),  # api_call calls db_query
            (db_sym.symbol_id, ui_sym.symbol_id),       # db_query calls render_ui (type reference)
        ],
    )
    
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=3,
        max_seeds=3,
    )
    
    result = await graph.expand_candidates((CandidateSymbol(db_sym, 1.0),))
    
    # Since it is depth=1, from db_sym (domain db):
    # - render_ui (domain ui) is type 'reference' (cross-domain), should be BLOCKED at distance 0
    # - api_call (domain backend) is type 'reference_reverse' (cross-domain), should be ALLOWED via Caller Exemption
    
    assert "api_call" in result.expanded_symbols
    assert "render_ui" not in result.expanded_symbols
    assert result.meta["domain"] == "db"


@pytest.mark.asyncio
async def test_static_scc_and_degree_caching(tmp_path: Path) -> None:
    # Verify Tarjan calculation runs only once and is cached on the bridge instance
    sym1 = _symbol("service/api.py", "api_call", 1, score=1.0)
    symbols = [sym1]
    
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={"service/api.py": [sym1]},
        symbols_by_id={s.symbol_id: s for s in symbols},
        reference_edges=[],
    )
    
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=2,
        max_seeds=2,
    )
    
    original_find_sccs = graph._find_sccs
    graph._find_sccs = MagicMock(side_effect=original_find_sccs)
    
    # First expansion - computes and caches
    await graph.expand_candidates((CandidateSymbol(sym1, 1.0),))
    assert graph._find_sccs.call_count == 1
    assert graph._scc_map is not None
    assert graph._node_degrees is not None
    
    # Second expansion - should use cache
    await graph.expand_candidates((CandidateSymbol(sym1, 1.0),))
    assert graph._find_sccs.call_count == 1


@pytest.mark.asyncio
async def test_smoothed_graph_score_cap(tmp_path: Path) -> None:
    # Verify that lexical cap correctly limits scores but preserves a 0.12 baseline floor
    # We use non-framework names so they don't get seed framework dampening
    sym1 = _symbol("service/api.py", "my_boarding_pass", 1, score=1.0)
    sym2 = _symbol("service/api.py", "fetch_details", 10, score=0.0)
    symbols = [sym1, sym2]
    
    repo_map = RepoMap(
        project_root=tmp_path,
        symbols=symbols,
        all_symbols=symbols,
        symbols_by_file={"service/api.py": symbols},
        symbols_by_id={s.symbol_id: s for s in symbols},
        reference_edges=[(sym1.symbol_id, sym2.symbol_id)],
    )
    
    graph = CursorGraphQueryBridge(
        _RepoMapService(repo_map),
        depth=1,
        top_symbols=2,
        max_seeds=2,
    )
    
    # Test case: lexical score is 0.0 -> cap is max(0.12, 0.0) = 0.12
    class FakeBridge:
        def __init__(self, concepts):
            self.concepts = concepts
            
    result = await graph.expand_candidates(
        (CandidateSymbol(sym1, 1.0),),
        bridge=FakeBridge(["completely_unrelated_concept"])
    )
    
    # Verify that fetch_details is capped to 0.12 (floor) because its lexical score is 0.0
    node_scores = {node.name: node.score for node in result.graph_nodes}
    assert node_scores["fetch_details"] == 0.12
