from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.context.retriever import ContextRetriever, build_context_queries
from src.indexer.ctags import CtagsIndexResult, CtagsSymbol
from src.indexer.repo_map import build_repo_map


@dataclass(frozen=True)
class FakeSymbol:
    file_path: str
    name: str
    kind: str
    start_line: int
    end_line: int
    signature: str
    score: float


class FakeRepoMap:
    def __init__(self, symbols: list[FakeSymbol]) -> None:
        self.symbols = symbols
        self.queries: list[str] = []

    def search(self, query: str, *, limit: int = 20) -> list[FakeSymbol]:
        self.queries.append(query)
        q = query.lower()
        return [
            symbol
            for symbol in self.symbols
            if q in symbol.name.lower()
            or q in symbol.file_path.lower()
            or q in symbol.signature.lower()
        ][:limit]


def test_context_retriever_aggregates_symbols_files_and_snippets(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text(
        "\n".join(
            [
                "from fastapi import FastAPI",
                "app = FastAPI()",
                "",
                '@app.get("/api/orders/query")',
                "def query_orders():",
                "    sql = 'SELECT * FROM view_ticket_report_detail'",
                "    return sql",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    repo_map = FakeRepoMap([
        FakeSymbol(
            file_path="main.py",
            name="query_orders",
            kind="function",
            start_line=4,
            end_line=7,
            signature="def query_orders()",
            score=0.9,
        )
    ])

    pack = ContextRetriever(project_root=tmp_path, repo_map=repo_map).retrieve(
        "把当前登机牌查询接口改成用视图查询",
        task_template="investigate",
    )

    assert pack.relevant_files == ("main.py",)
    assert pack.symbols[0].name == "query_orders"
    assert "view_ticket_report_detail" in pack.snippets[0].text
    assert pack.confidence >= 0.75
    assert not pack.missing_info
    assert pack.search_plan[0].patterns
    assert "登机牌" in repo_map.queries


def test_context_retriever_reports_missing_repo_map(tmp_path: Path) -> None:
    pack = ContextRetriever(project_root=tmp_path, repo_map=None).retrieve(
        "把当前登机牌查询接口改成用视图查询"
    )

    assert pack.confidence < 0.75
    assert "repo_map unavailable" in pack.missing_info
    assert "no relevant symbols found" in pack.missing_info


def test_build_context_queries_keeps_domain_expansions() -> None:
    queries = build_context_queries("把当前登机牌查询接口改成用视图查询")

    assert "登机牌" in queries
    assert "boarding_pass" in queries
    assert "query" in queries
    assert "view" in queries
    assert "api" in queries


def test_context_pack_planner_block_contains_structured_evidence(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text("def query_orders():\n    return 'ok'\n", encoding="utf-8")
    repo_map = FakeRepoMap([
        FakeSymbol(
            file_path="main.py",
            name="query_orders",
            kind="function",
            start_line=1,
            end_line=2,
            signature="def query_orders()",
            score=0.9,
        )
    ])
    pack = ContextRetriever(project_root=tmp_path, repo_map=repo_map).retrieve(
        "把当前登机牌查询接口改成用视图查询"
    )

    block = pack.to_planner_block()

    assert '<context_pack source="ContextRetriever">' in block
    assert "confidence:" in block
    assert "relevant_files:" in block
    assert "main.py" in block
    assert "query_orders" in block


def test_context_pack_agent_json_contains_builder_sections(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text(
        "\n".join([
            "def helper():",
            "    return 'x'",
            "",
            "def query_orders():",
            "    return helper()",
        ])
        + "\n",
        encoding="utf-8",
    )
    repo_map = FakeRepoMap([
        FakeSymbol(
            file_path="main.py",
            name="query_orders",
            kind="function",
            start_line=4,
            end_line=5,
            signature="def query_orders()",
            score=0.9,
        ),
        FakeSymbol(
            file_path="main.py",
            name="helper",
            kind="function",
            start_line=1,
            end_line=2,
            signature="def helper()",
            score=0.7,
        ),
    ])

    pack = ContextRetriever(project_root=tmp_path, repo_map=repo_map).build(
        user_request="修改 main.py 里的 query_orders",
        current_files=("main.py",),
    )
    data = pack.to_agent_json()

    assert data["schema"] == "mitkii.context_pack.v1"
    assert data["task"]["mode"] == "edit"
    assert data["candidate_files"][0]["file"] == "main.py"
    assert "explicit_file" in data["candidate_files"][0]["reasons"]
    assert data["focused_snippets"]
    assert data["evidence"]
    assert "delete_file" in data["tool_policy"]["denied_tools"]
    assert data["budget"]["max_input_tokens"] == 24000


def test_context_pack_merges_previous_handoff(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text("def query_orders():\n    return 'ok'\n", encoding="utf-8")
    repo_map = FakeRepoMap([
        FakeSymbol(
            file_path="main.py",
            name="query_orders",
            kind="function",
            start_line=1,
            end_line=2,
            signature="def query_orders()",
            score=0.9,
        )
    ])

    pack = ContextRetriever(project_root=tmp_path, repo_map=repo_map).build(
        user_request="修改 query_orders",
        previous_handoff={
            "facts": [{"source_subtask": "st-1", "fact": "query_orders is target"}],
            "evidence": [{"source_subtask": "st-1", "file": "main.py", "symbol": "query_orders"}],
            "known_negatives": [{"source_subtask": "st-1", "reason": "no matches in tests"}],
            "next_focus": [{"source_subtask": "st-1", "focus": "main.py"}],
        },
        mode="edit",
    )
    data = pack.to_agent_json()

    assert any(item.get("type") == "prior_handoff" for item in data["evidence"])
    assert any(item.get("type") == "prior_fact" for item in data["evidence"])
    assert data["known_negatives"]
    assert pack.metadata["previous_next_focus"] == "main.py"


def test_context_pack_scores_recent_files(tmp_path: Path) -> None:
    target = tmp_path / "recent.py"
    target.write_text("def touched():\n    return 1\n", encoding="utf-8")
    repo_map = FakeRepoMap([])

    pack = ContextRetriever(project_root=tmp_path, repo_map=repo_map).build(
        user_request="继续修改刚才的文件",
        recent_files=("recent.py",),
        mode="edit",
    )

    assert pack.candidate_files[0]["file"] == "recent.py"
    assert "recent_edit" in pack.candidate_files[0]["reasons"]


def test_context_builder_uses_repo_map_reference_graph() -> None:
    fixture = Path(__file__).resolve().parents[1] / "fixtures" / "repo_map_sample"
    repo_map = build_repo_map(fixture, top_k=50)

    pack = ContextRetriever(project_root=fixture, repo_map=repo_map).build(
        user_request="检查 AppService 的调用链",
        mode="diagnose",
    )

    assert "AppService -> BaseService" in pack.call_chain
    assert any(symbol.name == "BaseService" for symbol in pack.symbols)
    assert any(snippet.file_path == "base.py" for snippet in pack.focused_snippets)


def test_context_builder_expands_two_layer_function_call_chain(tmp_path: Path) -> None:
    main = tmp_path / "main.py"
    main.write_text(
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
    indexed = CtagsIndexResult(
        symbols=[
            CtagsSymbol("main.py", "query_orders", "function", 1, 2, "def query_orders()"),
            CtagsSymbol("main.py", "build_order_query", "function", 4, 5, "def build_order_query()"),
            CtagsSymbol("main.py", "format_order_response", "function", 7, 8, "def format_order_response()"),
        ],
        references=[
            ("main.py:query_orders:1", "build_order_query"),
            ("main.py:build_order_query:4", "format_order_response"),
        ],
        source="test",
    )
    repo_map = build_repo_map(tmp_path, indexed=indexed)

    pack = ContextRetriever(project_root=tmp_path, repo_map=repo_map).build(
        user_request="修改 query_orders",
        mode="edit",
    )

    assert "query_orders -> build_order_query" in pack.call_chain
    assert "build_order_query -> format_order_response" in pack.call_chain
    assert any(symbol.name == "format_order_response" for symbol in pack.symbols)


def test_build_context_queries_with_jieba_and_mixed_inputs() -> None:
    # 1. Test Chinese tokenization via jieba (length limit relaxed)
    queries = build_context_queries("项目里面改签逻辑有哪些")
    # Verify that it extracted correct segmented words
    assert "改签" in queries
    assert "逻辑" in queries

    # 2. Test mixed Chinese and English inputs
    queries_mixed = build_context_queries("修改 query_orders 改签状态")
    assert "query_orders" in queries_mixed
    assert "改签" in queries_mixed
    assert "状态" in queries_mixed

    # 3. Test empty-result fallback query
    queries_fallback = build_context_queries("  ")
    assert isinstance(queries_fallback, list)

    queries_fallback_2 = build_context_queries("a")
    assert "a" in queries_fallback_2

