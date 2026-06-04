from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.context.retriever import ContextRetriever, build_context_queries


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
