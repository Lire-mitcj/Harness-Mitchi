from pathlib import Path

from src.indexer.ctags import CtagsIndexResult, CtagsSymbol
from src.indexer.proto_links import (
    build_proto_reference_edges,
    generated_go_names,
    scan_proto_definitions,
)
from src.indexer.repo_map import build_repo_map, _symbol_id


def test_scan_proto_definitions_extracts_service_and_packages(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "order.proto").write_text(
        'option go_package = "example.com/demo/api;apiv1";\n'
        "service OrderService {\n"
        "  rpc GetOrder(GetOrderRequest) returns (Order);\n"
        "}\n"
        "message Order {\n"
        "  string id = 1;\n"
        "}\n",
        encoding="utf-8",
    )

    defs = scan_proto_definitions(tmp_path)
    by_name = {item.name: item for item in defs}
    assert by_name["OrderService"].kind == "service"
    assert by_name["OrderService"].go_package == "example.com/demo/api"
    assert by_name["Order"].kind == "message"
    assert generated_go_names("OrderService", "service") == [
        "OrderService",
        "OrderServiceClient",
        "OrderServiceServer",
        "UnimplementedOrderServiceServer",
        "NewOrderServiceClient",
        "RegisterOrderServiceServer",
    ]


def test_build_proto_reference_edges_links_proto_to_generated_go(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "order.proto").write_text(
        "service OrderService {}\n",
        encoding="utf-8",
    )
    (tmp_path / "api" / "order_grpc.pb.go").write_text(
        "package apiv1\n\n"
        "type OrderServiceClient struct{}\n"
        "type OrderServiceServer interface{}\n"
        "type UnimplementedOrderServiceServer struct{}\n",
        encoding="utf-8",
    )

    go_sym = CtagsSymbol(
        file_path="api/order_grpc.pb.go",
        name="OrderServiceClient",
        kind="struct",
        start_line=3,
        end_line=3,
        signature="type OrderServiceClient struct{}",
    )
    symbol_nodes = {
        ("api/order.proto", "OrderService", 1): "api/order.proto:OrderService:1",
        (go_sym.file_path, go_sym.name, go_sym.start_line): _symbol_id(go_sym),
    }
    name_to_ids = {
        "OrderService": ["api/order.proto:OrderService:1"],
        "OrderServiceClient": [_symbol_id(go_sym)],
    }
    file_nodes = {
        "api/order.proto": "file:api/order.proto",
        "api/order_grpc.pb.go": "file:api/order_grpc.pb.go",
    }

    edges = build_proto_reference_edges(
        tmp_path,
        symbol_nodes=symbol_nodes,
        name_to_ids=name_to_ids,
        file_nodes=file_nodes,
    )
    assert ("api/order.proto:OrderService:1", _symbol_id(go_sym)) in edges
    assert (_symbol_id(go_sym), "api/order.proto:OrderService:1") in edges


def test_build_repo_map_includes_proto_reference_edges(tmp_path: Path) -> None:
    (tmp_path / "api").mkdir()
    (tmp_path / "api" / "order.proto").write_text(
        "service OrderService {}\n",
        encoding="utf-8",
    )
    (tmp_path / "api" / "order_grpc.pb.go").write_text(
        "package apiv1\n\n"
        "type OrderServiceClient struct{}\n",
        encoding="utf-8",
    )

    indexed = CtagsIndexResult(
        symbols=[
            CtagsSymbol(
                file_path="api/order_grpc.pb.go",
                name="OrderServiceClient",
                kind="struct",
                start_line=3,
                end_line=3,
                signature="type OrderServiceClient struct{}",
            )
        ],
        references=[],
        source="test",
    )
    repo_map = build_repo_map(tmp_path, top_k=20, indexed=indexed)

    proto_sid = next(
        sym.symbol_id
        for sym in repo_map.all_symbols
        if sym.file_path == "api/order.proto" and sym.name == "OrderService"
    )
    go_sid = next(
        sym.symbol_id
        for sym in repo_map.all_symbols
        if sym.name == "OrderServiceClient"
    )
    assert (proto_sid, go_sid) in repo_map.reference_edges
