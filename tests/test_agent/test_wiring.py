from __future__ import annotations

from src.agent.wiring import (
    integration_layer,
    mount_setup_confirmed,
    peer_layers_skip_wiring,
    task_needs_integration_wiring,
    wiring_probe_symbols,
)


def test_task_needs_integration_wiring_opt_out_for_mention_yaml() -> None:
    task = "更新 noise_policy.yaml 的 bot_nicknames，修复 @艾特 检测"
    assert not task_needs_integration_wiring(task)


def test_task_needs_integration_wiring_opt_in_for_mount_task() -> None:
    task = "把 list.py 的 router 挂到 main.py include_router"
    assert task_needs_integration_wiring(task)


def test_integration_layer_classifies_entry_and_infra_paths() -> None:
    assert integration_layer("cmd/server/main.go") == "entry"
    assert integration_layer("agentmesh_orchestrator/conf/noise_policy.yaml") == "infrastructure"
    assert integration_layer("app/interfaces/grpc_server.py") == "handler"
    assert peer_layers_skip_wiring("agentmesh_orchestrator/internal/rules.py")


def test_mount_setup_confirmed_uses_language_profile_patterns() -> None:
    py_code = "app = FastAPI()\napp.include_router(router)\n"
    assert mount_setup_confirmed(py_code, file_path="main.py")
    go_code = "func main() {\n    http.Handle(\"/\", handler)\n}\n"
    assert mount_setup_confirmed(go_code, file_path="cmd/main.go")


def test_wiring_probe_symbols_follow_language_profile() -> None:
    assert "create_app" in wiring_probe_symbols("main.py")
    assert "main" in wiring_probe_symbols("cmd/server/main.go")
    assert "Application" in wiring_probe_symbols("src/Application.java")
