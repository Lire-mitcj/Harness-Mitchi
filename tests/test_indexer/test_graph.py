from src.indexer.graph import build_module_index, resolve_import_targets


def test_resolve_import_full_module_path() -> None:
    file_nodes = {
        "src/planner/task_tree.py": "file:src/planner/task_tree.py",
        "src/utils.py": "file:src/utils.py",
    }
    module_index = build_module_index(file_nodes)
    hits = resolve_import_targets(
        "src.planner.task_tree",
        module_index=module_index,
        file_nodes=file_nodes,
    )
    assert hits == ["file:src/planner/task_tree.py"]


def test_resolve_import_stem_fallback_capped() -> None:
    file_nodes = {
        "a/utils.py": "file:a/utils.py",
        "b/utils.py": "file:b/utils.py",
        "c/utils.py": "file:c/utils.py",
        "d/utils.py": "file:d/utils.py",
    }
    module_index = build_module_index(file_nodes)
    hits = resolve_import_targets(
        "unknown.mod",
        module_index=module_index,
        file_nodes=file_nodes,
        max_targets=2,
    )
    assert len(hits) <= 2
