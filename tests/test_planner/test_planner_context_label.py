from src.planner.planner_node import _PROJECT_CONTEXT_LABEL, PlannerNode


def test_planner_uses_project_context_label() -> None:
    node = PlannerNode(client=object())  # type: ignore[arg-type]
    msgs = node.plan_messages("fix bug", "<repo_map></repo_map>", discovery_manifest=None)
    project_block = next(
        m
        for m in msgs
        if m.get("role") == "system" and _PROJECT_CONTEXT_LABEL in m["content"]
    )
    assert f"{_PROJECT_CONTEXT_LABEL}:" in project_block["content"]
    assert "Project structure:" not in project_block["content"]


def test_planner_messages_include_context_pack_block() -> None:
    node = PlannerNode(client=object())  # type: ignore[arg-type]
    context_pack = (
        '<context_pack source="ContextRetriever">\n'
        "confidence: 0.90\n"
        "relevant_files:\n"
        "- main.py\n"
        "</context_pack>"
    )

    msgs = node.plan_messages(
        "把当前登机牌查询接口改成用视图查询",
        "<repo_map></repo_map>",
        discovery_manifest=context_pack,
    )

    user_block = msgs[-1]["content"]
    assert "<context_pack" in user_block
    assert "main.py" in user_block


def test_patch_plan_messages_use_patch_plan_system_prompt() -> None:
    node = PlannerNode(client=object())  # type: ignore[arg-type]

    msgs = node.patch_plan_messages(
        "把当前登机牌查询接口改成用视图查询",
        "<repo_map></repo_map>",
        context_pack="<context_pack>confidence: 0.90</context_pack>",
    )

    assert "PatchPlan Planner" in msgs[0]["content"]
    assert "No markdown" in msgs[0]["content"]
    assert "Output ONE TaskTree JSON" not in msgs[0]["content"]
    assert '"patch_plan"' in msgs[0]["content"]
    assert "<context_pack>" in msgs[-1]["content"]
