from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.planner.kinds import SubTaskKind
from src.planner.task_tree import SubTaskNode, SubTaskStatus, TaskTree
from src.planner.tool_policy import default_allowed_tools, normalize_allowed_tools

_REQUIRED_NODE_FIELDS = (
    "id",
    "kind",
    "description",
    "acceptance_criteria",
    "allowed_tools",
    "context_files",
    "depends_on",
)
_VALID_KINDS = frozenset(k.value for k in SubTaskKind)


@dataclass
class PlannerParseResult:
    """Outcome of parsing Planner LLM output before PlanGate."""

    raw: str
    payload: dict[str, Any]
    tree: TaskTree
    json_ok: bool = False
    schema_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.json_ok and not self.schema_errors

    @property
    def all_errors(self) -> list[str]:
        errors: list[str] = []
        if not self.json_ok:
            errors.append(
                "Invalid JSON — output ONE raw object with double-quoted keys, "
                'e.g. {"root_task":"...","nodes":[...]}. No markdown fences or prose.'
            )
        errors.extend(self.schema_errors)
        return errors


def normalize_planner_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Fill common small-model omissions before schema validation."""
    if not isinstance(payload, dict):
        return payload

    out = dict(payload)
    nodes_in = out.get("nodes")
    if not isinstance(nodes_in, list):
        return out

    nodes_out: list[Any] = []
    for idx, item in enumerate(nodes_in):
        if not isinstance(item, dict):
            nodes_out.append(item)
            continue
        node = dict(item)
        node["id"] = str(node.get("id") or f"st-{idx + 1}").strip() or f"st-{idx + 1}"

        if "allowed_tools" not in node and "tools" in node:
            node["allowed_tools"] = node["tools"]

        deps = node.get("depends_on")
        if not isinstance(deps, list):
            node["depends_on"] = [] if idx == 0 else [_previous_node_id(nodes_out, idx)]
        elif idx > 0 and not deps:
            node["depends_on"] = [_previous_node_id(nodes_out, idx)]

        if not isinstance(node.get("context_files"), list):
            node["context_files"] = []
        if not isinstance(node.get("allowed_tools"), list):
            node["allowed_tools"] = []
        if not node["allowed_tools"]:
            try:
                node["allowed_tools"] = default_allowed_tools(SubTaskKind(str(node.get("kind") or "")))
            except ValueError:
                node["allowed_tools"] = []
        if not isinstance(node.get("handoff_outputs"), list):
            node["handoff_outputs"] = []
        if not isinstance(node.get("requires_handoff"), list):
            node["requires_handoff"] = []

        crit = node.get("acceptance_criteria")
        if not isinstance(crit, str) or not crit.strip():
            desc = str(node.get("description") or "Complete subtask")
            node["acceptance_criteria"] = desc[:120]

        if "needs_l1" not in node:
            node["needs_l1"] = False

        for key in ("description", "acceptance_criteria"):
            value = node.get(key)
            if isinstance(value, str) and len(value) > 120:
                node[key] = value[:117].rstrip() + "..."

        nodes_out.append(node)

    _repair_sequential_dependencies(nodes_out)

    out["nodes"] = nodes_out
    root = out.get("root_task")
    if (not isinstance(root, str) or not root.strip()) and nodes_out:
        first = nodes_out[0]
        if isinstance(first, dict):
            out["root_task"] = str(first.get("description") or "Task")[:200]
    return out


def _previous_node_id(nodes: list[Any], idx: int) -> str:
    if idx <= 0:
        return "st-1"
    previous = nodes[idx - 1]
    if isinstance(previous, dict):
        raw_id = str(previous.get("id") or "").strip()
        if raw_id:
            return raw_id
    return f"st-{idx}"


def _repair_sequential_dependencies(nodes: list[Any]) -> None:
    latest_design_id = ""
    latest_edit_id = ""
    for idx, node in enumerate(nodes):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or f"st-{idx + 1}")
        kind = str(node.get("kind") or "")
        deps = [
            str(dep)
            for dep in (node.get("depends_on") or [])
            if isinstance(dep, str) and dep.strip()
        ]
        requires = {
            str(item)
            for item in (node.get("requires_handoff") or [])
            if isinstance(item, str)
        }
        if kind == SubTaskKind.EDIT.value and latest_design_id:
            if "PATCH_INTENT_JSON" in requires or idx > 0:
                if latest_design_id not in deps:
                    deps.append(latest_design_id)
        if kind == SubTaskKind.VERIFY.value and latest_edit_id:
            if latest_edit_id not in deps:
                deps.append(latest_edit_id)
        node["depends_on"] = deps
        if kind == SubTaskKind.DESIGN.value:
            latest_design_id = node_id
        elif kind == SubTaskKind.EDIT.value:
            latest_edit_id = node_id


def validate_planner_payload(payload: dict[str, Any]) -> list[str]:
    """Schema checks on parsed Planner JSON (before TaskTree normalization)."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["Root must be a JSON object."]

    root_task = payload.get("root_task")
    if not isinstance(root_task, str) or not root_task.strip():
        errors.append("root_task must be a non-empty string.")

    nodes = payload.get("nodes")
    if not isinstance(nodes, list):
        errors.append("nodes must be an array.")
        return errors
    if not nodes:
        errors.append("nodes must contain at least one subtask.")
        return errors

    ids: list[str] = []
    for idx, item in enumerate(nodes, start=1):
        label = f"nodes[{idx - 1}]"
        if not isinstance(item, dict):
            errors.append(f"nodes[{idx - 1}] must be object, got {type(item).__name__}: {item!r}")
            continue
        for key in _REQUIRED_NODE_FIELDS:
            if key == "allowed_tools":
                if "allowed_tools" not in item and "tools" not in item:
                    errors.append(f"{label} missing required field 'tools'.")
            else:
                if key not in item:
                    errors.append(f"{label} missing required field '{key}'.")
        node_id = item.get("id")
        if isinstance(node_id, str) and node_id.strip():
            ids.append(node_id.strip())
        kind = str(item.get("kind") or "")
        if kind and kind not in _VALID_KINDS:
            errors.append(f"{label} kind '{kind}' invalid — use diagnose|design|edit|verify|shell.")
        desc = item.get("description")
        if isinstance(desc, str) and len(desc) > 120:
            errors.append(f"{label} description exceeds 120 characters.")
        crit = item.get("acceptance_criteria")
        if isinstance(crit, str) and len(crit) > 120:
            errors.append(f"{label} acceptance_criteria exceeds 120 characters.")
        
        tools = item.get("allowed_tools") or item.get("tools")
        if tools is not None and not isinstance(tools, list):
            errors.append(f"{label} tools must be an array.")
        elif isinstance(tools, list):
            for tool in tools:
                if not isinstance(tool, str):
                    errors.append(f"{label} tools must contain strings only.")
                    break
        for list_field in ("context_files", "depends_on", "handoff_outputs", "requires_handoff"):
            val = item.get(list_field)
            if val is not None and not isinstance(val, list):
                errors.append(f"{label} {list_field} must be an array.")

    if len(ids) != len(set(ids)):
        errors.append("Duplicate subtask id in nodes.")

    for idx, item in enumerate(nodes[1:], start=2):
        if not isinstance(item, dict):
            continue
        deps = item.get("depends_on")
        if not isinstance(deps, list) or not deps:
            errors.append(
                f"nodes[{idx - 1}] must set depends_on (e.g. [\"st-1\"]) when order matters."
            )

    return errors


def build_task_tree_from_payload(
    payload: dict[str, Any],
    *,
    fallback_task: str,
) -> TaskTree:
    """Build TaskTree from payload without silent fallback."""
    nodes: list[SubTaskNode] = []
    for idx, item in enumerate(payload.get("nodes") or [], start=1):
        if not isinstance(item, dict):
            continue
        raw_kind = str(item.get("kind") or SubTaskKind.EDIT.value)
        try:
            kind = SubTaskKind(raw_kind)
        except ValueError:
            kind = SubTaskKind.EDIT
        needs_l1 = item.get("needs_l1")
        raw_tools = item.get("allowed_tools") or item.get("tools")
        tools_list = (
            [str(t) for t in raw_tools if isinstance(t, str)]
            if isinstance(raw_tools, list)
            else []
        )
        nodes.append(
            SubTaskNode(
                id=str(item.get("id") or f"st-{idx}"),
                description=str(item.get("description") or "Unnamed subtask"),
                kind=kind,
                acceptance_criteria=str(item.get("acceptance_criteria") or ""),
                allowed_tools=normalize_allowed_tools(kind, tools_list or None),
                needs_l1=needs_l1 if isinstance(needs_l1, bool) else None,
                context_files=[str(p) for p in item.get("context_files") or []],
                depends_on=[str(d) for d in item.get("depends_on") or []],
                status=SubTaskStatus(str(item.get("status", SubTaskStatus.PENDING))),
                checkpoint_id=item.get("checkpoint_id"),
                handoff_outputs=[str(h) for h in item.get("handoff_outputs") or []],
                requires_handoff=[str(r) for r in item.get("requires_handoff") or []],
            )
        )
    return TaskTree(
        root_task=str(payload.get("root_task") or fallback_task),
        nodes=nodes,
    )
