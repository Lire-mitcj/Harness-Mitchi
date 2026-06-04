from __future__ import annotations

from src.harness.edit.target import EditTarget


def format_edit_targets_block(targets: tuple[EditTarget, ...] | list[EditTarget]) -> str:
    if not targets:
        return ""
    lines = [
        "Edit targets (Harness-resolved — use replace_symbol, NOT edit_file):",
        "",
    ]
    for target in targets[:3]:
        preview = target.original_source
        if len(preview) > 6000:
            preview = preview[:6000] + "\n…[truncated]"
        lines.extend([
            f'<edit_target path="{target.path}" symbol="{target.symbol}" '
            f'lines="{target.start_line}-{target.end_line}" anchor="{target.anchor_hash}">',
            "<original>",
            preview.rstrip("\n"),
            "</original>",
            "</edit_target>",
            "",
        ])
    lines.extend([
        "replace_symbol workflow:",
        "1. Modify the code inside <original> per the subtask.",
        "2. Call replace_symbol(path, symbol, new_body) with the FULL revised block.",
        "3. new_body must differ from <original>; include def/header lines.",
        "Harness verifies anchor hash and re-resolves symbol lines if the file drifted.",
    ])
    return "\n".join(lines)
