from __future__ import annotations

import re
from pathlib import Path
from typing import Any

def normalize_grep_search_args(
    arguments: dict[str, Any],
    *,
    subtask: Any = None,
    hint_text: str = "",
) -> dict[str, Any]:
    """Fill missing grep_search.pattern so small models cannot brick the tool call."""
    args = dict(arguments)
    pattern = args.get("pattern")
    if isinstance(pattern, str) and pattern.strip():
        return args

    blob = hint_text
    if subtask is not None:
        blob = " ".join(
            part
            for part in (
                subtask.description,
                subtask.acceptance_criteria or "",
                " ".join(subtask.context_files or []),
                hint_text,
            )
            if part
        )

    tokens: list[str] = []
    for word in (
        "register",
        "signup",
        "transaction",
        "commit",
        "rollback",
        "session",
        "sp_register",
        "passenger",
    ):
        if word in blob.lower():
            tokens.append(word)

    for match in re.finditer(r"\b(sp_[A-Za-z0-9_]+)\b", blob):
        tokens.append(match.group(1))

    if subtask and subtask.context_files and "path" not in args:
        args["path"] = subtask.context_files[0]

    args.setdefault("path", ".")
    args["pattern"] = "|".join(dict.fromkeys(tokens)) if tokens else r"def |commit|rollback|register"
    return args


def normalize_shell_exec_args(
    arguments: dict[str, Any],
    *,
    project_root: Path,
) -> dict[str, Any]:
    """Default shell cwd to project_root; reject hallucinated paths like /workspace."""
    args = dict(arguments)
    root = project_root.resolve()
    wd = args.get("working_dir")
    if not isinstance(wd, str) or not wd.strip():
        args["working_dir"] = str(root)
        return args
    candidate = Path(wd).expanduser()
    if not candidate.is_absolute():
        candidate = (root / candidate).resolve()
    else:
        candidate = candidate.resolve()
    if candidate.is_dir():
        args["working_dir"] = str(candidate)
    else:
        args["working_dir"] = str(root)
    return args
