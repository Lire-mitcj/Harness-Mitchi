from __future__ import annotations

"""Block reads of MitKII internal framework source unless the user explicitly targets it."""

FRAMEWORK_PATH_PREFIXES = (
    "src/harness",
    "src/agent",
    "src/cli",
    "prompts",
)


def normalize_rel_path(path: str) -> str:
    return path.replace("\\", "/").lstrip("./")


def is_blocked_framework_path(rel_path: str) -> bool:
    rel = normalize_rel_path(rel_path)
    for prefix in FRAMEWORK_PATH_PREFIXES:
        if rel == prefix or rel.startswith(f"{prefix}/"):
            return True
    return False


def extract_framework_path_mentions(user_message: str) -> list[str]:
    """Framework paths the user explicitly named in their request."""
    msg = user_message.replace("\\", "/")
    mentioned: list[str] = []
    seen: set[str] = set()

    for token in msg.replace(",", " ").split():
        cleaned = normalize_rel_path(token.strip("\"'`;:()[]{} "))
        if not cleaned or not is_blocked_framework_path(cleaned):
            continue
        key = cleaned.rstrip("/")
        if key not in seen:
            seen.add(key)
            mentioned.append(key)
    return mentioned


def user_allows_framework_path(user_message: str, rel_path: str) -> bool:
    """Allow reading framework code only when the user cited that path or a parent prefix."""
    rel = normalize_rel_path(rel_path)
    allowed_prefixes = extract_framework_path_mentions(user_message)
    if not allowed_prefixes:
        return False

    for prefix in allowed_prefixes:
        if rel == prefix or rel.startswith(f"{prefix}/") or prefix.startswith(f"{rel}/"):
            return True
    return False


def blocked_framework_reads(
    user_message: str,
    paths: list[str],
) -> list[str]:
    """Return framework paths that should be denied for this user turn."""
    blocked: list[str] = []
    for raw in paths:
        rel = normalize_rel_path(raw)
        if is_blocked_framework_path(rel) and not user_allows_framework_path(user_message, rel):
            blocked.append(rel)
    return blocked


def format_framework_read_denial(blocked_paths: list[str]) -> str:
    paths = ", ".join(blocked_paths)
    return (
        f"Read blocked for internal framework path(s): {paths}. "
        "Harness scoring (L0 lint/tests, L1 rubric judge, L2 warnings) runs automatically "
        "after you edit files — do NOT read src/harness, src/agent, src/cli, or prompts source. "
        "Write or edit the user's target file directly using scorer feedback when rewrite starts. "
        "To modify framework code, the user must name the exact path (e.g. src/agent/loop.py) "
        "in their request."
    )


def format_framework_browse_denial(blocked_paths: list[str], tool_name: str) -> str:
    paths = ", ".join(blocked_paths)
    return (
        f"{tool_name} blocked for internal framework path(s): {paths}. "
        "Do not browse or search MitKII framework source (src/harness, src/agent, src/cli, "
        "prompts). Focus on the user's project files. To inspect framework code, the user must "
        "name the exact path in their request."
    )


def collect_browse_paths(tool_name: str, arguments: dict[str, object]) -> list[str]:
    paths: list[str] = []
    if tool_name in {"list_dir", "glob_files", "grep_search"}:
        p = arguments.get("path")
        if isinstance(p, str) and p.strip() and p.strip() not in {".", "./"}:
            paths.append(p)
    return paths


def blocked_framework_browse(
    user_message: str,
    tool_name: str,
    arguments: dict[str, object],
) -> list[str]:
    """Return framework paths that should be denied for list_dir/grep/glob."""
    blocked: list[str] = []
    for raw in collect_browse_paths(tool_name, arguments):
        rel = normalize_rel_path(raw)
        if is_blocked_framework_path(rel) and not user_allows_framework_path(user_message, rel):
            blocked.append(rel)
    return blocked
