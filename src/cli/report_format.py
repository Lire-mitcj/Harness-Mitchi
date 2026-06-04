from __future__ import annotations

import re

from src.llm.dsml import strip_dsml_text

_TABLE_SEP = re.compile(r"^\|[\s\-:|]+\|$")
_MARKDOWN_BOLD = re.compile(r"\*\*(.*?)\*\*")
_NUMBERED_LABEL = re.compile(r"^\d+\.\s*(?:\*\*)?(.*?)(?:\*\*)?[：:]\s*(.*)$")


def summarize_line(text: str, *, max_len: int = 140) -> str:
    one_line = " ".join((text or "").split())
    if len(one_line) <= max_len:
        return one_line
    return one_line[: max_len - 1] + "…"


def _parse_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


_SKIP_SECTION_HEADERS = frozenset({
    "acceptance criteria status",
    "next subtask (for the orchestrator)",
    "next subtask",
    "what i found",
    "key insight",
    "relevant files and evidence",
    "发现总结",
    "完成条件",
})

_KIND_ZH = {
    "diagnose": "诊断",
    "edit": "编辑",
    "verify": "验证",
    "shell": "命令",
}


def format_report_for_terminal(text: str) -> str:
    """Convert markdown-heavy executor output to plain terminal text."""
    text = strip_dsml_text(text)
    lines_out: list[str] = []
    table_headers: list[str] = []

    for raw in (text or "").splitlines():
        stripped = raw.strip()
        if not stripped:
            if lines_out and lines_out[-1] != "":
                lines_out.append("")
            table_headers = []
            continue

        lower = stripped.lower().rstrip(":")
        if lower in _SKIP_SECTION_HEADERS or lower.startswith("diagnosis summary"):
            table_headers = []
            continue
        if stripped in _SKIP_SECTION_HEADERS:
            table_headers = []
            continue

        stripped = _strip_inline_markdown(stripped)

        if stripped.startswith("|"):
            if _TABLE_SEP.match(stripped):
                continue
            cells = _parse_table_row(stripped)
            if not cells:
                continue
            if not table_headers:
                table_headers = cells
                continue
            if len(cells) >= 3:
                lines_out.append(f"  • {cells[0]} — {cells[1]} ({cells[2]})")
            elif len(cells) == 2:
                lines_out.append(f"  • {cells[0]} — {cells[1]}")
            else:
                lines_out.append(f"  • {cells[0]}")
            continue

        table_headers = []
        if stripped.startswith("#"):
            title = re.sub(r"^#+\s*", "", stripped).strip()
            if title:
                if lines_out:
                    lines_out.append("")
                lines_out.append(title)
            continue
        numbered = _NUMBERED_LABEL.match(stripped)
        if numbered:
            label, rest = numbered.groups()
            lines_out.append(f"• {label.strip()}: {rest.strip()}")
            continue
        if stripped.startswith(("✅", "✓")):
            lines_out.append("✓ " + stripped.lstrip("✅✓ ").strip())
            continue
        if stripped.startswith(("• ", "- ", "* ")):
            lines_out.append(f"  {stripped.lstrip('*-• ')}")
            continue
        lines_out.append(stripped)

    while lines_out and lines_out[-1] == "":
        lines_out.pop()
    return "\n".join(lines_out)


def milestone_from_report(text: str, *, max_len: int = 72) -> str:
    """One-line subtask completion hint for milestones."""
    cleaned_text = strip_dsml_text(text)
    formatted = format_report_for_terminal(cleaned_text)
    bullet_lines: list[str] = []
    for line in formatted.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        if cleaned.startswith("•"):
            bullet_lines.append(cleaned.lstrip("• ").strip())
            continue
        if cleaned.endswith(":") or cleaned.endswith("—"):
            continue
        if cleaned.startswith(("✅", "✓")):
            return summarize_line(cleaned, max_len=max_len)
    if bullet_lines:
        return summarize_line(bullet_lines[0], max_len=max_len)
    for line in formatted.splitlines():
        cleaned = line.strip()
        if cleaned and len(cleaned) >= 20 and not cleaned.endswith(":"):
            return summarize_line(cleaned, max_len=max_len)
    return summarize_line(cleaned_text, max_len=max_len)


def compact_diagnose_report(text: str, *, max_lines: int = 10) -> str:
    """Short plain-text diagnose report for terminal — drop orchestrator meta boilerplate."""
    formatted = format_report_for_terminal(text)
    kept: list[str] = []
    for line in formatted.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        lower = cleaned.lower().rstrip(":")
        if lower in _SKIP_SECTION_HEADERS or lower.startswith("diagnosis summary"):
            continue
        if cleaned.startswith(("✅ Met", "✓ Met", "Met —")):
            continue
        kept.append(cleaned if cleaned.startswith("  ") else cleaned)
        if len(kept) >= max_lines:
            break
    return "\n".join(kept)


def _strip_inline_markdown(text: str) -> str:
    return _MARKDOWN_BOLD.sub(r"\1", text).replace("`", "")
