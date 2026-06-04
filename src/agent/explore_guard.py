from __future__ import annotations

import re

_GREP_QUERY_LINE = re.compile(
    r"^\s*-\s+'(?P<pattern>.*?)'\s+in\s+(?P<scope>.+?)\s*$",
    re.MULTILINE,
)
_READ_RANGE_LINE = re.compile(
    r"^\s*-\s+(?P<label>[\w./-]+(?::\d+(?:-\d+)?)?)\s*$",
    re.MULTILINE,
)
_MAP_QUERY_LINE = re.compile(
    r"^\s*-\s+(?P<query>.+?)\s*$",
    re.MULTILINE,
)


def normalize_grep_query(
    pattern: str,
    *,
    path: str | None = None,
    glob: str | None = None,
    include: str | None = None,
) -> str:
    p = _canonical_grep_pattern(pattern)
    norm_path, norm_glob = _normalize_grep_scope(path, glob=glob, include=include)
    scope = norm_path or norm_glob or "*"
    return f"grep:{p}@{scope}"


def _canonical_grep_pattern(pattern: str) -> str:
    p = " ".join((pattern or "").strip().split())
    # Models often vary harmless regex escaping across ReAct turns. Canonicalize
    # common literal punctuation so duplicate-search detection still works.
    replacements = {
        r"\"": '"',
        r"\(": "(",
        r"\)": ")",
        r"\/": "/",
    }
    for old, new in replacements.items():
        p = p.replace(old, new)
    return p


def _normalize_grep_scope(
    path: str | None,
    *,
    glob: str | None = None,
    include: str | None = None,
) -> tuple[str | None, str | None]:
    """Canonicalize path/glob so `./app.py` and `app.py` dedup together."""
    if isinstance(path, str) and path.strip():
        rel = path.strip().replace("\\", "/").lstrip("./")
        if rel in {"", "."}:
            g = glob or include
            if isinstance(g, str) and g.strip():
                return None, g.strip().replace("\\", "/")
            return None, "*"
        if "*" in rel or "?" in rel:
            return None, rel
        return rel, None
    g = glob or include
    if isinstance(g, str) and g.strip():
        return None, g.strip().replace("\\", "/")
    return None, "*"


def parse_read_path_with_lines(raw: str) -> tuple[str, int | None, int | None]:
    """Split ``app.py:1580-1660`` into path + optional line range."""
    text = raw.strip().replace("\\", "/")
    match = re.match(r"^(?P<path>.+?):(?P<start>\d+)(?:-(?P<end>\d+))?$", text)
    if not match:
        return text, None, None
    end_s = match.group("end")
    return (
        match.group("path"),
        int(match.group("start")),
        int(end_s) if end_s else None,
    )


def normalize_map_query(query: str) -> str:
    q = " ".join((query or "").strip().split())
    return f"map:{q.casefold()}"


def normalize_read_range(
    path: str,
    *,
    start_line: int | None = None,
    end_line: int | None = None,
) -> str:
    rel = path.replace("\\", "/").lstrip("./")
    if start_line is not None and end_line is not None:
        return f"read:{rel}:{start_line}-{end_line}"
    if start_line is not None:
        return f"read:{rel}:{start_line}+"
    return f"read:{rel}"


class ExploreCommandTracker:
    """Block duplicate grep/read exploration within one subtask run."""

    def __init__(
        self,
        *,
        grep_dedup_limit: int = 1,
        read_dedup_limit: int = 1,
        map_dedup_limit: int = 1,
    ) -> None:
        self.grep_dedup_limit = max(1, grep_dedup_limit)
        self.read_dedup_limit = max(1, read_dedup_limit)
        self.map_dedup_limit = max(1, map_dedup_limit)
        self._grep_counts: dict[str, int] = {}
        self._read_counts: dict[str, int] = {}
        self._map_counts: dict[str, int] = {}

    def seed_from_digest(self, digest: str) -> None:
        """Treat digest-listed queries/ranges as already executed (retry handoff)."""
        if not digest.strip():
            return
        for match in _GREP_QUERY_LINE.finditer(digest):
            scope = match.group("scope").strip()
            norm_path, norm_glob = _normalize_grep_scope(scope)
            key = normalize_grep_query(
                match.group("pattern"),
                path=norm_path,
                glob=norm_glob,
            )
            self._grep_counts[key] = max(self._grep_counts.get(key, 0), self.grep_dedup_limit)
        section = _extract_section(digest, "Line ranges already read:")
        if section:
            for match in _READ_RANGE_LINE.finditer(section):
                label = match.group("label")
                key = f"read:{label.replace(chr(92), '/')}"
                self._read_counts[key] = max(self._read_counts.get(key, 0), self.read_dedup_limit)
        section = _extract_section(digest, "Files already read:")
        if section:
            for part in section.replace("  - ", "").split(","):
                rel = part.strip().replace("\\", "/").lstrip("./")
                if rel:
                    key = f"read:{rel}"
                    self._read_counts[key] = max(self._read_counts.get(key, 0), self.read_dedup_limit)
        section = _extract_section(digest, "Repo map searches already run:")
        if section:
            for match in _MAP_QUERY_LINE.finditer(section):
                key = normalize_map_query(match.group("query"))
                self._map_counts[key] = max(self._map_counts.get(key, 0), self.map_dedup_limit)

    def check_grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob: str | None = None,
        include: str | None = None,
    ) -> str | None:
        norm_path, norm_glob = _normalize_grep_scope(path, glob=glob, include=include)
        key = normalize_grep_query(pattern, path=norm_path, glob=norm_glob)
        runs = self._grep_counts.get(key, 0)
        if runs >= self.grep_dedup_limit:
            scope_label = norm_path or norm_glob or "*"
            return (
                f"Blocked duplicate grep_search: {pattern!r} in {scope_label} "
                f"already run this subtask. Use the session summary / prior read output, "
                "then edit_file with a unique multi-line old_string."
            )
        return None

    def check_map(self, query: str) -> str | None:
        key = normalize_map_query(query)
        runs = self._map_counts.get(key, 0)
        if runs >= self.map_dedup_limit:
            return (
                f"Blocked duplicate map_search: {query!r} already run this subtask. "
                "Use the session summary / repo map hits, then edit_file with a "
                "unique multi-line old_string."
            )
        return None

    def record_map(self, query: str) -> None:
        key = normalize_map_query(query)
        self._map_counts[key] = self._map_counts.get(key, 0) + 1

    def record_grep(
        self,
        pattern: str,
        *,
        path: str | None = None,
        glob: str | None = None,
        include: str | None = None,
    ) -> None:
        norm_path, norm_glob = _normalize_grep_scope(path, glob=glob, include=include)
        key = normalize_grep_query(pattern, path=norm_path, glob=norm_glob)
        self._grep_counts[key] = self._grep_counts.get(key, 0) + 1

    def seed_read_slices(self, slices: dict[str, tuple[int, int]]) -> None:
        """Mark diagnose/preload line ranges as already read (cross-subtask handoff)."""
        for rel, (start, end) in slices.items():
            key = normalize_read_range(rel, start_line=start, end_line=end)
            self._read_counts[key] = max(self._read_counts.get(key, 0), self.read_dedup_limit)

    def check_read(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> str | None:
        key = normalize_read_range(path, start_line=start_line, end_line=end_line)
        runs = self._read_counts.get(key, 0)
        if runs >= self.read_dedup_limit:
            return (
                f"Blocked duplicate read on '{path}'"
                + (f" lines {start_line}-{end_line}" if start_line else "")
                + ": content is in the session summary. "
                "Proceed with edit_file using a unique old_string (include surrounding lines)."
            )
        return None

    def record_read(
        self,
        path: str,
        *,
        start_line: int | None = None,
        end_line: int | None = None,
    ) -> None:
        key = normalize_read_range(path, start_line=start_line, end_line=end_line)
        self._read_counts[key] = self._read_counts.get(key, 0) + 1


def _extract_section(text: str, header: str) -> str:
    idx = text.find(header)
    if idx < 0:
        return ""
    rest = text[idx + len(header) :]
    out: list[str] = []
    started = False
    for line in rest.splitlines():
        if line.startswith("  - "):
            out.append(line)
            started = True
        elif started:
            break
        elif line.strip():
            break
    return "\n".join(out)
