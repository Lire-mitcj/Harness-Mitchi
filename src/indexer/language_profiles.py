"""Language profiles for grep, view, and indexer symbol extraction.

Each profile bundles regex dialects for definition/import discovery, symbol
extraction from match lines, and optional mount/wiring heuristics. Tools select
profiles from file extension or project stack instead of hard-coding Python.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Sequence


@dataclass(frozen=True, slots=True)
class LanguageProfile:
    """Per-language patterns used by grep/view/indexer enrichment."""

    id: str
    extensions: frozenset[str]
    # grep symbol mode: templates with {name} placeholder (re.escape applied by caller)
    symbol_mode_templates: tuple[str, ...]
    # grep import mode
    import_mode_templates: tuple[str, ...]
    # line classifiers (no placeholder)
    definition_line_re: re.Pattern[str]
    import_line_re: re.Pattern[str]
    mount_line_res: tuple[re.Pattern[str], ...] = ()
    # extract_symbol_from_match_line — each pattern must have one capture group
    symbol_extractors: tuple[re.Pattern[str], ...] = ()
    # view_symbol_code block starts — group 1 is symbol name when present
    block_start_res: tuple[re.Pattern[str], ...] = ()
    default_globs: tuple[str, ...] = ()
    discovery_patterns: tuple[str, ...] = ()
    wiring_probe_symbols: tuple[str, ...] = ()


def _compile(pattern: str, flags: int = 0) -> re.Pattern[str]:
    return re.compile(pattern, flags)


PYTHON = LanguageProfile(
    id="python",
    extensions=frozenset({".py"}),
    symbol_mode_templates=(
        r"\b(?:async\s+def|def|class)\s+{name}\b",
    ),
    import_mode_templates=(
        r"\b(?:import|from)\b.*{name}",
    ),
    definition_line_re=_compile(r"\b(?:async\s+def|def|class)\s+"),
    import_line_re=_compile(r"\b(?:import|from)\s+"),
    mount_line_res=(
        _compile(r"\binclude_router\s*\("),
        _compile(r"\bFastAPI\s*\("),
        _compile(r"\bcreate_app\s*\("),
        _compile(r"\badd_exception_handler\s*\("),
        _compile(r"@(?:app|router)\."),
    ),
    symbol_extractors=(
        _compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
        _compile(
            r"@(?:app|router)\.\w+\([^)]*\).*?\b(?:async\s+def|def)\s+([A-Za-z_][A-Za-z0-9_]*)"
        ),
    ),
    block_start_res=(
        _compile(r"\b(?:async\s+def|def|class)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ),
    default_globs=("*.py",),
    discovery_patterns=(
        "@router\\.",
        "build_router",
        "include_router",
        "FastAPI",
        "create_app",
        "@app\\.exception_handler",
    ),
    wiring_probe_symbols=("create_app", "app", "wire_routes", "application"),
)

GO = LanguageProfile(
    id="go",
    extensions=frozenset({".go"}),
    symbol_mode_templates=(
        r"\bfunc\s+(?:\([^)]*\)\s+)?{name}\b",
        r"\btype\s+{name}\b",
    ),
    import_mode_templates=(
        r'"(?:[^"]*/)?{name}"',
        r"\bimport\s+.*{name}",
    ),
    definition_line_re=_compile(r"\b(?:func|type|struct|interface)\s+"),
    import_line_re=_compile(r"\bimport\s+"),
    mount_line_res=(
        _compile(r"\bhttp\.Handle(?:Func)?\s*\("),
        _compile(r"\bgrpc\.Register\w+Server\s*\("),
        _compile(r"\bRegister\w+Server\s*\("),
    ),
    symbol_extractors=(
        _compile(r"\bfunc\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)"),
        _compile(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ),
    block_start_res=(
        _compile(r"\bfunc\s+(?:\([^)]*\)\s+)?([A-Za-z_][A-Za-z0-9_]*)"),
        _compile(r"\btype\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ),
    default_globs=("*.go",),
    discovery_patterns=(
        "func.*Handler",
        "grpc\\.",
        "http\\.Handle",
        "package main",
    ),
    wiring_probe_symbols=("main", "Run", "NewServer", "Serve", "Start"),
)

JAVA = LanguageProfile(
    id="java",
    extensions=frozenset({".java"}),
    symbol_mode_templates=(
        r"\b(?:class|interface|enum)\s+{name}\b",
        r"\b(?:public|protected|private|static|final|synchronized|abstract)\s+"
        r"[\w<>,\s\[\]]+\s+{name}\s*\(",
    ),
    import_mode_templates=(
        r"\bimport\s+[\w.]*\.{name}\b",
        r"\bimport\s+static\s+[\w.]*\.{name}\b",
    ),
    definition_line_re=_compile(
        r"\b(?:class|interface|enum|@(?:RestController|Service|Repository|Component))\b"
    ),
    import_line_re=_compile(r"\bimport\s+"),
    mount_line_res=(
        _compile(r"@RestController"),
        _compile(r"@RequestMapping"),
        _compile(r"@Autowired"),
        _compile(r"@Bean"),
    ),
    symbol_extractors=(
        _compile(r"\b(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"),
        _compile(
            r"\b(?:public|protected|private)\s+[\w<>,\s\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
        ),
    ),
    block_start_res=(
        _compile(r"\b(?:class|interface|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"),
        _compile(
            r"\b(?:public|protected|private)\s+[\w<>,\s\[\]]+\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("
        ),
    ),
    default_globs=("*.java", "*.xml"),
    discovery_patterns=(
        "@RestController",
        "@Service",
        "@Repository",
        "@Autowired",
        "@GetMapping",
    ),
    wiring_probe_symbols=("Application", "main", "SpringBootApplication"),
)

PROTO = LanguageProfile(
    id="proto",
    extensions=frozenset({".proto"}),
    symbol_mode_templates=(
        r"\b(?:service|message|enum|rpc)\s+{name}\b",
    ),
    import_mode_templates=(
        r'import\s+"[^"]*{name}',
    ),
    definition_line_re=_compile(r"\b(?:service|message|enum|rpc)\s+"),
    import_line_re=_compile(r'\bimport\s+"'),
    symbol_extractors=(
        _compile(r"\b(?:service|message|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"),
        _compile(r"\brpc\s+([A-Za-z_][A-Za-z0-9_]*)\s*\("),
    ),
    block_start_res=(
        _compile(r"\b(?:service|message|enum)\s+([A-Za-z_][A-Za-z0-9_]*)"),
    ),
    default_globs=("*.proto",),
    discovery_patterns=(
        "service ",
        "rpc ",
        "message ",
        "option go_package",
    ),
)

SQL = LanguageProfile(
    id="sql",
    extensions=frozenset({".sql"}),
    symbol_mode_templates=(),
    import_mode_templates=(),
    definition_line_re=_compile(
        r"\bCREATE\s+(?:OR\s+REPLACE\s+)?(?:TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER)\b",
        re.IGNORECASE,
    ),
    import_line_re=_compile(r"$^"),
    symbol_extractors=(
        _compile(
            r"(?is)\bCREATE\s+(?:OR\s+REPLACE\s+)?"
            r"(?:DEFINER\s*=\s*\S+\s+)?(?:TEMP\s+|TEMPORARY\s+)?"
            r"(?:TABLE|VIEW|PROCEDURE|FUNCTION|TRIGGER|EVENT)\s+"
            r"(?:IF\s+NOT\s+EXISTS\s+)?"
            r"(?:`([^`]+)`|\"([^\"]+)\"|'([^']+)'|([A-Za-z_][A-Za-z0-9_]*))"
        ),
    ),
    default_globs=("*.sql",),
    discovery_patterns=("CREATE TABLE", "CREATE VIEW"),
)

ALL_PROFILES: tuple[LanguageProfile, ...] = (PYTHON, GO, JAVA, PROTO, SQL)

_PROFILE_BY_ID: dict[str, LanguageProfile] = {p.id: p for p in ALL_PROFILES}
_EXTENSION_TO_PROFILE: dict[str, LanguageProfile] = {}
for _profile in ALL_PROFILES:
    for _ext in _profile.extensions:
        _EXTENSION_TO_PROFILE[_ext] = _profile


def profile_for_extension(extension: str) -> LanguageProfile | None:
    return _EXTENSION_TO_PROFILE.get(extension.lower())


def profile_for_path(file_path: str) -> LanguageProfile | None:
    if not file_path:
        return None
    suffix = "." + file_path.rsplit(".", 1)[-1] if "." in file_path else ""
    return profile_for_extension(suffix)


def profiles_for_ids(ids: Sequence[str]) -> tuple[LanguageProfile, ...]:
    out: list[LanguageProfile] = []
    seen: set[str] = set()
    for lang_id in ids:
        profile = _PROFILE_BY_ID.get(lang_id)
        if profile is None or profile.id in seen:
            continue
        seen.add(profile.id)
        out.append(profile)
    return tuple(out)


def searchable_extensions(profile_ids: Sequence[str]) -> frozenset[str]:
    exts: set[str] = set()
    for lang_id in profile_ids:
        profile = _PROFILE_BY_ID.get(lang_id)
        if profile is not None:
            exts.update(profile.extensions)
    return frozenset(exts)


def build_symbol_mode_pattern(name: str, profiles: Sequence[LanguageProfile]) -> str:
    escaped = re.escape(name)
    alts: list[str] = []
    for profile in profiles:
        for template in profile.symbol_mode_templates:
            alts.append(template.format(name=escaped))
    if not alts:
        return rf"\b(?:async\s+def|def|class)\s+{escaped}\b"
    return "(?:" + "|".join(alts) + ")"


def build_import_mode_pattern(name: str, profiles: Sequence[LanguageProfile]) -> str:
    escaped = re.escape(name)
    alts: list[str] = []
    for profile in profiles:
        for template in profile.import_mode_templates:
            alts.append(template.format(name=escaped))
    if not alts:
        return rf"\b(?:import|from)\b.*{escaped}"
    return "(?:" + "|".join(alts) + ")"


def profiles_from_include_glob(include: str | None) -> tuple[LanguageProfile, ...] | None:
    """Narrow active profiles when grep include glob targets one language."""
    if not include:
        return None
    lowered = include.casefold()
    matched: list[LanguageProfile] = []
    for profile in ALL_PROFILES:
        for glob in profile.default_globs:
            if glob.strip("*").casefold() in lowered or lowered.endswith(
                glob.lstrip("*").casefold()
            ):
                matched.append(profile)
                break
    return tuple(dict.fromkeys(matched)) if matched else None


@lru_cache(maxsize=1)
def default_profiles() -> tuple[LanguageProfile, ...]:
    return ALL_PROFILES


def extract_symbol_with_profiles(
    content: str,
    profiles: Sequence[LanguageProfile],
) -> str:
    for profile in profiles:
        for pattern in profile.symbol_extractors:
            match = pattern.search(content)
            if not match:
                continue
            for group in match.groups():
                if group:
                    return group
    return ""


def classify_line_with_profiles(
    line: str,
    *,
    pattern: str,
    profiles: Sequence[LanguageProfile],
) -> str:
    text = str(line or "").strip()
    if not text:
        return "usage"
    for profile in profiles:
        if profile.definition_line_re.search(text):
            return "definition"
    if SQL.definition_line_re.search(text):
        return "schema"
    for profile in profiles:
        for mount_re in profile.mount_line_res:
            if mount_re.search(text):
                return "mount"
    for profile in profiles:
        if profile.import_line_re.search(text):
            return "import"
    leaf = str(pattern or "").split(".")[-1].strip()
    if leaf and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", leaf):
        if re.search(rf"\b{re.escape(leaf)}\s*\(", text):
            if not any(p.definition_line_re.search(text) for p in profiles):
                return "call_site"
    return "usage"
