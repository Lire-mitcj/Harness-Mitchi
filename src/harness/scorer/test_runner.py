from __future__ import annotations

import asyncio
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

FRAMEWORK_MARKERS: dict[str, list[str]] = {
    "pytest": ["pytest.ini", "pyproject.toml", "conftest.py", "setup.cfg"],
    "jest": ["jest.config.js", "jest.config.ts", "jest.config.mjs"],
    "vitest": ["vitest.config.ts", "vitest.config.js"],
    "mocha": [".mocharc.yml", ".mocharc.json"],
}

FRAMEWORK_COMMANDS: dict[str, list[str]] = {
    "pytest": ["python", "-m", "pytest", "--tb=short", "-q"],
    "jest": ["npx", "jest", "--bail"],
    "vitest": ["npx", "vitest", "run"],
    "mocha": ["npx", "mocha"],
}

TEST_FILE_PATTERNS: dict[str, list[str]] = {
    "pytest": ["test_*.py", "*_test.py"],
    "jest": ["*.test.ts", "*.test.tsx", "*.test.js", "*.spec.ts", "*.spec.js"],
    "vitest": ["*.test.ts", "*.test.tsx", "*.spec.ts"],
    "mocha": ["*.test.js", "*.spec.js"],
}


@dataclass(slots=True)
class LayerScore:
    layer: str
    passed: bool
    details: str
    items: list[dict[str, Any]] = field(default_factory=list)


class TestRunner:
    """L0 programmatic scorer: discovers and runs related tests."""

    def __init__(self, project_root: Path | None = None) -> None:
        self._root = project_root or Path.cwd()

    async def run(self, context: Any) -> LayerScore:
        framework = self._detect_test_framework()
        if framework is None:
            return LayerScore(
                layer="L0:test",
                passed=True,
                details="No test framework detected — skipping.",
            )

        changed = getattr(context, "changed_files", []) or []
        related = self._find_related_tests(changed, framework)

        if not related:
            return LayerScore(
                layer="L0:test",
                passed=True,
                details="No related test files found for changed files.",
            )

        cmd = list(FRAMEWORK_COMMANDS[framework]) + related
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=str(self._root),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        except asyncio.TimeoutError:
            return LayerScore(
                layer="L0:test",
                passed=False,
                details=f"Tests timed out after 120s ({framework})",
            )
        except FileNotFoundError:
            return LayerScore(
                layer="L0:test",
                passed=True,
                details=f"{framework} not found on PATH — skipping.",
            )

        output = stdout.decode(errors="replace") + stderr.decode(errors="replace")
        passed = proc.returncode == 0

        return LayerScore(
            layer="L0:test",
            passed=passed,
            details=f"{framework}: {'PASSED' if passed else 'FAILED'} ({len(related)} test files)",
            items=[{"output": output[-3000:], "files": related}],
        )

    def _find_related_tests(
        self,
        changed_files: list[str],
        framework: str,
    ) -> list[str]:
        patterns = TEST_FILE_PATTERNS.get(framework, [])
        test_dir_candidates = ["tests", "test", "__tests__", "spec"]

        all_tests: set[Path] = set()
        for pattern in patterns:
            all_tests.update(self._root.rglob(pattern))
        for d in test_dir_candidates:
            td = self._root / d
            if td.is_dir():
                for pattern in patterns:
                    all_tests.update(td.rglob(pattern))

        if not changed_files:
            return [str(t.relative_to(self._root)) for t in sorted(all_tests)[:20]]

        changed_stems = set()
        for f in changed_files:
            stem = Path(f).stem
            changed_stems.add(stem)
            changed_stems.add(f"test_{stem}")
            changed_stems.add(f"{stem}_test")
            changed_stems.add(f"{stem}.test")
            changed_stems.add(f"{stem}.spec")

        related: list[str] = []
        for test_path in sorted(all_tests):
            if test_path.stem in changed_stems or any(
                cs in test_path.stem for cs in changed_stems
            ):
                related.append(str(test_path.relative_to(self._root)))

        return related[:30]

    def _detect_test_framework(self) -> str | None:
        for framework, markers in FRAMEWORK_MARKERS.items():
            for marker in markers:
                if (self._root / marker).exists():
                    if framework == "pytest" and marker == "pyproject.toml":
                        content = (self._root / marker).read_text(errors="replace")
                        if "[tool.pytest" in content or "pytest" in content.lower():
                            return framework
                        continue
                    return framework

        if shutil.which("pytest"):
            return "pytest"
        return None
