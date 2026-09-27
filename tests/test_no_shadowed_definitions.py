"""No test module may define the same top-level name twice.

WHY THIS EXISTS. ``tests/test_scale_and_feature.py`` carried SIX tests defined
twice -- 34 ``def test_`` lines, 28 collected. Python binds the second
definition and silently discards the first, so pytest ran 28 of the 34 and
reported a clean, healthy file. The shadowed copies had never run at all.

That is the rig's signature defect wearing its most comfortable costume: a test
that never runs looks EXACTLY like a test that passes. Nothing in the output
distinguishes them -- the same green dot, the same count -- and it was found by
reading, not by running. A whole-file duplicate is the easy case; the general
shape is any name that shadows an earlier one in the same module, and the only
place to catch it is before a suite that cannot see it reports success.

WHAT IT CHECKS. Every ``tests/*.py`` is parsed with ``ast`` (not imported: a
module that raises on import must still be checked) and each top-level
``def``/``class`` name must be unique within its file. The failure message
names the file, the duplicated name, and every line it appears on, so the fix
is mechanical.

Scope is deliberately the RIG's own test modules. It is not a general linter:
redefinition is occasionally intentional (a conditional platform override), and
a test that cannot tell those apart would be deleted rather than obeyed.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import pytest

TESTS_DIR = Path(__file__).resolve().parent


def _test_modules() -> list[Path]:
    return sorted(TESTS_DIR.glob("*.py"))


def test_this_check_has_something_to_check() -> None:
    """A positive control for the file list itself.

    The scan below is a loop over ``_test_modules()``: if that returned an empty
    list -- a renamed directory, a changed pattern -- every assertion inside it
    would be skipped and the test would report success for having checked
    nothing. That is the same failure this file exists to prevent, one level up.
    """
    files = _test_modules()
    assert len(files) > 5, (
        f"only {len(files)} test modules found under {TESTS_DIR}; this check "
        "would pass vacuously, which is the defect it exists to catch"
    )
    names = {f.name for f in files}
    assert Path(__file__).name in names, (
        "this file is not among those it scans, so the scan is pointed at the "
        "wrong directory"
    )


@pytest.mark.parametrize("path", _test_modules(), ids=lambda p: p.name)
def test_no_top_level_name_is_defined_twice(path: Path) -> None:
    """Every top-level ``def``/``class`` in ``path`` is bound exactly once."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    seen: dict[str, list[int]] = defaultdict(list)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            seen[node.name].append(node.lineno)

    duplicated = {name: lines for name, lines in seen.items() if len(lines) > 1}
    assert not duplicated, "\n".join(
        f"{path.name}: {name} defined {len(lines)} times at lines "
        f"{', '.join(str(n) for n in lines)} -- the last binding wins and the "
        f"earlier one{'s' if len(lines) > 2 else ''} NEVER RUNS"
        for name, lines in sorted(duplicated.items())
    )
