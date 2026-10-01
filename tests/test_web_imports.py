"""Every import reachable from ``web/`` must exist in the web IMAGE.

WHY THIS EXISTS. The dispatch deploy crashlooped and was rolled back; the site
was down for about six minutes. The cause was one import:

    /opt/app/config.py:112
    ModuleNotFoundError: No module named 'mse_maxwell_solver'

``web/config.py`` and ``web/meshing.py`` imported the SOLVER package, which
lives in ``dockerfiles/maxwell_solver/`` -- a directory ``dockerfiles/web/
Dockerfile`` does not copy. The image built, pushed, and died at container
start. The whole suite was green, because every test puts the solver directory
on ``sys.path`` before importing anything; nothing in the suite started the web
IMAGE, so nothing asked the first question the container asks.

That is this rig's signature defect with a new costume: a check that cannot fail
looks exactly like a check that passes. Every unit test proves the code WORKS
given the modules; none of them proved the modules are THERE.

WHAT THIS CHECKS, and what it deliberately cannot. It parses every ``web/*.py``
with ``ast`` (not importing, so a module that raises on import is still checked)
and resolves each top-level import against three things:

  1. the standard library,
  2. ``dockerfiles/web/requirements.txt``,
  3. a module file the web Dockerfile actually COPYs.

The COPY list is read OUT of the Dockerfile rather than restated here, because a
list retyped in a test checks the test's own arithmetic and stays green when the
Dockerfile changes -- the failure mode this whole file is about.

It is a static check, so it proves a resolver CAN find each name, not that the
image builds. ``tests/image_smoke.sh`` is the one that builds and runs it, and
that is the check that would have caught this on the first try; this one runs in
a second and catches the same class at commit time.

SCOPE: top-level module names only. A conditional import inside a function (the
app has several, deliberately late) is still parsed -- ``ast.walk`` finds every
``Import``/``ImportFrom`` at any depth, which is where the crashloop's import
would have been caught had the module not also been imported at module level.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
DOCKERFILE = ROOT / "dockerfiles" / "web" / "Dockerfile"
REQUIREMENTS = ROOT / "dockerfiles" / "web" / "requirements.txt"


def _dockerfile_copies() -> dict[str, Path]:
    """Map module name -> the repo file the web Dockerfile copies for it.

    Read from the Dockerfile, not retyped. Only ``COPY`` lines whose source is a
    ``.py`` file matter here, plus the wholesale ``COPY web/ ...``: a directory
    copy makes every module in it importable under its own name.
    """
    copied: dict[str, Path] = {}
    for raw in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line.upper().startswith("COPY "):
            continue
        parts = line.split()
        # COPY <src>... <dest>
        for token in parts[1:-1]:
            src = ROOT / token
            if token.endswith(".py"):
                copied[src.stem] = src
            elif token.endswith("/"):
                for module in src.glob("*.py"):
                    copied[module.stem] = module
    return copied


def _requirement_name(spec: str) -> str:
    """The importable name at the head of a PEP 508 requirement string.

    Split on the first character that can only start a version specifier, an
    extra, or a marker, rather than chaining ``.split()`` calls. The chained
    version this replaces read ``botocore (<1.44.0,>=1.43.103)`` as
    ``botocore (`` -- which is not a module name, so the requirement counted for
    nothing. It passed only because the SAME package appears a second time
    under an extra (``botocore[crt] ...; extra == "crt"``) whose parsing
    happened to work. That is the wrong reason for a check to be green, and it
    is the same mechanism that made this check believe the web image had scipy.

    ``spec`` here has already had its marker stripped by the caller.
    """
    return re.split(r"[\s\[<>=!~;(]", spec.strip(), maxsplit=1)[0]

def _requirement_names() -> set[str]:
    """Every import name the pinned requirements provide, TRANSITIVELY.

    Read from ``pip``'s own metadata for the installed packages rather than from
    the requirement lines, because the interesting cases are exactly the ones
    not named there: the app imports ``numpy`` (which ``trimesh`` requires) and
    ``botocore`` (which ``boto3`` requires), and neither appears in the file. A
    check that only read the file's own names would red-flag both -- and the
    tempting fix for a noisy check is to add the names to an ignore list, which
    is how a check stops checking.

    Asking the installed distribution is also the honest direction: it answers
    what the image WILL contain, rather than what a reader guesses a pinned
    version drags in.
    """
    from importlib.metadata import PackageNotFoundError, distribution

    names: set[str] = set()
    for raw in REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        name = line.split("==")[0].split(">=")[0].split("[")[0].strip()
        names.add(name)
        try:
            for required in distribution(name).requires or ():
                spec, _, marker = required.partition(";")
                if "extra" in marker:
                    # An OPTIONAL extra is NOT installed by a plain pin. Counting
                    # it is how this check came to believe the web image had
                    # scipy -- trimesh declares `scipy; extra == "easy"` -- while
                    # the container died on `import scipy` at the first pc
                    # submission. A marker naming an extra means "not unless you
                    # asked for that extra", so it is not evidence of presence.
                    continue
                names.add(_requirement_name(spec))
        except PackageNotFoundError:
            # Not installed in this interpreter; the name itself still counts,
            # and the parameter names a package rather than failing.
            pass
    # pip's distribution name is not always the import name.
    for dist_name, import_name in (("psycopg2-binary", "psycopg2"),
                                   ("pyyaml", "yaml"), ("pillow", "PIL")):
        if dist_name in names:
            names.add(import_name)
    # A distribution name imports under a normalised spelling: '-' and '.' are
    # '_' in a module name.
    return {n.replace("-", "_").replace(".", "_") for n in names if n}


def _web_modules() -> dict[str, Path]:
    return {p.stem: p for p in sorted(WEB.glob("*.py"))}


def _top_level_imports(path: Path) -> set[str]:
    """Every top-level module name imported ANYWHERE in ``path``.

    ``ast.walk``, not ``tree.body``: the imports this check exists for are the
    late ones inside functions as much as the module-level ones, and a scan that
    only read the top level would miss exactly the shape that crashed.
    """
    found: set[str] = set()
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            # level>0 is a relative import, which resolves inside the package.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def test_the_scan_can_actually_see_the_web_modules():
    """A positive control on the discovery, before trusting any negative.

    If ``web/`` were renamed, or the glob changed, ``_web_modules()`` would come
    back empty, the parametrised test below would collect nothing, and the suite
    would report success for having checked nothing -- the exact failure this
    file exists to prevent, one level up.
    """
    modules = _web_modules()
    assert len(modules) >= 8, (
        f"only {len(modules)} modules found under {WEB}; this scan would pass "
        "vacuously")
    assert "app" in modules and "config" in modules, (
        f"expected app.py and config.py among {sorted(modules)}")


@pytest.mark.parametrize("name", sorted(_web_modules()))
def test_every_web_import_resolves_inside_the_web_image(name: str) -> None:
    """No web import may require a module the web Dockerfile does not ship."""
    module = _web_modules()[name]
    available = (set(sys.stdlib_module_names) | _requirement_names()
                 | set(_dockerfile_copies()) | set(_web_modules()))
    unresolved = sorted(
        imported for imported in _top_level_imports(module)
        if imported not in available
    )
    assert not unresolved, (
        f"web/{module.name} imports {unresolved}, which is neither stdlib, nor "
        f"in {REQUIREMENTS.name}, nor a file the web Dockerfile copies "
        f"(copies: {sorted(_dockerfile_copies())}). The web image would build "
        f"and then die at container start with ModuleNotFoundError -- the "
        f"dispatch crashloop. Add the module to the shared mse_common (which "
        f"both images COPY) or to the Dockerfile, not to a sys.path dance."
    )


def test_the_solver_package_is_not_reachable_from_the_web_image():
    """The specific import that took the site down, asserted directly.

    The generic check above would catch it, but this states the rule in the
    words of the incident: the web image does not contain the solver package,
    so nothing under ``web/`` may import it. A future ``mse_common`` refactor
    that moved the shared pieces BACK into ``mse_meshing`` would pass the
    generic check for a solver-directory module and fail here.
    """
    solver_modules = {"mse_maxwell_solver", "mse_meshing", "mse_fem",
                      "mse_eigen", "mse_fields", "mse_storage", "mse_db"}
    offenders: list[str] = []
    for module in _web_modules().values():
        for imported in sorted(_top_level_imports(module) & solver_modules):
            offenders.append(f"web/{module.name} -> {imported}")
    assert not offenders, (
        "the web image does not ship the solver package, but these web modules "
        f"import it: {offenders}. Put what they need in mse_common, which both "
        "Dockerfiles COPY."
    )
