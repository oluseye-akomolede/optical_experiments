"""What the images install, and where they put the DDL.

Two things about a deployed image cannot be checked by running the app from the
checkout, because the checkout has neither the flat layout nor the installed
distribution:

* the dependency pins must match what the image actually contains, or "pin it"
  is a story rather than a guarantee;
* the DDL must be found at a fixed path, not by climbing relative directories,
  which is what broke the first deploy.

These are static tests over the Dockerfile and the requirements files. They do
NOT build anything: a build is minutes of network on the openrig VM, and the
lead owns that. What they can do is fail loudly the moment a `>=` reappears or
the ENV that makes the path canonical is removed.

The pins were read from the built images with `pip freeze` -- see
``PINNED_FROM`` below, which records the tag so the next reader can re-derive
them rather than trusting this file.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB_REQS = ROOT / "dockerfiles" / "web" / "requirements.txt"
SOLVER_REQS = ROOT / "dockerfiles" / "maxwell_solver" / "requirements.txt"
WEB_DOCKERFILE = ROOT / "dockerfiles" / "web" / "Dockerfile"
SOLVER_DOCKERFILE = ROOT / "dockerfiles" / "maxwell_solver" / "Dockerfile"

#: the image tag the pins were read from, for re-derivation:
#:   docker run --rm --entrypoint pip <image>:<tag> freeze
PINNED_FROM = "e1962d0"

#: requirement lines that are deliberately not pinned.
#:
#: Empty, and it should stay that way: every direct dependency of both images
#: was readable from the freeze, so an exception here would be a hole in the
#: guarantee rather than pragmatism. Listed explicitly rather than as a comment
#: so that adding one is a visible act.
ALLOWED_UNPINNED: set[str] = set()


def _requirement_lines(path: Path) -> list[str]:
    """The actual requirement lines: comments and blanks dropped."""
    lines = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            lines.append(line)
    return lines


@pytest.mark.parametrize("path", [WEB_REQS, SOLVER_REQS],
                         ids=["web", "solver"])
def test_no_bare_range_specifiers(path: Path):
    """No `>=`, `>`, `~=`, `<=` or bare `*`: a range specifier is not a pin.

    This is the test the lead asked for. A `>=` means the next build can install
    a different version than the one that was validated, which is exactly the
    class of change that is invisible until something breaks in production.
    """
    offenders = []
    for line in _requirement_lines(path):
        name = re.split(r"[<>=!~]", line, maxsplit=1)[0].strip()
        if name in ALLOWED_UNPINNED:
            continue
        if "==" not in line:
            offenders.append(line)
    assert not offenders, (
        f"{path.name} has unpinned requirements: {offenders}. "
        f"Pin with `==` to the version in the image (see PINNED_FROM)."
    )


@pytest.mark.parametrize("path", [WEB_REQS, SOLVER_REQS],
                         ids=["web", "solver"])
def test_every_direct_requirement_is_pinned_exactly_once(path: Path):
    """Exactly one `==` per line, and no `===` or environment markers.

    `===` is the arbitrary-equality operator, which pins the literal string
    rather than a version; markers like `; python_version < "3.12"` would let a
    different resolution happen than the one that was frozen.
    """
    for line in _requirement_lines(path):
        assert line.count("==") == 1, f"not a single exact pin: {line!r}"
        assert "===" not in line, f"arbitrary equality is not a pin: {line!r}"
        assert ";" not in line, f"environment marker changes resolution: {line!r}"
        assert "!=" not in line, f"exclusion is not a pin: {line!r}"


@pytest.mark.parametrize("path", [WEB_REQS, SOLVER_REQS],
                         ids=["web", "solver"])
def test_no_editable_or_vcs_or_path_requirements(path: Path):
    """A requirement must come from the index, not a local path or a git URL.

    A path requirement would silently resolve to whatever is on the machine
    doing the build, which for an image is the checkout -- so the image would
    depend on the build host in a way the freeze inside it cannot show.
    """
    for line in _requirement_lines(path):
        assert not line.startswith("-"), f"pip option in requirements: {line!r}"
        assert "git+" not in line, f"VCS requirement: {line!r}"
        assert not line.startswith("."), f"path requirement: {line!r}"
        assert "/" not in line.split("==")[0], f"path-like requirement: {line!r}"


def test_the_web_image_pins_flask_and_gunicorn():
    """The two that decide how the app is served must be present and pinned."""
    pinned = dict(
        line.split("==", 1) for line in _requirement_lines(WEB_REQS)
    )
    assert "flask" in pinned
    assert "gunicorn" in pinned


def test_the_solver_image_pins_its_scientific_stack():
    """numpy/scipy/tetgen are the solve; an unpinned one is an unnoticed change."""
    pinned = dict(
        line.split("==", 1) for line in _requirement_lines(SOLVER_REQS)
    )
    for name in ("numpy", "scipy", "tetgen", "pyvista"):
        assert name in pinned, f"{name} must be a pinned direct requirement"


# ---------------------------------------------------------------------------
# the DDL path inside the images
# ---------------------------------------------------------------------------
def test_the_web_dockerfile_declares_the_schema_path():
    """MSE_SCHEMA_PATH makes the DDL location a fact about the image.

    Without it the path is derived by climbing relative directories from
    ``web/db.py``, which is precisely how the first deploy broke: the resolver's
    first candidate is the image layout, and it only worked because the climb
    happened to land on it.
    """
    text = WEB_DOCKERFILE.read_text(encoding="utf-8")
    assert "MSE_SCHEMA_PATH=" in text
    assert "/opt/mse/001_schema.sql" in text


def test_the_web_dockerfile_copies_the_ddl_to_that_path():
    """The declared path must be the path the file is actually copied to."""
    text = WEB_DOCKERFILE.read_text(encoding="utf-8")
    copy_lines = [
        line for line in text.splitlines()
        if line.startswith("COPY") and "db/" in line
    ]
    assert copy_lines, "nothing copies db/ into the web image"
    dest = copy_lines[0].split()[-1].rstrip("/")
    assert dest == "/opt/mse", f"DDL copied to {dest}, but ENV points at /opt/mse"
    assert "/opt/mse/001_schema.sql" == f"{dest}/001_schema.sql"


def test_the_solver_image_does_not_ship_the_ddl():
    """The solver must not carry the schema: it does not apply it.

    Its presence was both dead weight and a standing invitation to think a Job
    could provision its own schema, which is the assumption that produced the
    second deploy failure.
    """
    text = SOLVER_DOCKERFILE.read_text(encoding="utf-8")
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert not (stripped.startswith("COPY") and "db/" in stripped), (
            f"the solver image still copies the DDL: {stripped!r}"
        )


def test_the_web_initcontainer_and_env_agree():
    """End to end across files: the initContainer runs db.apply_schema().

    It resolves through MSE_SCHEMA_PATH, which the Dockerfile sets to a file the
    Dockerfile also copies. If either half moves, this fails rather than
    producing an image that crash-loops at startup.
    """
    import sys

    sys.path.insert(0, str(ROOT / "web"))
    import db as WEB_DB  # noqa: PLC0415

    dockerfile = WEB_DOCKERFILE.read_text(encoding="utf-8")
    env_line = next(
        line for line in dockerfile.splitlines() if "MSE_SCHEMA_PATH=" in line
    )
    declared = env_line.split("MSE_SCHEMA_PATH=", 1)[1].strip().strip('"')

    copied = {
        line.split()[-1].rstrip("/")
        for line in dockerfile.splitlines()
        if line.startswith("COPY") and "db/" in line
    }
    assert any(declared.startswith(d) for d in copied), (
        f"MSE_SCHEMA_PATH={declared} is not under any COPY destination {copied}"
    )
    # and the filename is the one the resolver looks for
    assert declared.endswith(WEB_DB.SCHEMA_FILENAME)
