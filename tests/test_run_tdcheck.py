"""scripts/run_tdcheck.sh -- checking a --dir mirror before the browser starts.

``--dir`` refuses a mirror that is missing a file the check reads, so an
incomplete mirror is named as one instead of crashing the check and looking
like a page defect. That promise is only as wide as the list in the shell
script, and the list is worth nothing if it drifts behind the check: a file
``tdcheck.mjs`` reads and the runner does not name is a crash, not a refusal
(``files/trace.bin`` was exactly that -- read unconditionally, listed nowhere).

So the list is not restated here. The first test derives it from tdcheck.mjs's
own source -- every ``path.join(DIR, '...')`` it reads by name -- and requires
the runner to cover it, so adding a read to the check without adding it to the
runner fails here. The second drives the refusal itself.

Tier 1: no network, no browser, no docker (``docker`` is a stub that records
what it was asked for; the pre-check must refuse before it is ever run).
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHECK = ROOT / "dockerfiles" / "rendercheck" / "tdcheck.mjs"
RUNNER = ROOT / "scripts" / "run_tdcheck.sh"

#: A path the check reads out of TD_DIR by name, as written in its source.
READ = re.compile(r"""path\.join\(DIR,\s*(['"])([^'"]+)\1\)""")
#: The runner's list: `for f in a b c; do`.
LIST = re.compile(r"for f in ([^;]*); do")

#: A few of the check's reads, named so a rename this test cannot follow is a
#: failure here rather than a derivation that silently agrees with itself.
PINNED = {"job.html", "status.json", "files/spectrum.json", "files/trace.bin"}


def reads() -> set[str]:
    """Every file tdcheck.mjs takes out of TD_DIR by name.

    Not the ones it looks up indirectly (the frames its manifest names, and the
    /static paths the page itself asks for): those follow from ``files`` and
    from the modules below, and cannot be listed in advance.
    """
    return {m.group(2) for m in READ.finditer(CHECK.read_text())}


def listed() -> set[str]:
    m = LIST.search(RUNNER.read_text())
    assert m, f"no `for f in ...; do` pre-check in {RUNNER.name}"
    return set(m.group(1).split())


def test_the_runner_refuses_a_mirror_missing_any_file_the_check_reads():
    got = reads()
    assert got >= PINNED, f"the check no longer reads {PINNED - got}: fix the derivation"
    missing = got - listed()
    assert not missing, f"run_tdcheck.sh's pre-check misses what tdcheck.mjs reads: {sorted(missing)}"


def test_the_refusal_names_the_hole_and_starts_nothing(tmp_path):
    """The forced case: a mirror with every listed file but ``trace.bin``."""
    mirror = tmp_path / "mirror"
    for f in ["job.html", "status.json", "files/spectrum.json",
              "static/tdplot.js", "static/viewer.js", "static/tdmovie.js"]:
        p = mirror / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    marker = tmp_path / "docker-argv"
    stub = tmp_path / "bin"
    stub.mkdir()
    docker = stub / "docker"
    docker.write_text(f'#!/bin/sh\necho "$@" >> "{marker}"\nexit 1\n')
    docker.chmod(0o755)
    env = {**os.environ, "PATH": str(stub), "DOCKER_CONFIG": str(tmp_path)}
    p = subprocess.run(["/bin/sh", str(RUNNER), "--dir", str(mirror)],
                       capture_output=True, text=True, env=env)
    assert p.returncode == 2, f"refusal should exit 2: {p.returncode}\n{p.stdout}{p.stderr}"
    assert "files/trace.bin is not readable" in p.stderr, p.stderr
    # `docker rm -f` from the exit trap is expected; `docker run` is not: the
    # browser must never have been reached on a mirror with a hole in it.
    calls = marker.read_text() if marker.exists() else ""
    assert "run" not in calls.split(), f"docker was run: {calls!r}"
