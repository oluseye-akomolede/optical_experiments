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
what it was asked for; the pre-check must refuse before it is ever run). The cap
that bounds the run is real: ``timeout`` is symlinked into the stub directory,
because the runner reaching it is what these tests are here to show.
"""

from __future__ import annotations

import json
import os
import re
import shutil
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
PINNED = {"job.html", "status.json", "outline.glb",
          "files/spectrum.json", "files/trace.bin"}


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


def build_mirror(mirror: Path, frames=None, skip: set[str] = frozenset(),
                 docker_body: str = 'echo "$@" >> "$MARKER"\nexit 1') -> Path:
    """A mirror holding every file run_tdcheck.sh's own pre-check lists.

    Built from that list rather than a second copy of it here, so a file the
    runner starts demanding later is present for every case automatically and
    each test keeps forcing only the one hole it means to force. ``docker`` is
    stubbed beside it (PATH is set by ``run``), because nothing here reaches a
    browser.
    """
    for f in sorted(listed() - set(skip)):
        p = mirror / f
        if any(other.startswith(f + "/") for other in listed()):
            p.mkdir(parents=True, exist_ok=True)     # a listed directory, e.g. files
            continue
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("x")
    if frames is not None:
        (mirror / "files/spectrum.json").write_text(json.dumps({"frames": frames}))
    stub = mirror.parent / "bin"
    stub.mkdir(exist_ok=True)
    docker = stub / "docker"
    docker.write_text(f"#!/bin/sh\n{docker_body}\n")
    docker.chmod(0o755)
    # ``run`` points PATH at this stub directory alone, so a real tool the
    # runner or a stub calls has to be reachable from it. ``timeout`` is the one
    # that matters: it is the cap, and without it the runner dies at "timeout:
    # not found" (127), which is not a verdict about the page. Symlinked rather
    # than stubbed, because the real one is what the runner is tested against.
    for tool in ("timeout", "sleep"):
        where = shutil.which(tool)
        assert where, f"{tool} is not on PATH; the runner's cap needs it"
        if not (stub / tool).exists():
            (stub / tool).symlink_to(where)
    return stub


def run(mirror: Path, stub: Path, **env_extra):
    env = {**os.environ, "PATH": str(stub), "DOCKER_CONFIG": str(stub.parent),
           "MARKER": str(stub.parent / "docker-argv"), **env_extra}
    return subprocess.run(["/bin/sh", str(RUNNER), "--dir", str(mirror)],
                          capture_output=True, text=True, env=env, timeout=60)


def test_the_refusal_names_the_hole_and_starts_nothing(tmp_path):
    """The forced case: a mirror with every listed file but ``trace.bin``."""
    mirror = tmp_path / "mirror"
    stub = build_mirror(mirror, skip={"files/trace.bin"})
    marker = tmp_path / "docker-argv"
    p = run(mirror, stub)
    assert p.returncode == 2, f"refusal should exit 2: {p.returncode}\n{p.stdout}{p.stderr}"
    assert "files/trace.bin is not readable" in p.stderr, p.stderr
    # `docker rm -f` from the exit trap is expected; `docker run` is not: the
    # browser must never have been reached on a mirror with a hole in it.
    calls = marker.read_text() if marker.exists() else ""
    assert "run" not in calls.split(), f"docker was run: {calls!r}"


def test_the_cap_follows_the_movies_own_frame_count(tmp_path):
    """What bounds the run is the movie's length, not a constant.

    Measured: a fixed 600 s is not enough for a correct run -- the real job
    180cb65f (60 frames, two-digit names) was cut after 45 PASS, and the
    120-frame open-boundary fixture mirror needed ~740 s on this loaded VM --
    while a short movie finishes in a fraction of that. The cap is derived from the
    manifest the mirror holds, and the runner says the number it chose.
    """
    for count in (1, 60, 120):
        mirror = tmp_path / f"m{count}" / "mirror"
        stub = build_mirror(mirror, frames={"count": count, "digits": 3},
                            docker_body="exit 0")      # the check passed
        p = run(mirror, stub)
        assert p.returncode == 0, p.stdout + p.stderr
        assert f"cap: {180 + 12 * count}s for {count} frames" in p.stdout, p.stdout


def test_the_cap_defaults_to_the_solvers_maximum_and_can_be_overridden(tmp_path):
    """A manifest that will not read, and an operator who knows it is slower."""
    mirror = tmp_path / "unreadable" / "mirror"
    stub = build_mirror(mirror)          # files/spectrum.json is there, but not JSON
    p = run(mirror, stub)
    assert f"cap: {180 + 12 * 120}s for 120 frames" in p.stdout, p.stdout
    mirror = tmp_path / "over" / "mirror"
    stub = build_mirror(mirror, frames={"count": 120, "digits": 3})
    p = run(mirror, stub, TDCHECK_TIMEOUT="1234")
    assert "cap: 1234s for 120 frames" in p.stdout, p.stdout


def test_a_run_cut_by_the_cap_is_named_and_is_not_a_page_defect(tmp_path):
    """The forced case: the container outlives the cap and is stopped.

    124 is timeout(1)'s own status, not the page's verdict, so the runner must
    name the cap and the frame count in plain words. Left to itself the check's
    last words are a stack trace about a closed page, which names the symptom
    and reads like a broken page.
    """
    mirror = tmp_path / "mirror"
    # exec, so the signal the cap sends lands on the sleep itself and the stub
    # does not leave an orphan behind to outlive the run.
    stub = build_mirror(mirror, frames={"count": 120, "digits": 3},
                        docker_body='case " $* " in *" run "*) exec sleep 30 ;; esac\nexit 0')
    p = run(mirror, stub, TDCHECK_TIMEOUT="1")
    assert p.returncode == 124, f"a cut run should exit 124: {p.returncode}\n{p.stdout}{p.stderr}"
    assert "harness timed out after 1s (120 frames; TDCHECK_TIMEOUT=1)" in p.stderr, p.stderr
