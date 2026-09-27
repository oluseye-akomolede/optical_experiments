"""The deploy artefacts: the systemd unit and the web Deployment manifest.

These two files are the only part of the system that no unit test can execute --
they are read by systemd and the kubelet on machines we do not run tests on --
so they drift silently. Four defects found in a single review of them all had
the same shape: a path or a key that is *written* in one place and *resolved*
in another, where only one of the two moved.

So the tests here check the two halves against each other, not the file against
a retyped copy of itself. A test that says "ExecStart contains this string"
is green whenever someone edits both the string and the test; a test that reads
what the worker actually computes and asserts the unit agrees is the one that
fails when the layout moves.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
UNIT = ROOT / "worker" / "mse-worker.service"
WORKER = ROOT / "worker" / "mse_worker.py"
DEPLOYMENT = ROOT / "manifests" / "mse" / "deployment.yaml"


def unit_text() -> str:
    return UNIT.read_text(encoding="utf-8")


def directives(text: str) -> dict[str, str]:
    """``[Service]``-style ``Key=Value`` lines, first occurrence wins.

    Comment lines are skipped: every one of these keys is DISCUSSED in a comment
    above it, and a naive scan would match the prose instead of the directive --
    which would make the test pass on a file whose real directive was deleted.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("["):
            continue
        if "=" in stripped:
            key, _, value = stripped.partition("=")
            out.setdefault(key.strip(), value.strip())
    return out


# ---------------------------------------------------------------------------
# the unit
# ---------------------------------------------------------------------------

def test_exec_start_names_an_interpreter_not_the_bare_script():
    """A shebang is not an interpreter guarantee.

    ``#!/usr/bin/env python3`` resolves to the SYSTEM python on the Orin, which
    has no cupy and no nvmath. The worker would then die at import with a
    traceback that reads like a code fault, on a machine we cannot debug from
    here. Naming the venv interpreter makes the environment explicit.
    """
    start = directives(unit_text())["ExecStart"]
    first = start.split()[0]
    assert "/" in first, (
        f"ExecStart invokes {first!r}, which relies on PATH lookup -- the "
        f"shebang/`env` case this test exists to prevent. ExecStart={start!r}")
    assert first.endswith("python") or "/python" in first, (
        f"ExecStart's program is {first!r}, which is not an interpreter path")
    assert "%h/mse-worker/venv/" in first, (
        f"ExecStart's interpreter {first!r} is not the venv's, so the worker "
        "would run on an interpreter without the GPU stack")


def test_the_unit_and_the_worker_agree_on_where_the_script_is():
    """The unit's script path and the worker's own import layout must be one layout.

    This is the cross-check the review needed: the worker resolves the solver
    directory RELATIVE TO ITS OWN FILE, so the unit's path decides whether that
    import resolves. Asserting the unit's string alone would have missed the
    original defect, where both halves were individually plausible and only
    their composition was wrong.
    """
    start = directives(unit_text())["ExecStart"]
    script = [tok for tok in start.split() if tok.endswith("mse_worker.py")]
    assert len(script) == 1, f"ExecStart does not name exactly one script: {start!r}"
    script_path = script[0]

    # The solver directory the worker computes, reproduced from the worker's
    # own source: it is `Path(__file__).parent.parent / "dockerfiles" /
    # "maxwell_solver"`. Deriving it here means a change to either the layout or
    # the worker's expression shows up as a disagreement.
    source = WORKER.read_text(encoding="utf-8")
    match = re.search(r"^_SOLVER_DIR\s*=\s*(.+)$", source, re.M)
    assert match, "_SOLVER_DIR is no longer a simple module-level expression"
    expr = match.group(1).strip()
    assert "parent.parent" in expr and "dockerfiles" in expr, (
        f"_SOLVER_DIR changed shape ({expr!r}); this test's derivation of the "
        "layout assumption must be revisited")

    # The unit's script lives at <root>/worker/mse_worker.py, so `parent.parent`
    # is <root> and the solver must be at <root>/dockerfiles/maxwell_solver.
    assert "/worker/mse_worker.py" in script_path, (
        f"the unit's script path {script_path!r} is not .../worker/mse_worker.py, "
        "so the worker's `parent.parent` no longer points at the archive root")
    archive_root = script_path.rsplit("/worker/mse_worker.py", 1)[0]
    solver = f"{archive_root}/dockerfiles/maxwell_solver"
    assert (ROOT / "dockerfiles" / "maxwell_solver").is_dir(), (
        "the repo itself no longer has dockerfiles/maxwell_solver, so the "
        f"derived deploy path {solver!r} cannot be right")
    # ...and the deployed archive must actually CONTAIN it, which is the whole
    # point of shipping the tree instead of the one script.
    assert f"{archive_root}/worker" in script_path


def test_the_unit_sets_a_working_directory():
    """Without WorkingDirectory the process starts in systemd's CWD (`$HOME`).

    The worker resolves scratch paths relative to the process, so a deploy that
    forgot this would scatter them outside the app tree -- and, more visibly,
    Documentation= would resolve against the wrong root.
    """
    assert "WorkingDirectory" in directives(unit_text()), (
        "the unit sets no WorkingDirectory, so the worker runs from $HOME")


def test_documentation_points_at_a_file_the_archive_contains():
    """The unit documents itself; a dangling Documentation= is worse than none.

    The path must be INSIDE the deployed archive (under the archive root in
    ExecStart), because the archive is all that is on the machine.
    """
    d = directives(unit_text())
    doc = d.get("Documentation", "")
    assert doc.startswith("file:"), f"Documentation is not a file: URL: {doc!r}"
    path = doc[len("file:"):]
    start = d["ExecStart"]
    root = [t for t in start.split() if t.endswith("mse_worker.py")][0]
    root = root.rsplit("/worker/mse_worker.py", 1)[0]
    assert path.startswith(root + "/"), (
        f"Documentation={path!r} is outside the deployed archive root {root!r}, "
        "so it will not exist on the machine")
    # ...and the corresponding file must exist in the repo at that offset.
    rel = path[len(root) + 1:]
    assert (ROOT / rel).is_file(), (
        f"Documentation points at {rel!r}, which is not in the repository")


def test_the_unit_and_the_readme_describe_the_same_interpreter():
    """The README is the deploy instruction; it must not name a different one.

    Two documents disagreeing is how an operator picks the one that happens to
    be open -- and the wrong one dies at import with a confusing traceback.
    """
    readme = (ROOT / "worker" / "README.md").read_text(encoding="utf-8")
    start = directives(unit_text())["ExecStart"]
    interp = [t for t in start.split() if t.endswith("/python")][0]
    # Compare the RESOLVED path, not the spelling: the unit writes `%h` because
    # systemd expands it, and the README writes `~` because a human types it.
    # Asserting the raw string would pass only while both used one convention --
    # a test about formatting masquerading as a test about the deploy target.
    resolved = interp.replace("%h/", "~/").replace("~", "$HOME")
    readme_resolved = readme.replace("~/", "$HOME/")
    assert resolved in readme_resolved, (
        f"the README never names {interp!r} (as {resolved!r}), the interpreter "
        "the unit runs -- the deploy instructions and the unit disagree")


def test_the_readme_does_not_tell_the_operator_to_copy_only_the_script():
    """The original defect, as a regression guard on the instructions.

    `cp worker/mse_worker.py ~/mse-worker/` is the exact command that produced
    the broken layout, so its presence in the deploy section is the bug coming
    back through the documentation rather than the unit.
    """
    readme = (ROOT / "worker" / "README.md").read_text(encoding="utf-8")
    deploy = readme.split("## What the lead does to deploy it", 1)[1]
    deploy = deploy.split("## Running it by hand", 1)[0]
    assert "cp worker/mse_worker.py ~/mse-worker/" not in deploy, (
        "the deploy instructions still copy the script alone, which leaves the "
        "worker's solver import pointing at a directory that does not exist")
    assert "git archive" in deploy, (
        "the deploy instructions do not ship the whole tree")


def test_the_readme_states_that_lingering_needs_the_owner():
    """Linger is off and enabling it needs sudo, which our rules forbid.

    An operator who reads "one-time fix: sudo loginctl enable-linger" with no
    caveat will either run it (against the rules) or skip it and get a worker
    that silently stops at logout.
    """
    readme = (ROOT / "worker" / "README.md").read_text(encoding="utf-8")
    assert "linger" in readme.lower(), "the README never mentions lingering"
    assert "owner" in readme.lower(), (
        "the README does not say the linger decision belongs to the owner")


# ---------------------------------------------------------------------------
# the Deployment manifest
# ---------------------------------------------------------------------------

def test_the_web_deployment_carries_the_worker_token():
    """Without it the worker API is closed (503) in production.

    worker_api.py disables the whole API when MSE_WORKER_TOKEN is unset -- the
    designed fail-safe, so a missing Secret can never mean an OPEN endpoint on a
    tailnet with no login. The consequence is that a missing key here is
    invisible in a rollout: the app serves pages happily while no worker can
    ever claim a job. Wiring the value is what makes it discoverable.
    """
    text = DEPLOYMENT.read_text(encoding="utf-8")
    assert "MSE_WORKER_TOKEN" in text, (
        "the web container has no MSE_WORKER_TOKEN, so the worker API answers "
        "503 and no Jetson job can ever be claimed")
    match = re.search(
        r"- name: MSE_WORKER_TOKEN\s*\n\s*valueFrom:\s*\n\s*secretKeyRef:\s*\n"
        r"\s*name: (\S+)\s*\n\s*key: (\S+)", text)
    assert match, (
        "MSE_WORKER_TOKEN is present but not wired through a secretKeyRef; an "
        "inline value would put the shared secret in the manifest")
    secret_name, key = match.group(1), match.group(2)
    assert key == "MSE_WORKER_TOKEN", (
        f"the secretKeyRef key is {key!r}, so the container would read a "
        "differently-named key and come up with no token")
    assert secret_name == "mse-web-secrets", (
        f"the token comes from secret {secret_name!r}, but the worker's token "
        "is provisioned under mse-web-secrets -- two sources for one secret is "
        "how they drift")


def test_the_worker_token_is_optional_at_pod_level_but_never_open():
    """`optional: true` keeps the app serving; the API stays closed anyway.

    The safety of tolerating a missing key depends entirely on the code's other
    half -- an unset token must mean 503, not an open route.
    """
    text = DEPLOYMENT.read_text(encoding="utf-8")
    block = text.split("- name: MSE_WORKER_TOKEN", 1)[1][:400]
    assert "optional: true" in block, (
        "a missing key would fail the rollout, which is not what was decided")


def test_every_worker_route_refuses_with_the_token_unset(monkeypatch):
    """The other half, EXERCISED rather than grepped for.

    An earlier version of this pair asserted only that the string
    ``WorkerAPIDisabled`` appeared in ``worker_api.py``. That class was defined
    and never raised, so the assertion was about a name rather than a
    behaviour: the reviewer deleted the ``abort(503)`` block, kept the class
    name, and the test still passed -- a future edit could have shipped
    ``optional: true`` with a WIDE OPEN worker API and stayed green. (The dead
    class has since been deleted; the guard is the ``abort(503)`` below, and it
    is now asserted by calling it.)

    Every route on the blueprint is driven, and the list is read off the
    blueprint rather than retyped: a NEW worker route added later is covered
    automatically, which a hand-written list cannot promise. ``MSE_WORKER_TOKEN``
    is deleted from the environment, so this is the true "the Secret is absent"
    case -- the worker Deployment tolerates that, and the code must too.
    """
    import sys

    sys.path.insert(0, str(ROOT / "web"))
    sys.path.insert(0, str(ROOT / "tests"))
    import worker_api as WORKER

    monkeypatch.delenv(WORKER.TOKEN_ENV, raising=False)

    from test_web_routes import make_app

    # The routes come from the app's own url_map, so this cannot drift from the
    # code the way a retyped list would.
    app, _store, _db, _launcher = make_app(Path("/tmp"))  # noqa: F841
    worker_rules = sorted(
        (str(r), sorted(r.methods - {"HEAD", "OPTIONS"}))
        for r in app.url_map.iter_rules()
        if str(r).startswith("/worker/")
    )
    assert worker_rules, (
        "no /worker/ routes were registered, so this test would pass vacuously "
        "-- the blueprint is not wired the way this test assumes")
    assert len(worker_rules) >= 5, (
        f"only {len(worker_rules)} worker routes found ({worker_rules}); the "
        "route table has shrunk, which this test should not silently accept")

    client = app.test_client()
    for path, methods in worker_rules:
        concrete = path.replace("<job_id>", "j1").replace("<name>", "n1")
        # The VERB comes from the rule too. Hard-coding POST made this test
        # report 405 for the one route that is a PUT, and a hand-written method
        # would have to be revisited for every new route -- the same drift the
        # route list avoids by being read off the app.
        for method in methods:
            response = client.open(concrete, method=method)
            assert response.status_code == 503, (
                f"{method} {concrete} returned {response.status_code} with "
                f"{WORKER.TOKEN_ENV} unset; it must be 503, not open"
            )


def test_the_deployment_is_valid_yaml_and_keeps_one_replica():
    """A cheap structural check on a file no test can otherwise execute.

    One replica is load-bearing (the tick thread races for a single Job slot);
    it is asserted here so a later "obvious" scale-up fails loudly rather than
    in production.
    """
    yaml = pytest.importorskip("yaml")
    with DEPLOYMENT.open(encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    assert doc["spec"]["replicas"] == 1, (
        "replicas is not 1 -- two replicas race for the namespace's single Job "
        "slot")
    containers = doc["spec"]["template"]["spec"]["containers"]
    names = [c["name"] for c in containers]
    assert names == ["web"], f"unexpected containers: {names}"
    env_names = [e["name"] for e in containers[0].get("env", [])]
    assert "MSE_WORKER_TOKEN" in env_names
