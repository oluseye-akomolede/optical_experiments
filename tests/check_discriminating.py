"""Prove a regression test fails without its fix -- in one command.

A test that passes both before and after a fix is worthless, and "I checked it
fails" is a claim a reviewer cannot re-derive without doing the mutation by
hand. This tool makes the mutation mechanical: it applies a named reversion to
the source, runs the tests, restores the file, and reports which tests failed.

    python tests/check_discriminating.py            # run every case
    python tests/check_discriminating.py isfile     # run one case
    python tests/check_discriminating.py --list

Exit status is 0 only if every case's tests changed from passing to failing,
which is what "discriminating" means. It never leaves the tree modified: the
original bytes are restored in a ``finally``, and the restoration is verified by
hash rather than assumed.

**One run at a time.** This tool rewrites the working tree and puts it back, so
two concurrent runs will each restore over the other's reversion -- and the
damage is silent, because a case whose file was restored mid-run reports as
"not discriminating" while a case that finishes during the other's apply can
leave a reverted file behind. A lock file makes the second run refuse rather
than corrupt. This is not hypothetical: on 2026-09-27 the reviewer ran this in
my worktree while I was committing in it, and a leftover ``bench_box.py``
reversion -- the bench skipping its production solve, the exact defect the bench
case exists to catch -- sat in the tree until I noticed it by hand.

This exists because a reviewer asked for revert-and-rerun fixtures to be left in
place. A script is better than a scratch commit: the fixture is checked in, it
cannot be lost, and the failure count is reproducible from the tests alone.
"""

from __future__ import annotations

import argparse
import atexit
import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / ".check-discriminating.lock"
WEB_DB = ROOT / "web" / "db.py"
SOLVER_DB = ROOT / "dockerfiles" / "maxwell_solver" / "mse_db.py"
SCHEMA_SQL = ROOT / "db" / "001_schema.sql"


def take_lock() -> None:
    """Refuse to run if another instance is mid-reversion.

    ``O_CREAT | O_EXCL`` is atomic, so two processes cannot both believe they
    hold it.

    A stale lock from a killed run is NOT cleared automatically, and that is
    deliberate rather than an oversight. A run killed mid-case leaves a *dirty
    tree* -- the file it was reverting is still reverted -- and the stale lock is
    the only thing stopping the next run from building on that. The mutation
    window is most of the run, so a kill almost always lands on a reverted file.
    Clearing a stale lock by pid-liveness or a timeout would let run two revert
    files on top of run one's half-restored tree. The refusal is load-bearing;
    clear it by hand, after ``git status``.
    """
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        try:
            # ``read_text`` succeeds on the window between the create above and
            # the write below, returning '' -- so the OSError fallback does not
            # cover it and the message would read "in progress (pid )."
            holder = LOCK.read_text().strip() or "unknown"
        except OSError:
            holder = "unknown"
        raise SystemExit(
            f"another check_discriminating run is in progress (pid {holder}).\n"
            "It rewrites the tree and restores it, so running two at once would "
            "corrupt both.\n"
            f"If that process is gone, the tree may be half-reverted -- run "
            f"'git status' before removing {LOCK} and retry."
        )
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    atexit.register(release_lock)


def release_lock() -> None:
    try:
        LOCK.unlink()
    except OSError:
        pass


@dataclass
class Case:
    """One reversion: an exact substitution, and the tests that must fail."""

    name: str
    why: str
    edits: list[tuple[Path, str, str]]
    expect_failing: set[str] = field(default_factory=set)
    test_file: str = "tests/test_schema_paths.py"

    def apply(self) -> None:
        for path, old, new in self.edits:
            text = path.read_text(encoding="utf-8")
            if old not in text:
                raise SystemExit(
                    f"{self.name}: could not find the text to revert in "
                    f"{path.relative_to(ROOT)} -- has the code changed?\n"
                    f"  looked for: {old!r}"
                )
            path.write_text(text.replace(old, new), encoding="utf-8")

    def snapshot(self) -> dict[Path, bytes]:
        return {path: path.read_bytes() for path, _, _ in self.edits}

    def restore(self, saved: dict[Path, bytes]) -> None:
        for path, data in saved.items():
            path.write_bytes(data)
            # verify, do not assume: a partial restore would poison later cases
            digest = hashlib.sha256(data).hexdigest()
            now = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != now:
                raise SystemExit(f"FAILED TO RESTORE {path} -- tree is dirty")


CASES: list[Case] = [
    Case(
        name="schema-path-climb",
        why="the fixed-number-of-dirnames climb that broke the deployed image",
        edits=[(
            WEB_DB,
            "def _schema_path() -> str:",
            # reinstate the old repo-only resolution
            "def _schema_path() -> str:\n"
            "    return os.path.join(\n"
            "        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),\n"
            "        'db', SCHEMA_FILENAME,\n"
            "    )\n\n\n"
            "def _unused_schema_path() -> str:",
        )],
        # Six, not four: restoring the fixed climb also breaks the two
        # directory cases (they resolve through the same function) and the
        # end-to-end initContainer test, which really does exercise the layout.
        # The count is asserted exactly, so a *new* test that stops catching
        # this regression shows up as a mismatch rather than passing quietly.
        # Six. test_web_error_names_every_path_it_tried is here because the
        # reverted climb raises its own FileNotFoundError naming one path
        # instead of delegating to the candidate list. It is NOT
        # test_web_candidates_cover_both_layouts_without_existing: that one
        # calls schema_candidates() directly, which this reversion does not
        # touch, so it passes and belongs out of the set.
        expect_failing={
            "test_web_resolves_the_image_layout",
            "test_web_prefers_the_env_override",
            "test_web_error_names_every_path_it_tried",
            "test_a_directory_at_the_override_alone_raises_clearly",
            "test_a_directory_beside_the_module_is_not_the_schema",
            "test_the_web_initcontainer_script_still_works_in_the_image_layout",
        },
    ),
    Case(
        name="isfile",
        why="os.path.exists accepts a directory, which then fails in open()",
        edits=[
            (WEB_DB, "if os.path.isfile(path):", "if os.path.exists(path):"),
            (SOLVER_DB, "if os.path.isfile(resolved):", "if os.path.exists(resolved):"),
        ],
        expect_failing={
            "test_web_skips_a_directory_at_the_override",
            "test_solver_skips_a_directory_at_the_override",
            "test_a_directory_at_the_override_alone_raises_clearly",
            "test_a_directory_beside_the_module_is_not_the_schema",
        },
    ),
    Case(
        name="bare-create-schema",
        why="CREATE SCHEMA IF NOT EXISTS needs database-level CREATE",
        edits=[(
            SCHEMA_SQL,
            "        CREATE SCHEMA mse;\n    END IF;",
            "        CREATE SCHEMA IF NOT EXISTS mse;\n    END IF;",
        )],
        # Three, and note the middle one: test_the_guard_still_contains_a_real
        # _create_schema asserts `"CREATE SCHEMA mse;" in guard`, and
        # "CREATE SCHEMA IF NOT EXISTS mse;" does not contain that substring --
        # so it fails here too. An earlier expectation of two was wrong.
        expect_failing={
            "test_the_ddl_has_no_bare_create_schema",
            "test_the_schema_creation_is_guarded_on_pg_namespace",
            "test_the_guard_still_contains_a_real_create_schema",
        },
    ),
    Case(
        name="raise-instead-of-create",
        why="a RAISE would remove the only path that provisions a fresh schema",
        edits=[(
            SCHEMA_SQL,
            "        CREATE SCHEMA mse;\n    END IF;",
            "        RAISE EXCEPTION 'schema missing';\n    END IF;",
        )],
        expect_failing={
            "test_the_guard_still_contains_a_real_create_schema",
            "test_the_schema_creation_is_guarded_on_pg_namespace",
        },
    ),
    Case(
        name="solve-climb",
        why="the solver's own copy climbed one level too far in the image",
        edits=[(
            SOLVER_DB,
            "        candidates = [os.path.join(here, \"db\", SCHEMA_FILENAME)]",
            "        candidates = [os.path.join(os.path.dirname(here), \"db\", SCHEMA_FILENAME)]",
        )],
        # One: test_solver_resolves_the_image_layout is the test that catches
        # this in the image. test_solver_skips_a_directory_at_the_override
        # passes anyway -- its override is a directory, so the resolver moves
        # on and finds the real repo candidate whichever way the first entry
        # points. Kept out of the set rather than listed and tolerated.
        expect_failing={
            "test_solver_resolves_the_image_layout",
        },
    ),
]

WEB_REQS = ROOT / "dockerfiles" / "web" / "requirements.txt"
SOLVER_REQS = ROOT / "dockerfiles" / "maxwell_solver" / "requirements.txt"
WEB_DOCKERFILE = ROOT / "dockerfiles" / "web" / "Dockerfile"
SOLVER_DOCKERFILE = ROOT / "dockerfiles" / "maxwell_solver" / "Dockerfile"

TEST_FILE = "tests/test_schema_paths.py"
IMAGE_TEST_FILE = "tests/test_image_contracts.py"

# One reversion per image-contract guarantee. Each is a real regression that
# shipped in this project or was proposed: a `>=` that drifted, an ENV that was
# never added, a path that stopped matching its COPY, and the solver's dead DDL.
for _reqs, _label in ((WEB_REQS, "web"), (SOLVER_REQS, "solver")):
    _first = next(
        line for line in _reqs.read_text(encoding="utf-8").splitlines()
        if "==" in line and not line.lstrip().startswith("#")
    )
    CASES.append(Case(
        name=f"unpinned-{_label}",
        why=f"a `>=` in {_label} requirements re-opens the version drift",
        edits=[(_reqs, _first, _first.replace("==", ">="))],
        # Parametrised tests report as name[web] / name[solver]: pytest appends
        # the id, so the bare name never matches. The third is the per-image
        # test that names the requirement that was unpinned.
        expect_failing={
            f"test_no_bare_range_specifiers[{_label}]",
            f"test_every_direct_requirement_is_pinned_exactly_once[{_label}]",
            ("test_the_web_image_pins_flask_and_gunicorn" if _label == "web"
             else "test_the_solver_image_pins_its_scientific_stack"),
        },
        test_file=IMAGE_TEST_FILE,
    ))

FIELDS = ROOT / "dockerfiles" / "maxwell_solver" / "mse_fields.py"
SAMPLER_TEST_FILE = "tests/test_field_sampler.py"

# BUG A: the (chunk, n_tets, 3) broadcast that OOM-killed the Job. Restoring it
# leaves the located set identical -- which is exactly why it needs a test that
# watches the allocation rather than the answer, and why the exactness tests
# all pass here. An earlier draft of this expectation also listed
# test_the_candidate_set_is_bounded_by_cell_size_not_tet_count; that was wrong,
# and the tool said so. candidate_counts re-gathers candidates through
# _candidate_pairs on its own, so it does not observe the broadcast at all. The
# set below is what was measured: exactly one test, and it is the one written
# for this bug.
CASES.append(Case(
    name="broadcast-against-every-tet",
    why="the (chunk, n_tets, 3) difference array that OOM-killed the Job",
    edits=[(
        FIELDS,
        "        owner, cand = _candidate_pairs(chunk, tree, radius)\n"
        "        if cand.size == 0:\n"
        "            continue\n"
        "\n"
        "        diff = chunk[owner] - v0_all[cand]                # (C, 3)\n"
        "        b = np.einsum(\"cij,cj->ci\", inv_J[cand], diff)    # (C, 3)",
        "        # THE ORIGINAL BUG, restored for the discriminating check\n"
        "        d = chunk[:, None, :] - v0_all[None, :, :]        # (m, n_tets, 3)\n"
        "        n_tets = tets.shape[0]\n"
        "        owner = np.repeat(np.arange(chunk.shape[0]), n_tets)\n"
        "        cand = np.tile(np.arange(n_tets), chunk.shape[0])\n"
        "        bb1 = np.einsum(\"mtj,tj->mt\", d, inv_J[:, 0, :])\n"
        "        bb2 = np.einsum(\"mtj,tj->mt\", d, inv_J[:, 1, :])\n"
        "        bb3 = np.einsum(\"mtj,tj->mt\", d, inv_J[:, 2, :])\n"
        "        b = np.stack([bb1, bb2, bb3], axis=-1).reshape(-1, 3)",
    )],
    expect_failing={
        "test_the_locator_does_not_broadcast_against_every_tet",
    },
    test_file=SAMPLER_TEST_FILE,
))

# The shape of "reuse" that would still be wrong: locating per mode, inside
# sample_field, instead of once per job in write_results. Reverting the
# once-per-job call is a call-site change in the solver, not in this module, so
# what is checked here is the module-level contract it rests on -- that the
# located result is a pure function of (points, tets, samples) and carries no
# eigenvector, so reuse cannot change the field.
CASES.append(Case(
    name="locator-depends-on-the-mode",
    why="if locating were mode-dependent, reusing one result would change the field",
    edits=[(
        FIELDS,
        "    radius = _LOCATE_RADIUS_FACTOR * r_max",
        "    radius = _LOCATE_RADIUS_FACTOR * r_max * (1.0 + 1e-12)",
    )],
    # A deliberately invisible perturbation: it changes no answer, so nothing
    # should fail. This case exists to record that the "locate once" claim is
    # NOT proved by a reversion of this shape -- it is proved by
    # test_locating_once_and_reusing_matches_locating_per_mode being a pure
    # function of its inputs, and by write_results calling it outside the mode
    # loop. An empty expected set is a real assertion: it says the reviewer can
    # restore this and watch the suite stay green, so any future test that
    # claims to catch mode-dependence by this route is caught lying.
    expect_failing=set(),
    test_file=SAMPLER_TEST_FILE,
))

WEB_APP = ROOT / "web" / "app.py"
WEB_TEST_FILE = "tests/test_web_routes.py"
#: the launcher's own tests, where the real 403 branch is driven through a fake _client
LAUNCHER_TEST_FILE = "tests/test_web_app.py"

# BUG B: a dead Job's row staying `running` forever. Two reversions, because the
# fix has two halves and they fail in opposite directions -- dropping the failure
# reconcile leaves dead rows running, and dropping the "is it still alive?" check
# fails healthy solves. A fix that only passes one of them is not the fix.
CASES.append(Case(
    name="never-reconcile-running-rows",
    why="without the reconcile, a Job that dies leaves its row running forever",
    edits=[(
        WEB_APP,
        "        jobs_db.mark_failed(job[\"job_id\"], JOBS.failure_message(reason))\n"
        "        failed += 1\n",
        "        continue                           # BUG: dead row left running\n",
    )],
    # The reconcile picks ``reason`` first and fails the row at the bottom of
    # the loop, so the reversion drops that one ``mark_failed``. Everything
    # before it still runs, including the too-large-for-the-cluster reroute:
    # that is a different defect, and its tests live in tests/test_routing.py.
    # Five, not the four the names suggest. The fifth is
    # test_a_dead_job_frees_the_slot_for_the_next_upload, and it belongs: that
    # test asserts the queued upload starts once the dead row is failed, so a
    # reversion that stops failing the row fails it too. Two tests catching one
    # defect from different ends (the row's state, and the queue moving behind
    # it) is the intended shape here -- see the comment in that test.
    expect_failing={
        "test_a_running_row_whose_job_vanished_is_failed_not_requeued",
        "test_a_job_reporting_deadline_exceeded_fails_the_row_with_that_reason",
        "test_a_job_reporting_oomkilled_fails_the_row_with_that_reason",
        "test_a_job_failed_for_an_unknown_reason_does_not_leak_the_kubernetes_token",
        "test_a_dead_job_frees_the_slot_for_the_next_upload",
    },
    test_file=WEB_TEST_FILE,
))

CASES.append(Case(
    name="absent-means-dead",
    why="treating 'no Failed condition' as 'dead' fails every healthy solve",
    edits=[(
        WEB_APP,
        "        elif name and name in present:\n"
        "            continue                       # alive and well; not ours to touch",
        "        elif name and name in present:\n"
        "            pass                           # BUG: alive, but failed anyway",
    )],
    # Four, not the one. Two extras are the slot tests, and they are not
    # incidental: a healthy Job failed by this bug is now terminal, so the tick
    # reaps it and the next upload starts, which is precisely the "every upload
    # fails" damage described above. They are the same defect measured on the
    # queue rather than on the row. The fourth is the transient-error test,
    # which came later and pins the same healthy-Job survival from the other
    # side: its second tick sees a present, unfailed Job, and this reversion
    # fails it there too.
    expect_failing={
        "test_a_live_job_is_never_failed_or_deleted",
        "test_a_queued_or_running_job_is_never_deleted",
        "test_the_tick_does_nothing_when_the_slot_is_occupied",
        "test_a_transient_list_error_then_a_good_list_behaves_normally",
    },
    test_file=WEB_TEST_FILE,
))

BENCH = ROOT / "tests" / "bench_box.py"
SOLVER_TEST_FILE = "tests/test_maxwell_solver.py"

# The bench's writing stage, un-wired: the exact gap that let the ~15 GB sampler
# reach the cluster. Reverting the stage to "collect the keys and write nothing"
# is what a bench that stopped after the solve looked like.
CASES.append(Case(
    name="bench-skips-the-writing-stage",
    why="the bench stopping after the solve is how BUG A reached the cluster",
    edits=[(
        BENCH,
        "    t0 = time.time()\n"
        "    summary = SOLVER.solve(\n"
        "        job_id, units, n_index, n_modes, stores, stores,\n"
        "        max_tets=max_tets, grid=grid, verbose=verbose,\n"
        "    )\n"
        "    summary[\"wall_s\"] = time.time() - t0",
        "    t0 = time.time()\n"
        "    summary = {\"timings\": {}, \"n_tets\": 0, \"n_modes\": n_modes}\n"
        "    summary[\"wall_s\"] = time.time() - t0",
    )],
    expect_failing={
        "test_the_bench_writing_stage_drives_the_production_solve",
        "test_the_bench_reports_the_writing_stage_among_its_timings",
    },
    test_file=SOLVER_TEST_FILE,
))

JOBS_MODULE = ROOT / "web" / "jobs.py"

# The 403 that the reviewer found: swallowing an unreadable cluster into an empty
# result, so "_fail_dead_jobs" cannot tell "not allowed to look" from "nothing is
# there" and fails a live solve -- then deletes it on a later tick.
#
# One expected failure, and it took measuring to learn that. The two 403 tests
# that drive doubles pass with the swallow restored, and no expected set can fix
# that: the callers were repaired separately, by a placeholder that turns "could
# not ask" into a shut gate, so a double that answers empty leaves the tick doing
# the same correct thing either way. The launcher's own raise -- the part that
# stops the empty answer arising at all -- is reachable only by reading the
# source, which is what ``test_the_launcher_raises_on_a_403_instead_of_answering_
# empty`` does. So the reversion is caught by exactly that one probe.
CASES.append(Case(
    name="403-swallowed-as-empty",
    why="an unreadable cluster must not look like an empty one",
    edits=[
        # the two raise sites, back to the e5b2e3e behaviour
        (
            JOBS_MODULE,
            "            if exc.status == 403:\n"
            "                raise CannotAskCluster(\n"
            "                    \"not allowed to list solver jobs\"\n"
            "                ) from exc",
            "            if exc.status == 403:\n"
            "                return []",
        ),
        # the sentence that says it raises, so the source stays coherent with
        # the reverted behaviour rather than a docstring contradicted by code
        (
            JOBS_MODULE,
            "        Raises on 403 rather than returning ``[]``. An empty list is a *claim*",
            "        Returns ``[]`` when the list call is forbidden. An empty list is a claim",
        ),
    ],
    # Two, and they are the *behavioural* tests in test_web_app.py: the reversion
    # is caught by driving the real 403 branch through a fake `_client`, which is
    # what the lead pointed out was possible all along. This case used to expect a
    # source probe instead; that probe is gone.
    expect_failing={
        "test_active_job_names_raises_on_a_403_instead_of_returning_empty",
        "test_job_states_raises_on_a_403_instead_of_returning_empty",
    },
    test_file=LAUNCHER_TEST_FILE,
))

# The split answer: one list call raising while the other answers. This is the
# shape that shows the placeholder is not what protects `_fail_dead_jobs` -- an
# empty dict from `job_states` reads as "nothing failed", and the running row is
# failed. Fixing it required `_fail_dead_jobs` to require BOTH calls to succeed,
# so this reversion is the guard's own test.
CASES.append(Case(
    name="reconcile-on-a-partial-answer",
    why="failing rows off a list call that never answered",
    edits=[(
        WEB_APP,
        "    try:\n"
        "        names = launcher.active_job_names()\n"
        "    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop\n"
        "        log(f\"[mse-web] could not list solver jobs: {exc}\")\n"
        "        return 0\n"
        "    try:\n"
        "        states = launcher.job_states()\n"
        "    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop\n"
        "        log(f\"[mse-web] could not read job states: {exc}\")\n"
        "        return 0\n"
        "    present = set(names)",
        "    try:\n"
        "        present = set(_active_jobs(launcher))\n"
        "        states = launcher.job_states()\n"
        "    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop\n"
        "        log(f\"[mse-web] could not read job states: {exc}\")\n"
        "        return 0",
    )],
    # One, not two, and the reason is worth keeping. Only the first direction
    # discriminates: it needs `job_states` to *answer*, which the old combined
    # try/except does not guard. The mirror -- `job_states` raising while
    # `active_job_names` answers -- is caught by the old code too, because a
    # raise from either call inside the single try reaches the same handler. So
    # the mirror pins the behaviour without distinguishing the two shapes, and
    # saying so is better than leaving a second name here that never fires.
    expect_failing={
        "test_one_call_raising_while_the_other_answers_still_fails_nothing",
    },
    test_file=WEB_TEST_FILE,
))

CASES.append(
        Case(
            name="no-schema-path-env",
    why="without MSE_SCHEMA_PATH the DDL path is a relative climb again",
    edits=[(WEB_DOCKERFILE, "ENV MSE_SCHEMA_PATH=/opt/mse/001_schema.sql\n",
            "")],
    expect_failing={
        "test_the_web_dockerfile_declares_the_schema_path",
        "test_the_web_initcontainer_and_env_agree",
    },
    test_file=IMAGE_TEST_FILE,
))

CASES.append(Case(
    name="schema-path-mismatches-copy",
    why="ENV pointing somewhere the DDL is not copied is a crash-loop at startup",
    edits=[(WEB_DOCKERFILE, "/opt/mse/001_schema.sql", "/opt/somewhere/001_schema.sql")],
    # Two, not three: test_the_web_dockerfile_copies_the_ddl_to_that_path reads
    # the COPY line, which this reversion leaves alone, so it still passes.
    # The ENV-declares and the cross-file agreement are what catch a mismatch.
    expect_failing={
        "test_the_web_dockerfile_declares_the_schema_path",
        "test_the_web_initcontainer_and_env_agree",
    },
    test_file=IMAGE_TEST_FILE,
))

CASES.append(Case(
    name="solver-ships-the-ddl",
    why="the solver carrying the schema re-invites a Job applying DDL",
    edits=[(SOLVER_DOCKERFILE, "COPY dockerfiles/maxwell_solver/ /opt/app/\n",
            "COPY dockerfiles/maxwell_solver/ /opt/app/\nCOPY db/ /opt/app/db/\n")],
    expect_failing={"test_the_solver_image_does_not_ship_the_ddl"},
    test_file=IMAGE_TEST_FILE,
))

CASES.append(Case(
    name="path-requirement",
    why="a local path requirement makes the image depend on the build host",
    # Four, not one: replacing a pin with a path also removes the `==`, so the
    # pin tests and the flask-presence check fail with it. Every one of these is
    # the same real defect seen from a different test, which is the point.
    edits=[(WEB_REQS, "flask==3.1.3", "./vendor/flask")],
    expect_failing={
        "test_no_editable_or_vcs_or_path_requirements[web]",
        "test_no_bare_range_specifiers[web]",
        "test_every_direct_requirement_is_pinned_exactly_once[web]",
        "test_the_web_image_pins_flask_and_gunicorn",
    },
    test_file=IMAGE_TEST_FILE,
))

RESONANCES = ROOT / "dockerfiles" / "maxwell_solver" / "resonances.py"
RESONANCES_TEST_FILE = "tests/test_resonances.py"

# Each test in tests/test_resonances.py names the mutation it goes red on;
# these are those mutations, with the exact failing set each one produces.

CASES.append(Case(
    name="resonances-gamma-halved",
    why="a decay rate off by a factor reports Q wrong by that factor",
    edits=[(
        RESONANCES,
        "    f = -np.angle(u) / (2 * np.pi * dt)\n"
        "    gamma = -np.log(np.abs(u)) / dt\n"
        "    modes = list(zip(f, gamma, a))",
        "    f = -np.angle(u) / (2 * np.pi * dt)\n"
        "    gamma = -np.log(np.abs(u)) / (2 * dt)\n"
        "    modes = list(zip(f, gamma, a))",
    )],
    expect_failing={
        "test_q_1e5_from_a_twentieth_of_a_decay_time[False-fdm]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[False-pencil]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-fdm]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-pencil]",
        "test_q_1e5_at_1536_nm_in_si_units",
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-pencil]",
        "test_noise_sweep_bars_bound_the_error[False]",
        "test_noise_sweep_bars_bound_the_error[True]",
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
        "test_real_path_equals_complex_path[fdm]",
        "test_real_path_equals_complex_path[pencil]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-f-per-sample",
    why="a frequency in cycles per sample instead of Hz lands nothing in the window",
    edits=[(
        RESONANCES,
        "    f = -np.angle(u) / (2 * np.pi * dt)\n"
        "    gamma = -np.log(np.abs(u)) / dt\n"
        "    modes = list(zip(f, gamma, a))",
        "    f = -np.angle(u) / (2 * np.pi)\n"
        "    gamma = -np.log(np.abs(u)) / dt\n"
        "    modes = list(zip(f, gamma, a))",
    )],
    expect_failing={
        "test_q_1e5_from_a_twentieth_of_a_decay_time[False-fdm]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[False-pencil]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-fdm]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-pencil]",
        "test_q_1e5_at_1536_nm_in_si_units",
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-pencil]",
        "test_noise_sweep_bars_bound_the_error[False]",
        "test_noise_sweep_bars_bound_the_error[True]",
        "test_pure_noise_reports_nothing[False-fdm]",
        "test_pure_noise_reports_nothing[True-fdm]",
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
        "test_real_path_equals_complex_path[fdm]",
        "test_real_path_equals_complex_path[pencil]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-fdm-threshold-coarse",
    why="a coarse U0 threshold projects out the weaker of two close modes",
    edits=[(RESONANCES, "SINGULAR_THRESHOLD = 1e-7", "SINGULAR_THRESHOLD = 1e-1")],
    expect_failing={
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-fdm]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-pencil-rank-one",
    why="a pencil capped at one pole cannot hold a pair",
    edits=[(RESONANCES, "        K = min(K, L // 2)", "        K = min(K, 1)")],
    expect_failing={
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-pencil]",
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
        "test_real_path_equals_complex_path[pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-pencil]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-no-background",
    why="without the pencil's far poles a strong mode outside the window biases the pair inside",
    edits=[(
        RESONANCES,
        "                    background: bool = True, **options) -> list[Resonance]:",
        "                    background: bool = False, **options) -> list[Resonance]:",
    )],
    expect_failing={
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-bars-too-narrow",
    why="a noise variance 100 times too small gives bars 10 times too narrow",
    edits=[(
        RESONANCES,
        "    return modes, sigma2, cramer_rao(modes, dt, N, sigma2, real=real)",
        "    return modes, sigma2, cramer_rao(modes, dt, N, sigma2 / 100, real=real)",
    )],
    expect_failing={
        "test_noise_sweep_bars_bound_the_error[False]",
        "test_noise_sweep_bars_bound_the_error[True]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-pencil]",
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-accept-everything",
    why="with no significance test every pole fitted to noise is reported",
    edits=[(
        RESONANCES,
        "        if gains[best] < min_significance ** 2 * quad:",
        "        if gains[best] < 0:",
    )],
    expect_failing={
        "test_pure_noise_reports_nothing[False-fdm]",
        "test_pure_noise_reports_nothing[True-fdm]",
        "test_noise_sweep_bars_bound_the_error[False]",
        "test_noise_sweep_bars_bound_the_error[True]",
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-fdm]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-pencil-shift-swapped",
    why="the pencil's shift the wrong way round gives the poles 1/u (Q changes sign)",
    edits=[(
        RESONANCES,
        "    V1, V2 = Vk[:, :-1], Vk[:, 1:]",
        "    V1, V2 = Vk[:, 1:], Vk[:, :-1]",
    )],
    expect_failing={
        "test_raw_fdm_and_raw_pencil_agree[False]",
        "test_raw_fdm_and_raw_pencil_agree[True]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[False-pencil]",
        "test_real_path_equals_complex_path[pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-fdm]",
        "test_two_close_modes_beside_a_strong_distant_one[window0-pencil]",
        "test_two_close_modes_beside_a_strong_distant_one[window1-pencil]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-real-half-amplitude",
    why="a real record's cosine amplitude is twice the complex half-amplitude",
    edits=[(
        RESONANCES,
        "            d = 2 * d[:K]",
        "            d = d[:K]",
    )],
    expect_failing={
        "test_real_path_equals_complex_path[fdm]",
        "test_real_path_equals_complex_path[pencil]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-fdm]",
        "test_q_1e5_from_a_twentieth_of_a_decay_time[True-pencil]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-fisher-unguarded",
    why="the pseudo-inverse of a singular Fisher matrix gives a duplicated pole half the bars and says nothing",
    edits=[(
        RESONANCES,
        "    cov, determined = _fisher_inverse(F)",
        "    cov, determined = np.linalg.pinv(F, hermitian=True), np.ones(4 * K, bool)",
    )],
    expect_failing={
        "test_duplicate_poles_get_no_bars[False]",
        "test_duplicate_poles_get_no_bars[True]",
        "test_near_duplicate_pair_across_the_condition_limit[False]",
        "test_near_duplicate_pair_across_the_condition_limit[True]",
        "test_weak_nuisance_mode_costs_nobody_their_bars[False]",
        "test_weak_nuisance_mode_costs_nobody_their_bars[True]",
    },
    test_file=RESONANCES_TEST_FILE,
))

CASES.append(Case(
    name="resonances-nyquist-unchecked",
    why="a window above Nyquist must be refused, not aliased",
    edits=[(
        RESONANCES,
        "        if f_min < 0 or f_max > nyq:",
        "        if f_min < 0:",
    )],
    expect_failing={
        "test_bad_inputs_are_refused_with_a_message",
    },
    test_file=RESONANCES_TEST_FILE,
))

FARFIELD = ROOT / "dockerfiles" / "maxwell_solver" / "farfield.py"
FARFIELD_TEST_FILE = "tests/test_farfield.py"

# The near-to-far-field transform: one reversion per mutation a test docstring
# in tests/test_farfield.py names. Every formula sign and weight here has a
# closed-form dipole result that reads it, and the sets below are what was
# measured, not what was hoped: a mutation that breaks the formula breaks
# several closed forms at once, and a normalised pattern or a ratio is blind to
# a missing weight, which is why the absolute-power tests exist.
#: The closed-form tests a wrong far-field FORMULA (a sign in the equivalence
#: or in the phase) turns red. test_halving_the_spacing_converges is not among
#: them: it compares two resolutions of the same wrong answer and stays green,
#: which is why it is never the only test on a formula.
_FORMULA_TESTS = {
    "test_z_dipole_intensity_is_sin_squared", "test_far_field_power_equals_surface_power",
    "test_x_dipole_pattern", "test_collection_fraction_matches_closed_form",
    "test_off_centre_dipole_in_non_cubic_box_has_the_same_pattern",
    "test_pec_image_reproduces_the_dipole_over_a_mirror",
    "test_open_bottom_without_image_misses_the_reflection",
}
#: What a wrong far-field SCALE turns red: the absolute-intensity and the
#: power tests, and the convergence test's power-against-closed-form line.
_SCALE_TESTS = {
    "test_z_dipole_intensity_is_sin_squared", "test_far_field_power_equals_surface_power",
    "test_halving_the_spacing_converges",
    "test_pec_image_reproduces_the_dipole_over_a_mirror",
    "test_open_bottom_without_image_misses_the_reflection",
}
for _name, _why, _old, _new, _fails in (
    ("farfield-m-sign",
     "M_s = +n x E makes the J and M halves of the equivalence interfere the wrong way",
     "        ms.append(-np.cross(n, e))          # M_s = -n x E",
     "        ms.append(np.cross(n, e))           # M_s = -n x E",
     _FORMULA_TESTS),
    ("farfield-exponent-sign",
     "exp(+ik r^.r') is the far field of the mirrored surface; a centred cube's pattern cannot see it",
     "        phase = np.exp(-1j * k * (rh @ points.T))         # (chunk, n)",
     "        phase = np.exp(1j * k * (rh @ points.T))          # (chunk, n)",
     _FORMULA_TESTS),
    ("farfield-no-area-weight",
     "a sample without its cell area: a ratio on uniform cells stays right, the power scales with "
     "the sample count and the non-cubic box weights its faces wrongly",
     "    jw = J * dS[:, None]\n    mw = M * dS[:, None]",
     "    jw = J + 0.0 * dS[:, None]\n    mw = M + 0.0 * dS[:, None]",
     _SCALE_TESTS | {"test_off_centre_dipole_in_non_cubic_box_has_the_same_pattern"}),
    ("farfield-scale",
     "dropping k from i k / (4 pi): every normalised pattern passes, the absolute intensity and power do not",
     "    scale = 1j * k / (4.0 * math.pi)",
     "    scale = 1j / (4.0 * math.pi)",
     _SCALE_TESTS),
    ("farfield-e-phi-sign",
     "E_phi with +L_theta: the x-dipole pattern reads it directly; the z-dipole's E_phi, zero only "
     "because its J and M parts cancel, becomes 2 eta N_phi",
     "                    scale * (ETA0 * N_p - L_t))",
     "                    scale * (ETA0 * N_p + L_t))",
     {"test_x_dipole_pattern", "test_collection_fraction_matches_closed_form",
      "test_z_dipole_intensity_is_sin_squared", "test_halving_the_spacing_converges",
      "test_off_centre_dipole_in_non_cubic_box_has_the_same_pattern"}),
    ("farfield-poynting-half",
     "the time average without its 1/2 doubles the surface power",
     "        poynting = 0.5 * np.real(np.cross(face.E, np.conj(face.H), axis=0))  # (3, n_u, n_v)",
     "        poynting = np.real(np.cross(face.E, np.conj(face.H), axis=0))  # (3, n_u, n_v)",
     {"test_z_dipole_surface_power_matches_jackson", "test_far_field_power_equals_surface_power",
      "test_pec_image_reproduces_the_dipole_over_a_mirror",
      "test_open_bottom_without_image_misses_the_reflection"}),
    ("farfield-na-as-angle",
     "the NA used as the cone angle in radians reads 0.124 for the z-dipole instead of 0.208, "
     "and NA 1 is no longer the hemisphere",
     "    inside = cone_power(faces, frequency, math.asin(na), **kwargs)",
     "    inside = cone_power(faces, frequency, na, **kwargs)",
     {"test_collection_fraction_matches_closed_form",
      "test_collection_fraction_is_one_at_full_aperture"}),
    ("farfield-image-sign",
     "a magnetic-wall image (horizontal J kept, vertical negated) instead of the PEC one",
     "        J_img = J * np.array([-1.0, -1.0, 1.0])",
     "        J_img = J * np.array([1.0, 1.0, -1.0])",
     {"test_pec_image_reproduces_the_dipole_over_a_mirror",
      "test_bottom_face_on_the_mirror_plane_is_redundant_with_the_image"}),
    ("farfield-in-plane-order",
     "the +-y faces with (z, x) in-plane axes: sample_box stays consistent with face_points, "
     "only the layout test reads the documented order",
     "_IN_PLANE = {0: (1, 2), 1: (0, 2), 2: (0, 1)}",
     "_IN_PLANE = {0: (1, 2), 1: (2, 0), 2: (0, 1)}",
     {"test_in_plane_axes_are_the_non_normal_axes_in_increasing_order"}),
    ("farfield-square-cell-area",
     "a cell area of du^2: invisible wherever the cells are square, wrong on the 18 x 16 x 20 box "
     "and the graded face",
     "        return np.diff(self.u_edges)[:, None] * np.diff(self.v_edges)[None, :]",
     "        return np.diff(self.u_edges)[:, None] * np.diff(self.u_edges)[None, :]",
     {"test_off_centre_dipole_in_non_cubic_box_has_the_same_pattern",
      "test_graded_edges_use_each_cells_own_area",
      "test_in_plane_axes_are_the_non_normal_axes_in_increasing_order"}),
    ("farfield-first-cell-area",
     "every cell weighted by the first cell's width: a uniform grid cannot tell",
     "        return np.diff(self.u_edges)[:, None] * np.diff(self.v_edges)[None, :]",
     "        return np.full(self.u_edges.size - 1, self.u_edges[1] - self.u_edges[0])[:, None] "
     "* np.diff(self.v_edges)[None, :]",
     {"test_graded_edges_use_each_cells_own_area",
      "test_in_plane_axes_are_the_non_normal_axes_in_increasing_order"}),
):
    CASES.append(Case(name=_name, why=_why, edits=[(FARFIELD, _old, _new)],
                      expect_failing=set(_fails), test_file=FARFIELD_TEST_FILE))

FDTD = ROOT / "dockerfiles" / "maxwell_solver" / "fdtd.py"
PHASOR_TEST_FILE = "tests/test_fdtd_phasor_box.py"

# The phasor box monitor in fdtd.simulate: one reversion per mutation a test
# docstring in tests/test_fdtd_phasor_box.py names, with the exact set each
# turns red (measured). The far-field pattern and the per-sample dipole fit
# see every sign, offset and interpolation slip; only the power and the fitted
# moment's size see the normalisation, which a normalised pattern cannot.
_PHASOR_SHAPE = {"test_the_monitors_far_field_is_the_hertzian_sin_squared",
                 "test_the_monitors_fields_are_one_dipoles_e_and_h"}
for _name, _why, _old, _new, _fails in (
    ("phasor-half-step",
     "H stamped at (n+1) dt like E: the power barely moves (0.9954), the pattern and the "
     "E-H moment phase (6.09 deg) do",
     '"H": xp.asarray(scale * np.exp(phase_sign * omega * (steps + 0.5) * dt),',
     '"H": xp.asarray(scale * np.exp(phase_sign * omega * (steps + 1.0) * dt),',
     _PHASOR_SHAPE),
    ("phasor-dft-sign",
     "a running DFT with exp(-i w t): conjugate phasors, which are not a Maxwell solution at w; "
     "the surface power cannot see it",
     "        phase_sign = 1j\n",
     "        phase_sign = -1j\n",
     _PHASOR_SHAPE),
    ("phasor-uv-swap",
     "the face arrays transposed to (v, u): shape-compatible on a cube, so only the field reads it; "
     "a non-square face is refused by farfield.Face",
     '                    np.stack([c[f] for c in entry["E"]]),\n'
     '                    np.stack([c[f] for c in entry["H"]])))',
     '                    np.stack([c[f].T for c in entry["E"]]),\n'
     '                    np.stack([c[f].T for c in entry["H"]])))',
     _PHASOR_SHAPE | {"test_the_monitor_runs_on_the_device_fake_and_returns_host_faces",
                      "test_without_the_option_nothing_changes"}),
    ("phasor-one-sided",
     "H read half a cell outside the face instead of interpolated across it",
     "                    out = ((above * np.take(out, [0], axis=ax)\n"
     "                            + below * np.take(out, [1], axis=ax))\n"
     "                           / (below + above))",
     "                    out = np.take(out, [1], axis=ax)",
     _PHASOR_SHAPE),
    ("phasor-corner",
     "an in-plane node sample taken at the cell's corner instead of the mean of its two ends",
     "                out = 0.5 * (np.take(out, range(n - 1), axis=ax)\n"
     "                             + np.take(out, range(1, n), axis=ax))",
     "                out = np.take(out, range(n - 1), axis=ax)",
     _PHASOR_SHAPE),
    ("phasor-normalisation",
     "1/sum(w) for 2/sum(w): every power a quarter, every normalised pattern unchanged",
     "        scale = (2.0 / total) * weights[:, None]",
     "        scale = (1.0 / total) * weights[:, None]",
     {"test_the_monitors_surface_power_is_the_delivered_power",
      "test_the_monitors_fields_are_one_dipoles_e_and_h"}),
    ("phasor-one-frequency",
     "every frequency's column phased at the first frequency: the undriven one reads the drive",
     "        omega = 2.0 * np.pi * freqs[None, :]",
     "        omega = 2.0 * np.pi * freqs[None, :1]",
     {"test_the_monitors_surface_power_is_the_delivered_power"}),
    ("phasor-flux-box-rule",
     "the flux box's CPML rule, forgetting the half cell of H a phasor face reads outside itself",
     "        if depth[a] and (lo[a] < depth[a] + 1 or hi[a] > n - 2 - depth[a]):",
     "        if depth[a] and (lo[a] < depth[a] or hi[a] > n - 1 - depth[a]):",
     {"test_a_phasor_box_in_the_absorber_is_refused"}),
    ("phasor-bloch-allowed",
     "a paired Bloch run allowed a phasor box: its cos set alone transformed as the field",
     "    if phasor_box is not None and paired:",
     "    if phasor_box is not None and paired and False:",
     {"test_a_bloch_run_refuses_a_phasor_box"}),
):
    CASES.append(Case(name=_name, why=_why, edits=[(FDTD, _old, _new)],
                      expect_failing=set(_fails), test_file=PHASOR_TEST_FILE))

SOURCE_EPS_TEST_FILE = "tests/test_fdtd_source_eps.py"
PAPER1 = ROOT / "scripts" / "paper1_nanowire.py"

# The permittivity the update applies at a point source (change
# source-cell-permittivity): fdtd.source_permittivity, simulate's
# "source_eps", and the paper-1 arm that reads it. Each set is measured.
_CLAIMS = "test_a_metal_claims_exactly_the_samples_its_update_rescales"
for _name, _why, _path, _old, _new, _fails, _file in (
    ("source-eps-cell-value",
     "source_permittivity returns the layout's node value, ignoring the metal's eps_inf",
     FDTD,
     "    return eps / (1.0 + rescale)\n",
     "    return eps\n",
     {"test_a_metal_face_applies_eps_inf_and_only_that_reads_true",
      f"{_CLAIMS}[node0-z-1.0]", f"{_CLAIMS}[node1-z-1.0]",
      f"{_CLAIMS}[node4-x-1.0]"},
     SOURCE_EPS_TEST_FILE),
    ("source-eps-wrong-component",
     "the metal's claim read for the next component's sample, not the source's own",
     FDTD,
     "        eps_inf = applied(component, cell, shape, periodic)\n",
     "        eps_inf = applied(COMPONENTS[(COMPONENTS.index(component) + 1) % 3],"
     " cell, shape, periodic)\n",
     {"test_a_metal_face_applies_eps_inf_and_only_that_reads_true",
      f"{_CLAIMS}[node0-z-1.0]", f"{_CLAIMS}[node3-z-4.0]",
      f"{_CLAIMS}[node4-x-1.0]", f"{_CLAIMS}[node5-x-4.0]"},
     SOURCE_EPS_TEST_FILE),
    ("source-eps-simulate-no-terms",
     "simulate's source_eps computed without the run's polarisation terms",
     FDTD,
     "                                      ade_terms=terms, boundary=boundary)\n",
     "                                      ade_terms=(), boundary=boundary)\n",
     {"test_a_metal_face_applies_eps_inf_and_only_that_reads_true"},
     SOURCE_EPS_TEST_FILE),
    ("paper1-eps-cell",
     "the paper-1 arm hands its readings the layout's node value instead of the applied one",
     PAPER1,
     "    return source, terms, volume, index, eps_applied\n",
     "    return source, terms, volume, index, eps_cell\n",
     {"test_the_source_line_prints_the_cell_and_the_applied_permittivity"},
     "tests/test_paper1_nanowire.py"),
    ("paper1-eps-print",
     "the paper-1 source line prints the cell value where it names the applied one",
     PAPER1,
     '          f" (cell), {eps_applied:.4f} (applied at E{args.axis})"\n',
     '          f" (cell), {eps_cell:.4f} (applied at E{args.axis})"\n',
     {"test_the_source_line_prints_the_cell_and_the_applied_permittivity"},
     "tests/test_paper1_nanowire.py"),
):
    CASES.append(Case(name=_name, why=_why, edits=[(_path, _old, _new)],
                      expect_failing=set(_fails), test_file=_file))


COLLECTION_TEST_FILE = "tests/test_paper1_collection.py"
MEMORY_TEST_FILE = "tests/test_fdtd_memory.py"

# The paper-1 collection arm (change paper1-collection): the box placement, its
# refusals, the arm's reading, the off-by-default path, and the phasor box in
# the memory estimate. Each set is measured.
_ENCLOSE = {f"test_the_box_encloses_the_whole_scene_inside_the_open_pad[{k}]"
            for k in ("tiny", "tight", "paper", "run-1")}
_COUNTED = "test_a_phasor_box_is_counted_as_the_monitor_allocates_it"
_NO_ROOM = "test_a_layout_with_no_room_for_the_box_is_refused"
_VACUUM = "test_the_arm_reads_the_free_dipole_in_vacuum"
_PRICED = "test_the_box_is_priced_into_the_estimate_the_caps_read"
_LADDER_PRICES = "test_the_ladder_prices_the_box_on_every_rung"
_LADDER_REFUSES = "test_the_ladder_refuses_a_rung_with_no_room"
_OFF = "test_without_the_flag_the_run_is_the_one_it_was"
_INSIDE = {f"test_a_phasor_box_run_stays_inside_its_estimate[{k}]"
           for k in ("box0-frequencies0-60-20", "box1-frequencies1-80-10",
                     "box2-frequencies2-60-30")}
for _name, _why, _path, _old, _new, _fails, _file in (
    ("collection-parts-frame",
     "the parts' bounds taken in their own centred frame, not the layout's (origin at the "
     "bounding box's minimum corner): the box shifts by half the mirror",
     PAPER1,
     "    return (0.0, 0.0, 0.0), (width, width, geom[\"z_top\"])\n",
     "    return (-0.5 * width, -0.5 * width, 0.0), (0.5 * width, 0.5 * width,"
     " geom[\"z_top\"])\n",
     _ENCLOSE | {_NO_ROOM, _VACUUM, _PRICED, _LADDER_PRICES, _LADDER_REFUSES, _OFF},
     COLLECTION_TEST_FILE),
    ("collection-no-clearance",
     "the high faces placed without the 2 layers of clearance from the parts",
     PAPER1,
     "        hi_range = (last + COLLECTION_CLEAR_CELLS, n - 2 - pml)\n",
     "        hi_range = (last, n - 2 - pml)\n",
     _ENCLOSE | {_NO_ROOM, _VACUUM, _LADDER_REFUSES},
     COLLECTION_TEST_FILE),
    ("collection-absorber",
     "the low faces placed from layer 1, forgetting the absorber",
     PAPER1,
     "        lo_range = (pml + 1, first - COLLECTION_CLEAR_CELLS)\n",
     "        lo_range = (1, first - COLLECTION_CLEAR_CELLS)\n",
     _ENCLOSE | {_VACUUM},
     COLLECTION_TEST_FILE),
    ("collection-parity",
     "any pair of layers taken, so centre +/- half_extent is not the faces picked",
     PAPER1,
     "            if (lo + hi) % 2 == 0 and (best is None or i + j < best[0]):\n",
     "            if best is None or i + j < best[0]:\n",
     _ENCLOSE | {_VACUUM},
     COLLECTION_TEST_FILE),
    ("collection-shell-metal",
     "the faces' vacuum check reads the permittivity only, not the metal masks",
     PAPER1,
     "                if np.any(eps[shell] != 1.0) or np.any(solid[shell]):\n",
     "                if np.any(eps[shell] != 1.0):\n",
     {"test_a_face_through_material_is_refused"},
     COLLECTION_TEST_FILE),
    ("collection-pulse-allowed",
     "the arm allowed on a pulse record, which has no one frequency to read",
     PAPER1,
     "        if args.profile != \"cw\":\n",
     "        if False:\n",
     {"test_the_arm_refuses_what_it_cannot_read"},
     COLLECTION_TEST_FILE),
    ("collection-whole-record",
     "the box read from step 0, through the ramp, not over the CW reading's tail",
     PAPER1,
     "            \"start\": int(steps) - tail, \"steps\": tail}\n",
     "            \"start\": 0, \"steps\": tail}\n",
     {_VACUUM, _PRICED, _OFF},
     COLLECTION_TEST_FILE),
    ("collection-of-all",
     "the fraction printed over all the radiated power, not the upward hemisphere",
     PAPER1,
     "    fraction = farfield.collection_fraction(faces, frequency, na)\n",
     "    fraction = cone / sphere\n",
     {_VACUUM, _OFF},
     COLLECTION_TEST_FILE),
    ("collection-run-without-box",
     "the run never handed the box: the arm has no faces to read",
     PAPER1,
     "    extra = ({} if phasor is None else\n",
     "    extra = ({} if True else\n",
     {_VACUUM, _OFF},
     COLLECTION_TEST_FILE),
    ("collection-unpriced",
     "the run path's estimate without the box: the caps check a run smaller than the one run",
     PAPER1,
     "                            pec=lay[\"pec\"], **phasor_estimate(phasor))\n",
     "                            pec=lay[\"pec\"])\n",
     {_PRICED},
     COLLECTION_TEST_FILE),
    ("collection-ladder-unpriced",
     "the ladder's rungs priced without the box",
     PAPER1,
     "                est[\"bytes\"] += fdtd.phasor_box_bytes(\n",
     "                est[\"bytes\"] += 0 * fdtd.phasor_box_bytes(\n",
     {_LADDER_PRICES},
     COLLECTION_TEST_FILE),
    ("memory-phasor-uncounted",
     "memory_estimate ignores a phasor box",
     FDTD,
     "    if phasor_box is not None:\n        total += phasor_box_bytes(",
     "    if False:\n        total += phasor_box_bytes(",
     {_COUNTED},
     MEMORY_TEST_FILE),
    ("memory-phasor-uncounted-arm",
     "the same, seen from the paper-1 run path",
     FDTD,
     "    if phasor_box is not None:\n        total += phasor_box_bytes(",
     "    if False:\n        total += phasor_box_bytes(",
     {_PRICED},
     COLLECTION_TEST_FILE),
    ("memory-phasor-tables",
     "the phase tables left out of the box's bytes",
     FDTD,
     "    total += 2 * 16 * count * int(steps)         # the E and H phase tables\n",
     "",
     {_COUNTED},
     MEMORY_TEST_FILE),
    ("memory-phasor-assembly",
     "the faces' assembly after the run left out: the estimate is under the measured peak",
     FDTD,
     "    return int(total + max(step, assembly))\n",
     "    return int(total + step)\n",
     {_COUNTED} | _INSIDE,
     MEMORY_TEST_FILE),
    ("memory-phasor-estimate-run",
     "estimate_run takes the box and does not pass it on",
     FDTD,
     "                                 max_frames=max_frames, phasor_box=phasor_box,\n",
     "                                 max_frames=max_frames, phasor_box=None,\n",
     {_COUNTED},
     MEMORY_TEST_FILE),
):
    CASES.append(Case(name=_name, why=_why, edits=[(_path, _old, _new)],
                      expect_failing=set(_fails), test_file=_file))


SETTLING_TEST_FILE = "tests/test_cw_settling.py"
_ONE_EXP = "test_one_exponential_approach_is_class_b_with_its_own_tau_and_limit"
_TWO_EXP = "test_two_time_constants_is_an_approach_that_says_it_holds_more_than_one"
_GROWTH = "test_a_slow_exponential_growth_is_class_c_and_never_a_purcell_factor"
_LINE = "test_a_straight_rise_and_a_short_record_are_class_d"
_COVER = "test_the_error_bars_cover_the_truth_on_noisy_records"
_SCALE = "test_the_fit_is_as_well_conditioned_at_any_scale"
_PHASE_LOCK = "test_the_curve_is_cut_at_the_end_s_own_phase"
_VACUUM_BOX = "test_an_open_vacuum_box_swings_by_its_own_prediction"
_BOX_LIMIT = "test_the_lossy_box_s_limit_is_the_one_a_longer_record_reaches"
_LOCATOR = "test_the_locator_line_says_when_one_mode_cannot_make_the_curve"
_ROUND_TRIP = "test_the_saved_series_gives_the_identical_reading"
_SWING = {f"test_the_window_phase_swing_is_the_leakage_times_the_quadrature[{q}]"
          for q in ("100.0", "2000.0")}

# The CW arm's settling curve and its class (change cw-settling). Each set is
# measured: the reversion applied, the file run, exactly these tests red.
for _name, _why, _old, _new, _fails in (
    ("settling-free-phase",
     "the curve's cuts at free 5% marks, not a whole number of periods before the end",
     "            cut = n - int(round(j * per))\n",
     "            cut = n - int(round(target * per))\n",
     {_PHASE_LOCK}),
    ("settling-ramp-fitted",
     "cuts whose window still holds the drive's ramp are fitted",
     "                     \"fitted\": F is not None and cut - window >= ramp})\n",
     "                     \"fitted\": F is not None})\n",
     {_GROWTH, _LINE, _ONE_EXP, _PHASE_LOCK, _COVER, _LOCATOR, _BOX_LIMIT,
      _VACUUM_BOX}),
    ("settling-limit-sign",
     "the approach's limit taken as c - D/u",
     "        out.update(F_inf=float(c + D / u) * scale,\n",
     "        out.update(F_inf=float(c - D / u) * scale,\n",
     {_ONE_EXP, _BOX_LIMIT}),
    ("settling-independent-cuts",
     "the fit's errors computed as if cuts sharing window samples were independent",
     "    fit = fit_approach(t, F, floor=floor, corr=corr)\n",
     "    fit = fit_approach(t, F, floor=floor, corr=None)\n",
     {_COVER}),
    ("settling-raw-scale",
     "the fit run on raw readings, so a large F loses a covariance direction",
     "    scale = float(np.max(np.abs(raw))) or 1.0\n",
     "    scale = 1.0\n",
     {_COVER, _SCALE}),
    ("settling-no-growth-class",
     "a negative rate never classed as growth",
     "    elif fit[\"u\"] + SIGNIFICANCE * fit[\"u_err\"] < 0.0:\n",
     "    elif fit[\"u\"] + SIGNIFICANCE * fit[\"u_err\"] < -math.inf:\n",
     {_GROWTH}),
    ("settling-misfit-as-noise",
     "the settled test reads a one-exponential misfit's residual as noise",
     "    if abs(drift) <= SIGNIFICANCE * drift_err:\n",
     "    if abs(drift) <= SIGNIFICANCE * math.sqrt(2.0) * max(noise, fit[\"rms\"]):\n",
     {_TWO_EXP}),
    ("settling-phase-once",
     "the phase prediction counts the -f image's leakage once, not through both transforms",
     "            \"quadrature\": quadrature, \"predicted\": 2.0 * leak * quadrature\n",
     "            \"quadrature\": quadrature, \"predicted\": 1.0 * leak * quadrature\n",
     {_VACUUM_BOX} | _SWING),
    ("settling-phase-end-window",
     "the quadrature read from the end window as it is, its own share of the swing left in",
     "    inphase *= abs(float(coef[0]) / ratios[-1]) if ratios[-1] else 0.0\n",
     "    inphase *= 1.0\n",
     {_VACUUM_BOX} | _SWING),
    ("settling-no-phase-floor",
     "the curve's floor taken as zero instead of the swing at the cuts' phase mismatch",
     "        floor = phase[\"swing\"] * 4.0 * math.pi * miss\n",
     "        floor = 0.0 * miss\n",
     {_VACUUM_BOX, _PHASE_LOCK}),
    ("settling-locator-never-longer",
     "the locator line never says the curve is too slow for the locator's mode",
     "        if tau - SIGNIFICANCE * err > tau_l + SIGNIFICANCE * err_l:\n",
     "        if False:\n",
     {_LOCATOR}),
    ("settling-series-float64",
     "the series saved as float64, not at the dtype the run recorded",
     "    np.savez_compressed(path, drive=np.asarray(drive), source_field=np.asarray(field),\n",
     "    np.savez_compressed(path, drive=np.asarray(drive),"
     " source_field=np.asarray(field, dtype=np.float64),\n",
     {_ROUND_TRIP}),
    ("settling-series-no-eps",
     "the series saved without the applied permittivity the reading needs",
     "                        eps_r=float(eps_r), index=float(index),\n",
     "                        index=float(index),\n",
     {_ROUND_TRIP}),
):
    CASES.append(Case(name=_name, why=_why, edits=[(PAPER1, _old, _new)],
                      expect_failing=set(_fails), test_file=SETTLING_TEST_FILE))


BASIN_TEST_FILE = "tests/test_decay_fit_basin.py"
_B1060 = "test_a_1060_fs_mode_between_two_grid_columns_recovers_its_tau"
_BSWEEP = "test_no_wrong_basin_across_the_linewidth_to_grid_ratio"
_BWIDE = "test_a_line_wider_than_the_band_is_searched_across_its_width"
_BEQUAL = "test_two_equal_modes_are_flagged_ambiguous"
_BNOISE = "test_a_mode_under_the_noise_and_noise_alone_are_flagged"
_BBIAS = "test_a_second_mode_that_biases_tau_is_flagged_below_the_factor"
_BWEAK = "test_a_clean_mode_and_a_weak_second_mode_are_not_flagged"
_BADOPT = "test_the_answer_does_not_depend_on_which_cell_the_scan_ranked_first"
_BDEGEN = "test_a_degenerate_tau_direction_is_flagged"
_BARM = "test_the_arm_prints_the_reason_where_tau_would_be"

# fit_decay's search and its ambiguity flag (change decay-fit-basin). Each set
# is measured: the reversion applied, the file run, exactly these tests red.
for _name, _why, _old, _new, _fails in (
    ("basin-coarse-omega",
     "the scan's omega step asked for at a twentieth of the band, not 1/(8 windows)"
     " (padding then only to the record's own power of two)",
     "    step = min(2.0 * math.pi / (DECAY_PAD * window), (hi_w - lo_w) / 20.0)\n",
     "    step = (hi_w - lo_w) / 20.0\n",
     {_BSWEEP}),
    ("basin-pad-one",
     "the scan's omega bins 1/window apart instead of 1/(8 windows)",
     "DECAY_PAD = 8 ",
     "DECAY_PAD = 1 ",
     {_BSWEEP}),
    ("basin-two-bin-window",
     "the descent's omega window two bins either side at every tau, not the line's width",
     "        half = max(2.0 * step_w, 1.0 / float(taus[i]))\n",
     "        half = 2.0 * step_w\n",
     {_BWIDE}),
    ("basin-no-rival-flag",
     "a rival minimum's residual never makes the fit ambiguous",
     "    if ratio is not None and ratio < AMBIGUITY_RATIO:\n",
     "    if False:\n",
     {_BEQUAL, _BNOISE, _BBIAS, _BARM}),
    ("basin-factor-three",
     "the ambiguity factor 3 instead of the measured 12",
     "AMBIGUITY_RATIO = 12.0 ",
     "AMBIGUITY_RATIO = 3.0 ",
     {_BBIAS}),
    ("basin-factor-fifty",
     "the ambiguity factor 50 instead of the measured 12",
     "AMBIGUITY_RATIO = 12.0 ",
     "AMBIGUITY_RATIO = 50.0 ",
     {_BWEAK}),
    ("basin-no-degenerate-flag",
     "a degenerate tau direction in the covariance is not a reason",
     "    if tau_err is None:\n        reasons.append(",
     "    if False:\n        reasons.append(",
     {_BDEGEN}),
    ("basin-arm-reads-q",
     "the arm reads a Q from an ambiguous fit instead of stopping at the reason",
     "    if ambiguous:\n        # Before the Q line",
     "    if False:\n        # Before the Q line",
     {_BARM}),
    ("basin-arm-prints-tau",
     "the arm prints an ambiguous tau as tau = ... +- ...",
     "                f\" {ambiguous})\" if ambiguous else\n",
     "                f\" {ambiguous})\" if False else\n",
     {_BARM}),
    ("basin-rival-not-adopted",
     "a rival that refines lower than the scan's first cell is not made the fit",
     "        if other[0] < rss:\n",
     "        if False:\n",
     {_BADOPT}),
):
    CASES.append(Case(name=_name, why=_why, edits=[(PAPER1, _old, _new)],
                      expect_failing=set(_fails), test_file=BASIN_TEST_FILE))

# The whole search put back as it was before the change: 21 omegas across the
# band, the best cell, three coordinate sweeps whose steps shrink after the
# first, and no rival.
_NEW_SEARCH = (
    "    minima = _local_minima(S)\n"
    "    i, j = (int(k) for k in minima[0])\n"
    "    at_edge = i == taus.size - 1\n"
    "    rss, tau, omega = refine(i, j)\n")
_OLD_SEARCH = (
    "    omegas = np.linspace(lo_w, hi_w, 21)\n"
    "    grid = [(rss_at(g_tau, g_om)[0], float(g_tau), float(g_om))\n"
    "            for g_tau in taus for g_om in omegas]\n"
    "    rss, tau, omega = min(grid)\n"
    "    at_edge = math.isclose(tau, float(taus[-1]), rel_tol=1e-9)\n"
    "    step_tau = (taus[-1] - taus[0]) / (taus.size - 1)\n"
    "    step_w = (omegas[-1] - omegas[0]) / (omegas.size - 1)\n"
    "    for sweep in range(3):\n"
    "        tau = _golden(lambda v: rss_at(v, omega)[0],\n"
    "                      max(tau - 1.5 * step_tau, 1e-3 * window), tau + 1.5 * step_tau)\n"
    "        omega = _golden(lambda v: rss_at(tau, v)[0],\n"
    "                        omega - 1.5 * step_w, omega + 1.5 * step_w)\n"
    "        if sweep == 0:\n"
    "            step_tau, step_w = step_tau * 0.05, step_w * 0.05\n"
    "    minima = []\n")
CASES.append(Case(
    name="basin-old-search",
    why="fit_decay's search as it was: 21 omegas, the best cell, a shrinking descent",
    edits=[(PAPER1, _NEW_SEARCH, _OLD_SEARCH)],
    expect_failing={_B1060, _BSWEEP, _BEQUAL, _BNOISE, _BBIAS, _BWEAK, _BADOPT, _BARM},
    test_file=BASIN_TEST_FILE))


def run_case(case: Case) -> bool:
    saved = case.snapshot()
    ok = True
    try:
        case.apply()
        # ``-B`` / PYTHONDONTWRITEBYTECODE, and it is not an optimization. This
        # tool reverts a source file and runs pytest against it; without this,
        # pytest imports the *reverted* module and writes a matching .pyc, which
        # then outlives the restore. The next run serves that stale bytecode
        # while the source on disk is correct -- so tests fail for a reason that
        # is not in the tree, every case reports them as "failed but not
        # expected", and the checker exits 1 on a tree that is fine. Reproduced
        # on 2026-09-27: a web/db.py compiled before the isfile change made the
        # four test_schema_paths.py cases fail while the source used isfile.
        # The failure moves between runs, which is what makes it worth fixing
        # rather than re-reading: it is the same defect this file warns about in
        # its own docstring -- a half-reverted artefact outliving the run that
        # made it. The cure is not to compile it in the first place.
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run(
            [sys.executable, "-B", "-m", "pytest", case.test_file, "-q",
             "--no-header", "-p", "no:randomly", "--tb=no"],
            cwd=str(ROOT), capture_output=True, text=True, env=env,
        )
        # "Unrunnable" and "passing" must not be the same input. If pytest never
        # ran -- wrong interpreter, import error, a collection failure -- stdout
        # holds no FAILED lines and every expected set would compare empty and
        # report MISMATCH, or worse, a case expecting nothing would report OK.
        # That is the same class of error as the 403 defect: an empty result
        # standing in for "we could not ask".
        if proc.returncode not in (0, 1):
            print(f"\n=== {case.name} ===")
            print(f"  COULD NOT RUN: pytest exited {proc.returncode}")
            tail = (proc.stderr or proc.stdout).strip().splitlines()[-4:]
            for line in tail:
                print(f"    {line}")
            ok = False
            return ok
        failing = set()
        for line in proc.stdout.splitlines():
            if not line.startswith("FAILED"):
                continue
            # "FAILED tests/x.py::test_name - AssertionError: ..." -- take the
            # nodeid, drop the reason, or the parsed name keeps the - Assert...
            # tail and never matches the expected set.
            node = line.split(" - ", 1)[0].strip()
            if "::" in node:
                failing.add(node.split("::")[-1])
        print(f"\n=== {case.name} ===")
        print(f"why: {case.why}")
        print(f"reverted -> {len(failing)} failing test(s); "
              f"expected {len(case.expect_failing)}")
        missing = case.expect_failing - failing
        if failing != case.expect_failing:
            ok = False
            print("  MISMATCH")
            if missing:
                print(f"    expected to fail but passed: {sorted(missing)}")
            extra = failing - case.expect_failing
            if extra:
                print(f"    failed but not expected: {sorted(extra)}")
        else:
            print(f"  OK - exactly the expected tests fail, and only they")
        return ok
    finally:
        case.restore(saved)
        print("  source restored and verified by hash")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("case", nargs="?", help="which case to run")
    parser.add_argument("--list", action="store_true", help="list the cases")
    args = parser.parse_args(argv)

    # Fail here rather than reporting 18 clean mismatches. This checker shells
    # out with sys.executable; run it under an interpreter without pytest and
    # every subprocess dies on import, stdout holds no FAILED lines, and each
    # case compares an empty set. On 2026-09-27 that cost a round of redesigning
    # tests that were never the problem.
    import importlib.util
    if importlib.util.find_spec("pytest") is None:
        raise SystemExit(
            f"this checker runs pytest as a subprocess of {sys.executable!r}, "
            "which cannot import it -- use the venv:\n"
            "  .venv-mse/bin/python tests/check_discriminating.py"
        )

    # One run at a time -- taken before any case can revert a file.
    take_lock()

    if args.list:
        for case in CASES:
            print(f"{case.name:28} {case.why}")
        return 0

    chosen = CASES
    if args.case:
        chosen = [c for c in CASES if c.name == args.case]
        if not chosen:
            raise SystemExit(f"no such case: {args.case}")
    results = [run_case(case) for case in chosen]
    print()
    if all(results):
        print(f"all {len(chosen)} case(s) discriminating: every reversion is "
              "caught by the tests that claim to cover it")
        return 0
    print("NOT discriminating: at least one reversion went uncaught")
    return 1


if __name__ == "__main__":
    sys.exit(main())
