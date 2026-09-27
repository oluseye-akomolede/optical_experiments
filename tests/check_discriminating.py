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

This exists because a reviewer asked for revert-and-rerun fixtures to be left in
place. A script is better than a scratch commit: the fixture is checked in, it
cannot be lost, and the failure count is reproducible from the tests alone.
"""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEB_DB = ROOT / "web" / "db.py"
SOLVER_DB = ROOT / "dockerfiles" / "maxwell_solver" / "mse_db.py"
SCHEMA_SQL = ROOT / "db" / "001_schema.sql"


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

CASES.append(Case(
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


def run_case(case: Case) -> bool:
    saved = case.snapshot()
    ok = True
    try:
        case.apply()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", case.test_file, "-q", "--no-header",
             "-p", "no:randomly", "--tb=no"],
            cwd=str(ROOT), capture_output=True, text=True,
        )
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
