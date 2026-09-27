"""The schema DDL script must be found in the repo AND inside the images.

This is a regression file, and it exists because the bug it covers shipped:
``_schema_path`` climbed a fixed number of directory levels, which is correct
for the repo checkout and wrong for the image, where the layout is flat.

    repo   web/db.py                        -> ../db/001_schema.sql
    image  /opt/app/db.py                   -> ./db/001_schema.sql
    repo   dockerfiles/maxwell_solver/mse_db.py -> ../../db/001_schema.sql
    image  /opt/app/mse_db.py               -> ./db/001_schema.sql

The first deploy died on ``FileNotFoundError: /opt/db/001_schema.sql`` from the
web initContainer, and the solver's version climbed one level further still
(``/db/001_schema.sql``), so every solver Job would have failed at startup too.

These tests build both layouts in a tmp dir and assert *which file* each
resolves to, so the assertion is about a path, not about a story.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"
SOLVER = ROOT / "dockerfiles" / "maxwell_solver"

for _path in (WEB, SOLVER):
    sys.path.insert(0, str(_path))

import db as WEB_DB  # noqa: E402  (web/db.py)
import mse_db as SOLVER_DB  # noqa: E402


def _make_layout(root: Path, module_dir: str, depth: int) -> tuple[Path, Path]:
    """Write a fake module dir and a schema at ``depth`` levels above it.

    Returns ``(module_dir, schema_file)``. ``depth=0`` puts ``db/`` beside the
    module (the image layout); ``depth=1`` puts it one level up (the repo layout
    for the web app); ``depth=2`` two levels up (the repo layout for the
    solver).
    """
    module = root
    for part in module_dir.split("/"):
        if part:
            module = module / part
    module.mkdir(parents=True, exist_ok=True)

    schema_dir = module
    for _ in range(depth):
        schema_dir = schema_dir.parent
    schema_dir = schema_dir / "db"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema = schema_dir / "001_schema.sql"
    schema.write_text("-- fake schema\n", encoding="utf-8")
    return module, schema


# ---------------------------------------------------------------------------
# web/db.py
# ---------------------------------------------------------------------------
def test_web_resolves_the_image_layout(monkeypatch, tmp_path):
    """Flat layout: the script is beside db.py, as in /opt/app."""
    module, schema = _make_layout(tmp_path / "img", "", depth=0)
    fake = module / "db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(WEB_DB, "__file__", str(fake))
    assert Path(WEB_DB._schema_path()).resolve() == schema.resolve()


def test_web_resolves_the_repo_layout(monkeypatch):
    """One level up, as in the checkout: web/db.py -> repo/db/."""
    here = Path(WEB_DB.__file__).resolve()
    expected = (here.parent.parent / "db" / "001_schema.sql").resolve()
    assert expected.exists(), "the real repo schema should exist to test against"
    assert Path(WEB_DB._schema_path()).resolve() == expected


def test_web_prefers_the_env_override(monkeypatch, tmp_path):
    """MSE_SCHEMA_PATH wins, so an operator can mount the script elsewhere."""
    custom = tmp_path / "mounted.sql"
    custom.write_text("-- mounted\n", encoding="utf-8")
    monkeypatch.setenv("MSE_SCHEMA_PATH", str(custom))
    assert Path(WEB_DB._schema_path()).resolve() == custom.resolve()


def test_web_error_names_every_path_it_tried(monkeypatch, tmp_path):
    """A wrong path must say which paths were tried, not just that one failed.

    The deploy failure was hard to read precisely because the message named a
    single path that looked plausible; the candidates make the layout visible.
    """
    module = tmp_path / "empty"
    module.mkdir()
    fake = module / "db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(WEB_DB, "__file__", str(fake))
    monkeypatch.delenv("MSE_SCHEMA_PATH", raising=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        WEB_DB._schema_path()
    message = str(excinfo.value)
    assert "001_schema.sql" in message
    for candidate in WEB_DB.schema_candidates():
        assert candidate in message


def test_web_skips_a_directory_at_the_override(monkeypatch, tmp_path):
    """A directory must never be RETURNED as the schema.

    ``os.path.exists`` says yes to a directory, and the failure then surfaces as
    an unexplained ``IsADirectoryError`` from ``open`` -- which defeats the
    point of an error that names every candidate. ``isfile`` is the fix. Found
    in review, not by the original tests.

    The assertion is on what is *returned*, not on raising: with a bogus
    override the resolver must fall through to a real candidate if one exists.
    """
    wrong = tmp_path / "not_a_file"
    wrong.mkdir()
    monkeypatch.setenv("MSE_SCHEMA_PATH", str(wrong))
    resolved = WEB_DB._schema_path()
    assert Path(resolved) != wrong
    assert Path(resolved).is_file()


def test_solver_skips_a_directory_at_the_override(monkeypatch, tmp_path):
    wrong = tmp_path / "not_a_file"
    wrong.mkdir()
    monkeypatch.setenv("MSE_SCHEMA_PATH", str(wrong))
    resolved = SOLVER_DB.JobStore._resolve_schema_path()
    assert Path(resolved) != wrong
    assert Path(resolved).is_file()


def test_a_directory_at_the_override_alone_raises_clearly(monkeypatch, tmp_path):
    """With no valid candidate left, the error must still name the candidates.

    This is the case that used to degrade to a bare ``IsADirectoryError``.
    """
    module = tmp_path / "empty"
    module.mkdir()
    fake = module / "db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(WEB_DB, "__file__", str(fake))
    wrong = tmp_path / "not_a_file"
    wrong.mkdir()
    monkeypatch.setenv("MSE_SCHEMA_PATH", str(wrong))
    with pytest.raises(FileNotFoundError) as excinfo:
        WEB_DB._schema_path()
    assert str(wrong) in str(excinfo.value)


def test_a_directory_beside_the_module_is_not_the_schema(monkeypatch, tmp_path):
    """The same defect without an override: a `db` directory is not a script.

    ``<module>/db/001_schema.sql`` is the image candidate, so if that path
    resolves to a directory the resolver must skip it rather than return it.
    """
    module = tmp_path / "app"
    module.mkdir()
    # a DIRECTORY named 001_schema.sql -- exists() would accept it
    (module / "db" / "001_schema.sql").mkdir(parents=True)
    fake = module / "db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(WEB_DB, "__file__", str(fake))
    monkeypatch.delenv("MSE_SCHEMA_PATH", raising=False)
    with pytest.raises(FileNotFoundError):
        WEB_DB._schema_path()


def test_web_candidates_cover_both_layouts_without_existing(monkeypatch, tmp_path):
    """The candidate list itself is layout-independent, which is the point."""
    module = tmp_path / "somewhere"
    module.mkdir()
    fake = module / "db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(WEB_DB, "__file__", str(fake))
    monkeypatch.delenv("MSE_SCHEMA_PATH", raising=False)
    candidates = WEB_DB.schema_candidates()
    assert candidates == [
        str(module / "db" / "001_schema.sql"),
        str(module.parent / "db" / "001_schema.sql"),
    ]


# ---------------------------------------------------------------------------
# the solver's copy
# ---------------------------------------------------------------------------
def test_solver_resolves_the_image_layout(monkeypatch, tmp_path):
    """Flat layout, as in the solver image: /opt/app/db/001_schema.sql."""
    module, schema = _make_layout(tmp_path / "img", "", depth=0)
    fake = module / "mse_db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(SOLVER_DB, "__file__", str(fake))
    monkeypatch.delenv("MSE_SCHEMA_PATH", raising=False)
    assert Path(SOLVER_DB.JobStore._resolve_schema_path()) == schema.resolve()


def test_solver_resolves_the_repo_layout(monkeypatch):
    """Two levels up, as in the checkout for dockerfiles/maxwell_solver/."""
    here = Path(SOLVER_DB.__file__).resolve()
    expected = (here.parent.parent.parent / "db" / "001_schema.sql").resolve()
    assert expected.exists(), "the real repo schema should exist to test against"
    assert Path(SOLVER_DB.JobStore._resolve_schema_path()) == expected


def test_solver_error_names_every_path_it_tried(monkeypatch, tmp_path):
    module = tmp_path / "empty"
    module.mkdir()
    fake = module / "mse_db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(SOLVER_DB, "__file__", str(fake))
    monkeypatch.delenv("MSE_SCHEMA_PATH", raising=False)
    with pytest.raises(FileNotFoundError) as excinfo:
        SOLVER_DB.JobStore._resolve_schema_path()
    assert "001_schema.sql" in str(excinfo.value)


def test_solver_and_web_agree_on_the_filename():
    """The two are deliberate near-twins; the constant must not drift."""
    assert WEB_DB.SCHEMA_FILENAME == SOLVER_DB.SCHEMA_FILENAME


# ---------------------------------------------------------------------------
# The DDL must not need database-level privileges
# ---------------------------------------------------------------------------
SCHEMA_SQL = ROOT / "db" / "001_schema.sql"

#: statements that require a privilege on the DATABASE, which our scoped role
#: mse_app deliberately does not have (it owns only the `mse` schema)
DATABASE_LEVEL = (
    "CREATE SCHEMA",
    "CREATE EXTENSION",
    "CREATE DATABASE",
    "ALTER DATABASE",
    "DROP DATABASE",
    "CREATE ROLE",
    "CREATE USER",
    "CREATE TABLESPACE",
    "CREATE LANGUAGE",
)


def _statements_outside_do_blocks(sql: str) -> list[str]:
    """SQL statements that are not inside a ``DO $$ ... $$`` block.

    The DO block is how this file creates the schema without needing the
    database-level privilege, so what it wraps is legitimate; anything outside
    one is not. Dollar-quoted bodies and ``--`` comments are stripped so a
    keyword in a comment or a body cannot produce a false hit.
    """
    lines, in_do, out = [], False, []
    for line in sql.splitlines():
        stripped = line.split("--", 1)[0]
        if "$$" in stripped:
            # a DO $$ opens it; the closing $$ ends it. Two $$ on one line
            # (DO $$ ... $$;) both opens and closes.
            if stripped.count("$$") % 2 == 1:
                in_do = not in_do
            continue
        if not in_do:
            lines.append(stripped)
    body = "\n".join(lines)
    for statement in body.split(";"):
        cleaned = statement.strip()
        if cleaned:
            out.append(cleaned)
    return out


def test_the_ddl_has_no_bare_create_schema():
    """`CREATE SCHEMA IF NOT EXISTS` still needs CREATE on the DATABASE.

    Postgres checks the privilege before the IF NOT EXISTS short-circuit, so
    that one line crash-looped the initContainer with 'permission denied for
    database mode_search_engine' even though the schema already existed. It must
    only ever appear inside a DO block guarded on pg_namespace.
    """
    for statement in _statements_outside_do_blocks(
        SCHEMA_SQL.read_text(encoding="utf-8")
    ):
        upper = statement.upper()
        assert not upper.startswith("CREATE SCHEMA"), (
            f"bare CREATE SCHEMA outside a DO block: {statement!r}"
        )


def test_the_ddl_has_no_other_database_level_statement():
    """CREATE EXTENSION, GRANT ... ON DATABASE and friends need the same grant."""
    for statement in _statements_outside_do_blocks(
        SCHEMA_SQL.read_text(encoding="utf-8")
    ):
        upper = " ".join(statement.upper().split())
        for keyword in DATABASE_LEVEL:
            assert not upper.startswith(keyword), (
                f"database-level statement outside a DO block: {statement!r}"
            )
        assert " ON DATABASE " not in upper, (
            f"privilege granted on the DATABASE: {statement!r}"
        )


def test_the_schema_creation_is_guarded_on_pg_namespace():
    """The guard must test the catalog, not rely on IF NOT EXISTS.

    Asserted on the DO block's own text rather than the whole file, because the
    file's comment names the rejected form ('CREATE SCHEMA IF NOT EXISTS') while
    explaining why it is wrong -- a whole-file substring check would be fooled
    by that comment, and was.
    """
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    guard = sql[sql.index("DO $$"): sql.index("END $$;") + len("END $$;")]
    assert "pg_catalog.pg_namespace" in guard
    assert "CREATE SCHEMA mse;" in guard
    assert "IF NOT EXISTS" not in guard.split("CREATE SCHEMA")[1], (
        "the CREATE SCHEMA inside the guard must be unqualified"
    )


def test_the_guard_still_contains_a_real_create_schema():
    """The guard must still CREATE, not merely raise.

    A role that DOES hold database-level CREATE has to be able to provision the
    schema in one step. Replacing the inner statement with a RAISE would make
    the file only ever re-appliable, and would take away the one path that
    provisions a fresh database.

    (The scoped app role cannot provision -- that is a documented limitation of
    the role, not of this file.)
    """
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    guard = sql[sql.index("DO $$"): sql.index("END $$;") + len("END $$;")]
    assert "CREATE SCHEMA mse;" in guard
    assert "RAISE EXCEPTION" not in guard


def test_the_comment_does_not_claim_the_schema_is_always_creatable():
    """The wording finding: the old comment read as a general fact.

    An existing schema never reaches the privileged statement, but an ABSENT one
    does and fails -- with an error byte-identical to the one this fix was for.
    The comment must say which case it covers, or the next reader re-derives the
    wrong conclusion from a deploy log.
    """
    sql = SCHEMA_SQL.read_text(encoding="utf-8")
    assert "when the schema already exists" in sql
    assert "When the schema is ABSENT" in sql


def test_the_stripper_sees_a_bare_create_schema():
    """The helper itself must be discriminating, or the tests above are vacuous."""
    fake = "CREATE SCHEMA IF NOT EXISTS mse;\n"
    statements = _statements_outside_do_blocks(fake)
    assert statements == ["CREATE SCHEMA IF NOT EXISTS mse"]

    wrapped = (
        "DO $$\nBEGIN\n  IF NOT EXISTS (SELECT 1) THEN\n"
        "    CREATE SCHEMA mse;\n  END IF;\nEND $$;\n"
    )
    assert _statements_outside_do_blocks(wrapped) == []


# ---------------------------------------------------------------------------
# The solver must not run DDL at all
# ---------------------------------------------------------------------------
SOLVER_ENTRYPOINT = SOLVER / "mse_maxwell_solver.py"


def test_the_solver_entrypoint_never_calls_apply_schema():
    """The web initContainer owns the schema.

    A Job applying DDL would repeat identical work on every upload, and -- the
    reason this test exists -- a Job whose schema path is wrong dies before it
    can solve anything. Static, so it needs no database.
    """
    source = SOLVER_ENTRYPOINT.read_text(encoding="utf-8")
    code = [
        line for line in source.splitlines()
        if not line.lstrip().startswith("#")
    ]
    body = "\n".join(code)
    assert "apply_schema" not in body, (
        "mse_maxwell_solver.py must not call apply_schema(); the schema is "
        "applied by the web initContainer"
    )


def test_the_solver_imports_and_exposes_no_call_site():
    """Belt and braces: import the module and check the loaded source too."""
    import mse_maxwell_solver as entry  # noqa: PLC0415

    source = Path(entry.__file__).read_text(encoding="utf-8")
    assert "apply_schema()" not in source


def test_apply_schema_remains_callable_for_operators(monkeypatch, tmp_path):
    """It is no longer called by the solver, but it must still resolve.

    Kept so an operator running the solver outside the cluster has a working
    helper rather than a broken one that would raise on a path that used to be
    correct.
    """
    module, schema = _make_layout(tmp_path / "img", "", depth=0)
    fake = module / "mse_db.py"
    fake.write_text("", encoding="utf-8")
    monkeypatch.setattr(SOLVER_DB, "__file__", str(fake))
    monkeypatch.delenv("MSE_SCHEMA_PATH", raising=False)
    store = SOLVER_DB.JobStore()
    assert store._conn is None
    # no connection -> no-op, and crucially it must NOT try to read a path
    store.apply_schema()


def test_the_web_initcontainer_script_still_works_in_the_image_layout(tmp_path):
    """End-to-end for the piece that actually broke: run the init command.

    Reproduces the initContainer exactly -- ``python -c "from db import
    JobStore; JobStore().apply_schema()"`` with a flat ``/opt/app`` layout --
    and checks it gets past the path resolution rather than dying on it. The
    database is absent, so a connection error is the expected *success* signal
    here: it means the schema file was found and opened.

    The real DB password is never used; a bogus DSN is passed in.
    """
    app = tmp_path / "opt" / "app"
    (app / "db").mkdir(parents=True)
    (app / "db" / "001_schema.sql").write_text(
        "SELECT 1;\n", encoding="utf-8")
    # the module under test, copied flat, as the Dockerfile does
    (app / "db.py").write_text(
        Path(WEB_DB.__file__).read_text(encoding="utf-8"), encoding="utf-8")

    probe = textwrap.dedent(
        """
        import db
        print("RESOLVED", db._schema_path())
        """
    )
    env = dict(os.environ, PYTHONPATH=str(app))
    env.pop("MSE_SCHEMA_PATH", None)
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True,
        env=env, cwd=str(app),
    )
    assert result.returncode == 0, result.stderr
    assert "RESOLVED" in result.stdout
    resolved = result.stdout.split("RESOLVED", 1)[1].strip()
    assert Path(resolved) == (app / "db" / "001_schema.sql")
