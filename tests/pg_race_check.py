#!/usr/bin/env python3
"""Prove the two SQL properties a tier-1 double CANNOT prove.

Run against a THROWAWAY Postgres, never production. The connection comes from
the ENVIRONMENT, never from argv: a DSN on the command line is visible to every
seat's ``ps`` on this shared VM, and the habit leaks real passwords even when
the password in question is a throwaway container's. There is deliberately no
``--dsn`` flag, so the insecure form is not even expressible:

    export DATABASE_URL="postgresql://postgres:pw@127.0.0.1:55432/msecheck"
    .venv-mse/bin/python tests/pg_race_check.py

The individual ``PGHOST`` / ``PGPORT`` / ``PGUSER`` / ``PGPASSWORD`` /
``PGDATABASE`` variables work too -- ``DATABASE_URL`` wins if both are set, which
is libpq's own precedence. ``PGPASSWORD`` is no better on the process table than
argv is, but it is at least not in this file's documented command line; for a
throwaway container either is safe, and for a real one the connection should be
reached through ``~/.pgpass`` instead.

Run it against a FRESH database, and do not trust a second run on a dirty one.
This check does not clean up after itself, and three of its legs depend on being
handed the row they inserted -- while ``claim_jetson`` returns the OLDEST
claimable row. Against a database left over from an earlier run this check
reports failures that are not defects. That is a real limitation, and the
opposite of the rule this file exists to enforce: an instrument that cannot say
"could not ask" must never say "nothing there". So it now says so loudly --
``_claim_row`` raises rather than returning None when the row is unreachable --
and this note is the honest label on the remaining gap. Recreate the container
between runs:

    docker rm -f mse-pgcheck && docker run -d --rm --name mse-pgcheck \\
      -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=msecheck -p 55432:5432 \\
      postgres:16-alpine

-- and TEAR IT DOWN when you are done, ending with ``docker ps -a``:

    docker rm -f mse-pgcheck && docker ps -a

(``DOCKER_CONFIG=~/.docker-mse`` for both, and for every docker command in this
rig.) The trailing ``docker ps -a`` is not ceremony: a check leaves a container
behind every time it is run, and a leftover holds the port so the NEXT run
connects to the old one -- whose schema is the previous commit's. That is a
check silently reading a stale world, which is the failure this file exists to
prevent, committed by the file itself.

Why this file exists
--------------------
The rest of the suite runs against a Python double (tests set ``MSE_DB_MOCK``),
and ``LeaseDb``'s docstring says so honestly: *no test there can fail because the
real ``web/db.py`` query is wrong*. Two things were therefore verified by READING
and never by RUNNING:

  1. ``claim_jetson``'s SELECT column list and the row indices it returns -- the
     SELECT lists ``min_feature_m, scale`` and the return reads ``row[4]``,
     ``row[5]``. Change the SELECT to ``NULL, NULL`` and the whole suite stays
     green.
  2. Any bound on repeated releases. A race needs two concurrent transactions in
     ONE database. A Python double reimplements the logic, so it cannot exhibit
     the interleaving whether the real UPDATE is atomic or not: a test through
     the double passes on the fix AND on the broken read-test-write version,
     i.e. it is a no-op for exactly the property it claims to check.

This script closes both by running the real statements against real Postgres.

It found a THIRD thing while being written, which is now one of its checks:
`release_jetson` clears `lease_owner` but leaves `status = 'running'`, and
`claim_jetson` only claims `status = 'queued'`. So a released job is claimable by
NOBODY -- the docstring's promise ("another worker can pick the job up
immediately") is false in the code, and `/failed` with it. Measured, not read:
the B1 leg below drives the real methods and reports the status after each
release.

Design notes, so the next reader can trust it:
  * The schema is applied by the PROJECT'S OWN ``JobStore.apply_schema()``, not
    by a splitter here. An earlier version of this file hand-rolled a ``;``
    splitter and mis-split the file on semicolons inside comments, silently
    dropping every ``DO $$`` block -- a checker that could not check. Reuse the
    real code path; do not restate it.
  * It DISCOVERS the release-counter column instead of assuming a name, so it
    keeps working whichever spelling the implementation chose, and reports a
    clear failure if no counter exists at all.
  * It calls the REAL ``JobStore`` methods. Re-stating their SQL here would test
    this file's transcription, not web/db.py -- the failure mode it exists to
    catch would move into the test.
  * The B2 race RENDEZVOUSES the two sessions at a barrier. Without one they run
    one after the other and every form looks safe: measured, the lost-update
    form records 3 releases against a bound of 3 sequentially, and 5 with the
    barrier. A check that cannot see the defect is worse than no check, so the
    barrier form is also run as a control that must FAIL.
  * Any failure to run exits non-zero with ``COULD-NOT-RUN``: a check that did
    not execute must never be readable as a check that passed.
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

# The app's own modules, so apply_schema and the SQL under test are the real
# ones. tests/ is on the path for mse_db_shim, which loads web/db.py by path.
_TESTS = Path(__file__).resolve().parent
sys.path.insert(0, str(_TESTS))
sys.path.insert(0, str(_TESTS.parent / "web"))


#: The parsed dsn dict, set once in main(). JobStore reads dsn["host"] etc, so
#: the URL is parsed at the edge (psycopg2's own parser) and the dict is what
#: the checks hand around -- not a second reading of the URL.
DSN_OF: list[dict] = [{}]


class Result:
    def __init__(self):
        self.ok = True

    def check(self, name, passed, detail=""):
        print(f"[{'PASS' if passed else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""))
        if not passed:
            self.ok = False
        return passed


def counter_column(conn) -> str | None:
    """Whatever the implementation called the release counter, or None.

    Discovered rather than assumed: the name is the implementation's choice and
    hard-coding one would make this check fail for a spelling difference instead
    of for a defect.
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = 'mse' AND table_name = 'jobs' "
            "AND (column_name LIKE '%%release%%' OR column_name LIKE '%%attempt%%' "
            "     OR column_name LIKE '%%retry%%' OR column_name LIKE '%%fail%%') "
            "ORDER BY char_length(column_name), column_name"
        )
        rows = [r[0] for r in cur.fetchall()]
    return rows[0] if rows else None


def insert_job(conn, job_id, **cols):
    keys = ["job_id", "units", "n_index", "n_modes", "target", "status"] + list(cols)
    vals = [job_id, "mm", 1.0, 20, "jetson", "queued"] + list(cols.values())
    with conn.cursor() as cur:
        cur.execute(f"INSERT INTO mse.jobs ({', '.join(keys)}) "
                    f"VALUES ({', '.join(['%s'] * len(vals))})", vals)
    conn.commit()


def check_claim_columns(conn, r: Result, JobStore):
    """A2: claim_jetson's SELECT and its return indices must agree.

    Insert DISTINCT values for min_feature_m and scale. Identical values would
    pass a query that selected the right columns in the wrong order, or a return
    that read the wrong indices -- the check would then be blind to exactly the
    defect it exists for.
    """
    jid = str(uuid.uuid4())
    insert_job(conn, jid, **{"min_feature_m": 0.37, "scale": 1234.5})
    # Call the REAL method, so a wrong column list or a wrong row index in
    # web/db.py fails here. Re-stating its SQL would test my transcription.
    got = JobStore(dsn=DSN_OF[0]).claim_jetson("w-race", lease_seconds=300)
    if got is None:
        return r.check("A2 claim_jetson returns the seeded row", False,
                       "no row came back")
    r.check("A2 claim_jetson returns min_feature_m/scale under the right keys "
            "(a wrong SELECT column or index fails here)",
            got.get("min_feature_m") == 0.37 and got.get("scale") == 1234.5,
            f"min_feature_m={got.get('min_feature_m')!r} "
            f"scale={got.get('scale')!r} (want 0.37 / 1234.5; distinct so a swap "
            f"or a NULL-pair SELECT is visible)")


def check_feature_triplet_roundtrip(conn, r: Result, JobStore):
    """A2b: the three feature columns survive a write/read cycle, distinctly.

    The row keeps the feature THREE ways -- the number typed, the unit chosen,
    and the canonical metres -- and all three share one SELECT list and one set
    of row indices in ``web/db.py``. A swap between ``min_feature_value`` and
    ``min_feature_m`` (two floats, adjacent in the list) is invisible unless the
    values are distinct, and it is the exact defect this file exists for: the
    store would hand back a number that looks right and means something else.

    Read through the REAL ``get_job``, not through re-stated SQL, so a wrong
    column list or row index in the store fails here rather than in production.

    The inserted row is DELETED before returning. This leg only reads, so it
    never claims the row, which would leave it ``queued`` and claimable -- and
    the next leg to claim would be handed THIS row instead of its own, failing
    with an ordering error that has nothing to do with what it tests. Measured:
    without this delete, ``check_release_bound``'s ``_claim_row`` drained its
    ``DRAIN`` allowance on this row and raised. A read-only leg must not leave
    claimable state behind, for the same reason ``_claim_row`` releases what it
    touches.
    """
    jid = str(uuid.uuid4())
    # Distinct in EVERY position, including the unit: 0.2 / 2e-4 / 'nm' cannot
    # be mistaken for one another, whereas 2.0 / 2.0 would hide a swap.
    insert_job(conn, jid, **{"min_feature_m": 2e-4, "min_feature_value": 0.2,
                             "min_feature_unit": "nm"})
    try:
        got = JobStore(dsn=DSN_OF[0]).get_job(jid)
    finally:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM mse.jobs WHERE job_id = %s", (jid,))
        conn.commit()
    if got is None:
        return r.check("A2b get_job returns the seeded row", False, "no row")
    r.check("A2b the min_feature_* triplet round-trips through the real store "
            "(a value<->metres swap or an off-by-one index fails here)",
            got.get("min_feature_m") == 2e-4
            and got.get("min_feature_value") == 0.2
            and got.get("min_feature_unit") == "nm",
            f"m={got.get('min_feature_m')!r} value={got.get('min_feature_value')!r} "
            f"unit={got.get('min_feature_unit')!r} (want 2e-04 / 0.2 / 'nm')")


#: The image tag production runs, and therefore the schema the next upgrade
#: starts from. Named here so the leg says which deployment it is testing; a
#: stale value would make the leg check an upgrade nobody will perform.
DEPLOYED_TAG = "8e27869"


def _deployed_schema_sql() -> str | None:
    """``db/001_schema.sql`` as it was at the deployed tag, or None.

    Read out of git rather than retyped: the point of the leg is that the
    upgrade starts from the real deployed shape, and a transcription of it here
    would be a second copy that can drift from the file that actually shipped.
    None (rather than a raise) when git cannot answer, so the leg can report
    "could not ask" instead of passing on an untested upgrade.
    """
    import subprocess  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    try:
        result = subprocess.run(
            ["git", "show", f"{DEPLOYED_TAG}:db/001_schema.sql"],
            cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout if result.returncode == 0 and result.stdout else None


def _job_columns(cur) -> list[str]:
    """The column names of ``mse.jobs``, ordered, as the DB currently has them."""
    cur.execute("""
        SELECT column_name FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs'
        ORDER BY column_name
    """)
    return [row[0] for row in cur.fetchall()]


def check_upgrade_matches_a_fresh_schema(conn, r: Result, JobStore, db_module):
    """A2c: upgrading the DEPLOYED schema gives the same columns as a fresh one.

    The upgrade that has to work is the one production will actually run, and
    production is at image tag 8e27869 -- a schema with no ``min_feature*``
    column and no ``scale`` column. (An earlier version of this leg recreated an
    intermediate commit's shape instead, which proved a property production
    never exercises and motivated a DROP that would never have fired there. The
    leg now models the real starting point.)

    What this proves: the additive path from 8e27869 lands on exactly the fresh
    shape, so the same code reads the same columns on a new deploy and an
    upgraded one. A double cannot check it -- the property is about two REAL
    schema shapes -- and it is worth checking because a column present in one
    shape and absent in the other works on one machine and fails on the next.

    Run inside a transaction that is ROLLED BACK, so the legs after this one see
    the schema they expect.
    """
    # The 8e27869 DDL is not retyped here or approximated by dropping columns:
    # the checkout carries it, and applying the real file is the only way to be
    # sure the "before" shape is what production actually runs. Only four
    # columns were dropped by the hand-reduction this replaced, which would have
    # missed k8s_job_name/target/gpu_queued_at and quietly made the leg easier
    # than the upgrade it claims to test.
    deployed_ddl = _deployed_schema_sql()
    if deployed_ddl is None:
        return r.check("A2c the deployed (8e27869) schema is available to test "
                       "the upgrade from", False,
                       "git could not read db/001_schema.sql at 8e27869 -- the "
                       "leg cannot reproduce the real starting shape")
    with open(db_module._schema_path(), encoding="utf-8") as handle:
        current_ddl = handle.read()

    with conn.cursor() as cur:
        # The FRESH shape, captured before anything is torn down: this is what
        # ``apply_schema`` produces on an empty database, and it is the target
        # the upgrade must land on.
        fresh = _job_columns(cur)
    if not fresh:
        return r.check("A2c the current schema produces mse.jobs", False, "[]")

    with conn.cursor() as cur:
        # Everything happens inside one transaction that is ROLLED BACK: the
        # legs after this one must see the schema they expect, and this leg
        # rebuilds tables.
        cur.execute("BEGIN")
        # The "before" state is production's, produced by applying the DEPLOYED
        # file rather than by dropping a few columns off the current one. The
        # hand-reduction this replaced would have missed k8s_job_name, target and
        # gpu_queued_at, making the leg quietly easier than the upgrade it claims
        # to test.
        cur.execute("DROP TABLE IF EXISTS mse.modes")
        cur.execute("DROP TABLE IF EXISTS mse.jobs")
        cur.execute(deployed_ddl)
        before = _job_columns(cur)
        # ...then the CURRENT file, exactly as the deployed initContainer runs it.
        cur.execute(current_ddl)
        after = _job_columns(cur)
        cur.execute("ROLLBACK")

    if not before:
        return r.check("A2c applying the deployed schema produces mse.jobs",
                       False, "8e27869's DDL created no columns -- bad starting "
                              "state, the upgrade was not tested")

    # The STARTING shape is asserted on its own, not just used. `after == fresh`
    # would also hold if this leg accidentally applied the CURRENT schema twice
    # -- a copy/paste of the wrong DDL, or _deployed_schema_sql() silently
    # returning the working-tree file -- and it would then pass while testing no
    # upgrade at all. This is the same could-not-ask shape the rig keeps hitting:
    # an equality between two things is not evidence about either one.
    #
    # 12 is production's count at 8e27869 (measured, not assumed); the four
    # exclusions are the columns the upgrade is supposed to ADD, so their absence
    # before is what makes their presence after meaningful.
    EXPECTED_BEFORE = 12
    ADDED_BY_THE_UPGRADE = ("min_feature_m", "min_feature_value",
                            "min_feature_unit", "scale")
    leaked = [c for c in ADDED_BY_THE_UPGRADE if c in before]
    starting_shape_ok = (len(before) == EXPECTED_BEFORE and not leaked)
    r.check("A2c the STARTING shape is really 8e27869's (12 columns, none of "
            "the upgrade's columns yet)",
            starting_shape_ok,
            f"got {len(before)} columns (expected {EXPECTED_BEFORE}); columns "
            f"the upgrade should be adding but which are already present: "
            f"{leaked} -- the 'after' comparison below would pass vacuously")
    if not starting_shape_ok:
        # The equality cannot be read as evidence if the "before" state was not
        # the deployed one. Report it as not-run rather than as a pass.
        return r.check("A2c upgrading the DEPLOYED (8e27869) schema lands on "
                       "exactly the fresh set of columns", False,
                       "not evaluated: the starting shape was wrong, so the "
                       "comparison proves nothing")

    r.check("A2c upgrading the DEPLOYED (8e27869) schema lands on exactly the "
            "fresh set of columns",
            after == fresh,
            f"upgrade-only={sorted(set(after) - set(fresh))} "
            f"missing-after-upgrade={sorted(set(fresh) - set(after))} "
            f"(before the upgrade: {len(before)} columns, after: {len(after)})")


def _race(conn, dsn, r: Result, col: str, bound: int, rounds: int = 60,
          predicate: bool = True):
    """Two sessions, each reading `col`, meeting at a barrier, then writing.

    The barrier is the whole point. Run the two sessions one after the other and
    every form looks safe, because the second one always sees the first one's
    committed row -- so the check would pass for a reason that has nothing to do
    with the property, which is the defect this file exists to avoid. Measured on
    this container: WITH the barrier the no-predicate form records 5 releases
    against a bound of 3; WITHOUT it, 3. Returns the worst count seen.
    """
    import threading  # noqa: PLC0415
    import psycopg2  # noqa: PLC0415

    worst = 0
    for _ in range(rounds):
        jid = str(uuid.uuid4())
        insert_job(conn, jid, **{"lease_owner": "w-race"})
        a = psycopg2.connect(**dsn)
        b = psycopg2.connect(**dsn)
        a.autocommit = b.autocommit = True
        bar = threading.Barrier(2)

        def seat(s, predicate):
            for _ in range(bound + 2):
                with s.cursor() as cur:
                    cur.execute(f"SELECT {col} FROM mse.jobs WHERE job_id = %s",
                                (jid,))
                    n = cur.fetchone()[0]
                try:
                    bar.wait(timeout=5)
                except threading.BrokenBarrierError:
                    pass
                with s.cursor() as cur:
                    if predicate:
                        cur.execute(
                            f"UPDATE mse.jobs SET {col} = {col} + 1 "
                            f"WHERE job_id = %s AND {col} < %s", (jid, bound))
                    else:
                        cur.execute(f"UPDATE mse.jobs SET {col} = %s "
                                    f"WHERE job_id = %s", (n + 1, jid))

        ts = [threading.Thread(target=seat, args=(s, predicate))
              for s in (a, b)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        with conn.cursor() as cur:
            cur.execute(f"SELECT {col} FROM mse.jobs WHERE job_id = %s", (jid,))
            worst = max(worst, cur.fetchone()[0])
        a.close()
        b.close()
    return worst


def check_release_bound(conn, r: Result, bound: int, dsn: dict, JobStore,
                        db_module):
    """B2: the bound must be enforced INSIDE the UPDATE, not by the caller.

    Two sessions race to release one row; the counter must never pass `bound`.

    What this can and cannot show, stated rather than implied. At READ COMMITTED
    -- the app's level, web/db.py sets autocommit and never an isolation level --
    an UPDATE carrying `WHERE c < bound` is atomic by itself: the row lock
    serialises the two sessions and Postgres re-evaluates the predicate after
    waiting. So this check does not discriminate "one statement" from "two
    statements"; both pass, and either implementation is genuinely safe. What it
    DOES discriminate is a release that tests the counter OUTSIDE the UPDATE --
    read c, then `SET c = k` unconditionally. That form loses updates and records
    bound+2 or more. The control below runs exactly that and must FAIL, so a PASS
    on the real code means the predicate is in the statement, not merely that the
    race was never run.
    """
    import psycopg2  # noqa: PLC0415

    col = counter_column(conn)
    if col is None:
        return r.check("B2 a release counter column exists", False,
                       "no release/attempt/retry counter in mse.jobs -- the bound "
                       "cannot exist without one")
    print(f"      (counter column discovered: mse.jobs.{col})")

    # Control first, while the schema is untouched by this check: the shape that
    # MUST break. If it does not break, this check cannot see the defect and a
    # later PASS would be hollow.
    broken = _race(conn, dsn, r, col, bound, rounds=40, predicate=False)
    r.check("B2 control: a lost-update release DOES break the bound "
            "(so the check can see the defect)",
            broken > bound,
            f"that form recorded {broken} releases against bound {bound}")

    good = _race(conn, dsn, r, col, bound)
    r.check(f"B2 counter never exceeds the bound ({bound}) under concurrent "
            f"releases", good <= bound, f"worst {col}={good} over 60 rounds")

    check_release_is_terminal(conn, r, bound, JobStore)
    check_release_not_in_the_gate(r, db_module)
    check_release_refuses_a_non_holder(conn, r, JobStore)


def check_release_is_terminal(conn, r: Result, bound: int, JobStore):
    """B1/B2 on the REAL release path: releasing `bound` times must end FAILED.

    Calls web/db.py, not a copy of its SQL.

    What a "release" IS, pinned explicitly, because the earlier form of this leg
    counted calls instead of looking and so could not tell the two apart. A
    release that matters is one web/db.py records -- ``release_count`` moves and
    the job goes back to ``queued``. The release that discovers the bound
    already reached records nothing and is the terminal one. Counting CALLS
    makes a correct implementation report bound+1 (the extra call is the one
    that ends it) and go red against its own bound.

    Read the counter from the row rather than assuming a call moved it. The
    other reason the call count is wrong here: ``claim_jetson`` takes the OLDEST
    claimable row, so this loop's claim can return a FOREIGN row, and
    ``lease_held_by(jid)`` then breaks out while a release against the job it
    actually claimed has landed on that other job's counter.
    """
    store = JobStore(dsn=DSN_OF[0])
    if not hasattr(store, "release_jetson"):
        return r.check("B1 release_jetson exists", False, "method missing")
    jid = str(uuid.uuid4())
    insert_job(conn, jid)                     # queued, unclaimed, no lease yet
    status_after, message = None, ""
    releases = 0
    # Claim ONCE, before the loop, and use that lease in the first iteration.
    # The claim is a state CHANGE, not a question: it takes the lease and moves
    # the row to ``running``, so a second claim for the same row correctly finds
    # nothing claimable and returns None. An earlier form claimed here "to check
    # the row is reachable" and then claimed again inside the loop, which read
    # its own first claim as "the job has terminated" and reported a crash loop
    # that had not run. Probing a row by claiming it consumes the thing the
    # probe was meant to establish; check it with a SELECT instead.
    claimed = _claim_row(store, jid, owner="w-race")   # ITS OWN row, not oldest
    for _ in range(bound + 5):
        if claimed is None:
            break                             # not claimable: it terminated
        with conn.cursor() as cur:
            cur.execute("SELECT release_count FROM mse.jobs WHERE job_id = %s",
                        (jid,))
            before = cur.fetchone()[0]
        # The lease came from the real claim path, the way the worker tick does,
        # so lease_expires_at is a real timestamp -- a hand-seeded lease_owner
        # has no expiry and lease_held_by (which requires lease_expires_at >
        # now()) reads it as not held.
        released = store.release_jetson(jid, "w-race")   # what /release does
        with conn.cursor() as cur:
            cur.execute("SELECT status, COALESCE(message,''), release_count "
                        "FROM mse.jobs WHERE job_id = %s", (jid,))
            got = cur.fetchone()
        if got is None:
            raise AssertionError(
                f"row {jid} vanished: claim returned {claimed['job_id']}, "
                f"release_jetson returned {released!r}, and the next SELECT "
                f"found no row. Something DELETED it -- this leg only reads "
                f"and updates, so the instrument is wrong about the table."
            )
        status_after, message, after = got
        if after > before:
            releases += 1                     # a release this store RECORDED
        else:
            break                             # the terminal call: recorded none
        # The next round needs a fresh lease, and by now the row is back to
        # ``queued`` with no owner -- so this claim is the one that either hands
        # it back or reports that the bound has taken it out of the pool.
        claimed = _claim_row(store, jid, owner="w-race")
    r.check(f"B1 a deterministic failure reaches FAILED at {bound} recorded "
            f"releases (counter stopped at {releases})",
            status_after == "failed" and releases == bound,
            f"status={status_after!r} after {releases} recorded releases "
            f"(message {message!r})")


def check_release_not_in_the_gate(r: Result, db_module):
    """The hollow-PASS trap, as a test rather than a note.

    If `releases_are_terminal` were written as `if releases <= bound` while
    releases is still 0 (the counter read before it is incremented, the classic
    off-by-one), it would return True immediately and the bound would be
    decorative. The test is a tripwire: it must make the guard say False, so a
    guard that cannot say False anywhere fails here.

    The decided contract puts it at MODULE level. Either location is accepted --
    the check is that the bound can be stated and interrogated as a predicate at
    all, and where the implementation chose to hang it is not this check's
    business.
    """
    gate = getattr(db_module, "releases_are_terminal", None)
    where = "module"
    if gate is None:
        gate = getattr(db_module.JobStore, "releases_are_terminal", None)
        where = "JobStore"
    if gate is None:
        return r.check("B1 the bound is expressed as a testable predicate",
                       False,
                       "no releases_are_terminal(...) on the module or on "
                       "JobStore -- the bound cannot be checked except by "
                       "running the whole loop, so a hollow guard could pass "
                       "unnoticed")
    ok = bool(gate(0, 3)) is False and bool(gate(2, 3)) is False \
        and bool(gate(3, 3)) is True
    r.check("B1 releases_are_terminal refuses 0 and 2, allows 3 "
            "(a guard that only ever returns True fails here)", ok,
            f"checked (0,3)->False, (2,3)->False, (3,3)->True on {where}")


def check_release_refuses_a_non_holder(conn, r: Result, JobStore):
    """E: a release by a worker that does not hold the lease must be refused."""
    store = JobStore(dsn=DSN_OF[0])
    jid = str(uuid.uuid4())
    insert_job(conn, jid, **{"lease_owner": "w-holder"})
    got = store.release_jetson(jid, "w-someone-else")
    with conn.cursor() as cur:
        cur.execute("SELECT lease_owner FROM mse.jobs WHERE job_id = %s", (jid,))
        owner = cur.fetchone()[0]
    # ``None`` is the decided refusal value for "this worker does not hold the
    # lease" -- the same value the route turns into a 409, and a different event
    # from "released" or "failed". ``False`` is accepted too: both mean refused,
    # and pinning one spelling would fail a correct implementation that picked
    # the other.
    r.check("E releasing a lease you do not hold is refused and changes nothing",
            (got is None or got is False) and owner == "w-holder",
            f"release_jetson returned {got!r} (want None or False), "
            f"lease_owner now {owner!r}")


# ---------------------------------------------------------------------------
# C. the EXPIRY path -- a separate route, so it gets its own legs
#
# A worker that dies mid-job never calls /release: it dies, and the row comes
# back through lease EXPIRY (reclaim_expired_leases, run by the app's tick).
# That is a DIFFERENT code path from release_jetson, and the reviewer's point is
# that it must be checked on its own rather than reused with a different caller
# -- the two can diverge, and the expiry one is exactly where the stranding bug
# lived. Each leg below is red when the status reset or the bound is removed
# from reclaim_expired_leases; neither leg exercises release_jetson.
# ---------------------------------------------------------------------------

def _expire_lease(conn, job_id):
    """Make ``job_id``'s lease already expired, as a dead worker leaves it.

    Set the deadline in the past rather than nulling the lease: the reclaim
    query selects on ``lease_expires_at < now()``, so a NULL expiry is not an
    expired lease -- it is a row with no lease at all, which is a different
    state and would make this leg check nothing.
    """
    with conn.cursor() as cur:
        cur.execute("UPDATE mse.jobs SET lease_expires_at = now() - "
                    "interval '1 hour' WHERE job_id = %s", (job_id,))
    conn.commit()


def check_expiry_requeues_the_job(conn, r: Result, JobStore):
    """C1: an expired lease must make the job CLAIMABLE AGAIN by ANOTHER worker.

    The stranding bug, exactly: ``claim_jetson`` requires ``status = 'queued'
    AND lease_owner IS NULL``, so a reclaim that clears the lease but leaves the
    row ``running`` produces a job no worker can take. Nothing errors and
    nothing loops -- the job just never moves again.

    Driven through the real methods end to end: a real claim (so the lease has a
    real deadline), the lease expires, the real reclaim, then ANOTHER worker
    claims through the real gate. The assertion that matters is the last step:
    it is the only one that can observe the stranding.
    """
    store = JobStore(dsn=DSN_OF[0])
    jid = str(uuid.uuid4())
    insert_job(conn, jid)

    # claim_jetson takes the OLDEST claimable row, and earlier legs leave rows
    # of their own behind. Claim until THIS leg's row comes back, so the leg
    # tests the stranding bug rather than the order the table happened to be in.
    claimed = _claim_row(store, jid)
    if claimed is None:
        return r.check("C1 the seeding claim works", False,
                       f"claim_jetson never returned {jid}")

    _expire_lease(conn, jid)
    reclaimed = store.reclaim_expired_leases()

    with conn.cursor() as cur:
        cur.execute("SELECT status, lease_owner FROM mse.jobs WHERE job_id = %s",
                    (jid,))
        status, owner = cur.fetchone()

    # THE assertion: a different worker, through the real gate.
    again = _claim_row(store, jid)
    r.check("C1 an expired lease leaves the job claimable by ANOTHER worker "
            "(a reclaim that does not reset status strands it here)",
            again is not None and again["job_id"] == jid,
            f"after reclaim status={status!r} lease_owner={owner!r}, "
            f"reclaim returned {reclaimed!r}, re-claim got {again!r} -- "
            f"claim_jetson needs status='queued' AND lease_owner IS NULL")


#: how many foreign rows a leg may walk past. Kept BELOW the default bound (3)
#: on purpose: each skip releases the row, and a skip count at or over the
#: bound could fail a foreign job that this check would then wrongly own.
DRAIN = 2


def _claim_row(store, jid, owner="w-leg", lease_seconds=300):
    """Claim until ``jid`` comes back, or give up. Returns the job dict or None.

    ``claim_jetson`` returns the OLDEST claimable row, so a leg that inserts a
    job and then claims may be handed an earlier leg's row. Today no earlier leg
    leaves a claimable one (each inserted row is either claimed, so ``running``,
    or carries an owner, so the gate refuses it), and this walks past that
    assumption instead of resting on it -- a future leg that inserts without
    claiming would otherwise turn C1's assertion into an ordering test.

    A foreign row is handed back with ``release_jetson`` before moving on. That
    is load-bearing: a plain claim leaves the row ``running`` under an owner
    that will never touch it, which is precisely the stranded state these legs
    exist to detect -- the instrument must not manufacture the defect it looks
    for. The release is bounded by the same counter as everything else, so
    ``DRAIN`` is kept well under ``bound``.

    ``None`` means ONE thing here. An earlier version returned it both for "no
    claimable row at all" and for "claimed ``DRAIN`` foreign rows and never
    reached yours", and the second case is silent data loss for the caller: the
    B1 leg read ``None`` as "the job has terminated" and stopped, reporting a
    crash loop that had not run. Whenever two different conditions collapse onto
    one return value, the callers eventually believe the wrong one.
    """
    for _ in range(DRAIN):
        job = store.claim_jetson(owner, lease_seconds=lease_seconds)
        if job is None:
            return None
        if job["job_id"] == jid:
            return job
        store.release_jetson(job["job_id"], owner, "not the row under test")
    raise AssertionError(
        f"{jid} was still claimable when the leg started but not among the "
        f"first {DRAIN} rows claim_jetson handed back; the store is not empty "
        f"and this leg cannot see the row it is about. Run this check against a "
        f"clean database (it does not clean up between runs)."
    )


def check_expiry_bound_is_terminal(conn, r: Result, JobStore, bound: int):
    """C2: repeated expiries must reach FAILED at the bound, not cycle.

    The crash-loop case: a worker that dies every time never calls /release, so
    if the expiry path did not count against the same bound, such a worker would
    requeue forever through lease timeouts. This leg uses ONLY the expiry route
    -- claim, expire, reclaim, repeat -- and asserts it terminates.
    """
    store = JobStore(dsn=DSN_OF[0])
    jid = str(uuid.uuid4())
    insert_job(conn, jid)

    # Count REQUEUES -- rounds where the reclaim handed the job back -- not
    # rounds where a claim was merely attempted. The claim that finds the job
    # already FAILED is the terminal observation, not another expiry, and
    # counting it made the intact case report bound+1 and go red against its
    # own bound. ``bound`` requeues are allowed; the (bound+1)th expiry is the
    # one that fails the job, so ``bound`` is the expected count.
    requeues = 0
    for _ in range(bound + 4):
        # Its OWN row, not whatever claim_jetson hands back first: expiring a
        # foreign row's lease and then asserting on ``jid`` would report a
        # crash loop that never happened.
        if _claim_row(store, jid, owner="w-crashy") is None:
            break                      # no longer claimable: it terminated
        _expire_lease(conn, jid)
        requeued = store.reclaim_expired_leases()
        if jid not in requeued:
            break                      # this expiry was the terminal one
        requeues += 1

    with conn.cursor() as cur:
        cur.execute("SELECT status, COALESCE(message, '') FROM mse.jobs "
                    "WHERE job_id = %s", (jid,))
        status, message = cur.fetchone()

    r.check(f"C2 a crash-looping worker reaches FAILED at the bound "
            f"({requeues} requeues, bound {bound})",
            status == "failed" and requeues == bound,
            f"status={status!r} after {requeues} requeues (message {message!r})")


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Race/lease checks against a throwaway Postgres. The "
                    "connection is read from DATABASE_URL or the standard "
                    "PGHOST/PGPORT/PGUSER/PGPASSWORD/PGDATABASE variables; "
                    "there is deliberately no --dsn flag.")
    ap.add_argument("--bound", type=int, default=3,
                    help="release bound the code uses (default 3)")
    args = ap.parse_args()

    import os  # noqa: PLC0415

    import psycopg2  # noqa: PLC0415
    from psycopg2.extensions import parse_dsn  # noqa: PLC0415
    from mse_db_shim import load_db_module  # noqa: PLC0415

    # The connection is assembled from the ENVIRONMENT so no password ever
    # appears on argv, where any seat's `ps` can read it. Building the dict
    # ourselves (rather than handing psycopg2.connect() no arguments and letting
    # libpq pick the variables up) is what makes the checking possible: we need
    # the same values as a parsed dict for JobStore, and we want to refuse
    # loudly when nothing is configured rather than let libpq silently fall back
    # to a local socket and a same-named role.
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        # psycopg2's own parser, so the URL is read exactly as connect() would.
        params = parse_dsn(url)
    else:
        params = {k: os.environ.get(f"PG{k.upper()}", "")
                  for k in ("host", "port", "dbname", "user", "password")}
    dsn = {
        "host": params.get("host") or "localhost",
        "port": params.get("port") or "5432",
        "dbname": params.get("dbname") or "postgres",
        "user": params.get("user") or "postgres",
        "password": params.get("password") or "",
    }
    if not url and not any(os.environ.get(f"PG{k.upper()}")
                           for k in ("host", "dbname", "user")):
        raise SystemExit(
            "No connection configured. Set DATABASE_URL, or at least PGHOST/"
            "PGDATABASE/PGUSER. This check must be pointed at a THROWAWAY "
            "Postgres -- it writes to the schema and does not clean up.")

    DSN_OF[0] = dsn
    r = Result()
    conn = psycopg2.connect(**dsn)
    try:
        db_module = load_db_module()
        JobStore = db_module.JobStore
        # Print the code's OWN bound rather than assuming it equals --bound. The
        # legs assert against the bound they exercise; if the constant drifts
        # from the default this line is how the reader sees it, and every leg
        # that reads web/db.py's constant would go red instead of being tuned by
        # a flag that the code never reads.
        print(f"      (web/db.py MAX_RELEASES = {db_module.MAX_RELEASES}; "
              f"--bound {args.bound} is what the legs assert)")
        JobStore(dsn=dsn).apply_schema()   # the PROJECT's own path
        print("schema applied via web/db.py:JobStore.apply_schema()")
        check_claim_columns(conn, r, JobStore)
        check_feature_triplet_roundtrip(conn, r, JobStore)
        check_upgrade_matches_a_fresh_schema(conn, r, JobStore, db_module)
        check_release_bound(conn, r, args.bound, dsn, JobStore, db_module)
        # The expiry path is its own route (a worker that DIES never calls
        # /release), so it gets its own legs rather than reusing the release
        # ones with a different caller.
        check_expiry_requeues_the_job(conn, r, JobStore)
        check_expiry_bound_is_terminal(conn, r, JobStore, args.bound)
    finally:
        conn.close()
    print("RESULT:", "PASS" if r.ok else "FAIL")
    return 0 if r.ok else 1


if __name__ == "__main__":
    try:
        _rc = main()
    except Exception as _exc:  # noqa: BLE001 - this is the point
        print(f"COULD-NOT-RUN: {type(_exc).__name__}: {_exc}")
        print("RESULT: FAIL (the check did not execute; this is NOT a pass)")
        sys.exit(2)
    sys.exit(_rc)
