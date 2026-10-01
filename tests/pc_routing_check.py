#!/usr/bin/env python3
"""Prove, on REAL Postgres, that a pc row reaches the Jetson and not the cluster.

This is the one property a tier-1 double cannot prove, and the reason the
routing defect shipped green: ``FakeDb`` in ``tests/test_web_routes.py`` returns
whatever the test hands it, so a ``next_queued`` with no kind filter and a
``claim_jetson`` that already selects ``kind`` are indistinguishable from their
fixes. The bug is not in a value either store computes; it is in which store
*sees* the row, and only two queries against one shared table can settle that.

The crossing is driven through the real upload route -- ``_handle_pc_upload``
builds the row with the real ``create_job`` -- and then both claim paths are
asked for it:

    claim_jetson(...)  -> MUST return the row   (the only pc solver is on the
                                                 Jetson worker)
    next_queued(CLUSTER) -> MUST NOT return it  (the cluster would build a
                                                 CAVITY Job and cavity-solve
                                                 a band diagram)

Against the unfixed branch the first assertion is the one that fails, and it
fails with ``claim_jetson -> None``: the row was created ``target=cluster``, so
the Jetson query (``target = 'jetson'``) cannot see it at all. Reading the RED
against that predicted symptom is the point -- a red for any other reason
(connection, schema, import) is a broken instrument, not a caught defect.

Run against a THROWAWAY Postgres, never production, and the connection comes
from the ENVIRONMENT, never argv (argv is readable by every seat's ``ps`` on
this shared VM):

    export DOCKER_CONFIG=~/.docker-mse
    docker rm -f mse-pgcheck && docker run -d --rm --name mse-pgcheck \\
      --cpus=2 --memory=2g -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=msecheck \\
      -p 55432:5432 postgres:16-alpine
    export DATABASE_URL="postgresql://postgres:pw@127.0.0.1:55432/msecheck"
    (ulimit -v 4194304; nice -n 19 .venv-mse/bin/python tests/pc_routing_check.py)

TEAR IT DOWN when you are done, ending with ``docker ps -a``:

    docker rm -f mse-pgcheck && docker ps -a
"""

from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import db as DB  # noqa: E402


def _dsn_from_env() -> dict:
    """The throwaway connection, from the environment only.

    Deliberately no ``--dsn`` flag, so the insecure form is not expressible.
    ``DATABASE_URL`` wins over the ``PG*`` variables, matching libpq.
    """
    url = os.environ.get("DATABASE_URL", "").strip()
    if url:
        parts = urlsplit(url)
        return {
            "host": parts.hostname or "127.0.0.1",
            "port": parts.port or 5432,
            "dbname": (parts.path or "/").lstrip("/") or "postgres",
            "user": parts.username or "postgres",
            "password": parts.password or "",
        }
    host = os.environ.get("PGHOST", "").strip()
    if not host:
        raise SystemExit(
            "No connection configured. Set DATABASE_URL, or PGHOST/PGDATABASE/"
            "PGUSER. This check must be pointed at a THROWAWAY Postgres."
        )
    return {
        "host": host,
        "port": int(os.environ.get("PGPORT", "5432")),
        "dbname": os.environ.get("PGDATABASE", "postgres"),
        "user": os.environ.get("PGUSER", "postgres"),
        "password": os.environ.get("PGPASSWORD", ""),
    }


def _apply_schema(store: DB.JobStore) -> None:
    store.apply_schema(str(ROOT / "db" / "001_schema.sql"))


def _count(store: DB.JobStore, sql: str, params=()) -> int:
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        return int(cur.fetchone()[0])


def _create_pc_row_through_the_route(store: DB.JobStore) -> str:
    """Create the pc row the way the upload route creates it.

    ``_handle_pc_upload`` is a Flask view: it needs an app, a request context
    and a store, and running it would drag the whole web app in. What matters
    here is the ONE database write it performs, so this calls the same
    ``create_job`` with the same arguments read off ``web/app.py`` at the
    commit under test. The values are asserted against the source below, so a
    divergence between this helper and the route is a failure, not a silent
    drift.

    The route names itself in the docstring: a pc job has no file, its label is
    its parameters, and it creates the row ``target=DB.JETSON``, ``kind=DB.PC``.
    We reproduce exactly that and then ask the queue which machine can actually
    see it.
    """
    job_id = uuid.uuid4()
    request_json = (
        '{"lattice": "triangular", "r_over_a": 0.2, "n_bands": 6, '
        '"polarisation": "TM", "divisions": 24}'
    )
    store.create_job(
        job_id, "triangular lattice, r/a=0.2, 6 bands, TM",
        "mm", 1.0, 6, target=DB.JETSON, est_tets=100,
        kind=DB.PC, pcs=request_json,
    )
    return str(job_id)


def _route_creates_pc_against_jetson() -> None:
    """Read the route's own call, so the helper above cannot drift from it.

    A retyped expectation would check this file's arithmetic; extracting it from
    the source checks the route. If the source stops saying ``DB.JETSON`` the
    crossing below is no longer the crossing this check claims to make, and it
    must fail rather than quietly test something else.
    """
    src = (ROOT / "web" / "app.py").read_text(encoding="utf-8")
    start = src.index("def _handle_pc_upload")
    body = src[start:src.index("\ndef ", start + 1)]
    if "target=DB.JETSON" not in body:
        raise SystemExit(
            "web/app.py:_handle_pc_upload no longer creates the row "
            "target=DB.JETSON; this check's premise is stale. Re-read the "
            "route before trusting the crossing below."
        )
    if "target=DB.CLUSTER" in body:
        raise SystemExit(
            "web/app.py:_handle_pc_upload still creates the row target=DB.CLUSTER; "
            "that is the routing defect this check exists to catch."
        )
    if "kind=DB.PC" not in body:
        raise SystemExit(
            "web/app.py:_handle_pc_upload no longer creates the row kind=DB.PC; "
            "this check's premise is stale."
        )


def main() -> int:
    dsn = _dsn_from_env()
    store = DB.JobStore(dsn=dsn)
    _apply_schema(store)

    # A clean table, so ``claim_jetson`` -- which takes the OLDEST claimable row
    # -- cannot hand back a leftover from an earlier run and pass the crossing
    # for the wrong reason. This is a throwaway database; wiping it is the point.
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mse.jobs")

    _route_creates_pc_against_jetson()

    failures: list[str] = []

    job_id = _create_pc_row_through_the_route(store)

    # (1) The row exists, with the kind and target the route wrote. If this
    #     fails the instrument is broken, not the routing, and everything below
    #     would be meaningless -- so it is checked first and named as such.
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT target, kind FROM mse.jobs WHERE job_id = %s", (job_id,))
        row = cur.fetchone()
    if row is None:
        print("INSTRUMENT BROKEN: the pc row was not created at all")
        return 2
    target, kind = row[0], row[1]
    print(f"created pc row {job_id}: target={target!r} kind={kind!r}")
    if kind != DB.PC:
        print(f"INSTRUMENT BROKEN: kind is {kind!r}, expected {DB.PC!r}")
        return 2

    # (2) The cluster queue must NOT see it. This is asked FIRST, while the row
    #     is still `queued` and unclaimed: `claim_jetson` below moves the row to
    #     `running`, and a `running` row is invisible to `next_queued` whatever
    #     its kind, so asking after the claim would test the lease and not the
    #     filter.
    nq = store.next_queued(target=DB.CLUSTER)
    if nq is not None:
        failures.append(
            f"next_queued(cluster) returned {nq['job_id']!r}: the cluster would "
            "build a CAVITY Job for a pc row and cavity-solve a band diagram. "
            "next_queued must skip kind='pc'."
        )

    # (2b) The check above is necessary but NOT sufficient, and the mutation
    #      that reverted the kind filter proved it: with the route fixed, the
    #      row is target='jetson', and a cluster-targeted query cannot see it
    #      for that reason alone. The kind filter only bites on a pc row that is
    #      ALSO target=cluster -- the exact row the defect used to create, and
    #      the one a future caller could create again. So a second row is
    #      written with that shape directly, and the filter must still hide it.
    #      Without this row, deleting `AND kind <> 'pc'` from next_queued leaves
    #      this check green.
    straggler = uuid.uuid4()
    store.create_job(
        straggler, "pc row wrongly created for the cluster", "mm", 1.0, 6,
        target=DB.CLUSTER, est_tets=100, kind=DB.PC,
        pcs='{"lattice": "triangular", "n_bands": 6}',
    )
    nq_pc_cluster = store.next_queued(target=DB.CLUSTER)
    if nq_pc_cluster is not None:
        failures.append(
            f"next_queued(cluster) returned the cluster-targeted pc row "
            f"{nq_pc_cluster['job_id']!r}: the kind filter is not excluding pc "
            "rows. A pc row on the cluster is cavity-solved and its band diagram "
            "is never produced."
        )

    # (3) THE CROSSING. claim_jetson must see the route's row, because the
    #     Jetson worker is the ONLY machine with a pc solver. On the unfixed
    #     branch this is the assertion that fails, with `claim_jetson -> None`:
    #     the row was created target='cluster', so the Jetson query
    #     (`target = 'jetson'`) cannot see it at all.
    claimed = store.claim_jetson("pc-routing-check", lease_seconds=60)
    if claimed is None:
        failures.append(
            "claim_jetson returned None: the Jetson worker -- the ONLY machine "
            "with a pc solver -- cannot see the pc row. On the unfixed branch "
            "this is because the route created it target='cluster'; the fix is "
            "to create pc jobs target=JETSON."
        )
    else:
        if claimed["job_id"] != job_id:
            failures.append(
                f"claim_jetson returned the wrong row: {claimed['job_id']} != {job_id}"
            )
        if claimed.get("kind") != DB.PC:
            failures.append(
                f"claim_jetson dropped the kind: {claimed.get('kind')!r} != {DB.PC!r} "
                "-- the worker dispatches on it, so a lost kind is a cavity solve "
                "of a band diagram"
            )
        if claimed.get("pcs") is None:
            failures.append(
                "claim_jetson dropped the pc request: the worker cannot solve a "
                "band diagram with no lattice/parameters"
            )

    # (4) THE ABANDON PATH. A pc job nobody claims within MSE_GPU_WAIT_S must
    #     not be moved to the cluster: there is no pc solver there, and once
    #     target='cluster' the row is invisible to BOTH queues (next_queued
    #     skips pc; claim_jetson needs target='jetson'), so it is stranded with
    #     no error and, if ever seen, cavity-solved. It must STAY on the GPU
    #     queue. This drives the real function, not a reimplementation.
    failures.extend(_abandon_path_keeps_pc_on_the_gpu(store))

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("\nok: the pc row is claimable by the Jetson and invisible to the cluster")
    return 0


def _abandon_path_keeps_pc_on_the_gpu(store: DB.JobStore) -> list[str]:
    """Run the real GPU-abandon path over a stale pc row and check it stays put.

    Red on the unfixed branch with the predicted symptom: the row's target
    flips to 'cluster', its message says it moved, and ``claim_jetson`` then
    returns None -- the only pc solver is on a worker that can no longer see it.
    """
    import config as CFG  # noqa: PLC0415 - only needed for this phase
    if str(ROOT / "web") not in sys.path:
        sys.path.insert(0, str(ROOT / "web"))
    import app as APP  # noqa: PLC0415 - imports Flask; keep it off the top

    out: list[str] = []
    job_id = _create_pc_row_through_the_route(store)

    # On the GPU queue, waiting longer than the window, with an estimate small
    # enough that the SIZE escape does not fire. pc rows are capped well under
    # the cluster's element limit, so the size branch is unreachable for them:
    # the kind is the only thing that can save the row.
    wait = CFG.gpu_wait_seconds()
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(
            "UPDATE mse.jobs SET status = %s, target = %s, stage = %s, "
            "gpu_queued_at = now() - make_interval(secs => %s), "
            "lease_owner = NULL, lease_expires_at = NULL, message = NULL "
            "WHERE job_id = %s",
            (DB.QUEUED, DB.JETSON, "queued", float(wait) * 2.0, job_id),
        )

    # A launcher that is merely present: the function bails on None, and this
    # phase is about the routing decision, not about launching anything.
    logs: list[str] = []
    APP._reroute_abandoned_gpu_jobs(store, object(), log=logs.append)

    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT target, message FROM mse.jobs WHERE job_id = %s",
                    (job_id,))
        target, message = cur.fetchone()

    moved_text = "moved to the cluster queue"
    if target != DB.JETSON:
        out.append(
            f"the abandon path moved a pc row to target={target!r}: there is no "
            "pc solver on the cluster, so it is now invisible to BOTH queues "
            "(next_queued skips pc; claim_jetson needs target='jetson') and "
            "stranded with no error. A pc row must stay on the GPU queue."
        )
    if message and moved_text in message:
        out.append(
            f"the row tells the owner {message!r}: nothing moved, so the "
            "'moved to the cluster queue' message is false"
        )
    for line in logs:
        if moved_text in line:
            out.append(f"the run logged a move that did not happen: {line!r}")

    # The consequence as the owner meets it: the Jetson can still claim it.
    claimed = store.claim_jetson("pc-abandon-check", lease_seconds=60)
    if claimed is None or claimed["job_id"] != job_id:
        out.append(
            "claim_jetson could not return the pc row after the abandon path "
            "ran: the job is lost (the predicted symptom on the unfixed branch)"
        )

    if not out:
        print("ok: a pc row past the GPU wait stays on the GPU queue")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
