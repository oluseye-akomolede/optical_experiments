#!/usr/bin/env python3
"""Prove, on REAL Postgres, that a td row never leaves the Jetson queue (FDTD 4.2).

The cluster runs only the cavity solver, so a td row must stay ``target =
'jetson'`` through every path that can change its routing, and the cluster
queue must never see it. A tier-1 double cannot settle this: the properties
live in SQL WHERE clauses (``next_queued``'s kind whitelist,
``reroute_to_cluster``'s kind guard, the lease paths' untouched ``target``).

Phase 0 is the 4.1 read side: drop ``tds``, re-apply ``001_schema.sql`` (twice),
then read a td row back through ``get_job``, ``list_jobs`` and (phase 2)
``claim_jetson``. Predicted red: removing the schema's ADD COLUMN, a reader
taking tds from the pcs slot, or a list SELECT without tds.

Each phase drives the real ``JobStore`` (and, for the abandon pass and the
reaper, the real ``app`` functions) over one td row:

    create_job(target=cluster)  -> refused (ValueError naming the Jetson)
    next_queued(cluster/jetson) -> None    (the kind whitelist)
    claim_jetson                -> the row, kind 'td'
    release_jetson              -> queued, still target 'jetson'
    lease expiry + reclaim      -> queued, still target 'jetson'
    abandon pass past the wait  -> stays 'jetson'; a cavity CONTROL beside it
                                   moves, so the pass demonstrably can move rows
    reaper over a running row   -> not failed (no k8s Job is its normal state)

Predicted red on the pre-4.2 branch: ``next_queued(JETSON)`` returns the td row
(the filter was ``kind <> 'pc'``) and ``create_job(target=cluster)`` succeeds.

Run against a THROWAWAY Postgres, never production; the connection comes from
the ENVIRONMENT, never argv:

    export DOCKER_CONFIG=~/.docker-mse
    docker rm -f mse-pgcheck; docker run -d --rm --name mse-pgcheck \\
      --cpus=2 --memory=2g -e POSTGRES_PASSWORD=pw -e POSTGRES_DB=msecheck \\
      -p 55432:5432 postgres:16-alpine
    export DATABASE_URL="postgresql://postgres:pw@127.0.0.1:55432/msecheck"
    (ulimit -v 4194304; nice -n 19 .venv-mse/bin/python tests/td_routing_check.py)

TEAR IT DOWN when you are done, ending with ``docker ps -a``:

    docker rm -f mse-pgcheck && docker ps -a
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "tests"))

import db as DB  # noqa: E402
from pc_routing_check import _apply_schema, _dsn_from_env  # noqa: E402

TDS = {"boundary": "open", "source_kind": "point", "profile": "pulse"}


def _row(store, job_id):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT target, status, kind, message FROM mse.jobs "
                    "WHERE job_id = %s", (job_id,))
        return cur.fetchone()


def _set(store, job_id, sql, params=()):
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute(f"UPDATE mse.jobs SET {sql} WHERE job_id = %s",
                    (*params, job_id))


def _new_td(store) -> str:
    job_id = str(uuid.uuid4())
    store.create_job(job_id, "u.obj", "um", 1.0, 0, target=DB.JETSON,
                     est_tets=1_000, kind=DB.TD, tds=dict(TDS))
    return job_id


def _busy(store, job_id) -> dict:
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT target, status, release_count, busy_since, "
                    "busy_until, message FROM mse.jobs WHERE job_id = %s",
                    (job_id,))
        keys = ("target", "status", "release_count", "busy_since",
                "busy_until", "message")
        return dict(zip(keys, cur.fetchone()))


def _has_tds(store) -> bool:
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM information_schema.columns WHERE "
                    "table_schema = 'mse' AND table_name = 'jobs' "
                    "AND column_name = 'tds'")
        return cur.fetchone()[0] == 1


class _NoJobsLauncher:
    """A cluster that answered and has no Jobs: the reaper's hardest case."""
    dry_run = False

    def active_job_names(self):
        return []

    def job_states(self):
        return {}


def main() -> int:
    import app as APP  # noqa: PLC0415 - imports Flask; keep it off the top
    import config as CFG  # noqa: PLC0415

    store = DB.JobStore(_dsn_from_env())
    bad: list[str] = []
    quiet = lambda _m: None  # noqa: E731

    # 0. Upgrade: a database from before 4.1 has no tds column. Applying
    #    001_schema.sql must add it (the spec says idempotently), and a td row
    #    must then read back through every reader with its own request.
    _apply_schema(store)
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mse.jobs")
        cur.execute("ALTER TABLE mse.jobs DROP COLUMN IF EXISTS tds")
        cur.execute("ALTER TABLE mse.jobs DROP COLUMN IF EXISTS busy_since")
        cur.execute("ALTER TABLE mse.jobs DROP COLUMN IF EXISTS busy_until")
    if _has_tds(store):
        bad.append("INSTRUMENT: tds is still there after DROP COLUMN, so the "
                   "upgrade below proves nothing")
    _apply_schema(store)
    _apply_schema(store)  # twice: the ADD COLUMN must be idempotent
    if not _has_tds(store):
        bad.append("001_schema.sql did not add mse.jobs.tds to an older database")
        for line in bad:
            print("FAIL:", line)
        return 1
    read_id = _new_td(store)
    got = store.get_job(read_id)
    if got is None or got["kind"] != DB.TD or got["tds"] != TDS or got["pcs"] is not None:
        bad.append(f"get_job did not return the td request: {got!r}")
    listed = [j for j in store.list_jobs() if j["job_id"] == read_id]
    if len(listed) != 1 or listed[0]["tds"] != TDS or listed[0]["kind"] != DB.TD:
        bad.append(f"list_jobs did not return the td request: {listed!r}")
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mse.jobs")

    # 1. Refused at creation off the Jetson.
    try:
        store.create_job(str(uuid.uuid4()), "u.obj", "um", 1.0, 0,
                         target=DB.CLUSTER, kind=DB.TD, tds=dict(TDS))
        bad.append("create_job accepted a td row with target='cluster'")
    except ValueError as exc:
        if "Jetson" not in str(exc):
            bad.append(f"the refusal does not name the Jetson: {exc}")

    # 2. Invisible to the cluster queue, visible to the Jetson claim.
    job_id = _new_td(store)
    for target in (DB.CLUSTER, DB.JETSON):
        got = store.next_queued(target)
        if got is not None:
            bad.append(f"next_queued({target!r}) returned the {got['kind']!r} row; "
                       "the cluster launcher would fail it")
    claimed = store.claim_jetson("td-check", lease_seconds=60)
    if (claimed is None or claimed["job_id"] != job_id
            or claimed["kind"] != DB.TD or claimed["tds"] != TDS):
        bad.append(f"claim_jetson did not return the td row: {claimed!r}")

    # 3. Release keeps it on the Jetson.
    if store.release_jetson(job_id, "td-check", "test release") != "released":
        bad.append("release_jetson did not release the held td row")
    target, status, _k, _m = _row(store, job_id)
    if (target, status) != (DB.JETSON, DB.QUEUED):
        bad.append(f"after release: target={target!r} status={status!r}")

    # 4. Lease expiry keeps it on the Jetson.
    store.claim_jetson("td-check", lease_seconds=60)
    _set(store, job_id, "lease_expires_at = now() - interval '1 minute'")
    reclaimed = store.reclaim_expired_leases()
    target, status, _k, _m = _row(store, job_id)
    if job_id not in reclaimed or (target, status) != (DB.JETSON, DB.QUEUED):
        bad.append(f"after lease expiry: reclaimed={reclaimed} "
                   f"target={target!r} status={status!r}")

    # 5. The abandon pass past the wait leaves it; a cavity control moves.
    control = str(uuid.uuid4())
    store.create_job(control, "c.obj", "mm", 1.0, 10, target=DB.JETSON,
                     est_tets=1_000)
    old = float(CFG.gpu_wait_seconds()) * 2.0
    for jid in (job_id, control):
        _set(store, jid, "gpu_queued_at = now() - make_interval(secs => %s)",
             (old,))
    if store.reroute_to_cluster(job_id, "direct"):
        bad.append("reroute_to_cluster moved a td row (its SQL kind guard is gone)")
    logs: list[str] = []
    APP._reroute_abandoned_gpu_jobs(store, object(), log=logs.append)
    if _row(store, job_id)[0] != DB.JETSON:
        bad.append("the abandon pass moved the td row off the Jetson")
    if _row(store, control)[0] != DB.CLUSTER:
        bad.append("INSTRUMENT: the cavity control did not move, so the abandon "
                   "pass never ran its move and phase 5 proves nothing")
    if not any("'td' job" in line for line in logs):
        bad.append(f"the abandon pass did not log keeping the td row: {logs}")

    # 6. The reaper leaves a running td row with no k8s Job.
    _set(store, job_id, "status = %s, stage = %s, k8s_job_name = NULL",
         (DB.RUNNING, DB.RUNNING))
    APP._fail_dead_jobs(_NoJobsLauncher(), store, log=quiet)
    target, status, _k, message = _row(store, job_id)
    if status != DB.RUNNING or target != DB.JETSON:
        bad.append(f"the reaper touched the running td row: target={target!r} "
                   f"status={status!r} message={message!r}")

    # 7. Jetson busy (4.2): a deferral requeues without spending a release,
    #    sits out of the claim for BUSY_RETRY_SECONDS, keeps its first
    #    busy_since, and fails only once BUSY_WAIT_MINUTES have passed since it.
    with store._connect() as conn, conn.cursor() as cur:
        cur.execute("DELETE FROM mse.jobs")
    busy_id = _new_td(store)
    if store.claim_jetson("td-check", lease_seconds=60) is None:
        bad.append("INSTRUMENT: the busy row was not claimable to begin with")
    if store.defer_busy(busy_id, "someone-else", "x") is not None:
        bad.append("defer_busy accepted a lease this owner does not hold")
    if store.defer_busy(busy_id, "td-check", "3.1 GB free, 6 GB needed") != "deferred":
        bad.append("defer_busy did not defer the held td row")
    first = _busy(store, busy_id)
    if first["status"] != DB.QUEUED or first["target"] != DB.JETSON:
        bad.append(f"after a deferral: {first!r}")
    if first["release_count"] != 0:
        bad.append(f"a deferral spent a release: release_count={first['release_count']}")
    if first["busy_since"] is None or first["busy_until"] is None:
        bad.append(f"a deferral did not set busy_since/busy_until: {first!r}")
    if "6 GB needed" not in (first["message"] or ""):
        bad.append(f"the waiting message lost the worker's reason: {first['message']!r}")
    if store.claim_jetson("td-check", lease_seconds=60) is not None:
        bad.append("a just-deferred row was claimed straight back (busy_until ignored)")
    _set(store, busy_id, "busy_until = now() - interval '1 second'")
    if store.claim_jetson("td-check", lease_seconds=60) is None:
        bad.append("a deferred row never became claimable again after busy_until")
    _set(store, busy_id, "busy_since = busy_since - interval '5 minutes'")
    kept = _busy(store, busy_id)["busy_since"]
    if store.defer_busy(busy_id, "td-check", "still busy") != "deferred":
        bad.append("a second deferral inside the wait did not defer")
    if _busy(store, busy_id)["busy_since"] != kept:
        bad.append("a second deferral reset busy_since, so the 60-minute wait "
                   "would never end")
    _set(store, busy_id, "busy_until = NULL, busy_since = now() - "
         "make_interval(mins => %s)", (DB.BUSY_WAIT_MINUTES + 1,))
    store.claim_jetson("td-check", lease_seconds=60)
    if store.defer_busy(busy_id, "td-check", "still busy") != "failed":
        bad.append("a deferral past BUSY_WAIT_MINUTES did not fail the job")
    last = _busy(store, busy_id)
    if last["status"] != DB.FAILED or last["message"] != DB.BUSY_GAVE_UP_MESSAGE:
        bad.append(f"after the wait ran out: {last!r}")
    control = str(uuid.uuid4())
    store.create_job(control, "c.obj", "mm", 1.0, 10, target=DB.JETSON,
                     est_tets=1_000)
    store.claim_jetson("td-check", lease_seconds=60)
    if store.defer_busy(control, "td-check", "x") is not None:
        bad.append("defer_busy deferred a cavity row; only td waits for memory")

    for line in bad:
        print("FAIL:", line)
    if not bad:
        print("ok: a td row stays on the Jetson through create, both queues, "
              "release, lease expiry, the abandon pass and the reaper; a busy "
              "deferral is uncounted, held out, and bounded at "
              f"{DB.BUSY_WAIT_MINUTES} minutes")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
