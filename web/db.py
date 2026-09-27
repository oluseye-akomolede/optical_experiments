"""Postgres access for the web app: the ``mse`` schema and the upload queue.

A deliberate near-twin of ``dockerfiles/maxwell_solver/mse_db.py``: the two
components ship as separate images, so the app does not import the solver
package. The duplicated part is a dozen lines of connection handling; the SQL is
kept identical by hand and both sides are parameterised -- no value is ever
interpolated into a statement.

Uploads that are waiting to run live here: an upload row is created with status
``queued`` and the background tick promotes the oldest one when no solver Job is
running.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone

#: job states. `queued` is ours (waiting for the single Job slot); the rest are
#: written by the solver as it moves through its stages.
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"

#: terminal states
FINISHED = (DONE, FAILED)

#: which machine a job is for. ``cluster`` is a k8s Job (the pre-routing default,
#: and what every existing row did); ``jetson`` waits for a GPU worker to claim
#: it through the worker API.
CLUSTER = "cluster"
JETSON = "jetson"
TARGETS = (CLUSTER, JETSON)

#: How many times a jetson job may be handed back with no verdict before it is
#: failed for good. Releases are legitimate for transient reasons; an unbounded
#: number of them is an infinite loop, so there is a cap. Three is enough for a
#: genuine restart or a flapping network to clear without a human, and small
#: enough that a genuinely broken job reaches a terminal state quickly instead
#: of cycling through the GPU queue forever.
MAX_RELEASES = 3


def releases_are_terminal(count: int, bound: int = MAX_RELEASES) -> bool:
    """Whether ``count`` releases have used up the allowance and the job is done.

    The bound as a PREDICATE rather than a comparison repeated at each of the
    three sites that enforce it (``release_jetson``, ``reclaim_expired_leases``
    and the SQL's own ``WHERE ... < %s``). It exists so the bound can be checked
    without driving a whole claim/release loop: a check that can only reach the
    bound by running the loop cannot tell a working bound from one that is
    decorative, because a guard that always returns False never fires and a test
    that never observes it firing still passes.

    ``count`` is the number of releases ALREADY recorded, so ``MAX_RELEASES``
    releases are permitted and the next one is over: ``releases_are_terminal(3,
    3)`` is True, ``(2, 3)`` is False. This is the same comparison the SQL makes
    -- ``release_count < bound`` allows the write, so a row AT the bound takes
    the failure branch.
    """
    return int(count) >= int(bound)


def dsn_from_env() -> dict:
    """Connection parameters from the environment (never from a file)."""
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432"),
        "dbname": os.environ.get("DB_NAME", "mode_search_engine"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
    }


#: filename of the DDL script, and where it lives relative to this module.
SCHEMA_FILENAME = "001_schema.sql"


def schema_candidates() -> list[str]:
    """Every path the DDL script may live at, most specific first.

    The script sits next to this module's directory in **both** layouts, but at
    a different depth, and climbing a fixed number of levels is what broke the
    deployed image:

    * image: ``web/`` is copied to ``/opt/app/`` and ``db/`` to ``/opt/app/db/``,
      so the script is ``<module dir>/db/001_schema.sql``.
    * repo: this file is ``web/db.py`` and the script is ``repo/db/``, so it is
      ``<module dir>/../db/001_schema.sql``.

    ``MSE_SCHEMA_PATH`` wins if set, so an operator can point at a mounted copy
    without rebuilding. The caller reports every candidate it tried, because a
    bare FileNotFoundError naming one wrong path is what made this hard to see.
    """
    override = os.environ.get("MSE_SCHEMA_PATH")
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(here, "db", SCHEMA_FILENAME),              # image layout
        os.path.join(os.path.dirname(here), "db", SCHEMA_FILENAME),  # repo layout
    ]
    if override:
        candidates.insert(0, override)
    return candidates


def _schema_path() -> str:
    """The first candidate that is a real file. Raises if none are.

    ``isfile``, not ``exists``: a directory passes ``exists`` and then fails at
    ``open`` with an unexplained IsADirectoryError, which defeats the point of
    reporting every candidate tried.
    """
    candidates = schema_candidates()
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "could not find " + SCHEMA_FILENAME + "; looked in: "
        + ", ".join(candidates)
    )


class JobStore:
    """Thin wrapper around ``mse.jobs`` / ``mse.modes``.

    Creates a fresh connection per operation. The app is one replica serving one
    owner, and the jobs are minutes apart, so a connection pool would be more
    moving parts than the traffic justifies -- and it keeps the background tick
    from holding a connection open while Flask is idle.
    """

    def __init__(self, dsn: dict | None = None):
        self.dsn = dsn or dsn_from_env()

    @contextmanager
    def _connect(self):
        import psycopg2

        conn = psycopg2.connect(
            host=self.dsn["host"], port=self.dsn["port"],
            dbname=self.dsn["dbname"], user=self.dsn["user"],
            password=self.dsn["password"],
        )
        conn.autocommit = True
        try:
            yield conn
        finally:
            conn.close()

    # -- schema ------------------------------------------------------------
    def apply_schema(self, path: str | None = None) -> None:
        """Apply ``db/001_schema.sql`` idempotently (runs in the initContainer)."""
        with open(path or _schema_path(), encoding="utf-8") as fh:
            sql = fh.read()
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(sql)

    # -- writes ------------------------------------------------------------
    def create_job(self, job_id, original_filename: str, units: str,
                   n_index: float, n_modes: int, target: str = CLUSTER,
                   est_tets: int | None = None,
                   min_feature_mm: float | None = None,
                   scale: float = 1.0) -> None:
        """Create the upload row. Status starts ``queued``, not ``running``.

        ``target`` is the machine the upload-time estimate chose. It is recorded
        rather than recomputed later because the estimate is the *decision*, and
        a wrong route is diagnosed by comparing it against the count the solver
        reports. A job queued for the Jetson also gets ``gpu_queued_at``, which
        is what the fallback timer counts from.

        ``min_feature_mm`` and ``scale`` are recorded because they are *inputs*
        to the mesh, not derivable from what is stored: the same .obj at scale
        1000 and units mm is a different object, and a job whose frequency looks
        wrong by 1000x is diagnosed by reading this row, not by re-reading the
        upload.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO mse.jobs "
                "(job_id, original_filename, units, n_index, n_modes, status, stage, "
                " target, est_tets, min_feature_mm, scale, gpu_queued_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, "
                "        CASE WHEN %s = 'jetson' THEN now() END)",
                (str(job_id), original_filename, units, float(n_index),
                 int(n_modes), QUEUED, QUEUED, target,
                 None if est_tets is None else int(est_tets),
                 None if min_feature_mm is None else float(min_feature_mm),
                 float(1.0 if scale is None else scale),
                 target),
            )

    def mark_failed(self, job_id, message: str) -> None:
        """Record an upload that never reached the solver."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s, message = %s, "
                "finished_at = now() WHERE job_id = %s",
                (FAILED, FAILED, message, str(job_id)),
            )

    def mark_running(self, job_id, k8s_job_name: str) -> None:
        """Claim a queued job for the Job we just created."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s, k8s_job_name = %s "
                "WHERE job_id = %s",
                (RUNNING, "meshing", k8s_job_name, str(job_id)),
            )

    def reroute_to_cluster(self, job_id, reason: str) -> bool:
        """Move a jetson job back to the cluster queue. Returns whether it moved.

        This is the fallback for a GPU that never came: the job's estimate put it
        on the jetson queue, no worker claimed it inside ``MSE_GPU_WAIT_S`` (or a
        worker claimed it and died), and the cluster can still solve it. It is
        the same job, so nothing is re-uploaded -- only ``target`` changes and
        the queue position resets, which is honest: it is genuinely queued again.

        Guarded on ``target = 'jetson'`` so two ticks racing cannot reroute (and
        re-log) the same job twice, and so a job a worker is *actively* holding
        cannot be yanked out from under it. Returns False when nothing moved,
        which is not an error: it means someone else got there first.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET target = %s, stage = %s, message = %s, "
                "lease_owner = NULL, lease_expires_at = NULL, gpu_queued_at = NULL "
                "WHERE job_id = %s AND target = %s AND status = %s",
                (CLUSTER, QUEUED, reason, str(job_id), JETSON, QUEUED),
            )
            return cur.rowcount > 0

    def reclaim_expired_leases(self) -> list[str]:
        """Expire leases past their deadline. Returns the job ids reclaimed.

        An expired lease means the worker died: it renews on every progress
        call, so a lease that ran out with no renewal has no live owner. The
        lease is cleared AND the row is put back to ``queued``, which is what
        makes it claimable again -- another worker takes it, and if none does
        the ``MSE_GPU_WAIT_S`` fallback reroutes it to the cluster.

        The status reset is load-bearing and was missing. ``claim_jetson``
        requires ``status = 'queued' AND lease_owner IS NULL``, so clearing the
        lease alone leaves the row ``running`` and NO worker can claim it. The
        job is then stranded rather than looping: nothing errors, nothing
        retries, it simply never moves again. A comment here used to call
        leaving it ``running`` deliberate ("the job genuinely is mid-flight");
        that was wrong, because the row is mid-flight only while somebody holds
        it, and an expired lease means nobody does.

        THIS COUNTS AS A RELEASE, and that is the whole reason the count is not
        confined to the ``/release`` route. A worker that crashes mid-job every
        time never calls ``/release`` -- it dies -- so if only that route
        incremented the counter, a crash-looping worker would requeue through
        lease timeouts forever and never reach the bound. Past the bound the job
        is failed here instead, which is the only way a crash loop terminates.

        One statement per job, with the bound in the WHERE, for the same reason
        as ``release_jetson``: a racer that checks the count first can requeue a
        job that should have failed.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, release_count FROM mse.jobs "
                "WHERE lease_owner IS NOT NULL AND lease_expires_at < now()",
            )
            rows = [(str(r[0]), int(r[1])) for r in cur.fetchall()]
            over = []
            if rows:
                # Claimable again, and counted -- but only while under the bound.
                #
                # ``status`` goes back to ``queued``, and that is NOT cosmetic:
                # ``claim_jetson`` requires ``status = 'queued' AND lease_owner
                # IS NULL``. Clearing the lease alone leaves the row ``running``,
                # which NO worker can claim -- the job is stranded, not looping.
                # That was a real defect here (the reviewer measured it on
                # Postgres), and it is the failure mode with no symptom at all:
                # no error, no loop, just a job that never moves again.
                cur.execute(
                    "UPDATE mse.jobs SET lease_owner = NULL, "
                    "lease_expires_at = NULL, status = %s, stage = %s, "
                    "gpu_queued_at = now(), "
                    "release_count = release_count + 1 "
                    "WHERE lease_owner IS NOT NULL AND lease_expires_at < now() "
                    "AND release_count < %s",
                    (QUEUED, "queued", MAX_RELEASES),
                )
                # Past the bound: terminal, and named for why.
                cur.execute(
                    "SELECT job_id FROM mse.jobs "
                    "WHERE lease_owner IS NOT NULL AND lease_expires_at < now() "
                    "AND release_count >= %s",
                    (MAX_RELEASES,),
                )
                over = [str(r[0]) for r in cur.fetchall()]
                if over:
                    cur.execute(
                        "UPDATE mse.jobs SET status = %s, stage = %s, message = %s, "
                        "finished_at = now(), lease_owner = NULL, "
                        "lease_expires_at = NULL "
                        "WHERE job_id = ANY(%s) AND lease_owner IS NOT NULL "
                        "AND lease_expires_at < now()",
                        (FAILED, FAILED,
                         "the GPU worker stopped responding and the job was "
                         "retried too many times; try again later",
                         over),
                    )
        # Reclaimed-and-requeued only: a job that was just failed is terminal,
        # and reporting it here would tell the caller to go looking for a worker
        # that does not exist. A uniform "all expired ids" return is the
        # instrument lying about which rows moved.
        return [job_id for job_id, _count in rows if job_id not in set(over)]

    def set_k8s_job_name(self, job_id, k8s_job_name: str) -> None:
        """Record the Job name even if the Job could not be created."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET k8s_job_name = %s WHERE job_id = %s",
                (k8s_job_name, str(job_id)),
            )

    # -- reads -------------------------------------------------------------
    def get_job(self, job_id) -> dict | None:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, created_at, original_filename, units, n_index, "
                "n_modes, status, stage, message, k8s_job_name, peak_rss_mb, "
                "finished_at, target, est_tets, min_feature_mm, scale "
                "FROM mse.jobs WHERE job_id = %s",
                (str(job_id),),
            )
            row = cur.fetchone()
        return _job_row(row) if row else None

    def list_jobs(self, limit: int = 25) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, created_at, original_filename, units, n_index, "
                "n_modes, status, stage, message, k8s_job_name, peak_rss_mb, "
                "finished_at, target, est_tets, min_feature_mm, scale "
                "FROM mse.jobs ORDER BY created_at DESC LIMIT %s",
                (int(limit),),
            )
            rows = cur.fetchall()
        return [_job_row(r) for r in rows]

    def list_modes(self, job_id) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT mode_index, k0_per_m, freq_hz, vacuum_wavelength_m "
                "FROM mse.modes WHERE job_id = %s ORDER BY mode_index",
                (str(job_id),),
            )
            rows = cur.fetchall()
        return [
            {"index": int(r[0]), "k0_per_m": r[1], "freq_hz": r[2],
             "vacuum_wavelength_m": r[3]}
            for r in rows
        ]

    # -- the queue ---------------------------------------------------------
    def next_queued(self, target: str = CLUSTER) -> dict | None:
        """Oldest upload still waiting for the Job slot on ``target``.

        Filtered by target: the namespace admits exactly one ``batch/v1`` Job, so
        a jetson row reaching the launcher would occupy that slot while the app
        waited for a GPU worker that is not coming. Jetson rows are claimed by
        the worker API instead -- see ``claim_jetson``.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, original_filename, units, n_index, n_modes, est_tets "
                "FROM mse.jobs WHERE status = %s AND target = %s "
                "ORDER BY created_at LIMIT 1",
                (QUEUED, target),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"job_id": str(row[0]), "original_filename": row[1],
                "units": row[2], "n_index": row[3], "n_modes": row[4],
                "est_tets": row[5]}

    def jetson_waiting(self, older_than_seconds: float) -> list[dict]:
        """Jetson rows nobody has claimed for longer than the wait window.

        ``gpu_queued_at`` is the clock, not ``created_at``: a job that was tried
        on the cluster and rerouted has an old ``created_at`` but has only just
        become a GPU candidate, and counting from ``created_at`` would reroute it
        back to the cluster immediately.

        ``lease_owner IS NULL`` excludes the row a worker is actively holding --
        a slow solve is not an abandoned one, and pulling a live job's row out
        from under it would duplicate the work it is already doing.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, est_tets FROM mse.jobs "
                "WHERE target = %s AND status = %s AND lease_owner IS NULL "
                "AND gpu_queued_at IS NOT NULL "
                "AND gpu_queued_at < now() - make_interval(secs => %s) "
                "ORDER BY gpu_queued_at",
                (JETSON, QUEUED, float(older_than_seconds)),
            )
            return [{"job_id": str(r[0]), "est_tets": r[1]} for r in cur.fetchall()]

    def claim_jetson(self, owner: str, lease_seconds: float) -> dict | None:
        """Atomically take the oldest unclaimed jetson job for ``owner``.

        ``FOR UPDATE SKIP LOCKED`` is the whole point: two workers polling at the
        same instant must get *different* jobs or no job, never the same one. A
        plain ``SELECT`` followed by an ``UPDATE`` would let both read the same
        row and both start solving it -- the classic double-claim, and it would
        only show up as two workers fighting over one job's output files.

        Returns ``None`` when there is nothing to claim, which is the normal
        case (204), not an error.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, units, n_index, n_modes, min_feature_mm, scale "
                "FROM mse.jobs "
                "WHERE target = %s AND status = %s AND lease_owner IS NULL "
                "ORDER BY created_at "
                "FOR UPDATE SKIP LOCKED LIMIT 1",
                (JETSON, QUEUED),
            )
            row = cur.fetchone()
            if row is None:
                return None
            job_id = str(row[0])
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s, k8s_job_name = NULL, "
                "lease_owner = %s, "
                "lease_expires_at = now() + make_interval(secs => %s) "
                "WHERE job_id = %s",
                (RUNNING, "queued", owner, float(lease_seconds), job_id),
            )
        return {"job_id": job_id, "units": row[1], "n_index": row[2],
                "n_modes": row[3], "min_feature_mm": row[4], "scale": row[5],
                "lease_owner": owner}

    def renew_lease(self, job_id, owner: str, lease_seconds: float) -> bool:
        """Extend ``owner``'s lease. Returns whether it still holds it.

        Guarded on ``lease_owner = owner``, so a worker whose lease expired and
        was taken by someone else cannot renew its way back in -- it gets False
        and must stop writing. This is the only defence against two workers
        writing one job's files after a network partition.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET lease_expires_at = "
                "now() + make_interval(secs => %s) "
                "WHERE job_id = %s AND lease_owner = %s",
                (float(lease_seconds), str(job_id), owner),
            )
            return cur.rowcount > 0

    def set_jetson_stage(self, job_id, stage: str) -> None:
        """Record a worker-reported stage on a jetson job.

        Deliberately unguarded on the lease: this is a progress label, not a
        result. A worker that lost its lease mid-solve writing a stale stage
        costs the owner a slightly wrong word on a page that is about to be
        rewritten by the winner; refusing it would cost nothing but would put a
        lease check on the heartbeat path, where a false negative stops a live
        solve.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET stage = %s, status = %s "
                "WHERE job_id = %s AND target = %s",
                (stage, RUNNING, str(job_id), JETSON),
            )

    def finish_jetson(self, job_id, owner: str) -> bool:
        """Mark a jetson job done, if ``owner`` still holds the lease.

        Returns whether it moved. The caller has already checked the lease, but
        this checks again inside the same statement: between the check and the
        write, a lease can expire and another worker can take the job -- and the
        loser must not be the one that marks it done.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s, finished_at = now(), "
                "lease_owner = NULL, lease_expires_at = NULL "
                "WHERE job_id = %s AND lease_owner = %s",
                (DONE, DONE, str(job_id), owner),
            )
            return cur.rowcount > 0

    def fail_jetson(self, job_id, owner: str, message: str) -> bool:
        """Mark a jetson job FAILED for good, if ``owner`` still holds the lease.

        TERMINAL, and that is the point. The earlier behaviour -- releasing the
        lease and requeueing -- meant a deterministic failure (a mesh the
        mesher refuses, a box that is too detailed, a solver exception) came
        straight back to the same or the next worker, which failed it the same
        way. Forever. The message was logged and nowhere else, so the owner saw
        a job that never finished and no reason for it.

        ``finished_at`` is set and the lease cleared in the same statement, so
        the row is in exactly the state the cluster's own failure path leaves it
        in: the owner's job page reads it through the same column.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s, message = %s, "
                "finished_at = now(), lease_owner = NULL, lease_expires_at = NULL "
                "WHERE job_id = %s AND lease_owner = %s",
                (FAILED, FAILED, message, str(job_id), owner),
            )
            return cur.rowcount > 0

    def release_jetson(self, job_id, owner: str, message: str = "") -> str | None:
        """Hand a job back with no verdict -- or FAIL it if it has been released
        too many times. One statement, one transaction.

        Returns ``"released"``, ``"failed"`` (the bound was reached and the job
        is now terminal) or ``None`` (this owner does not hold the lease -- the
        caller answers 409, never a silent success).

        For TRANSIENT reasons only -- a SIGTERM shutdown mid-job, a CUDA init
        failure, a lost connection. Not a verdict about the job, so the row goes
        back to ``queued`` and another worker (or this one, later) can take it.

        WHY ONE STATEMENT. The bound has to hold when two releases race. A
        read-count-then-decide form lets both callers read ``count = 2``, both
        conclude they are under the bound, and both requeue -- the job that
        should have failed at 3 instead goes round again, which is exactly the
        non-terminating loop this whole mechanism exists to remove. Putting the
        counter in the ``UPDATE``'s WHERE clause makes the database do the
        comparison, so a losing racer matches zero rows and takes the failure
        branch below.

        WHY THE COUNT IS ON EVERY PATH BACK TO CLAIMABLE. ``/release`` is not
        the only way a jetson row returns to the queue: a lease EXPIRY (a worker
        that died mid-job) requeues it too. If only ``/release`` incremented the
        count, a worker that crashed every time would requeue forever through
        the timeout path and never hit the bound. See ``reclaim_expired_leases``.
        """
        with self._connect() as conn, conn.cursor() as cur:
            # Try the bounded release first. The WHERE carries the bound, so a
            # racer that loses sees rowcount 0 and falls through to the failure.
            cur.execute(
                "UPDATE mse.jobs SET lease_owner = NULL, lease_expires_at = NULL, "
                "status = %s, stage = %s, gpu_queued_at = now(), "
                "release_count = release_count + 1 "
                "WHERE job_id = %s AND lease_owner = %s AND release_count < %s",
                (QUEUED, "queued", str(job_id), owner, MAX_RELEASES),
            )
            if cur.rowcount > 0:
                return "released"
            # Zero rows: either the lease is not ours (409) or the bound is
            # reached. Distinguish them, because they are not the same event.
            cur.execute(
                "SELECT release_count FROM mse.jobs "
                "WHERE job_id = %s AND lease_owner = %s",
                (str(job_id), owner),
            )
            row = cur.fetchone()
            if row is None:
                return None
            # Over the bound and still ours: fail it, in this same transaction.
            # The message is written from THIS attempt -- the failure the owner
            # should see is the one that finally ended it, not the first one,
            # which would name an earlier cause than the run that gave up.
            #
            # The branch is guarded on the same predicate the SQL uses, so the
            # two cannot drift: a row that reaches here with ``row[0]`` under the
            # bound means the UPDATE above matched nothing for some OTHER reason,
            # and failing the job on that would be the instrument deciding the
            # verdict. It cannot happen at READ COMMITTED (the row lock makes the
            # second statement see the first's result), so it is a guard rather
            # than a path -- but a guard that cannot refuse is not a guard.
            if not releases_are_terminal(row[0]):
                return None
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s, message = %s, "
                "finished_at = now(), lease_owner = NULL, lease_expires_at = NULL "
                "WHERE job_id = %s AND lease_owner = %s",
                (FAILED, FAILED,
                 message or f"the GPU worker could not complete this job after "
                            f"{row[0]} attempts; try again later",
                 str(job_id), owner),
            )
            return "failed"

    def lease_held_by(self, job_id, owner: str) -> bool:
        """Whether ``owner`` currently holds ``job_id``'s lease and it is live.

        Every worker write goes through this. Two separate conditions, and both
        matter: the owner must match (a stale worker must not write) and the
        lease must not have expired (a worker that vanished and came back must
        not write either). Checking only the owner would let a worker that slept
        through its lease resume writing over its successor's output.
        """
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM mse.jobs WHERE job_id = %s AND lease_owner = %s "
                "AND lease_expires_at > now()",
                (str(job_id), owner),
            )
            return cur.fetchone() is not None

    def queue_position(self, job_id) -> int | None:
        """1-based place in the queue, or None once the job is no longer queued."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT created_at, status FROM mse.jobs WHERE job_id = %s",
                        (str(job_id),))
            row = cur.fetchone()
            if not row or row[1] != QUEUED:
                return None
            cur.execute(
                "SELECT count(*) FROM mse.jobs WHERE status = %s AND created_at < %s",
                (QUEUED, row[0]),
            )
            ahead = int(cur.fetchone()[0])
        return ahead + 1

    def count_running(self) -> int:
        """Jobs the app believes are running (belt and braces; the Job list is
        the authority)."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM mse.jobs WHERE status = %s", (RUNNING,))
            return int(cur.fetchone()[0])

    # NOTE: there is deliberately no `requeue_stale` here any more. It put rows
    # whose Job had vanished back into `queued`, which sounds self-healing but
    # re-runs a solve that has already died -- an object that OOMs would OOM on
    # every attempt, forever. A row that is `running` with no Job behind it did
    # not finish, so the tick now fails it (see app._fail_dead_jobs) and the
    # owner is told, rather than watching a spinner that never ends.


def _job_row(row) -> dict:
    return {
        "job_id": str(row[0]),
        "created_at": _iso(row[1]),
        "original_filename": row[2],
        "units": row[3],
        "n_index": row[4],
        "n_modes": row[5],
        "status": row[6],
        "stage": row[7],
        "message": row[8],
        "k8s_job_name": row[9],
        "peak_rss_mb": row[10],
        "finished_at": _iso(row[11]),
        "target": row[12],
        "est_tets": row[13],
        "min_feature_mm": row[14],
        "scale": row[15],
    }


def _iso(value) -> str | None:
    """timestamptz -> ISO 8601 in UTC, so the browser can localise it."""
    if value is None:
        return None
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()
    return str(value)
