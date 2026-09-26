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
                   n_index: float, n_modes: int) -> None:
        """Create the upload row. Status starts ``queued``, not ``running``."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "INSERT INTO mse.jobs "
                "(job_id, original_filename, units, n_index, n_modes, status, stage) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (str(job_id), original_filename, units, float(n_index),
                 int(n_modes), QUEUED, QUEUED),
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
                "finished_at FROM mse.jobs WHERE job_id = %s",
                (str(job_id),),
            )
            row = cur.fetchone()
        return _job_row(row) if row else None

    def list_jobs(self, limit: int = 25) -> list[dict]:
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, created_at, original_filename, units, n_index, "
                "n_modes, status, stage, message, k8s_job_name, peak_rss_mb, "
                "finished_at FROM mse.jobs ORDER BY created_at DESC LIMIT %s",
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
    def next_queued(self) -> dict | None:
        """Oldest upload still waiting for the Job slot."""
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "SELECT job_id, original_filename, units, n_index, n_modes "
                "FROM mse.jobs WHERE status = %s ORDER BY created_at LIMIT 1",
                (QUEUED,),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {"job_id": str(row[0]), "original_filename": row[1],
                "units": row[2], "n_index": row[3], "n_modes": row[4]}

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

    def requeue_stale(self, job_ids) -> int:
        """Put jobs the cluster no longer knows about back in the queue.

        ``job_ids`` are the ones the tick found neither running nor queued; a
        Job deleted by hand (or lost with the node) would otherwise leave its
        row stuck in ``running`` forever.
        """
        ids = [str(j) for j in job_ids]
        if not ids:
            return 0
        with self._connect() as conn, conn.cursor() as cur:
            cur.execute(
                "UPDATE mse.jobs SET status = %s, stage = %s "
                "WHERE status = %s AND job_id = ANY(%s)",
                (QUEUED, QUEUED, RUNNING, ids),
            )
            return cur.rowcount


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
