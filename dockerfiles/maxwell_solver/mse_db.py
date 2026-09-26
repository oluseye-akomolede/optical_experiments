"""Postgres bookkeeping for MSE jobs.

All statements are parameterised; no value is ever interpolated into SQL. The
schema lives in the ``mse`` schema (see ``db/001_schema.sql``).

The connection is optional: the test suite and the ``--local-*`` path run
without a database, so every helper here no-ops when no connection was given.
"""

from __future__ import annotations

import os
from contextlib import contextmanager

#: stage names the UI polls for, in order
STAGES = ("meshing", "assembling", "solving", "writing", "done")

#: filename of the DDL script, kept the same as web/db.py's constant
SCHEMA_FILENAME = "001_schema.sql"


def dsn_from_env() -> dict:
    """Connection parameters from the environment (never from a file)."""
    return {
        "host": os.environ.get("DB_HOST", "localhost"),
        "port": os.environ.get("DB_PORT", "5432"),
        "dbname": os.environ.get("DB_NAME", "mode_search_engine"),
        "user": os.environ.get("DB_USER"),
        "password": os.environ.get("DB_PASSWORD"),
    }


class JobStore:
    """Thin wrapper around the ``mse`` schema. Safe to use as a no-op."""

    def __init__(self, dsn: dict | None = None):
        self.dsn = dsn or dsn_from_env()
        self._conn = None

    # -- lifecycle ---------------------------------------------------------
    def connect(self):
        import psycopg2

        self._conn = psycopg2.connect(
            host=self.dsn["host"],
            port=self.dsn["port"],
            dbname=self.dsn["dbname"],
            user=self.dsn["user"],
            password=self.dsn["password"],
        )
        self._conn.autocommit = True
        return self

    def close(self):
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self.connect()

    def __exit__(self, *exc):
        self.close()

    # -- helpers -----------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self._conn is not None

    @contextmanager
    def _cursor(self):
        if self._conn is None:
            yield None
            return
        cur = self._conn.cursor()
        try:
            yield cur
        finally:
            cur.close()

    def apply_schema(self, path: str | None = None):
        """Apply ``db/001_schema.sql`` idempotently.

        The file is a single multi-statement script (including a ``DO $$`` block
        that renames the legacy ``spark_app_name`` column); psycopg2 sends it as
        one execute, which the server runs as a single transaction.

        Nothing in the solver calls this: the web app's initContainer owns the
        schema and applies it before any Job is queued. It is kept for
        operators running the solver outside the cluster (``--local-*`` paths do
        not touch the database at all), and resolves its path the same
        candidate-list way as the web app.

        The two resolvers are deliberately not mirror copies the whole way down
        -- the connection models differ (``_connect`` vs ``_cursor`` plus a
        ``None`` guard) -- so what is actually shared and tested is the filename
        constant and the candidate SEMANTICS: env override first, then beside
        the module, then one level up. The bodies may change independently.
        """
        if self._conn is None:
            return
        if path is None:
            path = self._resolve_schema_path()
        with open(path, encoding="utf-8") as fh:
            sql = fh.read()
        with self._cursor() as cur:
            cur.execute(sql)

    @staticmethod
    def _resolve_schema_path() -> str:
        """First existing candidate, else a clear error naming all of them."""
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [os.path.join(here, "db", SCHEMA_FILENAME)]
        candidates.extend(
            os.path.join(os.path.dirname(here), *([os.pardir] * n),
                         "db", SCHEMA_FILENAME)
            for n in range(1, 4)
        )
        override = os.environ.get("MSE_SCHEMA_PATH")
        if override:
            candidates.insert(0, override)
        for candidate in candidates:
            resolved = os.path.normpath(candidate)
            # isfile, not exists: a directory would pass exists and then fail at
            # open() with an unexplained IsADirectoryError
            if os.path.isfile(resolved):
                return resolved
        raise FileNotFoundError(
            "could not find " + SCHEMA_FILENAME + "; looked in: "
            + ", ".join(os.path.normpath(c) for c in candidates)
        )

    def set_stage(self, job_id, status: str, stage: str | None = None,
                  message: str | None = None, k8s_job_name: str | None = None,
                  peak_rss_mb: float | None = None):
        """Update the solver's progress on a job.

        Only the *column names* are ever joined into the statement text; they
        come from this fixed set of literals and are never user input. Every
        value goes through a ``%s`` placeholder.
        """
        with self._cursor() as cur:
            if cur is None:
                return
            fields = ["status = %s"]
            params: list = [status]
            if stage is not None:
                fields.append("stage = %s")
                params.append(stage)
            if message is not None:
                fields.append("message = %s")
                params.append(message)
            if k8s_job_name is not None:
                fields.append("k8s_job_name = %s")
                params.append(k8s_job_name)
            if peak_rss_mb is not None:
                fields.append("peak_rss_mb = %s")
                params.append(float(peak_rss_mb))
            if status in ("done", "failed"):
                fields.append("finished_at = now()")
            params.append(str(job_id))
            cur.execute(
                f"UPDATE mse.jobs SET {', '.join(fields)} WHERE job_id = %s",
                params,
            )

    def insert_modes(self, job_id, rows):
        """Insert one row per mode: ``(index, k0_per_m, freq_hz, wavelength_m)``."""
        if self._conn is None or not rows:
            return
        with self._cursor() as cur:
            if cur is None:
                return
            cur.executemany(
                "INSERT INTO mse.modes "
                "(job_id, mode_index, k0_per_m, freq_hz, vacuum_wavelength_m) "
                "VALUES (%s, %s, %s, %s, %s) "
                "ON CONFLICT (job_id, mode_index) DO UPDATE SET "
                "k0_per_m = EXCLUDED.k0_per_m, "
                "freq_hz = EXCLUDED.freq_hz, "
                "vacuum_wavelength_m = EXCLUDED.vacuum_wavelength_m",
                [(str(job_id), i, k0, f, wl) for (i, k0, f, wl) in rows],
            )

    def fail(self, job_id, message: str, peak_rss_mb: float | None = None):
        """Record a human-readable failure.

        ``message`` is expected to be the plain-English text the owner sees on
        the job page, not a stack trace.
        """
        try:
            self.set_stage(job_id, "failed", message=message,
                           peak_rss_mb=peak_rss_mb)
        except Exception:  # noqa: BLE001 - never mask the original error
            pass

    def delete_modes(self, job_id):
        """Clear a job's modes (a re-run must not leave stale higher indices)."""
        if self._conn is None:
            return
        with self._cursor() as cur:
            if cur is None:
                return
            cur.execute("DELETE FROM mse.modes WHERE job_id = %s", (str(job_id),))
