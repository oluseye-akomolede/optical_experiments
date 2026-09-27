-- MSE job bookkeeping.
--
-- Applied idempotently by the web app's initContainer, which runs before the
-- web container and therefore before any solver Job can be queued. The solver
-- does NOT apply it: it only reads and writes rows, and a Job that ran DDL
-- would repeat identical work on every upload. Parameterised SQL only: values
-- are never interpolated into statements.
--
-- The tables live in their own `mse` schema so this project cannot collide with
-- the other databases that share this Postgres instance.

-- Create the schema only when it is actually missing.
--
-- NOT `CREATE SCHEMA IF NOT EXISTS mse`: Postgres checks CREATE privilege on
-- the DATABASE before it evaluates the IF NOT EXISTS short-circuit, so that
-- statement fails even when the schema is already present. Our scoped role
-- mse_app owns the `mse` schema but deliberately has no database-level CREATE,
-- and it should not be granted one merely to make RE-application idempotent.
--
-- Guarding on pg_namespace instead means that when the schema already exists it
-- never reaches a statement that needs the database privilege, so re-running
-- this file against a deployed database is free.
--
-- When the schema is ABSENT this still needs database-level CREATE, which the
-- scoped app role does not have: the inner CREATE SCHEMA then fails with the
-- same bare "permission denied for database ..." as before, so a deploy log
-- alone cannot distinguish "schema missing" from "schema present". Provisioning
-- a missing schema is therefore an operator's one-off with a role that holds the
-- grant (CREATE SCHEMA mse AUTHORIZATION <app_role>), not something this file
-- can do on a fresh database. The CREATE is kept, not replaced by a RAISE, so
-- that a role which does hold the grant still provisions in one step.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = 'mse'
    ) THEN
        CREATE SCHEMA mse;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS mse.jobs (
    job_id            uuid PRIMARY KEY,
    created_at        timestamptz NOT NULL DEFAULT now(),
    original_filename text,
    units             text        NOT NULL DEFAULT 'mm',
    n_index           double precision NOT NULL DEFAULT 1.0,
    n_modes           integer     NOT NULL DEFAULT 20,
    status            text        NOT NULL DEFAULT 'queued',
    stage             text,
    message           text,
    k8s_job_name      text,
    peak_rss_mb       double precision,
    finished_at       timestamptz
);

-- The solver used to run as a SparkApplication; it is now a plain batch/v1 Job,
-- so the column is named for what it holds. `IF NOT EXISTS` cannot rename, so
-- upgrade an existing table in place.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs'
          AND column_name = 'spark_app_name'
    ) AND NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs'
          AND column_name = 'k8s_job_name'
    ) THEN
        ALTER TABLE mse.jobs RENAME COLUMN spark_app_name TO k8s_job_name;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs'
          AND column_name = 'peak_rss_mb'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN peak_rss_mb double precision;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS mse.modes (
    job_id                uuid    NOT NULL REFERENCES mse.jobs(job_id) ON DELETE CASCADE,
    mode_index            integer NOT NULL,
    k0_per_m              double precision,
    freq_hz               double precision,
    vacuum_wavelength_m   double precision,
    PRIMARY KEY (job_id, mode_index)
);

CREATE INDEX IF NOT EXISTS jobs_status_idx ON mse.jobs (status);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON mse.jobs (created_at DESC);
