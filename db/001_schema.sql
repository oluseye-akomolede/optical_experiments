-- MSE job bookkeeping.
--
-- Applied idempotently by the web app (and by the solver at start-up) so the
-- schema exists whichever component gets there first. Parameterised SQL only:
-- values are never interpolated into statements.
--
-- The tables live in their own `mse` schema so this project cannot collide with
-- the other databases that share this Postgres instance.

CREATE SCHEMA IF NOT EXISTS mse;

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
