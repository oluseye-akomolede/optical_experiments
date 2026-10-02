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
    finished_at       timestamptz,
    -- The mesh inputs the owner may set. Present here as well as in the upgrade
    -- block below so a fresh database and an upgraded one end up identical.
    --
    -- The smallest feature is a PHYSICAL length: the owner types a number and
    -- picks nm/um/mm, INDEPENDENT of the object's `units`. `min_feature_m` is
    -- the canonical metres value the solver uses; `min_feature_value` and
    -- `min_feature_unit` are the input as typed, kept so the page can show the
    -- owner their own number back. All three NULL means "the physics edge
    -- alone". Three columns rather than one because a length whose unit is
    -- implicit is exactly what the bug this replaces was made of.
    min_feature_m     double precision,
    min_feature_value double precision,
    min_feature_unit  text,
    scale             double precision NOT NULL DEFAULT 1.0,
    -- How many times a GPU worker has handed this job back with no verdict
    -- (POST /worker/jobs/<id>/release). It exists so that a job which can never
    -- complete cannot loop forever: releases are legitimate for transient
    -- reasons, but a bounded number of them. See the cap in web/worker_api.py.
    release_count     integer NOT NULL DEFAULT 0,
    -- The per-part refractive indices, as a JSON object mapping the .obj's part
    -- NAME to that part's `n`. The name is the key because the name is what the
    -- owner assigned the material to; geometry is re-derived from the upload, so
    -- an index into the part list would be a second, weaker key that can drift
    -- from the first.
    --
    -- NULL means "no materials table": the one-part vacuum path, exactly as
    -- before this column existed. A NULL is not the empty object -- `{}` would
    -- be a table with no parts, which is a different and wrong statement -- so
    -- the column stays NULL rather than being defaulted, and every reader
    -- treats NULL and absent alike.
    materials         jsonb,
    -- The job KIND. `cavity` is the 3-D vector solver this table has always
    -- held; `pc` is a 2-D photonic-crystal band diagram. It is a discriminator
    -- so the two kinds are views of one table rather than two tables: both go
    -- through the same claim -> solve -> store path and want the same status,
    -- lease and routing columns.
    --
    -- The DEFAULT 'cavity' is what keeps every existing row and every existing
    -- reader on the path they were already on: a row written before this column
    -- existed is a cavity job, so the default is not a placeholder guess but the
    -- true value. `kind` is NOT NULL for that reason -- unlike `materials`,
    -- there is no "absent" state, only "cavity".
    kind              text        NOT NULL DEFAULT 'cavity',
    -- The photonic-crystal request and its result, as JSON: the request
    -- (lattice, r_over_a, n_rod, n_bg, n_bands) and the computed bands.
    --
    -- NULL means "not a pc job", exactly as NULL `materials` means the one-part
    -- path. It is NOT defaulted to `{}`: a cavity job has no photonic-crystal
    -- request, and `{}` would claim one with every field missing.
    pcs               jsonb,
    -- The time-domain (`td`) request, as JSON: the keys `worker/td_solve.py`
    -- reads (REQUEST_KEYS). NULL for every other kind, for the reason `pcs` is.
    tds               jsonb
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

-- --------------------------------------------------------------------------
-- Routing and the GPU worker (dispatch 2/3)
--
-- Additive only: existing rows keep working, and every column is nullable or
-- defaulted so a Job launched before this change is unaffected. The web app's
-- initContainer re-applies this file on every deploy, so each addition is
-- guarded the same way the earlier ones are.
-- --------------------------------------------------------------------------
DO $$
BEGIN
    -- Which machine this job is for: 'cluster' (a k8s Job, as before) or
    -- 'jetson' (waiting for a GPU worker to claim it). Existing rows default
    -- to 'cluster', which is what they actually did.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'target'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN target text NOT NULL DEFAULT 'cluster';
    END IF;

    -- The upload-time tet estimate that chose `target`. Kept because it is the
    -- number the routing decision was made on, and a wrong route is diagnosed
    -- by comparing it with the count the solver actually reported.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'est_tets'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN est_tets bigint;
    END IF;

    -- Lease: which worker holds this job and until when. A lease that has
    -- expired is how a dead worker is detected -- there is no other signal,
    -- because the worker's whole contract is to renew it.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'lease_owner'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN lease_owner text;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'lease_expires_at'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN lease_expires_at timestamptz;
    END IF;

    -- When this job first became claimable on the jetson queue, so the
    -- fallback can tell "nobody has picked this up for MSE_GPU_WAIT_S" from
    -- "it is simply new".
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'gpu_queued_at'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN gpu_queued_at timestamptz;
    END IF;

    -- The two mesh inputs the owner can set. Both are recorded rather than
    -- derived: the min_feature_* triplet is the smallest feature the modes must
    -- resolve (NULL = the physics edge alone; see the CREATE TABLE above for
    -- why it is three columns), and scale multiplies every vertex before
    -- anything measures it. Without these a job's frequencies cannot be
    -- explained from its row alone -- the same .obj at scale 1000 is a
    -- different object, and the row would not say so.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'min_feature_m'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN min_feature_m double precision;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'min_feature_value'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN min_feature_value double precision;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'min_feature_unit'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN min_feature_unit text;
    END IF;

    -- NO DROP for the superseded `min_feature_mm`, deliberately. An earlier
    -- version of this file dropped it, to make fresh and upgraded databases
    -- match for any DB that had run the intermediate commit -- and that was the
    -- wrong trade. PRODUCTION never had the column: it runs 8e27869, which
    -- predates the commit that introduced it (verified with
    -- ``git merge-base --is-ancestor 8e27869 25ad6d8``; 8e27869 has no
    -- ``min_feature`` or ``scale`` column at all). So the DROP would never fire
    -- in production, and its only effect there would be a destructive statement
    -- running on every app start in exchange for tidying databases that are
    -- thrown away. A schema script that DROPs on every boot to fix dev
    -- leftovers is a worse thing to own than a dead column in a scratch DB.
    --
    -- The upgrade that matters is from 8e27869, and it is additive: the three
    -- columns below plus `scale`. That path is what A2c in tests/pg_race_check.py
    -- builds and checks.

    -- NOT NULL DEFAULT 1: an old row is an unscaled upload by definition, so
    -- backfilling 1 is the true value, not a placeholder that has to be
    -- interpreted. The default is what keeps `scale` from ever being NULL in a
    -- multiplication.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'scale'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN scale double precision NOT NULL DEFAULT 1.0;
    END IF;

    -- How many times a worker has released this job with no verdict. Additive
    -- and defaulted to 0, so every existing row means "released zero times",
    -- which is the true value rather than a placeholder.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'release_count'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN release_count integer NOT NULL DEFAULT 0;
    END IF;

    -- The materials table, for databases created before multi-material uploads.
    -- Deliberately NULLABLE with no default: NULL is the true value for every
    -- existing row (they are one-part vacuum uploads), and unlike `scale` there
    -- is no sensible non-NULL placeholder -- `{}` would claim a table the owner
    -- never entered. See the column comment above the CREATE for why the part
    -- name is the key.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'materials'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN materials jsonb;
    END IF;

    -- The job kind, for databases created before photonic crystals. Added with
    -- the DEFAULT so existing rows take the true value ('cavity') and a NOT
    -- NULL holds from the start -- the same shape as `release_count` above.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'kind'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN kind text NOT NULL DEFAULT 'cavity';
    END IF;

    -- The photonic-crystal request/result JSON, nullable for the same reason as
    -- `materials`: NULL is the true value for every pre-existing cavity row.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'pcs'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN pcs jsonb;
    END IF;

    -- The time-domain request JSON, for databases created before `td`.
    -- Nullable: NULL is the true value for every pre-existing row.
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'tds'
    ) THEN
        ALTER TABLE mse.jobs ADD COLUMN tds jsonb;
    END IF;
END $$;

-- The claim query scans for the oldest unleased jetson job, so it wants an
-- index on exactly that. Partial, because leased and cluster rows are never
-- candidates and the jetson queue stays small next to the archive.
CREATE INDEX IF NOT EXISTS jobs_jetson_claim_idx
    ON mse.jobs (created_at)
    WHERE target = 'jetson' AND status = 'queued' AND lease_owner IS NULL;
