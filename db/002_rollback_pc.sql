-- Rollback for the photonic-crystal change (openspec/changes/photonic-crystal-bands).
--
-- Drops the two columns that change added to mse.jobs. `db/001_schema.sql` is
-- written to be applied idempotently and never rolls anything back, so undoing
-- the change needs its own file: without it, rolling the app back to the commit
-- before photonic crystals leaves two columns behind that no code reads, and
-- the next forward migration meets a table that is neither the old shape nor
-- the new one.
--
-- The schema qualifier is `mse`, read from `db/001_schema.sql` where every
-- statement is `mse.jobs` (the `mse` schema exists so this project cannot
-- collide with the other databases on the shared Postgres instance). The
-- reviewer's two notes disagreed about this, so it is taken from the file, not
-- from either note.
--
-- WHEN TO RUN THIS: only after the app has been rolled back to a revision whose
-- code does not read `kind` or `pcs`, and only on a database you intend to roll
-- back. It DESTROYS the pc request/result JSON and every row's kind. Take a
-- dump first if any band diagram in the table is still wanted:
--
--     pg_dump --schema=mse "$DATABASE_URL" > mse-before-rollback.sql
--
-- Run it the same way the schema is applied:
--
--     psql "$DATABASE_URL" -f db/002_rollback_pc.sql
--
-- `IF EXISTS` on both, so a rollback run twice is a no-op rather than an error,
-- matching the forward file's idempotence.

BEGIN;

-- The band-diagram request/result JSON. Every value here is a photonic-crystal
-- payload; nothing else ever wrote this column.
ALTER TABLE mse.jobs DROP COLUMN IF EXISTS pcs;

-- The job kind. Dropping it also drops its DEFAULT ('cavity') and the NOT NULL
-- with it, which is the pre-photonic shape: a table with one implicit kind.
ALTER TABLE mse.jobs DROP COLUMN IF EXISTS kind;

COMMIT;
