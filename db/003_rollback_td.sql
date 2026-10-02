-- Rollback for the time-domain change (openspec/changes/time-domain-fdtd).
--
-- Drops the one column that change added to mse.jobs, `tds`, for the reason
-- `002_rollback_pc.sql` exists: `db/001_schema.sql` is applied idempotently and
-- never rolls anything back.
--
-- WHEN TO RUN THIS: only after the app has been rolled back to a revision whose
-- code does not read `tds`. It DESTROYS every time-domain request. Rows with
-- kind = 'td' are left in place but no longer runnable by the older code, which
-- refuses unknown kinds; delete or fail them first if that matters. Take a dump
-- first:
--
--     pg_dump --schema=mse "$DATABASE_URL" > mse-before-rollback.sql
--     psql "$DATABASE_URL" -f db/003_rollback_td.sql
--
-- `IF EXISTS`, so running it twice is a no-op.

BEGIN;

ALTER TABLE mse.jobs DROP COLUMN IF EXISTS tds;

COMMIT;
