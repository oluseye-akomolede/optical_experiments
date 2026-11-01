## MODIFIED Requirements

### Requirement: A time-domain job SHALL be stored as kind `td` with a keyed request

A `td` job SHALL be stored in the jobs table with `kind = 'td'` and its request
in a `tds` JSON column, NULL for every other kind (including `elec`, whose
request is its own `elecs` column, `electrical-jobs`). The request keys SHALL be
exactly the worker's `REQUEST_KEYS` list (`boundary`, `source_kind`, `profile`,
`frequency_hz`, `bandwidth`, `position`, `direction`, `polarisation`,
`samples`, `sample_duration_s`, `duration_s`, `probes`, `dx_m`, `coatings`).
Every key SHALL be present; a key that does not apply to the chosen source holds null. The
worker SHALL refuse a request with a missing key, naming every missing key, and
SHALL refuse a `tds` that is not an object. Keys outside the list SHALL be
ignored, so results written back to the row are never read as input. The TF/SF
box is derived from the geometry, not stored. The claim payload SHALL always
carry `tds` (null for other kinds). Creating a job with a kind outside the known
kinds (`cavity`, `pc`, `td`, `elec`) SHALL be refused before any row is written. The
column is added idempotently by `db/001_schema.sql` and dropped by
`db/003_rollback_td.sql`.

#### Scenario: a missing request key is named
- **WHEN** a claimed `td` job's `tds` lacks one or more keys
- **THEN** the worker refuses it with a message naming every missing key

#### Scenario: a misspelt kind never reaches the table
- **WHEN** a job is created with kind `TD`
- **THEN** it is refused and no INSERT is sent

#### Scenario: the worker dispatches every kind by name
- **WHEN** the worker claims a job
- **THEN** it decides the kind once, before fetching anything: `cavity` (or no
  kind, a row older than the column) and `td` fetch their upload, `pc` fetches
  nothing, `elec` fetches its upload, and a `td` job is reported as "solving", never "meshing"
- **AND** a kind outside the worker's list fails terminally with a message
  naming it, before any input is fetched, and is never solved as a cavity
- **AND** the worker's kind list equals the app's
