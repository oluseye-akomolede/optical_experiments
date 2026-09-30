# job-lifecycle

## ADDED Requirements

### Requirement: A job is a row in `mse.jobs` with four states

A job SHALL be a row in `mse.jobs` whose status is one of `queued`, `running`,
`done`, `failed`. `done` and `failed` SHALL be the finished states. Source:
`web/db.py`, `db/001_schema.sql`.

#### Scenario: a job is created
- **WHEN** a job is created
- **THEN** its row is inserted with status and stage both `queued`

#### Scenario: a restored job carries the same status
- **WHEN** the app repairs a terminal job whose lease was left set
- **THEN** it appends to `message` and does NOT change the row's status

### Requirement: Claiming a job takes a lease atomically

A claim SHALL select at most one claimable row (`target = 'jetson'`,
`status = 'queued'`, `lease_owner IS NULL`) with
`FOR UPDATE SKIP LOCKED LIMIT 1`, and SHALL, in the same transaction, set
`status = 'running'`, `lease_owner` to the worker's id, and
`lease_expires_at = now() + lease_seconds`. Source: `web/db.py`.

#### Scenario: no claimable row
- **WHEN** a worker claims and no job is claimable
- **THEN** the app returns 204 with an empty body

#### Scenario: the claim is atomic under concurrency
- **WHEN** two workers claim concurrently
- **THEN** they never receive the same job

### Requirement: A lease is renewed only by its owner

A lease SHALL be renewable only by the worker that holds it. Renewal SHALL be
guarded on `lease_owner = owner`, so a worker whose lease has been reclaimed
cannot renew back into the job. Source: `web/db.py`, `web/worker_api.py`.

#### Scenario: a worker whose lease expired tries to report progress
- **WHEN** a worker renews a lease it no longer holds
- **THEN** the renewal affects no row and the progress route returns 409

### Requirement: Release clears the lease and resets the status

Every path that clears `lease_owner` SHALL set a claimable status in the SAME
statement; no path may clear the lease while leaving the row unclaimable.
A release SHALL return the row to `queued` and SHALL increment
`release_count`. Source: `web/db.py`.

#### Scenario: a transient fault releases a job
- **WHEN** a worker reports a transient fault
- **THEN** the row's lease is cleared, its status is `queued`, and
  `release_count` is incremented

#### Scenario: a lease clearing must not strand a row
- **WHEN** any lease-clearing path in the app is exercised
- **THEN** the row's status is claimable (not a stale `running`)

### Requirement: Repeated releases terminate in a failure

A job SHALL be retried at most `MAX_RELEASES` (3) times. At the bound, the job
SHALL be marked `failed` rather than requeued, so a crash-looping worker's job
does not requeue forever. Source: `web/db.py`.

#### Scenario: the retry bound is reached
- **WHEN** a job's `release_count` is at the bound and another release or lease
  expiry occurs
- **THEN** the app marks it `failed` with a message that the worker repeatedly
  stopped responding

### Requirement: A crashed worker's row is recovered by lease expiry

A lease expiring while the row is `running` SHALL be reclaimed by the tick:
the lease is cleared, the row is requeued, and `release_count` is incremented
(an expiry counts as a release, because a worker that dies never calls
`/release`). Source: `web/db.py`, `web/app.py`.

#### Scenario: a worker dies mid-job
- **WHEN** the tick finds a `running` row whose lease has expired below the bound
- **THEN** the row is requeued and its `release_count` is incremented

### Requirement: The tick runs the reapers on a fixed interval

The app SHALL run its reapers from one ticker thread every `MSE_TICK_SECONDS`
(default 5 s): reclaim expired leases, reroute abandoned GPU jobs after the GPU
wait window, fail cluster jobs whose Kubernetes Job is gone or failed, and
delete finished Jobs that still hold a quota slot. The dead-job reaper SHALL
skip `target = 'jetson'` rows. Source: `web/app.py`, `web/db.py`.

#### Scenario: a Jetson job is not reaped by the cluster reaper
- **WHEN** the dead-job reaper examines a row whose target is `jetson`
- **THEN** it skips it

#### Scenario: the cluster cannot be asked
- **WHEN** the reaper cannot list or inspect Jobs
- **THEN** it fails no job and returns without guessing

#### Scenario: a GPU job is abandoned
- **WHEN** a queued `jetson` job has waited longer than the GPU wait window and
  its estimate is within the cluster's size limit
- **THEN** it is rerouted to the cluster

### Requirement: Failure reaches the user through `message`

The status route SHALL return 200 with the row's state regardless of state, and
SHALL carry the failure in `message`, not a separate failure body. Failure
messages SHALL be plain English and SHALL NOT interpolate the internal reason
token. Source: `web/app.py`, `web/jobs.py`.

#### Scenario: a cluster job is OOM-killed
- **WHEN** a cluster Job's failure reason is `OOMKilled`
- **THEN** the message asks the user to try a simpler object or fewer modes

#### Scenario: the reason is unknown
- **WHEN** the failure reason is not recognised
- **THEN** the generic plain-English message is used

#### Scenario: a completed job reports its modes
- **WHEN** the status route is asked for a `done` job
- **THEN** the payload includes the job's modes

### Requirement: The worker API is authenticated and single-purpose per route

Every worker route SHALL require a bearer token
(`MSE_WORKER_TOKEN`, compared with `hmac.compare_digest`). The surface SHALL be:
claim, fetch input, report progress, upload an allowed result file, report done
with modes, report failed (terminal), and release (transient). Uploaded file
names SHALL be checked against an allow-list (`modes.json`, `surface.glb`,
`mode_NN.bin`, `mode_NN.vtu`) with path components rejected. Source:
`web/worker_api.py`.

#### Scenario: the token is missing or wrong
- **WHEN** a route is called with no token, or a wrong token
- **THEN** the app returns 401 (or 503 if no token is configured)

#### Scenario: a file name escapes the allow-list
- **WHEN** an upload names a file outside the allow-list, or a name containing a
  path component
- **THEN** the app returns 400 and stores nothing

#### Scenario: an oversized upload
- **WHEN** an upload's `Content-Length`, or its actual body, exceeds the file
  size bound
- **THEN** the app returns 413
