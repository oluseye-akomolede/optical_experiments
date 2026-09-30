# jetson-worker

## ADDED Requirements

### Requirement: One job at a time, gated before claiming

The worker SHALL run at most one job at a time. It SHALL acquire the bench lock
`~/mse-bench/.gpu.lock` with `flock(LOCK_EX | LOCK_NB)` BEFORE claiming a job,
and SHALL NOT claim while the lock is held. The job it takes SHALL come from an
atomic single-row claim: `SELECT ... FOR UPDATE SKIP LOCKED LIMIT 1` over rows
`target = 'jetson' AND status = 'queued' AND lease_owner IS NULL`. Source:
`worker/mse_worker.py`, `web/db.py`.

#### Scenario: the bench lock is held
- **WHEN** the worker cannot take the bench lock
- **THEN** it logs that the lock is held and does not claim
- **AND** it never claims a job and then leaves it queued

#### Scenario: two workers poll at the same instant
- **WHEN** two workers issue the claim query concurrently
- **THEN** they receive different jobs or no job, never the same one

### Requirement: The worker's memory is bounded by a cgroup MemoryMax

The systemd unit SHALL bound the worker with `MemoryMax=5G`, and SHALL NOT use
`ulimit -v`, which stops CUDA from initialising on the Jetson. Source:
`worker/mse-worker.service`, `worker/README.md`.

#### Scenario: the unit bounds memory as a cgroup
- **WHEN** the `mse-worker` user unit is inspected
- **THEN** it sets `MemoryMax=5G` and no address-space ulimit

### Requirement: A crashed worker's job is recovered by lease expiry

A worker killed with SIGKILL (including an OS OOM kill) SHALL NOT be expected to
report anything: no Python exception is raised, so neither `/failed` nor
`/release` is called. The app SHALL recover the row by lease expiry, clearing the
lease, returning the row to `queued`, and INCREMENTING `release_count`. Source:
`web/db.py`, `worker/mse_worker.py`.

#### Scenario: a job's lease expires
- **WHEN** a running job's lease passes `lease_expires_at` with `release_count`
  below the bound
- **THEN** the app clears the lease, sets the row back to `queued`, and
  increments `release_count`

#### Scenario: a crash-looping worker must not requeue forever
- **WHEN** a job's lease expires and its `release_count` is already at the bound
  (`MAX_RELEASES`, 3)
- **THEN** the app marks the job FAILED instead of requeuing it
- **AND** the message tells the user the worker repeatedly stopped responding

### Requirement: A job that is the problem is failed, not requeued

The worker SHALL distinguish a transient fault (release, requeue) from a
job-side fault (fail). A `WorkerError` reporting that the worker was asked to
stop SHALL be transient and SHALL release the row. Source:
`worker/mse_worker.py`.

#### Scenario: the worker is asked to stop between operations
- **WHEN** the worker sees SIGTERM between operations
- **THEN** it releases the job as transient and the row is requeued
- **AND** it does not mark the job FAILED

### Requirement: The worker never holds the store token

The worker SHALL use only the app and a `LocalStore`, and SHALL NOT construct an
S3/MinIO client or read `MINIO_*` variables. It SHALL NOT log its API token:
`Config.__repr__` and `WorkerAPI.__repr__` SHALL print a character count, not the
value, and traceback lines SHALL be redacted. Source: `worker/mse_worker.py`.

#### Scenario: the worker's configuration is printed
- **WHEN** a `Config` or `WorkerAPI` is formatted for a log line
- **THEN** the token appears only as a character count

#### Scenario: the leak check on worker source
- **WHEN** the worker source is searched for store construction or `MINIO_`
- **THEN** no S3 store is constructed and no `MINIO_*` variable is read
