# job-lifecycle delta

## MODIFIED Requirements

### Requirement: Failure reaches the user through `message`

The status route SHALL return 200 with the row's state regardless of state, and
SHALL carry the failure in `message`, not a separate failure body. Failure
messages SHALL be plain English and SHALL NOT interpolate the internal reason
token. Source: `web/app.py`, `web/jobs.py`.

For a GPU job on the cluster's A4000s, a failure that names an out-of-memory
condition SHALL reach the user as the same plain sentence the Jetson path uses
-- that the model did not fit the GPU -- and SHALL NOT become the generic
"contact support" text.

#### Scenario: a cluster job is OOM-killed
- **WHEN** a cluster Job's failure reason is `OOMKilled`
- **THEN** the message asks the user to try a simpler object or fewer modes

#### Scenario: a GPU job runs out of card memory
- **WHEN** a GPU Job's failure output names an out-of-memory condition
- **THEN** the message says the model did not fit the GPU
- **AND** it is not the "contact support" text

#### Scenario: the reason is unknown
- **WHEN** the failure reason is not recognised
- **THEN** the generic plain-English message is used

#### Scenario: a completed job reports its modes
- **WHEN** the status route is asked for a `done` job
- **THEN** the payload includes the job's modes
