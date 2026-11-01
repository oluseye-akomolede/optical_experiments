## MODIFIED Requirements

### Requirement: A pc job SHALL be routed to the Jetson worker, and the cluster SHALL refuse it

The pc solver SHALL run on the Jetson worker only. A pc job SHALL therefore be
created with `target = jetson`, so the worker that can solve it is the worker
that claims it. The cluster queue SHALL NOT hand a pc row to the launcher: the
launcher builds exactly two Job shapes, the cavity solver for `cavity` and the
electrical solver for `elec` (`electrical-jobs`), chosen by the row's `kind`,
and has no pc shape, so a pc row reaching it would be solved as the wrong kind
and the row's `kind` would be lost. The cluster side
SHALL refuse a pc row by NAMING its kind, at both the queue and the launcher, so
a routing mistake is a loud failure rather than a band diagram that never
appears.

#### Scenario: the pc upload creates the row for the Jetson, not the cluster
- **WHEN** a valid pc form is uploaded
- **THEN** the created row's `target` is `jetson`
- **AND** its `kind` is `pc`
- **AND** the cluster worker never sees it

#### Scenario: the cluster queue skips a pc row
- **WHEN** a pc row with `target = cluster` is queued (a row that should not
  exist, but which the queue must not solve if it does)
- **THEN** `next_queued(cluster)` does not return it
- **AND** a cavity row and an `elec` row for the cluster are still returned, so
  the filter removes only pc rows

#### Scenario: a pc row past the GPU wait stays on the GPU queue
- **WHEN** a pc row has waited longer than `MSE_GPU_WAIT_S` for a worker, and
  the GPU-abandon path runs over it
- **THEN** its `target` is still `jetson`, and `claim_jetson` still returns it
- **AND** it is not moved to the cluster, so it is never stranded with no queue
  that can see it
- **AND** the row does not tell the owner it was "moved to the cluster queue",
  because nothing moved
- **AND** `reroute_to_cluster` refuses a non-cavity row in its own SQL guard
  (`kind = 'cavity'`), so no caller can move one, not merely the one caller
  that skips pc rows today

#### Scenario: the cluster launcher refuses a pc row by name
- **WHEN** the cluster launcher is handed a job whose `kind` is `pc`
- **THEN** it raises, and the refusal names the kind and the Jetson
- **AND** it creates neither a cavity Job nor an electrical Job for it

#### Scenario: the crossing is proven on the real store, not a double
- **WHEN** a pc row is created through the real upload route's store call on
  real Postgres
- **THEN** `claim_jetson` returns it
- **AND** `next_queued(cluster)` does not
- **AND** the check is red against the unfixed branch, failing as
  `claim_jetson -> None` because the row was created `target = cluster`
