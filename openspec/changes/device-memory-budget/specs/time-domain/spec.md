# time-domain

## MODIFIED Requirements

### Requirement: A run SHALL be refused above its limits before it is queued

The estimate SHALL be computed from the form. A request SHALL be refused,
before it is queued, when it exceeds the product's step ceiling (20 000 steps,
computed from the run's own dt) or its time ceiling (30 minutes, computed from
the measured Jetson rate for that configuration: boundary kind and whether
dispersive metals are present). The refusal SHALL name the limit, the value
that exceeded it, and the form lever that reduces it.

The memory ceiling SHALL NOT be a fixed number. The web app runs on the control
plane, not on the device, so it SHALL check only the limits above and SHALL pass
`fdtd.NO_DEVICE`: the page prices the run (bytes, steps, seconds) and says the
device decides its memory. The device's own check SHALL be applied by the
worker, which runs on the device, through `fdtd.admission_budget` and
`fdtd.check_limits`.

#### Scenario: too many steps is refused readably
- **WHEN** the requested duration on the grid's dt needs more than 20 000 steps
- **THEN** the page shows the refusal with the step count, the limit and the
  lever, and the server does not respond 500

#### Scenario: the page prices memory without refusing on it
- **WHEN** the form's estimate for a request is larger than any fixed ceiling
  the product once had (for example the 11.55 GB paper-1 dx-5 nm record)
- **THEN** the page prices the run -- its estimate, which the movie-window and
  step checks read -- and admits the request to the queue without comparing the
  byte figure to a ceiling of its own, because the only honest ceiling is the
  device's and it is read on the device

#### Scenario: the estimate counts the steps the run will take
- **WHEN** a run's dt is lowered below 0.99 of Courant by a dispersive metal
- **THEN** the estimate's step count uses that lowered dt, the same dt the
  run takes

## ADDED Requirements

### Requirement: The memory and cell ceilings SHALL follow the device the run lands on

`fdtd.admission_budget(xp=None)` SHALL return the admission ceiling of the device
the calling process runs on: when `xp` is cupy, the device's own memory
(`cupy.cuda.Device().mem_info`) plus the bytes cupy's pool already holds; else
the cgroup memory limit when one is set (`/sys/fs/cgroup/memory.max`, or v1's
`memory/memory.limit_in_bytes`), because that is what the run may hold even when
the host has more; else the kernel's `MemAvailable`. The ceiling SHALL be the
binding figure times `fdtd.ADMISSION_MARGIN` (0.9), and the budget SHALL carry
that figure, its source and its free and capacity values, so a refusal can name
all three.

`fdtd.check_limits(estimate, *, max_steps=None, max_seconds=None, max_cells=None,
max_bytes=None, budget=None)` SHALL take the budget: `budget=None` SHALL read
`admission_budget()` of the calling process at call time, `budget=fdtd.NO_DEVICE`
SHALL mean the caller is not on the run's device and SHALL check only steps and
seconds, and an explicit `max_bytes` SHALL override the budget. A memory refusal
SHALL name the byte estimate, the ceiling, the source of the ceiling and the
margin that was applied, and the form lever that reduces it. The cell ceiling
SHALL follow the budget instead of being an independent number: with no explicit
`max_cells` it SHALL be the most cells that fit `max_bytes` at the cheapest
configuration's per-cell cost, and that cost SHALL be measured from
`memory_estimate`'s own accounting -- the difference between two grids that
differ only in their cell count -- not retyped from its formula.

`fdtd.MAX_CELLS` and `fdtd.MAX_BYTES` SHALL be removed. `fdtd.MAX_STEPS` and
`fdtd.MAX_SECONDS` SHALL keep their values and their meaning.

#### Scenario: the A4000's own memory admits the record the fixed ceiling refused
- **WHEN** `admission_budget` is given a device reporting 16 GB total and 14.5 GB
  free, with no cgroup limit and cupy's pool empty
- **THEN** its ceiling is 13.05 GB (14.5 GB x 0.9), and `check_limits` admits the
  paper-1 dx-5 nm record's 11.55 GB estimate that the 3.0 GB constant refused

#### Scenario: the container's limit is the ceiling on the Jetson's host memory
- **WHEN** the process has no GPU and its cgroup limit is 14 GiB, while the host
  reports 25 GB available
- **THEN** the budget's capacity is the 14 GiB limit and its ceiling is
  min(limit, available) x 0.9, with the limit named as the source, because a run
  may hold the container's memory and not the host's

#### Scenario: the refusal names the ceiling, its source and the margin
- **WHEN** an estimate of 14.0 GB is checked against the budget of the scenario
  above
- **THEN** the `EstimateError` names 14.00 GB, the ceiling (13.5 GB: the 14 GiB
  limit is 15.03 GB, x 0.9), the source (this container's memory limit) and the
  0.9 margin, and keeps naming the form lever

#### Scenario: the cell ceiling follows the budget and can still fire
- **WHEN** an estimate is checked whose `bytes` is under the ceiling but whose
  `cells` x the cheapest per-cell cost is over the same ceiling, as an estimate
  whose two figures disagree would be
- **THEN** the refusal names the cell ceiling the budget implies, so the check
  catches an estimator that prices fewer bytes than its own cell count can hold,
  rather than a second opinion about the machine

#### Scenario: the worker defers a run that does not fit now
- **WHEN** a `td` job's estimate fits the device's capacity budget but exceeds
  the device's free memory at admission, times the same margin
- **THEN** the worker leaves the job queued with its busy status instead of
  starting it against a full device, and a job over the capacity budget is
  refused with the estimate's own message

## REMOVED Requirements

### Requirement: The cell ceiling SHALL be a per-run argument, raisable by the bench script alone

**Reason**: the 8,000,000-cell constant describes no device; the ceiling now
follows the device's memory budget (above).

**Migration**: `--max-cells N` on the paper-1 bench script still sets an explicit
cell ceiling; with no flag the bench run is admitted against the device's budget.

### Requirement: The memory ceiling SHALL be a per-run argument, raisable only by the bench script

**Reason**: the 3.0e9-byte constant refused the paper-1 dx-5 nm record (11.55 GB)
that the bench's own 16 GB card holds; the ceiling now comes from the device.

**Migration**: `--max-bytes GB` on the paper-1 bench script still sets an explicit
byte ceiling; with no flag the bench run is admitted against the device's budget.
