# time-domain

## MODIFIED Requirements

### Requirement: A run SHALL be refused above its limits before it is queued

The estimate SHALL be computed from the form. A request SHALL be refused,
before it is queued, when it exceeds the product's step ceiling (20 000 steps,
computed from the run's own dt) or its time ceiling (30 minutes, computed from
the measured Jetson rate for that configuration: boundary kind and whether
dispersive metals are present). The refusal SHALL name the limit, the value
that exceeded it, and the form lever that reduces it.

The cell and memory ceilings SHALL NOT be fixed numbers. They are the budget of
the device the run lands on (a separate requirement, below). The web app runs
on the control plane, not on that device, so it SHALL price the run (bytes,
cells, steps, seconds) and SHALL check only the two limits above, passing
`fdtd.NO_DEVICE`: it SHALL NOT refuse a request for memory or cells, because
any figure it could compare against would be the control plane's, not the
solver's. The device's own check SHALL be applied where the run happens: the
worker, through `fdtd.admission_budget` and `fdtd.check_limits`, refusing a run
that cannot fit that device before it is solved, and deferring one that does
not fit the free memory at that moment.

#### Scenario: an oversized grid is refused readably
- **WHEN** the requested frequency and size give more cells than the budget of
  the device the run lands on allows (the product once capped these at 8 M
  cells, on the page; the page now prices the run and queues it, and the cell
  count is checked where the run happens)
- **THEN** the device-side refusal names the cell count, the ceiling in bytes,
  the figure it came from and the lever, and the server does not respond 500

#### Scenario: too many steps is refused readably
- **WHEN** the requested duration on the grid's dt needs more than 20 000 steps
- **THEN** the page shows the refusal with the step count, the limit and the
  lever, and the server does not respond 500

#### Scenario: too much memory is refused readably
- **WHEN** the estimated memory exceeds the budget of the device the run lands
  on
- **THEN** the refusal names the byte estimate, the ceiling, the figure it came
  from (the GPU's free memory, the container's limit or the system's available
  memory) and the lever, and the server does not respond 500

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
  solver reports, and a duration of a whole number of steps is not rounded up
  by floating-point noise

#### Scenario: an unmeasured configuration is timed at the slower rate
- **WHEN** a run's configuration has no row of its own in the Jetson rate table
  (periodic or mixed faces, or metal walls with a dispersive metal)
- **THEN** it is timed at the slower measured rate (open, or open with gold),
  so the 30-minute limit never admits a run on a guessed faster rate

#### Scenario: the run time errs long
- **WHEN** the run time is estimated
- **THEN** each configuration's rate is the slowest row it measured over every
  recorded Jetson run, kept in the code as cited data rather than retyped
- **AND** the estimate divides by a 0.9 safety factor on that rate, since
  run-to-run spread is about 3% under background load and an under-estimate
  would admit a run over 30 minutes

#### Scenario: every exceeded limit is named at once
- **WHEN** a request exceeds more than one limit
- **THEN** one refusal names each of them, in the order cells, steps, memory,
  time, and a value exactly at a limit is admitted

#### Scenario: the memory estimate is checked against the solver's real peak
- **WHEN** the solver runs a tiny grid with metal, open or periodic faces, a
  conductor mask, the peak-field record or a Lorentz-Drude metal
- **THEN** the host-measured peak is between 0.8 and 1.0 of the estimate (the
  estimate includes the device allocator's cached blocks, which numpy does not
  keep), and each option's measured increment within 15% of its estimated
  increment
- **AND** every row of the Jetson rate table (metal, open and open-plus-gold at
  1 M and 4 M cells) measures between 0.9 and 1.0 of the estimate, so the
  memory refusal never admits a run the device cannot hold
- **AND** a rate script measures cell-steps/s and peak device memory for metal,
  open and open-plus-gold runs at about 1 M and 4 M cells on the Jetson, beside
  the estimate, and the run-time limit's rates come from those measurements

### Requirement: The probes SHALL be priced in every estimate of the run that records them

`fdtd.memory_estimate` and `fdtd.estimate_run` SHALL take `probes` (and
`probe_steps`, defaulting to the run's own step count) and count
`4 x probes x probe_steps` bytes, the same four bytes a float32 trace is,
because that is what the run holds in device memory and what the device's
admission ceiling must hold. `fdtd_layout.estimate` SHALL pass the planned
layout's own probe count, and the worker SHALL pass its layout's, so the
product's memory check prices the traces it will record. The bench script's
ladder SHALL price the same line the run will place, by building it from the
plan through one function both paths call.

#### Scenario: a run's estimate grows by its traces
- **WHEN** `fdtd.estimate_run` is called on the same nodes with `probes=0` and with `probes=n`
- **THEN** the two `bytes` differ by exactly `4 n steps`, and `dt` and `steps` do not change: probes cost memory and no time

#### Scenario: the ladder prices the run's own line
- **WHEN** `--estimate --axis-probes` is read
- **THEN** each rung's bytes include `4 x 3 x layers x steps` for the line that rung's plan would place

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
