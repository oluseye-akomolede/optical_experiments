# photonic-bands

## ADDED Requirements

### Requirement: A photonic-crystal request SHALL be described by exactly the fields it is solved with

The form SHALL supply `lattice` (`square` | `triangular`), `r_over_a`,
`n_rod`, `n_bg`, `n_bands`, `divisions` and `polarisation` (`TM` | `TE`).
`divisions` SHALL be part of the stored request even though the plan's §5 lists
only the first four: it changes the mesh, so a row that omitted it could not say
what was solved. `k_points` SHALL NOT be a form field: the path is derived from
the lattice, so the owner cannot ask for a path that is not the
irreducible-zone path. The row's `pcs` JSON SHALL carry exactly the keys the
worker reads (`web/pc_form.request_json`), so no field is re-derived at solve
time from a default the owner never saw.

#### Scenario: the stored request round-trips to the worker
- **WHEN** a valid pc form is planned and enqueued
- **THEN** the `pcs` column's keys are exactly `lattice`, `r_over_a`, `n_rod`,
  `n_bg`, `n_bands`, `divisions`, `polarisation`
- **AND** the solver consumes the row's own values, not a default

#### Scenario: a missing request field is refused by name, not silently defaulted
- **WHEN** the worker is handed a request missing a key it needs
- **THEN** it refuses naming the missing key
- **AND** it does not substitute a default

### Requirement: Every refusal SHALL name its own field or limit

Validation SHALL raise an error the owner can act on, and each refusal SHALL
name the offending value and the boundary it crossed. No refusal may be a bare
"too big", because a summed message cannot say which lever to pull.

#### Scenario: each request field is refused naming itself
- **WHEN** `lattice` is not one of the two choices
- **THEN** the refusal names the value and lists the choices
- **WHEN** `polarisation` is not `TM` or `TE`
- **THEN** the refusal names the value, lists the choices, and states that TM is
  `E_z` and TE is `H_z`
- **WHEN** `r_over_a` is not strictly between 0 and 0.5
- **THEN** the refusal states the range and the value
- **WHEN** either refractive index is not positive
- **THEN** the refusal names WHICH index (rod or background) and the value
- **WHEN** `n_bands` is below 1 or above the limit
- **THEN** the refusal states the bound and the value
- **WHEN** `divisions` is below the solver's floor of 4, or above the limit
- **THEN** the refusal names the bound; the floor SHALL NOT be clamped, because a
  silent bump to 4 solves a different problem than the one asked for

#### Scenario: the two caps are refused separately, each with its own number
- **WHEN** the derived k-point count exceeds `MAX_K_POINTS`
- **THEN** the refusal says the path's k-point count and that limit, and names
  reducing points per segment
- **AND** it does not mention the node cap
- **WHEN** the derived node count exceeds `MAX_NODES`
- **THEN** the refusal says the node count and that limit, and names reducing
  divisions
- **AND** it does not mention the k-point cap
- **AND** each cap can be tripped alone and only its own refusal text changes

### Requirement: The unit cell SHALL be meshed conformally, with periodic pairs by coordinate

The cell SHALL be triangulated with `r/a` sampled on the grid so that no
triangle straddles the rod interface (a triangle is inside iff all its vertices
are inside, outside iff all are outside). The circle SHALL be discretised finely
enough that no grid point can steal a chord from it: arc spacing SHALL NOT
exceed the grid spacing. Periodic boundary nodes SHALL be paired BY COORDINATE
under the two lattice translations (a slave whose master is the same point
translated by `a1` or `a2`), never by index, so a change in vertex ordering
cannot silently rewire the periodicity.

#### Scenario: no triangle straddles the rod interface
- **WHEN** the cell is meshed at any admissible `r/a`
- **THEN** every triangle is wholly inside or wholly outside the rod
- **AND** the eps of a triangle is `n_rod^2` or `n_bg^2`, never mixed

#### Scenario: the conforming circle does not move the pairing
- **WHEN** the rod mesh is refined to the conforming circle
- **THEN** the periodic pairs are the same set the coarse mesh produced

#### Scenario: every boundary node is paired exactly once
- **WHEN** the cell is meshed at a given number of divisions
- **THEN** each node on the periodic boundary is a slave exactly once
- **AND** the corner node carries both translations `(1, 1)`
- **AND** the number of pairs matches the boundary set, not a hard-coded count

### Requirement: The mesh SHALL support rods and holes, and their fills

The unit cell SHALL support a circular rod of `n_rod` in a background of `n_bg`,
and the inverse (a hole), by the `geometry` and `fill` parameters. A stripe
geometry SHALL exist solely for the 1-D oracle, and SHALL snap its fill to a
grid column that stays strictly inside the cell, never straddling the interface.

#### Scenario: the rod gets n_rod and the background gets n_bg
- **WHEN** a rod cell is meshed
- **THEN** the triangles whose centroid is inside the circle have
  `eps = n_rod^2`
- **AND** those outside have `eps = n_bg^2`
- **AND** both regions are non-empty

#### Scenario: a stripe snaps to a grid column and does not straddle
- **WHEN** a stripe cell is meshed at a fill between grid columns
- **THEN** the built fill is snapped to the nearest grid column
- **AND** no triangle straddles the stripe interface

### Requirement: The two polarisations SHALL solve the two distinct scalar operators

The lattice SHALL be in x–y with z out of plane. `TM` (named for `E_z`) SHALL
solve `-lap(psi) = (w/c)^2 eps psi` — plain stiffness, eps on the MASS. `TE`
(named for `H_z`) SHALL solve `-div((1/eps) grad psi) = (w/c)^2 psi` — `1/eps` on
the STIFFNESS, plain mass. The mapping SHALL be recorded as the geometric fact
`z-hat . x-hat = 0`, not as a convention, because the PhC names the polarisation
for the out-of-plane field while the 1-D layer problem names it for the
interface normal, and mixing them is the feature's easiest error.

#### Scenario: the weight parameter separates the two operators
- **WHEN** the TM operator is assembled
- **THEN** eps appears on the mass and stiffness is plain
- **WHEN** the TE operator is assembled
- **THEN** `1/eps` appears on the stiffness and mass is plain

#### Scenario: the two operators coincide only when eps is uniform
- **WHEN** eps is uniform
- **THEN** the TM and TE spectra coincide
- **WHEN** eps is non-uniform and the k-point has a non-zero in-plane component
- **THEN** TM and TE differ

### Requirement: The k-path SHALL be the irreducible-zone path, derived from the basis

The square lattice SHALL walk `Gamma-X-M-Gamma`; the triangular lattice SHALL
walk `Gamma-M-K-Gamma`. Points SHALL be returned in Cartesian `2 pi / a` units
with shared endpoints counted exactly once, the path starting and closing at
Gamma. The triangular corners SHALL be the first-Brillouin-zone vertices nearest
Gamma, derived from the reciprocal basis and asserted against their closed forms
(`|M| = 2 pi / (sqrt3 a)`, `|K| = 4 pi / (3a)`), NOT taken from the textbook
`K = (b1 + b2)/3`, which is correct only for one basis orientation and gives
`|K| = 2.42` here against the physical `4.19`.

#### Scenario: the path reaches the high-symmetry points
- **WHEN** the square path is built with `n` points per segment
- **THEN** the labels are Gamma, X, M, Gamma
- **AND** sample `n` is `[0.5, 0]` (X), sample `2n` is `[0.5, 0.5]` (M)

#### Scenario: the triangular corners are the physical corners, not the naive formula
- **WHEN** the triangular path is built
- **THEN** `|M| = 2 pi / sqrt(3)` and `|K| = 4 pi / 3` to 1e-12
- **AND** the naive `(b1 + b2)/3` norm differs from `4 pi / 3` by more than 1e-3,
  so the assertion discriminates

### Requirement: The Bloch condition SHALL be a periodic-pair fold with a phase, solved by sparse shift-invert

The periodic slave nodes SHALL be folded onto their masters carrying the Bloch
phase `exp(i k . T)`, producing a complex Hermitian generalised eigenproblem per
k-point. The lowest `n_bands` frequencies SHALL be found by sparse shift-invert
`eigsh` at a NEGATIVE shift, because at Gamma the lowest eigenvalue is exactly 0
and a zero shift is singular. Frequencies SHALL be returned as the dimensionless
`w a / (2 pi c)`.

#### Scenario: the eigenvalue-to-frequency conversion is consistent with the phase
- **WHEN** bands are computed for an empty lattice
- **THEN** the lowest band at Gamma is 0 to 1e-10
- **AND** the second band at Gamma is 1.0 to 1e-2 (the free-photon value)
- **AND** a value near `2 pi` there indicates the Bloch phase dropped its `2 pi`
  factor or the band output skipped its `/2 pi` — the two omissions cancel on
  band 1 at X and M, so only this landmark detects them

### Requirement: The result SHALL be k-path labels, k-points and bands, and the gaps SHALL be computed from the arrays

The payload written to the job SHALL be JSON-ready, carrying
`labels` (the path), `k_points` (per point, in `2 pi / a`), and `bands` (one
vector of `n_bands` per point, in `w a / (2 pi c)`). Band gaps SHALL be computed
by the pure module `web/static/bands.js` from the arrays, never read off the
drawn plot: for each adjacent band pair, a gap exists where band `n`'s maximum
over all k sits below band `n+1`'s minimum over all k. The gap SHALL report
`lower`, `upper`, `midgapRatio` (`dw / w_midgap`, the standard figure of merit)
and `index` (the lower band's 1-based number). Absence of a gap SHALL be
reported as an empty array — an honest absence the caller must state — never as
a fabricated window.

#### Scenario: a gap is computed from the whole path, not one k-point
- **WHEN** `bandGaps` is given a payload whose bands cross
- **THEN** no gap is reported for the crossing pair
- **WHEN** band n is everywhere below band n+1
- **THEN** the gap's `lower` is band n's maximum over k and `upper` is band n+1's
  minimum over k
- **AND** `midgapRatio` is `(upper - lower) / ((upper + lower) / 2)`

#### Scenario: an empty diagram reports no gaps rather than a fake one
- **WHEN** the payload has fewer than two bands, or no band pair separates
- **THEN** `bandGaps` returns an empty array
- **AND** the UI says there is no gap rather than drawing an empty band

#### Scenario: the x-axis is path length, and a flat band does not break the range
- **WHEN** the diagram is drawn
- **THEN** the x-axis is cumulative path length, not the k-index
- **AND** a flat band (min === max, as the empty lattice produces) yields a
  finite y-range rather than a degenerate one

### Requirement: A photonic job SHALL reuse the cavity job path, discriminated by kind

A pc job SHALL live in the same `mse.jobs` table as a cavity job, told apart by a
`kind` discriminator (`cavity` | `pc`), and SHALL move through the same claim →
solve → store path. A cavity job handed to the pc solver (or the reverse) SHALL
be refused, not solved, and the refusal SHALL distinguish "the wrong kind" from
"a valid kind with an empty request".

#### Scenario: the pc solver refuses a cavity job
- **WHEN** a job whose `kind` is `cavity` is handed to the pc solver
- **THEN** it is refused as the wrong kind
- **AND** this is distinct from a pc job whose `pcs` request is NULL

#### Scenario: a pc job is solved with every request field it carries
- **WHEN** a valid pc job is solved
- **THEN** every request field the row carries is used
- **AND** the k-path comes from the lattice, not from a value on the row

### Requirement: A pc job SHALL be routed to the Jetson worker, and the cluster SHALL refuse it

The pc solver SHALL run on the Jetson worker only. A pc job SHALL therefore be
created with `target = jetson`, so the worker that can solve it is the worker
that claims it. The cluster queue SHALL NOT hand a pc row to the launcher: the
launcher builds exactly one Job shape, the cavity solver, so a pc row reaching
it would be cavity-solved and the row's `kind` would be lost. The cluster side
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
- **AND** a cavity row for the cluster is still returned, so the filter removes
  only pc rows

#### Scenario: the cluster launcher refuses a pc row by name
- **WHEN** the cluster launcher is handed a job whose `kind` is `pc`
- **THEN** it raises, and the refusal names the kind and the Jetson
- **AND** it does not create a cavity Job for it

#### Scenario: the crossing is proven on the real store, not a double
- **WHEN** a pc row is created through the real upload route's store call on
  real Postgres
- **THEN** `claim_jetson` returns it
- **AND** `next_queued(cluster)` does not
- **AND** the check is red against the unfixed branch, failing as
  `claim_jetson -> None` because the row was created `target = cluster`

### Requirement: The validation controls SHALL be analytic or published, stated with their tolerances before running

Control A, the empty lattice (`n_rod = n_bg = 1`), SHALL reproduce the folded
free-space light line: `lam = 0` at Gamma is the one exact landmark on a P1 mesh
(asserted at 1e-10); X and M (0.5 and `sqrt2/2`) SHALL be asserted as a RELATIVE
bound of 1e-3 at the production mesh, with the convergence order measured in
[1.8, 2.2] — asserting equality at X and M would assert something false, since
the scheme is second order. Control B, the 1-D layered cell, SHALL match the
exact transfer-matrix band edges, and the oracle SHALL be validated against an
independent ODE shooting solve BEFORE any mesh is compared to it. Control C, a
published benchmark, SHALL be a NUMERIC-vs-NUMERIC comparison against a printed
result whose source was read, with the tolerance stated before the run. Every
dispersion claim SHALL be measured against one of these.

#### Scenario: Control A's exact landmark and its relative bounds
- **WHEN** the empty lattice is solved at the production mesh
- **THEN** the lowest band at Gamma is 0 to 1e-10
- **AND** X and M are within a relative 1e-3 of 0.5 and `sqrt2/2`
- **AND** the second band at Gamma is within 1e-2 of 1.0

#### Scenario: Control A's convergence discriminates a wrong operator
- **WHEN** the mesh is refined h → h/2 → h/4 at M
- **THEN** the error decreases monotonically
- **AND** the observed order `log2(err(h)/err(h/2))` lies in [1.8, 2.2]

#### Scenario: Control B's oracle is validated on its own before any mesh
- **WHEN** the transfer-matrix oracle is compared to an ODE shooting solve
- **THEN** the worst relative difference is below 1e-10
- **AND** a deliberately wrong admittance fails that same check
- **AND** the s and p polarisations coincide at eps = 1 and differ at eps = 4

#### Scenario: Control C reproduces a published benchmark gap
- **WHEN** the MPB tutorial's triangular rod lattice (TM, `r = 0.2 a`,
  `eps = 12`, rods in air, resolution 32) is solved
- **THEN** its TM 1–2 gap, taken as `bandGaps` takes it (max of band 1 over k,
  min of band 2), matches the tutorial's printed
  `0.275065617068082 → 0.446289918847647` (47.47%) to within 1% of the midgap
- **AND** the source is the MPB Scheme tutorial's "Triangular Lattice" section,
  read directly, and the tolerance is stated before the run
- **AND** the benchmark can FAIL: `r = 0.3` moves the gap centre outside that bar

### Requirement: A benchmark's quoted figures SHALL come from a source that was read

A published number used as a control SHALL be taken from a source that was
actually read in the work, and the change SHALL name it. A figure carried from
memory, or from a host that could not be reached, SHALL NOT be stated as a
target.

#### Scenario: an unreachable or remembered figure is not a benchmark
- **WHEN** a candidate benchmark's source cannot be read, or the figure is from
  memory
- **THEN** that figure is not used as a control, and the omission is recorded
  rather than quietly retargeted

## Out of scope

3-D slabs, dispersion and loss, defects and waveguides, non-circular rods,
radius optimisation, complex-ω quality factors, and finite-height
effective-index slabs (roadmap §6). `r/a` beyond where the circles still fit is
a refusal, not a clamp.
