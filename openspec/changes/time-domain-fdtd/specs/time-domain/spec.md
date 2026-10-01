# time-domain

## ADDED Requirements

### Requirement: A time-domain run SHALL use a stable Yee time step

The solver SHALL step E and H on a graded tensor-product Yee grid with
dt = 0.99/(c·√(1/dx_min²+1/dy_min²+1/dz_min²)), so it never violates the 3-D
Courant limit of the smallest cell.

#### Scenario: a closed metal box conserves energy
- **WHEN** a pulse is launched in a lossless PEC box and the source has ended
- **THEN** the total electromagnetic energy stays within 1% of its value at
  source end for the rest of the run

### Requirement: The source SHALL be the one the user defined

A `td` request SHALL carry the source kind (`point` | `plane`), the time profile
(`cw` | `pulse` | `drawn`), the frequency, the bandwidth for a pulse, the
polarisation, and the position for `point` or the direction for `plane`. A
`plane` source with boundary `metal` SHALL be refused before it is queued, with
a message that names the conflict. Every source profile, including `drawn`,
SHALL be zero-mean (no DC content), because a soft current with a net integral
leaves a static field that no absorbing boundary removes.

#### Scenario: a pulse sits at the requested frequency
- **WHEN** a `pulse` source with centre frequency f is run
- **THEN** the spectrum of the source's own time profile peaks at f within 1%

#### Scenario: plane wave in a metal cavity is refused
- **WHEN** the form asks for `plane` with boundary `metal`
- **THEN** the request is refused with a readable message and no job is queued

#### Scenario: a source leaves no static field
- **WHEN** any source profile is integrated over the run
- **THEN** its integral is below 1e-3 of the integral of its absolute value

#### Scenario: a CW source settles with no offset
- **WHEN** a `cw` profile's running integral is averaged over whole periods
  after its turn-on ramp
- **THEN** the average is below 1e-4 of that integral's swing (a ramped
  `envelope × sin` alone settles 1% off zero, which a long run's total
  integral does not show; the profile is therefore the exact derivative of
  `-envelope × cos(ωt)/ω`, the same pure sine once the ramp ends)

### Requirement: A metal-box pulse SHALL ring at the box's resonances

A pulse in a closed metal box SHALL excite the box's own resonances and no
others, so the time-domain solver agrees with the analytic cavity and with the
cavity solver.

#### Scenario: benchmark V1
- **WHEN** a broadband pulse is run in an empty rectangular metal box
- **THEN** each spectrum peak in the pulse band is within 1% of an analytic
  rectangular-cavity frequency, AND every analytic mode in the band that the
  source couples to (nonzero source-component field at the source cell)
  appears as a peak; a missing mode fails the benchmark
- **AND** peaks are matched to modes one-to-one
- **AND** the recorded window is at least 4 / (the smallest gap between
  distinct analytic modes in the search band), so neighbouring modes don't merge
  into one peak; if that needs more than the configured maximum record, the
  benchmark refuses and names the pair rather than running
- **AND** the 1% tolerance is widened only to twice the grid's own dispersion
  error, the output says so when it is, and a widened run does not report a
  spec pass

#### Scenario: V1 agrees with the cavity solver
- **WHEN** the V1 box is also solved by the cavity (Nedelec) solver
- **THEN** each V1 peak is within 1% of the matching cavity-solver eigenfrequency,
  for the box the grid actually has, using the mean of a degenerate set
- **AND** if the cavity solver fails its own check or returns fewer modes than
  were chased, the comparison fails instead of comparing fewer rows

### Requirement: An open boundary SHALL absorb outgoing waves

With boundary `open`, a 10-cell CPML SHALL absorb waves leaving the grid, so no
reflection from the grid edge is shown as part of the simulation.

#### Scenario: benchmark V2
- **WHEN** a CW point source runs in open vacuum to steady state
- **THEN** at three or more probe distances r ≥ 2λ from the source, broadside
  to the dipole (perpendicular to its polarisation, where the far field is
  largest), the field amplitude follows 1/r within 5% (amplitude from a
  least-squares fit at the source frequency)
- **AND** a probe that saw nothing, or whose fits over the two halves of the
  recorded window differ by more than 1% (not yet steady), fails the benchmark

#### Scenario: benchmark V3
- **WHEN** a plane-wave pulse crosses empty open space and leaves the grid
- **THEN** three grid transit times after the source ends, the residual field
  energy over the physical region (CPML excluded) is below 1e-4 of the peak
  energy, AND throughout the run the scattered-field region's field stays below
  1e-3 of the incident peak amplitude
- **AND** it is checked for every direction and transverse polarisation on a
  grid graded along the propagation axes, since the 1-D line takes that axis's
  spacings
- **AND** the scattered-field region is judged at every sample of every E
  component, each at its own Yee position against the box faces, from a
  running maximum of |E| rather than from probes
- **AND** a run with no incident wave in the box, or with a non-finite result,
  fails the benchmark

### Requirement: The plane-wave source SHALL be a clean total-field/scattered-field box

A `plane` source SHALL use a TF/SF box with a 1-D auxiliary incident grid that
uses the same dt and the same graded spacings along the propagation axis. The
box SHALL sit at least 3 cells from every object and at least 3 cells from the
CPML. A polarisation parallel to the propagation direction SHALL be refused
before it is queued. The box is given as two inclusive corner nodes; the
incident wave is defined in one background medium, so every cell within 3 of a
face, on either side, SHALL be that medium (objects sit wholly inside the core
or wholly outside), and a box that breaks either clearance SHALL be refused
with a message naming the axis or the offending node. The 1-D line is padded
beyond the 3-D grid and closed by its own CPML, so its own end reflections
never reach the box.

#### Scenario: a plane wave leaves nothing outside its box
- **WHEN** a plane-wave pulse in any of the six directions, with either
  transverse polarisation, crosses an empty box on a graded grid
- **THEN** just outside every face the field stays below 1e-3 of the incident
  peak, AND a dielectric block inside the same box scatters above 1e-2 to the
  same probes

#### Scenario: a box too close to an object or the CPML is refused
- **WHEN** an object, a conductor or the CPML lies within 3 cells of a box face
- **THEN** the run is refused with a message naming the clearance and where it
  is broken

#### Scenario: a longitudinal polarisation is refused
- **WHEN** the form asks for a plane wave polarised along its direction
- **THEN** the request is refused with a readable message and no job is queued

### Requirement: A run SHALL be refused above its limits before it is queued

The estimate SHALL be computed from the form. A request SHALL be refused,
before it is queued, when it exceeds any of: 8 M cells; 20 000 steps; a GPU
memory budget of 3.0 GB (fields, CPML auxiliaries and ADE storage, computed
from the form); or an estimated run time of 30 minutes, computed from the
measured Jetson rate for that configuration (boundary kind and whether
dispersive metals are present). The refusal SHALL name the limit, the value
that exceeded it, and the form lever that reduces it.

#### Scenario: an oversized grid is refused readably
- **WHEN** the requested frequency and size give more than 8 M cells
- **THEN** the page shows the refusal with the cell count, the limit and the
  lever, and the server does not respond 500

#### Scenario: too many steps is refused readably
- **WHEN** the requested duration on the grid's dt needs more than 20 000 steps
- **THEN** the page shows the refusal with the step count, the limit and the
  lever, and the server does not respond 500

#### Scenario: too much memory is refused readably
- **WHEN** the estimated GPU memory exceeds 3.0 GB
- **THEN** the page shows the refusal with the byte estimate, the limit and the
  lever, and the server does not respond 500

### Requirement: The worker SHALL run a time-domain job only with the memory it needs

The worker SHALL run at most one GPU job at a time when one of them is `td`.
Before starting a `td` job it SHALL check free device memory against the job's
estimate; when memory is short it SHALL leave the job queued with a status
saying the Jetson is busy, not fail it. A `td` job SHALL wait at most 60
minutes as "Jetson busy" and then fail with a readable message. A `td` job
SHALL never be routed to the cluster, which cannot run it, by any path that can
change its routing (the unclaimed-GPU fallback, the abandoned-GPU reroute, the
reaper and release). A device out-of-memory error during the run (another team's job
started after the check) SHALL fail the job with a readable message, not a
generic error. A `td` job SHALL be stopped after 45 minutes of run time (the
30-minute estimate limit plus a margin) and fail with a message naming that
limit.

#### Scenario: a busy Jetson delays, not fails
- **WHEN** a `td` job's estimate exceeds the Jetson's free memory at start
- **THEN** the job stays queued with a "Jetson busy" status and runs once
  memory is free

#### Scenario: a Jetson that stays busy fails readably
- **WHEN** free memory stays short for 60 minutes
- **THEN** the job fails with a message saying the Jetson stayed busy, and it was
  never moved to the cluster queue

#### Scenario: memory taken mid-run fails readably
- **WHEN** the device runs out of memory during a `td` run
- **THEN** the job fails with a message naming the memory shortage

#### Scenario: an overlong run is stopped
- **WHEN** a `td` run passes 45 minutes of run time
- **THEN** it is stopped and the job fails with a message naming the limit

### Requirement: Dispersive metals SHALL follow their published model

Gold, silver, aluminium, copper and chromium SHALL be modelled by the
Lorentz-Drude fits of Rakić et al., Applied Optics 37, 5271 (1998), with
ε∞ = 1 and the paper's parameters in eV (ω_p, f_j, Γ_j, ω_j), stepped with
auxiliary differential equations. ADE storage SHALL cover only the metal cells'
bounding box. An E sample on a metal face SHALL take the metal. A run whose
pulse band leaves the paper's fitted range for a metal in the scene SHALL be
refused before it is queued.

#### Scenario: the material table matches the paper
- **WHEN** ε(ω) is computed from the library at three photon energies per metal
- **THEN** it matches an independently published Rakić Lorentz-Drude n, k
  dataset (cited in the test) within 2%

#### Scenario: benchmark V4
- **WHEN** a pulse reflects at normal incidence from a gold slab at least 10
  skin depths thick, with at least 4 cells per skin depth
- **THEN** the reflectance, measured as |scattered/incident|² at a
  scattered-field probe, is within 0.02 (absolute) of the analytic Fresnel
  reflectance for the same model, over the band where the incident spectrum is
  at least 1e-2 of its peak

### Requirement: Thin layers SHALL be resolved, not lost

The grid SHALL place at least 4 cells across every layer thinner than the
coarse cell, or across the metal skin depth where that is thinner, with a
neighbour spacing ratio of at most 1.3.

#### Scenario: the grid builder keeps its own rules
- **WHEN** the grid builder is given layers thinner than the coarse cell
- **THEN** every such layer has at least 4 cells across it and every neighbour
  spacing ratio is at most 1.3

#### Scenario: benchmark V5
- **WHEN** a dielectric film of index n, far thinner than the coarse cell,
  fills the gap between two PEC plates of lateral size a × b with PEC side walls,
  driven by a source with E normal to the plates
- **THEN** its resonances are within 1% of the parallel-plate lateral modes
  f = (c/2n)·√((m/a)² + (p/b)²) with m, p ≥ 1, which depend on n and not on the
  film thickness, and every such mode in the pulse band appears

### Requirement: A drawn waveform SHALL be checked against the grid

A `drawn` source SHALL be a list of at most 10 000 dimensionless amplitude
samples with a duration in seconds. It SHALL be resampled to dt and then its
mean SHALL be removed (the form says so), and it is zero after its duration; the
run's own duration sets the step count. The grid's resolvable frequency is
f_res = c / (10·|n|_max·Δ_max), where Δ_max is the largest cell spacing. A
drawn waveform with more than 1% of its spectral energy above f_res SHALL be
refused before it is queued, naming f_res.

#### Scenario: an under-resolved drawing is refused before queueing
- **WHEN** a drawn waveform has a sharp edge putting more than 1% of its
  spectral energy above f_res
- **THEN** the form shows the refusal, naming f_res, and no job is queued

### Requirement: The scene's physical size SHALL come from the owner's units and scale

A `td` request SHALL take its physical size from the same `units` and `scale`
form fields the cavity tool uses (size × scale × unit), and the estimate SHALL
be computed from that physical size.

#### Scenario: a 10 nm flat film in a 1 µm scene fits
- **WHEN** a 1 µm scene holding an axis-aligned 10 nm gold film is requested at a
  visible-band pulse
- **THEN** the estimate is below 8 M cells AND below 20 000 steps on the graded
  grid, and the run is accepted

#### Scenario: a curved thin shell is estimated, not assumed
- **WHEN** a curved object of 1 µm with a 10 nm coating is requested
- **THEN** the estimate reports the cell count the graded grid actually needs, and
  a request above the cap is refused naming that count

### Requirement: A periodic cell SHALL reproduce the crystal's bands

A `td` request MAY declare periodic faces with a Bloch wave vector k. The run
SHALL then model an infinite crystal of that cell. Except at k = 0 and the zone
edge, the cos and sin fields SHALL be stepped together with cross-coupled face
updates, through a face hook in the core.

#### Scenario: benchmark V6
- **WHEN** a pulse is run in one cell of a 2-D crystal extruded along z, with
  Bloch-periodic x and y faces at a chosen k, periodic z faces with kz = 0, for
  each polarisation (TM: E along z; TE: H along z)
- **THEN** the spectrum's peaks are within 1% of the photonic-crystal band
  solver's frequencies at that k for that polarisation, and every band in the
  pulse band appears
