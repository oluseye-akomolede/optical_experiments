# time-domain

## ADDED Requirements

### Requirement: A time-domain run SHALL use a stable Yee time step

The solver SHALL step E and H on a graded tensor-product Yee grid with
dt = 0.99/(c·√(1/dx_min²+1/dy_min²+1/dz_min²)), so it never violates the 3-D
Courant limit of the smallest cell. When a dispersive metal is in the scene,
dt SHALL be reduced to 0.98 of the largest step at which the metal's update is
stable (von Neumann on a homogeneous metal with the finest spacings), and the
run SHALL report the dt it used.

#### Scenario: a metal shrinks dt only as far as it must
- **WHEN** 5 nm gold is stepped at 0.99 of the Courant dt
- **THEN** the stability check finds a growing mode (factor above 1.1) and the
  run uses a dt between 0.9 and 0.99 of the Courant one
- **AND** 2 nm gold runs at the Courant dt unchanged
- **AND** with no metal, the same check reads 1.01 of the Courant dt as
  growing, so it can see an unstable step

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

#### Scenario: the layer's reflection is measured, not assumed
- **WHEN** a pulse meets the 10-cell layer, measured on the exact 1-D reduction
  of the solver's update with the solver's own layer coefficients
  (`scripts/fdtd_cpml_reflection.py`)
- **THEN** the reflected amplitude is below 1e-3 of the incident at normal
  incidence, and below 5e-3 from 0 to 75 degrees at 10 to 40 cells per
  wavelength
- **AND** the same measurement with no layer (a bare wall) reads a reflection
  above 0.5, so an instrument that sees nothing cannot pass the layer

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

### Requirement: Periodic faces at k = 0 SHALL repeat the cell

The solver SHALL take a boundary per axis, `(bx, by, bz)`, each `metal`, `open`
or `periodic` (a single kind still applies to all three). A `periodic` axis
models the cell repeated forever along it with k = 0: the curl differences wrap
from the last sample to the first. Node `n-1` of a periodic axis is the same
point as node 0, one period on: the period is the sum of the axis's `n-1`
spacings, the wrap's dual width is the mean of the last and first primal
spacings, and `eps_r`, the conductor mask and every metal region SHALL hold the
same material on those two node slabs, or the run is refused naming the axis.
A point source on node 0 of a periodic axis is refused, naming node `n-1`. The
slot at index `n-1` of a half-position sample along a periodic axis is a ghost:
it is never written for E and never reported as a field value. Per-axis
boundaries are a solver API; the form keeps `metal | open` until phase 6, and a
Bloch phase (k ≠ 0) is phase 6.

#### Scenario: the wrap is the cell repeated
- **WHEN** a point source on uniformly spaced periodic x and y axes is moved by
  a shift that crosses the seam on both axes
- **THEN** every E component's peak field is the unshifted one rolled by that
  shift, within 1e-6 of the peak, where the field varies across the period by
  more than 1e-2 of the peak
- **AND** an off-by-one in the wrap (the wrong source slab, the wrong copy, or
  a periodic axis updated only over its interior) fails it

#### Scenario: a closed periodic box conserves energy
- **WHEN** a pulse runs in a box periodic on all three axes, graded on each
- **THEN** after the source ends the conserved energy varies by less than 1e-5
  of its mean, and a wrong wrap width fails it

#### Scenario: a plane wave across periodic faces
- **WHEN** a plane wave travels along an `open` axis with both other axes
  `periodic` and its box spanning them (node 0 to node `n-1`)
- **THEN** every E component's peak is uniform across the periodic axes within
  1e-6 of the incident peak, every ghost slot reads 0, and the field outside
  the two injecting faces stays below 1e-3 of the incident
- **AND** benchmark V3 passes in that setup
  (`scripts/fdtd_bench_v3.py --periodic-transverse N`)
- **AND** a plane wave is refused with a message when its travel axis is not
  `open`, when its two transverse axes are not both `open` or both
  `periodic`, when any axis is `metal`, or when its box does not span a
  periodic axis

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
never reach the box. With both transverse axes periodic, the box spans them,
the clearance applies along the travel axis only, and the wave enters through
the two faces normal to its direction.

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

### Requirement: A time-domain job SHALL be stored as kind `td` with a keyed request

A `td` job SHALL be stored in the jobs table with `kind = 'td'` and its request
in a `tds` JSON column, NULL for every other kind. The request keys SHALL be
exactly the worker's `REQUEST_KEYS` list (`boundary`, `source_kind`, `profile`,
`frequency_hz`, `bandwidth`, `position`, `direction`, `polarisation`,
`samples`, `sample_duration_s`, `duration_s`, `probes`, `dx_m`, `coatings`).
Every key SHALL be present; a key that does not apply to the chosen source holds null. The
worker SHALL refuse a request with a missing key, naming every missing key, and
SHALL refuse a `tds` that is not an object. Keys outside the list SHALL be
ignored, so results written back to the row are never read as input. The TF/SF
box is derived from the geometry, not stored. The claim payload SHALL always
carry `tds` (null for other kinds). Creating a job with a kind outside the known
kinds (`cavity`, `pc`, `td`) SHALL be refused before any row is written. The
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
  nothing, and a `td` job is reported as "solving", never "meshing"
- **AND** a kind outside the worker's list fails terminally with a message
  naming it, before any input is fetched, and is never solved as a cavity
- **AND** the worker's kind list equals the app's

### Requirement: The upload page SHALL offer a time-domain arm that posts exactly what the form reads

The upload page's mode selector SHALL offer a time-domain choice. Its arm SHALL
show the shared upload (file, units, scale) and the td fields the form reads,
with the same names and choices; fields the chosen source or profile does not
use SHALL be hidden. Only fields marked required SHALL ever block a submission,
and a control that is hidden SHALL be neither required nor posted.

#### Scenario: names and choices match the parser
- **WHEN** the page is rendered
- **THEN** its td control names are exactly the ones `web/td_form.py` reads, and
  each choice list equals the tuple the form checks it against

#### Scenario: the page's own defaults are accepted
- **WHEN** the td arm is submitted with its default values, a one-part upload,
  a source position and a probe
- **THEN** the job is queued

#### Scenario: optional fields never block, hidden fields never post
- **WHEN** a cavity upload is submitted with the optional scale left blank, or a
  file is chosen and the crystal mode is then picked
- **THEN** the browser accepts the cavity form, and the crystal form posts no file

### Requirement: Each part of a multi-part time-domain upload SHALL get its own material

The owner's own example, a thin film between two conducting plates, is three
parts with three materials. When a time-domain upload has two or more parts,
the page SHALL ask the app's part parser (`/parts`) for the part names and show
one material picker per part: a refractive index, `pec`, or one of the
dispersive metals. Each picker posts `part_index.<part name>`. One material
SHALL never be applied to every part on the owner's behalf. A missing, unknown
or malformed material SHALL be refused before queueing, naming the part.
Each part MAY also be coated (next requirement).

#### Scenario: a picker per part, named from the parser
- **WHEN** a three-part .obj is chosen with the time-domain arm selected
- **THEN** the page shows three pickers, one per part name `/parts` returned,
  posting `part_index.<name>`, and the single refractive-index field is hidden

#### Scenario: the posted table reaches the run
- **WHEN** the owner picks `Au`, `1.5` and `pec` for three parts and submits
- **THEN** the queued request's materials table maps each part to that choice

#### Scenario: a missing or unknown material is refused naming the part
- **WHEN** a part has no material, a material names a part not in the upload,
  or a material is neither an index of 1 or more nor a known metal or `pec`
- **THEN** the upload is refused before queueing, and the message names the part

#### Scenario: no per-part pickers, no multi-part run
- **WHEN** a multi-part time-domain upload arrives with no materials
- **THEN** it is refused in plain words naming its parts, and nothing is queued

#### Scenario: `/parts` names a td scene the way the td POST cuts it
- **WHEN** the time-domain arm asks `/parts` for a scene whose parts are not
  nested (a film between two plates)
- **THEN** it returns the names the td POST's own split gives, not the cavity
  parser's refusal that the first part must enclose the rest

### Requirement: A part MAY be coated, and the coating SHALL lie on the grid as a shell over the part

A coating is a material (a refractive index, `pec` or a dispersive metal) plus
a thickness on a part's surface. The page SHALL offer one for every part of a
time-domain upload, one-part uploads included: a coating material picker
(default "no coating") posting `part_coating.<part name>`, and a thickness in
the form's length unit posting `part_coating_thickness.<part name>`. The
stored request SHALL carry the key `coatings`: null, or part name →
[material, thickness in metres].

The coating SHALL be a shell of its thickness OUTSIDE the part's surface: every
point within that distance of the surface and outside the part takes the
coating's material, so a core coated in gold lays out as a gold shell around
the core's material. The shell SHALL grow into open space or into a part that
encloses the coated part (displacing that host). It SHALL NOT be painted into
the coated part, the parts inside it, a neighbour that does not enclose it (a
neighbour keeps its volume, so a coating on a face flush against a neighbour is
absent there), or a metal boundary's wall. Where two shells meet, the shared-face
rule decides: the conductor, else the higher index. The grid SHALL be refined by
the grid builder's coating rule on each axis of the coated part (two thin layers
at its faces where both faces are flat, else its whole span at thickness / 4),
plus a metal coating's skin-depth spacing.

#### Scenario: a gold-coated core lays out as a gold shell over its core
- **WHEN** a dielectric part is coated in 10 nm of gold
- **THEN** the nodes over each face within 10 nm of it are gold nodes, the
  part's own nodes keep its index, and the nodes beyond the shell are vacuum;
  the shell's faces are grid planes when the part's faces are flat

#### Scenario: a node on a coating's face is decided by rule, not rounding
- **WHEN** a node lies within the grid's face-line tolerance (the tolerance the
  grid snaps face lines with) of the coated part's surface or of the shell's
  outer face
- **THEN** it is on that face, and the shared-face rule decides it: on the
  part's surface the conductor, else the higher index, of part and coating; on
  the outer face the coating, unless the space it grows into ranks higher (open
  space ranks as index 1). So a 50 nm gold coating on a dielectric box is the
  same number of nodes thick on all six faces, wherever the box sits in the
  mesh frame

#### Scenario: a coating on a curved part follows its surface
- **WHEN** a sphere is coated
- **THEN** the coating's nodes are the ones whose distance from the sphere's
  surface is under the thickness, outside it

#### Scenario: a coating displaces its host and spares its neighbour
- **WHEN** a coated part sits inside a host part and flush against a neighbour
- **THEN** the shell replaces the host's material around it, and the
  neighbour keeps its own material within the shell's reach

#### Scenario: a coating on the cavity's own wall is refused
- **WHEN** a part that encloses a metal-walled cavity (no part encloses it) is
  coated
- **THEN** the request is refused naming the part: its shell would lie in the
  wall

#### Scenario: the source, probes, estimate and TF/SF box see the coating
- **WHEN** a part carries a conductor coating
- **THEN** a source or probe inside the coating is refused naming "the coating
  of part <name>", the form's estimate counts the coating's metal and is never
  below the solved run's, and a plane wave's TF/SF box encloses the shells

#### Scenario: a bad coating is refused naming the part
- **WHEN** a coating names a part not in the upload, has an unknown material,
  or a thickness that is missing, not a number, or not positive
- **THEN** the upload is refused before queueing, and the message names the part

### Requirement: Parts that touch SHALL meet on a face one of them owns, and SHALL NOT overlap

A film flush against its plates shares a face with each. A grid node on a
shared face SHALL belong to the conductor (`pec` or a dispersive metal) over
any dielectric, and between two dielectrics to the higher refractive index,
whatever the file order. The same rule SHALL name the part a source or probe
sits in. Two parts, neither inside the other, that share volume SHALL be
refused naming both parts; nested parts are not overlap (the innermost owns
its volume). The upload SHALL be refused before queueing when a sampled check
finds shared volume: for each pair neither of which encloses the other, a
16 x 16 x 16 lattice of cell centres in the intersection of their bounding
boxes is tested against both parts. Flush neighbours meet in a box of zero
thickness and SHALL NOT be probed. The lattice can step over a sliver thinner
than its spacing, so the voxeliser's check on every grid node, in the worker,
SHALL stay authoritative and fail the job readably. Nesting is decided by an
exact containment test, asked only where one part's bounding box lies within
the other's.

#### Scenario: a film's faces go to its plates
- **WHEN** a dielectric slab and a `pec` (or metal) slab share a face
- **THEN** every node on that face is in the conductor mask, in either file order

#### Scenario: a film between conductor plates keeps its drawn thickness
- **WHEN** a dielectric film lies flush between two conductor plates, its
  faces anywhere relative to the coarse lattice
- **THEN** every flat face that bounds a part's box is a grid line along its
  normal, the nodes on the film's faces are conductor nodes, and the
  conductor walls sit exactly the film's thickness apart; a curved part's box
  adds no grid line

#### Scenario: the form's memory estimate still errs high
- **WHEN** a metal part's face lies on a grid node
- **THEN** the form's estimate counts every node whose dual cell overlaps the
  part's bounding box, so it is never below the solved run's

#### Scenario: between dielectrics the higher index owns the face
- **WHEN** slabs of index 2.0 and 1.5 share a face
- **THEN** the face's nodes take index 2.0, in either file order

#### Scenario: overlapping parts are refused by name
- **WHEN** two parts, neither enclosing the other, share volume
- **THEN** the run fails with a message naming both parts; touching or nested
  parts run

#### Scenario: an overlap is refused at upload
- **WHEN** a td upload holds two parts, neither enclosing the other, that share
  volume the lattice samples
- **THEN** the upload is refused before queueing, naming both parts, and no job
  is created; a flush stack of three parts is queued, with no lattice probed

#### Scenario: a flush film is taken at upload, and an open part is refused by name
- **WHEN** a td upload is a film lying flush between two plates, so the parts
  share faces and their merged surface is open
- **THEN** the part list and the td POST both take it (the upload's whole-mesh
  closed check, whose fallback is the cavity parser's touch rule, does not run
  for td); a td upload with a part that has holes is refused naming that part

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

#### Scenario: waiting for memory does not use up the retries
- **WHEN** the worker defers a `td` job because the Jetson is busy, any number
  of times within the 60 minutes
- **THEN** the job's release count is unchanged (a busy machine is no verdict
  on the job), the 60 minutes are counted from the FIRST deferral and later
  deferrals do not restart them, and the job is not handed out again for 60
  seconds after each deferral, so the worker does not claim it straight back
- **AND** only a `td` job can be deferred this way, and only by the worker that
  holds its lease
- **AND** a release or an expired lease requeues the job with a fresh wait: a
  deferral after that starts a new 60 minutes, so an old wait cannot fail it

#### Scenario: a Jetson that stays busy fails readably
- **WHEN** free memory stays short for 60 minutes
- **THEN** the job fails with a message saying the Jetson stayed busy, and it was
  never moved to the cluster queue
- **AND** it is finished like any failed job: its lease is released and its
  finish time is set

#### Scenario: memory taken mid-run fails readably
- **WHEN** the device runs out of memory during a `td` run
- **THEN** the job fails with a message naming the memory shortage

#### Scenario: a kernel out-of-memory kill fails the job once
- **WHEN** the kernel's OOM killer ends a `td` run with SIGKILL (on the Orin
  the GPU shares system RAM, so this, not an exception, is a common way to run
  out of memory)
- **THEN** the job fails once with the out-of-memory message, and is never
  requeued as a counted release
- **AND** this holds because the run solves in a spawned child process, with
  the worker (holding the lease) reading exit status -9 as out of memory; the
  child ignores SIGTERM and SIGINT, so a service stop does not end a run; an
  error raised in the child reaches the worker's classifier with its type (or
  its type's name, when it cannot be pickled) and its cause chain; and the
  worker's unit sets `OOMPolicy=continue`, so systemd does not stop the
  service when the child is killed
- **AND** only SIGKILL reads as out of memory: a child that exits non-zero or
  dies of another signal (SIGABRT, say) fails with the plain "stopped
  unexpectedly" message naming its exit status, never the out-of-memory one

#### Scenario: a run stuck inside one step is still stopped
- **WHEN** a `td` child is still running 5 minutes past the 45-minute limit
  (its own stop is checked only between steps, so it is stuck inside one)
- **THEN** the worker kills it and fails the job with the 45-minute message
- **AND** the worker times the run from when the child reports `solving`, the
  moment the child's own 45 minutes start, never from spawn, so a long setup
  does not eat into the run's time
- **AND** before `solving` arrives, setup is capped at 50 minutes from spawn;
  a child that reaches it is killed and the job fails with a plain message
  saying the setup took too long
- **AND** the worker reads every message the child has already sent before it
  looks at either clock, and an outcome found there wins over a deadline: a
  child that finished is never failed as too slow
- **AND** a child that has sent its outcome but has not exited within 60
  seconds is killed, and every child is reaped (no zombie) before the worker
  goes on

#### Scenario: an overlong run is stopped
- **WHEN** a `td` run passes 45 minutes of run time
- **THEN** it is stopped and the job fails with a message naming the limit
- **AND** the run time is checked before the first step and every 64 steps
  after it, on the host without reading any device value, so the stop comes at
  most 64 steps late and the check adds no device sync; the message names the
  step reached, the total, and the advice (increase dx or shorten the duration)

#### Scenario: free memory is what a new run can take on the Orin
- **WHEN** the worker checks free memory before a `td` run
- **THEN** it compares the estimate with the kernel's MemAvailable (the Orin's
  GPU shares system RAM) plus the blocks this process's cupy memory pool
  already holds free, and runs on numpy when no device is present

#### Scenario: a time-domain verdict is never retried
- **WHEN** a `td` job is refused (layout, mesh, estimate or limits) or stopped
  (out of memory, 45 minutes)
- **THEN** it is failed with that message shown to the owner unchanged, never
  released for another attempt, whatever error it was raised from; a device
  error the solver does not translate is classified like any other job's

### Requirement: Dispersive metals SHALL follow their published model

Gold, silver, aluminium, copper and chromium SHALL be modelled by the
Lorentz-Drude fits of Rakić et al., Applied Optics 37, 5271 (1998), with
ε∞ = 1 and the paper's parameters in eV (ω_p, f_j, Γ_j, ω_j), stepped with
auxiliary differential equations. ADE storage SHALL cover only the metal cells'
bounding box. An E sample on a metal face SHALL take the metal. A run whose
pulse band leaves the paper's fitted range for a metal in the scene SHALL be
refused before it is queued.

#### Scenario: the stepped metal has the model's susceptibility
- **WHEN** a metal's term is driven by a sinusoidal field at 20 time steps
  per radian (Au at 1 and 3 eV, Ag and Cr at 2 eV)
- **THEN** the polarisation it builds matches the model's χ(ω) within 1.5%

#### Scenario: a face sample takes the metal and eps_inf = 1
- **WHEN** a single metal node sits next to a dielectric
- **THEN** each E component takes the metal at exactly the two samples whose
  segment touches the node
- **AND** those samples step with ε∞ = 1 whatever the neighbouring permittivity

#### Scenario: a metal near a plane wave's box is refused
- **WHEN** a metal region lies within 3 cells of a plane wave's box faces
- **THEN** the run is refused with a message naming a dispersive metal and
  its node

#### Scenario: the material table matches the paper
- **WHEN** ε(ω) is computed from the library at 12 wavelengths per metal,
  spread through the published table from its shortest to its longest
- **THEN** n + ik matches an independently published Rakić Lorentz-Drude n, k
  dataset (cited in the test) within 1e-3 (relative) at every one, tight
  enough that a 10% typo in a single model parameter fails

#### Scenario: benchmark V4
- **WHEN** a pulse reflects at normal incidence from a gold slab at least 10
  skin depths thick, with at least 4 cells per skin depth, on an `open` travel
  axis with periodic transverse faces (so the slab is infinite, as the Fresnel
  result assumes)
- **THEN** the reflectance, measured as |scattered/incident|² at a
  scattered-field probe, is within 0.02 (absolute) of the analytic Fresnel
  reflectance for the same model, over the band where the incident spectrum is
  at least 1e-2 of its peak

#### Scenario: the V4 bench judges only what it measured
- **WHEN** the V4 bench runs
- **THEN** it takes the incident spectrum from a reference run with the same
  metal term over an empty region (so both runs share one time step), and
  transforms both traces without a window
- **AND** it fails if either trace still holds more than 1e-3 of its peak in
  its last 5% (a cut-off record), or if the ratio is not finite in the band
- **AND** it exits 2 without running when the pulse band leaves the metal's
  fit, and exits 2 rather than passing when the setup has fewer than 4 cells
  per skin depth or a slab under 10 skin depths
- **AND** it fails unless the judged bins span the band it checked the metal
  and skin depths over, each end within one bin and with no hole, since a
  shorter record gives fewer, wider bins over the same band
- **AND** it runs with the solver's default 10-cell CPML

#### Scenario: a metal slab across periodic faces is uniform
- **WHEN** a plane wave meets a metal slab that spans every periodic axis
- **THEN** E inside the slab is the same at every transverse sample within
  1e-6 of its peak (the metal term advances every sample the E update does,
  including the last row of a periodic axis)

### Requirement: Thin layers SHALL be resolved, not lost

The grid SHALL place at least 4 cells across every layer it is given that is
thinner than the coarse cell, or across the metal skin depth where that is
thinner, with a neighbour spacing ratio of at most 1.3. For a `td` upload the
layers are the ones the layout rule detects; a thin feature it does not
detect is reported by a caveat on the result, not resolved.

#### Scenario: the grid builder keeps its own rules
- **WHEN** the grid builder is given layers thinner than the coarse cell
- **THEN** every such layer has at least 4 cells across it and every neighbour
  spacing ratio is at most 1.3

#### Scenario: a layer can cap its own spacing
- **WHEN** a layer carries a spacing (a metal surface gets its skin depth at the
  top of the band divided by 4, a curved coating its thickness divided by 4)
- **THEN** no cell inside the layer is wider than that spacing, and the layer's
  edges are grid nodes

#### Scenario: a coating follows the region's shape
- **WHEN** a region spanning lo..hi on an axis is coated with thickness t
- **THEN** a region with flat faces normal to that axis adds the two thin layers
  lo−t..lo and hi..hi+t, and any other region adds one layer lo−t..hi+t capped at
  t/4, so a curved coating costs more cells than a flat one

#### Scenario: a grading that cannot converge is refused
- **WHEN** repairing neighbour-ratio violations grows an axis past a fixed
  multiple of its first node count, or does not settle within the repair limit
- **THEN** the builder raises a named grid error instead of refining further

#### Scenario: a cut cell takes the volume average
- **WHEN** a node's dual cell (halfway to each neighbour, clipped at the domain
  ends) is split between materials
- **THEN** its permittivity is the average over the dielectric samples of that
  cell; a node at least half of whose samples are one metal is that metal's
  node; a node with no dielectric sample gets permittivity 1; and nothing
  outside the domain is sampled

#### Scenario: benchmark V5
- **WHEN** a dielectric film of index n, far thinner than the coarse cell,
  fills the gap between two PEC plates of lateral size a × b with PEC side walls,
  driven by a source with E normal to the plates
- **THEN** its resonances are within 1% of the parallel-plate lateral modes
  f = (c/2n)·√((m/a)² + (p/b)²) with m, p ≥ 1, which depend on n and not on the
  film thickness, and every such mode in the pulse band appears

#### Scenario: how the V5 bench decides
- **WHEN** the V5 bench judges a run
- **THEN** "far thinner" means at most a quarter of the coarse cell; the film is
  run at two thicknesses and both are gated against the same modes; peaks are
  matched to modes one to one, closest first; a peak more than 1% from every mode
  of the family fails the run; a mode whose source × probe overlap is under 0.05
  is listed as not gated; and a setup whose grid dispersion at the top of the band
  exceeds half the tolerance, or whose film axis breaks the builder's rules, exits
  2 (not a spec setup) unless the run already failed; a missing module or a
  crash in the bench's own setup exits 2 ("did not run"), while an exception
  raised by the solver during a run exits 1 (a FAIL), so a solver bug never
  reads as a bench problem

### Requirement: The upload's parts SHALL be voxelised onto the grid

Each sample of a node's dual cell SHALL take the material of the innermost
uploaded part containing it. Outside every part the grid SHALL be a conductor
for `metal` and vacuum for `open`. Every part SHALL be given a material: a
refractive index of at least 1, `pec`, or one of the Rakić metals.

#### Scenario: nested parts in any file order
- **WHEN** an upload holds a part nested inside another, in either file order
- **THEN** every sample inside the inner part takes the inner part's material,
  matching the cavity tool's winding-number test point for point

#### Scenario: a sample on a mesh edge's line
- **WHEN** an axis-aligned part puts its edges on the grid's sample lines
- **THEN** those samples are classified as the winding-number test classifies
  them (queries are shifted ~1e-6 of the scene size so no ray meets an edge)

#### Scenario: outside the shape
- **WHEN** a `metal` run is voxelised
- **THEN** every node whose cell lies outside the parts is in the conductor mask;
  for an `open` run those nodes are vacuum (permittivity 1)

#### Scenario: a materials table that does not fit the upload
- **WHEN** the table leaves a part out, names a part the file lacks, gives an
  unknown material name, or gives an index below 1 or not finite
- **THEN** the run is refused with a message naming the part(s) and, for a
  mismatch, the file's parts

### Requirement: A `td` upload SHALL be laid out on its grid by one fixed rule

The worker SHALL turn a `td` job's upload into a grid by this rule, and phase
5's estimate SHALL use the same function, so the run that is estimated is the
run that is solved.

- **Parts.** The upload's named groups, each a closed surface, refused by name
  otherwise.; the closed check is per part, never on the merged mesh, so parts
  may share faces. There is no enclosing-wall rule: in `open` space the parts are
  separate objects. Each part takes its entry in the job's `materials` table
  (a refractive index of at least 1, `pec`, or a Rakić metal name); a
  one-part upload with no table takes the job's `n_index`.
- **Frame.** Metres are mesh units × the unit's metres × `scale`. The origin is
  the minimum corner of the parts' combined bounding box. `position` and every
  probe are metres in this frame.
- **Extent.** For `metal` the grid is exactly the bounding box: outside the
  parts is the conductor. For `open` every side is padded by 16 cells of
  `dx_m` (the 3-cell TF/SF clearance, 3 more cells, and the 10-cell CPML), for
  a point source as well as a plane wave.
- **Plane wave.** The TF/SF box is the bounding box grown by the 3-cell
  clearance on every side. A plane wave with `metal` is refused, as in phase 2.
- **Cells.** The coarse cell is `dx_m`. On each axis a part whose
  bounding-box extent is under 4 `dx_m` adds a layer over that extent with
  spacing a quarter of the extent, so it has at least 4 cells across. A metal
  part adds, on each axis, a layer 3 skin depths deep inside each of its two
  bounding-box faces (the whole extent when it is thinner than 6), with spacing
  a quarter of the skin depth at the top of the source band; where both apply
  the smaller spacing wins. The skin depth is `fdtd_grid.skin_depth(metal, f)`
  = c / (2π f κ), κ the imaginary part of √ε(f) of the Rakić model; the worker
  and the estimate both call that function. The top of the band is
  `frequency_hz + bandwidth` for a pulse and `frequency_hz` for `cw` and
  `drawn`.
- **Thin shells.** The layers above are the only thin features the layout
  detects (axis-aligned extents and metal bounding-box faces). A curved thin
  shell is voxelised at the grid it gets, and the estimate counts that grid;
  the result carries the caveat "thin curved features are resolved only at
  the coarse cell" whenever a part is not a box (its volume is under 0.99
  of its bounding box's), since its thin walls may not be refined. The
  thin-layer requirement applies to the detected layers.
- **Refusals.** A probe or point-source position outside the grid, and a
  boundary other than `metal` or `open`, are refused before anything is
  allocated, naming the value. A point source or probe whose nearest node is a
  conductor (the `metal` boundary's outside, a `pec` part, or a Rakić-metal
  part's interior) is refused, naming the position and the part, since its
  source would inject nothing and its trace would read as "no response".
  With `open`, a point source or probe whose nearest node lies in the 10-cell
  CPML at either end of any axis is refused, naming the position and the
  range it must lie in: the layer absorbs a source as it is injected and
  damps what a probe reads. Periodic boundaries stay a phase 6 option.

#### Scenario: the frame is the bounding box's corner in metres
- **WHEN** a one-part box spanning 2..6 mm in x is laid out with units `mm`
  and scale 1
- **THEN** the grid's x nodes run from 0 to 4 mm for `metal`, and from
  -16 `dx_m` to 4 mm + 16 `dx_m` for `open`

#### Scenario: a thin metal plate is resolved at its skin depth
- **WHEN** a gold plate thinner than 6 skin depths lies in `open` space
- **THEN** no cell across the plate is wider than a quarter of gold's skin depth
  at the top of the band, and the plate's faces are grid nodes

#### Scenario: separate objects need no wall
- **WHEN** an `open` upload holds two closed parts, neither inside the other
- **THEN** it is laid out without the cavity solver's one-wall refusal

#### Scenario: a part thinner than 4 cells gets 4 cells
- **WHEN** a dielectric plate 1.5 `dx_m` thick lies in `open` space
- **THEN** at least 4 cells lie across the plate and no cell across it is wider
  than a quarter of its thickness

#### Scenario: a curved part carries the thin-feature caveat
- **WHEN** a part is not a box (a sphere, say)
- **THEN** the result carries the caveat that thin curved features are resolved
  only at the coarse cell, and a box-only upload carries no such caveat

#### Scenario: a source or probe inside a conductor is refused
- **WHEN** a point source or probe sits on a conductor node (outside the parts
  with `metal`, inside a `pec` part, or inside a Rakić-metal part)
- **THEN** the job is refused before allocation, naming the position and the
  part (or the metal boundary)

#### Scenario: a source or probe inside the CPML is refused
- **WHEN** with `open`, a point source or probe lies on the grid but its
  nearest node is inside the 10-cell CPML
- **THEN** the job is refused before allocation, naming the position and the
  range outside the layer

#### Scenario: a probe off the grid is named
- **WHEN** a probe lies outside the laid-out grid
- **THEN** the job is refused before allocation, naming the probe and the grid's
  extent

### Requirement: The solver SHALL record a bounded field movie of |E|
On request the solver SHALL record |E| at up to 60 steps spread evenly over the
run, the last being the final step, on the nodes a per-axis stride keeps. The
stride SHALL be the smallest whole number that brings each axis to at most 64
samples. Each frame SHALL be taken after the step's E update, source and
conductors, so it shows no field inside a conductor. The three components SHALL
be combined where they are stored, without interpolation, and squared in float64
so small fields keep their digits. The frames SHALL come back as host float32
whatever array library ran the solve, and the memory estimate SHALL count them.

#### Scenario: the frames match the probes
- **WHEN** a run records frames and probes at the same nodes
- **THEN** each frame's value at a probe node is the magnitude of that probe's
  three components at the frame's step, to float32 precision

#### Scenario: a long run is capped
- **WHEN** a 130 × 64 × 65-node grid runs for 1,000 steps
- **THEN** the movie has 60 frames ending on step 999, with strides (3, 1, 2)
  and frames of 44 × 64 × 33 samples

#### Scenario: a bad frame plan is refused
- **WHEN** a frame step is repeated or outside the run, or a stride is below 1
- **THEN** the solver refuses before stepping, naming the bad value

### Requirement: A time-domain job's results SHALL be three kinds of flat file
A finished `td` job SHALL store `trace.bin` (every probe's trace, float32
little-endian, C order, shaped probes × steps), one `frame_NN.bin` per movie
frame (two-digit, 1-based, |E| float32 little-endian, C order, in the grid's
x, y, z order) and `spectrum.json` (the manifest: time step, step count, the
band, each probe's position, component and amplitude spectrum within the band,
the trace and frame shapes, each frame sample's position along x, y and z in
metres (the grid is graded, so a viewer cannot assume even spacing), and the
layout caveats). The spectra SHALL be scaled so the largest peak across all
probes is 1, stated in the manifest as `spectrum_scale: "largest peak = 1"`, so
they compare across probes and do not grow with the step count. A probe with
no signal in the band SHALL be stored as flat zero with a caveat naming it,
never NaN, even when no probe has any signal. The manifest SHALL be
written last, so its presence means every other file exists. A non-finite value
in a trace, frame or spectrum SHALL fail the job before anything is written.
The worker's upload list and the app's accepted names SHALL agree on every one
of these names, and SHALL refuse near misses (`frame_1.bin`, `frame_001.bin`,
a path separator).

#### Scenario: the spectrum does not grow with the run
- **WHEN** the same steady wave is recorded for twice as many steps
- **THEN** the stored spectrum still peaks at 1 at the same frequency, with the
  same shape around the peak counted in frequency bins

#### Scenario: a probe the source cannot excite
- **WHEN** a probe's trace is zero throughout
- **THEN** its stored spectrum is all zeros, the other probes keep their scale,
  and the caveats say that probe recorded no signal in the band

#### Scenario: the manifest places every movie sample
- **WHEN** a movie keeps every `stride`-th node of a graded grid
- **THEN** `frames.positions_m` lists, per axis, exactly those nodes in metres,
  in the probes' frame, one per sample; a count that disagrees with the frame
  shape fails the job before anything is written

#### Scenario: the files decode to the solver's own output
- **WHEN** a tiny `td` job finishes
- **THEN** `trace.bin` decodes to the solver's traces and each `frame_NN.bin`
  to its frame, exactly, and `spectrum.json` names their shapes

#### Scenario: a run that dies part way leaves no manifest
- **WHEN** writing stops after `trace.bin` or a frame
- **THEN** no `spectrum.json` exists for the job

#### Scenario: NaN is refused before writing
- **WHEN** a trace or frame holds NaN or infinity
- **THEN** the job fails naming which output, and no file is written

#### Scenario: a near-miss name is refused by both sides
- **WHEN** the worker or the app is offered `frame_1.bin`, `frame_001.bin` or
  `x/frame_01.bin`
- **THEN** both refuse it

#### Scenario: the page reads td results through files/
- **WHEN** a td job is done
- **THEN** its status carries `spectrum_url` and `trace_url` (both under
  `/jobs/<id>/files/`) and none of `modes`, `manifest_url`, `surface_url` or
  `bands`, and `files/trace.bin`, `files/spectrum.json` and
  `files/frame_NN.bin` return the stored bytes

#### Scenario: td results are not served early or invented
- **WHEN** the job is not done, the named frame was not written, or the name is
  not one the solver writes
- **THEN** the files/ route answers 404

### Requirement: A finished time-domain job's page SHALL plot each probe's trace and spectrum
The job page of a finished `td` job SHALL show a card with every probe's trace
against time and its amplitude spectrum over the manifest's band (the
spectrum axis labelled with the manifest's scale, "largest peak = 1"), both with
labelled axes in SI-prefixed units, a legend naming each probe's component and
position in its line's colour, and the run's caveats. The cavity viewer, mode
table and band card SHALL stay hidden. A trace longer than the plot is wide
SHALL be drawn as each pixel column's minimum and maximum, not by decimation,
so a CW trace does not alias into a false beat. A manifest or trace that cannot
be fetched SHALL be named in the card's caption, never left as an empty plot.

#### Scenario: a finished td job shows its probes
- **WHEN** a finished `td` job with two probes is opened
- **THEN** the card shows two trace lines and two spectrum lines, a time axis in
  ps or ns, a frequency axis in GHz or THz, and a legend whose swatches match
  the line colours, and the 3-D viewer, mode table and band card are hidden

#### Scenario: a trace that cannot be fetched is named
- **WHEN** the trace request answers 502
- **THEN** the caption says the probe traces could not be loaded and names the
  HTTP status

#### Scenario: caveats reach the page
- **WHEN** the manifest lists caveats
- **THEN** each one is listed on the card

### Requirement: A finished time-domain job's page SHALL play its field movie
The job page SHALL play the run's |E| frames in a 3-D view drawn with the cavity
view's raymarch shader. Each frame SHALL be resampled from the graded grid onto
an evenly spaced lattice using `frames.positions_m`, so a thin film is drawn at
its true width. Every frame SHALL share ONE colour scale (a high percentile of
|E| over the whole movie), and the caption SHALL state the frame number, its
time, that scale and the drawing floor, read from the same constants the
drawing uses. A slider SHALL select any frame and a button SHALL play and pause.

#### Scenario: frames play at their own times
- **WHEN** the owner opens a finished td job with a recorded movie
- **THEN** the movie shows frame 1, the slider spans every frame, and moving it
  to frame n captions frame n at `frames.steps[n-1] × dt_s`

#### Scenario: a movie that cannot be shown says why
- **WHEN** a frame cannot be fetched, the run recorded no movie, or every frame
  is zero
- **THEN** the caption names the failing frame and its HTTP status, says the run
  recorded no field movie, or says there is no field to draw -- never a still
  or empty picture with no explanation

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

#### Scenario: the drawing is read the way it was drawn
- **WHEN** a drawn waveform of N samples and duration T is resampled
- **THEN** the samples span [0, T] evenly, both ends included, joined by
  straight lines; the grid reads them at the run's own steps (after any
  dispersive dt reduction), and the waveform's sum over those steps is zero

#### Scenario: detail finer than dt is judged, not decimated away
- **WHEN** a drawing's sample spacing is finer than dt
- **THEN** its spectral energy above f_res is measured at the drawing's own
  spacing, so detail that reading at dt would alias cannot pass the check

#### Scenario: a drawing with nothing left is refused
- **WHEN** a drawing is constant (nothing remains once its mean is removed) or
  lasts under two time steps
- **THEN** it is refused with a message saying so, not run as a silent source

#### Scenario: the solver repeats the check as a backstop
- **WHEN** a drawn source reaches the solver
- **THEN** the solver refuses it, naming f_res, if more than 1% of its energy
  is above f_res computed from the dielectric index alone; the queue-time
  check, which knows the metals' |n|, is the one that keeps the job out of the
  queue

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

The solver takes `bloch=(kx, ky, kz)` in rad/m; the field obeys
F(x + L) = exp(i k L) F(x) across each periodic axis of period L (the sum of
its `n-1` spacings). The core stays real float32. A phase k L within 1e-12 of
0 or pi runs the real core alone, the copy slabs taking the factor +1 or -1.
Any other phase steps a second, sin, set of the six field arrays beside the
cos set through the same update; only the cos set is driven, and the face hook
fills the H ghost with exp(i k L) times half sample 0 and E node 0 with
exp(-i k L) times node `n-1`, mixing the two sets. The result adds
`traces_sin` (the probes on the sin set, absent on the real path); energy sums
both sets, a frame is the complex field's |E|, and `peak` is the cos set's.
A non-zero k on an axis that is not periodic, a malformed wave vector, and a
paired run with a plane wave or a polarisation term (a metal) are refused with
a message. The form option is a later phase-6 task.

#### Scenario: a Bloch cell is one cell of the supercell
- **WHEN** a cell with a dielectric rod, graded along x, runs at k = 2 pi p /
  (M L) on its periodic axes with a point source, and an M-cell supercell runs
  at k = 0 driven by the same source times cos(k.R) in the cell at R
- **THEN** in every cell R, every probed E component (seam nodes on every axis
  included) equals cos(k.R) times the cos trace minus sin(k.R) times the sin
  trace, within 5e-6 of the peak, for phases paired on two axes, at the zone
  edge alone, and at the zone edge with a paired axis
- **AND** a wrong sign on either coupling, an unstepped sin set, a wrong period
  or a wrong zone-edge sign fails it

#### Scenario: a closed Bloch box conserves energy
- **WHEN** a pulse runs in a graded box periodic on all three axes with a
  paired phase on each
- **THEN** after the source ends the energy summed over both sets varies by
  less than 1e-5 of its mean

#### Scenario: k = 0 is the plain periodic run
- **WHEN** a periodic run with an open axis, a conductor, a dielectric, energy,
  peak and frames is made with `bloch` omitted or zero
- **THEN** every output is bit-identical to the phase-2 (6.1a) solver's

#### Scenario: benchmark V6
- **WHEN** a pulse is run in one cell of a 2-D crystal extruded along z, with
  Bloch-periodic x and y faces at a chosen k, periodic z faces with kz = 0, for
  each polarisation (TM: E along z; TE: H along z)
- **THEN** the spectrum's peaks are within 1% of the photonic-crystal band
  solver's frequencies at that k for that polarisation, and every band in the
  pulse band appears

#### Scenario: benchmark V6 checks k against the exact light lines
- **WHEN** the same cell holds one homogeneous medium of index n and runs at
  several non-zero k, below and at the zone edge (X and M included)
- **THEN** every excitable folded light line f a / c = |k + G| / n in the
  pulse band has a peak within 1% of it, and no peak is 1% from every line
- **AND** a tiny form of this check (20 cells, four k) runs locally in tier 1,
  so a phase that ignored k or scaled it wrongly fails a local test
