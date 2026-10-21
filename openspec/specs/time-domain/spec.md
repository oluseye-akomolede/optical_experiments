# time-domain Specification

## Purpose
What a time-domain (`td`) job does: the FDTD solver that runs a pulse through
the uploaded scene, the materials and boundaries it accepts, the probe traces,
spectra and field movie it writes, how the job page shows them, and the
benchmarks and in-browser checks that hold each of those to a stated bound.

## Requirements

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
auxiliary differential equations. Gold SHALL also be available as the second
model fitted to Palik's 1985 table (see "A second gold model SHALL fit Palik's
table"), chosen by name, and
the Rakić fits stay the default so an unaltered run models gold as it did
before. ADE storage SHALL cover only the metal cells'
bounding box. An E sample on a metal face SHALL take the metal. A run whose
pulse band leaves the fitted range of the model it selected for a metal in the
scene SHALL be refused before it is queued, the refusal naming that model's
source.

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
refractive index of at least 1, `pec`, or one of the library’s dispersive metals (`fdtd.METALS`).

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
  otherwise; the closed check is per part, never on the merged mesh, so parts
  may share faces. There is no enclosing-wall rule: in `open` space the parts are
  separate objects. Each part takes its entry in the job's `materials` table
  (a refractive index of at least 1, `pec`, or a library metal name); a
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
  bounding-box faces, with spacing a quarter of the skin depth at the top of
  the source band; where both apply the smaller spacing wins. It gets **one
  layer across its whole extent at that spacing** instead when the two face
  layers would meet or cover the part already, or when what they leave between
  them would be narrower than the wider of one `dx_m` and one skin depth: such
  a middle is filled with a few cells across it -- four at a quarter of the
  gap, two and three for middles in between -- and while the gap is under one
  skin depth those cells are finer than the skin layers themselves: measured
  at dx 10 nm, middles of 6.1, 11.1, 16.1 and 26.1 nm came out in 1.528, 5.556,
  5.352 and 6.528 nm (0.21x-0.90x of the spacing), and Run C's 3.2 nm middle
  in four cells of 0.801 nm. A gap over one skin depth does not need the
  one-layer rule: with room around the part, middles from 31 to 726 nm
  measured 0.997x-1.000x of the spacing.
  One layer across the part takes under `24 + 4*max(dx_m, skin depth)/skin
  depth` cells across it (28 where a coarse cell is no coarser than the skin
  depth) at a quarter of the skin depth, never finer than the cells it
  replaces. The
  skin depth is
  `fdtd_grid.skin_depth(metal, f)`
  = c / (2π f κ), κ the imaginary part of √ε(f) of that metal’s model; the worker
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
  conductor (the `metal` boundary's outside, a `pec` part, or a library-metal
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

#### Scenario: a metal part just over 6 skin depths gets no sliver in its middle
- **WHEN** a metal part is thick enough that two face layers leave less than one
  `dx_m`, or less than one skin depth, of bulk between them
- **THEN** the part is laid out with one layer across its whole extent at a
  quarter of the skin depth: no cell across it is finer than half that, it
  takes no more cells than that spacing needs, and its faces are grid nodes

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
  with `metal`, inside a `pec` part, or inside a library-metal part)
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

On request the solver SHALL record |E| at up to the requested number of steps
(default 60, at most 120) spread evenly over the requested window, the last
being the window's final step (the run's final step when no window is asked
for). The stride SHALL be the smallest whole number that brings each axis to at
most 64 samples. Each frame SHALL be taken after the step's E update, source
and conductors, so it shows no field inside a conductor. The three components
SHALL be combined where they are stored, without interpolation, and squared in
float64 so small fields keep their digits. The frames SHALL come back as host
float32 whatever array library ran the solve, and the memory estimate SHALL
count the requested number of frames.

#### Scenario: the frames match the probes
- **WHEN** a run records frames and probes at the same nodes
- **THEN** each frame's value at a probe node is the magnitude of that probe's
  three components at the frame's step, to float32 precision

#### Scenario: a long run is capped
- **WHEN** a 130 × 64 × 65-node grid runs for 1,000 steps with no window and no
  frame count
- **THEN** the movie has 60 frames ending on step 999, with strides (3, 1, 2)
  and frames of 44 × 64 × 33 samples

#### Scenario: a bad frame plan is refused
- **WHEN** a frame step is repeated or outside the run, or a stride is below 1
- **THEN** the solver refuses before stepping, naming the bad value

#### Scenario: a window inside the run
- **WHEN** the same grid records frames over a window that starts after step 0
- **THEN** every recorded step lies inside the window, the last is the window's
  last step, and the stride is unchanged from the whole-run movie

### Requirement: A time-domain job's results SHALL be three kinds of flat file

A finished `td` job SHALL store `trace.bin` (every probe's trace, float32
little-endian, C order, shaped probes × steps), one `frame_NNN.bin` per movie
frame (1-based, |E| float32 little-endian, C order, in the grid's x, y, z order;
named with two digits for a movie of up to 99 frames and three from 100, so a
120-frame movie ends at `frame_120.bin`) and `spectrum.json` (the manifest: time
step, step count, the band, each probe's position, component and amplitude
spectrum within the band, the trace and frame shapes, each frame sample's
position along x, y and z in metres (the grid is graded, so a viewer cannot
assume even spacing), and the layout caveats). The spectra SHALL be scaled so
the largest peak across all probes is 1, stated in the manifest as
`spectrum_scale: "largest peak = 1"`, so they compare across probes and do not
grow with the step count. A probe with no signal in the band SHALL be stored as
flat zero with a caveat naming it, never NaN, even when no probe has any signal.
The manifest SHALL be written last, so its presence means every other file
exists. A non-finite value in a trace, frame or spectrum SHALL fail the job
before anything is written. The worker's upload list and the app's accepted
names SHALL agree on every one of these names, and SHALL refuse near misses
(`frame_1.bin`, `frame_0001.bin`, a path separator).

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
- **THEN** `trace.bin` decodes to the solver's traces and each `frame_NNN.bin`
  to its frame, exactly, and `spectrum.json` names their shapes

#### Scenario: a run that dies part way leaves no manifest
- **WHEN** writing stops after `trace.bin` or a frame
- **THEN** no `spectrum.json` exists for the job

#### Scenario: NaN is refused before writing
- **WHEN** a trace or frame holds NaN or infinity
- **THEN** the job fails naming which output, and no file is written

#### Scenario: a near-miss name is refused by both sides
- **WHEN** the worker or the app is offered `frame_1.bin`, `frame_0001.bin` or
  `x/frame_01.bin`
- **THEN** both refuse it

#### Scenario: the page reads td results through files/
- **WHEN** a td job is done
- **THEN** its status carries `spectrum_url` and `trace_url` (both under
  `/jobs/<id>/files/`) and none of `modes`, `manifest_url`, `surface_url` or
  `bands`, and `files/trace.bin`, `files/spectrum.json` and
  `files/frame_NNN.bin` return the stored bytes

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

### Requirement: The field movie SHALL draw the uploaded object over the field

The field movie of a finished time-domain job SHALL draw the uploaded object's
outline -- every part of the upload, its edges and a faint shade of its
surfaces -- in the same 3-D view as the field, on every frame it shows, while
playing and while scrubbing. The outline's parts SHALL be placed by the job's
own `units` and `scale` alone (the upload's own coordinates multiplied by
`unit_scale(units) x scale`), with no further translation and no further
scaling: the movie's frames are stored in that same absolute frame -- the
upload's own metres, whose origin is the parts' combined bounding-box minimum --
so the outline needs no offset. The positions a frame samples MAY extend past
the object: an open boundary's scene is sampled with pad cells on every side of
the object, and those belong to the field's domain, not to the object. A job
whose outline cannot be built or fetched SHALL still play its movie and SHALL
say in the caption that the object outline could not be drawn and why -- never a
movie with no object and no explanation, and never a movie that stops.

Why: the owner asked for the uploaded object to be visible during the movies
(2026-10-02), and a frame drawn without it is a field in an empty box -- the
box's wireframe, not the object.

#### Scenario: an outline on every frame

- **WHEN** a finished td job with a movie is opened and its frames are stepped
  through and played
- **THEN** the uploaded object is drawn over the field on every frame shown,
  at the position the job's own units and scale put it

#### Scenario: an outline that cannot be drawn is named

- **WHEN** the job's outline cannot be built or fetched
- **THEN** the movie still plays its frames and the caption says the object
  outline could not be drawn, naming the reason

### Requirement: A job's object outline SHALL come from its stored upload

The uploaded object's outline SHALL be derivable from a job's own stored upload
and the job's own `units` and `scale` at the moment it is asked for, rather than
from anything the solver writes: every part of the upload SHALL be drawn, parts
SHALL be scaled by the same function and factor the solver used, and no part
SHALL be dropped or overwritten by another part's name. Jobs solved before this
existed SHALL therefore show their outline with no re-run, and the solver SHALL
need no change.

Why: the cavity view's surface file is written only by the cavity solver, in
the user's units and centred on its own origin, so it is not the object's place
in the movie's frame; deriving the outline on request is what lets the owner's
already-solved film job show its object.

#### Scenario: an already-solved job shows its object

- **WHEN** a job solved before this change is opened
- **THEN** its movie draws that job's upload, with no re-run

#### Scenario: a multi-part upload keeps every part

- **WHEN** the upload holds more than one part
- **THEN** every part is in the outline, at its own place

### Requirement: The movie page SHALL state where it drew the outline

The movie page SHALL expose, for the frame it is showing, the axis-aligned box
of the outline it actually added to its scene, in metres, in a form a check can
read back -- and SHALL state that no outline was drawn, rather than nothing,
when the outline could not be built.

Why: an outline drawn near the object instead of on it is the defect the owner
reported, and a check that only counts pixels cannot tell the two apart.

#### Scenario: an outline that is drawn

- **WHEN** the page draws the outline
- **THEN** the box it states is the outline's own, in metres

#### Scenario: an outline that is not drawn

- **WHEN** the outline cannot be fetched
- **THEN** the page states that none was drawn, and the caption says why

### Requirement: The render check SHALL verify the outline on every sampled frame

The render check SHALL read the box the page states it drew and require it,
within one grid cell, to be the box the outline artifact's own vertices span (so
a viewer that translated, rescaled or dropped a part is a failure, not a matter
of opinion), and SHALL require that same box to lie inside the box the
manifest's sample positions span, within one grid cell, on a mirrored real job
-- because a unit or a whole-frame error puts the object outside the field it
was solved in. It SHALL NOT require the drawn box to equal the sampled box: an
open boundary's samples carry pad cells past the object on every side and the
manifest records no boundary, so equality there would fail a correct job. It
SHALL further draw every frame it samples on the same page twice: once with the
outline and once with the outline route answering 404, the two runs differing in
nothing else, and require each sampled frame to gain a substantial number of
pixels from the outline -- including a frame reached by playing the movie, not
only by scrubbing -- while the run without the outline still draws its field.
The share of the frame's own lit pixels the outline must add SHALL be a kind
floor well below what a drawn object measures, not a pass mark set at the
measurement.

Why: a box compared against a centred object would pass for an outline drawn in
the sampled frame or shifted to the box's centre, so the comparison is only
worth making on an object that is off-centre -- the check's own fixture is
therefore two parts, the second at negative x, and the mirrored real job (the
owner's film job) is the same case on production's own artifacts. The pixel pair
is what shows the object is drawn rather than merely described and placed: the
stated box is read from the group the page built, which it builds whether or not
it adds it to the scene.

#### Scenario: the outline is drawn where the object is

- **WHEN** the check runs against a mirror whose object is not centred in its
  domain
- **THEN** the box the page states agrees with the outline artifact's own vertex
  box within one grid cell, and a translated, rescaled or partly dropped outline
  FAILs naming the axis and the two numbers

#### Scenario: the outline is drawn, not merely present

- **WHEN** the same frame is drawn with and without the outline
- **THEN** the frame with it has substantially more drawn pixels, on every
  sampled frame including one reached by playing, and the run without it is
  still a drawn field rather than a blank frame

### Requirement: A time-domain request SHALL carry an optional movie window and frame count

The `td` request SHALL carry `movie_start_s`, `movie_end_s` (seconds, or null
for the whole run) and `frame_count` (a whole number, or null for the default
60). A frame count above the solver's cap of 120, a window end at or before its
start, a window reaching outside the run, or a frame count larger than the
number of time steps in the window SHALL be refused with a message that names
the value, the run's own numbers, and what to change — never silently narrowed.
A job written before these keys existed SHALL still solve: an absent window key
means the whole run and an absent frame count means the default, exactly as
that job solved before.

#### Scenario: the form's refusals, in words
- **WHEN** a `td` form is posted with a movie ending before it starts, with a
  window past the end of the run, or with 200 frames
- **THEN** the upload is refused with a message naming that value and the run's
  own duration, frame cap or step count, and no job row is created

#### Scenario: a window with fewer steps than frames is refused
- **WHEN** a 20 fs window on a run whose dt is 0.5 fs is asked for 60 frames
- **THEN** the refusal says the window holds 40 time steps and 60 frames were
  asked for

#### Scenario: a job from before the window still solves
- **WHEN** a stored `tds` request has no `movie_start_s`, `movie_end_s` or
  `frame_count` key
- **THEN** the worker solves it over the whole run with 60 frames, as it did
  before this change

### Requirement: The movie's frames SHALL spread over its window and end at its end

The recorded steps SHALL be spread evenly over the requested window, with the
first step at or after the window's start and the last at or before its end,
exactly as the whole-run movie spreads over the run and ends on its final step.
The last frame SHALL be the last step of the window, and the frame count SHALL
be the requested one unless the window holds fewer steps.

#### Scenario: a windowed plan ends at the window's end
- **WHEN** a 1,000-step run records 12 frames over steps 400 to 800
- **THEN** the frames are 12 steps, all between 400 and 800, the last is 800,
  and no two are the same step

#### Scenario: the whole-run default is today's frames exactly
- **WHEN** a movie with no window and no frame count is planned
- **THEN** its steps are the ones `frame_plan` produced before this change, in
  the same order, for every step count

### Requirement: The movie's caption SHALL state the frame spacing and whether it can show the wave

The caption SHALL state the time between consecutive frames in fs (or the
unit that fits) AND in cycles of the source's centre frequency, and when that
spacing is more than half a cycle it SHALL say plainly that the video shows the
envelope rather than the oscillation and that a shorter window shows the wave.
The spacing SHALL come from the job's own recorded frame times, not from a
constant, since a spread integer step list can differ by one step between
pairs.

#### Scenario: a coarse movie says so
- **WHEN** a job's frames sit 3 cycles of its source apart
- **THEN** the caption gives the spacing in fs and in cycles and says the video
  shows the envelope, not the oscillation

#### Scenario: a fine movie does not
- **WHEN** a job's frames sit a tenth of a cycle apart
- **THEN** the caption gives the same two numbers and claims nothing about an
  envelope

### Requirement: Every frame name the solver can write SHALL be accepted by every list of result names

A movie frame SHALL be named `frame_NNN.bin`, one per frame, 1-based, with
enough digits for the solver's 120-frame cap. The worker's upload list, the
app's file route and the worker's own classifier SHALL accept every name the
solver writes, and SHALL still accept a two-digit name, so a job recorded
before this change keeps loading. Near misses that are still refused SHALL
include a one-digit number and a four-digit one. The manifest SHALL state how
many digits its own names use, so a reader never guesses.

#### Scenario: a 120-frame job's names all pass
- **WHEN** a movie has 120 frames
- **THEN** `frame_100.bin`, `frame_120.bin` and `frame_001.bin` are all
  accepted by the worker's upload list and by the app's file route, and all of
  them are classified as movie frames by the worker

#### Scenario: an older job's names still pass
- **WHEN** a job recorded before this change has `frame_01.bin`..`frame_60.bin`
- **THEN** every one of them is accepted and served, and the page asks for
  two-digit names because that job's manifest says two

#### Scenario: a near miss is still refused
- **WHEN** the name is `frame_1.bin`, `frame_0001.bin`, `frame_01.vtu` or
  carries a path separator
- **THEN** the upload list, the file route and the worker's classifier all
  refuse it

### Requirement: The memory estimate SHALL count the frame count the run will record

The memory estimate behind the form's cap check and the worker's own check
SHALL count the requested number of frames (and their float64 |E| working
arrays), not a fixed 60, so a run that would not fit at the requested frame
count is refused before it is queued.

#### Scenario: the estimate grows with the frame count
- **WHEN** the same run is estimated with 60 and with 120 frames
- **THEN** the second estimate is larger by about 60 frames' worth of the
  movie's bytes, and both match the solver's own measured peak

### Requirement: The job page SHALL take its movie's shape from the manifest

The page SHALL read the frame count, the digits of the file names and the frame
times from the job's manifest, so a windowed or 120-frame job plays without any
change to the page, and it SHALL state the loading cost it imposes.

#### Scenario: a windowed 120-frame job plays
- **WHEN** the owner opens a finished job whose movie has 120 frames over a
  window in the middle of the run
- **THEN** the slider spans 120 frames, each frame loads, and the caption's
  time is the frame's own time in the run, not its index times the spacing

### Requirement: The render check SHALL read its numbers from the manifest

The headless render check (`tdcheck.mjs`) SHALL take the probe count, the
units of the object and the frame count from the job's manifest rather than
from constants in the script, so it can run against a windowed 120-frame job.

#### Scenario: the check runs on a 120-frame windowed job
- **WHEN** the render check is pointed at a fixture whose manifest is a
  120-frame windowed movie
- **THEN** it checks every frame the manifest names, captions the last frame
  with the manifest's own count, and passes

### Requirement: A probe's legend SHALL state its position in the unit of the job's own grid

The legend of each probe on a finished time-domain job's page SHALL name the
probe's field component and its position, printing every coordinate in a length
prefix no finer than the job's own grid step -- the finest gap between the
sample positions the manifest records. A coordinate whose magnitude is below a
millionth of that step SHALL be printed as `0`: the stored position is
`origin_m + node`, so a probe on the frame's own origin arrives as what float64
leaves of two grid-sized numbers, and that residue must not choose the unit.
Where the manifest records no sample positions, the position SHALL still be
printed, under the previous rule (the largest coordinate picks the prefix), as
there is no grid to read a step from.

Why: prod job `aa6f083b` stored `[4.0e-20, -3.2e-19, 1.0e-19] m` for a probe on
a 20 nm grid's origin -- a cell-limit residue, ~1e-10 of a cell -- and the old
rule printed it in the prefix the residue needed: "0.0000400 fm", twelve orders
finer than the cell. The snap line sits four orders above the residue and six
below a real node.

#### Scenario: a probe on the frame's own origin

- **WHEN** a finished td job records a probe whose node is the frame origin, so
  its stored position is the remainder of two grid-sized numbers
- **THEN** each such coordinate is printed as `0`, and the others in the grid's
  own unit

#### Scenario: a probe one cell away

- **WHEN** the same job records a probe one grid step from that origin
- **THEN** the coordinate is printed in the grid's unit and is unchanged by the
  snapping rule

#### Scenario: a manifest with no sample positions

- **WHEN** the manifest records no sample positions (a job with no movie)
- **THEN** the legend still states the probe's position, with the prefix the
  largest coordinate picks

### Requirement: The render check SHALL verify the legend's stated position

`tdcheck.mjs` SHALL read the position back out of each legend line through the
page's own length-prefix table, and SHALL require every axis to be the
manifest's own probe node within one grid step (or, where the manifest records
no sample positions, within the three significant digits the legend prints). It
SHALL additionally require a coordinate the manifest puts below a millionth of
the grid step to be printed as `0`, and SHALL state on its output that it is
not checking that rule when it has no step to check it against.

Why: the check ran ALL PASS against a mirror of `aa6f083b` while the page
printed the residue in femtometres -- it verified the component and the swatch
colour, never the number after `at`.

#### Scenario: the production job's origin probe

- **WHEN** the check runs against a mirror of `aa6f083b`
- **THEN** it reads the legend's position as the manifest's node at the frame
  origin and accepts it, and against a page that printed the residue in
  femtometres the same check FAILs, naming the axis

### Requirement: The render check SHALL derive every expectation from the job it is served

`tdcheck.mjs` SHALL compute what it expects of the page from the job it is
served and from the page's own drawing rules, and SHALL carry no constant that
holds for one run's shape only. The probe count and the line and swatch counts
SHALL come from the manifest's `probes`; the time and frequency axis units SHALL
come from the page's own `siPrefix` applied to the run's span and band; the
trace's tick factor SHALL come from the page's own `axisExponent` applied to the
values the check decodes with the page's own `decodeTraces` (accepting both ends
of the page's 5% range padding); whether the run recorded a movie, how many
frames it has and the width of its file names SHALL come from the manifest's
`frames`. The page's plotting module SHALL be imported from the job directory,
so the rules applied are the ones the page was served.

Why: run against a real production job (1 probe, 525 steps at
`dt_s 3.8e-17`, band 250–650 THz), five checks FAILed a correct page — a
fabricated 2-probe line count, a fabricated 2-swatch count, `×10⁻¹³` against the
job's `1.43e-16` peak, `time (ps|ns)` against the page's `fs`, and
`frequency (GHz)` against the page's `THz`. A check that fails a correct page
cannot be pointed at production, which is where it must run.

#### Scenario: a one-probe femtosecond/terahertz production job passes
- **WHEN** the check runs against a mirror of a real job whose manifest names 1
  probe, a femtosecond span, a terahertz band, a `1.43e-16` trace peak and a
  movie with 120 three-digit frames
- **THEN** the per-probe line and swatch counts are checked against 1, the trace
  factor against `×10⁻¹⁶`, the axes against `fs` and `THz`, and every check
  passes because the page is right

#### Scenario: a job with different units and more probes is checked just as strictly
- **WHEN** the check runs against a job whose span picks a different prefix, whose
  band picks a different prefix, or which names more than one probe
- **THEN** the expectations follow that job, and none of the production job's
  numbers is assumed

#### Scenario: a job that recorded no movie is not a failure
- **WHEN** the served manifest has no `frames`
- **THEN** the check verifies the page says "This run recorded no field movie."
  and reports the movie cases as not applicable, instead of checking frames a
  job never wrote

### Requirement: The check SHALL be runnable against a mirrored real job

The check SHALL read a `TD_DIR` laid out as a job page plus what it serves
(`job.html`, `status.json`, `files/spectrum.json`, `files/trace.bin`, the frames
the manifest names, `static/`), and `scripts/tdcheck_fetch.py` SHALL mirror a
real job into that layout from the running app. A job SHALL be named by a full
id or by a prefix that matches exactly one job; an ambiguous or unknown prefix
SHALL be an error, never a guess about which run was mirrored. A fetch that does
not answer 200 SHALL abort naming the URL and the status, and
`scripts/run_tdcheck.sh --dir DIR` SHALL refuse a mirror missing any file the
check reads, so an incomplete mirror is named as such and never read as a page
defect. The runner SHALL NOT write to a mirror it is pointed at.

#### Scenario: a real job is mirrored and checked
- **WHEN** `scripts/tdcheck_fetch.py` is run against the app for a finished
  time-domain job and `scripts/run_tdcheck.sh --dir` is pointed at the result
- **THEN** the check runs the real page with the real assets and result files,
  and reaches an end with every check PASS or a named page defect

#### Scenario: a prefix that names two runs is refused
- **WHEN** the job argument matches two jobs
- **THEN** the fetch exits with an error naming how many it matched, and nothing
  is written

#### Scenario: a mirror with a hole is named, not judged as a page defect
- **WHEN** a mirror is missing a frame the manifest names, or a file the check
  reads
- **THEN** the run stops naming the missing file, and the page is not reported as
  broken

### Requirement: The check SHALL read the movie canvas without waiting on an animation frame

The check SHALL move the movie without any Playwright actionability wait: the
slider's value SHALL be set and its `input` event dispatched inside one
`page.evaluate`, and Play SHALL be a click dispatched in-page. It SHALL read the
movie canvas with a `toDataURL` read of the page's own canvas after forcing
`preserveDrawingBuffer: true` on every WebGL context from the page's first
context, and SHALL count the page's own animation frames so that it reads only
after the loop has drawn a further frame. A render loop that draws no frame
within 90 s SHALL be recorded as a FAIL naming the stalled loop, and SHALL NOT
hang the run.

Why: measured — against a real 58x53x48 movie the run hung twice, at
`locator('#td-scene').screenshot()` and at `page.fill('#td-movie-frame')`
(`Timeout 30000ms exceeded`), because a loaded movie starves
`requestAnimationFrame` and both Playwright calls wait on one.

#### Scenario: a real movie is moved and read to the end of the run
- **WHEN** the check steps the slider, clicks Play, and reads two different
  frames of a real movie
- **THEN** each read is a canvas read that needed no animation frame of its own,
  the two frames differ, and the run reaches its end

#### Scenario: a stalled render loop is a named failure
- **WHEN** the page's loop draws no frame for 90 s
- **THEN** the check records a FAIL naming the loop that drew nothing, and the
  run ends instead of hanging

### Requirement: The mirror SHALL name a movie's frames the way the page does

`scripts/tdcheck_fetch.py` SHALL derive a movie's frame file names from the
served manifest's own `frames.count` and `frames.digits`, and SHALL use the
page's own default width -- two digits, as `viewer.js` defaults -- when the
manifest carries no `digits` key, which is what a movie recorded before
three-digit names looks like. A mirror SHALL name exactly the files the page
will ask for: reading the key outright raises `KeyError` on such a job, so no
mirror is written and the run reports a harness fault.

#### Scenario: a pre-three-digit movie is mirrored

- **WHEN** the served manifest is `{"frames": {"count": 9}}`, with no `digits`
- **THEN** the fetch asks for `frame_01.bin` through `frame_09.bin`, the files
  the page's viewer asks for, and the mirror is complete

#### Scenario: a three-digit movie is unchanged

- **WHEN** the served manifest is `{"frames": {"count": 120, "digits": 3}}`
- **THEN** the fetch asks for `frame_001.bin` through `frame_120.bin`

### Requirement: The runner's cap SHALL follow the movie, and a cut run SHALL say so

`scripts/run_tdcheck.sh` SHALL cut the check at a cap derived from the movie's
own length -- `180 + 12 x frames` seconds, where `frames` is the mirror
manifest's `frames.count` (the solver's own 120-frame maximum when there is no
manifest to read) -- SHALL print the cap it chose, and SHALL let
`TDCHECK_TIMEOUT` override it. A fixed cap is not enough: the check draws every
frame through the container's software renderer, so a correct run outlives
600 s -- the real job 180cb65f (60 frames, two-digit names) was cut after
45 PASS, and the 120-frame open-boundary fixture mirror needed ~740 s on this
loaded VM. A run the cap cuts SHALL be reported as a cut run, never as a page
defect: the runner SHALL print `harness timed out after <cap>s (<frames> frames;
...)` and exit 124, the status `timeout(1)` itself reports, and `tdcheck.mjs`
SHALL print one plain `HARNESS TIMED OUT` line and exit 124 when the signal that
stops it arrives, in place of a stack trace about a closed page.

#### Scenario: a long movie gets a cap of its own

- **WHEN** the runner is pointed at a mirror whose manifest names 120 frames
- **THEN** it prints `cap: 1620s for 120 frames` and cuts the run there

#### Scenario: a manifest that will not read falls back to the solver's maximum

- **WHEN** the mirror holds no readable `files/spectrum.json`
- **THEN** the cap is the 120-frame one, and the run is bounded rather than
  unbounded or cut short

#### Scenario: an operator can raise the cap

- **WHEN** `TDCHECK_TIMEOUT=1234` is set
- **THEN** the runner cuts at 1234 s however long the movie is, and prints that
  number as the cap

#### Scenario: a run the cap cuts names itself and is not a page defect

- **WHEN** the container outlives the cap and is stopped
- **THEN** the runner prints the plain `harness timed out after ...` line naming
  the cap and the frame count, exits 124, and nothing in the output presents the
  run as a failed page

### Requirement: The cell ceiling SHALL be a per-run argument, raisable by the bench script alone

`fdtd.check_limits` SHALL take a third ceiling, `max_cells`, alongside `max_steps` and
`max_seconds`, defaulting to `fdtd.MAX_CELLS` read at call time and not bound at import, and
its refusal SHALL name the ceiling that actually applied rather than the product's constant.
`fdtd.MAX_CELLS` SHALL keep its value and its meaning: what changes is that the bench path can
be measured against a raised ceiling without the product's own moving. The paper-1 bench script
alone SHALL be able to raise it, through `--max-cells N`, and the flag SHALL only raise: a value
under the product's own SHALL be an argparse error, and `run_limits` SHALL clamp with `max()`
so a namespace that never met the parser reads the product's own. The web app and the k8s Job
path SHALL NOT be able to raise it: `worker/td_solve.py` and the web app SHALL call
`check_limits` with no ceiling, and no environment variable and no module global SHALL set one.
A run that used a raised ceiling SHALL say so on its `caps` log line, naming the cell ceiling in
the same parenthetical that names the raised steps and seconds.

The reason the third ceiling is needed at all is the grid's shape: on the paper's own scene a
dx-10 nm record with the open boundary held at r20's distance is 14,048,300 cells, over the 8 M
cap by 75.6 %, and the record that resolves the paper's 4 nm mode is that record at 993 nm --
100 periods, 39,650 steps.

#### Scenario: the product's own default is unchanged and one cell over it is refused
- **WHEN** `fdtd.check_limits` is called with no ceiling on an estimate of 8,000,000 cells
- **THEN** it admits it, and on 8,000,001 cells it raises `EstimateError` naming `8,000,000`,
  so the control is the product's own constant and not a number this change invented

#### Scenario: a raised ceiling is the ceiling the refusal names
- **WHEN** an estimate of 14,048,301 cells is checked against `max_cells=14,048,300`
- **THEN** the refusal names `14,048,300` and not `8,000,000`, because the log line's job is to
  name the ceiling that was in force

#### Scenario: the bench flag is the only way in, and it only raises
- **WHEN** `--max-cells 1000000` is parsed by the bench script
- **THEN** argparse refuses it (exit 2, `may only raise a cap`), and a hand-built namespace
  carrying `max_cells=1_000_000` reads `fdtd.MAX_CELLS` through `run_limits` instead

#### Scenario: the manifest's own record is refused by the product and priced with the flags
- **WHEN** the bench script's pricing arm (the ladder, which voxelises nothing) prices the
  manifest's own record -- `--hex paper --wavelength 993 --dx 10 --mirror-pad 600 --periods 100
  --pad-nm 320 --pml-nm 200`
- **THEN** with no bench flag not one rung is priced, each refused on every ceiling it exceeds --
  the ladder prints only the first 64 characters of a refusal, so the dx-10/pad-600 row reads
  `refused: The grid has 14,048,300 cells, over the limit of 8,000,000` and dx-12/pad-600 reads
  8,952,300 cells against the same limit, while dx 20 and 15 (and dx 12 at pad 400) lead with
  `The run needs 39,650 time steps, over the limit of 20,000` -- and with `--max-cells 15000000
  --max-steps 200000 --max-seconds 18000` the dx-10 row is priced at 14,048,300 cells,
  39,650 steps and 13,662.4 s in 1.63 GiB

#### Scenario: the wide-scan call sites do not diverge on which ceiling is in force
- **WHEN** the bench script's `check_limits` call sites are read
- **THEN** each hands its ceiling over (found by `re.findall`, each passing `**`), so the
  estimator's rungs and the run path cannot be priced against two different cell ceilings

### Requirement: The open boundary SHALL be nameable as a distance, so a resolution change is only a resolution change

`fdtd_layout.plan` and `fdtd_layout.layout` SHALL take `pad_cells` and `pml_cells` as
keyword-only arguments, defaulting to `fdtd_layout.OPEN_PAD_CELLS` and `fdtd.DEFAULT_PML_CELLS`
read at call time, and SHALL return the pair they planned with. Only the paper-1 bench script
SHALL pass them, through `--pad-nm NM` and `--pml-nm NM`, which name a DISTANCE: the pad is a
cell count, so the product's own pad is 320 nm at dx 20 and 160 nm at dx 10, and a dx-10 record
priced without these flags would move the open boundary at the same time as the grid's
resolution, leaving a difference from r20 attributable to neither. A distance that is not a
whole number of cells SHALL be rounded to the nearest cell, and a run SHALL report the boundary
it held in both cells and nanometres. The bench script SHALL refuse a distance under one cell,
SHALL refuse an absorber that is not inside the pad, and SHALL refuse a run that names an
absorber both in cells and in nanometres unless the two are the same cell count.

#### Scenario: the held distance is r20's own boundary, measured at r20's own dx
- **WHEN** the planner is asked for dx 20 with `--pad-nm 320 --pml-nm 200`
- **THEN** it returns the same grid as dx 20 with no flags at all -- 2,903,040 cells, 16 pad
  cells, 10 absorber cells, the box 2318.8 x 2318.8 x 2177.0 nm -- so the distance this change
  holds for dx 10 is the boundary r20 measured F = 4.38 on

#### Scenario: held at dx 10, r20's box is held and only the resolution moves
- **WHEN** the planner is asked for dx 10 with `--pad-nm 320 --pml-nm 200`
- **THEN** the box is r20's own 2318.8 x 2318.8 x 2177.0 nm, held by 32 pad cells and 20
  absorber cells, and the axes take 244 x 245 x 235 nodes against r20's 144 x 144 x 140:
  14,048,300 cells against 2,903,040 (4.84x) at an identical 39,650 steps and 8.353997e-18 s,
  because `dt` is set by the oxide's own 3.000 nm, which does not scale with `--dx`

#### Scenario: the product's boundary is not reachable from the product's own path
- **WHEN** the web form and the worker build their request and call `fdtd_layout.layout`
- **THEN** neither passes a pad or an absorber and the planner reads its own defaults, so a
  product run's boundary is the product's

#### Scenario: the three boundary refusals
- **WHEN** the bench script is asked for `--pad-nm 8` at dx 10, or for `--pml 20` together with
  `--pml-nm 100` at dx 10, or for `--pad-nm 100` with the product's own absorber
- **THEN** each is refused with exit 2 -- `under one cell at dx 10`, two absorbers that are not
  the same cell count (`--pml 20` against 10 cells, named as different absorbers), and an
  absorber that is not inside the pad (`the 10-cell absorber is not inside the 10-cell pad
  (100 nm in 100 nm)`) -- and no scene is built

### Requirement: The paper-1 ring-down fit SHALL scan omega finer than any mode's basin

`fit_decay` in `scripts/paper1_nanowire.py` SHALL scan its 31 log-spaced
taus and, at each tau, every omega of the band on zero-padded FFT bins at
most 1/(`DECAY_PAD` = 8) windows apart and never coarser than a twentieth of
the band. The scan SHALL only choose basins; every reported number SHALL come
from exact least squares.

#### Scenario: the cw-settling 1060 fs case on a synthetic tail
- **WHEN** a tail of 3,000 samples at 0.15 fs (a 450 fs window) holds one mode
  of tau 1060 fs midway between two of the old 21 columns of a +-15% band at
  331.3 THz, with 1e-3 of noise
- **THEN** tau is within 0.2% of 1060 fs (measured 1060.35 fs. The old search
  returned 108.05 fs with rms 0.38 of the peak. 40 noise seeds scatter by
  0.271 fs with a 0.004 fs bias), rms is under 2e-3 of the peak, and the fit
  is not ambiguous

#### Scenario: a sweep of linewidth against the old grid step
- **WHEN** tails with tau from 60 fs to 4000 fs (a line 0.43 to 0.016 of the
  old 4.97 THz step) sit at 0, 0.33, 0.5 and 0.83 of the way between two old
  columns, with 1e-3 of noise
- **THEN** every tau is within 1% of its truth. Measured over 60 such tails,
  the worst was 0.19%. The old search got 33 of 60 wrong by 14.6% to 97%.
  Its descent multi-started from the best 3 or 5 cells got 33 and 31 wrong.
  This search got none

#### Scenario: long, quiet lines need bins finer than the window
- **WHEN** tails with tau 1060, 4000 and 8000 fs under 1e-4 of noise sit
  0.083 to 0.75 of the way between old columns
- **THEN** every tau is within 1%. Measured over 60 such tails (700 to 8000
  fs, 12 places): bins 1/window apart left 5 wrong, 12.7% to 18.3% short with
  f right. Bins 1/(2, 4 or 8 windows) apart left none (worst 0.04%), all at
  about 550 ms a fit

### Requirement: The ring-down fit SHALL descend inside the chosen basin

`fit_decay` SHALL descend exactly from the best cell: tau within two grid
cells of the cell's, and omega at each tau within the larger of two bins and
the cell's own line (1/tau rad/s) either side, clamped to the band.

#### Scenario: a line wider than the band
- **WHEN** a 1.9 fs decay (the `--tiny` vacuum tail's shape: 2,571 samples
  of 0.02322 fs, a +-8% band at 333.1 THz) sits at 315 or 332.6 THz under
  1e-4 of noise
- **THEN** f is within 0.2 THz and tau within 0.5%, and rms is under 1.2e-4.
  Measured: within 0.02 THz and 0.05%, as the old search was. With omega held
  to two bins the fit was 6 to 31 THz off at rms 34 to 160 times the noise.
  The `--tiny` vacuum tail itself re-fits to 332.5800 THz and 1.9008 fs,
  against the old search's 332.5797 THz and 1.9009 fs at the same rms

### Requirement: The ring-down fit SHALL refine separated minima beside the fit

`fit_decay` SHALL refine up to `RIVALS` = 3 local minima of the scan, lowest
first, whose omega is separated from the fit's by more than twice the larger
of 1/(pi tau) and 1/window (times 2 pi). One that refines lower SHALL become
the fit. `rival` SHALL hold the best remaining one: its f, tau, and residual
over the fit's.

#### Scenario: the scan ranks a wrong cell first
- **WHEN** the scan's residual at one cell 12% below a 400 fs mode is forced
  under its minimum
- **THEN** the fit is still the mode within 1%, and that cell is the rival

### Requirement: The ring-down fit SHALL say when its tau is not a measurement

`ambiguous` SHALL be a reason, not None, when the rival's ratio is under
`AMBIGUITY_RATIO` = 12 or when `tau_err` is None. An undetermined `f_err`
SHALL NOT make a fit ambiguous: the pinv drops omega's direction for scale
alone, so it is None on well-fitted tails too.

#### Scenario: how the factor was measured
- **WHEN** 105 random tails are fitted: 60 two-mode records with a second
  amplitude of 0.05 to 1, and 45 one-mode records under 0.01 to 1 of noise
- **THEN** 16 fits were more than 10% off their mode's tau, and 14 of them had
  a rival at 1.05 to 11.1 times the fit's residual, so 12 is the round factor
  over those. The two it misses are a second mode at 0.12 amplitude whose
  rival was 122 times (12.2% off), and one at 0.25 amplitude that formed no
  separated minimum (12.8% off). 15 of the 60 two-mode tails formed none. The
  price is 29 fits within 10% flagged as well; the first flagged one-mode
  tail was at 0.16 noise (ratio 11.5, tau 1.7% off). The `rms ... against a
  ... peak` reading stays the check for a second mode the flag cannot see

#### Scenario: two modes of equal amplitude
- **WHEN** a tail holds two 1060 fs modes 10 THz apart, or a 400 fs and a
  250 fs mode 10 THz apart, of equal amplitude
- **THEN** the fit is ambiguous with a second minimum at the other mode, and
  the rival's residual is 1.000 and 1.48 times the fit's (measured)

#### Scenario: noise, alone or over a mode
- **WHEN** a tail is unit noise alone, or a 1060 fs mode under noise of its own
  amplitude
- **THEN** both are ambiguous, at ratios 1.0006 and 1.34 (measured. Noise alone
  over 16 seeds gave 1.000 to 1.003)

#### Scenario: the factor from both sides
- **WHEN** a 400 fs mode carries a 250 fs mode at 0.6 of its amplitude 10 THz
  above it, or at 0.2 of its amplitude 25 THz above it
- **THEN** the first is ambiguous, at ratio 4.11 with its tau 24% long. The
  second is not, at ratio 36.8 with its tau 3.3% long

#### Scenario: a degenerate tau direction
- **WHEN** the tail is constant (2,000 samples of 1e-9)
- **THEN** `tau_err` is None and `ambiguous` names the degenerate covariance

### Requirement: The ring-down arm SHALL print an ambiguous tau as its reason

When `fit_decay` returns `ambiguous`, the arm's `fit` line SHALL print
`tau AMBIGUOUS (<tau> fs is not a measurement: <reason>)` in place of
`tau = <tau> fs +- <err>`. The arm SHALL then print `unresolved: the fit is
ambiguous -- <reason>` and no Q and no three-sigma bound, and return 3. A fit
without the key, or with it None, SHALL print exactly as before.

#### Scenario: the arm on a canned record
- **WHEN** the arm (`_run` stubbed, everything after it real) reads two equal
  1060 fs modes 10 THz apart
- **THEN** it prints `tau AMBIGUOUS (` with the second minimum, no `Q` line
  and no `Q >`, and returns 3. On one clean 400 fs mode it prints
  `tau = ...` and its Q, and returns 0

### Requirement: The memory ceiling SHALL be a per-run argument, raisable only by the bench script

`fdtd.check_limits` SHALL take a fourth ceiling, `max_bytes`, alongside `max_steps`,
`max_seconds` and `max_cells`. Its default SHALL be `fdtd.MAX_BYTES` read at call time, and
its refusal SHALL name the ceiling that actually applied rather than the product's constant.
`fdtd.MAX_BYTES` SHALL keep its value (3.0e9 bytes) and its meaning: the product's own
ceiling is unchanged, and only a measured bench card is measured against a raised one.

The paper-1 bench script alone SHALL be able to raise it, through `--max-bytes GB`
(gigabytes, so `--max-bytes 14` is 14e9 bytes). The flag SHALL only raise: a value under the
product's own SHALL be an argparse error (exit 2), and `run_limits` SHALL clamp with `max()`
so a namespace that never met the parser reads the product's own. The web app and the k8s
Job path SHALL NOT be able to raise it: `worker/td_solve.py` and the web app SHALL call
`check_limits` with no ceiling, and no environment variable and no module global SHALL set
one. A run that used a raised memory ceiling SHALL say so on its `caps` log line, naming the
memory ceiling in the same parenthetical that names raised steps, seconds and cells.

The refusal this lifts was measured: the paper's Run B at dx 5 nm on the 16 GB A4000
estimates 11.55 GB of GPU memory and was refused after about two hours of setup under the
product's 3.0 GB ceiling.

#### Scenario: The default refuses what the card cannot do under the product's ceiling

- **WHEN** an estimate of 11.55e9 bytes is checked with no `max_bytes`
- **THEN** it SHALL be refused, and the refusal SHALL name "over the limit of 3.0 GB"

#### Scenario: A raised ceiling is the one the refusal names

- **WHEN** the same estimate is checked with `max_bytes=14e9`
- **THEN** it SHALL be admitted
- **AND** an estimate of 14e9 + 1 bytes SHALL be refused, naming "over the limit of 14.0 GB"
  and not the product's 3.0 GB

#### Scenario: The flag only raises

- **WHEN** the bench parser is given `--max-bytes 2`
- **THEN** it SHALL exit 2 with "may only raise a cap"
- **AND** a namespace carrying `max_bytes=1e9` SHALL resolve through `run_limits` to
  `fdtd.MAX_BYTES`, exactly as the other three ceilings do

#### Scenario: A memory-only raise is reported as a bench ceiling

- **WHEN** `report` runs with only `max_bytes` raised
- **THEN** its `caps` line SHALL carry the bench-ceiling parenthetical and name the memory
  ceiling (for example "14.0 GB")
- **AND** with no ceiling raised the same line SHALL carry no parenthetical

### Requirement: The CW arm's settling forecast SHALL follow the curve's slow tail

`classify_settling` in `scripts/paper1_nanowire.py` SHALL keep the
whole-curve fit (`whole`) and SHALL refit the forecast (`fit`) over the
fitted cuts in the record's last `TAIL_SHARE` = 0.5 (at least 5) when the
whole-curve residual is over `SIGNIFICANCE` = 3 x the floor. Otherwise the
forecast SHALL be the whole-curve fit itself. The class, the class line's
F_inf and the locator lines SHALL read the forecast.

#### Scenario: the 600-period run's curve
- **WHEN** the 20 cuts the run printed (31 to 600 periods, F 3.31597 to
  13.26, still rising) are classified with the run's floor (0.0552 / 2.83e4)
- **THEN** the whole-curve fit is the one the run printed (13.1935 +- 0.020,
  82.99 periods, 2.83e4 x the floor), and the forecast is the fit over the 10
  cuts from 338 periods: F_inf = 13.3068 +- 0.0039, tau_s = 115.1 periods,
  at or above the 13.26 already reached

#### Scenario: sums of two exponentials with known limits
- **WHEN** 1 - 0.7 exp(-t/fast) - 0.3 exp(-t/slow) is cut 20 times 30 apart
  (fast 10 or 25, slow 80 to 300) under 1e-5 of white noise, floor 1e-5
- **THEN** the forecast is within 3 of its errors and within 5e-4 of 1, and
  the whole-curve fit is at least 1e-3 off. Measured over 20 seeds x 6
  shapes: within 2.53 errors and 1.73e-4, against 1.6e-3 or more for the
  whole curve, and never flagged

#### Scenario: one clean exponential is not refit
- **WHEN** 1 - 0.8 exp(-t/tau) with tau 60, 100 or 200 is cut the same way
  under 1e-5 of noise, floor 1e-5
- **THEN** the forecast is the whole-curve fit, F_inf and tau_s value for
  value. Measured over 900 such records, the whole-curve residual reached
  1.44 x the floor, so none was refit. The cw-settling 30-period approach
  keeps tau_s = 30 periods and F_inf within 1e-5, unchanged

#### Scenario: the cw-settling two-time-constant record (a changed expectation)
- **WHEN** 1 - 0.5 exp(-t/10) - 0.3 exp(-t/60 periods) over 200 periods is
  read through the arm
- **THEN** the whole-curve fit still has tau_s between 10 and 60 (37.4) and
  a residual 1.1e4 x the floor, so the "more than this one time constant"
  words still print, on the refit line now. The forecast's tau_s is the slow
  60 periods within 5e-3 (measured 59.91) and its F_inf is the limit within
  1e-4 (measured 3.2e-5 under, where the old one was 1.72% under, below the
  end's 1.14%). The test now asserts the old bounds on `whole` and these on
  `fit`

#### Scenario: noisy records are refit too (a changed expectation)
- **WHEN** the cw-settling coverage records (200 noisy 30-period approaches
  over 60 periods, noise 0.05 of the in-phase amplitude) are classified
- **THEN** their noise is far above the phase floor, so all 200 are refit,
  and both the whole-curve fit and the forecast keep the old bounds: a z
  spread between 0.75 and 1.4, and 95% or more inside 3. Measured over 6
  seeds: the whole-curve fit 0.93-1.21 and 97.5-99.5%; the forecast
  1.12-1.34 and 96-99%, widest on the test's own seed

### Requirement: The CW arm SHALL refuse a forecast its own record contradicts

`forecast_check` SHALL call an approach's forecast inconsistent when the
last third's line moves by more than 3 of its errors and F_inf lies behind
the end's reading, below it on a rise or above it on a fall, by more than
`INCONSISTENT_SIGMA` = 2.5 of hypot(F_inf's error, the floor). The class
line SHALL then say INCONSISTENT, name the end and the gap, and print no
extrapolation.

#### Scenario: the 600-period run's old forecast
- **WHEN** the run's whole-curve forecast (13.1935 +- 0.020) is checked
  against its rising end (13.26), and against the same curve mirrored to fall
- **THEN** both are inconsistent at 3.33 of their errors, and the tail
  forecast of either is not

#### Scenario: correct forecasts are not refused
- **WHEN** single exponentials (tau 60 to 250 over 600, noise 1e-5 to 1e-3 of
  the limit, the end within noise of it at the longest) are classified
- **THEN** none is inconsistent. Measured: the wrong-side gap reached 1.92
  errors over 1623 such records, and 0.998 over 206 noisy series through the
  reading. Of 61 two-exponential whole-curve forecasts below their end, 30
  were caught at 2.5 (19 at 3), and the 600-period one at 3.33

#### Scenario: the end's error is the floor, not the last third's scatter
- **WHEN** the gap's error is taken from the fit's own per-point sigma
- **THEN** the 600-period forecast reads 1.05 errors behind its end and is
  not caught. The last third's line scatter, 0.0138 there against a floor of
  2e-6, is the curve's bend, not the end's noise

#### Scenario: the printed line
- **WHEN** the 600-period curve is replayed through the reading ten times
  faster (60 periods, 3,862 samples)
- **THEN** the refit line names the last 50% (11 cuts from 30 periods), the
  forecast lies 0.56% above the end where the whole-curve one lay 0.16%
  below, and the class line gives it as the extrapolation. With the refit off
  and the threshold at 0.5, the line says INCONSISTENT and prints no
  EXTRAPOLATION

### Requirement: The CW arm SHALL say when its forecast is unreliable

The class line SHALL call an approach's forecast UNRELIABLE when the
forecast fit's residual is over `FORECAST_RESIDUAL` = 3 x the floor. It SHALL
say that what was fitted still holds more than one time constant, or noise
the floor does not account for.

#### Scenario: the threshold
- **WHEN** 215 tail forecasts of two-exponential curves (fast 5 to 50, slow
  80 to 300, noise 1e-6 to 1e-4) are compared with their known limits
- **THEN** the 167 within 3 errors of the limit reached 2.74 x the floor and
  none is flagged. Of the 48 further off, 30 are flagged at 3, 26 at 5 and
  11 at 30. The other 18 sit at 0.72 to 1.72 x the floor (16 of them fast 50
  against slow 80 to 300), so no residual test can see them. The cw-settling
  two-time-constant forecast, 3.7 errors off, is flagged at 3.84

#### Scenario: the 600-period run's curve
- **WHEN** its tail forecast is classified
- **THEN** its residual is 1029 x the floor, so it is UNRELIABLE (and not
  inconsistent)

### Requirement: Each forecast rule SHALL be held by a test a reversion turns red

`tests/check_discriminating.py` SHALL hold a `tail-*` case for each rule
above. Each case SHALL apply, and SHALL turn exactly its named tests in
`tests/test_cw_settling.py` red.

#### Scenario: the cases and the re-measured cw-settling sets
- **WHEN** each of the 12 `tail-*` and 13 `settling-*` cases is run
- **THEN** each turns exactly its set red. Four `settling-*` sets moved. Free
  phase adds the two-time-constant test, whose forecast now has to land
  within 1e-4. Ramp-fitted drops from 8 tests to 4: the ramp's misfit sends
  the forecast to the last half, past the ramp. Limit-sign grows to 12, since
  every new test reads F_inf. Misfit-as-noise now reverts on `whole`'s
  residual (the forecast's is small) and turns 6 red. Independent-cuts
  follows `fit` to `whole`, its set unchanged

### Requirement: A second gold model SHALL fit Palik's table

The metal library SHALL offer `Au-palik`, a Lorentz-Drude `Metal` fitted to
E. D. Palik's handbook gold table over the table's rows where the model may be
asked to answer, so a validation run can be compared against the paper's own
source. The model SHALL be smooth as well as accurate: it is fitted under a
damping floor and judged on a dense grid between the table's rows, because a
pole that fits every row can still ring between two of them and fake a high-Q
mode in a ring-down. The fit's range SHALL be the rows it is fitted to
(1.10-1.90 eV, 653-1127 nm). The table's three rows past 1150 nm (1239.8,
1265.1 and 1291.5 nm) are not fitted: under the damping floor and the
smoothness envelope the model misses them by 4.12%, 19.18% and 20.37% against
the 3% gate, and their k falls then rises (8.77, 8.09, 8.25), so the fitted
range ends below them and a band that reaches them is refused. The dense-grid
criterion, not row accuracy, is what they are held to. The Rakić library and
its `Au` entry SHALL be unchanged, and every place that accepts a library
metal (the `_metal` lookup, `METALS`, the layout and voxel material checks,
and the skin-depth rule) SHALL accept the new name.

#### Scenario: the fit reproduces the table

- **WHEN** the model's complex permittivity is compared with every copied table
  point in the model's fitted range
- **THEN** every point is within 3% relative error, and the test states the
  measured maximum, where it falls, and the fitted range

#### Scenario: the model is smooth between the table's rows

- **WHEN** the model is evaluated every 2 nm over 600-1400 nm
- **THEN** Re eps is negative and Im eps positive at every point, and between
  the table's own rows the model's |eps| is within 1.5x of the linear
  interpolation between the neighbouring rows
- **AND** no pole has a damping below 0.05 eV, so nothing in the model can ring
  between two table rows

#### Scenario: the model is lossy, positive and stable

- **WHEN** the model is evaluated across the fitted band and its ADE term is
  checked at the bench's spacings
- **THEN** Im eps is positive at every point, every pole has f > 0 and
  Gamma > 0, and the growth factor is below 1 + `GROWTH_TOLERANCE`

#### Scenario: the Rakić library is untouched

- **WHEN** the library is listed
- **THEN** `METALS` contains both `Au` and `Au-palik`, and the published Rakić
  check still passes against `RAKIC_1998` with the same values

#### Scenario: a band outside the fit is refused naming its source

- **WHEN** a pulse band leaves `Au-palik`'s fitted range
- **THEN** the refusal names Palik's table and the range, as the Rakić
  refusals name Rakić

### Requirement: The bench SHALL choose the gold model by name

`scripts/paper1_nanowire.py` SHALL take `--gold-data {rakic,palik}`, defaulting
to `rakic` so earlier runs reproduce, SHALL use the choice wherever gold is
selected (the scene's mirror material, the reported skin depth, and the
estimate), and SHALL log which model the run used.

#### Scenario: the flag selects the material

- **WHEN** the script is parsed with and without `--gold-data palik`
- **THEN** the scene's mirror material is `Au-palik` and `Au` respectively, the
  estimate prices the same material, and the report names the model it ran

### Requirement: The run path SHALL check a pulse's band against its metals

`fdtd.source_band` SHALL be the single definition of a pulse's band: the
spectrum's width above `BAND_FLOOR` of its peak, `cw` and `drawn` having none,
and a band that reaches zero frequency being refused. The web form and the
worker SHALL each call `fdtd.check_scene_band` on the library metals the scene
names (parts and coatings), so a band outside a metal's fitted range is
refused where a run is accepted, not only where the bench runs. The paper-1
script, whose run is the one the refit protects, SHALL judge its mirror through
the same `fdtd.source_band` call and the same `check_metal_band` refusal, so
the same pulse is accepted or refused with the same numbers and the same
sentence as in the app; its `cw` and `drawn` sources ask nothing, as the app's
do. `pec` and refractive indices are not fitted models and SHALL NOT be
checked.

#### Scenario: the app refuses a band its metal cannot answer for

- **WHEN** an upload names a library metal and a pulse whose band leaves that
  metal's fitted range
- **THEN** the request is refused before it is queued, and the refusal names
  the metal, its source table and its range

#### Scenario: the worker refuses the same job

- **WHEN** a job reaches the worker with a band outside a named metal's fit,
  without having passed the form
- **THEN** the run is refused as a `TDSolveError` in the metal's own words

#### Scenario: a source without a declared band asks nothing

- **WHEN** the source is `cw` or `drawn`
- **THEN** the guard passes any named metal

#### Scenario: the paper-1 run refuses a band its gold cannot answer for

- **WHEN** the paper-1 script is given a pulse band and a gold model whose fit
  does not cover it
- **THEN** the scene is refused before it is built, with the same sentence the
  app gives for that pulse, and a `cw` source or a `--vacuum` control is not
  refused. At 900 nm the Palik fit accepts a pulse up to 83 nm of nominal
  half-width (band floor 1.103 eV) and refuses 84 nm (1.099 eV); Run C's
  `--band-half 100` (1.045-1.710 eV) is therefore refused under this guard,
  and whether the run narrows its pulse or the fit widens is not this
  change's call.

#### Scenario: the bench and the app share one band

- **WHEN** the bench computes the band it judges a metal against
- **THEN** it calls `fdtd.source_band`, the same function the app's guard uses,
  rather than spelling the formula out

### Requirement: Reports SHALL name the model that ran

Labels and reports that describe a dispersive metal SHALL name the library or
the selected model's source, never one specific table for every metal.

#### Scenario: the bench names its model

- **WHEN** the bench runs with any library metal
- **THEN** its header names that metal and the table its model was fitted to

### Requirement: The ring-down arm SHALL be able to record probes on the wire's axis, and their traces SHALL travel with the saved record

`scripts/paper1_nanowire.py --ringdown --save-tail PATH --axis-probes` SHALL add one probe per E component (`x`, `y`, `z` -- never a magnitude) on every z node of the wire's axis from the first node clear of the gold's top face (a millionth of a cell above it: the layout's point-in-part test reads a node on the face as on the mirror, whose field is zero) to three coarse cells above the wire's top facet, the in-plane position being the source's own node (which is the wire's axis: the layout builds the scene around the source's corner). The probes SHALL be placed from a `fdtd_layout.plan` taken before the voxeliser runs -- a probe must sit on a node, and only the plan's node list says where the nodes are -- and the run SHALL compare the plan's nodes with the layout's before stepping and refuse if they differ, so the line is on the nodes the run uses and not merely near them. The plan SHALL NOT pin a z line into the grid for the probes: a pinned line grades the axis it is added to, and probes placed after the grid would move it.

The traces SHALL be written into the `--save-tail` record beside the field: `probe_traces` (one row per probe), `probe_z_m` (each probe's node z read off the layout's nodes at the cell the layout placed it on, in the frame `--qd-below` uses), `probe_components`, the wire's own z span (`wire_z_m`) and the emitter's z (`emitter_z_m`). The writer SHALL take the layout the solver actually ran on -- whose probes are `(i, j, k, component)` four-tuples and which SHALL carry the parts' own boxes as `part_bounds`, passed through from its plan -- so the record's abscissa and its wire span come from the layout that was solved and not from a second plan of the same scene. `--axis-probes` SHALL be refused, before any scene is built, without `--ringdown` and without `--save-tail`, since the traces are read over the tail and would otherwise be recorded and thrown away -- except with `--estimate`, which prices the line and records nothing.

#### Scenario: a probe is on the node it names, on a graded axis
- **WHEN** the line is planned on Run C's own scene (dx 10 nm, the wire 162-1537 nm above the origin)
- **THEN** it holds 147 z nodes, 441 probes, from z 153 nm (the first node clear of the face at 150 nm) to z 1557 nm, and every probe's stored z is exactly the node coordinate it was built from (measured worst offset 0.0 m)

#### Scenario: the line ends where the geometry does
- **WHEN** the line is built for a scene whose wire top is at `hi` and whose gold top is at `lo`
- **THEN** its first z is the first node above `lo` by more than floating-point noise and its last is the last node at or below `hi + 3 dx`, and neither reaches into the absorber (`pml <= i <= len(nodes) - 1 - pml`), so a scene with no room is refused rather than probed inside a damped field

#### Scenario: the record is written from the layout that was solved
- **WHEN** the writer is given the layout the tiny (20 nm) scene's `solve_scene` returns, and its output is saved and read back
- **THEN** it writes one trace row per probe with that layout's own node z and component per row, the wire's span from the layout's `part_bounds` and the emitter's z from its source node, and `--axial-profile` on the written file prints one layer per z node (measured: 48 probes, 16 layers on that scene) -- the writer-to-reader path, not a hand-made record

### Requirement: The probes SHALL be priced in every estimate of the run that records them

`fdtd.memory_estimate` and `fdtd.estimate_run` SHALL take `probes` (and `probe_steps`, defaulting to the run's own step count) and count `4 x probes x probe_steps` bytes, the same four bytes a float32 trace is, because that is what the run holds in device memory and what `MAX_BYTES` must see. `fdtd_layout.estimate` SHALL pass the planned layout's own probe count, and the worker SHALL pass its layout's, so the product's memory check prices the traces it will record. The bench script's ladder SHALL price the same line the run will place, by building it from the plan through one function both paths call.

#### Scenario: a run's estimate grows by its traces
- **WHEN** `fdtd.estimate_run` is called on the same nodes with `probes=0` and with `probes=n`
- **THEN** the two `bytes` differ by exactly `4 n steps`, and `dt` and `steps` do not change: probes cost memory and no time

#### Scenario: the ladder prices the run's own line
- **WHEN** `--estimate --axis-probes` is read
- **THEN** each rung's bytes include `4 x 3 x layers x steps` for the line that rung's plan would place

### Requirement: The axial profile SHALL be read from the saved record, with no second run

`scripts/paper1_nanowire.py --axial-profile PATH [--axial-frequency-nm NM] [--axial-png OUT]` SHALL read a `--axis-probes` record and print, for every z layer, the power `|E|^2` of each component and their sum, normalised to the line's largest total, at the record's own drive frequency unless `--axial-frequency-nm` is given. Each probe's amplitude SHALL be the windowed DFT of its trace over the part of the record after the saved `start` step (the drive-free tail), not over the whole record. The profile SHALL also report the z of every interior local maximum of the summed power inside the wire's own span -- read from the record's `wire_z_m`, not retyped -- and a maximum at either end of the line SHALL NOT be reported as one, since it is the edge of the sampled range and not a maximum of the profile. A record without probe traces SHALL be refused with a message naming what it lacks. The PNG, which marks the wire's ends and the emitter's z from the record, SHALL be a convenience: a machine without matplotlib SHALL lose the picture and print the number.

#### Scenario: a standing wave's antinodes are where the physics puts them
- **WHEN** the record's traces are a standing wave of known wavelength and phase, sampled at the line's nodes
- **THEN** the reported maxima sit on the antinodes the wavelength and phase decide, within half a cell, and the count matches -- with the fixture's phase chosen so no antinode is at the centre of the line, so a reader that reported the centre would fail

#### Scenario: the profile reads the tail it was told to
- **WHEN** the same trace is read with the record's own `start` and with a later one
- **THEN** both are read from their own first step, and a `start` past the record's end is refused rather than silently reading nothing

#### Scenario: a record with no probes is refused, not read as empty
- **WHEN** `--axial-profile` is given a `--save-tail` record written without `--axis-probes`
- **THEN** it exits non-zero naming the missing traces, and `--axial-profile` together with any arm that would build a scene (`--ringdown`, `--estimate`, `--spectrum`, `--tiny`, `--save-tail`, `--save-series`, `--from-series`, `--collection-na`, `--locator-tau`, `--axis-probes`) SHALL be refused in `arm_conflicts`, so a flag that cannot be read is never accepted and ignored
