# maxwell-solver

## ADDED Requirements

### Requirement: A multi-material cavity SHALL be meshed to a per-region target edge

Each region SHALL be meshed to its own target edge

    edge_r = max(min(h * n_wall / n_r, thickness_r), h_min)

in METRES, where `h` is the wall's edge from the enclosed volume, the index ratio
is the physics (a region denser than the wall is finer; one less dense is not
coarser than the wall), `thickness_r` is the region's thickness clamp, and
`h_min = min_r` of the ramp `min(h, h * n_wall / n_r)` is the FINEST region's
edge. The floor at `h_min` SHALL guarantee the finest region is never moved, so
the finest region is held to exactly the edge the uniform rule gives it today and
the resolution the cavity is solved at does not move.

The per-region edges SHALL come from ONE helper (`region_edges`, reached through
`region_plan`) that both the mesher and the web estimate read, so the edge a
region is built to and the edge the route was decided on cannot drift apart. That
helper SHALL take and return METRES; the mesher converts to the upload's units at
its own boundary.

The mesher SHALL realise the per-region targets with TetGen's mesh-sizing
FUNCTION -- the `-m` switch, reached through the library's PUBLIC
`tetrahedralize(bgmesh=<pyvista.UnstructuredGrid>)` route, the grid carrying each
node's target in point data `target_size` on linear tetrahedra -- and SHALL NOT
pass `fixedvolume` or a positive `maxvolume` alongside it: either one overrides
the field and silently meshes uniformly, and the mesher SHALL refuse to build a
field whose switch set carries them. The background grid SHALL be packed by the
public route (`pyvista.UnstructuredGrid` with point data `target_size`), not by
hand: a background grid TetGen's field does not recognise was measured to ABORT
the process (SIGABRT, not a Python exception, so not catchable) rather than mesh
ungraded, and the public route is the one whose input contract is documented.
Where the per-region edges are all equal, the mesher SHALL use the existing
fixed-volume path unchanged, so an equal-index or one-part mesh is the mesh of
today byte for byte. Source: `dockerfiles/maxwell_solver/mse_common.py`,
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: a cavity whose core is denser than its wall
- **WHEN** the upload's regions do not all share one index
- **THEN** the wall's elements are coarser than the core's
- **AND** each region's achieved median edge is within the overshoot allowance of
  ITS OWN target edge, and the finest region's within that of `h_min`
- **AND** neither `fixedvolume` nor a positive global `maxvolume` accompanies the
  field

#### Scenario: every region shares one index
- **WHEN** a multi-part upload's regions all have the same index
- **THEN** the per-region edges are equal, `graded` is false, and the mesh is
  built by the existing fixed-volume path, byte for byte

#### Scenario: a single-material upload
- **WHEN** the upload has one part
- **THEN** nothing changes: the grading helper is not consulted and the mesh is
  the mesh of today

### Requirement: A region's target SHALL be clamped under its own thickness

A region's target edge SHALL be clamped to `CLAMP_THICKNESS_FRACTION * 2 *
V_filled_r / (A_boundary_r * scale^2)` in METRES, where `V_filled_r` is the
region's FILLED volume, `A_boundary_r` is its own surface area plus every child's
(in surface units), and `scale` converts the area to square metres. The clamp
SHALL only ever lower a target toward FINE (`min` against the ramp), and the
floor at `h_min` SHALL re-raise it, so the clamp can never coarsen a region and
never moves the finest region. When the clamp lowers a region's target, the job's
caveats SHALL say so, naming the region, the target it was given and the edge the
ramp asked for. Source: `dockerfiles/maxwell_solver/mse_common.py`.

Measured basis: the sizing field stops being honoured when a target approaches
the thickness of the region it describes. On three nested-cube fixtures the mesh
collapsed -- the FINE region losing its resolution too -- at 1.9-3x the shell's
thickness, while a target under the thickness held. The background mesh's own
RESOLUTION was tested as the cause and refuted: 0.22 against 0.08/0.05/0.04 gave
identical collapses (128 elements, core edge 0.1768 in every case), so the clamp
is on the target, not on the background mesh's step.

#### Scenario: the coarse target exceeds the region's thickness
- **WHEN** the graded edge for a region exceeds its clamp
- **THEN** it is lowered to the clamp and the job's caveats name the region
- **AND** it is never lowered below `h_min` -- the finest region's edge, the one
  a uniform mesh gives today -- so no clamp can coarsen the cavity's resolution

#### Scenario: the clamp does not bite
- **WHEN** every region's graded edge is under its clamp
- **THEN** no caveat is written

### Requirement: The sizing background grid SHALL be priced before it is built

A graded mesh's sizing field needs a tetrahedralization of the cavity to carry it,
and that grid is built BEFORE the object exists, on the machine that then builds
and solves the mesh while another job may be running. The mesher SHALL therefore
estimate the grid's element count from the cavity's ENCLOSED volume over its cell
volume BEFORE building it, SHALL hold it under a stated cap
(`MAX_BACKGROUND_TETS`), and SHALL COARSEN its cells to fit that cap rather than
refuse: a coarser transition costs mesh quality in the graded band, refusing costs
the answer. A coarsening can leave the finest region with no node of its own, and
the field then cannot produce that region's edge anywhere -- a mesh that comes
back quietly UNGRADED -- so the cells are never widened past the finest region's
own edge, and only when coarsening has run out there, with the volume term still
over the cap, does the mesher refuse: BEFORE building, with a plain message
naming that region, its edge and what the owner can change. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`.

Measured basis: the grid must be as fine as the fine edge and covered 8.6k-580k
elements on the fixtures measured -- up to ~19x the final mesh's own element
count. The estimate is the volume/cell-volume ratio, not the bounding box, so it
prices exactly the tetrahedralization about to be attempted. The cap's SIZE comes
from TetGen's own memory: ~589k cells of this PLC held 107-123 MB resident on the
VM (182-209 B/cell; 63 B/cell where the cells are far coarser than the facets), so
the cap is worth ~0.12 GB, a few per cent of a worker's RAM.

A BUILT grid can still exceed the cap, and that SHALL be reported, not refused:
once the cells are much coarser than the PLC's own facets, the facets set the
count and the cell edge stops moving it -- measured on the thick-wall fixture as
37.4k, 16.9k and 19.7k cells at an 8k, 3k and 1k cap, then 16,130 cells at 300, 60
and 20, with the cells 5.4x, 9.3x and 13.4x wider than the fine region's edge.
Refusing there would discard a build already paid for. The property grading
actually rests on -- the finest region keeping a node of its own -- is checked
separately and still refuses.

#### Scenario: the priced grid is over the cap
- **WHEN** the estimated element count exceeds `MAX_BACKGROUND_TETS`
- **THEN** the grid's cells are coarsened to fit the cap, but never past the
  finest region's own edge, and the mesh is still built graded, every region
  keeping its own achieved edge

#### Scenario: coarsening has run out at the finest region's edge
- **WHEN** the volume term is still over the cap with the cells widened to the
  finest region's own edge
- **THEN** the mesher refuses BEFORE building, naming that region, its edge and
  what the owner can change

#### Scenario: the built grid exceeds the cap
- **WHEN** the built grid holds more cells than the cap, because the PLC's own
  facets set the count
- **THEN** the mesher reports the count and its reason and builds on

#### Scenario: a grid inside the cap
- **WHEN** the estimated element count is inside the cap
- **THEN** the grid is built at its planned step, unchanged, and no cap applies

### Requirement: The multi-material benchmark SHALL carry its verdict in its exit code

The benchmark SHALL exit zero only when every criterion was measured AND passed.
A criterion that FAILED SHALL end the process with a non-zero DEFECT code and a
criterion that could not be measured SHALL end it with a distinct non-zero code,
so that a remote invocation cannot read a FAILED run as success. Its fine-region
criterion SHALL measure the achieved edge on the FINE REGION's own elements --
never the whole-mesh median, which a graded mesh's coarse regions dominate -- and
SHALL compare it with the same overshoot allowance the mesher itself enforces. The
benchmark's module docstring SHALL carry the invocation that works, including the
environment the solver backend needs. Source: `benchmarks/run_multimaterial_bench.py`.

#### Scenario: a criterion fails
- **WHEN** a case's worst error or its element count fails its criterion
- **THEN** the report prints the failed criterion and the process exits non-zero

#### Scenario: a graded case's fine region is scored
- **WHEN** a graded case is scored
- **THEN** its fine region's achieved median edge is taken from that region's own
  elements and compared with that region's own target

#### Scenario: the mesh block is missing
- **WHEN** a case has no mesh block, or its fine region holds no elements
- **THEN** the fine-region criterion is UNMEASURED, which fails the run, and it is
  never reported as a pass

## MODIFIED Requirements

### Requirement: The interior element size follows Weyl's law from enclosed volume

The target element size SHALL derive from the ENCLOSED volume (not the bounding
box) via Weyl's law, at a fixed number of edges per wavelength, and SHALL be
lowered (never raised) by the owner-declared smallest feature. That edge is the
WALL's edge, and the edge the whole cavity is meshed to while every region shares
one index. For a multi-region cavity it SHALL be refined per region, never
coarsened: the wall keeps it, a region denser than the wall is meshed finer in
proportion to its index, and no region is meshed coarser than the wall. Source:
`dockerfiles/maxwell_solver/mse_common.py`.

#### Scenario: a smallest feature is declared
- **WHEN** the declared feature is smaller than the physics-derived edge
- **THEN** the target edge is lowered to the feature

#### Scenario: a multi-region cavity
- **WHEN** the upload's regions do not all share one index
- **THEN** no region is meshed coarser than the Weyl edge, and the region a
  uniform mesh would have given that edge still receives exactly it

### Requirement: The surface can floor the mesh, so an over-refined surface is decimated

TetGen cannot make interior elements much larger than the surface facets, so the
mesher SHALL detect an over-refined surface (median surface edge below a trigger
fraction of the target edge) and decimate it before tetrahedralizing. The
decimation target SHALL be the target edge of the mesh being built: the COARSE
edge for a graded mesh, because the surface is the mesh's floor and decimating to
the finest edge would floor the coarse regions and defeat the grading; unchanged
for a uniform mesh. Source: `dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: an over-refined upload is meshed
- **WHEN** the upload's surface is finer than the target edge by the trigger
- **THEN** the surface is decimated before tetrahedralizing
- **AND** the interior does not take the surface's over-refinement

#### Scenario: a graded mesh's surface is over-refined
- **WHEN** the mesh is graded and the surface is finer than the coarse edge
- **THEN** the surface is decimated to the COARSE edge, so the coarse regions are
  not floored at the fine region's resolution

### Requirement: An achieved resolution worse than allowed is a failure

The mesher SHALL measure the achieved element edge (median) of a region from the
tets carrying that region's attribute -- never the whole mesh, whose median a
graded mesh's coarse regions dominate -- and SHALL FAIL if the FINEST region
overshoots its target by more than the allowed fraction; showing wrong
frequencies with confidence is worse than no answer. The finest region's target
IS `h_min`, the edge a uniform mesh gives that region today, and the allowance
SHALL never be looser than the uniform mesh's own, so grading cannot spend
resolution the cavity was previously solved at. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: the achieved edge overshoots
- **WHEN** the finest region's meshed edge exceeds the allowed overshoot
- **THEN** the job FAILS rather than reporting frequencies from the coarse mesh

#### Scenario: a graded mesh is measured
- **WHEN** the mesh is graded
- **THEN** the check is made on the finest region's own tets, so a coarse wall
  cannot hide or fake the resolution the core was solved at
- **AND** the finest region is held to the same edge a uniform mesh gives it

### Requirement: The solver writes a defined artifact set

A solved job SHALL write `surface.glb` (in the user's units, for the viewer),
`modes.json` (job metadata plus per-mode values), `mode_NN.bin` (float32, C
order, shape `[nz][ny][nx][3]`, `NaN` outside the object), and `mode_NN.vtu`
(an unstructured grid carrying `E` and `E_magnitude`). `modes.json` SHALL carry a
`mesh` block stating whether the mesh was graded, each region's target edge,
achieved median edge and filled volume, and any caveats the mesher recorded (the
clamp), for a graded mesh; for a one-part or equal-index upload the block keeps
its present fields. Source: `dockerfiles/maxwell_solver/mse_maxwell_solver.py`,
`dockerfiles/maxwell_solver/mse_fields.py`,
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: a mode field is written
- **WHEN** a mode's field is sampled and written
- **THEN** its bytes are `float32`, C order, with the three components last
- **AND** each mode is normalised so `max|E| == 1`

#### Scenario: a graded job is reported
- **WHEN** a job's mesh was graded
- **THEN** `modes.json`'s `mesh` block says `graded` and gives each region's
  target and achieved edge and its filled volume
- **AND** when the clamp lowered a region's target, the block carries the caveat
