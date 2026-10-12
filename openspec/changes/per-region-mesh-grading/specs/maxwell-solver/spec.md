# maxwell-solver

## ADDED Requirements

### Requirement: A multi-material cavity SHALL be meshed to a per-region target edge

Each region SHALL be meshed to its own target edge `h_r = min(h, h * n_wall /
n_r)`, where `h` is the wall's edge from the enclosed volume and the index ratio
is the physics: a region denser than the wall is finer, one less dense is not
coarser than the wall. The per-region edges SHALL come from ONE helper that both
the mesher and the web estimate read, so the edge a region is built to and the
edge the route was decided on cannot drift apart. The smallest of the per-region
edges SHALL be the finest region's edge `h_min`, identical to the edge the
uniform rule used, so the resolution the cavity is solved at does not move.

The mesher SHALL realise the per-region targets with TetGen's mesh-sizing
FUNCTION (a background mesh carrying each region's target, `bgmesh=`), and SHALL
NOT pass `fixedvolume` or a global `maxvolume` alongside it: either one overrides
the field and silently meshes uniformly. Where the per-region edges are all
equal, the mesher SHALL use the existing fixed-volume path unchanged. Source:
`dockerfiles/maxwell_solver/mse_common.py`,
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: a cavity whose core is denser than its wall
- **WHEN** the upload's regions do not all share one index
- **THEN** the wall's elements are coarser than the core's
- **AND** the core's achieved median edge is within the overshoot allowance of
  `h_min`
- **AND** neither `fixedvolume` nor a global `maxvolume` accompanies the field

#### Scenario: every region shares one index
- **WHEN** a multi-part upload's regions all have the same index
- **THEN** the per-region edges are equal and the mesh is built by the existing
  fixed-volume path, byte for byte

#### Scenario: a single-material upload
- **WHEN** the upload has one part
- **THEN** nothing changes: the grading helper is not consulted and the mesh is
  the mesh of today

## ADDED Requirements

### Requirement: A coarse region's target SHALL be clamped under its own thickness

A region's target edge SHALL be clamped to `CLAMP_THICKNESS_FRACTION *
2 * V_filled / A_boundary`, where `V_filled` is the region's FILLED volume and
`A_boundary` is its own surface area plus every child's. The clamp SHALL never
lower the finest region (it is never applied below `h_min`), and SHALL NOT
invert the ordering of the regions. When the clamp lowers a region's target, the
job's caveats SHALL say so, naming the region and the edge it was clamped to.
Source: `dockerfiles/maxwell_solver/mse_common.py`.

Measured basis: the sizing field stops being honoured when a target approaches
the thickness of the region it describes. On three nested-cube fixtures the mesh
collapsed -- the FINE region losing its resolution too -- at 1.9-3x the shell's
thickness, while a target under the thickness held. Background-mesh resolution
was tested as the cause and refuted: 0.22 vs 0.08/0.05/0.04 gave identical
collapses (128 elements, core edge 0.1768 in every case).

#### Scenario: the coarse target exceeds the region's thickness
- **WHEN** the graded edge for a coarse region exceeds the clamp
- **THEN** it is lowered to the clamp and the job's caveats name the region

#### Scenario: the clamp does not bite
- **WHEN** every region's graded edge is under its clamp
- **THEN** no caveat is written

## MODIFIED Requirements

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

The mesher SHALL measure the achieved element edge (median) of the FINEST region
-- the tets carrying that region's attribute, not the whole mesh, whose median a
graded mesh's coarse regions dominate -- and SHALL FAIL if it overshoots that
region's target by more than the allowed fraction; showing wrong frequencies with
confidence is worse than no answer. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: the achieved edge overshoots
- **WHEN** the finest region's meshed edge exceeds the allowed overshoot
- **THEN** the job FAILS rather than reporting frequencies from the coarse mesh

#### Scenario: a graded mesh is measured
- **WHEN** the mesh is graded
- **THEN** the check is made on the finest region's own tets, so a coarse wall
  cannot hide or fake the resolution the core was solved at

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
