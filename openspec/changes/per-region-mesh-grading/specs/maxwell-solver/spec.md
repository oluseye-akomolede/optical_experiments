# maxwell-solver

## ADDED Requirements

### Requirement: A multi-material cavity CAN be meshed to a per-region target edge, opt-in

Grading SHALL be OPT-IN. With the switch off (`mse_meshing.GRADED_MESHING`
False, the default, and no `graded=True` passed to `tetrahedralize`), a
multi-material upload SHALL be planned, estimated, meshed, checked and routed
exactly as the live requirements above say: uniformly at the finest region's
edge, through the existing `region_plan` / `estimate_tets_by_region` and the
per-region resolution guard. Making grading the product rule (the route and the
web estimate pricing a graded mesh) is step 3 of this change and is NOT part of
it.

When grading is switched on, each region SHALL be meshed to its own target edge

    edge_r = max(min(h * n_wall / n_r, clamp_r), h_min)

where `h` is the wall's edge, `n_wall` / `n_r` the indices, `clamp_r` the
region's thickness clamp, and `h_min` the finest region's edge as the live plan
computes it. The graded edges SHALL be an OVERLAY on the live plan
(`graded_region_plan`, `graded_region_edges` in `mse_common.py`), never a
replacement for it: the finest region keeps exactly the edge the uniform plan
gives it, and the per-region guard reads the same `own_edge_units` it reads for a
uniform mesh.

The mesher SHALL realise the targets with TetGen's mesh-sizing FUNCTION (`-m`)
through the public `tetrahedralize(bgmesh=<pyvista.UnstructuredGrid>)` route,
the grid carrying each node's target in point data `target_size`, and SHALL NOT
pass `fixedvolume` or a positive `maxvolume` alongside it (either overrides the
field and silently meshes uniformly); it SHALL refuse a switch set that carries
them. Where the graded edges are all equal the mesher SHALL use the uniform path
unchanged. A graded mesh SHALL NOT be retried at a finer edge: a graded build
over the element cap refuses. For a graded mesh the surface SHALL be decimated
to the COARSE edge (decimating to the finest would floor the coarse regions),
and the achieved-edge check SHALL be made on the finest region's own tets.
Source: `dockerfiles/maxwell_solver/mse_common.py`,
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: grading is off (the default)
- **WHEN** a multi-material upload is meshed and grading was not switched on
- **THEN** the plan, the estimate, the mesh and the route are the live uniform
  ones, and the report's `graded` is false

#### Scenario: a cavity whose core is denser than its wall, graded
- **WHEN** grading is on and the regions do not all share one index
- **THEN** the wall's elements are coarser than the core's
- **AND** the finest region's achieved median edge is within the overshoot
  allowance of `h_min`
- **AND** neither `fixedvolume` nor a positive global `maxvolume` accompanies
  the field

#### Scenario: every region shares one index, graded
- **WHEN** grading is on and every region has the same index
- **THEN** the graded edges are equal, `graded` is false, and the mesh is built
  by the uniform path

### Requirement: A graded region's target SHALL be clamped under its own thickness

A region's graded target SHALL be clamped to `CLAMP_THICKNESS_FRACTION * 2 *
V_filled_r / A_boundary_r`, where `V_filled_r` is the region's filled volume and
`A_boundary_r` its own surface area plus every child's. The clamp SHALL only
lower a target, and the floor at `h_min` SHALL re-raise it, so it never coarsens
a region and never moves the finest one. When it lowers a target, the plan's
caveats SHALL name the region, the target it was given and the edge the ramp
asked for. Measured basis: on three nested-cube fixtures the sizing field
collapsed (the fine region losing its resolution too) at 1.9-3x the shell's
thickness; background-grid resolution was tested as the cause and refuted.
Source: `dockerfiles/maxwell_solver/mse_common.py`.

#### Scenario: the coarse target exceeds the region's thickness
- **WHEN** a region's graded edge exceeds its clamp
- **THEN** it is lowered to the clamp, never below `h_min`, and a caveat names
  the region

#### Scenario: the clamp does not bite
- **WHEN** every region's graded edge is under its clamp
- **THEN** no caveat is written

### Requirement: The sizing background grid SHALL be priced before it is built

The mesher SHALL estimate the background grid's element count from the cavity's
ENCLOSED volume over its cell volume BEFORE building it, hold it under
`MAX_BACKGROUND_TETS` (the product's own element cap), and coarsen its cells to
fit rather than refuse, but never past the finest region's own edge; only when
coarsening has run out there does it refuse, before building, naming that
region, its edge and what the owner can change. A BUILT grid over the cap (the
PLC's facets set the count) SHALL be reported, not refused. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: the priced grid is over the cap
- **WHEN** the estimated grid count exceeds the cap
- **THEN** the cells are coarsened to fit, never past the finest region's edge,
  and the mesh is still built graded

#### Scenario: coarsening has run out at the finest region's edge
- **WHEN** the volume term is still over the cap at the finest region's edge
- **THEN** the mesher refuses before building, naming that region and its edge

#### Scenario: a grid inside the cap
- **WHEN** the estimated grid count is inside the cap
- **THEN** the grid is built at its planned step

### Requirement: The mesher SHALL report the mesh it built

After each `tetrahedralize` of a multi-material upload the mesher SHALL record
`LAST_MESH_REPORT`: whether the mesh was graded, each region's target edge,
achieved median edge and filled volume, and the plan's caveats; a uniform
multi-material mesh reports `graded` false. For a one-part upload it SHALL be
`None`. The solver SHALL write that report as an ADDITIVE `mesh` block in
`modes.json` when it is not `None`, so a one-part job's `modes.json` is
unchanged and no existing key changes. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`,
`dockerfiles/maxwell_solver/mse_maxwell_solver.py`.

#### Scenario: a graded job is reported
- **WHEN** a mesh was built graded
- **THEN** the report says `graded` and gives each region's target and achieved
  edge, and any clamp caveat

### Requirement: The grading benchmarks SHALL carry their verdict in their exit code

`benchmarks/run_multimaterial_bench.py` and `benchmarks/run_grading_matrix.py`
SHALL switch grading on for their graded side explicitly (the product default is
uniform) and SHALL exit zero only when every criterion was measured AND passed.
The accuracy bar SHALL be the UNIFORM mesh of the same object (owner, 2026-10-13);
an unsolved uniform reference is UNMEASURED, never a pass. Each non-answer SHALL
have its own exit code, distinct from PASS and FAIL and from each other: in the
matrix, FAIL 3, UNMEASURED 4, SAME MESH 5, UNRELIABLE 6, NO VERDICT 7, and TOO
CLOSE TO CALL 8 -- a case whose one-sided rule reads PASS or FAIL while the
case's own reference bound does not back it. Every verdict the criterion can
produce SHALL have an entry in the exit table. Source:
`benchmarks/run_multimaterial_bench.py`, `benchmarks/run_grading_matrix.py`.

#### Scenario: a criterion fails
- **WHEN** a case's worst error or element count fails its criterion
- **THEN** the report prints the failed criterion and the process exits non-zero

#### Scenario: the bound disagrees with the one-sided rule
- **WHEN** the case's shared-reference bound supports neither side, or the other
  side, of the one-sided rule's reading
- **THEN** the verdict is TOO CLOSE TO CALL and the process exits 8

#### Scenario: the uniform reference was not solved
- **WHEN** the uniform mesh could not be built or solved
- **THEN** the accuracy criterion is UNMEASURED and the run exits non-zero
