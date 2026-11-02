# maxwell-solver

## ADDED Requirements

### Requirement: A multi-material cavity is meshed to a per-region target edge, by default

This requirement SUPERSEDES, on archive, the ADDED requirement of the same name
in the unarchived change `per-region-mesh-grading`, whose text makes grading
opt-in with `mse_meshing.GRADED_MESHING` False. The rule, the clamp and the
overlay are that change's unchanged; what changes is which side is the default
and where the switch lives.

Grading SHALL be the PRODUCT RULE for a multi-part upload, and SHALL be
switchable OFF with one environment variable.

With grading ON -- the default -- a multi-material upload with more than one
region SHALL be meshed with TetGen's sizing field, each region at its own target
edge

    edge_r = max(min(h * n_wall / n_r, clamp_r), h_min)

where `h` is the wall's edge, `n_wall` / `n_r` the indices, `clamp_r` the
region's thickness clamp, and `h_min` the finest region's edge as the live plan
computes it. The graded edges SHALL be an OVERLAY on the live plan
(`graded_region_plan`, `graded_region_edges` in `mse_common.py`), never a
replacement for it: the finest region keeps exactly the edge the uniform plan
gives it, and the per-region guard reads the same `own_edge_units` it reads for a
uniform mesh.

#### Scenario: A multi-part upload is graded without being asked

- **WHEN** a multi-part upload is validated and meshed with grading ON (the
  default: `MSE_GRADED_MESHING` unset)
- **THEN** every region whose target edge differs from the finest edge is meshed
  at its own edge through the sizing field
- **AND** the finest region lands on the same edge the uniform plan gives it

#### Scenario: The estimate prices the mesh the mesher builds

- **WHEN** `web.meshing.estimate_tets` prices a multi-part upload
- **THEN** it reads the SAME switch the mesher reads
  (`mse_common.graded_meshing_enabled`)
- **AND** with grading ON it returns `estimate_tets_graded`'s number, and with
  grading OFF `estimate_tets_by_region`'s
- **AND** the two cannot disagree, because one accessor decides both

#### Scenario: One variable rolls grading back

- **WHEN** `MSE_GRADED_MESHING` is one of `0`, `false`, `no`, `off`
- **THEN** a multi-part upload is planned, estimated, meshed and artefacted
  uniformly at `h_min`, exactly as before this change
- **AND** the job page shows no graded-mesh block, because the mesh is not graded

#### Scenario: A one-part upload and an equal-edge multi-part upload are untouched

- **WHEN** an upload has one part, or several parts that all land on one edge
- **THEN** it is meshed by the same call it was meshed by before this change, and
  the graded path is not entered
