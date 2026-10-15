# maxwell-solver

## MODIFIED Requirements

### Requirement: An achieved resolution worse than allowed is a failure

The mesher SHALL measure the achieved element edge (median), and SHALL FAIL if
it overshoots the target by more than the allowed fraction; showing wrong
frequencies with confidence is worse than no answer. On a MULTI-REGION upload
the check SHALL be per region: each region's achieved median edge, measured over
that region's own tets from TetGen's per-tet region attributes, SHALL be compared
against the allowed fraction of THAT REGION's OWN required edge -- `h_r = h *
n_wall / n_r` -- and not against the wall's. The verdict SHALL be the worst
region in the mesh. Within the size cap, and while attempts remain, the mesher
SHALL retry FINER before failing; failing is the last resort. The refusal SHALL
name the region that could not be resolved and suggest fewer modes or a lower
refractive index for it; for a one-part upload, where there is no region to name,
the refusal SHALL instead name the object as too large or detailed. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`,
`dockerfiles/maxwell_solver/mse_common.py`.

#### Scenario: the achieved edge overshoots
- **WHEN** the meshed edge exceeds the allowed overshoot
- **THEN** the job FAILS rather than reporting frequencies from the coarse mesh

#### Scenario: a dense region is built coarser than its own rule
- **WHEN** a region's achieved median edge exceeds the allowed fraction of that
  region's own required edge, even though the mesh as a whole is inside the
  wall's allowance
- **THEN** the job FAILS, naming the region
- **AND** the message suggests fewer modes or a lower refractive index for that
  region, and not a smaller object

#### Scenario: a one-part upload is judged once
- **WHEN** the upload has a single region whose refractive index is the wall's
- **THEN** the per-region verdict is the whole-mesh verdict, and the refusal
  wording is the one-part wording

#### Scenario: every region is inside its own allowance
- **WHEN** each region's achieved median edge is within the allowed fraction of
  that region's own required edge
- **THEN** the job proceeds, whatever the region's refractive index
