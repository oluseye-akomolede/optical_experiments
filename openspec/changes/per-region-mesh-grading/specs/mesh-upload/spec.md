# mesh-upload

## MODIFIED Requirements

### Requirement: The upload is estimated and routed, or refused as too detailed

The app SHALL estimate the element count from the ENCLOSED volume, at the
effective edge length, with a conservative coefficient, and SHALL:

- ROUTE the job to the GPU worker when the estimate reaches the GPU threshold;
- otherwise ROUTE it to the cluster;
- REFUSE with a plain-English message when the estimate exceeds the refusal
  threshold (twice the real cap, because the estimate can be up to twice high).

For a multi-part upload the estimate SHALL be the sum over the regions of
`V_r / h_r^3`, each region's FILLED volume at its OWN target edge, which is the
same number as the one-part expression `V_cavity / h_min^3` when the edges are
equal. The estimate SHALL use the same per-region edges the mesher builds to, so
the route decided on cannot be the route for a mesh nobody builds.

The estimate SHALL return `0`, meaning "do not route on this", when the surface
is unusable or the formula fails -- not a small number. Source: `web/meshing.py`,
`dockerfiles/maxwell_solver/mse_common.py`, `web/config.py`.

#### Scenario: the object is too detailed
- **WHEN** the estimate exceeds the refusal threshold
- **THEN** the upload is refused, naming the estimate and the cap, and asking
  for a simpler model

#### Scenario: the estimate could not be computed
- **WHEN** the surface is unusable or the formula raises
- **THEN** the estimate is `0` and the job is not routed on that value

#### Scenario: a multi-part upload is estimated
- **WHEN** the upload has multiple parts
- **THEN** the estimate sums the FILLED volumes (a nested core is not
  double-counted), each at its OWN target edge, so a coarse wall does not count
  as if it were meshed at the core's edge

#### Scenario: every region shares one index
- **WHEN** a multi-part upload's regions all have the same index
- **THEN** the estimate is the SAME number the uniform formula produced, byte
  for byte, because the per-region edges are then equal

## ADDED Requirements

### Requirement: A multi-material job's page SHALL state how its mesh was graded

The job page SHALL show, for a job whose mesh was graded, each region's target
edge and achieved median edge at its own resolution, and SHALL show any caveat
the mesher recorded (a region whose target was clamped under its thickness). A
job that was not graded SHALL show nothing new. Source: `templates/job.html`,
`web/app.py`.

#### Scenario: a graded job's page
- **WHEN** the job's `mesh` block says the mesh was graded
- **THEN** the page shows the per-region resolutions and the caveats, in plain
  English

#### Scenario: a clamped region's page
- **WHEN** the mesher clamped a region's target under its thickness
- **THEN** the page says so, naming the region and the resolution it was given
