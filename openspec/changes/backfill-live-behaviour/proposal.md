# Backfill the live behaviour into OpenSpec

## Why

The rig adopted OpenSpec as the source of truth for what its code does
(owner, 2026-09-29), but `openspec/specs/` is empty: the live system's behaviour
exists only in code. This change writes a first spec for the behaviour that is
live TODAY, so a later change can be reviewed against a spec instead of against
memory, and so a reviewer can reject anything that removes spec'd behaviour
without a spec delta.

This is a BACKFILL, not a design. Nothing here is a proposed change to the
system; every requirement is written from a read of the live source, with the
file cited. Where the code has a limit or an absence, the spec says so.

## What

Eight capability specs, each with ADDED requirements:

- **mesh-upload** -- `.obj` intake, units/scale, named-part splitting, per-part
  index, watertightness, the element estimate and routing/refusal, plain-English
  refusals.
- **job-lifecycle** -- the `mse.jobs` row and its four states, atomic claim with
  a lease, owner-guarded renewal, release that resets status, the release bound,
  lease-expiry recovery, the tick's four reapers (and the Jetson skip), failure
  through `message`, and the worker API surface.
- **maxwell-solver** -- the Nédélec edge-element Maxwell eigenproblem, PEC by
  deletion, projected shift-invert Lanczos, spurious-mode assertion, SI
  frequency conversion, the one-factory backend routing BOTH factorisations,
  the checked GPU residual, the Weyl-law element size, the surface floor and
  decimation, hard resolution failure, and the artifact set.
- **jetson-worker** -- one job at a time gated before claiming, the cgroup
  MemoryMax, crash recovery by lease expiry, transient-vs-terminal verbs, and
  the token discipline.
- **results-api** -- results listed from the store not an index, list and fetch
  agreeing on one filename, opaque binary serving, the surface as its own
  artifact.
- **mode-viewer** -- the shared colour scale and the captioned faint component,
  the diverging signed scale, the field-off view, mode and component labelling.
- **render-check** -- the canvas capture (never a screenshot), judging the field
  not the box, the outline-only comparison, the below-floor frame, the gap the
  predicate does not close, and capturing the control after the page closes.
- **deployment-bounds** -- the `mse-power-cap` quota, thread counts matched to
  CPU limits, and the per-component images.

## What this change does NOT do

- It does not change any running code.
- It does not document the superseded Spark/Airflow path (the DAGs, the
  `mse-fem-modes`/`mse-meshes`/`mse-tetras` buckets, the old `mesh_metadata`
  tables). Those belong to the sys_dags rig's history, not this live system.
- It does not paper over an absence. Where the code has no writer for a field
  (for example `peak_rss_mb`), the spec's requirement is scoped to what is
  actually enforced; a future change adding the writer should add it here too.

## Follow-on

- Archive this change once a reviewer has checked each requirement against the
  live source.
- Not every requirement has a test yet. The backfill records the contract; a
  later change can add tests requirement by requirement, starting with the ones
  that most often break (the estimate's run-once units, the release/reset pair,
  the list/fetch filename agreement).
