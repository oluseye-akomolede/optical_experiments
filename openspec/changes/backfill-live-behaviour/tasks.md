# Tasks

## 1. Read the live behaviour, per capability

- [x] 1.1 mesh-upload: routes, units/scale, part splitting, estimate, refusals
- [x] 1.2 job-lifecycle: the row, claim/lease/release, reapers, messages, worker API
- [x] 1.3 maxwell-solver: meshing, FEM, eigen solve, backend seam, artifacts
- [x] 1.4 jetson-worker: concurrency, memory bound, crash recovery, token discipline
- [x] 1.5 results-api: store listing, index/fetch agreement, serving
- [x] 1.6 mode-viewer: colour scale, caption, field-off view
- [x] 1.7 render-check: capture, predicates, the control
- [x] 1.8 deployment-bounds: quota, threads, images

## 2. Write the capability specs

- [x] 2.1 mesh-upload/spec.md
- [x] 2.2 job-lifecycle/spec.md
- [x] 2.3 maxwell-solver/spec.md
- [x] 2.4 jetson-worker/spec.md
- [x] 2.5 results-api/spec.md
- [x] 2.6 mode-viewer/spec.md
- [x] 2.7 render-check/spec.md
- [x] 2.8 deployment-bounds/spec.md

## 3. Validate and hand off

- [x] 3.1 `openspec validate backfill-live-behaviour --strict` passes
- [ ] 3.2 Reviewer checks each requirement against the live source
- [ ] 3.3 Archive the change so `openspec/specs/` matches what is live

## 4. Known gaps to record, not hide

- [x] 4.1 `peak_rss_mb` has no writer in `web/` or `worker/` -- the spec does not
  claim it is populated
- [x] 4.2 the browser flow that POSTs to `/parts` is absent from this tree; the
  spec claims the server-side contract only
- [x] 4.3 the release bound's location disagrees between a schema comment and
  `web/db.py`; the spec cites the code
