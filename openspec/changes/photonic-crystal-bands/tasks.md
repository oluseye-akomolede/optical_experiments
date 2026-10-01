# Tasks — photonic-crystal-bands

Backfill: feature 2 is already built on this branch. These tasks record what
the change ships and what remains before the capability can be called
validated. Boxes ticked are done and verified on this branch.

## 1. Form and request

- [x] 1.1 `parse_request`: lattice, `r_over_a`, `n_rod`, `n_bg`, `n_bands`,
  `divisions`, `polarisation`; each refusal names its own field
- [x] 1.2 `divisions` stored as part of the request (it changes the mesh)
- [x] 1.3 `k_points` derived from the lattice, not a form field
- [x] 1.4 `cell_size` derives `(nodes, triangles)` from the solver's own
  `unit_cell_mesh`, not an independent estimate
- [x] 1.5 `check_caps` refuses each cap with its OWN number and sentence
- [x] 1.6 `plan_request` is the one entry point; it does not enqueue

## 2. Physics (`dockerfiles/maxwell_solver/pc.py`)

- [x] 2.1 `lattice_vectors`, `reciprocal_vectors` (dual to 1e-12)
- [x] 2.2 `high_symmetry_path`: Gamma-X-M-Gamma and Gamma-M-K-Gamma; triangular
  corners derived from the basis, asserted against closed forms
- [x] 2.3 `unit_cell_mesh`: conforming circle, grid-sampled eps, periodic pairs
  by coordinate (not index)
- [x] 2.4 `_p1_matrices` with the `weight` parameter separating the two operators
- [x] 2.5 `bloch_reduce`: fold masters with the phase `exp(i k . T)`
- [x] 2.6 `solve_bands`: sparse shift-invert at a negative shift; `w a / 2 pi c`
- [x] 2.7 `bands_from_mesh`: ONE assembly path shared by controls and solver
- [x] 2.8 `layered_*`: the 1-D transfer-matrix oracle (monodromy traces)

## 3. Worker and storage

- [x] 3.1 `worker/pc_solve.py` consumes the row's own request keys
- [x] 3.2 payload is JSON-ready: `labels`, `k_points`, `bands`
- [x] 3.3 a cavity job handed to the pc solver is refused as the wrong kind,
  distinct from a pc job with a NULL request
- [x] 3.4 a pc job reuses the cavity claim → solve → store path via `kind`

## 4. UI

- [x] 4.1 `web/static/bands.js` is pure: gaps computed from the arrays, not the
  plot
- [x] 4.2 `bandGaps` reports `lower`, `upper`, `midgapRatio`, `index`; no gap is
  an empty array
- [x] 4.3 `kPathAxis` uses path length, not k-index
- [x] 4.4 `bandRange` is finite for a flat band

## 5. Validation controls

- [x] 5.1 Control A — empty lattice: `lam = 0` at Gamma to 1e-10; X, M within
  relative 1e-3; Gamma band 2 within 1e-2 of 1.0
- [x] 5.2 Control A — second-order convergence, observed order in [1.8, 2.2]
  (heavy)
- [x] 5.3 Control B — layered cell vs transfer matrix; oracle validated against
  ODE shooting FIRST (worst rel < 1e-10)
- [x] 5.4 Control B — the polarisations differ at eps != 1 (so the control can
  fail)
- [x] 5.5 Control C — the published benchmark: MPB's triangular rod lattice
  (TM, `r = 0.2 a`, `eps = 12`, resolution 32), gap `0.275066 → 0.446290`
  (47.47%), matched to within 1% of midgap. Source read directly
  (mpb.readthedocs.io Scheme Tutorial, "Triangular Lattice"); tolerance stated
  before the run. Includes a can-fail precondition (`r = 0.3` moves the gap
  outside the bar). The row's original figure (Joannopoulos alumina, `eps = 8.9`,
  0.302–0.443) was DROPPED: the plan records that source as unreachable and the
  figure as from memory, so it is not a benchmark — see §4 of the plan and the
  spec's "a benchmark's quoted figures" requirement.
- [x] 5.6 Control C found a CALLER bug on its first run, not a solver bug. MPB
  quotes the mirror's permittivity (`eps = 12`), but `pc.py`'s cell parameters
  are refractive index (`unit_cell_mesh`: `eps_by_tri = n_rod ** 2`, pc.py:347).
  Passing `12` straight in made the gap read `0.0819 → 0.1523` (19.5%) — low by
  `sqrt(12)` and FLAT under refinement (div 16/24/32 → 0.08218/0.08196/0.08187),
  the signature of a units slip. Passed correctly (`n = sqrt(12)`) the solver
  reproduces MPB to 0.12% / 0.02%. Fixed in the test: the conversion is named in
  `MPB_TRI_N_ROD`, with the hazard recorded in the docstring and the proposal.
- [x] 5.7 The decisive split, as a permanent test:
  `test_uniform_cell_scales_as_one_over_sqrt_eps` runs the WHOLE cell at one
  permittivity (no rods), so no per-element sampling is in play, and requires
  band 1 to scale as `1/sqrt(eps)` against the eps=1 value measured in the same
  run. Uniform-wrong ⇒ assembly/normalisation fault; uniform-right-but-rods-wrong
  ⇒ per-element eps sampling. Verified it reddens on an assembly mutation
  (`eps_by_tri ** 0.5` on the mass) and on passing eps as index; both restored.

## 6. Change bookkeeping

- [x] 6.1 proposal, tasks, spec delta present
- [ ] 6.2 `openspec validate photonic-crystal-bands --strict` passes (this run)
- [x] 6.3 Control C added (5.5), so the capability's validation claim now rests
  on three controls, not two.

## 7. Routing fix (review REJECT: pc jobs could not be solved as wired)

Review proved on real Postgres that `_handle_pc_upload` created the row
`target = CLUSTER`, while the only pc solver is on the Jetson worker
(`claim_jetson` selects `target = 'jetson'`), and `next_queued` had no kind
filter — so `run_tick` handed the pc row to the cavity launcher, which built a
CAVITY Job for it and sent the row to `stage = meshing`.

- [x] 7.1 `_handle_pc_upload` creates the row `target = DB.JETSON`; there is no
  cluster pc path in this change
- [x] 7.2 `next_queued` skips `kind = 'pc'` (the pc row is filtered at the
  queue, so the launcher is never handed one on the normal path)
- [x] 7.3 `Launcher.create` raises `JobLaunchError` naming the kind when handed a
  non-cavity job; `run_tick` passes the row's `kind` so the guard is fed the real
  value, not a guess
- [x] 7.4 `real_solve`'s dispatch is explicit: `cavity` → cavity, `pc` → pc,
  anything else → a named failure (the silent `else` is gone). Worker-side pc
  tests added to `tests/test_worker.py` (there were none)
- [x] 7.5 `docs/photonic-crystal-plan.md` states there is no cluster fallback for
  pc yet
- [x] 7.6 Schema rollback written into the proposal (§ Rollback) and
  `db/002_rollback_pc.sql`, qualifier `mse` read from `db/001_schema.sql`,
  `DROP COLUMN IF EXISTS` so a re-run is a clean no-op
- [x] 7.7 CROSSING test on real Postgres (`tests/pc_routing_check.py`): creates
  the pc row via the real route's store call, asserts `claim_jetson` returns it
  and `next_queued(cluster)` does not. Red on the unfixed branch as
  `claim_jetson -> None` (read against the predicted symptom); green after. A
  second, explicitly `target = cluster` pc row pins the kind filter, which is
  only load-bearing for a pc row the queue could otherwise see
- [x] 7.8 Spec delta gains "A pc job SHALL be routed to the Jetson worker, and
  the cluster SHALL refuse it", with the four scenarios above
- [x] 7.9 The GPU-abandon path cannot strand a pc row (re-review F1):
  `reroute_to_cluster` refuses non-cavity rows in its own SQL guard
  (`AND kind = 'cavity'`), and `_reroute_abandoned_gpu_jobs` skips `kind='pc'`
  without showing the "moved to the cluster queue" message. Red-first test in
  `tests/pc_routing_check.py` drives the REAL abandon path on real Postgres:
  a pc row past `MSE_GPU_WAIT_S` stays `target=jetson`, `claim_jetson` returns
  it, and neither the row message nor the log says it moved. Mutations: revert
  the loop skip alone → holds (the SQL guard is load-bearing by itself); revert
  both → reddens with the predicted symptom (`target=cluster`, `claim_jetson`
  `None`)
- [x] 7.10 `docs/photonic-crystal-plan.md` — every remaining cluster-fallback
  claim removed (:35, :196, :262, :452 all now say Jetson only; :35 was a fourth
  site the review did not name)
