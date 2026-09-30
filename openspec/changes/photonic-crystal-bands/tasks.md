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
- [ ] 5.5 Control C — Joannopoulos alumina-rod TM gap, `eps = 8.9`, `r = 0.2 a`,
  gap expected 0.302–0.443; tolerance and quoted source stated before running.
  **NOT IMPLEMENTED** — this is the one control the brief asks for that has no
  test; it must be added before the capability is called validated.

## 6. Change bookkeeping

- [x] 6.1 proposal, tasks, spec delta present
- [ ] 6.2 `openspec validate photonic-crystal-bands --strict` passes (this run)
- [ ] 6.3 Control C added, or the capability's validation claim narrowed to match
