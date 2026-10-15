# Tasks — mesher-region-guard

Row `qitem-20261015050324-adc16144`. Step 1 of the lead's decision: fix the
guard, shipped with the estimate fix. Tier 1 only.

## Diagnosis (done, `af3598a`)

- [x] (a) Per-region achieved edge vs each region's OWN rule, at the product cap
      — 27 runs, worst 1.243x, `benchmarks/mesher_region_guard.py`
- [x] (b) Name the case, if any: none at the product cap; named under the cap
      retry
- [x] (c) The over-delivery cliff: cost-only or can it reach the guard
- [x] (d) The cap retry's coarsening, the guard's blindness, whether the user is
      told — `benchmarks/mesher_cap_retry.py`, fixture caps printed and labelled

## Step 1 — the per-region guard

- [x] `mse_common.region_edges`: return each region's OWN edge
      `h_r = h * n_wall / n_r` (no floor); docstring states the rule and keeps the
      "why the mesher still builds uniform" evidence
- [x] `mse_common.region_plan`: keep `edge_units` = `h_min` (the meshing edge the
      estimate and the app read), add `own_edge_units` per region for the guard
- [x] `mse_meshing._resolution_verdict`: worst region against
      `MAX_EDGE_OVERSHOOT * own_edge_units`; one-part path is the old expression
- [x] `mse_meshing.tetrahedralize`: `attrib` initialised for every pass; the four
      branch sites read the per-region verdict; the refine ratio is the worst
      region's
- [x] `mse_meshing._coarse_message`: region-naming arm, allowance interpolated
      from `MAX_EDGE_OVERSHOOT`; one-part wording unchanged
- [x] `mse_meshing.py:459-463`: the stale "SMALLEST h/n_region" comment corrected
- [x] No new constant; 1.6 and both arms (refine, refuse) kept

## Step 1 — tests (`tests/test_meshing_guard.py`)

- [x] A tiny two-region fixture, coarse core inside a fine wall, is REFUSED, and
      the message names the core
- [x] Reverting the verdict to the global comparison makes that test fail with
      its own message (mutation-checked) — and the other two stay green
- [x] Control: the same path ACCEPTS a two-region upload whose core is fine
- [x] A one-part upload's verdict is unchanged (same answer on both code paths,
      too-coarse and fine)
- [x] Full `tests/test_meshing_guard.py` green, bounded

## Spec

- [x] `skip_specs` dropped; `specs/maxwell-solver/spec.md` delta written for the
      requirement that actually carries the rule (the lead's message named
      `mesh-upload`; the requirement lives in `maxwell-solver`)
- [x] `openspec validate mesher-region-guard --strict` passes

## Step 2 — the reviewer's doc items, into `fix-upload-tet-estimate`

- [ ] Fold the reviewer's three doc items into `fix-upload-tet-estimate` when
      that change is next touched (items not yet enumerated to this seat)

## Step 3 — one Orin command

- [ ] The lead runs `benchmarks/estimate_vs_built.py --build` at the product cap
      on the Orin (real meshes; not this seat's tier)
