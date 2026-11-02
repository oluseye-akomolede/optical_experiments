# Tasks

## 1. One switch, read by both sides

- [x] 1.1 `mse_common.graded_meshing_enabled(environ=None)` and
      `GRADED_MESHING_ENV` / `GRADED_MESHING_OFF_VALUES`: unset means ON, only
      the explicit off spellings turn it off. Lives in `mse_common` because
      BOTH images carry that module.
- [x] 1.2 `mse_meshing.graded_meshing()` delegates to it; the module flag
      `GRADED_MESHING` is `None` by default (ask the environment) and stays only
      as a tier-1 test seam. `tetrahedralize(graded=None)` reads it.
- [x] 1.3 `GRADED_MESHING`'s docstring, the `tetrahedralize` docstring and the
      module comment say what the product now does and how to roll it back.

## 2. The estimate prices the graded mesh

- [x] 2.1 `web.meshing.estimate_tets` branches on
      `COMMON.graded_meshing_enabled()`: `estimate_tets_graded` when on,
      `estimate_tets_by_region` when off. One-part uploads unchanged.
- [x] 2.2 `estimate_tets_graded`'s "not on any product route" note replaced
      with the contract: it IS the multi-part estimate under grading, and the
      two functions are told apart by the same switch that picks the mesher's
      path.

## 3. Tests

- [x] 3.0 The solver Job is TOLD the setting the app priced with:
      `build_job_manifest(..., graded_meshing=...)` (pure, defaulting to the
      product rule) is fed from `K8sJobLauncher.manifest` via the same accessor
      `estimate_tets` used. Without it a rollback reached the web pod only, and
      `test_routing` now pins both directions.
- [x] 3.1 The memo's fakes accept the sizing field's keyword set, so a graded
      call reaches the branch instead of dying on a stale copy of the product
      parameter list.
- [x] 3.2 The existing controls are the graded default's controls, and they
      still hold: a one-part upload makes the same five-keyword call
      (`test_meshing_guard`), and an equal-edge multi-part upload never enters
      the sizing-field branch.
- [x] 3.3 Run `tests/test_meshing_guard.py`, `tests/test_estimate_families.py`,
      `tests/test_grading_matrix.py` at tier 1.

## 4. What is NOT done

- [ ] 4.1 A real multi-part upload graded end to end under the new default
      (tier 2/3), to confirm the estimate and the built mesh agree on a job that
      actually runs. This change is reversible with `MSE_GRADED_MESHING=0` on
      both deployments.
- [ ] 4.2 Archiving `per-region-mesh-grading`, whose ADDED requirement this
      change supersedes; until then the two changes disagree about the default
      and `openspec/specs/` reflects neither.
