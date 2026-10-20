# Tasks: palik-gold-smooth

## 1. Refit Au-palik under a damping floor

- [x] 1.1 `scripts/fit_palik_gold.py`: add `--gamma-floor` (default 0.05 eV),
      `--fit-max-nm` (default 1150) and `--envelope` (default 1.5), the last as
      a one-sided residual against 1.5x the linear interpolation between table
      rows on a 2 nm grid
- [x] 1.2 Print the dense-grid diagnostics the fit is accepted on (Re eps over
      1100-1300 nm, max Re eps and min Im eps over 600-1400 nm, the envelope
      ratio and where it peaks)
- [x] 1.3 Run the fit with the committed settings, record the literal, the
      fitted range and the max row error with its wavelength
- [x] 1.4 Replace the `PALIK_1985["Au-palik"]` literal and its comment with the
      result: the table, the fitted rows, the damping floor, the dense-grid
      measurements, the row error and the command that reproduces it

## 2. Dense-grid and row tests

- [x] 2.1 `tests/test_fdtd_metals.py`: dense-grid test (passive over
      600-1400 nm every 2 nm; `|eps|` within 1.5x of the interpolation between
      the table's rows; a bounded `|eps|` through the 1127-1240 nm gap)
- [x] 2.2 Row accuracy test on the fitted rows, asserting the measured maximum
      as well as the 3% gate
- [x] 2.3 The band-refusal test names Palik's table and the new fitted range
- [x] 2.4 State in the report that the dense-grid check fails on af2c616's
      literal (measure it, do not assert it from memory)

## 3. The band guard where runs are made

- [x] 3.1 `fdtd.BAND_FLOOR` and `fdtd.source_band(frequency, profile,
      bandwidth)`: the pulse's band above the floor, None for `cw`/`drawn`, and
      a refusal when the band reaches zero
- [x] 3.2 `fdtd.check_scene_band(materials, request)`: judge each named library
      metal against the request's band; `pec` and indices are not models
- [x] 3.3 `scripts/fdtd_bench_v4.py` takes its band from `fdtd.source_band`
      (one definition, the app's) and its header names the selected model's
      source
- [x] 3.4 `web/td_form.plan_request` calls the guard on the scene's metals
      (parts and coatings), so the refusal reaches the uploader in words
- [x] 3.5 `worker/td_solve.solve_td` calls the same guard before layout, so a
      job that bypasses the form meets it too (before layout: voxelising a grid
      only to refuse it wastes the run)
- [x] 3.6 Tests: `test_fdtd_metals.py` (the definition, the guard's scope),
      `test_td_form.py` (refused with the source named; in-band queued),
      `test_td_solve.py` (the worker's refusal, its default job still runs)

## 4. Labels

- [x] 4.1 Model-specific labels name the library or the selected model:
      `fdtd_voxel.py` (3), `purcell_reference.py` (5), `fdtd.py` (2),
      `fdtd_bench_v4.py` (1)

## 5. Validation and handoff

- [x] 5.1 `openspec validate palik-gold-smooth --strict` (valid) and
      `openspec validate --specs --strict` (baseline: 12 passed, 0 failed)
- [x] 5.2 Affected tests at tier 1 (`ulimit -v 4194304`, `nice -n 19`):
      `test_fdtd_metals.py` + `test_td_form.py` 111 passed in 51 s;
      `test_td_solve.py` 81 passed in 16 s; `test_fdtd_bench_v4.py` 10 passed;
      `test_paper1_nanowire.py -k "gold_flag or outside_its_models_fit"`
      2 passed in 179 s
- [x] 5.3 Commit locally (no push) and hand to the reviewer with the sha, the max
      row error and the dense-grid Re eps range over 1100-1300 nm

## 6. Reviewer rework (qitem-20261020164956)

- [x] 6.1 `scripts/paper1_nanowire.py` judges the mirror with
      `fdtd.source_band` and `check_metal_band`, not `frequency +/- bandwidth`:
      the pulse's 1% spectrum, the app's own numbers, and a `cw` source asks
      nothing
- [x] 6.2 `tests/test_paper1_nanowire.py`: the outside-the-fit test runs a
      `--profile pulse`; a `cw` Palik run at 1500 nm is accepted; Run C's
      900 nm pulse is parametrised at 83 nm (accepted, 1.103 eV floor), 84 nm
      and 100 nm (refused, same sentence as the app's `check_scene_band`)
- [x] 6.3 The tail-row rationale fixed everywhere it was copied (spec delta,
      proposal, `fdtd.py`, `tests/test_fdtd_metals.py`,
      `scripts/fit_palik_gold.py`, and the commit message): the rows past
      1150 nm are the table's tightest, not its sparsest, and are unfitted
      because they miss the 3% gate by 4.12/19.18/20.37% and their k is
      non-monotonic (8.77, 8.09, 8.25)
- [x] 6.4 The 1.47% claim dropped: the committed literal is 1.4951% at
      652.5 nm, which the row test asserts
- [x] 6.5 Re-run the affected tests and re-hand the new sha: `test_fdtd_metals.py`
      + `test_td_form.py` 111 passed (50 s); `test_paper1_nanowire.py -k
      "gold_flag or gold_run_outside or run_c_pulse"` 5 passed (164 s);
      `openspec validate palik-gold-smooth --strict` valid, `--specs --strict`
      12 passed, 0 failed

