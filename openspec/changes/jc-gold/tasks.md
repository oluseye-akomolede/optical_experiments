# jc-gold tasks

## 1. Spec

- [x] 1.1 `openspec/changes/jc-gold/specs/time-domain/spec.md`: modify the
      dispersive-metals and bench-selection requirements, add the Johnson &
      Christy model requirement
- [x] 1.2 `proposal.md`

## 2. Data and model

- [x] 2.1 `data/jc_au_nk.csv` from the 1972 table's 600-1300 nm rows, citation
      in the header
- [x] 2.2 Generalise `scripts/fit_palik_gold.py` (`--data`, `--name`,
      `--source`); the no-argument run still reproduces the Palik literal
- [x] 2.3 Fit `Au-jc` (4 poles, 256 starts, `--fit-max-nm 0` so all 9 rows
      are fitted) and add `fdtd.JOHNSON_CHRISTY_1972` to `LIBRARY`/`METALS`

## 3. Bench

- [x] 3.1 `--gold-data jc` on `scripts/paper1_nanowire.py`, mapped in
      `gold_metal`

## 4. Tests

- [x] 4.1 `tests/test_fdtd_metals.py`: table within the gate, smoothness,
      passive grid, stability at 2-20 nm, band refusal in its own words, and
      the 905 nm loss ratio against Palik
- [x] 4.2 `tests/test_paper1_nanowire.py`: `--gold-data jc` reaches the scene,
      the layout, the estimate and the report, and its band refusal names its
      own table

## 5. Verification

- [x] 5.1 `tests/test_fdtd_metals.py`, `tests/test_paper1_nanowire.py` and the
      suites that read the library (`test_td_form`, `test_td_solve`,
      `test_fdtd_layout`, `test_worker`, `test_routing`) green; strict
      validation of the change and the baseline, both counts reported.
      2026-10-23: `514 passed, 1 warning in 2073.90s` for those seven files
      (serial, single-threaded BLAS, under the 4 GiB cap);
      `openspec validate jc-gold --strict` -> "Change 'jc-gold' is valid";
      `openspec validate --specs --strict` -> 12 passed, 0 failed

## 6. Handoff

- [x] 6.1 Commit locally, report the sha to `mse-lead@mse` (no push)
