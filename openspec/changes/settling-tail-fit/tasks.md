# Tasks — settling-tail-fit

- [x] 1. Reproduce the defect on the 600-period run's printed curve:
  `fit_approach` over its 20 cuts gives F_inf = 13.1935 +- 0.020, tau_s
  82.99 periods, 2.83e4 x the floor, as printed. Over 27 two-exponential
  curves with known limits, the same fit forecast below the end on 18 or 19.
- [x] 2. Measure the candidate fixes on the curve and on those 27 curves
  (the run's cut grid and an even one, noise 0, 1e-6 and 1e-4). Two
  exponentials over the whole curve: residual 0.0196, limit 13.60-13.76 with
  the grid. Dropping cuts until the residual reaches the floor: best when
  noiseless, worst at 1e-4. The last half: never below the end, median
  error 3e-4 to 0.09 of the remaining rise. Chosen: the last half, only when
  the whole-curve residual is over 3 x the floor.
- [x] 3. Try two gates to tell noise from misfit, and keep neither. The two
  fits' F_inf disagree by up to 3.49 errors on noise alone and as little as
  0.48 on a misfit. The residual's von Neumann ratio is 0.85 or more on
  noise and up to 1.68 on misfits.
- [x] 4. `scripts/paper1_nanowire.py`: `TAIL_SHARE`, `INCONSISTENT_SIGMA`,
  `FORECAST_RESIDUAL`, `forecast_check`, `whole`/`fit`/`tail_from` in
  `classify_settling`, `_approach_text`, the refit line, and the
  INCONSISTENT and UNRELIABLE class lines in `settling_report`.
- [x] 5. Measure the thresholds (spec delta): the gap's error with the floor
  against the fit's sigma, the false-alarm records (1623 and 206), the 61
  old forecasts below their end, and the 215 tail forecasts for the
  residual threshold.
- [x] 6. Tests in `tests/test_cw_settling.py`, all pure numpy or under 4,000
  samples: the 600-period fixture, the check on its old forecast (rising and
  mirrored), four two-exponential recoveries, the clean exponential, the
  false-alarm set, and the replay through the printed lines. The two-time-
  constant and coverage tests are restated on `whole` and `fit`.
- [x] 7. `tests/check_discriminating.py`: 12 `tail-*` cases, and the
  `settling-*` sets re-measured. All 25 apply and turn exactly their
  named tests red.
- [x] 8. Verification: `OMP_NUM_THREADS=2 python -m pytest
  tests/test_cw_settling.py tests/test_paper1_nanowire.py
  tests/test_paper1_collection.py tests/test_decay_fit_basin.py -q`; each
  `settling-*` and `tail-*` case; `openspec validate settling-tail-fit
  --strict` and `openspec validate --specs --strict`. No full-size run.
- [ ] 9. Later, on the GPU: re-read the 600-period run's saved series with
  `--from-series` and read the refit line's F_inf, residual and class. If
  it is still UNRELIABLE, the curve's tail holds more than one time
  constant, and a longer record is the next measurement, not a richer model.
