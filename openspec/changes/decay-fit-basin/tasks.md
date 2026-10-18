# Tasks — decay-fit-basin

- [x] 1. Reproduce the wrong basin on a tiny synthetic tail. A 1060 fs mode
  midway between two of the old 21 columns (+-15% band at 331.3 THz, 3,000
  samples of 0.15 fs) gave 108.05 fs at rms 0.38 of the peak.
- [x] 2. Measure the three candidate fixes on 60 such tails (10 taus x 6
  places between columns, 1e-3 noise). Multi-starting the old descent from 3
  and 5 cells left 33 and 31 wrong; the FFT scan with a basin descent, none.
  The bin spacing on 60 long lines (700-8000 fs, 1e-4 noise) with the final
  descent: 1 bin per window left 5 wrong (13-18% short), 2, 4 and 8 none, all
  at about 550 ms a fit. Chosen: 8, four times the coarsest that worked.
- [x] 3. `scripts/paper1_nanowire.py`: `_decay_scan` (FFT scan),
  `_local_minima`, `fit_decay`'s basin descent, rivals, `rival` and
  `ambiguous`, plus the constants `DECAY_PAD`, `RIVALS`, `RIVAL_ITERATIONS`
  and `AMBIGUITY_RATIO`. The ring-down arm prints the reason and returns 3.
- [x] 4. Measure `AMBIGUITY_RATIO` on 105 random two-mode and noisy one-mode
  tails. 14 of the 16 fits more than 10% off had a rival at 1.05 to 11.1 x;
  chosen: 12. The two it misses (ratio 122, and no separated minimum) are
  stated in the spec.
- [x] 5. Tests: `tests/test_decay_fit_basin.py`, 10 tests on synthetic
  tails of 3,000 samples or fewer (about 20 s together).
- [x] 6. `tests/check_discriminating.py`: 11 `basin-*` cases. Each applies,
  and each turns exactly its named tests red.
- [x] 7. Verification: `OMP_NUM_THREADS=2 python -m pytest
  tests/test_decay_fit_basin.py tests/test_cw_settling.py
  tests/test_paper1_nanowire.py tests/test_paper1_collection.py -q`, the
  `basin-*` cases, and `--tiny --ringdown` set against main's output. No
  full-size run.
- [ ] 8. Later, on the GPU: re-fit the full-size ring-down's saved tail
  (`--save-tail`, `load_record`) with this fitter before reading 113.1 fs
  against anything. Read whether it is ambiguous and its rms against its
  peak. If it is still ambiguous, fit a two-mode model to the same record.
- [ ] 9. Separately: scale the Jacobian's columns before the pinv in
  `fit_decay`'s covariance. Measured on the 1060 fs tail: `tau_err` is 0.152
  fs where 40 seeds scatter by 0.271 fs, and the column-scaled covariance
  gives 0.279 fs. `f_err` would become a number (39 MHz against a 27 MHz
  scatter) instead of `undetermined`. This changes
  `test_a_positive_error_below_its_own_parameter_is_the_pinv_too`'s
  assertion that `f_err` is None, so it is its own change.
