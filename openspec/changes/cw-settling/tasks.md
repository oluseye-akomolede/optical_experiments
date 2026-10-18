# Tasks — cw-settling

- [x] 1. Measure the candidates that tiny grids can reach (proposal): the
  window-phase swing and its prediction (tiny paper scene in vacuum and with
  gold, a 25^3 open vacuum box, closed-form tones); the 2f-free reading over
  100 periods (CPML, gold ADE); a lossy PEC box with gold walls rung down and
  driven at its mode; `fit_decay` on boxes whose envelope decay is known.
- [x] 2. `scripts/paper1_nanowire.py`: `cw_powers` / `cw_reading` (the one
  reading, the existing two lines unchanged), `window_phase`,
  `settling_curve`, `hann_correlation`, `fit_approach`, `classify_settling`,
  `settling_report`, `locator_line`, `save_series` / `load_series`,
  `read_cw`, `from_series`; `--save-series`, `--from-series`,
  `--locator-tau`; their refusals in `arm_conflicts`. `fdtd.py` unchanged.
- [x] 3. Tests: `tests/test_cw_settling.py` (the four classes on exact
  inputs, error-bar coverage on noisy records, scale invariance, the phase
  swing against its prediction on closed-form tones and through the solver,
  the phase-locked cuts, the lossy box against its ring-down, the locator
  line, off-by-default invariance, the flags only adding lines, the .npz
  round trip, the unwritable path, the refusals).
- [x] 4. `tests/check_discriminating.py`: `settling-*` cases, each turning
  exactly its named tests red.
- [x] 5. Verification: `OMP_NUM_THREADS=2 python -m pytest
  tests/test_cw_settling.py tests/test_paper1_nanowire.py
  tests/test_paper1_collection.py -q`; the full `--tiny` run against main's
  output; `npx -y @fission-ai/openspec validate cw-settling --strict`.
- [ ] 6. Later, on the GPU: the full-size CW record at 988 nm, dx 10 nm, 300
  periods, with `--save-series` and `--locator-tau 113,3`; read its class,
  its phase line (the full-size quadrature ratio is not known here) and
  tau_s. Re-read the saved record at `--window-periods 16` with
  `--from-series` for the window-length candidate.
- [ ] 7. Separately: re-fit the full-size ring-down's saved tail with a band
  that resolves the line, and read the full-size log's rms against its peak,
  before the 113 fs is set against anything (proposal, finding 3). A fix to
  `fit_decay`'s frequency grid is its own change.
