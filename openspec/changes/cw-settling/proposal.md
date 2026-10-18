# cw-settling

## Why

The paper-1 CW arm (`scripts/paper1_nanowire.py`, `measure`) reads
F = P/P0 from `fdtd.delivered_power` / `fdtd.dipole_moment` over the last
`--window-periods` (default 8) of its record, and said whether that reading
had stopped moving with ONE line: the same reading on the record cut at two
thirds. The full-size runs at 988 nm, dx 10 nm read:

- 100-period record: F = 9.12, two-thirds 6.86, +32.8%;
- 300-period record: F = 12.7271, two-thirds 11.8688, +7.2%.

A ring-down locator on the same scene fitted tau = 113 +/- 3 fs (300 periods
is ~8.7 tau, the two-thirds cut ~5.8 tau), and one mode with that tau would
have moved the reading well under 1% between those cuts. So something slower
is moving it, or the reading itself is. The candidates (a second, longer-lived
mode; slow numerical growth in the gold's ADE, the CPML or a late-time
instability; the estimator: window length, window phase, the source's
start-up) cannot be told apart by two points, and the series the reading came
from was thrown away after the run.

This change makes the arm say which of four things its record is, with the
numbers that say so, and keeps the record. It does not claim a mechanism for
the full-size scene: that needs the GPU run, which this change prepares.

## What was measured here, on tiny grids (not the full-size answer)

**1. The reading swings with its window's phase, at twice the drive
frequency.** The field at a point source's own cell is almost all reactive:
its in-phase (power-carrying) part is a small residue of the half-step
average the delivered power takes. The Hann window passes a little of each
tone's -f image (3.65e-6 of it at 8 periods), which rotates at 2f against the
drive, and the quadrature multiplies it into the reading. Prediction: a swing
of +-2 x leakage x quadrature of F as the window's end moves through one
period (the image leaks through both the field's and the drive's transform).

| record | window | quadrature | swing measured | predicted |
|---|---|---|---|---|
| tiny paper scene, vacuum, 100 periods | 4 | 4149 | +-25.13% | +-25.14% |
| same | 8 | 4149 | +-3.033% | +-3.033% |
| same | 16 | 4149 | +-0.36% | +-0.36% |
| tiny paper scene, gold + InP, 100 periods | 8 | 1.57e5 | +-58.1% | +-58.1% |
| 25^3 open vacuum box (the test), 40 periods | 4 / 8 / 16 | 2344 | +-21.9 / 2.45 / 0.245% | to 1e-4 |

With the 2f term fitted out, the reading is one number at every window
(tiny vacuum: 1.0011 at 4, 8 and 16 periods; the 25^3 box: 0.97453, 0.97455,
0.97457). The default `--tiny` run's own "settling -2.4%" line is this swing
(0.40 of the most two readings can differ by it), and so is the tiny gold
scene's +109.5%. Whether the swing matters at full size depends on that
scene's quadrature ratio, which the arm now prints; nothing here measures it.

Free-phase cuts every 5% of a record alias the swing into a slow sawtooth
(tiny vacuum, 8 periods: +-0.15% on a reading flat to 6e-5), so the curve's
cuts are locked to the end's own phase.

**2. No slow drift from the CPML or the gold ADE on the tiny grids.** With the
2f term fitted out, the tiny vacuum scene (CPML only) reads 1.0011 to 6e-5
from 30 to 100 periods, and the tiny gold scene 0.00713 to 0.1%. A closed PEC
box with Rakić gold on all six walls (the test's lossy cavity) reaches the
limit a 2-decay-time record extrapolates to (561.1335) within 7e-6 after 10
decay times, with the same tau, and is not classed as growing.

**3. The ring-down fit can stop on a short tau.** `fit_decay` scans 21
frequencies across the band it is given and descends from the best grid
point. Where that grid step is wider than the mode's line, it returned a
short tau with a large residual (PEC boxes with Rakić gold walls, 40 nm
cells, a +-15% band, i.e. a 4.97 THz grid step):

| gold on | envelope tau | fit_decay, +-15% band | rms / peak | fit_decay, +-2% band |
|---|---|---|---|---|
| top (source at z node 8) | 1060 fs | 116.3 fs | 0.37 | 1059.5 fs |
| top (source at z node 9) | 1070.3 fs | 966.6 fs | 0.005 | 1059.7 fs |
| top + y walls | 251.5 fs | 113.3 fs | 0.133 | 251.7 fs |
| top + bottom | 473.1 fs | 118.1 fs | 0.225 | 474.3 fs |
| top, bottom, y | 196.0 fs | 182.8 fs | 0.010 | 195.8 fs |
| all six (the test) | 99.93 fs | 97.6 fs (99.75 at +-5%) | 0.003 | 99.8 fs |

At 116.3 fs the fit leaves 41% of the tail's energy in its residual, where
1060 fs at the same frequency leaves 1e-7. The full-size locator's 113 +/- 3
fs came from this fitter with the paper's 830-970 nm band (a 2.6 THz grid
step); whether it is a stuck fit is not known here. Its own log line ("rms
... against a ... peak") answers that, and a `--save-tail` record re-fitted
with a band that resolves the line answers it without a run. `fit_decay` is
NOT changed here (that would move the ring-down arm's numbers); the CW arm's
locator comparison names this as one of its two readings.

## What Changes

`scripts/paper1_nanowire.py`, the CW arm:

- `cw_powers` / `cw_reading`: the arm's one reading (`delivered_power`,
  `dipole_moment`, `larmor_power`, `purcell_factor` over the last
  `--window-periods` of the series handed in). The end line, the two-thirds
  line, every cut of the curve and a re-read of a saved series all go through
  it. The `measured` and `settling` lines are printed exactly as before.
- `window_phase`: the reading at 33 cuts across the record's last drive
  period, fitted as `a + b x + c cos 4 pi x + d sin 4 pi x`; prints the swing,
  the prediction (2 x Hann leakage at 2f x the field's quadrature ratio, read
  from the estimator's own half-step field) and the phase-free end reading,
  and how much of the two-thirds line's difference the phase alone allows.
- `settling_curve`: the same reading at one cut per 5% of the record, each
  cut a whole number of drive periods before the end (the one with the
  smallest fractional period within a third of the spacing), so every window
  ends at the end's phase. A cut is fitted once its window starts after the
  drive's own 5-period ramp (read from `fdtd.cw_profile`) and P > 0.
- `fit_approach`: `F(t) = F_inf - A exp(-t/tau_s)` written as
  `c + D (1 - e^{-u x})/u`, so the rate may change sign: u > 0 approach,
  u = 0 a straight rise, u < 0 growth. Unweighted least squares; 1-sigma
  errors from the sandwich covariance with the cuts' correlation
  (`hann_correlation`: their Hann windows' overlap, projected on the drive)
  and `s2 = rss / tr((I - H) V)`, floored at the phase floor; fitted on F over
  its own scale.
- `classify_settling`: (a) settled -- a line through the final third moves
  less than 3 of its errors (its own residual, the same sandwich, floored);
  (b) approaching -- the rate is more than 3 errors above zero, F_inf printed
  as an EXTRAPOLATION, with a flag when the residual is over 3 x the floor
  (more than one time constant); (c) growing -- more than 3 errors below
  zero, flagged as a likely numerical problem and "NOT a Purcell factor";
  (d) unresolved -- fewer than 5 fitted cuts, or a rate within its errors of
  zero. The floor is measured per record: the phase scan's swing x 4 pi x the
  largest phase mismatch left on the fitted cuts.
- `--locator-tau FS[,ERR]`: sets tau_s against a ring-down's tau in plain
  words: agrees, shorter, or longer beyond both errors ("a single mode with
  the locator's tau cannot make this curve ... or the locator's tau is not
  this mode's"), and the share of one such mode's rise left after the
  two-thirds cut.
- `--save-series PATH` (off by default): after the reading is printed, the
  drive and source field at their recorded dtype, with dt, frequency, cell
  volume, applied eps_r, host index, window and wavelength, as a compressed
  .npz at the caller's path. A failed write is a warning; the reading stands.
- `--from-series PATH`: the same reading, curve and class from a saved record
  with no scene and no run; `--window-periods` and `--locator-tau` apply.
- Refusals (`arm_conflicts`): `--save-series` / `--locator-tau` on a pulse
  arm, with `--estimate`, or with `--tiny`; `--from-series` with any flag
  that builds or runs a scene.

The curve, phase, fit and class lines are printed BY DEFAULT on the CW arm.
Why: they are what a GPU record is for, they cost ~50 readings of one window
(nothing beside the run), and they change no existing line: every line the
arm printed before is printed with the same text, in the same order, first.
Measured: the full `--tiny` run's output is identical to main's line for line
once the new block and the wall-clock lines are set aside, exit 0, 21
controls.

`dockerfiles/maxwell_solver/fdtd.py` is NOT changed: the arm composes its
existing public functions and reads two of its private helpers (`_window`,
`_windowed_at`) for the phase prediction and the correlation, without
changing them.

## Impact

- `scripts/paper1_nanowire.py` (CW arm, parser, `arm_conflicts`, `main`).
- Tests: `tests/test_cw_settling.py` (new, 27 tests, ~17 s),
  `tests/check_discriminating.py` (`settling-*` cases).
- No route, request field, schema, worker or solver change.
- Not done here: the full-size run (the GPU's); a fix to `fit_decay`'s
  frequency grid (a separate change: it moves the ring-down arm's numbers).
