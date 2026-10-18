# decay-fit-basin

## Why

`fit_decay` (`scripts/paper1_nanowire.py`) scanned 31 taus by 21 omegas and
descended from the best cell. Where the omega step (4.97 THz on a +-15% band
at 331 THz) is wider than a mode's line, the descent starts in a wrong basin.
Its tau bracket shrinks after the first sweep, so it cannot walk back
(why multi-starting it does not help). Change `cw-settling` measured 116.3
fs for a 1060 fs mode; a synthetic copy (3,000 samples) gives 108.05 fs at
rms 0.38 of peak. The full-size 113.1 +- 2.9 fs, with rms a quarter of its peak,
has the same shape.

## What Changes

- The scan puts omega on zero-padded FFT bins 1/(8 windows) apart at each
  tau, never coarser than a twentieth of the band; the descent stays in
  the best cell's basin (omega within a short tau's line). On 60 tails
  (lines 0.43 to 0.016 of the old step), wrong-basin answers (>5% off) were:
  old 33; old descent multi-started from 3 or 5 cells, 33 and 31; bins
  1/window, 9; 1/(4 windows) and 1/(8 windows), 0 (worst 0.19%).
- The three lowest separated minima are refined; one that refines lower
  becomes the fit; `rival` holds the best.
- `ambiguous` gives a reason when a rival's residual is under
  `AMBIGUITY_RATIO` = 12 times the fit's, or when tau's covariance direction
  is degenerate. The measured value: in 105 random two-mode and noisy
  one-mode records, 14 of the 16 fits >10% off had a rival at <= 11.1 x.
- The ring-down arm prints `tau AMBIGUOUS (... reason)` in place of
  `tau = ... +- ...`, prints no Q and no bound, and returns 3.

## Outputs that can move

- The `fit` line's f, tau and rms where the old fit sat in a wrong basin;
  elsewhere only at the descent's tolerance.
- An ambiguous fit's `fit` line, its `unresolved:` line, and exit code 3.
- `--tiny` lines quoting fitted values (band window, saved-tail re-fit,
  vacuum tail arm).
- Not moved: `f +- undetermined`. That is not ambiguity. The pinv drops
  omega because the Jacobian's columns differ in scale, even on good fits.
  The same scaling makes `tau_err` 0.56 of the measured scatter (0.152
  against 0.271 fs, 40 seeds). That is left for its own change.

## Impact

`scripts/paper1_nanowire.py`, `tests/test_decay_fit_basin.py`,
`tests/check_discriminating.py` (`basin-*`). Spec: `time-domain` (added).
