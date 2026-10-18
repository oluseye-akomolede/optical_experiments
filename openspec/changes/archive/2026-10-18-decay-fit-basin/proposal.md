# decay-fit-basin

## Why

`fit_decay` (`scripts/paper1_nanowire.py`) scanned 31 taus by 21 omegas and
descended from the best cell. Where the omega step (4.97 THz on a +-15% band
at 331 THz) is wider than a mode's line, the descent starts in a wrong basin.
Its tau bracket shrinks after the first sweep, so it cannot walk back
(so multi-starting it fails). Change `cw-settling` measured 116.3
fs for a 1060 fs mode; a synthetic copy (3,000 samples) gives 108.05 fs at
rms 0.38 of peak. The full-size 113.1 +- 2.9 fs, rms a quarter of peak,
looks the same.

## What Changes

- The scan puts omega on zero-padded FFT bins 1/(8 windows) apart at each
  tau, never coarser than a twentieth of the band; the descent stays in
  the best cell's basin (omega within a short tau's line). On 60 tails
  (lines 0.43 to 0.016 of the old step), wrong-basin answers (>5% off) were:
  old 33; old descent multi-started from 3 or 5 cells, 33 and 31; this,
  0 (worst 0.19%). On 60 long quiet lines, bins 1/window apart left 5
  (13-18% short), 1/(2 windows) and finer none.
- Three separated minima are refined; a lower one becomes the fit;
  `rival` holds the best.
- `ambiguous` gives a reason when a rival's residual is under
  `AMBIGUITY_RATIO` = 12 times the fit's, or when tau's covariance direction
  is degenerate. Measured: in 105 random two-mode and noisy
  one-mode records, 14 of the 16 fits >10% off had a rival at <= 11.1 x.
- The ring-down arm prints `tau AMBIGUOUS (... reason)` for `tau = ...`,
  no Q and no bound, and returns 3.

## Outputs that can move

- The `fit` line's f, tau and rms where the old fit sat in a wrong basin,
  else only at the descent's tolerance.
- An ambiguous fit's `fit` and `unresolved:` lines, and code 3.
- `--tiny`: omega is clamped to the band, so the narrow-window control
  stops at its edge (316.45 THz, was 313.70); the vacuum tail's f moves
  in the fourth decimal, tau and rms not at all.
- Not moved: `f +- undetermined`, which is not ambiguity: the pinv drops
  omega for the Jacobian's column scales, even on good fits. That scaling
  also makes `tau_err` 0.56 of the scatter (0.152 against 0.271 fs, 40
  seeds); left for its own change.

## Impact

`scripts/paper1_nanowire.py`; `tests/test_decay_fit_basin.py`;
`basin-*` in `tests/check_discriminating.py`; `time-domain` spec.
