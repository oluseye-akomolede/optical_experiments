# settling-tail-fit

## Why

The CW arm's settling forecast (`fit_approach` via `classify_settling`,
`scripts/paper1_nanowire.py`) fitted `F_inf - A exp(-t/tau_s)` unweighted
over every cut past the ramp. A 600-period run's curve rose to 13.26 and was
still rising, yet the fit printed F_inf = 13.1935 +- 0.02 (tau_s 82.99
periods, residual 2.83e4 x the floor). The early fast rise sets the fit, so
tau_s is short and F_inf low. Over 27 two-exponential curves with known
limits, this fit forecast below the end on 18 or 19.

## What Changes

- Where the whole-curve residual is over `SIGNIFICANCE` (3) x the floor,
  the forecast is refit over the cuts in the record's last half
  (`TAIL_SHARE` = 0.5). Otherwise the whole-curve fit stands, unchanged.
  Chosen over two exponentials (on the 600-period curve their residual
  stays 1e4 x the floor and the limit wanders 13.60-13.76) and over
  dropping cuts until the residual reaches the floor (worse with 1e-4
  noise: median error 0.15 of the remaining rise against 0.05-0.09).
- `forecast_check`: an F_inf behind a still-moving end by more than
  `INCONSISTENT_SIGMA` = 2.5 of hypot(its error, the floor) is printed as
  INCONSISTENT and no extrapolation is given. A forecast whose residual is
  over `FORECAST_RESIDUAL` = 3 x the floor is printed as UNRELIABLE.
- The 600-period curve now forecasts 13.3068 +- 0.0039 (tau_s 115.1
  periods), labelled UNRELIABLE (1029 x the floor).

## Outputs that can move

- `fit`: a new line naming the tail refit, only when the whole-curve
  residual is over 3 x the floor. The existing `fit` lines are unchanged.
- `class (b)`: F_inf and its error now come from the forecast. The old
  "more than this one time constant" clause moves to the refit line. An
  UNRELIABLE clause or an INCONSISTENT line can appear.
- The class itself, the `locator` lines and the returned `fit` follow the
  forecast's rate and tau_s.
- Not moved: the `measured`, `settling`, `phase` and `curve` lines, F, and
  any record whose whole-curve residual is at most 3 x the floor.

## Impact

`scripts/paper1_nanowire.py`, `tests/test_cw_settling.py`, `tail-*` and
re-measured `settling-*` cases in `tests/check_discriminating.py`,
`time-domain` spec. It amends `cw-settling`'s fit and class requirements;
see the delta.
