# time-domain

## ADDED Requirements

### Requirement: The CW arm's settling forecast SHALL follow the curve's slow tail

`classify_settling` in `scripts/paper1_nanowire.py` SHALL keep the
whole-curve fit (`whole`) and SHALL refit the forecast (`fit`) over the
fitted cuts in the record's last `TAIL_SHARE` = 0.5 (at least 5) when the
whole-curve residual is over `SIGNIFICANCE` = 3 x the floor. Otherwise the
forecast SHALL be the whole-curve fit itself. The class, the class line's
F_inf and the locator lines SHALL read the forecast.

#### Scenario: the 600-period run's curve
- **WHEN** the 20 cuts the run printed (31 to 600 periods, F 3.31597 to
  13.26, still rising) are classified with the run's floor (0.0552 / 2.83e4)
- **THEN** the whole-curve fit is the one the run printed (13.1935 +- 0.020,
  82.99 periods, 2.83e4 x the floor), and the forecast is the fit over the 10
  cuts from 338 periods: F_inf = 13.3068 +- 0.0039, tau_s = 115.1 periods,
  at or above the 13.26 already reached

#### Scenario: sums of two exponentials with known limits
- **WHEN** 1 - 0.7 exp(-t/fast) - 0.3 exp(-t/slow) is cut 20 times 30 apart
  (fast 10 or 25, slow 80 to 300) under 1e-5 of white noise, floor 1e-5
- **THEN** the forecast is within 3 of its errors and within 5e-4 of 1, and
  the whole-curve fit is at least 1e-3 off. Measured over 20 seeds x 6
  shapes: within 2.53 errors and 1.73e-4, against 1.6e-3 or more for the
  whole curve, and never flagged

#### Scenario: one clean exponential is not refit
- **WHEN** 1 - 0.8 exp(-t/tau) with tau 60, 100 or 200 is cut the same way
  under 1e-5 of noise, floor 1e-5
- **THEN** the forecast is the whole-curve fit, F_inf and tau_s value for
  value. Measured over 900 such records, the whole-curve residual reached
  1.44 x the floor, so none was refit. The cw-settling 30-period approach
  keeps tau_s = 30 periods and F_inf within 1e-5, unchanged

#### Scenario: the cw-settling two-time-constant record (a changed expectation)
- **WHEN** 1 - 0.5 exp(-t/10) - 0.3 exp(-t/60 periods) over 200 periods is
  read through the arm
- **THEN** the whole-curve fit still has tau_s between 10 and 60 (37.4) and
  a residual 1.1e4 x the floor, so the "more than this one time constant"
  words still print, on the refit line now. The forecast's tau_s is the slow
  60 periods within 5e-3 (measured 59.91) and its F_inf is the limit within
  1e-4 (measured 3.2e-5 under, where the old one was 1.72% under, below the
  end's 1.14%). The test now asserts the old bounds on `whole` and these on
  `fit`

#### Scenario: noisy records are refit too (a changed expectation)
- **WHEN** the cw-settling coverage records (200 noisy 30-period approaches
  over 60 periods, noise 0.05 of the in-phase amplitude) are classified
- **THEN** their noise is far above the phase floor, so all 200 are refit,
  and both the whole-curve fit and the forecast keep the old bounds: a z
  spread between 0.75 and 1.4, and 95% or more inside 3. Measured over 6
  seeds: the whole-curve fit 0.93-1.21 and 97.5-99.5%; the forecast
  1.12-1.34 and 96-99%, widest on the test's own seed

### Requirement: The CW arm SHALL refuse a forecast its own record contradicts

`forecast_check` SHALL call an approach's forecast inconsistent when the
last third's line moves by more than 3 of its errors and F_inf lies behind
the end's reading, below it on a rise or above it on a fall, by more than
`INCONSISTENT_SIGMA` = 2.5 of hypot(F_inf's error, the floor). The class
line SHALL then say INCONSISTENT, name the end and the gap, and print no
extrapolation.

#### Scenario: the 600-period run's old forecast
- **WHEN** the run's whole-curve forecast (13.1935 +- 0.020) is checked
  against its rising end (13.26), and against the same curve mirrored to fall
- **THEN** both are inconsistent at 3.33 of their errors, and the tail
  forecast of either is not

#### Scenario: correct forecasts are not refused
- **WHEN** single exponentials (tau 60 to 250 over 600, noise 1e-5 to 1e-3 of
  the limit, the end within noise of it at the longest) are classified
- **THEN** none is inconsistent. Measured: the wrong-side gap reached 1.92
  errors over 1623 such records, and 0.998 over 206 noisy series through the
  reading. Of 61 two-exponential whole-curve forecasts below their end, 30
  were caught at 2.5 (19 at 3), and the 600-period one at 3.33

#### Scenario: the end's error is the floor, not the last third's scatter
- **WHEN** the gap's error is taken from the fit's own per-point sigma
- **THEN** the 600-period forecast reads 1.05 errors behind its end and is
  not caught. The last third's line scatter, 0.0138 there against a floor of
  2e-6, is the curve's bend, not the end's noise

#### Scenario: the printed line
- **WHEN** the 600-period curve is replayed through the reading ten times
  faster (60 periods, 3,862 samples)
- **THEN** the refit line names the last 50% (11 cuts from 30 periods), the
  forecast lies 0.56% above the end where the whole-curve one lay 0.16%
  below, and the class line gives it as the extrapolation. With the refit off
  and the threshold at 0.5, the line says INCONSISTENT and prints no
  EXTRAPOLATION

### Requirement: The CW arm SHALL say when its forecast is unreliable

The class line SHALL call an approach's forecast UNRELIABLE when the
forecast fit's residual is over `FORECAST_RESIDUAL` = 3 x the floor. It SHALL
say that what was fitted still holds more than one time constant, or noise
the floor does not account for.

#### Scenario: the threshold
- **WHEN** 215 tail forecasts of two-exponential curves (fast 5 to 50, slow
  80 to 300, noise 1e-6 to 1e-4) are compared with their known limits
- **THEN** the 167 within 3 errors of the limit reached 2.74 x the floor and
  none is flagged. Of the 48 further off, 30 are flagged at 3, 26 at 5 and
  11 at 30. The other 18 sit at 0.72 to 1.72 x the floor (16 of them fast 50
  against slow 80 to 300), so no residual test can see them. The cw-settling
  two-time-constant forecast, 3.7 errors off, is flagged at 3.84

#### Scenario: the 600-period run's curve
- **WHEN** its tail forecast is classified
- **THEN** its residual is 1029 x the floor, so it is UNRELIABLE (and not
  inconsistent)

### Requirement: Each forecast rule SHALL be held by a test a reversion turns red

`tests/check_discriminating.py` SHALL hold a `tail-*` case for each rule
above. Each case SHALL apply, and SHALL turn exactly its named tests in
`tests/test_cw_settling.py` red.

#### Scenario: the cases and the re-measured cw-settling sets
- **WHEN** each of the 12 `tail-*` and 13 `settling-*` cases is run
- **THEN** each turns exactly its set red. Four `settling-*` sets moved. Free
  phase adds the two-time-constant test, whose forecast now has to land
  within 1e-4. Ramp-fitted drops from 8 tests to 4: the ramp's misfit sends
  the forecast to the last half, past the ramp. Limit-sign grows to 12, since
  every new test reads F_inf. Misfit-as-noise now reverts on `whole`'s
  residual (the forecast's is small) and turns 6 red. Independent-cuts
  follows `fit` to `whole`, its set unchanged
