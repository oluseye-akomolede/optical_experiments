# time-domain

## ADDED Requirements

### Requirement: The paper-1 CW arm SHALL print a settling curve from its own reading

After its existing `measured` and `settling` lines, the CW arm of
`scripts/paper1_nanowire.py` SHALL print its own reading (the last
`--window-periods` of the series it is handed) at one cut per 5% of the
record, each cut a whole number of drive periods before the end. A cut SHALL
enter the fit only once its window starts after the drive's ramp and its
delivered power is positive.

#### Scenario: free-phase cuts against locked ones
- **WHEN** a settled tone whose half-step field is 2000 times its own in-phase
  part, sampled 64.37 steps a period for 61.3 periods, is read at free 5%
  marks and by the curve
- **THEN** the free cuts spread by more than 0.04 of F (measured 0.059), the
  curve's by less than 0.005 (measured 0.0017, at a phase mismatch under
  0.004 periods), and the record is classed settled

### Requirement: The CW arm SHALL measure what the window's phase alone does to its reading

The arm SHALL read the record at cuts across its last drive period, fit
`a + b x + c cos 4 pi x + d sin 4 pi x`, and print the swing `|c, d|`, its
prediction (twice the Hann window's leakage at 2f times the field's
quadrature ratio, read from the estimator's own half-step field), the end
reading with the 2f term removed, and the share of the two-thirds line's
difference that the swing alone allows.

#### Scenario: closed-form tones
- **WHEN** settled tones with quadrature ratios of 100 and 2000 are read with
  4, 8 and 16-period windows
- **THEN** the swing is its prediction within 1e-3 (measured 2e-4 or better),
  the quadrature read back is the one put in within 1e-3, and the
  phase-free reading is one F at every window within 1e-5

#### Scenario: an open vacuum box through the solver
- **WHEN** an x dipole in a 25^3-node open vacuum box (dx 20 nm, CPML 6,
  900 nm, 40 periods) is read with 4, 8 and 16-period windows
- **THEN** the swing (measured 21.9%, 2.45% and 0.245% of F) is its
  prediction within 1e-3, the phase-free reading agrees across the windows
  within 2e-4 (measured 4e-5), the two-point difference lies between 0.1 and
  2 swings (measured 0.42 of 2), and the record is classed settled

### Requirement: The CW arm SHALL fit the curve with errors that allow for shared window samples

The arm SHALL fit `F(t) = F_inf - A exp(-t/tau_s)` with the rate free to
change sign, by unweighted least squares, with 1-sigma errors from the
sandwich covariance under the cuts' Hann-overlap correlation, floored at the
record's phase floor, and print F_inf, tau_s (fs and periods), their errors
and the residual against the floor.

#### Scenario: noisy records
- **WHEN** 200 records of a 30-period approach over 60 periods carry white
  noise on the field
- **THEN** (fit - truth) / error has a spread between 0.75 and 1.4 and 95% or
  more inside 3, for both F_inf and tau_s (measured 1.17 / 1.16 and 98-99%)

### Requirement: The CW arm SHALL state which of four classes its record is in

The arm SHALL print one class: (a) settled, when a line through the final
third moves less than 3 of its own errors; (b) approaching, when the rate is
more than 3 errors above zero, F_inf printed as an extrapolation; (c) growing
without a limit, when it is more than 3 errors below zero, never reported as
a Purcell factor; (d) unresolved otherwise, or under 5 fitted cuts.

#### Scenario: exact inputs
- **WHEN** the curve is read from 200-period closed-form records whose
  in-phase envelope is constant; 1 - 0.8 exp(-t/30 periods); 1 - 0.5
  exp(-t/10) - 0.3 exp(-t/60); 1 + 0.02 (exp(t/100) - 1); and a straight rise;
  and from a 14-period record
- **THEN** they are classed (a); (b) with tau_s = 30 periods and F_inf = the
  constant's F within 1e-5 and within 3 of their errors; (b) with tau_s
  between 10 and 60 and the more-than-one-time-constant flag; (c) with a
  growth time of 100 periods within 1e-4 and "NOT a Purcell factor"; (d); and
  (d) for want of cuts

#### Scenario: a cavity whose decay is measured by its own ring-down
- **WHEN** a 9 x 17 x 19-node PEC box with Rakić gold on all six walls is rung
  down (fit_decay over a +-5% band, and the log of its tail's envelope) and
  then driven at its mode for 2 and for 10 decay times
- **THEN** the 2-decay curve is classed (b) with tau_s the ring-down's within
  1e-3 (measured 99.748 fs against 99.75), the envelope agrees with the
  ring-down within 5e-3, F_inf agrees with the 10-decay record's within 2e-5
  (measured 7e-6), and the 10-decay record is not classed growing

### Requirement: The CW arm SHALL set its settling time against a locator's in plain words

With `--locator-tau FS[,ERR]` the arm SHALL say whether tau_s agrees with the
locator's tau within their errors, is shorter, or is longer beyond both; when
longer it SHALL say that a single mode with the locator's tau cannot make the
curve, and that either something slower moves the reading or the locator's
tau is not this mode's, without naming a mechanism. A curve with no tau_s
SHALL say so.

#### Scenario: three locators
- **WHEN** the 30-period approach (90.9 fs) is set against 90.9 fs and against
  9.09 fs, and a settled tone against 113 fs
- **THEN** the lines read "agrees", "cannot make this curve ... or the
  locator's tau is not", and "no tau_s to set against it"

### Requirement: The CW record SHALL be savable and re-readable, off by default

`--save-series PATH` SHALL write the CW record and every constant its reading
needs as a compressed .npz at the caller's path, after the reading is printed,
and a failed write SHALL leave the reading. `--from-series PATH` SHALL print
the same reading, curve and class from such a file without a run.

#### Scenario: the round trip
- **WHEN** a float32 record is saved through the arm and read back
- **THEN** the reading from the file is the arm's F bit for bit, and
  `--from-series` prints the run's `measured`, `settling`, `phase`, `curve`,
  `fit` and `class` lines unchanged

### Requirement: The new CW flags SHALL be refused where they would be inert

`--save-series` and `--locator-tau` SHALL be refused on a pulse arm, with
`--estimate` and with `--tiny`, and `--from-series` with any flag that builds
or runs a scene.

#### Scenario: inert combinations
- **WHEN** `--save-series` is given with `--spectrum`, `--ringdown`, `--tiny`
  or `--estimate`, `--locator-tau` with `--spectrum`, or `--from-series` with
  `--tiny`, `--save-series` or `--collection-na`
- **THEN** each is refused before any scene is built, naming the flag

### Requirement: Without the new flags the CW arm's existing output SHALL be unchanged

Every line the CW arm printed before this change SHALL be printed with the
same text and in the same order, before any new line, and no file SHALL be
written. The curve block is printed by default.

#### Scenario: the canned record and the tiny run
- **WHEN** `measure` runs on a canned record without the new flags, and the
  full `--tiny` run is compared with main's
- **THEN** the first two lines are the `measured` and `settling` lines as
  fdtd's own functions give them, nothing is written, and the `--tiny` output
  is main's line for line once the new block and the wall-clock lines are set
  aside
