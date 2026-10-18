# time-domain

## ADDED Requirements

### Requirement: The paper-1 ring-down fit SHALL scan omega finer than any mode's basin

`fit_decay` in `scripts/paper1_nanowire.py` SHALL scan its 31 log-spaced
taus and, at each tau, every omega of the band on zero-padded FFT bins at
most 1/(`DECAY_PAD` = 8) windows apart and never coarser than a twentieth of
the band. The scan SHALL only choose basins; every reported number SHALL come
from exact least squares.

#### Scenario: the cw-settling 1060 fs case on a synthetic tail
- **WHEN** a tail of 3,000 samples at 0.15 fs (a 450 fs window) holds one mode
  of tau 1060 fs midway between two of the old 21 columns of a +-15% band at
  331.3 THz, with 1e-3 of noise
- **THEN** tau is within 0.2% of 1060 fs (measured 1060.35 fs. The old search
  returned 108.05 fs with rms 0.38 of the peak. 40 noise seeds scatter by
  0.271 fs with a 0.004 fs bias), rms is under 2e-3 of the peak, and the fit
  is not ambiguous

#### Scenario: a sweep of linewidth against the old grid step
- **WHEN** tails with tau from 60 fs to 4000 fs (a line 0.43 to 0.016 of the
  old 4.97 THz step) sit at 0, 0.33, 0.5 and 0.83 of the way between two old
  columns, with 1e-3 of noise
- **THEN** every tau is within 1% of its truth. Measured over 60 such tails,
  the worst was 0.19%. The old search got 33 of 60 wrong by 14.6% to 97%.
  Its descent multi-started from the best 3 or 5 cells got 33 and 31 wrong.
  This search with bins 1/window apart got 9 wrong, and with bins 1/(4
  windows) apart none

### Requirement: The ring-down fit SHALL descend inside the chosen basin

`fit_decay` SHALL descend exactly from the best cell: tau within two grid
cells of the cell's, and omega at each tau within the larger of two bins and
the cell's own line (1/tau rad/s) either side, clamped to the band.

#### Scenario: a line wider than the band
- **WHEN** a 1.9 fs decay (the `--tiny` vacuum tail's shape: 2,571 samples
  of 0.02322 fs, a +-8% band at 333.1 THz) sits at 315 or 332.6 THz under
  1e-4 of noise
- **THEN** f is within 0.2 THz and tau within 0.5%, and rms is under 1.2e-4.
  Measured: within 0.02 THz and 0.05%, as the old search was. With omega held
  to two bins the fit was 6 to 31 THz off at rms 34 to 160 times the noise.
  The `--tiny` vacuum tail itself re-fits to 332.5800 THz and 1.9008 fs,
  against the old search's 332.5797 THz and 1.9009 fs at the same rms

### Requirement: The ring-down fit SHALL refine separated minima beside the fit

`fit_decay` SHALL refine up to `RIVALS` = 3 local minima of the scan, lowest
first, whose omega is separated from the fit's by more than twice the larger
of 1/(pi tau) and 1/window (times 2 pi). One that refines lower SHALL become
the fit. `rival` SHALL hold the best remaining one: its f, tau, and residual
over the fit's.

#### Scenario: the scan ranks a wrong cell first
- **WHEN** the scan's residual at one cell 12% below a 400 fs mode is forced
  under its minimum
- **THEN** the fit is still the mode within 1%, and that cell is the rival

### Requirement: The ring-down fit SHALL say when its tau is not a measurement

`ambiguous` SHALL be a reason, not None, when the rival's ratio is under
`AMBIGUITY_RATIO` = 12 or when `tau_err` is None. An undetermined `f_err`
SHALL NOT make a fit ambiguous: the pinv drops omega's direction for scale
alone, so it is None on well-fitted tails too.

#### Scenario: how the factor was measured
- **WHEN** 105 random tails are fitted: 60 two-mode records with a second
  amplitude of 0.05 to 1, and 45 one-mode records under 0.01 to 1 of noise
- **THEN** 16 fits were more than 10% off their mode's tau, and 14 of them had
  a rival at 1.05 to 11.1 times the fit's residual, so 12 is the round factor
  over those. The two it misses are a second mode at 0.12 amplitude whose
  rival was 122 times (12.2% off), and one at 0.25 amplitude that formed no
  separated minimum (12.8% off). 15 of the 60 two-mode tails formed none. The
  price is 29 fits within 10% flagged as well; the first flagged one-mode
  tail was at 0.16 noise (ratio 11.5, tau 1.7% off). The `rms ... against a
  ... peak` reading stays the check for a second mode the flag cannot see

#### Scenario: two modes of equal amplitude
- **WHEN** a tail holds two 1060 fs modes 10 THz apart, or a 400 fs and a
  250 fs mode 10 THz apart, of equal amplitude
- **THEN** the fit is ambiguous with a second minimum at the other mode, and
  the rival's residual is 1.000 and 1.48 times the fit's (measured)

#### Scenario: noise, alone or over a mode
- **WHEN** a tail is unit noise alone, or a 1060 fs mode under noise of its own
  amplitude
- **THEN** both are ambiguous, at ratios 1.0006 and 1.34 (measured. Noise alone
  over 16 seeds gave 1.000 to 1.003)

#### Scenario: the factor from both sides
- **WHEN** a 400 fs mode carries a 250 fs mode at 0.6 of its amplitude 10 THz
  above it, or at 0.2 of its amplitude 25 THz above it
- **THEN** the first is ambiguous, at ratio 4.11 with its tau 24% long. The
  second is not, at ratio 36.8 with its tau 3.3% long

#### Scenario: a degenerate tau direction
- **WHEN** the tail is constant (2,000 samples of 1e-9)
- **THEN** `tau_err` is None and `ambiguous` names the degenerate covariance

### Requirement: The ring-down arm SHALL print an ambiguous tau as its reason

When `fit_decay` returns `ambiguous`, the arm's `fit` line SHALL print
`tau AMBIGUOUS (<tau> fs is not a measurement: <reason>)` in place of
`tau = <tau> fs +- <err>`. The arm SHALL then print `unresolved: the fit is
ambiguous -- <reason>` and no Q and no three-sigma bound, and return 3. A fit
without the key, or with it None, SHALL print exactly as before.

#### Scenario: the arm on a canned record
- **WHEN** the arm (`_run` stubbed, everything after it real) reads two equal
  1060 fs modes 10 THz apart
- **THEN** it prints `tau AMBIGUOUS (` with the second minimum, no `Q` line
  and no `Q >`, and returns 3. On one clean 400 fs mode it prints
  `tau = ...` and its Q, and returns 0
