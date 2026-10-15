# resonance-extraction

## ADDED Requirements

### Requirement: A short record SHALL yield its resonances

`resonances.find_resonances(signal, dt, f_min, f_max)` SHALL fit a uniformly
sampled 1-D record (real or complex, `dt` in seconds) to a sum of damped
sinusoids `a exp(-2 pi i f t - gamma t)` and return every resonance with
`f_min <= f <= f_max` as `Resonance(f, Q, amplitude, error)`, sorted by `f`,
with `Q = pi f / gamma`, from a record much shorter than `1/gamma`. Only
numpy is required. Source: `dockerfiles/maxwell_solver/resonances.py`.

#### Scenario: Q = 1e5 from one twentieth of a decay time
- **WHEN** a single damped sinusoid with Q = 1e5 is sampled over 0.05 of its
  amplitude decay time 1/gamma, at 4 samples per period, with no noise
- **THEN** `f` is within 1e-7 relative and `Q` within 1% of the truth, for
  both `method="fdm"` and `method="pencil"`, for a complex and for a real
  record, and the same holds at f = c / 1536 nm in SI units
- **AND** a windowed FFT of the same record (`fdtd.spectrum`) has a bin
  width more than 60 times the linewidth, so it cannot give Q at all

#### Scenario: two modes two linewidths apart beside a strong distant mode
- **WHEN** the record holds two modes separated by twice their linewidth
  f/Q, with amplitudes 1 and 0.5, and a third mode 40 times stronger a few
  hundred Fourier bins away, with and without noise at 60 dB
- **THEN** a window holding all three returns three resonances, each within
  its own error bars of the truth, and a window holding only the pair
  returns the pair with the same accuracy (the distant mode is carried as a
  nuisance pole, not reported)

### Requirement: Real and complex records SHALL follow one stated convention

A complex record's positive `f` SHALL mean a phase rotating as
`exp(-2 pi i f t)` (Harminv's convention). A real record
(`numpy.isrealobj`) SHALL be fitted as the real part of the same sum, its
amplitude SHALL be the full cosine amplitude, and `reconstruct(result, dt,
n, real=True)` SHALL rebuild the fit. Source:
`dockerfiles/maxwell_solver/resonances.py` (module docstring,
`_pole_to_mode`, `polish`).

#### Scenario: the real path equals the complex path
- **WHEN** the same modes are sampled as a complex record and as its real
  part
- **THEN** both records give the same `f`, `Q` and `amplitude` within 1e-9
  relative, 1e-5 relative and 1e-6 relative

### Requirement: Error bars SHALL bound the error

`error` SHALL carry `error_sigma` (default 3) standard deviations of the
Cramér-Rao bound on `f`, `Q` and `|amplitude|` for the full multi-mode
model, at the maximum-likelihood fit (`polish=True`), with the noise
variance read from the window periodogram of the residual. Source:
`cramer_rao`, `polish`, `_noise_variance` in
`dockerfiles/maxwell_solver/resonances.py`.

#### Scenario: noise sweep
- **WHEN** one mode with gamma T = 0.5 is sampled with per-sample SNR of 0,
  10, 20, 40 and 60 dB, three seeds each
- **THEN** the reported `error.Q` bounds `|Q - Q_true|` and `error.f`
  bounds `|f - f_true|` at every seed and SNR
- **AND** the median of `|Q - Q_true| / error.Q` over the sweep is above
  0.05 (the bar is not loose)
- **AND** the median relative Q error at each SNR is within a factor 2 of
  `sqrt(6 / (SNR N)) / (gamma T)` for a complex record and twice that for
  a real one (the single-mode Cramér-Rao bound in the short-record limit)

### Requirement: Spurious modes SHALL not be reported

A mode SHALL be reported only if it passes two significance tests at
`min_significance` (default 6) noise standard deviations: the power its
pole removes from the residual given the poles already accepted, and,
after the polish, its amplitude against its own standard deviation from
the full Fisher matrix; `error.significance` SHALL be the latter ratio.
Source: `_greedy_select` and `_package` in
`dockerfiles/maxwell_solver/resonances.py`.

#### Scenario: the noise level is not lowered by noise-fitted poles
- **WHEN** the forward test judges a candidate
- **THEN** the noise level it is judged against comes from the record's
  own window periodogram first and then from the residual of the accepted
  modes only, never from a fit that includes the candidate

#### Scenario: pure noise
- **WHEN** the record is white Gaussian noise, real or complex, three seeds
- **THEN** `find_resonances` returns an empty list for both methods
- **AND** the control with `min_significance = 0` returns at least one mode,
  so the empty list is the test's doing, not an empty window

### Requirement: The two methods SHALL agree on their own

Without the polish (`polish=False`, which would otherwise take both to the
same maximum-likelihood point) the filter-diagonalization poles and the
matrix-pencil poles SHALL agree on every synthetic record: to 1e-8 relative
in `f` and 1e-3 relative in `Q` without noise, and within ten 1-sigma bars
of each other and of the truth at 40 dB. Source: `fdm`, `matrix_pencil` in
`dockerfiles/maxwell_solver/resonances.py`.

#### Scenario: raw agreement
- **WHEN** the Q = 1e5 record, the three-mode record and a 40 dB one-mode
  record are solved by both methods with `polish=False`
- **THEN** the counts match and every pair of poles agrees as stated
