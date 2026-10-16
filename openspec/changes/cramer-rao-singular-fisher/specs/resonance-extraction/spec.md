# resonance-extraction

## ADDED Requirements

### Requirement: A bar the record does not determine SHALL be NaN, never a number

`cramer_rao` SHALL scale the Fisher matrix to unit diagonal, SHALL treat an
eigenvalue below `1 / FISHER_CONDITION_LIMIT` (1e-12) of the largest as a
direction the record does not determine, and SHALL return NaN in all four
columns of every mode with a parameter along it, with the shape, column
order and signature unchanged, so `error.f`, `error.Q` and
`error.amplitude` read NaN. Source: `cramer_rao` in
`dockerfiles/maxwell_solver/resonances.py`.

#### Scenario: duplicate poles
- **WHEN** two identical modes (same f, gamma and amplitude) are given with
  the sweep's record (gamma T = 0.5, 40 dB), complex or real
- **THEN** both rows are NaN in all four columns
- **AND** the same mode alone has finite, positive bars
- **AND** the pseudo-inverse of the same Fisher matrix, the formula before
  this change, gives each of the pair exactly half those bars
- **AND** a third, resolved mode in the same call has the bars it has
  without the pair, to 1e-12 relative

#### Scenario: separated modes are unchanged
- **WHEN** two modes 159 Fourier bins apart, or one bin apart, are given
- **THEN** every bar is finite and equals the pseudo-inverse's to 1e-12
  relative (the three-mode record of the two-close-modes test, pair 0.32
  bins apart, to 1e-6)

#### Scenario: a near-duplicate pair on each side of the limit
- **WHEN** two equal modes are 0.02 Fourier bins apart (scaled condition
  number 0.8 of the limit)
- **THEN** both get finite bars, more than a thousand times the single-mode
  bars, within 2e-3 of the pseudo-inverse's
- **WHEN** they are 0.015 bins apart (4.4 times the limit)
- **THEN** both get NaN, where the pseudo-inverse still prints a number

#### Scenario: a weak nuisance mode costs nobody their bars
- **WHEN** the three-mode record carries a fourth mode at amplitude 1e-6,
  far outside the window, so that `matrix_rank` of the raw Fisher matrix
  falls below 16
- **THEN** every bar is finite, the three modes' bars are within 1e-3 of
  their values without the nuisance, and the nuisance's bars are 100 times
  those of the same mode at amplitude 1e-4
