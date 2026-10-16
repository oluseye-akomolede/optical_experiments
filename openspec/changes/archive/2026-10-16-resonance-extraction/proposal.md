# resonance-extraction

**New capability, post-processing only.** A numpy-only module,
`dockerfiles/maxwell_solver/resonances.py`, that takes a uniformly sampled
real or complex time record and returns the resonances inside a frequency
window as `(f, Q, amplitude, error)`, from a record much shorter than the
decay time. Nothing in `fdtd.py` or any existing module changes; the module
imports nothing from them, and its tests import `fdtd` only for `C0` (the
1536 nm example) and `spectrum` (the FFT control).

## Why

Paper 2 (arXiv 2505.00224, erbium ions in silicon nanophotonic cavities near
1536 nm) is about cavities with Q between 1e4 and more than 1e5. The FDTD
solver's `fdtd.spectrum` is a windowed FFT: it resolves a linewidth only when
the record is a few decay times long, which at Q = 1e5 and 195 THz is
hundreds of picoseconds, millions of steps, beyond `MAX_STEPS`. The resonance
parameters are nevertheless determined by a much shorter record, because a
damped sinusoid is a four-parameter model and a record of thousands of
samples over-determines it by far. Harmonic inversion fits that model.

## What Changes

- `resonances.find_resonances(signal, dt, f_min, f_max, ...)` implements the
  filter-diagonalization method of Mandelshtam & Taylor (1997), the algorithm
  of Harminv, over a user-given window, and the matrix pencil of Hua & Sarkar
  (1990) as an algebraically unrelated cross-check (`method="pencil"`).
- Both go through one post-processing pipeline: a statistical significance
  test against the noise level (not Harminv's eigen-residual, which this
  change measured to pass noise-fitted modes), a Levenberg-Marquardt
  maximum-likelihood polish, and Cramér-Rao error bars on f, Q and
  amplitude at the polished fit. The bars are what makes "error estimate"
  mean something: the raw methods scatter by a few times the bound, the
  polished fit sits on it.
- The numerical limits (round-off floor, record-length floor, resolution,
  the window and a mode outside it, cost) are stated in the module docstring
  and measured in `docs/resonances.md`.

## Impact

- New files: `dockerfiles/maxwell_solver/resonances.py`,
  `tests/test_resonances.py`, `docs/resonances.md`, this change.
- No route, payload, schema, Dockerfile, requirement file or manifest
  changes. scipy stays optional (unused).
- Review-risk questions: none answered yes. Nothing here runs in a job yet;
  wiring it into a `td` job's probe traces is a later change.
