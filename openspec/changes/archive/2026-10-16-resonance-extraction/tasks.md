# Tasks — resonance-extraction

Tier 1 only: every test is a closed-form synthetic record, a few seconds,
`OMP_NUM_THREADS=2`. No FDTD or FEM run.

## 1. Module `dockerfiles/maxwell_solver/resonances.py`

- [x] 1.1 `find_resonances(signal, dt, f_min, f_max, *, method, ...)` →
      sorted list of `Resonance(f, Q, amplitude, error)`; `error` is
      `ResonanceError(f, Q, amplitude, residual, significance)`
- [x] 1.2 `fdm`: Mandelshtam & Taylor U^0, U^1, U^2 by the closed-form
      geometric sums, streamed in blocks; U0 regularised by its eigenvalues
      (Harminv's threshold); amplitudes from `(b^T G0)^2`; real records get a
      mirrored window so the conjugate poles are in the basis
- [x] 1.3 `matrix_pencil`: Hankel SVD, rank from the singular-value noise
      floor, poles from the shifted right singular vectors, least-squares
      amplitudes; a random row subset (not a stride) above `MAX_ROWS`
- [x] 1.4 Selection: greedy forward selection of raw poles by the power they
      remove, in units of the noise variance read from the window
      periodogram of what is left; nuisance poles outside the window stay in
      the model (the pencil supplies them to the FDM path)
- [x] 1.5 `polish`: variable-projection Levenberg-Marquardt with the Kaufman
      Jacobian, step cap, damping; never worse than its start
- [x] 1.6 `cramer_rao`: full multi-mode Fisher matrix, scaled time; bars on
      f, Q (linear propagation) and |a|, `error_sigma` standard deviations
- [x] 1.7 Module docstring: conventions (sign of f, real amplitude), the
      bound formula, the numerical limits with numbers
- [x] 1.8 numpy only; scipy not imported

## 2. Tests `tests/test_resonances.py` (each names the mutation it catches)

- [x] 2.1 Q = 1e5, record 1/20 of a decay time: f to 1e-7 relative, Q to 1%
      (both methods, complex and real; and at 1536 nm using `fdtd.C0`)
- [x] 2.2 Two modes 2 linewidths apart plus a distant strong mode: all three
      found with the right f, Q and amplitude
- [x] 2.3 Noise sweep: Q error vs SNR reported; the error bar bounds the
      actual error at every SNR and is not loose (median ratio floor)
- [x] 2.4 Pure noise: nothing reported; the control with the significance
      test off reports many
- [x] 2.5 Raw FDM and raw pencil agree on every synthetic case (no polish)
- [x] 2.6 Real (cosine) path equals the complex path in f, Q, amplitude
- [x] 2.7 The FFT control: `fdtd.spectrum` on the same record cannot resolve
      the linewidth (its bin is wider than the line)
- [x] 2.8 Each test's mutation confirmed red before commit (list in
      `docs/resonances.md`)

## 3. Verification

- [x] `OMP_NUM_THREADS=2 python -m pytest tests/test_resonances.py tests/test_fdtd.py -q`
      green on the rebased branch; no image smoke test (nothing in an image
      changes); no e2e job
- [x] `openspec validate resonance-extraction --strict`

## 4. Archive prep: the NaN requirement (2026-10-16, row 5dd42c5e)

- [x] "Error bars SHALL bound the error" is scoped to the modes and
      parameters the record determines, and names the requirement
      `cramer-rao-singular-fisher` adds to this capability for the rest.
      Unscoped, the two merged as a contradiction (reviewer N3, row
      30a2e1a6): a bar cannot both bound an error and be NaN. Spec text
      only -- the bars the code produces are the ones the two requirements
      now agree on.
- [x] `openspec validate --all` 17 passed, and a dry archive of this change
      then `cramer-rao-singular-fisher` in a scratch worktree leaves
      `openspec/specs/resonance-extraction/spec.md` holding both
      requirements with no contradiction between them.
