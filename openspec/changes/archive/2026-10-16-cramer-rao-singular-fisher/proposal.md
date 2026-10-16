# cramer-rao-singular-fisher

**Bug fix in `resonances.cramer_rao`: a bar the record does not determine
is reported as NaN, never as a number.** Adds one requirement to the
`resonance-extraction` capability (under "Error bars SHALL bound the error");
nothing else in that spec changes.

## Why

`cramer_rao` inverted the Fisher matrix with `np.linalg.pinv`. When two
modes coincide (same f and gamma) the matrix is singular, and the
pseudo-inverse's two identical rows split one variance four ways: each of
the pair came back with bars exactly half the true single-mode bars, and
nothing told the caller. Measured on the sweep's record (gamma T = 0.5,
40 dB, complex and real): every one of the four bars at 0.500 of the single
mode's. A bar that is too small is the worst failure an error bar can have,
because it passes every "more than n of its own errors" test downstream.

The same inverse also mis-served a weak nuisance mode: at 1e-6 of the
others' amplitude its block of the raw matrix falls under `pinv`'s
truncation and its bars came out 2.7e9 to 3.1e9 times too small in the
amplitude columns and 5.1e12 to 1.9e14 in f and gamma (`matrix_rank` of
the raw matrix drops from 15 of its 16 parameters to 12 as the amplitude
goes from 1e-4 to 1e-6, while the scaled condition number does not move,
which is why rank is not the measure).

## What Changes

- `cramer_rao` scales the Fisher matrix to unit diagonal (invariant to the
  amplitude and units of every mode), takes its eigen-decomposition, and
  treats eigenvalues below `1 / FISHER_CONDITION_LIMIT` (1e-12, a named
  module constant with its choice explained beside it) of the largest as
  directions the record does not determine. A mode with any parameter along
  such a direction gets NaN in all four columns; the others get the inverse
  over the determined directions. Shape `(len(modes), 4)`, column order and
  signature are unchanged, so `_package` and every formatter see NaN bars
  (`error.f`, `error.Q`, `error.amplitude`) with no further change.
- Tests in `tests/test_resonances.py`: duplicate poles (NaN, with the
  single mode's bars finite and the pseudo-inverse's half-bars pinned),
  separated modes unchanged to 1e-12 relative, a near-duplicate pair on each
  side of the limit (0.02 bins at 0.8 of it, 0.015 bins at 4.4 times it), and
  the weak nuisance mode. Mutation case `resonances-fisher-unguarded` in
  `tests/check_discriminating.py`.

## Impact

- `dockerfiles/maxwell_solver/resonances.py`, `tests/test_resonances.py`,
  `tests/check_discriminating.py`, `docs/resonances.md`. numpy only, no new
  dependency, no route or payload change.
- Review-risk questions: none answered yes.
