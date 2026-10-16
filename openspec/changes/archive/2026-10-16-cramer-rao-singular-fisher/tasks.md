# Tasks — cramer-rao-singular-fisher

Tier 1 only: closed-form synthetic records, a few seconds, `OMP_NUM_THREADS=2`.

## 1. `dockerfiles/maxwell_solver/resonances.py`

- [x] 1.1 `FISHER_CONDITION_LIMIT = 1e12`, with the choice (the inverse's own
      round-off at the limit, the pseudo-inverse's truncation past it, the
      measured condition numbers at 0.1 and 0.01 bins) in the comment beside it
- [x] 1.2 `_fisher`: the matrix and T, split out of `cramer_rao`
- [x] 1.3 `_fisher_inverse`: unit-diagonal scaling, eigen-decomposition, the
      inverse over the determined directions, and per parameter whether it is
      determined (a zero diagonal is undetermined outright)
- [x] 1.4 `cramer_rao`: NaN in all four columns of an undetermined mode; shape,
      column order and signature unchanged; docstring says so
- [x] 1.5 Module docstring: the "Undetermined bars" numerical limit

## 2. Tests `tests/test_resonances.py`

- [x] 2.1 Duplicate poles: NaN, the single mode's bars finite, the pseudo-inverse's
      half-bars pinned, a third resolved mode keeps its bars (complex and real)
- [x] 2.2 Separated modes: bars equal the plain inverse to 1e-12 relative (1e-6 for
      the three-mode TRIO)
- [x] 2.3 Near-duplicate pair at 0.8 and 4.4 times the limit, the condition number
      recomputed in the test
- [x] 2.4 A nuisance mode at 1e-6 amplitude: rank of the raw matrix drops, every
      bar finite, the nuisance's bars at 1/|a|
- [x] 2.5 Mutation `resonances-fisher-unguarded` in `tests/check_discriminating.py`
      confirmed red on exactly 2.1, 2.3 and 2.4; recorded in `docs/resonances.md`

## 3. Verification

- [x] `OMP_NUM_THREADS=2 python -m pytest tests/test_resonances.py -q` green on
      the rebased branch (29 tests); no image smoke test (nothing in an image
      changes); no e2e job
- [x] `python tests/check_discriminating.py resonances-fisher-unguarded`
- [x] `openspec validate cramer-rao-singular-fisher --strict`
