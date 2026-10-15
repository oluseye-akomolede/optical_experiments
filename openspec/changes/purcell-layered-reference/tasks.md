# Tasks — purcell-layered-reference

Tier 1 only: reference maths against closed forms, no FDTD or FEM solve.

## 1. The module (`dockerfiles/maxwell_solver/purcell_reference.py`)

- [x] 1.1 `purcell_layered(wavelength_m, height_m, film_thickness_m,
      film_index, metal="Au", orientation="perp"|"par")` returning
      `PurcellResult(F, radiated, transmitted, nonradiative)`; `metal` is a
      Rakić name (`fdtd.permittivity`), `"PEC"` or a constant permittivity
- [x] 1.2 Stack Fresnel coefficients (Airy, air | film | substrate), the
      `exp(-i w t)` convention of `fdtd.permittivity`, every square root
      with `Im >= 0`
- [x] 1.3 Sommerfeld integral along `s = u - i h sin(pi u / s_b)` then the
      real axis to `exp(-50)` decay; composite Gauss-Legendre
- [x] 1.4 Far-field power into the air; transmitted flux into a lossless
      substrate including the forbidden-light band; `nonradiative` as the
      remainder
- [x] 1.5 `image_dipole_pec(kd, orientation)`: the mirror closed form, cited
- [x] 1.6 CLI: `python dockerfiles/maxwell_solver/purcell_reference.py`
      prints F and the radiated share at 850-950 nm, both orientations,
      for 6 nm above 12 nm of n = 1.45 on Au (the defaults)

## 2. The checks (`tests/test_purcell_reference.py`)

Each test's docstring names the mutation that turns it red; the mutation
sweep is recorded in `docs/purcell-reference.md`.

- [x] 2.1 PEC, no film: the image dipole to 1e-4 for x = 2kd in [0.1, 20]
      (agrees to 2.4e-14); PEC radiates everything
- [x] 2.2 Large height: the stationary-phase asymptote and its envelope,
      x in [30, 200], both orientations
- [x] 2.3 Film of index 1 (and of zero thickness) is no film, to 1e-12
- [x] 2.4 Lossless n = 1.5: power conserved to 1e-6 (holds to 1e-15), bare
      and under a 100 nm n = 1.45 film; `d -> 0` equals the exact closed
      form derived in the docs (2.30357446575872, 1.38809456316473)
- [x] 2.5 Gold at 2-5 nm: log-log slope of the non-radiative rate -2.96 /
      -2.99 (bound 0.06 from -3), quasi-static prefactor within 1 %
- [x] 2.6 Doubling points, halving or doubling the detour: < 1e-4 (1e-15);
      a grounded slab's real pole is detour-independent
- [x] 2.7 All 20 mutations applied one at a time go red in the named test
      and the restored module passes 21/21

## 3. Verification

- [x] `OMP_NUM_THREADS=2 python3 -m pytest tests/test_purcell_reference.py
      tests/test_fdtd.py -q` on the rebased branch
- [x] `npx @fission-ai/openspec validate purcell-layered-reference --strict`
- [ ] The FDTD Purcell measurement, once it exists, compared against the CLI
      table (the lead's call; not this change)
