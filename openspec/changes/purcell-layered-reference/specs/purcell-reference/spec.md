# purcell-reference

## ADDED Requirements

### Requirement: The reference SHALL give the exact Purcell factor of a dipole above a layered substrate

`purcell_layered(wavelength_m, height_m, film_thickness_m, film_index,
metal, orientation)` in `dockerfiles/maxwell_solver/purcell_reference.py`
SHALL return `F = P / P0` of an electric dipole in air at `height_m` above a
lossless film on a semi-infinite substrate, `orientation` `"perp"` (along
the normal) or `"par"`, from the Sommerfeld integral of the reflected field
(Novotny & Hecht ch. 10), `P0` the free-space dipole power.

#### Scenario: the substrate is named three ways
- **WHEN** `metal` is a Rakić 1998 library name, `"PEC"` or a number
- **THEN** the substrate permittivity is `fdtd.permittivity(metal, c /
  wavelength_m)` (never retyped), a perfect conductor, or that constant;
  any other name raises `fdtd.FDTDError`

#### Scenario: a perfect conductor is the image dipole
- **WHEN** `metal="PEC"`, no film, `x = 2 k d` anywhere in 0.1 to 20
- **THEN** `F` equals `1 + 3 (sin x / x^3 - cos x / x^2)` for the
  perpendicular dipole and `1 + (3/2)(sin x / x^3 - cos x / x^2 - sin x / x)`
  for the parallel one to 1e-4 (measured: 2.4e-14)

#### Scenario: gold quenches as 1 / d^3
- **WHEN** the substrate is `"Au"` at 900 nm, no film, `d` from 2 to 5 nm
- **THEN** the log-log slope of the non-radiative rate against `d` is -3
  within 0.06, and at 2 nm that rate is within 1 % of the quasi-static
  `(3/8) Im[(eps - 1)/(eps + 1)] / (k d)^3` (perpendicular; half that
  parallel), `eps` being `fdtd.permittivity("Au", c / 900 nm)`

#### Scenario: a film of index 1 is no film
- **WHEN** `film_index = 1` and `film_thickness_m = t`
- **THEN** every field of the result equals the bare-substrate result at
  height `height_m + t`, to 1e-12

#### Scenario: a dipole far away is unperturbed
- **WHEN** the substrate is `"Au"` and `x = 2 k d` is between 30 and 200
- **THEN** `F - 1` follows `-3 Re[r_p(0) e^{ix}] / x^2` (perpendicular)
  and `(3/2) Im[r_s(0) e^{ix}] / x` (parallel) to within `6 / x^3` and
  `3 / x^2` respectively, `r(0)` the normal-incidence Fresnel coefficient

### Requirement: The result SHALL split the power into radiated, transmitted and non-radiative parts

The result SHALL carry `radiated` (the far-field power into the air),
`transmitted` (the far-field power into a lossless substrate, including the
band that is evanescent in air; 0 for a metal or PEC) and `nonradiative`
(`F - radiated - transmitted`: absorbed, or guided), each in units of `P0`.

#### Scenario: a lossless substrate conserves power
- **WHEN** the substrate is a constant permittivity 2.25 (n = 1.5), bare or
  under 100 nm of n = 1.45 film, at any height from `lambda / 10^4` to
  `2 lambda`
- **THEN** `nonradiative` is 0 to 1e-6 (measured: 1e-15)
- **AND** at `d = lambda / 10^8` `F` is 2.30357446575872 (perpendicular)
  and 1.38809456316473 (parallel) to 1e-6, the exact `d -> 0` closed forms
  of `docs/purcell-reference.md`

#### Scenario: a perfect conductor radiates everything
- **WHEN** `metal="PEC"`, no film, `k d` of 0.1, 1 or 7
- **THEN** `transmitted` is 0 and `nonradiative` is 0 to 1e-9

### Requirement: The in-plane integral SHALL be taken on a path that avoids the surface-plasmon pole

The integral over the in-plane wavenumber SHALL run along a path deformed
into the lower half of the complex plane, under the surface-plasmon pole
and the `s = 1` branch point, then along the real axis until the evanescent
factor has decayed to `exp(-50)`; `F` SHALL NOT depend on the path or on
the number of quadrature points.

#### Scenario: the quadrature and the path are converged
- **WHEN** the Gauss points per panel are doubled, or the detour depth is
  halved or doubled, for 6 nm above 12 nm of n = 1.45 on Au at 900 nm and
  for a mirror at `k d = 5`
- **THEN** `F` changes by less than 1e-4 (measured: 3e-15)

#### Scenario: a real guided-mode pole is passed below
- **WHEN** the stack is 100 nm of n = 1.45 on a perfect conductor (a
  grounded slab, whose TM0 mode has a real pole in `1 < s < 1.45`)
- **THEN** `F` is the same for detour depths 0.25, 0.5 and 1.0 to 1e-9, and
  the guided power is reported as `nonradiative`

### Requirement: A CLI SHALL print the validation sweep

`python dockerfiles/maxwell_solver/purcell_reference.py` with no arguments
SHALL print, for a dipole 6 nm above 12 nm of n = 1.45 on Au, eleven rows
from 850 to 950 nm with `F` and the radiated share for both orientations;
`--height-nm`, `--film-nm`, `--film-index`, `--metal`, `--start-nm`,
`--stop-nm` and `--steps` SHALL change the sweep. It SHALL run with
`OMP_NUM_THREADS=2` and finish in under a second.

#### Scenario: the default sweep
- **WHEN** the CLI runs with no arguments
- **THEN** it prints a header row and eleven data rows, 850.0 to 950.0, and
  the 900 nm perpendicular `F` is the API's value (5.9321) to 1e-4
