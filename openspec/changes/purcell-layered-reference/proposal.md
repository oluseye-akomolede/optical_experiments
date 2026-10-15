# purcell-layered-reference

**New reference module, no solver change.** The FDTD solver is getting a
Purcell-factor measurement. To validate it independently this change adds an
exact, grid-free reference: the normalised decay rate `F` of an electric
dipole at height `d` above air / SiO2 film / semi-infinite gold, perpendicular
and parallel, against wavelength, from the Sommerfeld-integral form of the
reflected field (Novotny & Hecht ch. 10; Chance, Prock & Silbey 1978), with
the power split into radiated (air far field), transmitted (lossless
substrate far field) and non-radiative. Files added:
`dockerfiles/maxwell_solver/purcell_reference.py`,
`tests/test_purcell_reference.py`, `docs/purcell-reference.md`. No existing
module changes; `fdtd.permittivity` and `fdtd.RAKIC_1998` are imported, not
retyped.

## Why

A Purcell measurement in FDTD is sensitive to how the dipole is sampled, how
the metal is updated and how the emitted power is normalised. Each of those
can be wrong by tens of percent and still look like a plausible curve. The
layered-medium dipole is one of the few emitter problems with an exact
answer, so a reference for exactly the stack the solver will be run on (a
dipole 6 nm above 12 nm of SiO2 on gold, 850-950 nm) turns the validation
into a number with a bound.

## What changes

- A new capability `purcell-reference` with one API,
  `purcell_layered(wavelength_m, height_m, film_thickness_m, film_index,
  metal, orientation) -> PurcellResult(F, radiated, transmitted,
  nonradiative)`, and a CLI that prints the 850-950 nm sweep.
- The in-plane integral runs along a path deformed into the lower
  half-plane, under the surface-plasmon pole and the `s = 1` branch point;
  the real-axis tail is followed until `exp(-2 k d sqrt(s^2 - 1))` is
  `exp(-50)`.
- Six checks, each with a closed-form or conservation control and a named
  mutation that turns it red: the image dipole over a perfect conductor;
  the large-height asymptote; a film of index 1 is no film; a lossless
  n = 1.5 half-space conserves power and reproduces an exact `d -> 0`
  closed form; gold at 2-5 nm quenches as `1/d^3` with the quasi-static
  prefactor; quadrature and path knobs move `F` by under 1e-4.

## Impact

- Nothing live changes: no route, no job, no solver code path. The module is
  imported by its tests and the CLI only.
- `tests/`: 21 new tier-1 tests, 2.5 s in total, 2 threads.
