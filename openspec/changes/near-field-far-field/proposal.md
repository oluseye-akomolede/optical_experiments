# near-field-far-field

## Why

Paper 1 (arXiv 2601.04440: a quantum dot in an InP nanowire on 12 nm SiO2
over a gold mirror, emitting near 900 nm) reports that 74% of the emitted
light is collected within NA 0.8. To reproduce that number the solver needs a
post-processing step it does not have: from the tangential E and H on a closed
box around the emitter at one frequency, the far-field pattern in the upper
half-space, the fraction of the upward power inside a cone of given NA, and
the total radiated power. Nothing in `openspec/specs/` covers radiation into
free space; every requirement there is about a closed cavity or a grid.

## What Changes

- A new capability `far-field` with one new module,
  `dockerfiles/maxwell_solver/farfield.py`: Love's surface equivalence in free
  space (`J = n x H`, `M = -n x E`, Balanis *Antenna Theory* 3rd ed. §12.3),
  the midpoint rule over rectangular face cells, the radiation intensity
  `U(theta, phi)` in W/sr, Gauss-Legendre cone integrals for the collection
  fraction, and the surface Poynting flux for the total power.
- The input layout a field monitor has to produce is fixed by the spec
  (`Face`: outward normal, position, in-plane cell edges in increasing axis
  order, colocated Cartesian E and H at cell centres).
- A surface open at the bottom (top and four sides only) is accepted for the
  mirror case, with its approximation stated and measured; a `pec_mirror_z`
  option adds the perfect-conductor images and is exact for that mirror.
- Reference maths for the tests: the exact Hertzian-dipole field (Jackson
  eq. 9.18) and power (eq. 9.24).

Nothing is wired into `fdtd.py` or any job; no existing module changes.
Constants (`C0`, `EPS0`, `ETA0`, `DIRECTIONS`) are imported from `fdtd`.

## Impact

- New: `dockerfiles/maxwell_solver/farfield.py`, `tests/test_farfield.py`,
  eleven reversion cases in `tests/check_discriminating.py`.
- Specs: new `far-field` capability (ADDED requirements only).
- No route, payload, schema, image, manifest or resource-bound change. The
  tests are pure numpy, a 16^3-cell box, under two seconds at
  `OMP_NUM_THREADS=2`.
