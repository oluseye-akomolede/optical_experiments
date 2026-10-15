# far-field

## ADDED Requirements

### Requirement: A box of tangential fields SHALL transform to the far field by Love's equivalence in free space

`farfield.far_field(faces, frequency, theta, phi)` SHALL return the far-field
amplitudes `E_theta`, `E_phi` (with `exp(ikr)/r` stripped, V) and the
radiation intensity `U = |E|^2 / (2 eta0)` (W/sr) on the `(theta, phi)` grid
from `J = n x H`, `M = -n x E` radiating in free space (k = 2 pi f / c,
`exp(-i w t)`), by the midpoint rule over each face's cells
(`dockerfiles/maxwell_solver/farfield.py`).

#### Scenario: a z-dipole in vacuum radiates sin^2(theta)
- **WHEN** the exact field of a Hertzian dipole along z (Jackson eq. 9.18) is
  sampled on a cube 1 um on a side, 16 cells per axis, at 1 um wavelength
- **THEN** `U(theta, phi)` equals `c^2 eta0 k^4 |p|^2 sin^2(theta) / (32 pi^2)`
  to within 1% of its peak (measured 0.41%)

#### Scenario: an x-dipole radiates 1 - sin^2(theta) cos^2(phi)
- **WHEN** the same cube holds a dipole along x
- **THEN** `U / max U` equals `1 - sin^2(theta) cos^2(phi)` within 0.01
  (measured 0.0034)

#### Scenario: the frame is global, not the box's
- **WHEN** the z-dipole sits at (0.1, -0.15, 0.2) um in a 1.1 um box with
  corner (-0.4, -0.6, -0.3) um and 18 x 16 x 20 cells
- **THEN** its pattern matches the centred cube's within 0.01 (measured
  0.0043) and its `E_theta` equals the centred one times `exp(-i k r^.d)`
  within 1% of peak (measured 0.31%; the conjugate phase is 191% off)

### Requirement: The surface SHALL be given in one fixed face layout

A `Face` SHALL carry `normal` (one of `fdtd.DIRECTIONS`, outward),
`position` (metres, along that axis), `u_edges` (n_u + 1) and `v_edges`
(n_v + 1) cell edges in metres along the two non-normal axes in increasing
axis order, and colocated global Cartesian `E`, `H` of shape `(3, n_u, n_v)`
at the cell centres. A face that does not meet this SHALL be refused with
`FarFieldError` naming the face and the shape it needed.

#### Scenario: a wrong layout is refused
- **WHEN** a `Face` is built with `E` of shape `(3, 3, 4)` on a 4 x 4 cell face
- **THEN** `FarFieldError` says `face '+z': E must have shape (3, 4, 4) ...`

#### Scenario: the in-plane axes follow the documented order
- **WHEN** a "-y" face at y = 5 has `u_edges` (0, 1, 3) and `v_edges` (10, 12)
- **THEN** its two samples sit at x = 0.5 and 2.0, y = 5, z = 11, with areas
  2 and 4, and its unit normal is (0, -1, 0)

### Requirement: The power through the surface SHALL be the time-averaged Poynting flux

`farfield.surface_power(faces)` SHALL return `1/2 Re int (E x H*).n dS` in W
over the given faces, each cell weighted by its own area.

#### Scenario: Jackson's dipole power
- **WHEN** the z-dipole cube above is integrated
- **THEN** the flux equals `c^2 eta0 k^4 |p|^2 / (12 pi)` within 1%
  (measured 0.096%), and `int U dOmega` over the full sphere equals the flux
  within 1% (measured 0.096%)

#### Scenario: graded cells carry their own areas
- **WHEN** one in-plane axis of every face is cosine-graded (16 unequal cells)
- **THEN** the flux equals the uniform cube's within 0.2%

### Requirement: The collection fraction SHALL be the cone's share of the upward far-field power

`farfield.collection_fraction(faces, frequency, NA)` SHALL return
`int_{theta <= asin NA} U dOmega / int_{theta <= pi/2} U dOmega`, NA in air,
with Gauss-Legendre in cos(theta) (32 nodes) and 64 uniform phi nodes by
default. NA outside (0, 1] SHALL be refused with a message naming the range.

#### Scenario: closed-form cone fractions
- **WHEN** NA is 0.8 (theta_max 53.13 deg)
- **THEN** the z-dipole cube reads `1 - (3 cos - cos^3)/2 = 0.208` within 1%
  (measured 0.2076) and the x-dipole cube reads 0.496 within 1% (measured
  0.4961); NA 1 reads exactly 1

#### Scenario: halving the spacing changes the answer by less than the tolerance
- **WHEN** the z-dipole cube goes from 8 to 16 cells per axis
- **THEN** the pattern moves by less than 0.02 of peak (measured 0.010) and the
  NA 0.8 cone power by less than 0.5% (measured 0.021%), both towards the
  closed form (pattern error 0.014 -> 0.0034; power error 2.4e-4 -> 2.1e-5)

### Requirement: A surface open at the bottom SHALL be accepted, with its approximation stated

A list without the "-z" face SHALL transform like any other (top and four
sides only). The module docstring SHALL state that the free-space transform
then contains the direct wave and the reflection from the mirror patch inside
the box only, never the reflection from the mirror outside it, and SHALL
quote the measured error.

#### Scenario: a vertical dipole 0.3 wavelengths over a perfect mirror, no image
- **WHEN** the field of the dipole plus its same-sign image is sampled on the
  top and sides of a 1 x 1 x 1 um box whose bottom lies on z = 0 and
  transformed without `pec_mirror_z`
- **THEN** the upward far-field power is 0.55-0.60 of the surface flux
  (measured 0.573) and the pattern is 0.6-0.7 of peak off
  `sin^2(theta) cos^2(k h cos theta)` (measured 0.67): the stated
  approximation, pinned so the docstring cannot drift from the code

### Requirement: A perfect-conductor mirror SHALL be handled by image currents

With `pec_mirror_z=z_m` every included sample SHALL also radiate through its
image in the plane `z = z_m` (J: horizontal components negated; M: vertical
component negated), which is exact for a perfect conductor. A sample below
the plane SHALL be refused with `FarFieldError`.

#### Scenario: a vertical dipole 0.3 wavelengths over a perfect mirror
- **WHEN** the same open-bottom box is transformed with `pec_mirror_z=0`
- **THEN** the pattern equals `sin^2(theta) cos^2(k h cos theta)` within 0.01
  (measured 0.0065) and the upward far-field power equals the surface flux
  within 0.5% (measured 0.047%)
- **AND** adding the "-z" face on z = 0 changes the far field by less than
  1e-9 of peak (it cancels its own image)

#### Scenario: a sample below the mirror plane is refused
- **WHEN** `pec_mirror_z` is 0.5 um and the box starts at z = 0
- **THEN** `FarFieldError` says every sample must lie on or above the mirror plane
