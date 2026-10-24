## ADDED Requirements

### Requirement: The Drude pole SHALL be stepped as an exact decay

A metal model's `w0 = 0` pole is a free current obeying `J' = -G J + sigma E`,
with `sigma = eps0 f wp^2`. It SHALL be advanced by that equation's exact
solution over one time step, `J^{n+1} = exp(-G dt) J^n + sigma (1 -
exp(-G dt))/G E^n`, and its contribution to the E update SHALL be
`dt J^{n+1}`. It SHALL NOT be central-differenced: that form's characteristic
roots are 1 and `(1 - G dt/2)/(1 + G dt/2)`, so its static mode sits at exactly
1 at every finite dt -- a current mode that never decays, where the exact
solution decays as `exp(-G dt)`. The Lorentz poles (`w0 > 0`) keep their
central-difference recursion, and the frequency-domain permittivity the fits
are judged by is unchanged.

The growth check SHALL subtract each pole's contribution to the E update once
per step, `(d^{n+1} - d^n)`, matching what `advance` subtracts; it SHALL NOT
difference that contribution a second time.

#### Scenario: the static mode decays at every step
- **WHEN** any library metal's Drude pole is stepped at 1e-19, 1e-18, 4e-18,
  8.2e-18 or 2e-17 s
- **THEN** its per-step factor equals `exp(-G dt)`, is strictly inside the unit
  circle, and falls as the step grows

#### Scenario: the growth check certifies the runs' own steps
- **WHEN** the growth check runs for Rakic gold, Au-jc or Au-palik at Run I's
  dt and finest spacings, and again at Run K's
- **THEN** the spectral radius is at most `1 + GROWTH_TOLERANCE`
- **AND** a check that second-differences the Drude contribution reads 1.0379
  and 1.0634 there, and is refused
