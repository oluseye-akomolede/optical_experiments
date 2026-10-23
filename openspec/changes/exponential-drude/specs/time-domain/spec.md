## ADDED Requirements

### Requirement: The Drude pole SHALL be stepped as an exact decay

A metal model's `w0 = 0` pole is a free current obeying `J' = -G J + sigma E`,
with `sigma = eps0 f wp^2`. It SHALL be advanced by that equation's exact
solution over one time step, `J^{n+1} = exp(-G dt) J^n + sigma (1 -
exp(-G dt))/G E^n`, and its contribution to the E update SHALL be
`dt J^{n+1}`. It SHALL NOT be central-differenced: that form's characteristic
roots are 1 and `(1 - G dt/2)/(1 + G dt/2)`, so its static mode sits at exactly
1 at every finite dt -- neutral in the homogeneous block the growth check
models, and therefore reportable only as 1 to within rounding. The Lorentz
poles (`w0 > 0`) keep their central-difference recursion, and the
frequency-domain permittivity the fits are judged by is unchanged.

#### Scenario: the static mode decays at every step
- **WHEN** any library metal's Drude pole is stepped at 1e-19, 1e-18, 4e-18,
  8.2e-18 or 2e-17 s
- **THEN** its per-step factor equals `exp(-G dt)`, is strictly inside the unit
  circle, and falls as the step grows

#### Scenario: the growth check certifies the runs' own steps
- **WHEN** the growth check runs for Au-palik at Run I's dt and finest
  spacings, or for Au-jc at Run K's
- **THEN** the spectral radius is below `1 + GROWTH_TOLERANCE`
- **AND** for Au-jc it is strictly below 1, where the central-differenced form
  read `1 + 1.58e-8`
