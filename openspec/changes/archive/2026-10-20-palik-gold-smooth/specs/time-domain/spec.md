# time-domain (delta: palik-gold-smooth)

## MODIFIED Requirements

### Requirement: A second gold model SHALL fit Palik's table

The metal library SHALL offer `Au-palik`, a Lorentz-Drude `Metal` fitted to
E. D. Palik's handbook gold table over the table's rows where the model may be
asked to answer, so a validation run can be compared against the paper's own
source. The model SHALL be smooth as well as accurate: it is fitted under a
damping floor and judged on a dense grid between the table's rows, because a
pole that fits every row can still ring between two of them and fake a high-Q
mode in a ring-down. The fit's range SHALL be the rows it is fitted to
(1.10-1.90 eV, 653-1127 nm). The table's three rows past 1150 nm (1239.8,
1265.1 and 1291.5 nm) are not fitted: under the damping floor and the
smoothness envelope the model misses them by 4.12%, 19.18% and 20.37% against
the 3% gate, and their k falls then rises (8.77, 8.09, 8.25), so the fitted
range ends below them and a band that reaches them is refused. The dense-grid
criterion, not row accuracy, is what they are held to. The Rakić library and
its `Au` entry SHALL be unchanged, and every place that accepts a library
metal (the `_metal` lookup, `METALS`, the layout and voxel material checks,
and the skin-depth rule) SHALL accept the new name.

#### Scenario: the fit reproduces the table

- **WHEN** the model's complex permittivity is compared with every copied table
  point in the model's fitted range
- **THEN** every point is within 3% relative error, and the test states the
  measured maximum, where it falls, and the fitted range

#### Scenario: the model is smooth between the table's rows

- **WHEN** the model is evaluated every 2 nm over 600-1400 nm
- **THEN** Re eps is negative and Im eps positive at every point, and between
  the table's own rows the model's |eps| is within 1.5x of the linear
  interpolation between the neighbouring rows
- **AND** no pole has a damping below 0.05 eV, so nothing in the model can ring
  between two table rows

#### Scenario: the model is lossy, positive and stable

- **WHEN** the model is evaluated across the fitted band and its ADE term is
  checked at the bench's spacings
- **THEN** Im eps is positive at every point, every pole has f > 0 and
  Gamma > 0, and the growth factor is below 1 + `GROWTH_TOLERANCE`

#### Scenario: the Rakić library is untouched

- **WHEN** the library is listed
- **THEN** `METALS` contains both `Au` and `Au-palik`, and the published Rakić
  check still passes against `RAKIC_1998` with the same values

#### Scenario: a band outside the fit is refused naming its source

- **WHEN** a pulse band leaves `Au-palik`'s fitted range
- **THEN** the refusal names Palik's table and the range, as the Rakić
  refusals name Rakić

## ADDED Requirements

### Requirement: The run path SHALL check a pulse's band against its metals

`fdtd.source_band` SHALL be the single definition of a pulse's band: the
spectrum's width above `BAND_FLOOR` of its peak, `cw` and `drawn` having none,
and a band that reaches zero frequency being refused. The web form and the
worker SHALL each call `fdtd.check_scene_band` on the library metals the scene
names (parts and coatings), so a band outside a metal's fitted range is
refused where a run is accepted, not only where the bench runs. The paper-1
script, whose run is the one the refit protects, SHALL judge its mirror through
the same `fdtd.source_band` call and the same `check_metal_band` refusal, so
the same pulse is accepted or refused with the same numbers and the same
sentence as in the app; its `cw` and `drawn` sources ask nothing, as the app's
do. `pec` and refractive indices are not fitted models and SHALL NOT be
checked.

#### Scenario: the app refuses a band its metal cannot answer for

- **WHEN** an upload names a library metal and a pulse whose band leaves that
  metal's fitted range
- **THEN** the request is refused before it is queued, and the refusal names
  the metal, its source table and its range

#### Scenario: the worker refuses the same job

- **WHEN** a job reaches the worker with a band outside a named metal's fit,
  without having passed the form
- **THEN** the run is refused as a `TDSolveError` in the metal's own words

#### Scenario: a source without a declared band asks nothing

- **WHEN** the source is `cw` or `drawn`
- **THEN** the guard passes any named metal

#### Scenario: the paper-1 run refuses a band its gold cannot answer for

- **WHEN** the paper-1 script is given a pulse band and a gold model whose fit
  does not cover it
- **THEN** the scene is refused before it is built, with the same sentence the
  app gives for that pulse, and a `cw` source or a `--vacuum` control is not
  refused. At 900 nm the Palik fit accepts a pulse up to 83 nm of nominal
  half-width (band floor 1.103 eV) and refuses 84 nm (1.099 eV); Run C's
  `--band-half 100` (1.045-1.710 eV) is therefore refused under this guard,
  and whether the run narrows its pulse or the fit widens is not this
  change's call.

#### Scenario: the bench and the app share one band

- **WHEN** the bench computes the band it judges a metal against
- **THEN** it calls `fdtd.source_band`, the same function the app's guard uses,
  rather than spelling the formula out

### Requirement: Reports SHALL name the model that ran

Labels and reports that describe a dispersive metal SHALL name the library or
the selected model's source, never one specific table for every metal.

#### Scenario: the bench names its model

- **WHEN** the bench runs with any library metal
- **THEN** its header names that metal and the table its model was fitted to
