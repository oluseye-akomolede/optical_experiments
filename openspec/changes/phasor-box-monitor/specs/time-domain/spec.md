# time-domain

## ADDED Requirements

### Requirement: A phasor box SHALL record the closed surface the far-field transform reads

`fdtd.simulate(..., phasor_box=(centre, half_extent), phasor_frequencies=f)`
SHALL return `"phasor_faces"`: one list per frequency, in the order given, of
six `farfield.Face` in `fdtd.DIRECTIONS` order. The faces SHALL be the node
layers `centre +/- half_extent`, with positions and cell edges in metres from
node 0, and `E`, `H` `(3, n_u, n_v)` SHALL be colocated at the face-cell
centres (`dockerfiles/maxwell_solver/fdtd.py`).

#### Scenario: a dipole in vacuum, its power
- **WHEN** a CW z-dipole at the centre of a 37^3 vacuum grid (dx 10 nm, CPML
  6, 16 cells per wavelength) is monitored on a box of half-extent 8 cells
  over the last 8 periods with a Hann window
- **THEN** `farfield.surface_power` of its faces equals `fdtd.delivered_power`
  of the same run within 1% (measured 0.9984) and the flux box's
  `radiated_power` within 4% (measured 0.972 of it)

#### Scenario: a dipole in vacuum, its far field
- **WHEN** the same faces are transformed
- **THEN** `U / max U` equals `sin^2(theta)` within 0.02 over the sphere
  (measured 0.0081), and `int U dOmega` equals the surface power within 1.5%
  (measured 0.9949)

### Requirement: The phasors SHALL be a windowed DFT at each family's own sample time

Each phasor SHALL be `2 sum_n w_n F_n exp(+i 2 pi f t_n) / sum_n w_n` over the
steps from `phasor_start` on, `w` the `phasor_window` (`"hann"` or `None`),
with `t_n = (n + 1) dt` for E and `(n + 1/2) dt` for H. The sums SHALL stay
on the device and cross to the host once, after the run.

#### Scenario: E and H are one dipole's
- **WHEN** Jackson's dipole field is fitted to the vacuum run's face samples
  with one complex moment for E and one for H
- **THEN** the residuals are under 3% (measured 0.94% and 1.11%), the two
  moments differ in phase by under 1.5 degrees (measured 0.34; with H stamped
  at E's time it is 6.09) and the E moment is the drive's own within 3%
  (measured 1.0078)

#### Scenario: the device path
- **WHEN** the run goes through a device array that refuses every implicit
  host conversion
- **THEN** it completes and its faces equal the numpy run's exactly

### Requirement: Without a phasor box a run SHALL be unchanged

Without `phasor_box`, `"phasor_faces"` SHALL be `None` and every other record
SHALL be what it was before the option existed.

#### Scenario: the same run without the option
- **WHEN** a run is repeated with and without a phasor box
- **THEN** its traces and flux record are bit-identical and `"phasor_faces"`
  is `None` without it

### Requirement: A phasor box SHALL be refused where its surface would not measure the field

The box SHALL be refused with `FDTDError` (`fdtd.phasor_box_layers`) when a
face lies outside the live node layers `1..n-2`, when, on an axis with a CPML
`pml_cells` deep, a face lies outside `pml_cells + 1 .. n - 2 - pml_cells`
(each face reads H half a cell outside itself), on a paired Bloch run, when
only one of `phasor_box` and `phasor_frequencies` is given, when a frequency
is not positive and below `1 / (2 dt)`, and when `phasor_start` is not a step
of the run.

#### Scenario: a box touching the absorber
- **WHEN** a 37-node open axis has a 6-cell CPML and the box's faces are node
  layers 6..30, which a flux box accepts
- **THEN** `FDTDError` names the absorbing layer and the allowed layers 7..29

#### Scenario: a box off the grid
- **WHEN** the box's faces reach node layer 38 on a 37-node axis
- **THEN** `FDTDError` says a face must be inside the live layers 1..35

#### Scenario: a paired Bloch run
- **WHEN** a periodic cell runs at k L = 2 pi / 3 with a phasor box
- **THEN** `FDTDError` says to drop phasor_box or the Bloch phase, while
  k = 0 and k L = pi accept the box
