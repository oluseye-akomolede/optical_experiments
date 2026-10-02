# time-domain

## ADDED Requirements

### Requirement: A probe's legend SHALL state its position in the unit of the job's own grid

The legend of each probe on a finished time-domain job's page SHALL name the
probe's field component and its position, printing every coordinate in a length
prefix no finer than the job's own grid step -- the finest gap between the
sample positions the manifest records. A coordinate whose magnitude is below a
millionth of that step SHALL be printed as `0`: the stored position is
`origin_m + node`, so a probe on the frame's own origin arrives as what float64
leaves of two grid-sized numbers, and that residue must not choose the unit.
Where the manifest records no sample positions, the position SHALL still be
printed, under the previous rule (the largest coordinate picks the prefix), as
there is no grid to read a step from.

Why: prod job `aa6f083b` stored `[4.0e-20, -3.2e-19, 1.0e-19] m` for a probe on
a 20 nm grid's origin -- a cell-limit residue, ~1e-10 of a cell -- and the old
rule printed it in the prefix the residue needed: "0.0000400 fm", twelve orders
finer than the cell. The snap line sits four orders above the residue and six
below a real node.

#### Scenario: a probe on the frame's own origin

- **WHEN** a finished td job records a probe whose node is the frame origin, so
  its stored position is the remainder of two grid-sized numbers
- **THEN** each such coordinate is printed as `0`, and the others in the grid's
  own unit

#### Scenario: a probe one cell away

- **WHEN** the same job records a probe one grid step from that origin
- **THEN** the coordinate is printed in the grid's unit and is unchanged by the
  snapping rule

#### Scenario: a manifest with no sample positions

- **WHEN** the manifest records no sample positions (a job with no movie)
- **THEN** the legend still states the probe's position, with the prefix the
  largest coordinate picks

### Requirement: The render check SHALL verify the legend's stated position

`tdcheck.mjs` SHALL read the position back out of each legend line through the
page's own length-prefix table, and SHALL require every axis to be the
manifest's own probe node within one grid step (or, where the manifest records
no sample positions, within the three significant digits the legend prints). It
SHALL additionally require a coordinate the manifest puts below a millionth of
the grid step to be printed as `0`, and SHALL state on its output that it is
not checking that rule when it has no step to check it against.

Why: the check ran ALL PASS against a mirror of `aa6f083b` while the page
printed the residue in femtometres -- it verified the component and the swatch
colour, never the number after `at`.

#### Scenario: the production job's origin probe

- **WHEN** the check runs against a mirror of `aa6f083b`
- **THEN** it reads the legend's position as the manifest's node at the frame
  origin and accepts it, and against a page that printed the residue in
  femtometres the same check FAILs, naming the axis
