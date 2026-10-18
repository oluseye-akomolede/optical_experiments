# time-domain

## ADDED Requirements

### Requirement: The paper-1 collection arm SHALL enclose the whole scene in its phasor box

`scripts/paper1_nanowire.py --collection-na NA` SHALL run the CW arm with a
phasor box whose six faces enclose the wire, the oxide and the entire gold
pad, each face at least 2 node layers outside the parts, inside the layers
`pml + 1 .. n - 2 - pml`, and on vacuum in the voxelised layout.

#### Scenario: the box on a real layout
- **WHEN** the box is placed on the tiny plan, the paper's default plan and
  run 1's (dx 20, pad 400)
- **THEN** every face is at least 2 layers outside the plan's own part bounds
  and `fdtd.phasor_box_layers` accepts it with the run's absorber

### Requirement: The collection arm SHALL print the fraction and its sanity ratios

The arm SHALL read the box over the CW reading's own tail and print the
fraction of the upward far-field power inside `asin(NA)`, the upward and
total power, the box's surface power over the delivered power, and the
full-sphere far field over the surface power.

#### Scenario: the vacuum scene reads the free dipole
- **WHEN** the tiny scene with every material stripped (37 x 37 x 42 nodes, dx
  20 nm, 900 nm, x dipole, 16 periods read over the last 4) runs with
  `--collection-na 0.8`
- **THEN** the fraction is the free x dipole's 0.49600 within 0.003 (measured
  0.4960), and the surface power is the delivered power and the full-sphere
  far field the surface power within 2.5% (measured 0.9894 and 0.9947)

### Requirement: The mirror SHALL be inside the box, not modelled

The collection arm SHALL NOT use `farfield`'s `pec_mirror_z` image or a box
whose bottom lies above the gold: a free-space transform of a box that leaves
the mirror's currents outside drops the reflection, and a PEC image is an
infinite perfect mirror where the scene's is a finite gold pad.

#### Scenario: over a perfect conductor
- **WHEN** an x dipole lambda/4 above an exact Yee PEC plane (37 x 37 x 31
  nodes, 16 cells per wavelength) is read by a five-face box on the plane
  with the image, and by a closed box in air without it
- **THEN** the first reads the closed-form image dipole's NA 0.8 fraction
  0.7661 within 0.005 (measured 0.7666) and the second the free dipole's
  0.4960 within 0.01 (measured 0.4996)

#### Scenario: over a finite gold slab
- **WHEN** an x dipole 210 nm above a 360 x 360 x 60 nm Rakić gold slab at
  900 nm (37 x 37 x 35 nodes, dx 30 nm) is read by the full enclosure, by the
  same with every face 2 layers further out, by the PEC image on the slab's
  top and by a closed box in air
- **THEN** the two enclosures agree within 0.002 (measured 0.6121 and
  0.6121), while the PEC image reads 0.11 to 0.17 high (measured +0.143) and
  the box in air 0.085 to 0.15 low (measured -0.116)

### Requirement: The collection arm SHALL be off by default and refuse what it cannot read

Without `--collection-na` the script SHALL hand `fdtd.simulate` and
`fdtd.estimate_run` exactly the arguments it did before the arm existed and
print nothing new. The flag SHALL be refused with an NA outside (0, 1], on a
pulse record (`--spectrum`, `--ringdown`, `--profile pulse`), with `--tiny`
other than `--tiny --estimate`, and when the layout leaves no layer for a
face.

#### Scenario: the same run with and without the flag
- **WHEN** the tiny vacuum scene runs through `main` with the solve replaced
  by a canned record, with and without `--collection-na 0.8`
- **THEN** without it no phasor keyword reaches `simulate` or `estimate_run`,
  and with it every line of the plain run is printed unchanged and in order,
  the arm's lines added

#### Scenario: a pad with no room
- **WHEN** the tiny scene's pad is cut to 140 nm beside a 6-cell absorber
- **THEN** the placement is refused naming axis z and its layers, and the
  ladder prints that refusal on its dx-20 rungs

### Requirement: A phasor box SHALL be counted in the memory estimate

`fdtd.memory_estimate` and `fdtd.estimate_run` SHALL take `phasor_box`,
`phasor_frequencies` and `phasor_steps` and add `fdtd.phasor_box_bytes`: the
box's 36 accumulators and two phase tables as `_PhasorBox` allocates them,
plus the larger of a step's transients and the faces' assembly. Without the
keywords the estimate SHALL be unchanged. The paper-1 run path and ladder
SHALL price the box when the arm is on.

#### Scenario: against a real box
- **WHEN** a `_PhasorBox` of half-extent (7, 8, 9) at two frequencies over 40
  steps is built on a 22 x 24 x 26 grid
- **THEN** its accumulators and tables are the estimate's to the byte, the
  estimate rises by `phasor_box_bytes` exactly, and a numpy run's measured
  increment stays inside it (measured 0.50 to 0.83 of it on three boxes)
