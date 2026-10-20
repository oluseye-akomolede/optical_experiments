# time-domain (delta: skin-layer-no-sliver)

## MODIFIED Requirements

### Requirement: A `td` upload SHALL be laid out on its grid by one fixed rule

The worker SHALL turn a `td` job's upload into a grid by this rule, and phase
5's estimate SHALL use the same function, so the run that is estimated is the
run that is solved.

- **Parts.** The upload's named groups, each a closed surface, refused by name
  otherwise; the closed check is per part, never on the merged mesh, so parts
  may share faces. There is no enclosing-wall rule: in `open` space the parts are
  separate objects. Each part takes its entry in the job's `materials` table
  (a refractive index of at least 1, `pec`, or a library metal name); a
  one-part upload with no table takes the job's `n_index`.
- **Frame.** Metres are mesh units × the unit's metres × `scale`. The origin is
  the minimum corner of the parts' combined bounding box. `position` and every
  probe are metres in this frame.
- **Extent.** For `metal` the grid is exactly the bounding box: outside the
  parts is the conductor. For `open` every side is padded by 16 cells of
  `dx_m` (the 3-cell TF/SF clearance, 3 more cells, and the 10-cell CPML), for
  a point source as well as a plane wave.
- **Plane wave.** The TF/SF box is the bounding box grown by the 3-cell
  clearance on every side. A plane wave with `metal` is refused, as in phase 2.
- **Cells.** The coarse cell is `dx_m`. On each axis a part whose
  bounding-box extent is under 4 `dx_m` adds a layer over that extent with
  spacing a quarter of the extent, so it has at least 4 cells across. A metal
  part adds, on each axis, a layer 3 skin depths deep inside each of its two
  bounding-box faces, with spacing a quarter of the skin depth at the top of
  the source band; where both apply the smaller spacing wins. It gets **one
  layer across its whole extent at that spacing** instead when the two face
  layers would meet or cover the part already, or when what they leave between
  them would be narrower than one `dx_m`: a middle strip under a coarse cell is
  bulk metal that the thin-layer rule would cut into cells finer than the skin
  layers themselves, for no gain in resolution. The skin depth is
  `fdtd_grid.skin_depth(metal, f)`
  = c / (2π f κ), κ the imaginary part of √ε(f) of that metal’s model; the worker
  and the estimate both call that function. The top of the band is
  `frequency_hz + bandwidth` for a pulse and `frequency_hz` for `cw` and
  `drawn`.
- **Thin shells.** The layers above are the only thin features the layout
  detects (axis-aligned extents and metal bounding-box faces). A curved thin
  shell is voxelised at the grid it gets, and the estimate counts that grid;
  the result carries the caveat "thin curved features are resolved only at
  the coarse cell" whenever a part is not a box (its volume is under 0.99
  of its bounding box's), since its thin walls may not be refined. The
  thin-layer requirement applies to the detected layers.
- **Refusals.** A probe or point-source position outside the grid, and a
  boundary other than `metal` or `open`, are refused before anything is
  allocated, naming the value. A point source or probe whose nearest node is a
  conductor (the `metal` boundary's outside, a `pec` part, or a library-metal
  part's interior) is refused, naming the position and the part, since its
  source would inject nothing and its trace would read as "no response".
  With `open`, a point source or probe whose nearest node lies in the 10-cell
  CPML at either end of any axis is refused, naming the position and the
  range it must lie in: the layer absorbs a source as it is injected and
  damps what a probe reads. Periodic boundaries stay a phase 6 option.

#### Scenario: the frame is the bounding box's corner in metres
- **WHEN** a one-part box spanning 2..6 mm in x is laid out with units `mm`
  and scale 1
- **THEN** the grid's x nodes run from 0 to 4 mm for `metal`, and from
  -16 `dx_m` to 4 mm + 16 `dx_m` for `open`

#### Scenario: a thin metal plate is resolved at its skin depth
- **WHEN** a gold plate thinner than 6 skin depths lies in `open` space
- **THEN** no cell across the plate is wider than a quarter of gold's skin depth
  at the top of the band, and the plate's faces are grid nodes

#### Scenario: a metal part just over 6 skin depths gets no sliver in its middle
- **WHEN** a metal part is thick enough that two face layers leave less than one
  `dx_m` of bulk between them
- **THEN** the part is laid out with one layer across its whole extent at a
  quarter of the skin depth: no cell across it is finer than half that, it takes
  no more cells than that spacing needs, and its faces are grid nodes

#### Scenario: separate objects need no wall
- **WHEN** an `open` upload holds two closed parts, neither inside the other
- **THEN** it is laid out without the cavity solver's one-wall refusal

#### Scenario: a part thinner than 4 cells gets 4 cells
- **WHEN** a dielectric plate 1.5 `dx_m` thick lies in `open` space
- **THEN** at least 4 cells lie across the plate and no cell across it is wider
  than a quarter of its thickness

#### Scenario: a curved part carries the thin-feature caveat
- **WHEN** a part is not a box (a sphere, say)
- **THEN** the result carries the caveat that thin curved features are resolved
  only at the coarse cell, and a box-only upload carries no such caveat

#### Scenario: a source or probe inside a conductor is refused
- **WHEN** a point source or probe sits on a conductor node (outside the parts
  with `metal`, inside a `pec` part, or inside a library-metal part)
- **THEN** the job is refused before allocation, naming the position and the
  part (or the metal boundary)

#### Scenario: a source or probe inside the CPML is refused
- **WHEN** with `open`, a point source or probe lies on the grid but its
  nearest node is inside the 10-cell CPML
- **THEN** the job is refused before allocation, naming the position and the
  range outside the layer

#### Scenario: a probe off the grid is named
- **WHEN** a probe lies outside the laid-out grid
- **THEN** the job is refused before allocation, naming the probe and the grid's
  extent
