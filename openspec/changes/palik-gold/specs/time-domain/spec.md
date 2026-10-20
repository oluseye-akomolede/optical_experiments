## ADDED Requirements

### Requirement: A second gold model SHALL fit Palik's table

The metal library SHALL offer `Au-palik`, a Lorentz-Drude `Metal` fitted to
E. D. Palik's handbook gold table over the table's 600-1300 nm points, so a
validation run can be compared against the paper's own source. The Rakić
library and its `Au` entry SHALL be unchanged, and every place that accepts a
library metal (the `_metal` lookup, `METALS`, the layout and voxel material
checks, and the skin-depth rule) SHALL accept the new name.

#### Scenario: the fit reproduces the table

- **WHEN** the model's complex permittivity is compared with every copied table
  point in 600-1300 nm
- **THEN** every point is within 3% relative error, and the test states the
  measured maximum

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

### Requirement: The bench SHALL choose the gold model by name

`scripts/paper1_nanowire.py` SHALL take `--gold-data {rakic,palik}`, defaulting
to `rakic` so earlier runs reproduce, SHALL use the choice wherever gold is
selected (the scene's mirror material, the reported skin depth, and the
estimate), and SHALL log which model the run used.

#### Scenario: the flag selects the material

- **WHEN** the script is parsed with and without `--gold-data palik`
- **THEN** the scene's mirror material is `Au-palik` and `Au` respectively, the
  estimate prices the same material, and the report names the model it ran

## MODIFIED Requirements

### Requirement: Dispersive metals SHALL follow their published model

Gold, silver, aluminium, copper and chromium SHALL be modelled by the
Lorentz-Drude fits of Rakić et al., Applied Optics 37, 5271 (1998), with
ε∞ = 1 and the paper's parameters in eV (ω_p, f_j, Γ_j, ω_j), stepped with
auxiliary differential equations. Gold SHALL also be available as the second
model fitted to Palik's 1985 table (see "A second gold model SHALL fit Palik's
table"), chosen by name, and
the Rakić fits stay the default so an unaltered run models gold as it did
before. ADE storage SHALL cover only the metal cells'
bounding box. An E sample on a metal face SHALL take the metal. A run whose
pulse band leaves the fitted range of the model it selected for a metal in the
scene SHALL be refused before it is queued, the refusal naming that model's
source.

#### Scenario: the stepped metal has the model's susceptibility
- **WHEN** a metal's term is driven by a sinusoidal field at 20 time steps
  per radian (Au at 1 and 3 eV, Ag and Cr at 2 eV)
- **THEN** the polarisation it builds matches the model's χ(ω) within 1.5%

#### Scenario: a face sample takes the metal and eps_inf = 1
- **WHEN** a single metal node sits next to a dielectric
- **THEN** each E component takes the metal at exactly the two samples whose
  segment touches the node
- **AND** those samples step with ε∞ = 1 whatever the neighbouring permittivity

#### Scenario: a metal near a plane wave's box is refused
- **WHEN** a metal region lies within 3 cells of a plane wave's box faces
- **THEN** the run is refused with a message naming a dispersive metal and
  its node

#### Scenario: the material table matches the paper
- **WHEN** ε(ω) is computed from the library at 12 wavelengths per metal,
  spread through the published table from its shortest to its longest
- **THEN** n + ik matches an independently published Rakić Lorentz-Drude n, k
  dataset (cited in the test) within 1e-3 (relative) at every one, tight
  enough that a 10% typo in a single model parameter fails

#### Scenario: benchmark V4
- **WHEN** a pulse reflects at normal incidence from a gold slab at least 10
  skin depths thick, with at least 4 cells per skin depth, on an `open` travel
  axis with periodic transverse faces (so the slab is infinite, as the Fresnel
  result assumes)
- **THEN** the reflectance, measured as |scattered/incident|² at a
  scattered-field probe, is within 0.02 (absolute) of the analytic Fresnel
  reflectance for the same model, over the band where the incident spectrum is
  at least 1e-2 of its peak

#### Scenario: the V4 bench judges only what it measured
- **WHEN** the V4 bench runs
- **THEN** it takes the incident spectrum from a reference run with the same
  metal term over an empty region (so both runs share one time step), and
  transforms both traces without a window
- **AND** it fails if either trace still holds more than 1e-3 of its peak in
  its last 5% (a cut-off record), or if the ratio is not finite in the band
- **AND** it exits 2 without running when the pulse band leaves the metal's
  fit, and exits 2 rather than passing when the setup has fewer than 4 cells
  per skin depth or a slab under 10 skin depths
- **AND** it fails unless the judged bins span the band it checked the metal
  and skin depths over, each end within one bin and with no hole, since a
  shorter record gives fewer, wider bins over the same band
- **AND** it runs with the solver's default 10-cell CPML

#### Scenario: a metal slab across periodic faces is uniform
- **WHEN** a plane wave meets a metal slab that spans every periodic axis
- **THEN** E inside the slab is the same at every transverse sample within
  1e-6 of its peak (the metal term advances every sample the E update does,
  including the last row of a periodic axis)

### Requirement: The upload's parts SHALL be voxelised onto the grid

Each sample of a node's dual cell SHALL take the material of the innermost
uploaded part containing it. Outside every part the grid SHALL be a conductor
for `metal` and vacuum for `open`. Every part SHALL be given a material: a
refractive index of at least 1, `pec`, or one of the library’s dispersive metals (`fdtd.METALS`).

#### Scenario: nested parts in any file order
- **WHEN** an upload holds a part nested inside another, in either file order
- **THEN** every sample inside the inner part takes the inner part's material,
  matching the cavity tool's winding-number test point for point

#### Scenario: a sample on a mesh edge's line
- **WHEN** an axis-aligned part puts its edges on the grid's sample lines
- **THEN** those samples are classified as the winding-number test classifies
  them (queries are shifted ~1e-6 of the scene size so no ray meets an edge)

#### Scenario: outside the shape
- **WHEN** a `metal` run is voxelised
- **THEN** every node whose cell lies outside the parts is in the conductor mask;
  for an `open` run those nodes are vacuum (permittivity 1)

#### Scenario: a materials table that does not fit the upload
- **WHEN** the table leaves a part out, names a part the file lacks, gives an
  unknown material name, or gives an index below 1 or not finite
- **THEN** the run is refused with a message naming the part(s) and, for a
  mismatch, the file's parts

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
  bounding-box faces (the whole extent when it is thinner than 6), with spacing
  a quarter of the skin depth at the top of the source band; where both apply
  the smaller spacing wins. The skin depth is `fdtd_grid.skin_depth(metal, f)`
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
