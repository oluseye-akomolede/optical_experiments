## ADDED Requirements

### Requirement: Electrostatics SHALL solve Poisson on the existing multi-part tetrahedral mesh
`electrostatics.solve(points_m, tets, region_of_tet, eps_static, electrodes)` SHALL assemble
the P1 (linear nodal) stiffness matrix of ∇·(ε₀ε_s∇φ) = 0 on the mesh `mse_meshing.tetrahedralize`
returns, with ε_s taken per part from `eps_static` (a real ≥ 1 per part label). It SHALL hold
every node of each electrode part (a mapping of part label to volts) at that potential, apply a
zero-normal-field (natural) condition on the outer boundary, and solve the reduced SPD
system through `mse_backend.factor`. It SHALL refuse with `ValueError` a mesh with no electrode,
a part with no `eps_static` that is not an electrode, or an ε_s < 1.

#### Scenario: Static permittivity is never derived from the optical index
- **WHEN** a part has an optical material but no `eps_static` entry and is not an electrode
- **THEN** `solve` raises `ValueError` naming the part

#### Scenario: An electrode part needs no static permittivity
- **WHEN** an electrode part has no `eps_static` entry
- **THEN** `solve` accepts it, because its nodes are held at the electrode potential and its
  tets are not assembled

#### Scenario: The potential lies between the electrode potentials
- **WHEN** `solve` returns with electrodes held at potentials V_min ≤ V ≤ V_max
- **THEN** every nodal potential lies in [V_min, V_max] within 1e-9·(V_max − V_min) (the
  discrete maximum principle, checked on the bench meshes)

### Requirement: The field SHALL be sampled at a point and capacitances SHALL come from the residual charge
`electrostatics.field_at(solution, point_m)` SHALL return E = −∇φ in V/m in the tetrahedron that
contains the point, and SHALL raise `ValueError` for a point outside the mesh or inside an
electrode part. `electrostatics.capacitance_matrix(...)` SHALL drive each electrode at 1 V with
the others at 0 V. It SHALL read each electrode's charge from the assembled residual at that
electrode's nodes and return the Maxwell capacitance matrix in farads.

#### Scenario: Two-layer parallel plate
- **WHEN** a slab capacitor of two dielectric layers between plate electrodes is solved
- **THEN** the field in each layer matches the series-capacitor value within 1 % and the
  capacitance matches ε₀A/(t₁/ε₁ + t₂/ε₂) within 2 % (fringing kept small by the mesh's lateral extent)

#### Scenario: Concentric spheres
- **WHEN** two concentric sphere electrodes (radii a < b) with dielectric ε_s between are solved
- **THEN** C = 4πε₀ε_s·ab/(b − a) within 3 %

#### Scenario: Dielectric sphere in a uniform field
- **WHEN** a sphere of ε_s ∈ {4, 12.9} sits centred between distant plates that apply E₀
- **THEN** the field at the sphere centre is 3E₀/(ε_s + 2) within 5 %

#### Scenario: The capacitance matrix is symmetric
- **WHEN** `capacitance_matrix` is computed for three or more electrodes
- **THEN** C = Cᵀ within 1e-6 relative; C[i][i] > 0; and C[i][j] ≤ 0 for i ≠ j.
  C[i][j] is the charge induced on electrode `i` when electrode `j` is driven at
  1 V, so driving puts POSITIVE charge on the driven electrode.

#### Scenario: The rows sum to zero only when every held surface is in C
The zero-flux outer boundary makes the total charge zero by Gauss's law, so a row's
sum reports where the driven electrode's charge WENT, and that depends on whether
every surface held at a fixed potential is one of C's electrodes:
- **WHEN** every fixed-potential surface is an electrode of `capacitance_matrix`
  **THEN** each row of C sums to 0 to round-off (pinned within 1e-9 on the
  earthed two-plate fixture, measured ~1e-20)
- **WHEN** a surface is held at a fixed potential but is NOT one of C's electrodes
  (e.g. a grounded substrate the call does not name)
  **THEN** the charge that left for that surface is not in any row, so every row
  sums to ≥ 0 rather than 0. This is the honest statement of the property: the
  unconditional "each row sums to ≤ 0" this scenario used to assert was garbled
  and did not describe any physical quantity. The ≥ 0 case is NOT yet pinned by
  a test — no fixture here holds a surface the call omits — and is stated as an
  unimplemented check rather than a measured one.

### Requirement: Generality SHALL be tested on devices that are not a wire
The benches SHALL include at least two devices whose geometry is not the paper-1 wire: a
**parallel-plate capacitor** whose capacitance is C = ε₀ε_r A/d within 2 %, and a
**multi-terminal MOS-like stack** (a substrate, an oxide, two top electrodes at different
voltages) whose capacitance matrix is checked against the two-terminal series/parallel
closed forms and whose three-electrode symmetry is checked. Paper 1 SHALL be one fixture
among these, not the only one.

#### Scenario: Parallel plates match the closed form
- **WHEN** a parallel-plate capacitor with known area and gap is solved at two dielectric
  constants
- **THEN** its capacitance is ε₀ε_r A/d within 2 %, and the ratio of the two runs is the ratio
  of the dielectric constants

#### Scenario: A three-terminal stack has the right cross-couplings
- **WHEN** a substrate and two top electrodes are solved with the substrate grounded
- **THEN** the capacitance matrix matches the closed forms, C is symmetric, and the coupling
  between the two top electrodes is positive and smaller than either's coupling to ground


### Requirement: Electrostatics SHALL refuse a mesh over the admission budget before factorising
`solve` SHALL take `budget=None` (bytes). With `None` it SHALL read the machine's admission
budget at call time (`admission_budget()`, as the time-domain `check_limits` does), not a fixed
constant. Before assembly it SHALL estimate peak memory from the free-node count for the solve
path it will take (direct or iterative, below), using a per-node cost for each path measured on
the bench and recorded in the module. It SHALL raise `ValueError` naming the path, the estimate
and the budget when the estimate exceeds the budget.

#### Scenario: A mesh over a passed budget is refused before work
- **WHEN** `solve` is called on a small bench mesh with a `budget` below that mesh's estimate
- **THEN** `solve` raises before calling `mse_backend.factor` or the iterative solver

#### Scenario: The estimate follows the path
- **WHEN** the same mesh is estimated for the direct path and for the iterative path
- **THEN** the two estimates differ and each uses its own recorded per-node cost

### Requirement: Electrostatics SHALL solve large 3-D meshes iteratively, with a checked residual
`solve` SHALL use the direct path (`mse_backend.factor`) up to `direct_max_nodes` free nodes
(a module constant set from the bench, design estimate about 5 × 10⁵) and above it an
AMG-preconditioned conjugate-gradient solve (`pyamg`). The iterative path SHALL stop at a
relative residual ‖b − Ax‖/‖b‖ ≤ 1e-10. It SHALL recompute that residual from the assembled
matrix after the solve, not take it from the solver's own report, and SHALL raise
`RuntimeError` naming the residual when it is above the tolerance or the iteration cap is
reached. The result SHALL record which path ran and the residual.

#### Scenario: Both paths agree
- **WHEN** a bench mesh below `direct_max_nodes` is solved on both paths (forced by a keyword)
- **THEN** the nodal potentials agree within 1e-8·(V_max − V_min)

#### Scenario: An unconverged solve is refused
- **WHEN** the iterative path is run with an iteration cap too small to converge
- **THEN** `solve` raises `RuntimeError` naming the recomputed residual, and returns no field

### Requirement: The electrostatics mesh SHALL be graded per part through TetGen's sizing field
`mse_meshing.tetrahedralize` SHALL accept an opt-in `part_edge_m` mapping (part label to a
target edge in metres) and, when it is given, SHALL grade the mesh with TetGen's `-m` sizing
field: a coarse background mesh of the same PLC whose nodes carry the target edge of the part
they lie in, passed as `bgmesh`, with no global `maxvolume` and no `fixedvolume`. With
`part_edge_m=None` (the default) it SHALL run today's path unchanged, so the `maxwell-solver`
behaviour does not change. A target edge larger than the thinnest dimension of its part SHALL
be refused with `ValueError` naming the part, because it would drive the mesh onto the size
floor.

#### Scenario: Grading keeps the fine part fine and coarsens the rest
- **WHEN** a thin-layer bench geometry is meshed with a fine edge in the layer and a coarse
  edge in the surrounding part
- **THEN** the measured median edge in the layer is within 25 % of its target and the total
  tet count is below half that of the uniform mesh at the fine edge

#### Scenario: The default path is unchanged
- **WHEN** `tetrahedralize` is called without `part_edge_m`
- **THEN** it returns the same arrays as before this change for the existing test meshes

### Requirement: The paper-1 wire geometry SHALL solve in 3-D and respect its symmetry
The electrostatics bench SHALL include a reduced paper-1 scene: a hexagonal wire of ε_s = 12.5
on a 12 nm oxide (ε_s = 3.9) over a grounded mirror electrode, with a top electrode, in an air
box, meshed with `part_edge_m`. The result SHALL be a genuine 3-D field, not an axisymmetric
one: the bench SHALL print the field at the dot position on the wire axis, and the field at
the same height off the axis rotated by 60° (a symmetry of the hexagon) and by 30° (not one).

#### Scenario: The on-axis field points along the axis
- **WHEN** the reduced paper-1 scene is solved with the top electrode at V and the mirror at 0
- **THEN** at the on-axis dot position |E_x|, |E_y| ≤ 1e-2·|E_z| (the hexagon's symmetry)

#### Scenario: A 60° rotation is a symmetry of the solution
- **WHEN** the same solve is sampled at 0.8 R off the axis and at that point rotated by 60°
- **THEN** the two field magnitudes agree within 2 % (a check that fails if the mesh or the
  sampling breaks the hexagon's symmetry; the 30° point is printed for reading, not checked)

#### Scenario: The field converges under refinement
- **WHEN** the scene is solved at two gradings whose wire edges differ by 2
- **THEN** the on-axis |E_z| at the dot changes by less than 2 %

### Requirement: The electrostatics bench SHALL print its checks and exit nonzero on a failure
`scripts/electrostatics_bench.py` SHALL build each validation geometry above as a small
multi-part mesh. It SHALL run at tier 1 under a 4 GiB memory cap and print, per check, the
computed and reference values, the tolerance and the node count. It SHALL report the peak RSS,
and exit 1 if any check fails.

#### Scenario: A wrong permittivity fails the bench
- **WHEN** the per-part ε_s is ignored (all parts treated as vacuum)
- **THEN** the two-layer and dielectric-sphere checks fail and the bench exits 1

#### Scenario: A dropped residual check fails the bench
- **WHEN** the iterative path's recomputed residual is replaced by the solver's own report and
  the iteration cap is set too low
- **THEN** the both-paths-agree check fails and the bench exits 1
