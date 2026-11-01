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
- **THEN** C is symmetric within 1e-6 relative and each row sums to ≤ 0 off-diagonal

### Requirement: Electrostatics SHALL refuse a mesh above its node cap before factorising
`solve` SHALL count free nodes before assembly and raise `ValueError` naming the count and the
cap when the count exceeds `MAX_ELECTROSTATIC_NODES`. The default cap SHALL be stated in the
module and chosen from a measured peak RSS on the bench, not assumed.

#### Scenario: An oversized mesh is refused before work
- **WHEN** the mesh's free-node count exceeds `MAX_ELECTROSTATIC_NODES`
- **THEN** `solve` raises before calling `mse_backend.factor`

### Requirement: The electrostatics bench SHALL print its checks and exit nonzero on a failure
`scripts/electrostatics_bench.py` SHALL build each validation geometry above as a small
multi-part mesh. It SHALL run at tier 1 under a 4 GiB memory cap and print, per check, the
computed and reference values, the tolerance and the node count. It SHALL report the peak RSS,
and exit 1 if any check fails.

#### Scenario: A wrong permittivity fails the bench
- **WHEN** the per-part ε_s is ignored (all parts treated as vacuum)
- **THEN** the two-layer and dielectric-sphere checks fail and the bench exits 1
