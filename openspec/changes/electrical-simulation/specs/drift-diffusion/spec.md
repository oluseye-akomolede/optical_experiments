## ADDED Requirements

### Requirement: Drift-diffusion SHALL solve Poisson and both carrier continuities through DEVSIM
`drift_diffusion.solve(mesh, regions, doping, contacts, biases, form)` SHALL build a DEVSIM
device and solve the coupled Poisson, electron-continuity and hole-continuity equations with
Scharfetter-Gummel edge currents, Boltzmann statistics, Shockley-Read-Hall and radiative
recombination, and ohmic contacts. `form` SHALL be `"axisymmetric"` (a 2-D (r, z) mesh of an
equal-area cylinder, each node and edge weighted by 2πr) or `"3d"` (the tetrahedral mesh from
`mse_meshing.tetrahedralize`). Material parameters (ε_s, n_i, mobilities, lifetimes, B) SHALL be
inputs per region, each with its cited source recorded in the module. It SHALL return, per
bias, the terminal currents, the potential and both carrier densities at every node, and the
field at the dot position. It SHALL solve each bias by Newton from the previous one and raise
`RuntimeError` naming the bias when Newton does not converge.

#### Scenario: The built-in potential of an abrupt junction
- **WHEN** a 1-D-like abrupt p-n junction (a wide axisymmetric cylinder) at zero bias is solved
- **THEN** the potential step across it is (k_BT/q)·ln(N_A·N_D/n_i²) within 1 %

#### Scenario: The depletion width
- **WHEN** the same junction is solved at zero and at a reverse bias V_R
- **THEN** the depletion width matches √(2ε_s(V_bi + V_R)(N_A + N_D)/(q·N_A·N_D)) within 10 %
  (the depletion approximation is itself approximate at the edges)

#### Scenario: The ideal-diode slope
- **WHEN** a long p-n diode with only SRH recombination outside the depletion region is swept
  in forward bias at low injection
- **THEN** d(ln I)/dV over the low-injection range is q/(k_BT) within 3 % (ideality 1)

#### Scenario: Current is conserved
- **WHEN** any bias is solved
- **THEN** the currents of all contacts sum to zero within 1e-6 of the largest one

### Requirement: The mesh SHALL resolve the Debye length where carriers change, or the run SHALL be refused
Before solving, `solve` SHALL compute the extrinsic Debye length L_D = √(ε_s ε₀ k_BT/(q²N)) at
every doped node from its doping N, and SHALL refuse with `ValueError` (naming the region, the
worst L_D and the local edge) a mesh whose edge exceeds L_D/2 at any node within a stated
distance of a doping change. The refusal SHALL be the default; a coarse cross-check MAY pass
`allow_unresolved=True`, and its result SHALL then carry a caveat naming the unresolved regions.

#### Scenario: An unresolved junction is refused
- **WHEN** a 3-D mesh with a 20 nm junction edge is given 10¹⁸ cm⁻³ doping (L_D ≈ 4 nm)
- **THEN** `solve` raises `ValueError` naming the region and both lengths

### Requirement: The axisymmetric and coarse 3-D forms SHALL agree where both are resolved
For the p-i-n wire at a doping both forms resolve (design estimate ≤ 10¹⁶ cm⁻³), the
axisymmetric (r, z) form and the 3-D form SHALL give the same I-V.

#### Scenario: The two forms agree on a coarse p-i-n wire
- **WHEN** a small p-i-n hexagonal wire and its equal-area cylinder are solved at five forward
  biases
- **THEN** the currents agree within 5 % at every bias

#### Scenario: The cylinder's weight is the area
- **WHEN** a uniformly doped resistor cylinder of radius R is solved in the axisymmetric form
- **THEN** its current is the 1-D current density times πR² within 1 %

### Requirement: A voltage pulse SHALL give the carrier arrival profile at the dot
`drift_diffusion.transient(..., waveform, dt, t_end)` SHALL step DEVSIM's transient solve from
a solved bias through a voltage waveform. It SHALL return g(t), the rate at which electron-hole
pairs recombine radiatively inside the dot's region, and the contact currents. The result SHALL
conserve charge.

#### Scenario: Charge balance in a transient
- **WHEN** a forward voltage step is applied
- **THEN** the time-integrated contact current equals the change in stored charge plus the
  integrated recombination within 1 %

#### Scenario: The junction capacitance
- **WHEN** a small reverse-bias step ΔV is applied
- **THEN** the change in stored charge over ΔV matches ε_s·A/W at that bias within 5 %

### Requirement: The drift-diffusion bench SHALL print its checks and exit nonzero on a failure
`scripts/dd_bench.py` SHALL run every check above (tier 1 for the 1-D-like and small cases,
tier 2 or 3 for the coarse 3-D wire), print the computed value, the reference, the tolerance
and the unknown count per check, and the peak RSS, and exit 1 when any check fails. It SHALL
record DEVSIM's version.

#### Scenario: A central-difference current fails the bench
- **WHEN** the Scharfetter-Gummel edge current is replaced by a central difference of n and ψ
- **THEN** the built-in potential or ideal-diode check fails, or Newton fails, and the bench
  exits 1

#### Scenario: A dropped 2πr weight fails the bench
- **WHEN** the axisymmetric form omits the 2πr weight
- **THEN** the cylinder-area check fails and the bench exits 1
