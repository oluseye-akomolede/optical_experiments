## ADDED Requirements

### Requirement: The dot's carriers SHALL be solved in 3-D in the gate field
`stark.solve_carrier(box, mass_eff, potential_eV, n_states)` SHALL solve the single-band
effective-mass equation −(ħ²/2)∇·(m*⁻¹∇ψ) + Uψ = Eψ on a uniform 3-D Cartesian
finite-difference grid (7-point, Dirichlet ψ = 0 on the box faces). `box` gives the origin,
the spacing and the shape. `mass_eff` and the band-edge potential are given per grid point. It
SHALL return the lowest `n_states` energies and normalised states from a sparse eigensolve.
`stark.potential_in_field(box, band_edge_eV, field_fn, charge)` SHALL add the electrostatic
energy −qφ to the band edge, with φ interpolated from an `electrostatics` solution (or from a
uniform field for the checks), with the carrier's own sign of charge. It SHALL refuse with
`ValueError` a grid spacing above one quarter of the smallest confinement length it is given,
and a box whose grid would exceed the admission budget (estimated before the eigensolve).

#### Scenario: The 3-D harmonic oscillator
- **WHEN** a carrier of mass m sits in U = ½mω²r² on a box of 8 oscillator lengths per side
- **THEN** the lowest four distinct levels are ħω(n + 3/2) for n = 0..3, each within 1 %, with
  the degeneracies 1, 3, 6, 10 (split by less than 1 %)

#### Scenario: The cubic infinite well
- **WHEN** a carrier sits in a cube of side L with U = 0 inside and the box faces as the walls
- **THEN** the ground energy is 3π²ħ²/(2mL²) within 1 %

#### Scenario: The exact Stark shift of an oscillator
- **WHEN** a uniform field F is added to the 3-D oscillator for a carrier of charge q
- **THEN** the ground energy shifts by −q²F²/(2mω²) within 2 % for three values of F, and the
  state's centre moves by qF/(mω²) within 2 %

### Requirement: The exciton energy SHALL be the two carriers' levels plus a first-order Coulomb term
`stark.exciton(electron, hole, eps_static)` SHALL return E_X = E_g + E_e + E_h − J_eh, with the
first-order Coulomb integral J_eh = (e²/4πε₀ε_s)∫∫|ψ_e(r)|²|ψ_h(r′)|²/|r − r′| d³r d³r′,
computed by an FFT convolution on the zero-padded grid. It SHALL also return the overlap
|⟨ψ_e|ψ_h⟩|², and the separation of the electron and hole centres along the field. `E_g` is an
input per material.

#### Scenario: The Coulomb integral of two Gaussians
- **WHEN** the electron and the hole are both the ground state of the same 3-D oscillator of
  length l = √(ħ/mω)
- **THEN** J_eh = (e²/4πε₀ε_s)·√(2/π)/l within 2 %

#### Scenario: The overlap falls as the field pulls the carriers apart
- **WHEN** an electron and a hole in the same oscillator see a growing field
- **THEN** the overlap equals exp(−d²/(2l²)) within 2 %, where d is the exact separation of the
  two shifted ground states

### Requirement: The Stark sweep SHALL be fitted to a dipole and a polarisability
`stark.sweep(...)` SHALL compute E_X at each voltage of a list, from the field the
electrostatics solve gives at each voltage. It SHALL fit E_X(F_z) = E₀ − pF_z + βF_z² by least
squares (`coulomb=True` by default; `False` drops J_eh, for the exact check), and return E₀, p, β, the residual of the fit and the overlap at each voltage. The
dipole orientation SHALL be returned as in-plane (perpendicular to the wire axis): an assumption
of the heavy-hole single-band model, stated as such in the result, not computed.

#### Scenario: The fit recovers the exact coefficients
- **WHEN** the sweep runs on oscillator carriers in a uniform field, with the hole's oscillator
  centre offset by s along the field from the electron's
- **AND** the sweep is run with `coulomb=False`, so E_X is E_g + E_e + E_h and the reference
  is exact (J_eh also changes with the separation)
- **THEN** the fitted p is e·s within 2 % and the fitted β is the exact
  −e²/(2ω²)·(1/m_e + 1/m_h) within 3 %

### Requirement: The Stark bench SHALL print its checks and exit nonzero on a failure
`scripts/stark_bench.py` SHALL run every check above at tier 1 under a 4 GiB cap, print the
computed value, the reference, the tolerance and the grid size per check and the peak RSS, and
exit 1 when any check fails. A comparison with a published measured Stark curve of a nanowire
dot SHALL be printed for reading only, after its source has been fetched and is cited by
section and figure **(unsure: which paper; task 0.2)**, never as a pass condition.

#### Scenario: A dropped field term fails the bench
- **WHEN** `potential_in_field` ignores the field (returns the band edge alone)
- **THEN** the exact-Stark-shift and fit checks fail and the bench exits 1

#### Scenario: A wrong carrier sign fails the bench
- **WHEN** the hole is given the electron's sign of charge
- **THEN** the fit's p check fails (wrong sign) and the bench exits 1

### Requirement: The dot SHALL have a lateral extent, proven on exact answers
`exciton.dot_fill_fraction(box, well_height_nm, well_lateral_nm, well_mask)` SHALL give the dot
its lateral size as a rectangular box dot centred on the grid (height along z, the field axis;
one side or an (x, y) pair laterally), with the host as the barrier around it at the same depth
as above and below. `well_lateral_nm=None` SHALL mean the well fills the box laterally (the 1.9
model) and SHALL reproduce the 1.9 potential exactly. `exciton._potential_grid` and
`exciton.exciton_energy_eV` SHALL take `well_lateral_nm` and `well_mask` and pass them through.
The dot is a box, not a cylinder, because L3 runs on `stark.py`'s uniform Cartesian 7-point
grid (design §1), where a box's faces lie on grid planes and a cylinder's would be a staircase.

`well_mask="node"` SHALL be the 1.9 rule (a node is inside when it lies in the closed dot) and
stays the default; `"cell"` SHALL weight each node by the fraction of its dual cell inside the
dot, which preserves the well width on any grid. A lateral side at or beyond the box extent, a
lateral side with no depth, a non-positive side and an unknown mask SHALL be refused with
`ExcitonError`. `stark.hard_box_ground_eV` and `stark.finite_well_in_box` (the exact even ground
state and ⟨z²⟩ of a 1-D finite well inside hard walls) SHALL be the exact references.

#### Scenario: The hard-wall limit converges at second order
- **WHEN** a 6 × 6 × 4 nm cell-mask dot of depth 10⁶ eV sits in an 8 nm cube at 8, 16, 32 cells
- **THEN** the ground energy approaches K π² (2/6² + 1/4²)/m with a measured order above 1.8
  at both refinements

#### Scenario: The separable finite box well converges at second order
- **WHEN** V = V₀(3 − f_x − f_y − f_z) is built from the same per-axis cell fractions
- **THEN** the ground energy approaches the sum of three `finite_well_in_box` roots with a
  measured order above 1.8 at both refinements

#### Scenario: The dot reduces to the 1.9 well when it fills the box laterally
- **WHEN** `well_lateral_nm=None` with the cell mask
- **THEN** the ground energy approaches the 1-D z root plus the hard lateral box, order above 1.8

#### Scenario: The non-separable dot is bracketed and self-converges
- **WHEN** the 6 × 6 × 4 nm, 0.5 eV dot is solved at 16, 32 and 64 cells
- **THEN** E_sep(V₀/3) < E_dot < E_sep(V₀) on the 32-cell grid and, with the Richardson value
  of E_dot, against the exact continuum ends; and its self-convergence order is above 1.8

#### Scenario: The node mask's width error is visible
- **WHEN** the 1.9 bench well (4 nm, 0.8 eV, the dot electron, h_z = 0.4 nm) is solved with both
  masks
- **THEN** the node mask's ⟨z²⟩ lies above the exact `finite_well_in_box` value and converges to
  it at first order, and the cell mask's error is below 0.15 of the node mask's

#### Scenario: The bench reports binding three ways and does not merge them
- **WHEN** `scripts/stark_bench.py --lateral` runs (default side: the equal-area square of
  `dot_material.DOT_DIAMETER_NM`, never typed)
- **THEN** it prints, per depth, whether E_e lies below the continuum edge (V₀ with a lateral
  dot; V₀ plus the box column's lateral energy without), the probability inside the dot, and
  the 1.9 length criterion ⟨z²⟩_e ≤ h²/12 with its lateral analogue; and it refuses a lateral
  side not strictly inside `--transverse`
