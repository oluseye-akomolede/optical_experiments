# maxwell-solver Specification

## Purpose
What the solver computes: the electromagnetic modes of a perfectly conducting (PEC) cavity, filled with parts of given refractive index, found with the vector Maxwell equations on Nédélec elements.

## Requirements

### Requirement: The physics is the vector Maxwell eigenproblem on Nédélec elements

The solver SHALL assemble and solve the vector Maxwell eigenproblem
`∫ curl E · curl v = k² ∫ ε_r E · v` with `ε_r = n²` uniform, using lowest-order
Nédélec (Whitney) edge elements on linear tetrahedra, one DOF per edge. Source:
`dockerfiles/maxwell_solver/mse_fem.py`.

#### Scenario: the material enters only the mass matrix
- **WHEN** the element matrices are assembled with a refractive index
- **THEN** the curl-curl operator is untouched and the mass matrix is scaled by
  `ε_r`

### Requirement: The boundary condition is PEC, enforced by deletion

Edges on a boundary face SHALL be constrained by DELETING their rows and columns,
not by pinning them; a pinned edge would leave a spurious mode at the pin value.
Source: `dockerfiles/maxwell_solver/mse_fem.py`.

#### Scenario: boundary edges are constrained
- **WHEN** the reduced matrices are built
- **THEN** the constrained edges' rows and columns are removed
- **AND** the retained-index list is returned for mapping back

### Requirement: Modes are found with a projected shift-invert Lanczos

The lowest `n_modes` physical eigenvalues SHALL be found with a NEGATIVE shift
`s = (π/D)²` (so `A + sB` is symmetric positive definite), a projection that
removes the gradient null space, and `eigsh(..., which='LM')` mapped back to
physical eigenvalues `k²` in ascending order. Source:
`dockerfiles/maxwell_solver/mse_eigen.py`.

#### Scenario: the mesh is too coarse
- **WHEN** the free edge unknowns are fewer than `2·n_modes + 2`
- **THEN** the solver raises a "mesh is too coarse" error rather than returning
  garbage

#### Scenario: a non-physical mode is returned
- **WHEN** an eigenvalue falls at or below `SPURIOUS_FRACTION` of the largest
- **THEN** the solver RAISES rather than silently dropping it

### Requirement: Frequencies are reported in SI from a metre-scaled mesh

The mesh SHALL be scaled to metres before the physics, so `k`, frequency and
wavelength are all SI. Each mode SHALL report `k0_per_m` (free-space
wavenumber), `k_per_m` (in-medium wavenumber), `freq_hz` (vacuum frequency),
and `vacuum_wavelength_m`. Source: `dockerfiles/maxwell_solver/mse_eigen.py`,
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: a mode is reported
- **WHEN** a mode's eigenvalue `k²` is converted
- **THEN** `freq_hz = c · (k / n) / (2π)` with `c = 299792458 m/s`

### Requirement: The solver backend is one factory routing BOTH factorisations

The solver SHALL select its backend through ONE factory
(`MSE_SOLVER_BACKEND`, `cpu` default, `gpu`), and SHALL route BOTH the shift
matrix `K` and the projector matrix `S` through it (an OPinv application solves
against each). A `splu` call SHALL go through the module attribute, never a
module-level `from ... import splu`, so tests can observe both factorisations.
An unknown backend SHALL raise, not default. Source:
`dockerfiles/maxwell_solver/mse_backend.py`.

#### Scenario: an unknown backend name is configured
- **WHEN** `MSE_SOLVER_BACKEND` is set to an unrecognised value
- **THEN** the solver raises rather than silently using the CPU

#### Scenario: the CPU path factorises both matrices
- **WHEN** the CPU backend is used
- **THEN** both `K` and `S` are factorised with the symmetric SPD arguments

#### Scenario: the GPU stack is absent
- **WHEN** the GPU backend is requested but the CUDA stack is missing
- **THEN** the solver raises rather than falling back to the CPU

### Requirement: A GPU result is checked, not trusted

A GPU single-precision solve SHALL be refined on the host and its relative
residual SHALL be checked against a tolerance; a residual over the tolerance
SHALL RAISE, and the worst residual SHALL be surfaced in the run's report.
Source: `dockerfiles/maxwell_solver/mse_backend.py`.

#### Scenario: refinement fails to converge
- **WHEN** the refined GPU solve's relative residual exceeds the tolerance
- **THEN** the solver raises, naming the residual and the remedy

#### Scenario: the run reports its residual
- **WHEN** a solve completes
- **THEN** the reported backend description includes the worst refined residual

### Requirement: The interior element size follows Weyl's law from enclosed volume

The target element size SHALL derive from the ENCLOSED volume (not the bounding
box) via Weyl's law, at a fixed number of edges per wavelength, and SHALL be
lowered (never raised) by the owner-declared smallest feature. Source:
`dockerfiles/maxwell_solver/mse_common.py`.

#### Scenario: a smallest feature is declared
- **WHEN** the declared feature is smaller than the physics-derived edge
- **THEN** the target edge is lowered to the feature

### Requirement: The surface can floor the mesh, so an over-refined surface is decimated

TetGen cannot make interior elements much larger than the surface facets, so the
mesher SHALL detect an over-refined surface (median surface edge below a trigger
fraction of the target edge) and decimate it before tetrahedralizing. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: an over-refined upload is meshed
- **WHEN** the upload's surface is finer than the target edge by the trigger
- **THEN** the surface is decimated before tetrahedralizing
- **AND** the interior does not take the surface's over-refinement

### Requirement: An achieved resolution worse than allowed is a failure

The mesher SHALL measure the achieved element edge (median), and SHALL FAIL if
it overshoots the target by more than the allowed fraction; showing wrong
frequencies with confidence is worse than no answer. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: the achieved edge overshoots
- **WHEN** the meshed edge exceeds the allowed overshoot
- **THEN** the job FAILS rather than reporting frequencies from the coarse mesh

### Requirement: An open surface is rejected

The mesher SHALL reject a surface that is not watertight, naming the fault.
Source: `dockerfiles/maxwell_solver/mse_meshing.py`.

#### Scenario: the upload has holes
- **WHEN** the loaded surface is not watertight
- **THEN** the job fails with a message that the object is not a closed surface

### Requirement: The solver writes a defined artifact set

A solved job SHALL write `surface.glb` (in the user's units, for the viewer),
`modes.json` (job metadata plus per-mode values), `mode_NN.bin` (float32, C
order, shape `[nz][ny][nx][3]`, `NaN` outside the object), and `mode_NN.vtu`
(an unstructured grid carrying `E` and `E_magnitude`). Source:
`dockerfiles/maxwell_solver/mse_maxwell_solver.py`,
`dockerfiles/maxwell_solver/mse_fields.py`.

#### Scenario: a mode field is written
- **WHEN** a mode's field is sampled and written
- **THEN** its bytes are `float32`, C order, with the three components last
- **AND** each mode is normalised so `max|E| == 1`

### Requirement: Re-running a job does not leave stale modes

Before writing modes, the solver SHALL delete the job's existing mode rows, so a
re-run with fewer modes does not leave stale higher indices. Source:
`dockerfiles/maxwell_solver/mse_db.py`.

#### Scenario: a job is re-solved with fewer modes
- **WHEN** modes are inserted for a job that already had more
- **THEN** the higher stale indices are gone
