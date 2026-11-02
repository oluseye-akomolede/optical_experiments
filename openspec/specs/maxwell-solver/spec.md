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

An unavailable CUDA stack SHALL raise, never fall back: that is a configuration
fact, known before any work is done, and masking it turns a misconfigured GPU
run into a silent CPU run. An **out-of-memory** failure during a factorisation
or solve is different: it arrives after assembly has been paid for and reflects
transient contention on a shared device. On that error only, the solver SHALL
retry the same problem ONCE on the CPU, and SHALL record that it did.

#### Scenario: an unknown backend name is configured
- **WHEN** `MSE_SOLVER_BACKEND` is set to an unrecognised value
- **THEN** the solver raises rather than silently using the CPU

#### Scenario: the CPU path factorises both matrices
- **WHEN** the CPU backend is used
- **THEN** both `K` and `S` are factorised with the symmetric SPD arguments

#### Scenario: the GPU stack is absent
- **WHEN** the GPU backend is requested but the CUDA stack is missing
- **THEN** the solver raises rather than falling back to the CPU

#### Scenario: the GPU runs out of memory
- **WHEN** a GPU factorisation or solve raises an out-of-memory error
- **THEN** the solver retries the same problem once on the CPU and reports that
  it fell back

#### Scenario: the CPU cannot fit the problem either
- **WHEN** the fallback's problem does not fit this worker's memory limits
- **THEN** the solver raises without attempting the CPU solve, and the error
  names the GPU's out-of-memory failure as the cause and says the problem is
  too large to retry on the CPU

#### Scenario: the CPU solve runs out of memory
- **WHEN** the fallback itself raises an out-of-memory error
- **THEN** the solver raises; it does not retry a second time

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
frequencies with confidence is worse than no answer. On a MULTI-REGION upload
the check SHALL be per region: each region's achieved median edge, measured over
that region's own tets from TetGen's per-tet region attributes, SHALL be compared
against the allowed fraction of THAT REGION's OWN required edge -- `h_r = h *
n_wall / n_r` -- and not against the wall's. The verdict SHALL be the worst
region in the mesh. Within the size cap, and while attempts remain, the mesher
SHALL retry FINER before failing; failing is the last resort. The refusal SHALL
name the region that could not be resolved and suggest fewer modes or a lower
refractive index for it; for a one-part upload, where there is no region to name,
the refusal SHALL instead name the object as too large or detailed. Source:
`dockerfiles/maxwell_solver/mse_meshing.py`,
`dockerfiles/maxwell_solver/mse_common.py`.

#### Scenario: the achieved edge overshoots
- **WHEN** the meshed edge exceeds the allowed overshoot
- **THEN** the job FAILS rather than reporting frequencies from the coarse mesh

#### Scenario: a dense region is built coarser than its own rule
- **WHEN** a region's achieved median edge exceeds the allowed fraction of that
  region's own required edge, even though the mesh as a whole is inside the
  wall's allowance
- **THEN** the job FAILS, naming the region
- **AND** the message suggests fewer modes or a lower refractive index for that
  region, and not a smaller object

#### Scenario: a one-part upload is judged once
- **WHEN** the upload has a single region whose refractive index is the wall's
- **THEN** the per-region verdict is the whole-mesh verdict, and the refusal
  wording is the one-part wording

#### Scenario: every region is inside its own allowance
- **WHEN** each region's achieved median edge is within the allowed fraction of
  that region's own required edge
- **THEN** the job proceeds, whatever the region's refractive index

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

### Requirement: A failed solve tells the owner what actually happened

The owner-facing message for a failed solve SHALL state the cause it observed.
A memory failure on the solver machine SHALL be described as the machine being
out of memory or busy, and SHALL NOT be described as the object being too
complex or the user being advised to simplify it. When the GPU fell back to the
CPU, the run's reported backend SHALL say so. Source:
`dockerfiles/maxwell_solver/mse_maxwell_solver.py`.

This requirement records behaviour that ALREADY held: the OOM branch of
`owner_message` has named the machine rather than the object since 2026-09-27.
What the change adds is that the retry and that message read ONE word list, in
`mse_backend`, so a failure the solver retries cannot be described to the owner
as unsolvable. The scenarios below pin the existing wording.

#### Scenario: the GPU is out of memory and the CPU cannot hold the problem
- **WHEN** the GPU runs out of memory on a Jetson row and the worker's memory
  limits say the same problem cannot be solved on the CPU
- **THEN** the owner is told the machine's processor was busy and the job was
  too large to run on the main processor instead, and to try again later
- **AND** the message promises no automatic retry: on that path the worker
  fails the row, because the app never reads a re-route token for a Jetson row

#### Scenario: a solve is out of memory after the fallback
- **WHEN** a solve fails with an out-of-memory error on both backends
- **THEN** the message says the solver machine was out of memory, and does not
  tell the user the object is too complex

#### Scenario: an object is genuinely too large
- **WHEN** a solve is refused because the estimated memory exceeds the worker's
  limits, before any factorisation
- **THEN** the message may name the object's size as the reason

#### Scenario: the solver library does not load
- **WHEN** the CUDA maths library cannot be loaded for a run, so no
  factorisation happens at all (observed on the Jetson as the memory carve-out
  exhausting before `libcudss` could be mapped)
- **THEN** the message says the problem is on our side and not with the object,
  and does not ask the user to simplify the object

#### Scenario: the driver's allocator reports no memory
- **WHEN** the failure is the driver allocator naming `error 12` (ENOMEM)
- **THEN** it is treated as a memory failure, not as an unsolvable object
- **AND** the same allocator naming any other errno is not treated as one
