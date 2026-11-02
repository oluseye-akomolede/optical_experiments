## MODIFIED Requirements

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

## ADDED Requirements

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
