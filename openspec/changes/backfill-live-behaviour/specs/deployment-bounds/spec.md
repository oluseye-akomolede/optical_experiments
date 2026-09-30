# deployment-bounds

## ADDED Requirements

### Requirement: The namespace is capped by a ResourceQuota

The `mode-search-engine` namespace SHALL be bounded by a `ResourceQuota` named
`mse-power-cap` in `manifests/mse/power-cap.yaml`:

- `limits.cpu` and `requests.cpu`: **5**
- `limits.memory` and `requests.memory`: **18Gi**
- the quota SHALL also cap the number of Jobs and the total GPU count

A workload SHALL NOT be raised above these bounds without the owner's decision
(per the rig brief, raising quotas is an owner call).

#### Scenario: the quota admits the web app and a bounded solver job
- **WHEN** the quota and the `mse-container-defaults` LimitRange are applied
- **THEN** a workload with no explicit limits receives `cpu: 1`, `memory: 2Gi`
- **AND** no single container may request more than `cpu: 4`, `memory: 16Gi`

### Requirement: Thread counts SHALL equal the CPU limits

Every container that runs a numerically threaded workload SHALL set
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` (and the equivalent
BLAS variables) to its CPU limit, because a cgroup CPU limit does not change
`os.cpu_count()` and the workload would otherwise start one thread per HOST core
and be throttled. Source: `dockerfiles/maxwell_solver/Dockerfile`,
`dockerfiles/maxwell_solver/mse_maxwell_solver.py`.

#### Scenario: the solver image fixes its thread count
- **WHEN** the solver image is built
- **THEN** `OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` are all set to the image's CPU allowance (`4`)

#### Scenario: the solver caps its thread count at runtime
- **WHEN** the solver process starts
- **THEN** it sets the same thread variables before importing numpy, unless they
  are already set

### Requirement: Each component ships as its own image

The system SHALL be built from these images:

- `mse-web` — the Flask app and static viewer (`web/`, `manifests/mse/deployment.yaml`)
- `mse-maxwell-solver` — meshing, FEM, eigen solve, field sampling
  (`dockerfiles/maxwell_solver/`)
- the render-check image — a headless browser for `dockerfiles/rendercheck/`,
  run as a bounded Job (`manifests/mse/rendercheck-job.yaml`)

The web deployment SHALL run within `cpu: 500m` / `memory: 1Gi` limits and
`cpu: 250m` / `memory: 512Mi` requests.

#### Scenario: the render check runs as a bounded Job
- **WHEN** the render-check Job is created
- **THEN** it requests and limits `cpu: 2`, `memory: 2Gi`

#### Scenario: the web app is bounded well under the quota
- **WHEN** the web deployment rolls out
- **THEN** its limits (`500m` CPU, `1Gi`) leave the quota room for a solver Job
