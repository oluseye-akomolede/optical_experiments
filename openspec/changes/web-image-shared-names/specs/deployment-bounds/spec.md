## MODIFIED Requirements

### Requirement: Each component ships as its own image

The system SHALL be built from these images:

- `mse-web` — the Flask app and static viewer (`web/`, `manifests/mse/deployment.yaml`)
- `mse-maxwell-solver` — meshing, FEM, eigen solve, field sampling
  (`dockerfiles/maxwell_solver/`)
- the render-check image — a headless browser for `dockerfiles/rendercheck/`,
  run as a bounded Job (`manifests/mse/rendercheck-job.yaml`)

The web deployment SHALL run within `cpu: 500m` / `memory: 1Gi` limits and
`cpu: 250m` / `memory: 512Mi` requests.

Every module a web module imports SHALL be either standard library, a pinned
web requirement, or a file the web Dockerfile COPYs. A name that both images
need SHALL be defined in `mse_common`, which both Dockerfiles COPY and which
imports only the standard library and numpy — never in a solver module that imports the heavy stack. In
particular the web image SHALL NOT ship `mse_backend.py`, which imports scipy
at module level.

#### Scenario: the render check runs as a bounded Job
- **WHEN** the render-check Job is created
- **THEN** it requests and limits `cpu: 2`, `memory: 2Gi`

#### Scenario: the web app is bounded well under the quota
- **WHEN** the web deployment rolls out
- **THEN** its limits (`500m` CPU, `1Gi`) leave the quota room for a solver Job

#### Scenario: the web app imports with no solver-only module available
- **WHEN** the web image is built and `python -c "import app"` runs inside it
- **THEN** the import succeeds, because no module under `web/` requires a file
  the web Dockerfile does not COPY

#### Scenario: the backend names are one definition
- **WHEN** the GPU manifest names its backend
- **THEN** the value is `mse_common.GPU`, and `mse_backend.CPU` / `.GPU` /
  `.BACKENDS` are the same objects, not copies
