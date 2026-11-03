# gpu-solves Specification

## Purpose

How an owner upload that needs a GPU is run on the cluster's RTX A4000s as a
bounded Kubernetes Job, and how a job that does not fit the card still reaches
the owner with an honest message.

## ADDED Requirements

### Requirement: A GPU job is a bounded Kubernetes Job on one A4000

A GPU solve SHALL run as a `batch/v1` Job in namespace `mode-search-engine`,
pinned by `nodeSelector` to `nvidia.com/gpu.product: NVIDIA-RTX-A4000`, and
SHALL request exactly one GPU with `requests` equal to `limits`.

Because the quota is `requests.nvidia.com/gpu: '1'` and `count/jobs.batch: '1'`,
at most one GPU solve runs at a time. That is the owner's choice (one GPU, one
solve at a time), and it is enforced by the quota, not merely by the launcher.

The Job SHALL set `activeDeadlineSeconds`, so a run that hangs is stopped by the
cluster and not left occupying the single GPU slot indefinitely.

#### Scenario: the manifest a GPU job is submitted with
- **WHEN** a job routed to the GPU route is launched as a cluster Job
- **THEN** its container requests `nvidia.com/gpu: 1`
- **AND** `requests` and `limits` are equal for cpu, memory and the GPU
- **AND** `activeDeadlineSeconds` is a positive integer
- **AND** the pod's `nodeSelector` carries `nvidia.com/gpu.product:
  NVIDIA-RTX-A4000`

#### Scenario: thread counts match the CPU limit
- **WHEN** the GPU Job manifest is built
- **THEN** `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` and
  `NUMEXPR_NUM_THREADS` equal the container's CPU limit
- **AND** they are not left unset

### Requirement: The GPU route is chosen by a setting, and the Jetson stays reachable

The environment variable `MSE_GPU_ROUTE` SHALL select where a GPU-needing job
runs, and SHALL take the values `cluster` (default), `jetson`, and `fallback`:

- `cluster`: the A4000 Job.
- `jetson`: the existing pull worker.
- `fallback`: the A4000 Job when the GPU quota slot is free, otherwise the
  Jetson.

An unrecognised value SHALL be treated as `cluster` rather than raising, so a
typo in a deployment environment cannot take the site down.

The Jetson route and its worker SHALL remain in place. This capability adds a
route; it does not remove the Jetson.

#### Scenario: the default sends a GPU job to the cluster
- **WHEN** `MSE_GPU_ROUTE` is unset and an upload's estimate is at or above
  `gpu_min_tets()`
- **THEN** the job is created with the cluster-GPU target

#### Scenario: the Jetson is still selectable
- **WHEN** `MSE_GPU_ROUTE` is `jetson`
- **THEN** the same upload is created with the Jetson target
- **AND** no GPU Job manifest is built for it

#### Scenario: an unknown value does not break uploads
- **WHEN** `MSE_GPU_ROUTE` is set to a value that is not one of the three
- **THEN** the job is created with the cluster-GPU target
- **AND** the upload does not fail

### Requirement: A job too big for the card ends with an honest message

When a GPU job fails because it does not fit the card's memory, the row's
`message` SHALL state that in plain language -- that the model did not fit the
GPU -- and SHALL NOT say "contact support" or leave a bare exception. This is
the same contract the Jetson GPU-out-of-memory path already meets.

#### Scenario: an out-of-memory GPU job is explained, not blamed on the user
- **WHEN** a GPU job fails with an error that names an out-of-memory condition
- **THEN** the job's `message` says the model did not fit the GPU
- **AND** the message is not the "contact support" text
