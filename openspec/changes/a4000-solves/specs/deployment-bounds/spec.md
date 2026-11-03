# deployment-bounds delta

## MODIFIED Requirements

### Requirement: The namespace is capped by a ResourceQuota

The `mode-search-engine` namespace SHALL be bounded by a `ResourceQuota` named
`mse-power-cap` in `manifests/mse/power-cap.yaml`:

- `limits.cpu` and `requests.cpu`: **5**
- `limits.memory` and `requests.memory`: **18Gi**
- `requests.nvidia.com/gpu`: **1**
- the quota SHALL also cap the number of Jobs

The GPU bound is **1**, not 3, because the owner chose one solve at a time on
the A4000s. `count/jobs.batch: '1'` and the GPU bound therefore agree: a second
GPU job cannot be admitted while the first exists.

CPU and memory stay at **5 / 18Gi**. They are not coupled to the GPU count, and
the owner's decision raised the card count only; tightening them would remove
the headroom an app rollout needs while a solve runs. The GPU Job itself is
`4` CPU / `14Gi`, so Job plus app is `4.5` CPU / `15Gi` -- inside the cap.

A GPU Job SHALL be pinned to the A4000 by product, not by "any GPU":
`nodeSelector: {nvidia.com/gpu.product: NVIDIA-RTX-A4000}`. The manifest's
memory price is the A4000's, and the quota counts cards without naming a model,
so without the selector a different card could be scheduled and priced wrongly.

A workload SHALL NOT be raised above these bounds without the owner's decision
(per the rig brief, raising quotas is an owner call). The GPU bound is raised
0 -> 1 by the LEAD at deploy time, after the owner's recorded approval; no code
path applies it.

#### Scenario: the quota admits the web app and a bounded solver job
- **WHEN** the quota and the `mse-container-defaults` LimitRange are applied
- **THEN** a workload with no explicit limits receives `cpu: 1`, `memory: 2Gi`
- **AND** no single container may request more than `cpu: 4`, `memory: 16Gi`

#### Scenario: the tracked manifest states the GPU bound
- **WHEN** `manifests/mse/power-cap.yaml` is read
- **THEN** its `spec.hard` carries `requests.nvidia.com/gpu: '1'`

#### Scenario: the quota, not the launcher, serialises GPU solves
- **WHEN** a GPU Job exists and another GPU Job is submitted
- **THEN** the second is refused by the quota
- **AND** this happens without the launcher having to check
