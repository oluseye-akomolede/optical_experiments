# Tasks: run owner uploads' solves on the cluster A4000s

## 1. Routing

- [x] 1.1 `web/db.py`: add `CLUSTER_GPU = "cluster-gpu"` and extend `TARGETS`.
      The column already stores a plain string, so no migration is needed; note
      in the comment that old rows are unaffected.
- [x] 1.2 `web/config.py`: `gpu_route()` reading `MSE_GPU_ROUTE`, defaulting to
      `cluster`, and treating an unrecognised value as `cluster`.
- [x] 1.3 `web/app.py`: split the single comparison into "needs a GPU" and
      "which GPU machine", per `design.md`. Keep the CPU-cluster path unchanged.

**Done notes (1.x).** `next_queued()` takes a tuple of targets
(`CLUSTER_TARGETS`) so a `cluster-gpu` row is picked up by the same single Job
slot as a `cluster` one; a row's own `target` is what decides `gpu=` at launch,
so the queue cannot launch an A4000 Job as a CPU one. `GPU_ROUTE_FALLBACK` was
REMOVED from the config: its docstring described a quota-slot fallback that no
code implemented, and a route only an operator sets does not need a third
value. Unknown values still mean the default.

## 2. The GPU job

- [x] 2.1 `web/jobs.py`: `build_job_manifest(..., gpu=False)`. When true, add
      `nvidia.com/gpu: 1` to BOTH `requests` and `limits`, the
      `nodeSelector`, the four thread env vars set to the CPU limit, and keep
      `activeDeadlineSeconds`. Pure, as today.
- [x] 2.2 The GPU image: reuse `Dockerfile.bench-cuda`'s already-built image
      line; the solver image needs `MSE_SOLVER_BACKEND=cuda` in the manifest
      env, not a new image.
- [x] 2.3 `manifests/mse/power-cap.yaml`: `requests.nvidia.com/gpu: '1'`.
      Tracked only; the lead applies it.

**Done notes (2.x).** No new image was needed (2.2 was a check, not a build):
the manifest sets `MSE_SOLVER_BACKEND=cuda` and strips the CPU-only
`MSE_CPU_MAX_TETS`, which describes a machine the GPU Job is not running on.
The Job stays 4 vCPU / 14Gi, and the quota stays 5 CPU / 18Gi with only the GPU
count raised 0 -> 1. An earlier version of this change shrank the Job to 2.5
vCPU / 6Gi inside a 3 CPU / 7Gi quota, on the reasoning that 4/14Gi "only fitted
because the GPU count was 0". That reason is false: CPU and memory quota do not
depend on the GPU count. Job 4 + app 0.5 = 4.5 <= 5, and 14 + 1 = 15 <= 18Gi, so
the original size was already admissible -- and 3/7Gi left no headroom for an
app rollout beside a solve. Thread vars are the integer part of the CPU limit
(4), derived from the manifest's own limit rather than hardcoded.

## 3. The honest failure

- [x] 3.1 Confirm the existing GPU-OOM wording reaches a cluster GPU job's row.
      The Jetson path builds it in `worker/`; the cluster path reads the Job's
      failure output in `web/`. Route the A4000 case through the same sentence
      (spec: gpu-solves requirement 3).

**Done note (3.1).** It did NOT reach it honestly. `FAILURE_MESSAGES["OOMKilled"]`
said "Try a simpler object", which is false advice for a card that ran out of
memory -- and the A4000 is a tighter budget than the machine it replaced, so it
is the failure the owner meets first. The sentence now names the machine.

## 4. Tests (tier 1 only)

- [x] 4.1 A GPU manifest has `requests == limits` for cpu, memory and GPU, a
      positive `activeDeadlineSeconds`, and the A4000 nodeSelector.
- [x] 4.2 Thread env vars equal the CPU limit in the GPU manifest.
- [x] 4.3 `MSE_GPU_ROUTE`: `cluster` (and unset) -> cluster-gpu; `jetson` ->
      jetson; a nonsense value -> cluster-gpu.
- [x] 4.4 The manifest states `requests.nvidia.com/gpu: '1'`.
- [x] 4.5 An out-of-memory GPU failure produces the plain sentence, not
      "contact support".
- [x] 4.6 The CPU manifest is unchanged by the GPU support (no card, no
      nodeSelector, `cpu` backend).

**Done notes (4.x).** Each new test was mutation-checked: dropping `gpu=` at
the launch site turns 4.1 red, and dropping `CLUSTER_GPU` from
`CLUSTER_TARGETS` turns 4.1 and 4.5 red. Two existing tests pinned the OLD
contract and were updated to state the new one rather than deleted: the 10 mm
box's default target (Jetson -> A4000) and the Job's resource size (now read
from `CFG`, so a future quota move does not fail them for the wrong reason).
The `FakeDb`/`FakeLauncher` doubles were corrected to the real signatures --
a fake `next_queued(target=str)` would never return a `cluster-gpu` row, so
every routing test above would have passed by launching nothing.

## 5. Deployment (the lead, not this seat)

- [ ] 5.1 Raise the live quota 0 -> 1 and apply the Job manifest.
- [ ] 5.2 Smoke a SMALL mesh on the A4000 and read the log; verify a real pod
      starts (a server dry-run is not admissibility).
- [ ] 5.3 Rollback: `MSE_GPU_ROUTE=jetson` and re-apply the quota at 0.
