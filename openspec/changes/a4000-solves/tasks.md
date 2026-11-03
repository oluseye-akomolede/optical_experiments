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

## 4b. The two blockers the lead found before deploy

- [x] 4b.1 `MSE_SOLVER_BACKEND` comes from `mse_backend.GPU`, not a literal.
      The test compares it with `mse_backend.BACKENDS`; mutating back to
      `"cuda"` turns it red.
- [x] 4b.2 A separate `MSE_SOLVER_GPU_IMAGE`, used for GPU jobs and never
      falling back to `MSE_SOLVER_IMAGE`; a GPU job with no CUDA image is
      refused before the Job is created. The app still starts without it.
- [x] 4b.3 The false comment ("The image is built with the CUDA stack") is
      gone.
- [x] 4b.4 `dockerfiles/maxwell_solver/Dockerfile.gpu` + `requirements-gpu.txt`:
      a CUDA 12.6.3 base matching the A4000 nodes, with a BUILD-TIME probe that
      calls the solver's own `mse_backend._missing_gpu_package()` so a GPU image
      missing the stack cannot reach Harbor.
- [x] 4b.5 `nvmath-python==0.9.0` **and** `nvidia-cudss-cu12==0.7.1.6` both
      pinned explicitly, read from the Jetson by the lead (04:18Z). The `cu12`
      extra is deliberately NOT used: it also drags in nvidia-cuda-runtime-cu12
      beside the base image's own CUDA 12.6.
- [x] 4b.6 `dockerfiles/maxwell_solver/gpu_stack_probe.py`, run by the build:
      the solver's own import probe, then a `ctypes` load of the cuDSS shared
      object. Loading needs no GPU, so a stackless image fails at BUILD time.
- [x] 4b.7 Probe mutation-checked in both directions on the dev VM: with the
      cuDSS distribution absent it refuses with the "pin it" message; with
      0.7.1.6 installed it locates the library and tries to load it.

**Measured fact that broke the first probe:** `nvidia-cudss-cu12` 0.7.1.6 is
DATA-ONLY -- it installs `nvidia/cu12/lib/libcudss.so.0` and no importable
`nvidia.cudss` package, so `find_spec("nvidia.cudss")` fails on a correct
install and the first probe would have refused a good image. The probe now
locates the library through the `nvidia-cudss-cu12` DISTRIBUTION instead. The
wheel also does not declare its `libcublas.so.12` dependency (the CUDA base
supplies it); because the probe actually LOADS the library, that class of miss
fails the build -- and did, locally, which is how the dependency was found.

**A second probe bug, caught by the first real image build.**
`import_probe()` asked `mse_backend._missing_gpu_package()` and treated any
non-empty return as a failure. But that helper returns the string `"the CUDA
stack"` as a LABEL for an error message when NOTHING is missing, so the probe
refused a correct image, with the self-contradicting line "the CUDA stack is not
importable". It now imports `mse_backend._GPU_PACKAGES` one name at a time and
reports the name that actually failed; both directions are tested on the dev VM.
The lesson generalises: a helper whose "nothing wrong" case shares a type with
its "something wrong" case cannot be used as a boolean.

**The corrected probe passed a real image build** (openrig VM, repo-root build
context): `GPU STACK IMPORTS OK`, then `cuDSS LOADS OK: libcudss.so.0`, image
tagged, exit 0. That is the probe's first clean run on the image it guards --
before it, the only observed run was the buggy one that refused a good build.
One build-context trap found on the way: the COPY paths are repo-root-relative,
so passing `dockerfiles/maxwell_solver/` as the context fails; it must be the
repo root.

**Done notes (4b).** The old test asserted `== "cuda"`: it encoded the bug
rather than catching it, so it was replaced by a membership check against
`mse_backend.BACKENDS`. A first attempt made `_launcher` read the GPU image
strictly, which failed an existing test that builds the app without one --
that was a real deployment risk (a missing GPU setting would stop the site
starting) and the read was made lenient, with the refusal left at the launch
boundary where the Job is created.

## 5. Deployment (the lead, not this seat)

- [ ] 5.1 Raise the live quota 0 -> 1 and apply the Job manifest.
- [ ] 5.2 Smoke a SMALL mesh on the A4000 and read the log; verify a real pod
      starts (a server dry-run is not admissibility).
- [ ] 5.3 Rollback: `MSE_GPU_ROUTE=jetson` and re-apply the quota at 0.

**Lead's red-path check (2026-11-03, merged image).** In the built image, run as root
(as the image's own user, `pip uninstall` is refused and the package silently stays, so
a non-root mutation proves nothing): with `nvidia-cudss-cu12` uninstalled the probe exits 1
with "no nvidia-cudss-cu12 distribution ..."; with `nvmath-python` uninstalled it fails with
"'nvmath' is not importable". The intact image prints both OK lines and exits 0.
