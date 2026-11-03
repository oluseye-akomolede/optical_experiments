# Run owner uploads' solves on the cluster's A4000s

## Why

The owner (Slack, 2026-11-02 16:26Z), verbatim:

> "Use my A4000 cluster GPUs for solving instead of solving on the jetson, which is
> ostensibly an edge device. There should be 3 idle right now"

The Jetson is an edge device: one GPU, shared with nothing else, and a job that
does not fit its 8 GB has to be refused or run slowly on CPU. The cluster has
three idle RTX A4000s (16 GB each) on `sigma-worker1` and `sigma-worker2`; one is
already in use by `minecraft-server`. A larger-card solve is the difference
between an upload the product can take and one it has to turn away.

## What changes

Today a job is routed by ONE comparison: `est_tets >= gpu_min_tets()` sends it to
the Jetson's pull worker, otherwise it is a cluster CPU Job. The GPU route is a
single serial worker on somebody's desk.

The change makes the GPU route a **bounded Kubernetes Job** that requests one
A4000, reusing the CUDA image (`Dockerfile.bench-cuda`), the node pinning and the
`requests == limits` shape already proven by `manifests/bench/paper1-a4000-job.yaml`.

The work splits into three parts:

1. **Routing.** A new target, `cluster-gpu`, chosen when the estimate wants the
   GPU. The Jetson stops being the automatic GPU destination but stays as a
   fallback route; it is not removed (the owner's row says so explicitly).
2. **A GPU job manifest.** `build_job_manifest` grows a GPU variant: one
   `nvidia.com/gpu`, `requests == limits`, threads matched to the CPU limit, and
   an `activeDeadlineSeconds`. The GPU quota is raised 0 -> 1 **by the lead at
   deploy time**; this change edits the tracked `manifests/mse/power-cap.yaml`
   and never applies a live quota.
3. **The honest failure.** A job too big for 16 GB must end with the same
   human sentence the Jetson path gives, never "contact support".

## Impact

- Affected specs: `deployment-bounds` (the GPU quota bound), `job-lifecycle`
  (the GPU target and its terminal states). One new capability spec,
  `gpu-solves`.
- Affected code: `web/db.py` (a target), `web/app.py` (the routing decision),
  `web/jobs.py` (the manifest), `manifests/mse/power-cap.yaml`.
- Not removed: the Jetson worker and its fallback. This change adds a route; it
  does not retire one.

## Out of scope

- Running a GPU job from this seat. No GPU runs are made here (rig rule); the
  lead smokes a small mesh after deploy.
- Raising the live quota. That is the lead's action, after the owner's decision
  recorded on the row.
- Multi-GPU solves. The owner chose one GPU, one solve at a time.
