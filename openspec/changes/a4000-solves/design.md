# Design: how a job picks A4000, Jetson, or cluster CPU

## The decision to make

`web/app.py` today makes one comparison:

```python
target = DB.JETSON if est_tets >= CFG.gpu_min_tets() else DB.CLUSTER
```

That conflates two questions. "Does this need a GPU?" is a property of the
problem. "Which GPU machine runs it?" is a property of the fleet. The second
question now has two answers, so the first has to be asked separately.

## Routing rule

```
est_tets >= gpu_min_tets()          -> the GPU route
     MSE_GPU_ROUTE == "cluster"     -> cluster-gpu   (the A4000 Job)  [default]
     MSE_GPU_ROUTE == "jetson"      -> jetson        (the pull worker)
     MSE_GPU_ROUTE == "fallback"    -> cluster-gpu, falling back to jetson
                                       if the GPU quota has no free slot
otherwise                            -> cluster       (the CPU Job)
```

`MSE_GPU_ROUTE` is the single setting that moves the GPU route, and it is read
from the environment like every other knob in `web/config.py`. Three values, not
a boolean, because the honest requirements need a third state on rollout day:

- **`cluster` (default).** Every GPU job asks the A4000 route. This is what the
  owner asked for.
- **`jetson`.** The old behaviour, kept reachable so a rollback is one env
  change and not a revert.
- **`fallback`.** Ask for the A4000, but if the A4000 quota slot is occupied,
  route to the Jetson instead of failing. This is the transition state the row
  implies ("The Jetson stays a backup route until the A4000 path has run real
  uploads").

The default is `cluster` rather than `fallback` deliberately. A silent fallback
to the Jetson would make "the A4000 path works" unverifiable from the outside:
every job would succeed somewhere and the owner would never learn which. The
transition is a decision (set `fallback`), not a default that hides itself.

## Why one GPU, not up to three

The owner was still choosing between 1 and 3 when the row was written. The
quota raise is a single number (`requests.nvidia.com/gpu: '1'`), and the job
manifest asks for one. Concurrency is therefore 1, which is also what
`count/jobs.batch: 1` already enforces -- a second GPU job cannot be admitted
while the first exists, so asking for one GPU keeps the two limits consistent
rather than having the quota be the only thing that serialises them.

If the owner later wants 3, two numbers move together (the quota and the count
the launcher permits). They are not moved here.

## The 16 GB bound

An A4000 has 16 GB. The Jetson has 8 GB (less, shared). A job between those two
sizes is exactly the one this change is for -- it is refused today. Two things
must hold:

1. The job requests what it needs. Memory is derived from the estimator, as the
   CPU path already does, but the GPU path's request is bounded by the card.
2. When a job exceeds the card anyway (the estimate is an estimate), the failure
   is the SAME user-facing sentence the Jetson OOM path gives: what happened and
   what to do about it. Never "contact support", never a bare traceback.

Requirement 3 of the spec names both.

## What does NOT change

- The solver image and its CLI. The A4000 path runs the same `mse_maxwell_solver`
  with `MSE_SOLVER_BACKEND=cuda` instead of `cpu`; cuDSS, `EigenError` and the
  `namenames_an_out_of_memory` fallback are all already in the image.
- The Jetson worker. It keeps polling, keeps its route, and remains the answer
  to `MSE_GPU_ROUTE=jetson`.
- The CPU cluster path, byte for byte.

## The quota is a deploy-time action, not a commit

`manifests/mse/power-cap.yaml` is tracked for reference; the comment in it says
plainly that the live quota is set by control-operator and never raised from
here. This change sets the tracked value to `'1'` so the file and the cluster
agree, and the LEAD applies it after review. A test asserts the manifest says
`'1'`; no test and no code path applies it.

The known trap, recorded on `paper1-a4000-job.yaml`: `kubectl apply
--dry-run=server` answers "created" even while the quota refuses the pod,
because ResourceQuota admission is exempt from dry-run. A green dry-run is not
admissibility. The lead verifies with a real apply and a pod that starts.
