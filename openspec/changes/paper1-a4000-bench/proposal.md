# paper1-a4000-bench

## Why

The owner asked for the dx-10 grid check to run on an available A4000 rather than the Jetson
(18:43Z, relayed by the lead in the queue row at 18:45Z). The cluster route needs its own
artifacts for two reasons: no image of ours carries a CUDA runtime (the web image is
`python:3.12-slim`, and the FDTD path imports cupy lazily, so cupy is never importable there),
and the Jetson bench is one GPU behind a flock'd lock whose only wall bound is the launcher's
`MSE_CLOCK`.

## What this change adds

- `dockerfiles/maxwell_solver/Dockerfile.bench-cuda`: a pinned CUDA-12 runtime image carrying
  exactly the files `scripts/paper1_nanowire.py` imports, with cupy/numpy/scipy/trimesh pinned
  to the versions the bench path has been run with.
- `manifests/bench/paper1-a4000-job.yaml`: one bounded `batch/v1` Job on a single A4000, with
  the wall bound as the Job's own `activeDeadlineSeconds` and the result surviving in the pod
  log for 7 days.
- A new capability spec, `openspec/specs/cuda-bench`.

## Pricing (the product's own estimator, nothing run)

`scripts/paper1_nanowire.py --estimate`, the CW check (993 nm, 100 periods) and the ring-down
locator (1000 nm ± 100 nm, 30 tail periods). Both asked rungs — dx 10 nm with the Orin's own
pad of 600 nm — are **9,166,668 cells: over `fdtd.MAX_CELLS = 8,000,000` by 1,166,668
(14.6 %)**, so neither can be launched as things stand, with or without the bench flags: the
two bench knobs raise steps and seconds only (`fdtd.check_limits(max_steps, max_seconds)` in
`dockerfiles/maxwell_solver/fdtd.py`; the cell branch reads the module constant inline).

The tables and the derived Orin-rate seconds are in `tasks.md`. The nearest rung that fits
today is dx 12 nm at the same 600 nm pad: 6,397,264 cells, 39,650 steps, 0.79 GiB, 6,221.5 s
at the table's own rate.

## What this change does not do

- It does not raise the product's cell ceiling. A third bench knob (`--max-cells`, threaded
  like the other two, with the product path still bare) is **proposed** in `tasks.md` and left
  for the lead's word: it is a reviewed code change, not a silent one.
- It runs nothing on the cluster and pushes nothing: the Job is a manifest, and the image is
  built and smoke-tested on the VM only.
- It touches no product code, no DAG, no web app and no database.

## Impact

Two new files. The Job occupies the namespace's single Job slot under the `mse-power-cap`
ResourceQuota (1 Job) for up to its deadline, so it is launched when no owner upload is
solving.
