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
- `--max-cells`, `--pad-nm` and `--pml-nm` on `scripts/paper1_nanowire.py`: the bench path's own
  third ceiling, and the open boundary named as a distance instead of a cell count. Both are
  bench-only — the web form and the worker build their own request and call the planner with
  neither, so the product's own boundary and its own 8,000,000 cells are untouched.

## Pricing (the product's own estimator, nothing run)

`scripts/paper1_nanowire.py --estimate`, both records at dx 10 nm with the open boundary held
at r20's own distance: 320 nm of pad and 200 nm of absorber, which is the product's own 16/10
cells at dx 20 and reproduces r20's grid exactly — the same 2318.8 × 2318.8 × 2177.0 nm box.

Holding the boundary is what attached the number to a run. `fdtd_layout.OPEN_PAD_CELLS` is a
cell COUNT, so the product's pad is 320 nm at dx 20 and 160 nm at dx 10: a dx-10 record on the
defaults moves the boundary at the same time as the grid, and no difference from r20 could then
be attributed to either one (this scene's own record: 200 nm against 400 nm of pad moved F by
16 %). Held, the CW check is **14,048,300 cells, 39,650 steps, 1.63 GiB, 13,662.4 s** at the
estimator's own rate, i.e. 17,925 s (4 h 58 m) at the Orin's measured factor 1.312; the
ring-down locator is the same grid with a shorter record — 21,918 steps (its own 1000 nm
centre), 7,552.4 s = 9,909 s (2 h 45 m).

14,048,300 cells is over `fdtd.MAX_CELLS = 8,000,000` by 6,048,300 (75.6 %), so the record needs
a third bench ceiling — `--max-cells`, which the lead authorised on 2026-10-16 and this change
adds — rather than the rung the first pricing offered: at the held boundary dx 12 nm is
8,952,300 cells and still over the cap, and the nearest rung that fits is dx 15 nm (5,234,775
cells, 5,091.0 s = 6,679 s on the Orin).

The tables and the derived Orin-rate seconds are in `tasks.md`.

## What this change does not do

- It does not raise the product's cell ceiling. `--max-cells` is a third bench knob, threaded
  exactly like `--max-steps` and `--max-seconds`: read only from this script's own parser, with
  `fdtd.MAX_CELLS` and every product caller's bare `fdtd.check_limits(...)` unchanged.
- It runs nothing on the cluster and pushes nothing: the Job is a manifest, and the image is
  built and smoke-tested on the VM only.
- It touches no product code, no DAG, no web app and no database.

## Impact

Two new files, plus the three bench flags above. The Job occupies the namespace's single Job
slot under the `mse-power-cap` ResourceQuota (1 Job) for up to its deadline, so it is launched
when no owner upload is solving.
