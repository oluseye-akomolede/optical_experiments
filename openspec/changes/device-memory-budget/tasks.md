# Tasks: device memory budget

## 1. Spec and proposal
- [x] 1.1 Proposal: the owner's request, the two fixed numbers replaced, the
      decision that the page prices but does not refuse memory.
- [x] 1.2 Spec delta: MODIFIED "A run SHALL be refused above its limits before
      it is queued" (all eight scenarios carried; the two refusal scenarios now
      read at the device, because that is where the ceiling exists) and ADDED
      "The memory ceiling SHALL be the run's device budget, read on the device".

## 2. The budget in `fdtd`
- [x] 2.1 `fdtd.admission_budget(xp=None, margin=0.9)`: cupy's device free
      memory plus the default pool's cached blocks, else the cgroup v2/v1
      limit, else `MemAvailable`; ceiling = min(limit, free) x margin; the dict
      names the figure that bound it (`where`) for the refusal message.
- [x] 2.2 `fdtd.bytes_per_cell()` measured from `memory_estimate`'s own
      accounting (the difference of two pure-metal grids), so the derived cell
      ceiling cannot drift from the bytes it follows.
- [x] 2.3 `fdtd.check_limits` takes `budget=`; `max_cells` defaults to
      `max_bytes // bytes_per_cell()`; `fdtd.NO_DEVICE` means "this process is
      not the run's device" and skips the two device ceilings. `MAX_CELLS` and
      `MAX_BYTES` removed; `MAX_STEPS` and `MAX_SECONDS` kept.

## 3. Callers
- [x] 3.1 `worker/td_solve.py`: `check_limits(est, budget=admission_budget(xp))`
      refuses a run that can never fit the device; the busy gate now applies
      the same margin to the free-memory reading it defers on.
- [x] 3.2 `web/td_form.py`: prices the run and passes `NO_DEVICE`; the page no
      longer refuses on bytes or cells.
- [x] 3.3 `scripts/paper1_nanowire.py`: `run_limits` returns `None` for cells
      and bytes unless the flags set them (explicit ceilings, no product floor);
      the ladder prices against the device budget unless a flag pins one.

## 4. Tests
- [x] 4.1 `tests/test_fdtd_estimate.py`: the device-budget scenarios (cupy free
      plus pool, cgroup v2/v1 limits, no limit, no GPU, the derived cell
      ceiling firing on an inconsistent estimate, an explicit ceiling's
      precedence).
- [x] 4.2 `tests/test_td_form.py`: the page prices without refusing, and never
      reads a device budget (a poisoned reader would raise).
- [x] 4.3 `tests/test_paper1_nanowire.py`: `run_limits` ceilings and the
      ladder's refused rung, with an explicit `--max-bytes` so the verdict does
      not depend on the pricing host's memory.

## 5. Verification
- [x] 5.1 Probes: `test_fdtd_estimate`, `test_td_form`, `test_backend`,
      `test_routing`, `test_worker`, `test_fdtd_memory` green; the fast
      `test_paper1_axis_probes` + `test_fdtd_estimate` pair 43 passed.
- [x] 5.2 Bench suite (`test_paper1_nanowire.py`) green: 43 passed (25:04).
- [x] 5.3 `openspec validate device-memory-budget --strict` valid, and
      `openspec validate --specs --strict` unchanged at 12 passed, 0 failed.
- [ ] 5.4 No real solve: this change moves no physics and no grid; the GPU
      reading is exercised by a fake device, which is tier-1 and stated here as
      such.
