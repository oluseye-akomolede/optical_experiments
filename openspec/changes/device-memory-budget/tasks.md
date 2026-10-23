# Tasks: the memory and cell ceilings follow the device

Owner request (2026-10-23, via the lead): "a run should use all available
resources... size limits should follow the device... for every run (product
and bench), though the web requests are queued in the DB and the worker
processes them eventually... free at admission is the honest number."

## 1. The device's own budget

- [x] `fdtd.admission_budget(xp=None, *, margin, free=True, cgroup_paths,
      meminfo)` returns the ceiling of the device the process runs on.
      cupy: `Device().mem_info` free + the default pool's free blocks, capacity
      the device's total. Otherwise the cgroup limit (v2 `memory.max`, v1
      `memory.limit_in_bytes`; `max` and ~2^63 read as "no limit") when set, else
      `MemAvailable` from `/proc/meminfo`.
- [x] The ceiling is the binding figure x `ADMISSION_MARGIN` (0.9), and the
      budget carries `free`, `capacity`, `source`, `margin` and a readable
      `where` phrase naming the figure the ceiling was actually taken from (when
      a container's free memory binds, the phrase says so and gives the limit it
      is under, rather than naming the limit alone).
- [x] `free=False` gives the capacity-bound ceiling -- the worker's budget, so a
      device that is merely busy defers a run instead of refusing one it could
      hold.
- [x] `fdtd.bytes_per_cell()` measures the cheapest per-cell cost from
      `memory_estimate`'s own accounting (two grids differing only in cell
      count), not retyped from its formula.

## 2. The check follows the budget

- [x] `fdtd.check_limits` takes `budget`: `None` reads `admission_budget()` at
      call time; `fdtd.NO_DEVICE` checks only steps and seconds; an explicit
      `max_bytes` overrides the budget. `MAX_STEPS` / `MAX_SECONDS` keep their
      values and their meaning.
- [x] `fdtd.MAX_CELLS` and `fdtd.MAX_BYTES` removed.
- [x] The cell ceiling is derived from whichever byte ceiling applied (the most
      cells that fit it at `bytes_per_cell()`), so it fires on an estimate whose
      `bytes` and `cells` disagree rather than on a fixed product number.
- [x] A memory refusal names the estimate, the ceiling, its source and the
      margin, and keeps the metal-walls / dx / fewer-frames levers.

## 3. The callers

- [x] `worker/td_solve.py`: checks `admission_budget(xp, free=False)` on the
      device, and defers (`JetsonBusy`) when the estimate exceeds the free
      memory at admission x the same margin.
- [x] `web/td_form.py`: prices the run (bytes, steps, seconds) and passes
      `NO_DEVICE` -- the control plane cannot read the run's device.
- [x] `scripts/paper1_nanowire.py`: `run_limits` passes `None` for cells and
      bytes unless the flags set them (the device's budget applies), and the
      flags may now also set a ceiling below the device's.

## 4. Tests

- [x] `tests/test_fdtd_estimate.py`: budget sources (GPU free + pool, cgroup
      binding over host memory, no limit -> `MemAvailable`, capacity budget),
      the A4000 case (16 GB card, 14.5 GB free admits the 11.55 GB record and
      refuses 13.2 GB), a full container naming its free memory, the derived
      cell ceiling firing on a split estimate, the budget read at call time,
      `NO_DEVICE`, and the explicit-ceiling refusals. 28 passed.
- [ ] `tests/test_td_form.py`, `tests/test_td_solve.py` (via `test_worker.py`),
      `tests/test_paper1_nanowire.py`: update the scenarios that encoded the
      8,000,000-cell / 3.0 GB constants to the device budget (the page prices
      and admits; the worker refuses on capacity and defers on free memory).
- [ ] `tests/test_backend.py`, `tests/test_routing.py`, `tests/test_fdtd_memory.py`:
      any use of the removed constants.

## 5. Spec and proposal

- [x] Spec delta: the product's step and time ceilings unchanged; the page
      prices memory and the device decides; the memory and cell ceilings follow
      the device; the two fixed-ceiling requirements removed with migration.
- [x] `proposal.md` records the owner's request, the change and its risks.

## 6. Before handoff

- [ ] `openspec validate device-memory-budget --strict` and
      `openspec validate --specs --strict` (report both counts).
- [ ] The affected suites green, run one at a time at tier 1.
- [ ] Commit locally (no push; the lead pushes), then hand off to the lead with
      the sha and the test counts.
