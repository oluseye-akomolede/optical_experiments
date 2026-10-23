# Size limits follow the device the run lands on

## Why

The owner asked (Slack, 2026-10-23 04:42Z): "We should raise the limit generally. A run
should use all available resources." Today the FDTD solver refuses any run over two fixed
numbers -- 8,000,000 cells and 3.0e9 bytes -- regardless of the machine. Those numbers
describe no real device: they were sized for the Jetson, and they refuse a run the bench's
16 GB A4000 can hold (the paper-1 dx-5 nm record estimates 11.55 GB) and admit runs a
smaller box could not. A fixed ceiling cannot be right for a Jetson, an A4000 and a k8s
container at once; the device's own memory can.

## What changes

`fdtd.admission_budget(xp=None)` reads the memory of the device the calling process runs
on and returns the admission ceiling:

- **GPU (`xp` is cupy):** the device's own memory, `cupy.cuda.Device().mem_info`, plus the
  blocks cupy's pool already holds (free to reuse, not counted by the driver).
- **container:** the cgroup memory limit when one is set (`/sys/fs/cgroup/memory.max`, or
  v1's `memory/memory.limit_in_bytes`), because that is the memory the run may hold even
  when the host has more.
- **no cgroup limit:** the kernel's `MemAvailable`.

The ceiling is the binding figure times `ADMISSION_MARGIN` (0.9). Ten per cent is the
margin because the estimate already errs high in its own terms (frames at the requested
count, transients, the pool) but not against the *other tenants* of a shared device: the
A4000s are shared with the minecraft server and the Jetson's memory is shared with its OS,
so a run that plans to use every free byte dies when anything else grows. A tenth of the
device is the room that keeps a run that was admitted from failing at step 30,000 for a
reason outside its own accounting.

`fdtd.check_limits` takes the budget (`budget=`) instead of a fixed byte ceiling:
`max_bytes=None` now means "the device's budget", and the refusal names the figure it
compared to, where it came from and the margin (for example "over the admission budget of
12.9 GB -- the GPU's free memory at admission (14.3 GB free) x a 0.9 margin"). The cell
ceiling follows the budget instead of being an independent number: it is the most cells
that fit the budget at the cheapest configuration's per-cell cost, and that cost is
measured from `memory_estimate`'s own accounting (two grids differing only in cell count)
rather than retyped from its formula. It therefore fires only on an estimate whose `cells`
and `bytes` disagree -- an invariant check on the estimator, not a second opinion about the
machine. `MAX_CELLS` and `MAX_BYTES` are removed.

`MAX_STEPS` (20,000) and `MAX_SECONDS` (30 minutes) stay as they are: they guard a run
running away in time, not a device's size, and a bigger machine does not make a 47-minute
run part of the product.

The web app runs on the control plane, not on the device, so it cannot read the solver's
memory: it checks the device-independent limits (steps, seconds) at form time and passes
`fdtd.NO_DEVICE`, so the page no longer refuses a run on a memory figure it cannot know.
The worker, which runs on the device, applies the budget: it refuses a run the device could
never hold, and defers one that would fit but does not fit *now* (its existing busy
message), rather than starting it against a full card.

The paper-1 bench's `--max-cells` and `--max-bytes` remain, as explicit overrides rather
than permissions: with neither flag a bench run is admitted against the device's budget, and
a flag sets the ceiling it names (it may now also set one below the device's, since the
product no longer has a constant for it to raise from). `--max-steps` and `--max-seconds`
keep their only-raise rule against the product's two remaining constants.

## Cost and risk

- A run refused for memory now reaches the worker before learning it. The page still
  prices the run (bytes, steps, seconds) and says the device decides; the worker's refusal
  is the same sentence the page used to show, plus its source. Layout runs before the
  estimate in the worker, so a refused job costs its layout time, not its solve.
- The budget is read at admission, so two runs admitted together against one idle device
  can still collide; the worker's existing busy deferral is what serialises them, and the
  margin is what keeps a single run from using the whole card.
- `MAX_CELLS`/`MAX_BYTES` removal touches every caller and test that named them; the
  refusals keep their levers, and the spec's scenarios are replaced rather than dropped.
