# Where the solve-memory model came from (task 1.7)

`mse_electrostatics.estimate_memory` prices a solve before it allocates, so
`solve()` can refuse a mesh that would not fit. The constants it uses
(`DIRECT_MEMORY_TABLE`, `AMG_MEMORY_POWER_LAW`) are **measured**, not fitted here.

## How it was measured

`solve_memory_bench.py` runs one solve per size in a child process, records the
peak RSS growth over the empty-process baseline, and reports the bytes per free
unknown at each size. It fits both a power law and a linear law and prints each
one's worst relative error, so a bad model cannot be quoted by accident.

The measurement was run by **mse-lead** as a bounded Kubernetes Job:

| field | value |
|---|---|
| node | `sigma-worker2` (x86_64) |
| resources | 2 cpu, 6 GiB |
| image | `mse-maxwell-solver-gpu:28997f1`, **no GPU**, from `harbor.arcadia-ecs.local/mode-search-engine/` — the host every committed Job manifest uses (`manifests/bench/paper1-a4000-job.yaml`, `manifests/mse/deployment.yaml`) |
| code | `b3bbdc1` (branch `electrical-simulation-1.7`) |
| script | `dockerfiles/maxwell_solver/solve_memory_bench.py`, `--paths direct,amg` |

> **Provenance note.** The figures below are the lead's reported output from that
> Job. They were not captured into this repository directly — the Job had been
> cleaned up by the time this file was written, so the spec above is reconstructed
> from the row (`qitem-20261104005251-a44f6190`) and the resource limits instead
> of copied from a live manifest. If the model is ever re-measured, replace this
> section with the actual job yaml and its log.

**Why the cluster and not the Jetson:** the Jetson was unreachable for the whole
row (DNS resolves to 100.74.195.113; 0/2 pings and every SSH path times out at the
banner exchange). Solves now run on the cluster, so these are the right device's
numbers anyway.

## What was measured

### AMG-CG (sizes 12–30)

```
bytes = 12812 * n_free^0.91      worst point 2.0 % off
n_free range 1,859 .. 27,869     ~5 kB/unknown at the top of the range
```

A power law **fits** this path. It is applied as written over the measured range;
above 27,869 unknowns it is multiplied by `AMG_EXTRAPOLATION_MARGIN` (1.5) and the
estimate reports `extrapolated: True`.

### Direct LU (sizes 12–48)

| n_free | peak RSS growth |
|---:|---:|
| 1,859 | 10.9 MiB |
| 6,137 | 32.2 MiB |
| 14,375 | 122 MiB |
| 27,869 | 339 MiB |
| 47,915 | 723 MiB |
| 75,809 | 1,444 MiB |
| 112,847 | 4,797 MiB |

These are **MiB**, the bench's own column (`VmHWM` is KiB, printed as
`grow_kb / 1024`). `DIRECT_MEMORY_TABLE` stores them as `MiB × 1_048_576` bytes.
Reading them as MB would make every direct estimate 4.63 % **low** — the unsafe
direction for an OOM guard.

**No law fits this path.** A power law misses its own worst point by 27 %, linear
by 277 %. Bytes per unknown climb from 6,148 B to 44,574 B across the table
(5.86 → 42.5 kB), with a **2.2× jump** in the per-unknown rate between the last two
points. Over that same last step the *total* grows **3.3×** (1,444 → 4,797 MiB)
while the unknowns grow 1.49× — both statements are true of different quantities,
so say which one is meant. SuperLU's fill depends on the elimination ordering,
which changes strategy discretely with size. That is an algorithm switching, not a
curve.

So the direct path is **interpolated in log-log space between measured points**,
and **refused above 112,847 unknowns** (`EstimateError`, recommending AMG) rather
than extrapolated.

### The VM run that came before

The first attempt ran on the VM at four tiny sizes (1.9k–27.9k unknowns) and found
the per-unknown rate rising 6.2 → 12.9 kB. Neither law fit (33.8 % and 116.5 %
worst error) and the bench printed `usable model: NONE`. That was the correct
answer: at VM sizes the fill had not settled, and only a range wide enough to
cross SuperLU's strategy changes could separate the two paths' behaviour. The
cluster run is what made them distinguishable.

## Caveat carried forward

The crossover constant `AMG_NODE_CROSSOVER` (5 × 10⁵) sits **above** the direct
table, so it cannot be checked against a measurement on either side. It remains a
named, untested speed choice — see design.md §5. Every graded mesh the product
actually builds is above the table and takes AMG regardless, which makes the
crossover academic in practice but does not make it verified.
