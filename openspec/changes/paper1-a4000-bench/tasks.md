# paper1-a4000-bench — tasks

The lead's three steps (queue row 2026-10-16 18:44Z), then the lead's 19:0xZ decision to hold the
open boundary at r20's own distance. One commit each, local only; the lead merges and pushes.
Nothing runs on the cluster until review and the lead's go.

## 1. The CUDA bench image

- [x] Import closure derived from the script, not guessed: `.venv-mse/bin/python -I -c` loads
      `scripts/paper1_nanowire.py` with the same two `sys.path` entries it inserts itself, then
      lists every loaded module whose file is in the repo — `fdtd`, `fdtd_grid`, `fdtd_layout`,
      `fdtd_voxel`, `mse_common`, `mse_geometry` (all under `dockerfiles/maxwell_solver/`).
      Those six plus the script are what the image copies, in the repo's own relative layout.
- [x] Base tag pinned at or below the node's own CUDA: `nvidia/cuda:12.6.3-runtime-ubuntu24.04`
      (the A4000 nodes report `nvidia.com/cuda.runtime-version.full=12.7` with driver 565.57.01).
- [x] Packages pinned to the versions the bench path was RUN with — read on the Jetson:
      `cupy 14.2.0`, `numpy 2.2.6`, `scipy 1.15.3`, `trimesh 5.1.0` (the venv that produced r20).
- [x] Built on the VM with `DOCKER_CONFIG=~/.docker-mse`; **not pushed** (the lead pushes).
- [x] Smoke-tested on the VM with no GPU (`docker run --cpus=2 --memory=2g`): `--help`; a
      `--tiny --xp numpy` run through the script's own path; and `import cupy` inside it.
      Evidence in the queue row (the three outputs, verbatim).

## 2. The A4000 Job

- [x] `manifests/bench/paper1-a4000-job.yaml`: ns `mode-search-engine`, nodeSelector
      `nvidia.com/gpu.product=NVIDIA-RTX-A4000` and nothing else — as the row asked: which of
      the A4000 nodes carries it is left to the scheduler and the card's own allocatable, and
      the file says where to add a `kubernetes.io/hostname` if one is to be pinned by hand.
      **This is not a Job that can run today.** `mse-power-cap` carries
      `requests.nvidia.com/gpu` hard 0 / used 0 (live: the same line as
      `manifests/mse/power-cap.yaml`), so the pod is refused with `exceeded quota`; raising it
      0 → 1 is the owner's decision and the lead's apply, and the manifest's own header says
      so. `kubectl apply --dry-run=server` answered `created (server dry run)` here, and that
      is not evidence of admissibility: ResourceQuota admission is exempt from dry-run.
      Limits == requests (`nvidia.com/gpu 1`, `cpu 2`, `memory 8Gi`), OMP/MKL/OPENBLAS/NUMEXPR 2.
- [x] Memory justified from the pricing, with the boundary held: the priced dx-10 grid is
      **1.63 GiB** by the estimator's own storage for that record (1,754,244,508 bytes =
      1,673.0 MiB), so 20.4 % of the 8Gi request (8,192 MiB: the request is 4.9× that estimate)
      and 10.2 % of the card's 16,376 MiB. Both denominators are named because an earlier note
      here said "14.0 % of the request / 7.0 % of the card" from the *unheld* dx-10 grid
      (9,166,668 cells = 1,146.9 MiB); holding the boundary is 46 % more memory (1,673.0 ÷
      1,146.9), and the two are different records.
- [x] `backoffLimit: 0`, `restartPolicy: Never`, `ttlSecondsAfterFinished: 604800` (7 days) so
      the pod log — the result of a 5-hour run — outlives the queue.
- [x] `activeDeadlineSeconds: 28800` (8 h) as the wall bound, derived in the file from the
      step-3 pricing with the margin written out (13,662.4 s × 1.312 = 17,925 s, × 1.5 + 300 s
      = 27,188 s, rounded up to the next half hour); `MSE_CLOCK` has no counterpart here (there
      is no launcher).
- [x] Args via the manifest, `--xp cupy` present (the script's default is `numpy`, ~3× slower
      than the table's rate); the three raised bench ceilings are on the command line where a
      reviewer reads them.
- [x] `--pad-nm 320 --pml-nm 200` in the args, and the two flags themselves added to the script
      (step 5): the boundary has to be a distance for the records to be comparable to r20.
- [x] **The lead's word, 2026-10-16: option 1** — a third bench-only knob, `--max-cells N`,
      raised like the other two (step 4). The manifest passes it, so the args it ships are
      the ones it prices; the coarser rungs stay in the comments as the cheap alternative, with
      their own deadlines.

## 3. Pricing, from the product's own estimator (no run)

Every figure below is `scripts/paper1_nanowire.py --estimate` output. The table's rate is
constant across rungs: `cells × steps ÷ seconds = 40.77 Mcell-steps/s` (e.g. 2,903,040 × 39,650
÷ 2,823.3). The Orin's measured factor over that rate is **1.312** (r19 1.3116, r20 1.3119).

### The boundary is held, and the knob is what holds it

`fdtd_layout.OPEN_PAD_CELLS` is a cell **count**: the product's own pad is 320 nm at dx 20 —
which is what r20 ran — and 160 nm at dx 10, and `DEFAULT_PML_CELLS` likewise, 200 nm at dx 20
and 100 nm at dx 10. A dx-10 record on the product's defaults therefore moves the open boundary
and the wire's resolution at once, and no difference from r20 could be attributed to either. On
this scene that is not academic: 200 nm of pad against 400 nm moved F by **16 %**.

`--pad-nm`/`--pml-nm` (new, step 5) name it as a distance. The control is the coarse rung: at
dx 20 the knob's 320/200 nm **is** the product's own 16/10 cells, and the two plans are the same
grid — **2,903,040 cells, extents 2318.8 × 2318.8 × 2177.0 nm, finest 3.0000 nm**, i.e. r20's
own. Held at both dx, the dx-10 box is that same 2318.8 × 2318.8 × 2177.0 nm, so a difference
from r20 is resolution and nothing else.

Realised distances — the knob rounds to the **nearest** whole cell, so a rung whose dx does not
divide 320 nm cannot hold it exactly:

```
   dx 20  pad 16 cells = 320.00 nm  pml 10 = 200.00 nm   <- exact, and r20's own boundary
   dx 15  pad 21 cells = 315.00 nm  pml 13 = 195.00 nm   <- the nearest rung that fits 8 M cells
   dx 12  pad 27 cells = 324.00 nm  pml 17 = 204.00 nm   <- over 8 M, so not a way out
   dx 10  pad 32 cells = 320.00 nm  pml 20 = 200.00 nm   <- exact, the asked rung
```

### (a) The CW check — 993 nm, 100 periods, 8-period window

```
   dx    pad        cells    steps        s    GiB   dlam
 20.0    600    2,903,040   39,650   2823.3   0.36   9.93   <- r20's own rung (measured)
 15.0    600    5,234,775   39,650   5091.0   0.63   9.93   <- the nearest rung that fits
 12.0    600    8,952,300   39,650   8706.4   1.06   9.93   <- over 8 M, so not a way out
 10.0    600   14,048,300   39,650  13662.4   1.63   9.93   <- asked
```

`dlam` is rung-independent: it is set by the record's duration, and every rung here is 39,650
steps of the same dt (396.5 steps/period). The table prints these rows only because
`--max-cells` admits them; on the product's own ceiling the asked rung is refused before it is
priced.

### (b) The locator — ring-down, 1000 nm ± 100 nm, 30 tail periods

```
   dx    pad        cells    steps        s    GiB   dlam
 20.0    600    2,882,880   21,918   1549.8   0.36  18.11
 15.0    600    5,175,625   21,918   2782.4   0.62  18.11
 12.0    600    8,909,670   21,918   4789.8   1.05  18.11
 10.0    600   14,048,300   21,918   7552.4   1.63  18.11   <- asked
```

Two things to read off it. The locator's cells differ from the CW rung's at the same dx
(2,882,880 against 2,903,040 at dx 20) because a **pulse's band top sets the gold's skin depth**,
and so the gold's own cap: the two arms do not share a grid, and each is priced with its own
record. And the band is centred at **1000 nm**, not 993 nm, because the dx-20 resonance sits
outside the band arm's own default 830–970 nm window: the locator's job is to find where the
finer grid puts the mode. Read with the wrong centre (993 nm) the same record is 21,682 steps and
7,471.1 s — 1.1 % light — which is why the manifest names 1000 nm.

### The asked rung in full, and why it is 4.84× r20 and not 8× (the lead's two figures)

The CW check at dx 10 nm, mirror pad 600 nm, boundary held at r20's own 320/200 nm, 993 nm,
100 periods:

| | |
|---|---|
| cells | **14,048,300** (over `MAX_CELLS` 8,000,000 by 6,048,300 = **75.6 %**) |
| steps | **39,650** (100.0 periods; dt **8.3540e-18 s**, the same dt as r20) |
| memory | **1.63 GiB** (1,754,244,508 bytes = 1,673.0 MiB, the estimator's own storage) |
| estimator | **13,662.4 s** |
| Orin (×1.312, measured) | **17,925 s = 4 h 58 m** |
| A4000 | **unmeasured** — this Job is how it would be measured |

r20's own rung is 2,903,040 cells / 39,650 steps / 2,823.3 s at dx 20. The ratios are **4.839
cells**, **1.000 steps**, **4.839 seconds**. A uniform grid halved per axis would be 8× cells
and, at a halved dt, 2× steps — the "8× × 2× = 16×" the lead first priced. Both factors are
wrong, and the grid's own structure says why. `--dx` is the **base** spacing; the only layers
that take it are the three per-axis pad layers. Six further layer *requests* on this scene are
capped by three rules that never read `dx`, and what those rules produce is measured identical
at dx 20 and at dx 10:

- `thickness / MIN_CELLS` (`fdtd_grid.py:36`, `MIN_CELLS = 4`): the 12 nm oxide is 4 intervals
  of **3.0000 nm**, the grid's finest spacing — which sets `dt`, and with it the record's step
  count. Measured on the plan at both dx: `3.0000 nm x 4` in z, unchanged.
- `depth / CELLS_PER_SKIN` (`fdtd_layout.py:51,53`, `SKIN_DEPTHS = 3`, `CELLS_PER_SKIN = 4`):
  each gold face carries 12 intervals of `skin_depth("Au", band_top) / 4`, so 24 across y for
  two faces. Measured at both dx, unchanged: `6.6559 nm x 24` in y — for this record's band top,
  3.0191e14 Hz (993.0 nm), `fdtd_grid.skin_depth("Au", 3.0191e14) = 26.6235 nm`, and a face's
  clearances span `3 × 26.6235 = 79.87 nm`.
- a part thinner than `THIN_CELLS × dx` (`fdtd_layout.py:44`) gets four cells across.

Two things follow. Those caps are **this record's**: the skin depth follows the band top, so the
locator's pulse (3.3007e14 Hz = 908.3 nm → skin 27.1700 nm, cap 6.7925 nm) carries a slightly
coarser gold than the CW record at 993 nm (26.6235 nm, 6.6559 nm) — that is the whole of the
grid difference in table (b). And the pad is the opposite case: `OPEN_PAD_CELLS = BOX_CLEARANCE
+ 3 + DEFAULT_PML_CELLS = 16` **cells** (`fdtd_layout.py:41`), i.e. 16 × dx metres, the only
layer request that reads dx at all.

So the answer to *which regions* `--dx` refines: **the three per-axis pad layers, and nothing
else** — the background they cover. It does not refine the oxide, the gold's skin-depth shells,
the gold's body where a rule caps it, or any part thinner than the rule's own threshold.

That is also why the boundary decides the count. The pad is not a distance but
`OPEN_PAD_CELLS = 16` **cells**: at dx 10 the product's own pad is 160 nm, the box loses 320 nm
per axis, and the same record prices at **9,166,668 cells / 3.158× r20 / 8,914.8 s** — the figure
the lead was quoted before the boundary was held. Held, the box is r20's own and the count is
14,048,300, **4.839×**. Measured shapes, from the product's own plan:

```
dx 20.0  held 320/200 nm  shape (144, 144, 140)  cells  2,903,040  extents [2318.8, 2318.8, 2177.0]  finest 3.0000 nm
dx 10.0  held 320/200 nm  shape (244, 245, 235)  cells 14,048,300  extents [2318.8, 2318.8, 2177.0]  finest 3.0000 nm
```

The box is identical and the intervals are ~1.69× finer per axis (244/144 = 1.694, 245/144 =
1.701, 235/140 = 1.679) rather than 2×, because the capped layers keep their own spacing: they
take up node count per axis without refining, and the sub-intervals they impose flatten the
ratio. On the Orin the check is therefore **4 h 58 m** — not the ~16.5 h "8× × 2×" implies, and
not the 3 h 15 m the unheld boundary had priced.

### What the bench ceilings admit

- Steps: both records are 39,650 / 21,918 steps, over the product's 20,000 — admitted only with
  `--max-steps` (4b9d9fc).
- Seconds: 13,662.4 s and 7,552.4 s are both over the product's 1,800 s — admitted with
  `--max-seconds`, and the manifest carries 18,000 s (1.32× the CW estimate; still a pricing
  verdict, not a live bound).
- Cells: **14,048,300 > 8,000,000 — refused by the product, admitted only by `--max-cells`**
  (step 4, the lead's option 1); the manifest carries 15,000,000, 6.8 % over the priced record.
  Before step 4 the cell branch read the module constant inline and `run_limits` had no third
  key, so no bench flag could reach it.
- Memory: 1.63 GiB < `fdtd.MAX_BYTES` (3 GB) — no obstacle.

### The Orin's own seconds, and the deadline rule

13,662.4 s and 7,552.4 s × the measured 1.312: CW **17,925 s** (4 h 58 m), locator **9,909 s**
(2 h 45 m), and the nearest fitting rung dx 15 **6,679 s** (1 h 51 m). Rule used in the manifest:
`1.5 × Orin seconds + 300 s setup`, rounded up to the next half hour — CW dx 10 **28,800 s**
(8 h), locator **16,200 s** (4.5 h), dx 15 CW **10,800 s** (3 h).

## 4. The third bench ceiling — `--max-cells` (the lead's option 1, made)

- [x] `fdtd.check_limits(estimate, *, max_cells=None, max_steps=None, max_seconds=None)`:
      `max_cells` read at call time with `MAX_CELLS` as its None default, the cell branch
      comparing against the argument, and the refusal naming the ceiling that applied.
- [x] `fdtd.MAX_CELLS` unchanged at 8,000,000, and the docstring says what the product's own
      callers are (`worker/td_solve.py`, `web/td_form.py`, both calling bare).
- [x] `run_limits()` gains `max_cells` from `--max-cells`, parsed by the same `_raised_cap`
      factory (so a value below 8,000,000 is an argparse error, exit 2), clamped with `max()`
      so a namespace that never met the parser reads `fdtd.MAX_CELLS`.
- [x] The `caps` log line names the raised cell ceiling in the same parenthetical as the
      raised steps and seconds, on both the run path and the estimator's (`ladder`).
- [x] Tier-1 tests: the fdtd control checks 8,000,000 admitted / 8,000,001 refused / one cell
      over a *raised* ceiling refused naming the raised value and not 8,000,000; the script test
      prices the real dx-10 record at the held boundary, sees the product refuse it on cells
      alone (its steps and seconds fit), and sees `--max-cells 15000000 --max-steps 200000
      --max-seconds 18000` admit and price it; the widened `run_limits` and call-site controls
      carry the third key.
- [x] Delta `openspec/changes/paper1-a4000-bench/specs/time-domain/spec.md` (an ADDED
      requirement, so it cannot rewrite the two-ceiling paragraph that
      `paper1-nanowire-purcell` still owns — see "Archive order" below).
- [x] Image rebuilt and re-tagged from this commit and re-smoked; the manifest's tag updated
      to the new revision (the earlier images' script has no `--max-cells`, `--pad-nm` or
      `--pml-nm`, so their Job would exit on `unrecognized arguments`).

## 5. The open boundary as a distance — `--pad-nm` / `--pml-nm`

- [x] `fdtd_layout.plan(..., pad_cells=None, pml_cells=None)` and `layout(...)` alike: both
      read the same two hooks, so the estimator and the run cannot disagree about the boundary
      (`plan`'s node/cell count is what prices a record; `layout`'s is what runs it).
- [x] `scripts/paper1_nanowire.py`: `boundary_cells(args)` turns the two nm flags into whole
      cells -- `--pml-nm` (the distance) wins over `--pml` (the count) when both are named, and
      `--pml` alone still works -- and both of the scene's builders hand the pair to the
      planner.
- [x] A distance that is not a whole number of cells is **rounded to the nearest**, not up: the
      knob exists to hold the *reference's* distance, and nearest is the closest this dx can
      come to it (dx 15: 320/15 = 21.33 -> 21 cells = 315 nm, 200/15 -> 13 = 195; dx 12: 26.67
      -> 27 = 324, 16.67 -> 17 = 204). Every run's own log prints the distance it really held,
      so no log claims a boundary the grid did not have.
- [x] Three refusals keep the distance honest. Each is an exit 2 before a scene is built, and
      the tier-1 tests assert these exact texts through the script's own entry: a
      `--pad-nm`/`--pml-nm` under one cell at this dx ("so this run would have no boundary there
      at all"); `--pml` and `--pml-nm` naming different absorbers at this dx ("name one"); and
      an absorber that is not strictly inside the pad ("so the parts would be in the absorber").
- [x] The `geom` a scene carries includes the boundary it holds, in cells and in nm, and
      `ladder` prints it above the table; the run's own `boundary` line names it, so a record
      can be compared to r20 off its own log. The tiny arm's controls still hold at a boundary
      the caller moved (200/120 nm at its own 20 nm dx = 10/6 cells, 80 nm of open space).
- [x] Tier-1 controls: the dx-20 held plan is the product's own grid (2,903,040 cells, 16/10
      cells, the same extents) -- the knob reproduces the coarse run, which is what makes the
      dx-10 record comparable to it; the dx-10 held plan keeps that same box while its own grid
      is 244/245/235 cells; `boundary_cells` rounds as above; the estimator's plan and the run's
      layout are handed the same pair and return the same boundary; and the refusals are read
      both directly and through the script's own entry.
- [x] Delta `openspec/changes/paper1-a4000-bench/specs/time-domain/spec.md`: a second ADDED
      requirement, "The open boundary SHALL be nameable as a distance", stating the
      same-values default, the rounding rule, the cells-and-nm reading and the three refusals.

### Archive order (a real hazard, from the reviewer's own list)

`paper1-nanowire-purcell`'s `specs/time-domain/spec.md` MODIFIES the ceiling requirement and
its text says the product's callers can raise "both" ceilings. This change ADDS separate
requirements for the third ceiling and for the boundary distance instead of rewriting that
paragraph, so who archives first does not matter: an archive of `paper1-nanowire-purcell`
rewrites only what it names. When both are archived, grep the merged
`openspec/specs/time-domain/spec.md` for `max_cells` and for "both ceilings" — the two must not
contradict.

## Evidence for the row

- Commits: `f856bbf` is the code commit (script, layout, tests, specs, this file, the proposal)
  and it is the revision the image is built from, so it carries the image's tag; the commit after
  it lands the manifest that names `…/paper1-bench-cuda:f856bbf`.
- Built on the VM from that revision: `DOCKER_CONFIG=~/.docker-mse`, exit 0, 12/12 steps,
  `sha256:94c9507399a3b6f5a5d154826cae198dfe2ebfa6c74dae14f73d6edc892c5df2`, 4.74 GB. The
  build-time closure check printed `closure ok, cupy 14.2.0`, so the CUDA wheel is importable
  without a GPU.
- The image's seven python files are the revision's bytes, checked rather than assumed: each
  file's md5 inside the container equals `git show f856bbf:<path> | md5sum` for all seven
  (`paper1_nanowire.py` `468d3006…`, `fdtd.py` `d233b759…`, `fdtd_grid.py` `130a6791…`,
  `fdtd_layout.py` `7e328cf6…`, `fdtd_voxel.py` `4be5b67e…`, `mse_common.py` `a1a34af3…`,
  `mse_geometry.py` `2d5a3463…`).
- Smoke-tested on the VM, no GPU, `docker run --rm --cpus=2 --memory=2g`, all exit 0:
  `--entrypoint python3` → `cupy 14.2.0 numpy 2.2.6 scipy 1.15.3 trimesh 5.1.0 uid 10001`;
  `--help` lists `--pml`, `--pad-nm`, `--pml-nm` and `--max-cells`; `--tiny --pad-nm 200
  --pml-nm 120 --xp numpy` → the boundary line `boundary  open: pad 10 cells = 200 nm, absorber
  6 cells = 120 nm, 80 nm of open space (--pad-nm 200 --pml-nm 120)` and `tiny: all controls
  hold`; and the pricing arm with THIS manifest's own args reproduces the table's held dx-10 row
  and its boundary line — `10.0 600 14,048,300 39,650 13662.4 1.63 9.93 fits (the bench ceiling,
  not the tier-2 clock)`, under `boundary  held at 320 nm of pad and 200 nm of absorber at dx
  10.0; each rung's cell counts follow its own dx (32/20 cells at dx 10.0)`.
- Not observed, and not claimed: the `caps` line's `(bench ceiling: …, …, N cells)` clause
  prints only on a real run (`paper1_nanowire.py:930`), and no real run is affordable here — the
  reviewer said the same. It is read from the source and asserted by tier-1, no more.
- Tier-1 for this revision: the two files hold 54 tests and all 54 pass, in two complementary
  selections (`-k "ceiling or boundary or holds or nearest or estimator or prices or hand"` →
  17 passed / 37 deselected; its exact negation → 37 passed / 17 deselected), so the union is the
  whole pair rather than two overlapping hunks.
- The reviewer's two stale spots, both closed here and re-checked by grep at this revision:
  `manifests/bench/paper1-a4000-job.yaml`'s obstacle block now says the cell shortfall is resolved
  by this file's own `--max-cells` and the GPU quota is the only live obstacle (its one `refused`
  is the quota text); `proposal.md` no longer calls `--max-cells` proposed nor says the record
  cannot be launched.
- `openspec validate paper1-a4000-bench` → valid.
- `mse-harbor-pull` confirmed present in ns `mode-search-engine` and is what the live `mse-web`
  deployment uses to pull `harbor.arcadia-ecs.local/mode-search-engine/mse-web:1cca12b` — the
  manifest's pull secret and registry path are the ones a working consumer already uses.
- Pushed by nobody yet: the builder does not push; the image is tagged locally for the lead.
