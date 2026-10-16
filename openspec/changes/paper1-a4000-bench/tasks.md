# paper1-a4000-bench — tasks

The lead's three steps (queue row 2026-10-16 18:44Z). One commit each, local only; the lead
merges and pushes. Nothing runs on the cluster until review and the lead's go.

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
- [x] Memory justified from the pricing: the priced dx-10 grid is ~1.12 GiB (the table's
      ~131 bytes/cell × 9,166,668 cells; the same table prints 0.36 GiB at 2,903,040 cells) —
      1,146.9 MiB, so 14.0 % of the 8Gi request (8,192 MiB: the request is 7.1× the estimate)
      and 7.0 % of the card's 16,376 MiB.
- [x] `backoffLimit: 0`, `restartPolicy: Never`, `ttlSecondsAfterFinished: 604800` (7 days) so
      the pod log — the result of a 5-hour run — outlives the queue.
- [x] `activeDeadlineSeconds` as the wall bound, derived in the file from the step-3 pricing
      with the margin written out; `MSE_CLOCK` has no counterpart here (there is no launcher).
- [x] Args via the manifest, `--xp cupy` present (the script's default is `numpy`, ~3× slower
      than the table's rate); the raised bench ceilings are on the command line where a
      reviewer reads them.
- [x] **The lead's word, 2026-10-16: option 1** — a third bench-only knob, `--max-cells N`,
      raised like the other two (step 4). The manifest passes it, so the args it ships are
      the ones it prices; the dx-12/pad-600 variant stays in the comments as the cheap
      alternative, with its own deadline.

## 3. Pricing, from the product's own estimator (no run)

`--estimate` prints every rung; the asked rung is the last line. Rate is constant across the
table: `cells × steps ÷ seconds = 40.77 Mcell-steps/s` (e.g. 2,903,040 × 39,650 ÷ 2,823.3).

### (a) CW check — 993 nm, 100 periods

```
   dx    pad        cells    steps        s    GiB   dlam
 20.0    600    2,903,040   39,650   2823.3   0.36   9.93   <- r20's own rung (measured)
 15.0    600    4,381,245   39,650   4260.9   0.54   9.93
 12.0    600    6,397,264   39,650   6221.5   0.79   9.93
 10.0    400    6,040,468   39,650   5874.5   0.73   9.93
 10.0    600    refused: The grid has 9,166,668 cells, over the limit of 8,000,000.
```

The asked rung's own seconds are not printed (it is refused before the print), so they are
derived from the same table's rate: 9,166,668 × 39,650 ÷ 40.77 Mcell-steps/s = **8,914.8 s**
at the table's rate, **~1.12 GiB**, **9.93 nm per bin** (dlam is rung-independent: it is set by
the record's duration, and every rung here is 39,650 steps of the same dt, 396.5 steps/period).

### (b) Locator — ring-down, 1000 nm ± 100 nm, 30 tail periods

```
 20.0    600    2,882,880   21,918   1549.8   0.36  18.11
 12.0    600    6,363,236   21,918   3420.9   0.78  18.11
 10.0    400    6,040,468   21,918   3247.4   0.73  18.11
 10.0    600    refused: The grid has 9,166,668 cells, over the limit of 8,000,000.
```

Asked rung, derived the same way: 9,166,668 × 21,918 ÷ 40.77 Mcell-steps/s = **4,928.0 s**,
**~1.12 GiB**, **18.11 nm per bin**. The band is centred at 1000 nm and 100 nm wide because
the dx-20 resonance (993 nm) sits outside the band arm's default 830–970 nm window: the
locator's job is to find where the finer grid puts the mode, not to confirm the coarse one.
Both asked rungs' seconds were also computed **directly** through the product's own estimator
path (`fdtd_layout.estimate` on a record the cap refuses, which is why the table cannot print
them): the CW rung returns 8,914.8 s and the locator 4,928.0 s, to the printed figure. Both
asked rungs were confirmed, not just derived.

### The asked rung in full, and why it is 3.16× r20 and not 8× (the lead's two figures)

The CW check at dx 10 nm, pad 600 nm, 993 nm, 100 periods:

| | |
|---|---|
| cells | **9,166,668** (over `MAX_CELLS` 8,000,000 by 1,166,668 = **14.58 %**) |
| steps | **39,650** (100.0 periods; dt **8.3540e-18 s**, the same dt as r20) |
| memory | **1.12 GiB** |
| estimator | **8,914.8 s** |
| Orin (×1.312, measured) | **11,696.3 s = 3 h 14 m 56 s** |
| A4000 | **unmeasured** — this Job is how it would be measured |

r20's own rung is 2,903,040 cells / 39,650 steps / 2,823.3 s at the same dx 20. The ratios
are 3.158 cells, **1.000 steps**, 3.158 seconds. A uniform grid halved per axis would be 8×
cells and, at a halved dt, 2× steps — the "about 8× × 2× = 16×" the lead priced. Both factors
are wrong, and the grid's own structure says why. `--dx` is the **base** spacing; the only
layers that take it are the three per-axis pad layers. Six further layer *requests* on this
scene are capped, and the requests are byte-identical at dx 20 and at dx 10:

```
     160.00 ..   1838.80 nm   span 1678.80   cap dx          <- x, the whole extent
     160.00 ..   1838.80 nm   span 1678.80   cap dx          <- y
     160.00 ..   1697.00 nm   span 1537.00   cap dx          <- z
     160.00 ..    241.70 nm   span   81.70   cap 6.809 nm    <- gold face layer (x,y)
    1757.10 ..   1838.80 nm   span   81.70   cap 6.809 nm    <- gold face layer (x,y)
     160.00 ..    241.70 nm   span   81.70   cap 6.809 nm    <- gold face layer (x,y)
    1757.10 ..   1838.80 nm   span   81.70   cap 6.809 nm    <- gold face layer (x,y)
     160.00 ..     310.00 nm   span  150.00   cap 6.809 nm    <- gold body (z): 150 nm
     310.00 ..     322.00 nm   span   12.00   cap 3.000 nm    <- oxide (z): 12 nm
```

So the answer to *which regions* `--dx` refines: **only the pad-covered extent** (the
background and everything thicker than the rules below care about). It does **not** refine
the six capped regions, which are fixed by three rules that never read `dx`:
`thickness / MIN_CELLS` (12 nm ÷ 4 = **3.000 nm**, the oxide — and the grid's finest spacing,
which is why `dt` and the step count are dx-independent), `depth / CELLS_PER_SKIN` (the gold
skin depth 27.236 nm ÷ 4 = **6.809 nm**, and `SKIN_DEPTHS`× that in the faces' clearances,
81.70 nm), and a part thinner than `THIN_CELLS × dx` getting four cells across. And the pad
itself is `OPEN_PAD_CELLS = 16` **cells**, i.e. 16 × dx **metres**: halving dx shrinks the box
by 320 nm per axis, so the extent that does scale grows by 1.48× per axis, not 2× — measured
shapes, from the product's own plan:

```
dx 20.0  shape (142, 143, 139)  cells 2,822,534  extents_nm [2318.8, 2318.8, 2177.0]  finest 3.0000 nm
dx 10.0  shape (211, 212, 202)  cells 9,035,864  extents_nm [1998.8, 1998.8, 1857.0]  finest 3.0000 nm
```

interval ratio 1.4755 × 1.4825 × 1.4532 = 3.18, which is the 3.158 the estimator prints for
node counts. The true cost of the dx-10 CW check is therefore **3.16× r20's wall, not 16×**:
on the Orin, 3 h 15 m rather than the ~16.5 h that "8× cells × 2× steps" implies.

### What the bench ceilings admit

- Steps: both records are 39,650 / 21,918 steps, over the product's 20,000 — admitted only
  with `--max-steps` (the bench script's own flag, `run_limits` in
  `scripts/paper1_nanowire.py`, handed to `fdtd.check_limits`); this is what 4b9d9fc delivered.
- Seconds: 8,914.8 s and 4,928.0 s are both over the product's 1,800 s — admitted with
  `--max-seconds`, and the manifest carries 10,800 s (well above both, and still only a
  pricing verdict, not a live bound).
- Cells: **9,166,668 > 8,000,000 — refused by the product, admitted only by `--max-cells`**
  (step 4, the lead's option 1). Before step 4 the cell branch read the module constant
  inline and `run_limits` had no third key, so no bench flag could reach it.
- Memory: 1.12 GiB < `fdtd.MAX_BYTES` (3 GB) — no obstacle.

### The Orin-rate seconds and the deadline rule

The estimator's rate is 1.312× the Orin's measured rate (r20: 3,704.0 s measured ÷ 2,823.3 s
estimated = 1.3119; r19: 1,851.5 s measured ÷ 1,411.7 s derived the same way, 2,903,040 ×
19,825 ÷ 40.77 Mcell-steps/s = 1.3116). So what an Orin would really take:
CW **11,696 s** (3 h 15 m), locator **6,465 s** (1 h 48 m). Deadline rule used in the
manifest: `1.5 × Orin seconds + 300 s setup`, rounded up to the next half hour —
CW **18,000 s** (5 h), locator **10,800 s** (3 h), dx-12 CW (fits today) **12,600 s** (3.5 h).

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
- [x] Tier-1 tests: the new fdtd control checks 8,000,000 admitted / 8,000,001 refused /
      9,166,668 refused by the product and admitted by the flag / one cell over a *raised*
      ceiling refused naming 9,166,668 and not 8,000,000; the new script test prices the real
      dx-10 scene, sees the product refuse it on cells alone (its steps and seconds fit), and
      sees `--max-cells 10000000` admit it; the widened `run_limits` and call-site controls
      carry the third key.
- [x] Delta `openspec/changes/paper1-a4000-bench/specs/time-domain/spec.md` (an ADDED
      requirement, so it cannot rewrite the two-ceiling paragraph that
      `paper1-nanowire-purcell` still owns — see "Archive order" below).
- [x] Image rebuilt and re-tagged from this commit and re-smoked; the manifest's tag updated
      to the new revision (the d101aa8 image's script has no `--max-cells`, so its Job would
      exit on `unrecognized arguments`).
- [ ] **The Job itself still needs the lead and the owner:** `mse-power-cap` is 0 GPU, and the
      run starts the 16-hour clock in the handoff's own terms.

### Archive order (a real hazard, from the reviewer's own list)

`paper1-nanowire-purcell`'s `specs/time-domain/spec.md` MODIFIES the ceiling requirement and
its text says the product's callers can raise "both" ceilings. This change ADDS a separate
requirement for the third instead of rewriting that paragraph, so who archives first does not
matter: an archive of `paper1-nanowire-purcell` rewrites only what it names. When both are
archived, grep the merged `openspec/specs/time-domain/spec.md` for `max_cells` and for "both
ceilings" — the two must not contradict.

## Evidence for the row

- Image commit **d101aa8** (Dockerfile + this file + the spec delta); manifest commit follows it.
  The image tag is d101aa8: the revision the image was built from.
- Built on the VM: `DOCKER_CONFIG=~/.docker-mse`, `nice -n 19`, exit 0, 12/12 steps, image
  `4.74 GB`. The build-time closure check printed `#15 1.331 closure ok, cupy 14.2.0`, so the
  CUDA wheel is importable without a GPU.
- Smoke-tested on the VM, no GPU, `docker run --rm --cpus=2 --memory=2g`, all exit 0:
  - `--entrypoint python3 -c "import cupy, numpy, scipy, trimesh; ..."` →
    `cupy 14.2.0 numpy 2.2.6 scipy 1.15.3 trimesh 5.1.0`, `uid 10001`
  - `--help` → the script's own usage line
  - `--tiny --xp numpy` → `tiny: all controls hold`
- `openspec validate paper1-a4000-bench` → valid.
- `mse-harbor-pull` confirmed present in ns `mode-search-engine` and is what the live `mse-web`
  deployment uses to pull `harbor.arcadia-ecs.local/mode-search-engine/mse-web:1cca12b` — the
  manifest's pull secret and registry path are the ones a working consumer already uses.
- Pushed by nobody yet: the builder does not push; the image is tagged locally for the lead.
