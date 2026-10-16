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
      the A4000 nodes carries it is the quota's and the card's own allocatable's decision, and
      the file says where to add a `kubernetes.io/hostname` if one is to be pinned by hand.
      Limits == requests (`nvidia.com/gpu 1`, `cpu 2`, `memory 8Gi`), OMP/MKL/OPENBLAS/NUMEXPR 2.
- [x] Memory justified from the pricing: the priced dx-10 grid is ~1.12 GiB (the table's
      ~131 bytes/cell × 9,166,668 cells; the same table prints 0.36 GiB at 2,903,040 cells) —
      7× inside the 8Gi limit and 15 % of the card's 16376 MiB.
- [x] `backoffLimit: 0`, `restartPolicy: Never`, `ttlSecondsAfterFinished: 604800` (7 days) so
      the pod log — the result of a 5-hour run — outlives the queue.
- [x] `activeDeadlineSeconds` as the wall bound, derived in the file from the step-3 pricing
      with the margin written out; `MSE_CLOCK` has no counterpart here (there is no launcher).
- [x] Args via the manifest, `--xp cupy` present (the script's default is `numpy`, ~3× slower
      than the table's rate); the raised bench ceilings are on the command line where a
      reviewer reads them.
- [ ] **Blocked on the lead's word:** the args as shipped are refused by the cell cap (below).
      The manifest's comments carry the dx-12/pad-600 variant that fits today, with its own
      deadline, so the lead can pick either without an edit of substance.

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

### What the bench ceilings admit

- Steps: both records are 39,650 / 21,918 steps, over the product's 20,000 — admitted only
  with `--max-steps` (the bench script's own flag, `run_limits` in
  `scripts/paper1_nanowire.py`, handed to `fdtd.check_limits`); this is what 4b9d9fc delivered.
- Seconds: 8,914.8 s and 4,928.0 s are both over the product's 1,800 s — admitted with
  `--max-seconds`, and the manifest carries 10,800 s (well above both, and still only a
  pricing verdict, not a live bound).
- Cells: **9,166,668 > 8,000,000 — refused, and no bench flag can raise it.** The cell branch
  of `fdtd.check_limits` reads the module constant inline; `run_limits` has no third key.
- Memory: 1.12 GiB < `fdtd.MAX_BYTES` (3 GB) — no obstacle.

### The Orin-rate seconds and the deadline rule

The estimator's rate is 1.312× the Orin's measured rate (r20: 3,704.0 s measured ÷ 2,823.3 s
estimated = 1.3119; r19: 1,851.5 s measured ÷ 1,411.7 s derived the same way, 2,903,040 ×
19,825 ÷ 40.77 Mcell-steps/s = 1.3116). So what an Orin would really take:
CW **11,696 s** (3 h 15 m), locator **6,465 s** (1 h 48 m). Deadline rule used in the
manifest: `1.5 × Orin seconds + 300 s setup`, rounded up to the next half hour —
CW **18,000 s** (5 h), locator **10,800 s** (3 h), dx-12 CW (fits today) **12,600 s** (3.5 h).

## Proposed, NOT made here: the third bench ceiling

`--max-cells`, threaded exactly like the other two: `fdtd.check_limits(estimate, *,
max_cells=None)` reading `MAX_CELLS` at call time, `run_limits()` gaining the key from a
`--max-cells` parsed by the same `_raised_cap` factory (a value below the cap is refused), the
product entry points (`worker/td_solve.py`, `web/td_form.py`) still calling bare. One commit,
one delta on the `time-domain` spec's ceiling requirement, tier-1 tests in the same shape as
4b9d9fc's. **Waiting on the lead**: without it the dx-10 check cannot run on any machine, and
the alternative is dx 12 nm/pad 600 (6,397,264 cells) — the same pad, a coarser grid.

## Evidence for the row

- shas: (filled after each commit) — image build log, the three smoke outputs, `openspec
  validate paper1-a4000-bench`, and the affected tier-1 counts.
