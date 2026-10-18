# Tasks — paper1-collection

- [x] 1. Measure how a phasor box should meet the mirror (proposal):
  `tests/test_phasor_mirror.py`, a dipole over an exact Yee PEC plane against
  the closed-form image dipole, and over a finite Rakić gold slab with the
  full enclosure as the reference. Choice: the box encloses the whole scene.
- [x] 2. `fdtd.py`: `phasor_box_bytes`; `phasor_box` / `phasor_frequencies` /
  `phasor_steps` on `memory_estimate` and `estimate_run`. No phasor code
  changed (no bug found).
- [x] 3. `scripts/paper1_nanowire.py`: `--collection-na` (off by default);
  `scene_extent`, `collection_box`, `collection_plan`, `collection_line`,
  `collection`; the box priced in `filled` and in the ladder; the arm's
  refusals in `arm_conflicts`; `_run` hands `simulate` the box only when the
  arm is on.
- [x] 4. Tests: `tests/test_paper1_collection.py` (placement on four plans,
  no-room and material refusals, the arm's refusals, one vacuum run against
  the closed form, off-by-default invariance, the estimate and the ladder
  priced); `tests/test_fdtd_memory.py` (the box against a real
  `_PhasorBox`, and a measured run inside the estimate).
- [x] 5. `tests/check_discriminating.py`: `collection-*` and
  `memory-phasor-*`, 16 cases, each turning exactly its tests red.
- [x] 6. Verification: `OMP_NUM_THREADS=2 python -m pytest
  tests/test_paper1_collection.py tests/test_phasor_mirror.py
  tests/test_fdtd_memory.py tests/test_paper1_nanowire.py
  tests/test_fdtd_phasor_box.py tests/test_farfield.py -q`;
  `npx -y @fission-ai/openspec validate paper1-collection --strict`.
- [ ] 7. Later, on the GPU: the paper-1 CW run with `--collection-na 0.8` at
  the chosen rung (and the pad convergence the F reading already owes, since
  the fraction is the finite pad's), read against the paper's 74%.
