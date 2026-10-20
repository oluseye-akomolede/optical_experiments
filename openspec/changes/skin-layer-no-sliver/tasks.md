# Tasks: skin-layer-no-sliver

## 1. The layout rule

- [x] 1.1 `fdtd_layout.layout`: add the two face layers only when the middle
      they leave is at least one `dx_m`; otherwise one layer across the whole
      extent at a quarter of the skin depth
- [x] 1.2 Record in the code why the strip matters (the thin-layer rule cuts it
      into cells finer than the skin layers) and the measured case (150 nm
      mirror at dx 10 nm: 378,896 steps for a record that needs 121,121)

## 2. Tests

- [x] 2.1 `tests/test_fdtd_layout.py`: a 180 nm gold slab at dx 50 nm -- the
      fixture leaves a 6.1 nm strip (asserted) -- takes one uniform layer
      across it, no cell under half the cap, and no more cells than
      `ceil(thickness / (depth / 4))`
- [x] 2.2 Confirm the new test fails on the parent revision 2e7ae0e and state
      the measured numbers (36 cells, smallest 1.53 nm = 0.21x cap)
- [x] 2.3 The existing thin-plate and 4-cell tests still pass (the rule is
      unchanged for them): `tests/test_fdtd_layout.py` 68 passed

## 3. Run C

- [x] 3.1 `scripts/paper1_nanowire.planned_scene` on Run C's command
      (`--profile pulse --ringdown --tail-periods 300 --hex paper --qd-below 430
      --wavelength 900 --band-half 80 --dx 10 --mirror-pad 600 --pad-nm 320
      --pml-nm 200 --gold-data palik`): before, at 2e7ae0e, finest z cell
      0.801 nm at z=74.2..75.0 nm, 15,312,024 cells, 378,896 steps; after,
      finest 3.000 nm, 14,400,594 cells, 121,121 steps (the 15M-cell and
      200k-step caps). Rakić under the same command: 13,931,190 cells,
      116,455 steps, finest 3.000 nm.

## 4. Validation and handoff

- [x] 4.1 `openspec validate skin-layer-no-sliver --strict` and
      `openspec validate --specs --strict` (baseline 12 passed, 0 failed)
- [x] 4.2 `tests/test_fdtd_layout.py` at tier 1 (`ulimit -v 4194304`,
      `nice -n 19`): 68 passed
- [x] 4.3 Commit locally (no push) and hand to the lead with the sha and the
      before/after step counts
