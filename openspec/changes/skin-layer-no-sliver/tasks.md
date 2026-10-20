# Tasks: skin-layer-no-sliver

## 1. The layout rule

- [x] 1.1 `fdtd_layout.layout`: add the two face layers only when the middle
      they leave is at least the wider of one `dx_m` and one skin depth;
      otherwise one layer across the whole extent at a quarter of the skin
      depth
- [x] 1.2 Record in the code why the strip matters (graded_axis splits any gap
      under the coarse cell into 4 cells, finer than the skin layers) and the
      measured case (Run C's 150 nm mirror at dx 10 nm: a 3.21 nm strip,
      0.801 nm cells, 378,896 steps for a record that needs 121,121)
- [x] 1.3 Set the threshold at the wider of `dx_m` and one skin depth, not at
      `dx_m` alone: the gap between the face layers is filled with a few cells
      (four at a quarter of its width for 6.1 and 26.1 nm, two and three for
      11.1 and 16.1 nm), so middles of 6.1, 11.1, 16.1 and 26.1 nm came out in
      cells of 1.528, 5.556, 5.352 and 6.528 nm (0.21x-0.90x of the 7.245 nm
      spacing), and one skin depth is where the split meets that spacing (the
      finest cell at a 28.1 nm middle is 7.03 nm). One layer across a part in
      this band costs under 24 + 4*max(dx_m, depth)/depth cells (28 where the
      coarse cell is no coarser than the skin depth) at depth/4
- [x] 1.4 Show the threshold closes the measured band rather than moving it:
      with room around the slab (host 3000 nm tall, part 900 nm up), middles
      from 36.1 to 726.1 nm came out with a finest cell of exactly 7.2454 nm
      -- 1.000x the spacing -- and a 28.1 nm middle at 0.970x, so the two-layer
      branch is harmless above one skin depth. An earlier 31.1 nm / 0.86x and
      106 nm / 0.38x reading did not reproduce on this fixture and does not
      hold (31.1 nm reads 1.000x in a 400 nm and in a 3000 nm host)

## 2. Tests

- [x] 2.1 `tests/test_fdtd_layout.py`: a 180 nm gold slab at dx 50 nm -- the
      fixture leaves a 6.1 nm strip (asserted) -- takes one uniform layer
      across it, no cell under half the cap, and no more cells than
      `ceil(thickness / (depth / 4))`
- [x] 2.2 Confirm the new test fails on the parent revision 2e7ae0e and state
      the measured numbers (36 cells, smallest 1.53 nm = 0.21x cap)
- [x] 2.3 The existing thin-plate and 4-cell tests still pass (the rule is
      unchanged for them): `tests/test_fdtd_layout.py` 68 passed
- [x] 2.4 Second fixture for the band a coarse-cell threshold alone leaves
      open: 190 nm at dx 10 nm (middle 16.1 nm, over one coarse cell and under
      one skin depth). On 2e7ae0e it reads 29 cells across the part, against
      the 27 the 7.245 nm spacing needs, and a 5.352 nm middle cell

## 3. Run C

- [x] 3.1 `scripts/paper1_nanowire.planned_scene` on Run C's command
      (`--profile pulse --ringdown --tail-periods 300 --hex paper --qd-below 430
      --wavelength 900 --band-half 80 --dx 10 --mirror-pad 600 --pad-nm 320
      --pml-nm 200 --gold-data palik`): before, at 2e7ae0e, finest z cell
      0.801 nm at z=74.2..75.0 nm, 15,312,024 cells, 378,896 steps; after,
      finest 3.000 nm, 14,400,594 cells, 121,121 steps. Both revisions are
      over the app's caps (8,000,000 cells, 20,000 steps, 1,800 s) and only
      the fixed grid fits the bench Job's raised ones (15,000,000 cells,
      200,000 steps, 18,000 s). Rakić under the same command: 13,931,190
      cells, 116,455 steps, finest 3.000 nm.

## 4. Validation and handoff

- [x] 4.1 `openspec validate skin-layer-no-sliver --strict` and
      `openspec validate --specs --strict` (baseline 12 passed, 0 failed)
- [x] 4.2 `tests/test_fdtd_layout.py` at tier 1 (`ulimit -v 4194304`,
      `nice -n 19`): 68 passed
- [x] 4.3 Commit locally (no push) and hand to the lead with the sha and the
      before/after step counts
