# time-domain

## ADDED Requirements

### Requirement: The cell ceiling SHALL be a per-run argument, raisable by the bench script alone

`fdtd.check_limits` SHALL take a third ceiling, `max_cells`, alongside `max_steps` and
`max_seconds`, defaulting to `fdtd.MAX_CELLS` read at call time and not bound at import, and
its refusal SHALL name the ceiling that actually applied rather than the product's constant.
`fdtd.MAX_CELLS` SHALL keep its value and its meaning: what changes is that the bench path can
be measured against a raised ceiling without the product's own moving. The paper-1 bench script
alone SHALL be able to raise it, through `--max-cells N`, and the flag SHALL only raise: a value
under the product's own SHALL be an argparse error, and `run_limits` SHALL clamp with `max()`
so a namespace that never met the parser reads the product's own. The web app and the k8s Job
path SHALL NOT be able to raise it: `worker/td_solve.py` and the web app SHALL call
`check_limits` with no ceiling, and no environment variable and no module global SHALL set one.
A run that used a raised ceiling SHALL say so on its `caps` log line, naming the cell ceiling in
the same parenthetical that names the raised steps and seconds.

The reason the third ceiling is needed at all is the grid's shape: on the paper's own scene a
dx-10 nm record with the open boundary held at r20's distance is 14,048,300 cells, over the 8 M
cap by 75.6 %, and the record that resolves the paper's 4 nm mode is that record at 993 nm --
100 periods, 39,650 steps.

#### Scenario: the product's own default is unchanged and one cell over it is refused
- **WHEN** `fdtd.check_limits` is called with no ceiling on an estimate of 8,000,000 cells
- **THEN** it admits it, and on 8,000,001 cells it raises `EstimateError` naming `8,000,000`,
  so the control is the product's own constant and not a number this change invented

#### Scenario: a raised ceiling is the ceiling the refusal names
- **WHEN** an estimate of 14,048,301 cells is checked against `max_cells=14,048,300`
- **THEN** the refusal names `14,048,300` and not `8,000,000`, because the log line's job is to
  name the ceiling that was in force

#### Scenario: the bench flag is the only way in, and it only raises
- **WHEN** `--max-cells 1000000` is parsed by the bench script
- **THEN** argparse refuses it (exit 2, `may only raise a cap`), and a hand-built namespace
  carrying `max_cells=1_000_000` reads `fdtd.MAX_CELLS` through `run_limits` instead

#### Scenario: the manifest's own record is refused by the product and priced with the flags
- **WHEN** the bench script's pricing arm (the ladder, which voxelises nothing) prices the
  manifest's own record -- `--hex paper --wavelength 993 --dx 10 --mirror-pad 600 --periods 100
  --pad-nm 320 --pml-nm 200`
- **THEN** with no bench flag the two finer rungs are refused -- the dx-10/pad-600 row reads
  `refused: The grid has 14,048,300 cells, over the limit of 8,000,000`, the dx-12 row
  8,952,300 cells against the same limit, and dx 20 and 15 fit -- and with `--max-cells 15000000
  --max-steps 200000 --max-seconds 18000` the dx-10 row is priced at 14,048,300 cells,
  39,650 steps and 13,662.4 s in 1.63 GiB

#### Scenario: the wide-scan call sites do not diverge on which ceiling is in force
- **WHEN** the bench script's `check_limits` call sites are read
- **THEN** each hands its ceiling over (found by `re.findall`, each passing `**`), so the
  estimator's rungs and the run path cannot be priced against two different cell ceilings

### Requirement: The open boundary SHALL be nameable as a distance, so a resolution change is only a resolution change

`fdtd_layout.plan` and `fdtd_layout.layout` SHALL take `pad_cells` and `pml_cells` as
keyword-only arguments, defaulting to `fdtd_layout.OPEN_PAD_CELLS` and `fdtd.DEFAULT_PML_CELLS`
read at call time, and SHALL return the pair they planned with. Only the paper-1 bench script
SHALL pass them, through `--pad-nm NM` and `--pml-nm NM`, which name a DISTANCE: the pad is a
cell count, so the product's own pad is 320 nm at dx 20 and 160 nm at dx 10, and a dx-10 record
priced without these flags would move the open boundary at the same time as the grid's
resolution, leaving a difference from r20 attributable to neither. A distance that is not a
whole number of cells SHALL be rounded to the nearest cell, and a run SHALL report the boundary
it held in both cells and nanometres. The bench script SHALL refuse a distance under one cell,
SHALL refuse an absorber that is not inside the pad, and SHALL refuse a run that names an
absorber both in cells and in nanometres unless the two are the same cell count.

#### Scenario: the held distance is r20's own boundary, measured at r20's own dx
- **WHEN** the planner is asked for dx 20 with `--pad-nm 320 --pml-nm 200`
- **THEN** it returns the same grid as dx 20 with no flags at all -- 2,903,040 cells, 16 pad
  cells, 10 absorber cells, the box 2318.8 x 2318.8 x 2177.0 nm -- so the distance this change
  holds for dx 10 is the boundary r20 measured F = 4.38 on

#### Scenario: held at dx 10, r20's box is held and only the resolution moves
- **WHEN** the planner is asked for dx 10 with `--pad-nm 320 --pml-nm 200`
- **THEN** the box is r20's own 2318.8 x 2318.8 x 2177.0 nm, held by 32 pad cells and 20
  absorber cells, and the axes take 244 x 245 x 235 nodes against r20's 144 x 144 x 140:
  14,048,300 cells against 2,903,040 (4.84x) at an identical 39,650 steps and 8.353997e-18 s,
  because `dt` is set by the oxide's own 3.000 nm, which does not scale with `--dx`

#### Scenario: the product's boundary is not reachable from the product's own path
- **WHEN** the web form and the worker build their request and call `fdtd_layout.layout`
- **THEN** neither passes a pad or an absorber and the planner reads its own defaults, so a
  product run's boundary is the product's

#### Scenario: the three boundary refusals
- **WHEN** the bench script is asked for `--pad-nm 8` at dx 10, or for `--pml 20` together with
  `--pml-nm 100` at dx 10, or for `--pad-nm 100` with the product's own absorber
- **THEN** each is refused with exit 2 -- `under one cell at dx 10`, two absorbers that are not
  the same cell count (`--pml 20` against 10 cells, named as different absorbers), and an
  absorber that is not inside the pad (`the 10-cell absorber is not inside the 10-cell pad
  (100 nm in 100 nm)`) -- and no scene is built
