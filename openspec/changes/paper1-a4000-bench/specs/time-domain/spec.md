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
dx-10 nm record is 9,166,668 cells, over the 8 M cap by 14.6 %, and the record that resolves the
paper's 4 nm mode is that record at 993 nm -- 100 periods, 39,650 steps.

#### Scenario: the product's own default is unchanged and one cell over it is refused
- **WHEN** `fdtd.check_limits` is called with no ceiling on an estimate of 8,000,000 cells
- **THEN** it admits it, and on 8,000,001 cells it raises `EstimateError` naming `8,000,000`,
  so the control is the product's own constant and not a number this change invented

#### Scenario: a raised ceiling is the ceiling the refusal names
- **WHEN** an estimate of 9,166,669 cells is checked against `max_cells=9,166,668`
- **THEN** the refusal names `9,166,668` and not `8,000,000`, because the log line's job is to
  name the ceiling that was in force

#### Scenario: the bench flag is the only way in, and it only raises
- **WHEN** `--max-cells 1000000` is parsed by the bench script
- **THEN** argparse refuses it (exit 2, `may only raise a cap`), and a hand-built namespace
  carrying `max_cells=1_000_000` reads `fdtd.MAX_CELLS` through `run_limits` instead

#### Scenario: the manifest's own record is refused by the product and priced with the flags
- **WHEN** the bench script's pricing arm (the ladder, which voxelises nothing) prices the
  manifest's own record -- `--hex paper --wavelength 993 --dx 10 --mirror-pad 600 --periods 100`
- **THEN** with no bench flag every rung is refused and the dx-10/pad-600 row reads `refused:
  The grid has 9,166,668 cells, over the limit of 8,000,000`, and with `--max-cells 10000000
  --max-steps 200000 --max-seconds 10800` that same row is priced at 9,166,668 cells,
  39,650 steps and 8,914.8 s in 1.12 GiB

#### Scenario: the wide-scan call sites do not diverge on which ceiling is in force
- **WHEN** the bench script's `check_limits` call sites are read
- **THEN** each hands its ceiling over (found by `re.findall`, each passing `**`), so the
  estimator's rungs and the run path cannot be priced against two different cell ceilings
