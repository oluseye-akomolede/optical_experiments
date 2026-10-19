## ADDED Requirements

### Requirement: The memory ceiling SHALL be a per-run argument, raisable only by the bench script

`fdtd.check_limits` SHALL take a fourth ceiling, `max_bytes`, alongside `max_steps`,
`max_seconds` and `max_cells`. Its default SHALL be `fdtd.MAX_BYTES` read at call time, and
its refusal SHALL name the ceiling that actually applied rather than the product's constant.
`fdtd.MAX_BYTES` SHALL keep its value (3.0e9 bytes) and its meaning: the product's own
ceiling is unchanged, and only a measured bench card is measured against a raised one.

The paper-1 bench script alone SHALL be able to raise it, through `--max-bytes GB`
(gigabytes, so `--max-bytes 14` is 14e9 bytes). The flag SHALL only raise: a value under the
product's own SHALL be an argparse error (exit 2), and `run_limits` SHALL clamp with `max()`
so a namespace that never met the parser reads the product's own. The web app and the k8s
Job path SHALL NOT be able to raise it: `worker/td_solve.py` and the web app SHALL call
`check_limits` with no ceiling, and no environment variable and no module global SHALL set
one. A run that used a raised memory ceiling SHALL say so on its `caps` log line, naming the
memory ceiling in the same parenthetical that names raised steps, seconds and cells.

The refusal this lifts was measured: the paper's Run B at dx 5 nm on the 16 GB A4000
estimates 11.55 GB of GPU memory and was refused after about two hours of setup under the
product's 3.0 GB ceiling.

#### Scenario: The default refuses what the card cannot do under the product's ceiling

- **WHEN** an estimate of 11.55e9 bytes is checked with no `max_bytes`
- **THEN** it SHALL be refused, and the refusal SHALL name "over the limit of 3.0 GB"

#### Scenario: A raised ceiling is the one the refusal names

- **WHEN** the same estimate is checked with `max_bytes=14e9`
- **THEN** it SHALL be admitted
- **AND** an estimate of 14e9 + 1 bytes SHALL be refused, naming "over the limit of 14.0 GB"
  and not the product's 3.0 GB

#### Scenario: The flag only raises

- **WHEN** the bench parser is given `--max-bytes 2`
- **THEN** it SHALL exit 2 with "may only raise a cap"
- **AND** a namespace carrying `max_bytes=1e9` SHALL resolve through `run_limits` to
  `fdtd.MAX_BYTES`, exactly as the other three ceilings do

#### Scenario: A memory-only raise is reported as a bench ceiling

- **WHEN** `report` runs with only `max_bytes` raised
- **THEN** its `caps` line SHALL carry the bench-ceiling parenthetical and name the memory
  ceiling (for example "14.0 GB")
- **AND** with no ceiling raised the same line SHALL carry no parenthetical
