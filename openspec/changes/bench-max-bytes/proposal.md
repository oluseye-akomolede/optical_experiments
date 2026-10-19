# bench-max-bytes

## Why

The paper's Run B (dx 5 nm) was refused on the A4000 after about two hours of setup:
"The run needs about 11.55 GB of GPU memory, over the limit of 3.0 GB". `fdtd.check_limits`
takes `max_steps`, `max_seconds` and `max_cells` overrides but always compares `e["bytes"]`
against the module constant `fdtd.MAX_BYTES`; the bench script has no flag that raises the
memory ceiling, so the record cannot even be priced at that resolution under the card's real
16 GB. The 3.0 GB ceiling is right for the product (a Jetson-class worker) and stays exactly
what it is; this change gives the bench path the same per-run permission the other three
caps already have.

## What Changes

- `fdtd.check_limits(..., max_bytes=None)`: a fourth per-run ceiling, defaulting to
  `MAX_BYTES` read at call time like the other three; the refusal's text and `exceeded`
  tuple name the ceiling that applied, not the module's constant.
- `scripts/paper1_nanowire.py`: `--max-bytes GB` (gigabytes, so `14` is 14e9 bytes), built
  with `_raised_cap` so it may only raise a ceiling; `run_limits` clamps with `max()` and
  carries `max_bytes`; both `raised` computations read it, and the `caps` line's bench-ceiling
  parenthetical names the memory ceiling it was measured against.
- Spec: time-domain gains the memory-ceiling requirement beside the cell ceiling's.
- Tests: the default refuses 11.55e9 and names 3.0 GB; `max_bytes=14e9` admits it; one byte
  over the raised ceiling names 14.0 GB; moving `MAX_BYTES` moves the default; the parser
  refuses `--max-bytes 2` and `run_limits` clamps a hand-built namespace; a memory-only
  raise prints the parenthetical, and the plain call prints none.

## Out of scope

- Moving the memory check ahead of the voxelisation, so an over-ceiling record fails in
  seconds instead of after setup. Follow-up, not this change: it moves where a product
  refusal happens.
