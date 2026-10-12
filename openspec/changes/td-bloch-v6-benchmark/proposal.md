# td-bloch-v6-benchmark

## Why

`time-domain-fdtd` carried benchmark V6 (its task 6.2) as three scenarios under
"A periodic cell SHALL reproduce the crystal's bands". The Bloch core those
scenarios exercise is built and tested (supercell equivalence, energy
conservation, k = 0 identity), but the benchmark's full run on the Jetson has
not passed: the lead's six commands never all exited 0, and the one crystal
number reproduced so far was outside the 1% bound. Archiving the scenarios with
the rest would put unproven behaviour into `openspec/specs/time-domain/`.

## What Changes

- The three V6 scenarios move, word for word, from `time-domain-fdtd` to this
  change as one ADDED requirement.
- No code changes. The Bloch solver API stays as archived; it has no form
  option and no job can request it.

## Impact

- `openspec/specs/time-domain/`: the V6 scenarios are absent until this change
  archives.
- `scripts/fdtd_bench_v6.py`, `tests/test_fdtd_bench_v6.py`: unchanged.
