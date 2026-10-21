# Tasks — paper1-axis-probes

- [x] 1. Measure the line on Run C's own scene (proposal): plan its dx-10
  request twice, once without probes and once with the line placed from the
  first plan's nodes, and compare the two node lists (237 z nodes; before the
  face fix 148 layers and 444 probes from 150 nm). The line was then moved off
  the face node: the layout's point-in-part test reads a node within float
  noise of the gold's top face as on the mirror, so the line starts at the
  first node a millionth of a cell above it (see 3). Re-measured on the same
  scene after the fix: 147 layers, 441 probes, 153.000 to 1557.000 nm, all 441
  on their own nodes (worst offset 0.0 m, 0 nearest-node mismatches); the same
  scene's ring-down is 121,121 steps, so the line is 53,414,361 samples,
  213,657,444 B (213.7 MB) against a 3 GB cap.
- [x] 2. `fdtd.py`: `memory_estimate` / `estimate_run` take `probes` and
  `probe_steps`, priced `4 * probes * probe_steps`; `dt`, `steps` and `cells`
  unchanged by probes.
- [x] 3. `scripts/paper1_nanowire.py`: `AXIS_PROBE_ABOVE_CELLS`,
  `AXIS_COMPONENTS`, `axis_probe_points` (one probe per component per z node,
  from just above the gold's top face to three coarse cells above the wire's
  top, inside the absorber), `with_axis_probes` (a copy of the request),
  `planned_scene` (places the line from the plan, plans again and refuses if
  the node lists differ), `axis_probe_record` (the traces, their z read off the
  layout's own nodes, the wire span and the emitter z) and its hook in the save
  site, `axial_profile` / `axial_profile_command` (the table and the optional
  PNG), the `--axial-*` flags, the refusals in `arm_conflicts` (the read arm
  first, so a flag it cannot read is named before that flag's own rule; and
  `--axis-probes` alone is allowed with `--estimate`, which prices the line and
  records nothing), and the arm's early return in `main`, which turns a refused
  record into `refused: ...` and exit 2 rather than a traceback.
- [x] 4. `fdtd_layout.estimate`: passes the planned probe count, so the ladder
  and the run's own check price the line. `worker/td_solve.py`: passes its
  layout's count (0 until the job plans probes; the seam is priced when it is).
  On the Run C plan the estimate prices the same 441 probes at the scene's own
  121,121 ring-down steps: 4 x 441 x 121,121 = 213,657,444 B, the 213.7 MB the
  proposal quotes; the pulse profile's 10,482 steps are the shorter run's price
  (18,490,248 B).
- [x] 5. `tests/test_paper1_axis_probes.py` (13 tests): a probe on the node it
  names (plan and layout), the line's ends against the geometry and the
  absorber, the price (`4 n steps` on both estimate paths), the ladder's
  `--estimate --axis-probes` accepted and lined, a standing wave's antinodes,
  the tail (the record's own `start`, a later one, one past the end), a record
  without traces, the table the arm prints, and the flags the arm cannot read.
- [x] 6. Verification, all tier 1 (`ulimit -v 4194304`, `nice -n 19`, one
  process at a time): `tests/test_paper1_axis_probes.py` 13 passed in 15.68 s;
  `tests/test_fdtd_estimate.py tests/test_fdtd_memory.py tests/test_td_solve.py
  tests/test_td_form.py tests/test_paper1_collection.py` 219 passed in 88.42 s;
  `tests/test_fdtd_layout.py tests/test_paper1_nanowire.py` 110 passed in
  1,807.80 s. All exit 0. `openspec validate paper1-axis-probes --strict`:
  valid; `openspec validate --specs --strict`: 12 passed, 0 failed.
- [ ] 7. Later, on the bench: Run C's next ring-down with `--axis-probes`, and
  `--axial-profile` on its record, to read the antinode positions against the
  emitter's z. Not started here.
