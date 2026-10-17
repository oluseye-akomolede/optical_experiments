# Tasks — phasor-box-monitor

- [x] 1. `fdtd.py`: `phasor_box_layers` (the box check: live layers, CPML plus
  the half cell of H reach); `_PhasorBox` (per-face, per-component device
  accumulators, precomputed E and H phase tables, host interpolation to the
  face-cell centres, `farfield.Face` per frequency); `simulate` options
  `phasor_box`, `phasor_frequencies`, `phasor_start`, `phasor_window`, the
  Bloch refusal and the `"phasor_faces"` result; the `simulate` docstring.
- [x] 2. `tests/test_fdtd_phasor_box.py`: on a 37^3 vacuum grid at 16 cells
  per wavelength, the surface power against `delivered_power` and the flux
  box; the far field against sin^2 and its sphere integral against the
  surface power; the face samples against Jackson's dipole with one E and one
  H moment; the device fake; bit-identical runs without the option; refusals
  (CPML, off the grid, malformed options, Bloch). Each docstring names the
  measured numbers and the mutation it catches.
- [x] 3. `tests/check_discriminating.py`: one `phasor-*` case per named
  mutation (half step, DFT sign, u/v swap, one-sided H, corner sample,
  normalisation, one frequency column, flux-box CPML rule, Bloch refusal),
  each turning exactly its tests red.
- [x] 4. Verification: `OMP_NUM_THREADS=2 python -m pytest
  tests/test_fdtd_phasor_box.py tests/test_fdtd.py tests/test_farfield.py
  -q` passes; `python tests/check_discriminating.py phasor-<name>` for each
  case; `npx -y @fission-ai/openspec validate phasor-box-monitor --strict`.
  No image smoke test and no e2e job: no job can request the option.
- [ ] 5. Later, a separate change: the paper-1 run that reads the collection
  fraction within NA 0.8 from these faces, with the box over the gold mirror
  (`pec_mirror_z`, or an open bottom) and the option counted in
  `memory_estimate` if a job is to request it.
