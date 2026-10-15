# Tasks — near-field-far-field

- [x] 1. `farfield.py`: the `Face` layout and its checks, `sample_box`,
  `face_points`, `surface_power`, `far_field` / `radiation_pattern`,
  `cone_quadrature` / `cone_power` / `hemisphere_power` /
  `collection_fraction`, `pec_mirror_z` images, `hertzian_dipole` and
  `dipole_power` as the closed-form references.
- [x] 2. `tests/test_farfield.py`: the five checks of the job (z-dipole
  pattern and power, x-dipole pattern, NA 0.8 cone fraction against the
  closed form, halving the spacing, an off-centre dipole in a non-cubic box)
  plus the mirror cases and the layout contract. Each docstring names the
  mutation it catches.
- [x] 3. `tests/check_discriminating.py`: one reversion case per named
  mutation, with the exact set of tests each turns red, so the "goes red"
  claim is one command: `python tests/check_discriminating.py farfield-m-sign`
  (and the other `farfield-*` names; `--list` shows them).
- [x] 4. Verification: `OMP_NUM_THREADS=2 python -m pytest tests/test_farfield.py
  tests/test_fdtd.py -q` passes; `npx -y @fission-ai/openspec validate
  near-field-far-field --strict` passes. No image smoke test and no e2e job:
  nothing is wired into the solver or an image.
- [ ] 5. Later, a separate change: a phasor box monitor in `fdtd.simulate`
  that writes this layout (interpolating the Yee components to the face-cell
  centres), and the paper-1 run that reads 74% within NA 0.8 from it.
