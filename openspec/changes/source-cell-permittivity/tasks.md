# Tasks — source-cell-permittivity

- [x] 1. Read how `simulate` builds the E coefficients from the layout's
  `eps` (proposal, with line numbers): the node value at the component's own
  array index, no averaging in the solver, and the Lorentz-Drude rescale to
  `eps_inf = 1` on a metal's samples.
- [x] 2. `fdtd.py`: `LorentzDrude.samples` (one mask rule, used by `bind`),
  `LorentzDrude.applied_permittivity`, `source_permittivity`, and
  `simulate`'s `"source_eps"`; the readers' docstrings name it.
- [x] 3. `scripts/paper1_nanowire.py`: read with the applied value; the
  `source` line prints the cell and the applied permittivity.
- [x] 4. `tests/test_fdtd_source_eps.py`: (a) homogeneous dielectrics apply
  their cell value to the bit and `delivered_power` is unchanged bit for bit;
  (b) on a 37^3 open grid against the phasor box's surface power, a
  dielectric face reads true with the cell value (+0.0023, vacuum +0.0016),
  and a lossless metal face reads true only with the applied value (+0.0008
  against +3.0032 with the cell's); which samples a metal claims; the claim
  is the mask `bind` rescales. `tests/test_paper1_nanowire.py`: the source
  line and the returned value.
- [x] 5. `tests/check_discriminating.py`: `source-eps-cell-value`,
  `source-eps-wrong-component`, `source-eps-simulate-no-terms`,
  `paper1-eps-cell`, `paper1-eps-print`, each turning exactly its tests red.
- [x] 6. Verification: `OMP_NUM_THREADS=2 python -m pytest
  tests/test_fdtd_source_eps.py tests/test_fdtd_purcell.py
  tests/test_paper1_nanowire.py tests/test_fdtd.py tests/test_fdtd_metals.py
  tests/test_fdtd_phasor_box.py -q`;
  `npx -y @fission-ai/openspec validate source-cell-permittivity --strict`.
