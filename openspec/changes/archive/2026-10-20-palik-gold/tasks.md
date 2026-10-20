# palik-gold tasks

## 1. Data and fit

- [x] Copy the table's 600-1300 nm rows into `data/palik_au_nk.csv`, sourcing it
      to Palik 1985 via IMD (verified against the downloaded `.nk` file)
- [x] Fit a 4-pole Lorentz-Drude `Metal` (Drude f = 1, eps_inf = 1) with every
      f and Gamma positive, keeping `scripts/fit_palik_gold.py` to reproduce it
- [x] Record the fit's max relative eps error at each table point, and confirm
      the ADE growth check passes at 2-20 nm gold (max 2.11% at 1291 nm;
      growth check 23 passed in `tests/test_fdtd_metals.py`)

## 2. Library and lookup

- [x] Add `fdtd.PALIK_1985["Au-palik"]` with the literal, the source, the fit
      range and the max error in its comment
- [x] Make `_metal` and `METALS` see both libraries; keep `RAKIC_1998` untouched
- [x] Make `check_metal_band` name the model's own source

## 3. Bench flag

- [x] `--gold-data {rakic,palik}` (default rakic) selects the scene's mirror
      material and the skin depth the report prints
- [x] The report logs which model the run used (`gold` line); `--estimate`
      prices the same material (the layout's `metals` lookup)
- [x] Every other path that names the metal follows the choice: the skin-depth
      `stack` line and the estimate's gold-region check both go through
      `gold_metal(args)`; the `caps` line names no metal

## 4. Tests and validation

- [x] `tests/test_fdtd_metals.py`: every table point within 3%, Im eps > 0,
      poles positive, the Rakić `Au` values unchanged, the outside-band refusal
      names Palik, and the ADE growth check at bench spacings (23 passed,
      8.0 s)
- [x] `tests/test_paper1_nanowire.py`: the flag selects the right material,
      the default is Rakić, and the report names it (whole file: 38 passed,
      24:44)
- [x] Run only the affected test files at tier 1 and record the counts
      (`test_fdtd_metals.py`: 23 passed, 8.0 s; `test_paper1_nanowire.py`:
      38 passed, 24:44, both at the committed tree, `ulimit -v 4194304`,
      `nice -n 19`)
- [x] `openspec validate palik-gold --strict` (valid; `--specs --strict`:
      12 passed, 0 failed)
- [x] Commit locally (no push), report the sha and the fit's max error
