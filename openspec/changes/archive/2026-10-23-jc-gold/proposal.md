## Why

The owner approved the paper-1 run with Johnson & Christy's gold. The library
today offers Rakić's `Au` (the default) and Palik's `Au-palik`; the 1972
Johnson & Christy table is the source the owner chose for this run, and it is
a different measurement: at the paper's 905 nm its own rows interpolate to
Im eps 2.040 against Palik's 2.630, 22% less loss. Without a third model the
approved run can only be answered with a table the owner did not pick.

The fitter that produced `Au-palik` already does everything the second table
needs except name it, so this change generalises that script instead of
copying it.

## What Changes

- `data/jc_au_nk.csv`: the 1972 table's 9 points between 600 and 1300 nm
  (616.8-1216.0 nm) with the table's citation in the header, as data only.
- `fdtd.JOHNSON_CHRISTY_1972 = {"Au-jc": Metal(...)}`, merged into `LIBRARY`
  and `METALS`, so the `_metal` lookup, the layout and voxel checks, the web
  form's material list and the skin-depth rule accept it with no further code.
- `scripts/fit_palik_gold.py` takes `--data`, `--name` and `--source`; with no
  arguments it still prints the Palik literal it printed before (verified: the
  same parameters and the same 1.50% worst error).
- `scripts/paper1_nanowire.py` takes `--gold-data jc`, mapped to `Au-jc` in
  `gold_metal`, so the scene's mirror, the layout, the estimate and the report
  all read the same name.
- Tests: the table fits within the 3% gate (0.4080% at 892.0 nm), the model is
  smooth between rows (0.9919-1.0024x of the row interpolation inside the row
  span), passive over 600-1400 nm, stable at
  the bench's 2-20 nm spacings, its band refusal names its own table, and at
  905 nm its Im eps is 0.816x Palik's.

## Impact

- Affected specs: `time-domain` -- the dispersive-metals requirement and the
  bench's selection requirement are modified; the Johnson & Christy model is
  added as a requirement.
- Affected code: `dockerfiles/maxwell_solver/fdtd.py`,
  `scripts/fit_palik_gold.py`, `scripts/paper1_nanowire.py`,
  `tests/test_fdtd_metals.py`, `tests/test_paper1_nanowire.py`, and the new
  `data/jc_au_nk.csv`.
- No default changes: `rakic` stays the default, `Au-palik` and its literal are
  untouched, and the web form's behaviour changes only by one more selectable
  metal.
- No physics is re-measured here: the numbers are fits to a published table and
  no solve is run.
