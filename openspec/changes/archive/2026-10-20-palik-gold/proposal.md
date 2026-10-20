# Palik gold: a second gold model, fitted to Palik's own table

## Why

Paper 1 (the nanowire) states its gold comes from Palik's 1985 handbook, and its
Lumerical scene fits a dispersion model to that table. Our runs have used the
Rakić et al. 1998 Lorentz-Drude fit, which is a different measurement: at 900 nm
Palik gives eps = -36.1 + 2.59i and Rakić -27.6 + 2.45i. A mirror that absorbs
4.4% per bounce instead of 3.2% is a plausible part of the gap between our
F = 14 and the paper's 17, so Run C's F measurement should be able to use the
paper's own gold.

The owner chose this (Slack, 2026-10-20 14:32Z, reply "1"): use Palik's gold for
Paper 1, as the paper does.

## What Changes

- A new data file `data/palik_au_nk.csv`: the twelve Palik table rows between
  600 and 1300 nm (n, k against wavelength in Å), with the source cited
  (E. D. Palik, *Handbook of Optical Constants of Solids*, 1985, gold, p. 290,
  as distributed by IMD).
- `scripts/fit_palik_gold.py`: the fit that produces the model, so the literal
  in `fdtd.py` can be reproduced rather than trusted.
- `fdtd.PALIK_1985["Au-palik"]`: a `Metal` of the existing Lorentz-Drude form
  (Drude pole `f = 1`, four Lorentz poles, `eps_inf = 1`), every `f_j` and
  `Γ_j` positive so the ADE stays stable, fitted to the table's 653-1291 nm
  points with a worst relative complex-permittivity error of 2.11% (at
  1291 nm). `Metal` gains a `source` field so a band refusal names the table
  that model came from.
- `fdtd.LIBRARY` / `METALS` / `_metal` know both models. `RAKIC_1998` is
  byte-identical, `"Au"` remains the Rakić default, and `check_metal_band`
  still refuses out-of-fit bands, now naming the right source for either model.
- `scripts/paper1_nanowire.py`: `--gold-data {rakic,palik}` (default `rakic`).
  The flag selects the scene's mirror material, the skin depth the stack line
  reports, the material the estimate prices, and the model the report names.
- Tests: the fit reproduces every copied Palik point within 3% (and states the
  measured maximum); the model is lossy (Im eps > 0) over the fitted band and
  stable at the bench's 2, 5, 10 and 20 nm spacings; the Rakić table is
  unchanged and both names are listed; a Palik band outside the fit is refused
  naming Palik; the flag selects the priced and reported material.

## Out of scope

- Rakić stays the default; no existing run's numbers move unless the flag is
  passed.
- `eps_inf` stays 1 (the ADE is built on it); the fit absorbs the rest into the
  poles.
- The Palik table's 5635-6525 Å gap is not interpolated; the fit covers only
  the points that exist, and its declared range (0.96-1.9 eV) is those points.
- Rebuilding the paper1-bench-cuda image and Run C's F measurement are the
  lead's, after this is reviewed and merged.

## Impact

- Affected spec: `time-domain` — two ADDED requirements (the second model, the
  bench flag) and three MODIFIED ones: the dispersive-metals requirement (the
  library is no longer Rakić-only), and the voxel/layout requirements whose
  accepted-name rules now say "library metal" because they read `fdtd.METALS`,
  which now has both names. Their text is otherwise reproduced unchanged.
- Affected code: `dockerfiles/maxwell_solver/fdtd.py`,
  `scripts/paper1_nanowire.py`, `scripts/fit_palik_gold.py`, `data/`.
- Affected tests: `tests/test_fdtd_metals.py`, `tests/test_paper1_nanowire.py`.
- No change to the solver's numerics, the DAGs, the worker, or the API.
