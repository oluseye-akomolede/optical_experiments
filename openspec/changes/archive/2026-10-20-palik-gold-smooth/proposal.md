# Palik gold, smoothed: no spike between the table's rows, and a band guard where runs are made

## Why

`af2c616` added `Au-palik`, fitted so every one of the twelve copied table rows
was reproduced within 2.11%. Row accuracy was not enough. The fit used a
Lorentz pole at 1.017 eV with damping 1e-5 eV — a nearly undamped resonator
sitting between the 1127 and 1240 nm rows, where the table has no row to
outvote it. On a 0.05 nm grid its permittivity reaches |eps| = 7.3e4 at
1197.2 nm (-2.5e4 + 6.9e4 i), against the table's own |eps| of 63-77 there: a
spurious high-Q feature exactly in the band Run C's ring-down reads. A pulse
that asks in that gap would see a mode that is not the object's.

The fit also trusted points it should not: the table's last three rows (1239.8,
1265.1, 1291.5 nm) sit closest together of any in the table and are
non-monotonic (k runs 8.77, 8.09, 8.25), so a fit asked to hold them to 3%
buys that accuracy elsewhere. Under the damping floor and the smoothness
envelope it misses them by 4.12%, 19.18% and 20.37%, so the fitted range ends
below them.

## What Changes

- `scripts/fit_palik_gold.py` fits under a **damping floor** (0.05 eV on every
  pole, so nothing is high-Q) and a **one-sided envelope**: on a 2 nm grid over
  600-1400 nm, the model's `|eps|` may not exceed 1.5x the linear interpolation
  between the table's neighbouring rows. It fits only the rows at or below
  1150 nm (`--fit-max-nm`, the default), and reports the dense-grid
  diagnostics it is accepted on.
- `fdtd.PALIK_1985["Au-palik"]` is the new fit: worst row error **1.4951%
  (1.50% as the fit prints it) at 652.5 nm** over the fitted range 1.10-1.90 eV
  (653-1127 nm); on the 2 nm
  dense grid Re eps over 1100-1300 nm runs **-83.0 to -58.7** (all negative;
  max Re over 600-1400 nm is -1.31 at 600 nm, min Im is 1.17 at 672 nm), and
  the worst `|eps|` envelope ratio is **0.80**, at 1291 nm. The literal's
  comment records the table, the fit settings, these measurements and the
  reproduce command.
- **Scope change, stated:** the old spec promised every copied point within 3%.
  The three rows past 1150 nm are no longer fitted; their measured errors are
  **4.12% (1239.8 nm), 19.18% (1265.1 nm) and 20.37% (1291.5 nm)**, all outside
  the band the model accepts (a band below 1.10 eV is refused), and the
  dense-grid criterion carries them instead. This is a deliberate narrowing,
  recorded as a MODIFIED requirement with a spec delta.
- `fdtd.source_band(frequency, profile, bandwidth)` is one definition of a
  pulse's band (its spectrum above `BAND_FLOOR` of the peak; `cw` and `drawn`
  ask nothing; a band reaching zero is refused), and
  `fdtd.check_scene_band(materials, request)` judges the library metals a scene
  names. `web/td_form.plan_request` (parts and coatings, so the refusal reaches
  the uploader as text), `worker/td_solve.solve_td` (before layout, for jobs
  that bypass the form) and `scripts/paper1_nanowire.py` (through the same
  `source_band` call and the same refusal) all refuse a band outside a metal's
  fitted range. The bench takes its band from the same definition and its
  header names the selected model.
- Tests: a dense-grid test (passive every 2 nm, `|eps|` within the envelope,
  bounded through the 1127-1240 nm gap); row accuracy over the fitted rows
  asserting the measured maximum as well as the 3% gate; the band definition
  and the guard's scope; web and worker refusals with `pec`/index scenes
  unaffected; the paper-1 script refused with the app's own numbers.

## Consequence for Run C

Under the corrected guard, Run C at `--band-half 100` is refused: at 900 nm its
1% spectrum is 1.045-1.710 eV, outside `Au-palik`'s 1.10-1.90 eV fit. The
widest accepted pulse is `--band-half 83` (floor 1.1026 eV); 84 nm reaches
1.0993 eV and is refused. Whether Run C narrows its pulse or the fit is
widened is the lead's decision and is not made here.

## What Does Not Change

The Rakić table and its `Au` default are untouched; `Au-palik` is opt-in via
`--gold-data palik`. The fit's form (Drude plus four Lorentz poles, `eps_inf=1`,
positive `f` and `Γ`) is the solver's existing ADE form, so the stability test
is unchanged in kind.
