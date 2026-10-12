# The job page renders every material it stored

## Why

Live web (`a8d53a3`, and still `0966b10`) returns HTTP 500 for the owner's
time-domain job `180cb65f` (an Au / 1.5 / Au film). The traceback is the job
page: `templates/job.html` formats every stored material with `'%g'|format(n)`,
and a `td` job's stored table holds far more than a number.

`web/td_form.parse_materials` returns `{part: n | "pec" | <metal name>}` — a
per-part refractive index, the string `pec`, or a dispersive metal's name
(`fdtd.METALS`; the owner's film is `"Au"`). The cavity-era table assumed every
value was a number, so the first metal raised
`TypeError: must be real number, not str` and the page could not be opened at
all. The owner cannot see the job they just ran.

This was latent from the moment `td` materials landed: the same value reaches
`job.materials` for every time-domain job whose part is a metal or `pec`.

## What changes

- **One value → text mapping** (a Jinja filter, `material_label`, in
  `web/app.py`): a number keeps `%g`; `pec` becomes "perfect conductor (pec)",
  the wording the td form itself uses; any other string is shown as it is. The
  last arm is a fallback, not a raise — an unrecognised value renders as its
  own text, because a page that will not open is the worse failure.
- **The table names its kind.** A time-domain job's column is "Material", with a
  note that a metal name is the solver's dispersive model and `pec` is a perfect
  conductor; a cavity job keeps its "Refractive index" column and its
  "each meshed to its own resolution" note, unchanged.
- **Coatings are shown.** A coating is not in `job.materials` (it lives in the
  stored `tds` request — a part's base material and the film on it differ, and
  only the base material enters the solver's layout table), so a coated part
  currently reads as uncoated. A time-domain job's page now lists each coating
  with its material and its thickness in the unit the job was typed in.
- **A red-first test** in `tests/test_web_routes.py` that renders
  `job.html` for a stored td job with a metal part through the REAL template,
  plus one for a coated part.

## What this does not change

- Any solver, worker or storage behaviour. Nothing is written differently; only
  the page's rendering of a value the app already holds changes.
- A cavity job's page: same column, same wording, same `%g` numbers.
- The `materials` column's contents or its writers (`web/td_form.py`,
  `web/app.py`): the page meets the data where it is.
