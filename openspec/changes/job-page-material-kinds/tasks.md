# Tasks — job-page-material-kinds

## 1. The value → text mapping

- [x] 1.1 `web/app.py`: register the `material_label` Jinja filter. A number
      (not a bool) formats with `%g`; `pec` becomes "perfect conductor (pec)";
      any other string is shown as it is — the unknown-value arm is a fallback,
      never a raise.
- [x] 1.2 `web/templates/job.html`: every stored material goes through the
      filter (`value|material_label`), replacing `'%g'|format(n)`, which raised
      `TypeError` on the first string.

## 2. The table names its kind

- [x] 2.1 `web/templates/job.html`: a `td` job's column header is "Material"
      and its note explains the metal model and `pec`; a cavity job keeps
      "Refractive index" and the "meshed to its own resolution" note.

## 3. Coatings, which are not in `job.materials`

- [x] 3.1 `web/app.py`: `_coating_rows(job)` reads the stored `tds` request's
      `coatings` and returns part / material / thickness, the thickness shown in
      the unit the job was typed in; a value this build cannot read is shown as
      it stands rather than raising.
- [x] 3.2 `web/templates/job.html`: a `td` job with coatings lists them under
      the materials table.

## 4. Proof

- [x] 4.1 Red-before-fix: `test_a_td_job_page_renders_a_metal_part_without_a_500`
      (red at HEAD as the production `TypeError: must be real number, not str`)
      and `test_a_coated_part_shows_its_coating_on_the_job_page` (red at HEAD as
      the coating block's absence — all-float base materials, so it does not go
      red through its sibling's bug).
- [x] 4.2 The cavity regression (`test_the_job_page_renders_the_material_table_it_stored`)
      still passes: floats keep `%g`, the column and note are unchanged.
- [x] 4.3 The affected suites at tier 1 (tiny, `2` threads, `nice -n 19`).
