# mesh-upload

## ADDED Requirements

### Requirement: The job page SHALL render every material value its stored table can hold

A job's page SHALL show the material table from the job's STORED materials and
SHALL render every value that table can hold without failing: a refractive index
(a number, formatted `%g`), the string `pec`, or a dispersive metal's name. A
value the page does not recognise SHALL be shown as its own text rather than
raising, because a page that cannot be opened at all is the worse failure. The
`pec` and metal spellings are the ones the time-domain form itself uses, so a
reader meets the same words in the form and on the page. A cavity job's table
SHALL keep its "Refractive index" column and its wording; a time-domain job's
table SHALL label its column "Material" and say that a metal name is the
solver's dispersive model and `pec` is a perfect conductor. Source:
`web/app.py` (`material_label`), `web/templates/job.html`.

#### Scenario: a time-domain job with a metal part renders
- **WHEN** a stored `td` job's materials are `{"substrate": "Au", "film": 1.5,
  "top_lid": "pec"}`
- **THEN** the page renders (not a 500), showing `Au`, `1.5` and
  "perfect conductor (pec)" each against its own part, under a column labelled
  "Material"

#### Scenario: a cavity job is unchanged
- **WHEN** a stored cavity job's materials are `{"wall": 1.0, "core": 2.0}`
- **THEN** the table shows `1` and `2` under a "Refractive index" column, as it
  did before this change

#### Scenario: an unrecognised value does not take the page down
- **WHEN** a stored table holds a string that is neither `pec` nor a metal name
- **THEN** the page renders it as its own text instead of raising

### Requirement: A time-domain job's page SHALL show each part's coating

A time-domain job's page SHALL show the coatings stored in the job's REQUEST —
not in the materials table, because a part's base material and the film on it are
different things and only the base material enters the solver's layout table —
each coating with its material and its thickness in the unit the job was typed
in. A part with no coating SHALL NOT be shown as coated. A coating value this
build cannot read SHALL be shown as it stands rather than raising. Source:
`web/app.py` (`_coating_rows`), `web/templates/job.html`.

#### Scenario: a coated part shows its coating
- **WHEN** a stored `td` job requests a 20 nm coating of `Au` on part `film` and
  the job's unit is `um`
- **THEN** the page shows `film` with coating `Au` and thickness `0.02 um`

#### Scenario: an uncoated job shows no coating table
- **WHEN** a stored `td` job's request has no coatings
- **THEN** the page shows no coating block
