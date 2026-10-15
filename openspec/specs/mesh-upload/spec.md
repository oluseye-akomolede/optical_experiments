# mesh-upload Specification

## Purpose
How the owner's .obj uploads are accepted and turned into solver input: file type, units and scale, and splitting multi-part files into named parts.

## Requirements

### Requirement: The app accepts `.obj` uploads only

The app SHALL accept an upload named `meshfile` with suffix `.obj` and no other
format. Source: `web/meshing.py`, `web/config.py`.

#### Scenario: a non-obj file is uploaded
- **WHEN** the uploaded file's suffix is not `.obj`
- **THEN** the upload is refused with a message naming the supported format

#### Scenario: the file is empty
- **WHEN** the uploaded file has no bytes
- **THEN** the upload is refused as empty

#### Scenario: the file is over the size limit
- **WHEN** the uploaded file exceeds 50 MB
- **THEN** the upload is refused with the limit in the message

### Requirement: Units and scale are applied once, at the form

The object's units SHALL come from a fixed table
(`m`=1, `cm`=1e-2, `mm`=1e-3, `um`=1e-6, `nm`=1e-9, `in`=0.0254). A free scale
SHALL multiply every vertex before anything measures the object. The declared
smallest feature SHALL carry its OWN unit, chosen from a shorter list
(`nm`, `um`, `mm`) independent of the object's units. Source:
`dockerfiles/maxwell_solver/mse_common.py`, `web/app.py`, `web/config.py`.

#### Scenario: an object in mm with a feature in um
- **WHEN** a 10 mm object declares a 10 um feature
- **THEN** the feature's metres are computed with its OWN unit (1e-5 m), not the
  object's

#### Scenario: a scale is applied
- **WHEN** a scale factor is supplied
- **THEN** it multiplies the vertices once
- **AND** the estimate computes size x scale x unit_scale, never applying the
  unit twice

### Requirement: A multi-part `.obj` is split by named groups

The app SHALL split a multi-part upload by its `o`/`g` group names, and SHALL
refuse an upload whose named groups and closed parts do not correspond
one-for-one. The split SHALL use connected components (union-find on face
adjacency), not a graph engine. Source: `dockerfiles/maxwell_solver/mse_parts.py`.

#### Scenario: a group name is repeated
- **WHEN** the `.obj` names the same group more than once
- **THEN** the upload is refused, naming the duplicate

#### Scenario: faces outside any group
- **WHEN** the file has faces belonging to no named group
- **THEN** the upload is refused, telling the user to add an `o <name>` line

#### Scenario: unseen faces before the first group
- **WHEN** faces appear before any `o`/`g` line
- **THEN** they are a part named `(unnamed)`, and the count check still applies

### Requirement: Every part needs a refractive index

For a multi-part upload, the app SHALL require a refractive index for every
part, named by the part's name (not its index), and SHALL refuse an unknown part
name. Indices SHALL be numbers between 1.0 and 10.0. Source: `web/app.py`.

#### Scenario: a part is missing its index
- **WHEN** an upload names parts but omits one's refractive index
- **THEN** the upload is refused, naming the part(s) without an index

#### Scenario: a form names a part not in the upload
- **WHEN** the form supplies an index for a name not among the upload's parts
- **THEN** the upload is refused, listing the upload's actual parts

#### Scenario: an index is out of range
- **WHEN** a part's refractive index is below 1 or above 10
- **THEN** the upload is refused with the allowed range

### Requirement: An open surface is refused before solving

The app SHALL refuse an upload whose surface is not watertight, with a message
explaining that the object is treated as a cavity and must be sealed. Source:
`web/meshing.py`.

#### Scenario: the uploaded object has holes
- **WHEN** the surface is not closed
- **THEN** the upload is refused before any job is created

### Requirement: The upload is estimated and routed, or refused as too detailed

The app SHALL estimate the element count from the FILLED volume, by the
tetrahedral volume law -- `6*sqrt(2)` tetrahedra per `h^3` of filled volume at
edge `h` (`mse_common.TETS_PER_UNIT_VOLUME`) -- times a margin MEASURED over the
shape families the benchmark builds, and SHALL:

- ROUTE the job to the GPU worker when the estimate reaches the GPU threshold;
- otherwise ROUTE it to the cluster;
- REFUSE with a plain-English message when the estimate exceeds the refusal
  threshold: the real cap (600,000 elements) times `UPLOAD_REFUSE_FACTOR`.

The edge SHALL be the effective edge the MESHER applies, and on a multi-part
upload that is ONE edge for the whole cavity: the mesher computes a single
`maxvolume` from the finest region's narrowed edge and registers its regions by
seed with no per-region volume, so an estimate charging each region its own
phase edge would count a mesh the mesher never builds.

The margin SHALL be measured, and SHALL NOT be presented as a bound over the
product. What the mesher DELIVERS is not fixed by what it is asked for: TetGen
treats `maxvolume` as a ceiling and refines below it for quality, and the shipped
sweep at the product's own cap measured deliveries from 0.715x to 8.952x of the
volume law, so the estimate reads from 0.15x to 1.89x of the real mesh across
those families. The 2.10x reading is a SECOND setting -- the cube at half the
mode count -- and travels with it, not as part of the shipped band. The margin
SHALL be the worst under-estimate it covers (1.298 ->
1.35), and every family measured ABOVE it SHALL be recorded with its own numbers
and the direction it misses, because an estimate that reads low routes a job the
machine chosen for the predicted size cannot run.

The benchmark SHALL run at the PRODUCT's element cap -- `config.max_tets()`,
600,000 -- not at a smaller literal, and SHALL say in its header which cap and
which margin it used. A benchmark cap below a fixture's unconstrained count
measures the mesher's over-cap retry, which coarsens `maxvolume` and rebuilds, so
the number it reports is a mesh the product would never build. A smaller cap is
not a cheaper fixture.

The benchmark SHALL be runnable at more than one setting (``--family-scale-n``)
so the band can be measured rather than assumed, and the worst measured under-
and over-estimates SHALL be recorded with the settings and the family that
produced each. Both directions SHALL be recorded: a family whose delivery is
coarser than the law (the plate at 0.715x, where the estimate reads 1.89x of the
real mesh) and the families whose delivery is finer (the 3-region nested-thin-
sheet family at 8.952x, where the estimate reads 0.15x, recorded as a mesher
defect rather than answered by a larger margin).

The refusal factor SHALL be a product trade written down where the factor lives,
not a derivation from the estimate's over-read: over the measured spread
(estimates from 0.15x to 2.10x of the real mesh across the settings measured, the
top reading being the cube at half the mode count), no single factor both avoids
refusing a mesh that fits and avoids accepting one that does not. A factor at or
above the top of the band (2.1) admits an upload whose estimate is tight at up to
2.1 x 600,000 = 1.26M real elements, which the mesher's own cap then coarsens to
fit -- a quality loss, not the memory exhaustion the narrower factor risks. The
error is not one-sided: a family that reads LOW (measured 0.15x and 0.71x) is
ACCEPTED with its estimate inside the threshold while its real mesh can run to
1 / 0.15 ~ 6.7x that estimate, and only the mesher's over-cap retry keeps the
delivered count inside the cap. The factor bounds the ESTIMATE, not the delivery.
Wrongly ACCEPTING an oversized mesh is the worse failure and is the one not
chosen. Raising the refusal threshold from main's 727k to this change's 889k
law-elements also WIDENS the band in which the mesher coarsens: main retries
every mesh over the product's 600,000-element cap too, so this change extends
that retry band by 14 / 11.455 = 1.22x, and more uploads reach the retry whose
only lever is a larger `maxvolume`. The factor SHALL stay at the 2 the app ships
until the mesher's delivered edge is narrowed; the numbers, both failures, and
the reason are recorded with the constant in `web/config.py`.

The estimate SHALL return `0`, meaning "do not route on this", when the surface
is unusable or the formula fails -- not a small number. Source: `web/meshing.py`,
`dockerfiles/maxwell_solver/mse_common.py`, `web/config.py`.

#### Scenario: the object is too detailed
- **WHEN** the estimate exceeds the refusal threshold
- **THEN** the upload is refused, naming the estimate and the cap, and asking
  for a simpler model

#### Scenario: the estimate could not be computed
- **WHEN** the surface is unusable or the formula raises
- **THEN** the estimate is `0` and the job is not routed on that value

#### Scenario: a multi-part upload is estimated
- **WHEN** the upload has multiple parts, or a region denser than the wall
- **THEN** the estimate sums the FILLED volumes (a nested core is not
  double-counted) and applies the smallest effective edge to the whole cavity,
  which is the one edge the mesher's `maxvolume` is built from

#### Scenario: the estimate is compared against real meshes
- **WHEN** the benchmark builds every shape family (single part, 2-3 regions,
  thin and thick shells, off-centre, non-spherical) at the product's cap
- **THEN** the estimate-to-built ratio is reported per family, the worst reading
  in each direction is named with the family that produced it, and no family is
  asserted to sit at or above its mesh on the strength of the margin alone

#### Scenario: the estimate is shown to be better than the formula it replaced
- **WHEN** a test asserts that the replaced formula read worse than the new one
- **THEN** it does so by meshing a real fixture through the product's own calls
  and recomputing the old formula from that same mesh and edge, not against a
  hand-written element count, which would keep passing while the mesher changed
  under it

#### Scenario: the mesher delivers a mesh the request did not predict
- **WHEN** a family's built count lands more than the band above the law (the
  3-region nested-thin-sheet fixture: requested edge 0.3353 mm, 94.3% of its
  tetrahedra more than an octave finer, 419,176 elements built where the law
  says 46,825)
- **THEN** the finding is recorded in the change with the fixture's numbers and
  the mesher is left unmodified: covering it with the margin would read the
  other families 6-12x high, so narrowing the spread is a separate change

#### Scenario: the benchmark says which setting its numbers came from
- **WHEN** the benchmark is run at another setting (a different mode count or
  cap), or its numbers are quoted anywhere
- **THEN** the family, the setting and the estimate-to-built ratio travel
  together, and a ratio measured at one setting is not stated as a bound over
  the others

### Requirement: Refusals reach the user as plain English

Every refusal SHALL be a plain-English message shown to the user (a flash or a
JSON error), and SHALL NOT leak a stack trace or an internal token. Source:
`web/app.py`, `web/meshing.py`.

#### Scenario: a refusal is shown
- **WHEN** any upload validation fails
- **THEN** the user sees a message explaining what to change and how

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
each coating with its material and its thickness in the unit the coating was
typed in. A coating thickness SHALL be stored with that unit: the request SHALL
carry the time-domain form's own length unit (the one the thicknesses and cell
size were typed in), because the stored metre value does not say what unit it
came from, and the page SHALL divide by that same unit. The page SHALL NOT use
the object's upload unit, which is a different field of the same form. A request
stored before that unit existed has only metres, and the page SHALL then show the
value in metres rather than presenting a unit it cannot justify. A part with no
coating SHALL NOT be shown as coated. A coating value this build cannot read
SHALL be shown as it stands rather than raising. Source: `web/app.py`
(`_coating_rows`), `web/td_form.py` (`parse_request`), `web/templates/job.html`.

#### Scenario: a coated part shows its coating in the form's unit
- **WHEN** a stored `td` job was submitted with the time-domain form's length
  unit `nm` and the object's unit `um`, and requests a 10 nm coating of `Au` on
  part `film`
- **THEN** the page shows `film` with coating `Au` and thickness `10 nm` — the
  unit of the field the number was typed in, not the object's `um`

#### Scenario: a request with no stored unit shows metres
- **WHEN** a stored `td` job's request carries a coating thickness but no
  time-domain length unit, and its object unit is `um`
- **THEN** the page shows the thickness in metres, and not as a number in `um`

#### Scenario: an uncoated job shows no coating table
- **WHEN** a stored `td` job's request has no coatings
- **THEN** the page shows no coating block
