# mesh-upload

## MODIFIED Requirements

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
