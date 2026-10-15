# fix-upload-tet-estimate

## Why

`web/app.py` routes every upload (cluster / Jetson / refuse) on
`web.meshing.estimate_tets` *before any mesh exists*, and that estimate runs high
against the mesh the product's own mesher builds. Measured with
`benchmarks/estimate_vs_built.py` (step 1):

| measurement | est/built | built / (6*sqrt(2)*V/h^3) |
| --- | --- | --- |
| 9 distinct recorded matrix cases (two run dirs) | 1.53x - 2.73x, median 1.92x | 0.604 - 1.077 |
| tier-1 one-part 6 mm fixture | 1.65x | 0.998 |
| tier-1 nested fixture, core n=3.5 vs wall n=1 | 10.86x | 0.152 |

An over-estimate is not free: the refusal threshold is a 2x margin on the real
600,000-tet cap (`web/config.py`), and 2 of the 9 recorded cases measure above
2.0x (three-region 2.73x, c1.5-thick 2.24x), so today the site can refuse an
upload whose real mesh fits -- and route the rest to the slower machine on a
number that is not the job's size.

**CORRECTION (2026-10-15, step 2): the last two rows above are retired.** They
were taken with the benchmark's cap set to a 60,000 literal, and the mesher has
an over-cap retry that coarsens `maxvolume` and rebuilds -- so both rows measured
a mesh the product never builds. The nested fixture's real number at the
product's own cap (600,000) is 0.95x / 1.414, not 10.86x / 0.152, and the
"8.88x" and "worst under-estimate 1.047x" in the step-2 direction of 2026-10-15
04:11Z came from the same cap and are withdrawn with it. The benchmark now takes
its cap from `config.max_tets()` and prints it in its header, and the two tables
below are measured at that cap. Step 1's cause analysis stands: it was about the
FORMULA (the fitted `C = 14` and the finest-edge-on-the-whole-cavity rule), and
both are unchanged by the correction. What the correction changes is which family
is the problem: at the product's cap the nested fixture is an UNDER-read (0.95x,
old formula 1.17x), and the worst over-read on the instrument is the coarse plate
(1.89x, old formula 2.31x). The nested fixture's 41,999 built at the 60,000 cap
was itself the coarsened retry. The `--build` one-part fixture above is unaffected
(its mesh is far below any cap), which is why item 1 of the cause list still
reproduces exactly.

Not re-measured this turn: the "9 distinct recorded matrix cases" row is step 1's
reading of the grading bench's own recorded run dirs, taken at whatever cap THAT
bench ran with. It is left in place as recorded, and it is not one of the numbers
this change derives anything from.

The row's own numbers (`75,262` c2-thin, `161,389` c4-thin) are **retired**: the
lead established on 2026-10-15 that they are the grading bench's graded *planned*
count from its analytic table, not this app's estimate on those plans. Replaced
by the nine-family table in this proposal, where both sides are measured.

## Cause (step 1, measured; no product code changed)

The estimate was `est = C * V / h^3` with `C = CFG.est_tets_c() = 14.0` (the live
`mse-web` deployment sets no `MSE_EST_TETS_C`). The mesh it describes does not sit
on that law. A tetrahedralisation at edge `h` holds `6*sqrt(2) = 8.485` tets per
`h^3`; the mesher's own bound (`maxvolume = TETGEN_VOLUME_SLACK * h^3 /
(6*sqrt(2))`, slack 4.0) permits a mesh as coarse as `4^(1/3) = 1.587x` the
requested edge, and its surface facets and quality floors force one as fine as the
case needs. Three measured contributors:

1. **The constant alone:** `14 / (6*sqrt(2)) = 1.650`, exactly the one-part
   fixture's ratio (its mesh landed on the law: `built/volaw 0.998`).
2. **Where TetGen lands in the band**: the recorded cases span
   `built/volaw 0.604-1.077`, i.e. meshes whose achieved edge is 1.18x coarser to
   0.98x (2% finer) than the edge they were asked for --
   `achieved/h = (built/volaw)^(-1/3)`. That spread IS the 1.53x-2.73x spread.
3. **Counting the finest edge everywhere** (the row's hypothesis b): the estimate
   applies ONE `h` -- the narrowest region's, `h * n_wall / n_max` -- to the WHOLE
   cavity volume, so a small dense core makes the estimate `(n_max/n_wall)^3`
   larger than the same cavity meshed at the wall's edge (a core at n=3.5: 42.9x).
   The mesh does not follow it down: the mesher's acceptance guard still tests the
   achieved edge against `MAX_EDGE_OVERSHOOT * edge_m` (`mse_meshing.py:567`)
   measured BEFORE that narrowing (`:506` does the narrowing), so a mesh coarser
   than the edge it was targeted at can pass silently. Step 1 measured this on the
   nested fixture as 418,136 estimated vs 38,489 built, `achieved/h 1.864` --
   **but see the correction at the top: the 38,489 is the 60,000-cap coarsen-to-fit
   retry, and at the product's cap the same fixture builds 358,307 at
   `achieved/h 0.847`, inside the guard's 1.6. The guard MISMATCH is real in the
   code (item 2 of the found issue); the 1.864 that demonstrated it firing is not
   a product-path number.**

**Hypothesis c is excluded, not confirmed:** the estimate does not sum nested
region volumes twice. On the nested fixture the plan's summed `filled_volume_m3`
is 216.00 mm3, exactly the 6 mm box's volume -- the regions partition the cavity --
and the one-part path's `enclosure_volume_m3` for the same box is the same 216.00
mm3.

**No single constant fits.** On the recorded sample, never under-estimating needs
`C >= 8.485 * 1.077 = 9.14` (the c4-thick case), while being tight on a typical
case needs `C = 8.485 * 0.86 = 7.3`. Those are mutually exclusive, so re-fitting
`C` can trade one failure for the other but cannot remove either. The fix must
derive the count from what the mesher controls (its `maxvolume` and the fill it
achieves), not re-fit `C`.

## What Changes (step 2)

- **The formula, with nothing fitted into it:**
  `est = EST_TETS_MARGIN * 6*sqrt(2) * sum_r filled_V_r / h_r^3`
  (`web/meshing.py:estimate_tets`, `web/config.py`, `mse_common.estimate_tets_by_region`).
  The `6*sqrt(2)` is `mse_common.TETS_PER_UNIT_VOLUME` -- the same constant the
  mesher inverts to ask TetGen for an edge -- so the geometry part can only move
  if the mesh definition does. What used to be absorbed in `C = 14` is now the two
  things it really was: the law (1.650 of it) written down, and the margin
  (1.35) measured.
- **The edge is the edge the mesher applies.** The mesher computes ONE
  `maxvolume` for the whole cavity from the plan's `h_min`
  (`mse_meshing.py:506-509`) and registers its regions by seed with no per-region
  volume, so `h_r = h_min` for every region on today's mesher; the estimate sums
  the plan's `filled_volume_m3` per region anyway, so it follows the plan if a
  future mesher gains a per-region edge. `h_min` -- not the wall's edge -- is
  right on both sides of the current code: it is what `maxvolume` is built from,
  and it is what the estimate must charge to route a dense-cored upload to the GPU
  rather than let it die on the CPU.
- **The margin is a measurement, and it says so.** 1.35 is the worst
  UNDER-estimate of the shipped sweep at the product's cap (1.298, the 2-region
  dense core) rounded up ~4%; what it does NOT cover is written beside it with
  numbers in `web/config.py` -- a 3.5x finer core at 1.414 (0.95x), the same
  families at half the mode counts at 1.868 (0.72x), and a three-region
  nested-thin-sheet family at 8.952 (0.15x). The margin is not raised to cover
  those, and the arithmetic is plain: at 1.9 the worst over-read becomes 2.95x (the
  half-mode cube; the plate 2.66x), the plate's wrongful-refusal threshold falls
  from 635,556 to 451,579 real elements -- below the 600,000 cap, so the site
  would START refusing uploads that fit -- and 1.9 still does not reach 8.952, so
  the mesher's family would remain uncovered anyway. What the margin is NOT is a
  bound, and the spec says so.
- **The refusal factor is not re-derived and not widened:** `UPLOAD_REFUSE_FACTOR`
  is 2, main's value, with both failures and their numbers written beside it.
- **Instrument:** `benchmarks/estimate_vs_built.py` gains `--families` (nine
  shape families), `--max-tets` and `--family-scale-n`, so the band can be
  measured at more than one setting instead of assumed. Two new dev-only probes:
  `benchmarks/mesher_edge_cause.py` (what TetGen delivers against the
  `maxvolume` it was given, per region, with a no-region control) and
  `benchmarks/mesher_slack_probe.py` (what a tighter `TETGEN_VOLUME_SLACK`
  changes).
- **Tests on REAL meshes:** `tests/test_estimate_families.py` builds each family
  with the product's own calls and asserts the estimate sits inside the measured
  band -- `>= 1` on the six families whose meshes land near the law, and the two
  recorded under-reads pinned BY VALUE (0.15, 0.72) so a mesher fix or a raised
  margin reddens them rather than passing silently; it computes the replaced
  `C = 14` on the same real mesh and asserts the pair shows the old formula
  refusing a capping upload the new one accepts -- the old formula is the control,
  not a double. The estimate tests in `tests/test_routing.py` and
  `tests/test_web_routes.py` follow the margin API and the two-sided band; no
  `est_tets_c` / `MSE_EST_TETS_C` remains.
- **Spec:** `skip_specs` is removed and
  `specs/mesh-upload/spec.md` modifies *"The upload is estimated and routed, or
  refused as too detailed"*: the "conservative coefficient" it required was the
  cause, and its parenthetical "twice the real cap, because the estimate can be up
  to twice high" is what this measurement replaces (a two-sided trade, written
  with its numbers).
- **The mesher does not change in this change.** The guard finding below is
  recorded for the lead to schedule separately.

## The sweep (step 2, real meshes, tier 1, PRODUCT cap 600,000, margin 1.35)

Both tables are measured at `MAX_TETS = CFG.max_tets()` and reproduced in this
turn's run. `old/built` is the formula this change replaces; `achieved/h` is the
delivered volume-equivalent edge over the requested one.

```
family                     h_min mm   V mm3     est     built  est/built  built/volaw  old/built  achieved/h
one-part 6mm cube           0.6734   216.00    8104      4596     1.76       0.766       2.15      1.122
one-part plate 8x4x2        0.4489    64.00    8104      4294     1.89       0.715       2.31      1.134
one-part sphere r3          0.5412   112.12    8104      7586     1.07       1.264       1.31      0.912
one-part bumpy sphere r3    0.5430   113.26    8104      6357     1.27       1.059       1.56      0.996
2-region core 1.5mm n=2     0.4593   216.00   25535     24376     1.05       1.289       1.28      0.867
2-region dense core n=3.5   0.2625   216.00  136851    131631     1.04       1.298       1.27      0.871
3-region shell+core n=2/3   0.3353   208.00   63214    419176     0.15       8.952       0.18      0.317
2-region off-centre n=2     0.4625   216.00   25007     21649     1.16       1.169       1.41      0.929
2-region thin wall 0.4mm    0.5401   216.00   15708      9371     1.68       0.805       2.05      1.144
```

- **Worst under-estimate 0.95, and it is not one of the rows above**: it is the
  `--build` nested fixture (a 6 mm cube whose inner core is 3.5x finer -- the same
  object as the two-region dense-core row, built through the `--build` path):
  342,129 estimated against 358,307 built, `built/volaw 1.414`. The lowest ratio
  IN the table is the 2-region dense-core row at 1.04. Neither is covered by the
  margin, and both are recorded rather than hidden.
- **Worst over-estimate 1.89** (`one-part plate 8x4x2`: 8,104 estimated, 4,294
  built). The whole nine-family band is 0.15 .. 1.89, median 1.16.
- 1.35 is the margin; **the estimate is above the real mesh on eight of the nine
  families** and reads 0.15x on the ninth, which is the found issue below.
- `old/built` exceeds the new `est/built` by **exactly 14 / (6*sqrt(2) * 1.35) =
  1.222x on every row**. At the product's cap that is the whole of the formula
  change's effect on the over side: a uniform 22% tightening. It is NOT a 10x
  fix -- the 10.86x / 8.88x figures that motivated this step were the 60,000 bench
  cap's, and are withdrawn above.

At **half the mode counts** (`--family-scale-n 0.5`, a legitimate upload: the
user picks the mode count) the SAME nine families read est/built **0.72 .. 2.10**
(median 1.25), built/volaw 0.643 .. 1.868, achieved/h 0.798 .. 1.243:

- worst under-estimate **0.72** -- a bumpy sphere at h_min 0.6841 mm: 4,052
  estimated, 5,608 built, `built/volaw 1.868` (delivered edge 0.80x the request).
  The surface's own facet scale is the suspected floor; it is NOT measured here.
- worst over-estimate **2.10** -- the 6 mm cube at h_min 0.8484 mm: 4,052
  estimated, 1,931 built (`built/volaw 0.643`). The coarser the request, the more
  TetGen is entitled to coarsen below it, and the law cannot see that.

At **double the mode counts** (`--family-scale-n 2`), run by the LEAD on the Orin
against the same revision (2420635, the same bench, meshing only, cap 600,000),
the same nine families widen the band rather than shift it, and the recorded
under-read moves with the setting:

- the 3-region nested-thin-sheet family reads est/built **0.71** (`built/volaw
  1.909`, delivered 1.9x the law) -- at the shipped setting this is the 0.15
  family, so a ratio taken at one setting is 4.7x this one;
- the 2-region core 1.5mm family reads **0.94** (`built/volaw 1.438`).

The band across the three settings measured is **0.71 .. 2.10 plus the 0.15
point**. 1.35 does NOT cover either new row: the margin covers a family only
while `built/volaw` <= 1.35, and these deliver 1.909 and 1.438, so both read
under (est/built 0.71 and 0.94). What still stands is that each reads above 0.5,
i.e. inside the refusal factor 2 (only the 0.15 family falls outside it). The
half-mode-count run reproduces the cube's 2.10 and the bumpy sphere's 0.72 above.

So **1.35 is a measurement of one point in the (mode count, geometry) space, not a
bound over it**, and no single margin over the law can be a bound: `built/volaw`
is a property of the DELIVERED mesh, measured 0.643 to 8.952 on these fixtures,
and the request-side law is the only thing the estimator can compute before the
mesh exists.

## The spread is the mesher's, not the formula's

- **A facet floor was tried and rejected.** The mechanism above (fine facets
  flooring the delivered edge below the request) suggests `min(h, facet_edge)` as a
  term. It fails the middle of its own range: the plain sphere at the shipped scale
  has fine surface facets and a 0.5412 mm request and delivers a 0.5006 mm
  volume-equivalent edge -- 0.92x the request, `built/volaw` 1.264. A floor term
  would read it below the request, i.e. tens of percent high, where the mesh is 26%
  high. Making it fit would need a blend between the request and the facet scale --
  a fitted curve, which this change is forbidden to add and could not defend.
- **A tighter `maxvolume` does not narrow the spread -- it widens it.** Measured
  (`benchmarks/mesher_slack_probe.py`, no product code touched): at
  `TETGEN_VOLUME_SLACK` 4.0 the delivery spans `built/volaw` 0.715 .. 8.952; at
  2.0 it is **1.472 .. 3.818**; at 1.0 it is **3.295 .. 9.020**; at 0.5 **6.779 ..
  10.931** and two families fail outright (the mesh no longer fits the cap). The
  bound is a CEILING and TetGen refines below it for quality, so asking for a finer
  maximum makes the delivery finer, not the request more binding. (The probe
  patches BOTH bindings of the constant -- `mse_common` and `mse_meshing` -- and
  prints a control line proving the bound moved.)
- **The delivered edge is not derivable from the plan.** Nine families, near
  identical plans (one global edge, `maxvolume` from `h_min`), delivered
  volume-equivalent edges 0.1635 - 0.7361 mm; delivered/requested spans 0.32x to
  1.13x. Nothing in the plan distinguishes the rows.
- **The mesher has no per-region lever to steer it**: regions are registered by
  seed with `max_vol = 0.0` (`pytetgen.py:287`, `:336`), and the mesher's own
  comment records per-region volumes as measured INERT over nine orders of
  magnitude (a hand-built two-shell cube, identical under `varvolume=True`,
  `fixedvolume=True`, and neither).

## Found issue, reported to the lead and NOT fixed here

Measured on the nine families with `benchmarks/mesher_edge_cause.py` at the
product's cap (per-region tet volumes against the `maxvolume` the mesher actually
passed in; `over` and `worst` are the guard-facing direction, `fine` is the share
of tetrahedra at or below half the requested edge):

| family | tets over the bound | worst tet | fine | volume-equivalent delivered edge vs requested |
| --- | --- | --- | --- | --- |
| one-part 6mm cube | 0.3% | 1.15x | 15.1% | 0.7361 / 0.6734 = 1.09x |
| one-part plate 8x4x2 | 1.2% | 1.34x | 14.2% | 0.5020 / 0.4489 = 1.12x |
| one-part sphere r3 | 0.0% | 1.04x | 30.5% | 0.5006 / 0.5412 = 0.92x |
| one-part bumpy sphere r3 | 0.0% | 0.91x | 16.8% | 0.5327 / 0.5430 = 0.98x |
| 2-region core 1.5mm n=2 | 0.0% | 1.20x | 46.1% | 0.4221 / 0.4593 = 0.92x |
| 2-region dense core n=3.5 | 0.0% | 1.24x | 45.5% | 0.2406 / 0.2625 = 0.92x |
| 3-region shell+core n=2/3 | 0.0% | 1.14x | **94.3%** | 0.1635 / 0.3353 = **0.49x** |
| 2-region off-centre n=2 | 0.0% | 1.11x | 37.7% | 0.4391 / 0.4625 = 0.95x |
| 2-region thin wall 0.4mm | 0.0% | 0.98x | 8.2% | 0.5805 / 0.5401 = 1.07x |

Two defects, or one:

1. **The mesh can over-refine far below the `maxvolume` it is given, and nothing
   bounds it.** The three-region PLC is the extreme: 94.3% of its tetrahedra are
   at or below HALF the requested edge, the volume-equivalent delivered edge is
   0.49x the request, and it builds 419,176 elements where the law says 46,825
   (`built/volaw 8.952`). The per-region split shows it is not one region misbehaving
   -- all three are finer than asked (avg tet 0.049x / 0.018x / 0.007x of the
   bound). Over-refinement is legal (the bound is a maximum) but it is unbounded in
   the other direction, so it can multiply a job's real size by 3x or 9x with no
   signal at upload time. It is not only the multi-region path: the probe's
   no-region CONTROL for the thin wall is 91.2% fine against 8.2% for the same
   object meshed through the plan (the control is asked for a coarser edge, so the
   two are not comparable numbers of tetrahedra -- 5,927 vs 9,371 -- but both show
   the delivered mesh landing far from the request). **The one that matters for the
   product is the extreme:** the 3-region family asked for a 26% coarser edge
   (`h_min 0.4224` at half the mode counts) builds only 30,144 elements
   (`est/built 1.05`) -- a cliff, not a gradient.
2. **The too-coarse guard reads a different edge than `maxvolume` does.** The
   guard allows `MAX_EDGE_OVERSHOOT (1.6) * edge_m` where `edge_m` is the WALL's
   edge (`mse_meshing.py:567`, and `mse_common.effective_edge_m` documents that the
   multi-part branch "does not shrink it"), while `maxvolume` is built from
   `h_min` (`:507-509`). For the three-region family that licenses
   `1.6 * 3.0 = 4.8x` `h_min` before the guard can fire. **The mismatch is real in
   the code; the demonstration that it FIRES was not.** At the product's cap the
   delivered volume-equivalent edge is 0.32x-1.14x the requested one across the nine
   families, so every one of them is well inside the guard's 1.6 -- including the
   nested build fixture at `achieved/h 0.847`, the very object step 1 recorded at
   `1.864`. That 1.864 was the 60,000 bench cap's coarsen-to-fit retry, so the
   step-1 "the guard passed a 1.87x-coarse mesh" headline is withdrawn with the
   other cap-era numbers. What survives is the code fact: guard and `maxvolume`
   read different edges, and the guard's own bound (1.6) is within 1% of the
   mesher's volume slack in edge terms (`4^(1/3) = 1.587`), which is what it looks
   like it was meant to enforce. Whether the mismatch is reachable on any real
   upload is NOT established here -- it needs a fixture that over-refines by more
   than 1.6x, and this sweep's worst is 1.24x.

**Consequence for this change:** the estimator is faithful to the mesher's REQUEST
(one `maxvolume` from `h_min`) and CANNOT see the delivery, in either direction.
The spread is not fixable from inside the estimator, and the slack probe shows it
is not fixable by asking for a finer maximum either.

**What the residual mis-reads cost, measured.** Routing to the cluster happens
below an estimate of 15,000 and the solver refuses a cluster job above 150,000 real
elements (`mse_common.DEFAULT_CPU_MAX_TETS`), so a mis-route needs
`est/real < 15,000/150,000 = 0.1`. The worst ratio measured anywhere is **0.151**
(the 3-region at the shipped setting) -- within 1.5x of a mis-route but not
across it. On the refusal side, refusal fires above `888,889 * built/volaw` real
elements: 635,556 for the coarsest family at the shipped setting (above the
600,000 cap, so no wrongful refusal), and only the half-mode-count cube reaches
below it, to **571,000 -- 5% under the cap, the whole of this change's
wrongful-refusal exposure.** Main's fitted constant put the same threshold at
557,000 for the plate and the cube, so this change refuses strictly less than main
and never more.

The lead decides whether the mesher becomes its own row. It is NOT fixed here: the
change is scoped to the estimator (the lead's instruction of 2026-10-15 04:11Z,
"the mesher stays unchanged in this change").

## The refusal rule, derived from BOTH bounds

`est = margin * law` and `real = built/volaw * law`, so `est/real = 1.35 /
built/volaw`: the honest way to read the two failures is per family, not against
one constant.

- **Wrongly refusing an upload whose real mesh fits.** Refusal fires at
  `est > 2 x 600,000`, so it fires on real meshes above
  `2 x 600,000 x built/volaw / 1.35 = 888,889 * built/volaw`. At the shipped
  setting the coarsest family measured (`built/volaw 0.715`, the plate) puts that
  at **635,556 elements -- above the 600,000 cap, so no wrongful refusal there**,
  and likewise for all nine families. The only setting measured that reaches below
  the cap is the half-mode-count cube (`0.643`), at **571,000 -- 5% under the cap,
  and the whole of this change's wrongful-refusal exposure.** Main's fitted
  constant put the same threshold at 557,000 for the plate and the cube, so this
  change's failure mode here is strictly smaller than main's.
- **Wrongly accepting an upload whose real mesh does not fit.** Acceptance needs
  `est <= 1.2M`, so the largest real mesh that slips through at a family's ratio
  `r = est/real` is `1.2M / r`. At the SHIPPED setting the smallest ratio measured
  is 0.95, so nothing above 1.26M real elements is accepted (2.1x the cap); at half
  the mode counts the smallest is **0.72** (the bumpy sphere), so a real mesh up to
  **1.67M elements is accepted -- 2.8x the cap.** What happens to it is the
  mesher's over-cap retry: it coarsens `maxvolume` and rebuilds, so the failure is
  a silently reduced level of detail rather than memory exhaustion on the Orin.
  That is still the worse failure of the two, and it is the arm the lead refused to
  widen (2026-10-15 04:11Z).
- **No single factor avoids both**, because the spread is a property of the
  mesher's delivery and a factor is a scalar on the estimate. To never wrongly
  refuse it must be at least the worst over-estimate (1.89 at the shipped setting,
  2.10 at half the mode counts); to never wrongly accept it must be at most
  `1 / 2.10 = 0.48`, which refuses almost every upload. Both failures appear in the
  measured range -- the current factor wrongly refuses real meshes of 571,000-600,000
  and wrongly accepts real meshes of 600,000-1.67M -- and no scalar removes both.
- **Decision: the factor stays 2**, main's shipped value, with both failures and
  their numbers written beside it in `web/config.py`. On the floors measured at
  the product's cap, 2 sits just inside the first bound (2.1) and the second
  bound is not reachable at all -- the honest statement is that 2 wrongly accepts
  in principle and, on nine families measured, wrongly refuses on one setting by
  5%. The formula change alone moves the threshold 1.222x in the permissive
  direction, so this change refuses strictly less than main and never more;
  re-deriving the factor is the lead's call, and 2 becomes both safe and tight
  only once the mesher delivers the edge it asks for.

## Stop condition (the lead's, 2026-10-15 04:11Z)

The lead set one: if the spread cannot be brought near the old 2x without
modelling a mesher you would call broken, stop and report with the numbers, and
do not pick a factor.

**At the product's cap the spread does NOT need bringing near 2x -- it IS about
2x, and the premise that it was 8.88x was the 60,000 bench cap's artifact
(corrected at the top of this document).** Measured at `CFG.max_tets()`, the
nine-family band is `est/built` 0.95 .. 1.89 at the shipped setting and
0.72 .. 2.10 at half the mode counts. The `built/volaw` spread is 0.643 .. 8.952,
but one family (the 3-region nested thin sheets) produces the entire upper end;
without it the spread is 0.643 .. 1.414. So the honest report is:

- **the estimator is not the problem the step-2 direction feared.** At the
  product's cap the formula change is a principled 1.222x tightening plus the
  removal of a fitted constant nobody could read, not a rescue from 8.88x.
- **the one family that is 8.952x of the law is a mesher defect** (over-refinement
  far below the `maxvolume` it is given), recorded above with its fixture numbers
  and left unfixed by the lead's instruction.
- **the spread IS invariant to any constant** -- the change moves every row by the
  same 1.222x -- so no choice of margin or factor narrows it, and the slack probe
  shows a tighter `maxvolume` makes the delivery finer, not more predictable.
- **no factor was picked.** The factor stays at main's 2. The decision the lead
  reserved (re-deriving it, and whether the mesher becomes its own row) is
  escalated with these numbers.

## Impact

- **Live routing and refusal:** the estimate falls 1.222x, so jobs route to the
  cluster slightly less often and uploads are refused slightly less often. Refusal
  is strictly less likely than on main, never more; on the floors measured at the
  product's cap, no family is wrongly refused and routing needs a delivery 1.5x
  finer than the worst measured before it could mis-route.
- **Step 3 (the lead's, on the Orin):** run

  ```
  python benchmarks/estimate_vs_built.py --build --families \
      --family-scale-n <N> --n-modes <M> --max-tets 600000
  ```

  against pre-fix main and against this branch. `--families` is the nine families
  tabulated above; `--build`'s two fixtures (a plain 6 mm cavity and a nested one)
  are steered by `--size-mm`, `--core-ratio` and `--core-index` (defaults 6.0 /
  0.25 / 3.5). `--family-scale-n <N>` multiplies every family's mode count and
  `--n-modes <M>` sets the `--build` fixtures' count; the element count goes like
  the mode count, so pick `N` to put each family in the tens-to-hundreds of
  thousands. `--max-tets 600000` states the product's cap explicitly so the
  transcript carries it -- the benchmark defaults to `config.max_tets()` either
  way. Both sides are then real meshes at the product's cap. (No `--results`:
  that flag tabulates previously-recorded run dirs and is not part of this run.)
  The grading bench's `_build` key fix is NOT needed for this: step 3 runs this
  benchmark, not the grading branch, and the lead dropped that edit.
- **Not in this change:** the mesher's over-refinement and its guard's edge
  mismatch (found issue above); the refusal factor (the lead re-derives it); the
  facet-floor question (rejected here with its number).
