# fix-upload-tet-estimate -- tasks

Owner picked option 1 (fix the estimator) 2026-10-15. Row
`qitem-20261015000650-f986f488`. Branch `fix-mesh-estimate`, off MAIN `cdbeb49`
(NOT the grading branch). Tier 1 only: no FEM, no cluster, no Orin.

## 1. The cause (DONE -- step 1)

- [x] Read the live path end to end: `web/app.py:1113` -> `web.meshing.estimate_tets`
      -> `web/config.est_tets_c()`; the mesher's own edge resolution
      (`mse_meshing.py:463` -> `:506` -> `:567`) and `mse_common.region_edges`.
- [x] Confirm the live deployment uses the default: `mse-web` in ns
      `mode-search-engine` sets no `MSE_EST_TETS_C`, so `C = EST_TETS_C = 14.0`.
- [x] Build the instrument `benchmarks/estimate_vs_built.py` (dev-only) and take
      the measurement (tier 1, exit 0). Result on `--build`'s fixtures:

```
C = 14   tets per h^3 at edge h = 8.485   maxvolume = 4.0 * h^3/(6*sqrt(2))
one-part 6mm             h  0.6734 mm  est     9904  built     5989  est/built 1.65  built/volaw 0.998
nested 6mm core 0.25 n3.5 h 0.1934 mm  est   418136  built    38489  est/built 10.86 built/volaw 0.152
```

      **RETIRED (2026-10-15, step 2): these two rows were taken with the bench's
      cap set to a 60,000 literal, and the mesher has an over-cap retry that
      coarsens `maxvolume` and rebuilds, so both measured a mesh the product never
      builds.** At the product's cap (`config.max_tets()` = 600,000) the same
      fixtures read `1.35 / 0.998` and `0.95 / 1.414`. The benchmark now takes
      its cap from the product and prints it in its header.

- [x] Answer the row's three hypotheses by measurement:
      (a) the volume-per-tet constant -- YES, `14/(6*sqrt(2)) = 1.650`, the floor;
      (b) counting the finest edge everywhere -- YES, and the largest single
      contributor when a region is denser than the wall: `est` puts the narrowest
      region's edge on the WHOLE cavity, a `(n_max/n_wall)^3 = 42.9x` multiplier
      at n=3.5, which the mesh does not follow (that multiplier is what step 1
      measured as 10.86x at the 60,000 bench cap; at the product's cap the same
      fixture's `est/built` is 0.95x, because TetGen refines the dense core below
      the `maxvolume` it is asked for -- the over-read is real but small, and the
      direction on this fixture is the other one);
      (c) summing nested region volumes twice -- NO: the plan's summed
      `filled_volume_m3` for the nested 6 mm fixture is 216.00 mm3, exactly that
      box's volume, and matches the one-part path's `enclosure_volume_m3`.
- [x] Write the cause up in `proposal.md`.

## 1b. Correction to the row (lead's answer to Q1, 2026-10-15 00:30Z)

The row's `75,262` (c2-thin) and `161,389` (c4-thin) are the GRADING BENCH's
graded planned count from its analytic table, NOT main's `estimate_tets` on those
plans. **They are retired**; the dataset for step 3 is this benchmark's table,
both sides measured on it. The `mse-orin` / `mse-bench-results` rows step 1
printed are the grading bench's recorded PLANS, so they stay only as the shape
list; the numbers step 3 compares are the ones `--build` prints.

## 2. The agreed fix, and what the sweep measured (DONE -- step 2)

- [x] **The formula**, no fitted constant: `est = MARGIN * 6*sqrt(2) * sum_r
      filled_V_r / h_r^3`, with the geometry law written exactly
      (`mse_common.TETS_PER_UNIT_VOLUME`) and the margin a measurement. The
      mesher applies ONE edge to the whole cavity (`mse_meshing.py:507-509`
      builds a single `maxvolume` from the plan's `h_min`), so `h_r = h_min`
      describes the mesh the mesher really asks for; the per-region sum is
      written against the plan so it follows the plan if that changes.
- [x] **The margin is measured over the shape families**, not fitted:
      `benchmarks/estimate_vs_built.py` now builds nine families (single part,
      2 and 3 regions, thin and thick shells, off-centre, sphere, plate, bumpy
      sphere) on REAL meshes AT THE PRODUCT'S CAP (600,000; the bench used to use
      a 60,000 literal, which measured the mesher's coarsen-to-fit retry instead
      of the product). At its shipped scale the worst under-estimate it covers is
      1.298 -> `EST_TETS_MARGIN = 1.35`, and two families it does NOT cover are
      recorded by number: 1.414 (a 3.5x finer core, estimate 0.95x) and 1.868
      (the same families at `--family-scale-n 0.5`, the bumpy sphere, estimate
      0.72x). **So 1.35 is a measurement of one setting, not a bound**, and it is
      documented as such in `web/config.py`.
- [x] **The mesher's DELIVERY is what the spread is.** Measured at the product's
      cap the nine families read `built/volaw` 0.715-1.414 at the shipped setting
      (0.643 at half the mode counts) except one family at 8.952 (the 3-region
      nested thin sheets; the found issue). A
      `benchmarks/mesher_slack_probe.py` re-run at `TETGEN_VOLUME_SLACK`
      4.0/2.0/1.0/0.5 shows the delivery gets FINER as the bound tightens
      (0.715-8.952, then 1.472-3.818, then 3.295-9.020, then 6.779-10.931, with
      two families no longer fitting the cap at 0.5), so there is no slider that
      narrows it: the bound is a ceiling and TetGen refines below it.
- [x] **The refusal factor is NOT re-derived and NOT widened.** At the product's
      cap the measured band is 0.95x-1.89x at the shipped scale (0.72x-2.10x at
      half the modes); to never wrongly refuse the factor must be at least 2.10,
      and to never wrongly accept at most 0.48. Over that spread no single factor
      removes both failures, both of which appear in the measured range.
      `UPLOAD_REFUSE_FACTOR` stays at the 2 main ships, with both numbers written
      beside the constant; the formula change alone moves the threshold 1.222x in
      the permissive direction. Decision with the lead.
- [x] **Tests on real meshes**: `tests/test_estimate_families.py` builds real
      TetGen meshes and asserts the estimate sits in the measured band on each
      compliant family, pins the two under-read families (`0.15x`, `0.72x`) so a
      mesher fix reddens them, recomputes the replaced `C = 14` formula on the
      SAME mesh and asserts it falls outside the band (on the plate: new 1.89,
      old 2.31) plus the cap arithmetic that main refuses an upload that fits,
      and pins the refusal factor as the shipped value. The estimate tests in
      `tests/test_routing.py` and `tests/test_web_routes.py` use the margin API;
      no `est_tets_c` / `MSE_EST_TETS_C` remains anywhere.
- [x] **The mesher guard is recorded, not fixed** (the lead's Q3), and its
      demonstration is CORRECTED: the guard compares the achieved edge against
      `1.6 * edge_m` where `edge_m` is the WALL's edge, while `maxvolume` is built
      from `h_min` (`mse_meshing.py:506-509`, `:567`), so guard and request read
      different edges -- real in the code. But step 1's "a mesh 1.86x coarser than
      the mesher's own request passes" was measured at the 60,000 bench cap, and at
      the product's cap the same nested fixture delivers `achieved/h 0.847`,
      inside the guard. The mismatch is latent, not shown to fire: this sweep's
      worst delivered/requested edge is 1.24x, under the guard's 1.6. `proposal.md`
      carries both facts with the fixture numbers; the lead decides whether it
      becomes its own row.
- [x] **Bench fix**: the cap is taken from the product (`config.max_tets()`) and
      printed in the header, with `--max-tets` as a deliberate override; the
      over-refinement is visible per family (`--families` prints the fine-tet
      share). The grading bench's `_build` key fix at line 1 of section 3 is NOT
      needed and NOT wanted (the lead, 2026-10-15 04:11Z).
- [x] Spec: `skip_specs` gone; `specs/mesh-upload/spec.md` delta modifies the
      estimate requirement (the "twice the real cap, because the estimate can be
      up to twice high" parenthetical is replaced with the measured trade).
- [x] Tier-1 run of the affected suites green; local commit; sha to the lead.
- [x] Task 4 is "report to the lead": the lead tells the owner. There is no
      runtime-backend pick in this change.

## 3. Re-measure on the Orin (the lead's; the command is ready)

- [x] ~~Fix the grading bench's `_build` `edge_m` key.~~ **Dropped** (lead,
      2026-10-15 04:11Z): step 3 runs this benchmark, not the grading branch, and
      no grading-branch edit is wanted. This branch's copy is fixed anyway.
- [ ] Step 3 command (fixtures large enough for each family; --build so both
      sides are real meshes), to be handed to the lead:

```
python benchmarks/estimate_vs_built.py --build --families \
    --family-scale-n <N> --n-modes <M> --max-tets 600000
```

      Fixtures: the nine families in `FAMILIES` (one-part cube / plate / sphere
      / bumpy sphere; 2-region core-1.5mm / dense-core / off-centre / thin-wall;
      3-region shell+core) plus `--build`'s two (the plain 6 mm cavity and the
      nested 6 mm core). `--family-scale-n <N>` multiplies every family's mode
      count and `--n-modes <M>` sets the `--build` fixtures' count; the element
      count goes like the mode count cubed, so `--family-scale-n 2` (8x the
      elements) puts the coarser families in the hundreds of thousands and the
      3-region family near the cap. `--max-tets 600000` states the product's cap
      explicitly so the transcript carries it. Pre-fix main on this same command
      is the other side of the comparison.

## 4. Report

- [x] Report to the lead with the sha, the sweep table and the worst under- and
      over-estimates (step 2); the lead tells the owner. **The report leads with a
      correction:** the "worst over-estimate 8.88x" / "worst under-estimate
      1.047x" the step-2 direction of 2026-10-15 04:11Z reasons from were measured
      at the bench's 60,000 cap and do not describe the product. At the product's
      cap the band is 0.95x-1.89x at the shipped setting (0.72x-2.10x at half the
      mode counts) -- a spread of about 2x, not 8.5x. The lead's arithmetic on a
      factor of 9 still holds (a tight estimate is a tight estimate, so 9 still
      admits up to 5.4M elements), but the DIAGNOSIS it rests on -- "any single
      factor over a spread that wide" -- does not: over the real spread, main's
      2x is within 5% of never-wrongly-refusing, and the only family that forces
      headroom is the mesher's 8.952x over-refinement, not the estimator. The
      refusal factor is still NOT widened, and no factor is picked here.

## 5. Reviewer follow-ups (review row cdeda91b, ACCEPT 2420635)

- [x] Fold the three spec-delta doc items into this change:
      **a)** the low-reading consequence in the refusal-factor paragraph -- a
      family that reads low (0.15x, 0.71x) is accepted with its estimate inside
      the threshold while its real mesh can run to 1/0.15 ~ 6.7x it, so the factor
      bounds the ESTIMATE and not the delivery; **b)** the shipped-band nit -- the
      2.10x reading belongs to the cube at half the mode count, not to the shipped
      sweep, which tops out at 1.89x; **c)** the widened retry band -- main also
      retries every mesh over 600,000, so raising the refusal threshold 727k ->
      889k law-elements widens the coarsening retry by 14/11.455 = 1.22x.
- [x] Record the settings the band was measured at: the lead's Orin run at double
      the mode counts (`--family-scale-n 2`) adds the 3-region family at est/built
      0.71 (`built/volaw 1.909`) and the 2-region core 1.5mm family at 0.94
      (1.438); the half-mode-count run reproduces the cube at 2.10 and the bumpy
      sphere at 0.72. The band across the three settings is 0.71 .. 2.10 plus the
      0.15 point.
- [x] The `--max-tets 600000` override label compares values rather than testing
      for `None` (`c5d7e44`), the reviewer's item 5.
