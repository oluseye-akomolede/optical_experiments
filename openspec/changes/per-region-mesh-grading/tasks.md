# Tasks

## 1. The rule and the estimate (`mse_common.py`)

- [x] 1.1 `region_edges` returns the per-region graded edge
      `h_r = min(h, h * n_wall / n_r)`; `min` over the values is today's `h_min`.
- [x] 1.2 `region_boundary_area(parts, part_index)`: the region's own surface
      area plus every child's, in the part's own units (pairs with
      `filled_volume_m3`, which already subtracts the children's volumes).
- [x] 1.3 The thickness clamp in `region_edges`: `h_r <= CLAMP_THICKNESS_FRACTION
      * 2 * V_filled / A_boundary`, never applied to the finest region (never
      below `h_min`). Report which regions it lowered, so the caveat can name
      them.
- [x] 1.4 `estimate_tets_by_region` sums `c * V_r / h_r^3` per region. When the
      edges are equal this is byte-identical to today's `c * V_cavity / h_min^3`
      -- pinned by a test, not by inspection.
- [x] 1.5 `region_plan` / `region_plan_by_region` docstrings: the uniform rule is
      replaced by the graded rule; keep the record of why `add_region(max_vol)`
      is still inert and why the sizing field is the mechanism that works.

## 2. The mesher (`mse_meshing.py`)

- [x] 2.1 `_region_plan` carries the per-region targets in METRES (`edge_m`,
      `h_min_m`, `coarse_edge_m`, `unclamped_m`, `thickness_m`, `clamped`)
      beside the seeds, and `h_min_units = h_min_m / unit_scale` converts to the
      scaled surface's units once. The METRE space is not cosmetic: the thickness
      clamp divides by an area, so a rule that mixed the caller's upload units
      with metres would scale the clamp by the wrong power of the unit.
- [x] 2.2 `_sizing_background(plan, vertices, faces, canonical_offset,
      canonical_scale, units)` returns the `(points, tets, sizes)` triple: a
      coarse tetrahedralization of the canonical cavity with each node sized to
      the finest region adjacent to it (fallback: the coarse edge).
      `_sizing_background_grid(bg_points, bg_tets, bg_sizes)` packs that into a
      `pyvista.UnstructuredGrid` for TetGen's PUBLIC `bgmesh=` route
      (`point_data["target_size"]`). No private binding call on the path taken.
      Cost, measured on the 6 mm fixture: the background must be as fine as the
      fine edge, and it is ~19x the final mesh's tet count (8.6k-580k tets,
      up to 17 s). Recorded, not hidden; it is the price of the field.
- [x] 2.3 `tetrahedralize` branches: plan edges all equal -> the existing
      call, untouched (byte-identical); otherwise the sizing field with NEITHER
      `fixedvolume` NOR a global `maxvolume`. A guard raises if either is passed
      alongside the field.
- [x] 2.4 `achieved_edge_length(points, tets, mask=None)`: restrict the median to
      a region's tets. The graded check uses the finest region's mask against
      `MAX_EDGE_OVERSHOOT * h_min`.
- [x] 2.5 The decimation target for a graded mesh is the COARSE edge (`max_r
      h_r`); unchanged (finest edge) for a uniform mesh.
- [x] 2.6 The mesh report: `graded`, per-region `{target_edge_m,
      unclamped_edge_m, thickness_m, achieved_edge_m, filled_volume_m3, name}`,
      and `caveats` (the clamp when it bit, in plain English). Top level:
      `fine_region`, `target_edge_m`, `target_edge_units`, `coarse_edge_m`,
      `achieved_edge_m`, `fine_achieved_edge_m`. `LAST_ACHIEVED_EDGE_M` keeps its
      meaning for the uniform path.

## 3. The artifacts and the page

- [x] 3.1 The solver writes the `mesh` block into `modes.json` (graded flag,
      regions, caveats); the uniform/one-part block keeps its current fields.
- [x] 3.2 `templates/job.html` shows a graded job's per-region resolutions and
      its caveats, in plain English, and shows nothing new for a uniform job.
- [x] 3.3 `web/app.py` passes the block through without rewriting it.

## 4. Tests (tier 1, `nice -n 19`, `(ulimit -v 4194304; ...)`)

- [x] 4.1 The graded path on a nested-cube fixture: fewer tets than uniform AND
      the finest region's achieved edge within the allowance. Both directions
      pinned (the reduction control fails if grading is off; the resolution
      control fails if the field is not honoured).
- [x] 4.2 The clamp: a fixture whose coarse target exceeds its thickness is
      lowered, the caveat names the region, and a clamp that does not bite
      writes no caveat. The clamp never lowers the finest region.
- [x] 4.3 The guard: a graded call passing `fixedvolume` or a global
      `maxvolume` raises.
- [x] 4.4 The estimate: for a graded plan, `c * sum V_r/h_r^3`; for equal edges,
      the exact number the uniform formula gives (a regression pin on today's
      value).
- [x] 4.5 The control the lead asked for: a SINGLE-material job's mesh is
      byte-identical to today's, and fails if the graded path is taken.
- [x] 4.6 `region_boundary_area` and the thickness proxy on a fixture whose
      thickness is known analytically.

## 5. Record

- [x] 5.1 The row's transitions get the shas, the measurements, and the two
      refuted causes (background-mesh resolution; the file-route background mesh
      loading without error and grading nothing).
- [x] 5.2 The Jetson command for the concentric-sphere benchmark goes to the
      lead on the row (the lead runs it; this seat does not).
- [x] 5.3 The accuracy criterion is rebarred to the UNIFORM mesh (owner's
      decision, 2026-10-13), on the Orin measurement at `ebfd570`: n2 graded
      worst 0.341% vs uniform 0.559% (1.64x), n3 graded 0.749% with its uniform
      reference UNMEASURED (cuDSS `ALLOC_FAILED`, 199,062 tets). The old "within
      2x the empty sphere" bar is replaced by "no worse than the uniform mesh",
      and an unmeasured bar fails the run (INCOMPLETE, exit 4) instead of
      passing it. Tests cover the pass, fail, unmeasured-solve and flag-off arms.
      The later Orin run at `11357f4` solved n3's uniform reference: n3 graded
      0.749% vs uniform 0.662% (1.13x) -> FAIL (the run exited 3), n2 unchanged
      (0.341% vs 0.559%, 0.61x, PASS). Median edges at n3: graded core 0.664 /
      shell 0.9014 / wall 1.453 mm, uniform core 0.7085 / shell 0.7253 / wall
      0.5287 mm. n3 does not meet the bar on this measurement; the ALLOC_FAILED
      at `ebfd570` stands as that run's result.
- [x] 5.4 STEP 1 of row `600283f7` (owner's 10-13 13:00Z direction: "the meshing
      needs to work generally for any mesh supplied"): the cause of the n3 shell
      0.901 / wall 1.453 medians, from the plan dict's own numbers plus a tier-1
      analogue mesh. `benchmarks/probe_n3_bands.py` reproduces every number
      below (tier 1, no solver, no Orin); the quoted Orin numbers are the lead's
      `11357f4` measurement, a record and not an input to anything.

      (a) NEITHER REGION IS CLAMPED, and the `min(h, h * n_wall / n_r)` term
      produces NEITHER reported number. `region_plan` on the real fixture
      (b = 10 mm, a/b = 1/3, n_core = 3): wall asked 1.7884 = unclamped 1.7884,
      thickness 5.7693, `clamped` False; core asked 0.5961 = unclamped 0.5961,
      thickness 2.1882, `clamped` False. The clamp is refuted as the cause at
      this ratio; the `min` term only sets the core to h/3 = 0.5961 and leaves
      the wall at h = 1.7884. 0.9014 is neither of those, and 1.4530 is neither.

      (b) BOTH NUMBERS ARE WHATEVER THE SIZING BACKGROUND'S RAMP READS AT THE
      CENTRE OF A BAND, and every term the row named is refuted or local:

        * `finest-node-wins` is REAL BUT LOCAL, and it is what sets the ramp's
          lower end, not either reported number. Measured layer by layer in the
          analogue (section C of the probe; depth = (r - a_eff)/h_wall):
          within h_wall/5 of the interface 26-52% of tets are within 15% of
          h_min; by depth 0.4 h_wall, 0%; past 2 h_wall the median is 1.036
          x h_wall. So the field STEPS to the fine edge only at the interface
          and then RAMPS to the coarse edge over ~2 x h_wall.
        * `bg cell = min(coarse/2, fine)` is the background GRID's cell, and at
          n3 it is min(1.7884/2, 0.5961) = 0.5961 = h_min (the fine edge wins,
          exactly as the row's `min` says). It is visible in the mesh as the
          ramp's starting value -- h_min at the interface -- NOT as 0.901.
        * shell 0.9014 mm and wall 1.4530 mm are samples of that ramp. The
          shell band is an ABSOLUTE `SHELL_BAND_MM = 1.0`, so its median is a
          function of the band's WIDTH. Swept on the analogue (section D), in
          units of h_wall: width 0.278 -> median 0.439; 0.504 -> 0.477; 0.667 ->
          0.504; 1.111 -> 0.556. The Orin's band was 0.559 h_wall and its
          number 0.504; at the MATCHED relative width the analogue reads 0.477,
          a 5% discrepancy. So the analogue reproduces the STRUCTURE (a ramp
          whose samples move with the band width) and the order of magnitude,
          and NOT the number exactly. The earlier "same ratio to three digits"
          compared two different relative widths (0.667 here vs 0.559 there)
          and is WITHDRAWN; the width sweep is the evidence, and it is stronger
          than the coincidence was.
        * wall 1.4530 mm = 0.812 x h_wall. `wall` in that helper is `~core` --
          the WHOLE outer region -- so it mixes the ramp with the interior.
          Analogue, same mask: 0.7685 mm = 0.854 x h_wall (Orin 0.812). The
          wall's own BULK (r > a_eff + 1 mm) is 0.932 mm = 1.036 x h_wall with
          61% of its tets within 15% of h_wall. So the background DOES deliver
          the wall's asked edge in the wall's interior; the 0.81-0.85 figure is
          the band's mixture, not the rule's output.

      (c) THE OWNER'S QUESTION, answered in two halves -- the rule and the
      metric are not the same thing, and only one of them is job-shaped. The
      RULE is not fitted to one job: `region_plan` reads only the object's own
      geometry and indices, no per-job constant is in play at n3, and (a) shows
      the clamp never fires. The BENCH'S METRIC is job-shaped: `SHELL_BAND_MM =
      1.0` absolute mm is 10% of this b = 10 mm upload and 50% of a 2 mm one,
      and (b)'s sweep shows the number it reports moves by 0.12 h_wall as the
      width moves from 0.28 to 1.11 h_wall. A per-region report (the mesh
      report already emits each region's achieved edge), or a band width
      relative to a_eff or h_wall, does not have that failure mode. No bar
      computed from these bands should be carried across uploads until that
      changes; measuring it across shapes is step 2's matrix, not more arguing.

      (d) WHAT STEP 1 DOES NOT ESTABLISH, plainly: the accuracy bar itself. The
      bands are DIAGNOSTICS -- the criterion scores the spectrum -- so (b)
      explains the two medians the lead read, not the 0.749% / 0.662% FAIL.
      What the bands do show is the trade the grading makes: the uniform
      reference puts h_min = 0.5961 mm everywhere, so its outer region is 2.75x
      finer than the graded mesh's wall band (0.5287 vs 1.4530) for ~7.6x the
      elements (`estimate_tets`: 274,415 uniform vs 36,237 graded; the Orin
      built 199,062 for the uniform -- the estimator runs ~1.4x high, so its
      counts are upper bounds and the matrix's `--list` should be read as one).
      At n2 (h_wall/h_min = 2) that trade passes (0.341% vs 0.559%); at n3
      (ratio 3) it fails. The trade is RATIO-dependent, and a tier-1 probe
      cannot reach the achieved edge of a full-size graded mesh on this VM (no
      heavy compute), so sweeping the ratio is step 2's job by construction.

      Note for step 2's operation: `benchmarks/probe_n3_bands.py` is tier 1
      (24,887 tets, seconds) and must be run with OMP/MKL/OPENBLAS threads
      pinned to 1-2 -- unpinned it took 23 BLAS threads.

- [x] 5.5 STEP 2 of row `600283f7`: the Orin validation matrix, as a bench mode
      (`benchmarks/run_grading_matrix.py`: `--list` / `--case <label>` /
      `--json`), with its tier-1 suite in `tests/test_grading_matrix.py` (14
      tests, exit 0). Axes: contrast {1.5, 2, 3, 4}, shell thickness (clamp
      biting / not), regions {2, 3}, shape (non-spherical box, non-concentric
      offset core), reference (analytic / uniform-refined). The criterion per
      case is the owner's: graded error no worse than the uniform mesh's, both
      solved through the product's own pipeline and scored against the SAME
      reference at the same mode count. UNMEASURED -- a failed reference solve,
      a failed graded solve, a fixture the producer rejects -- is its own
      outcome and never a pass. The Orin runs themselves are the lead's; nothing
      in this entry was run on the Orin.

      SIZING, measured with the product's estimator (`web.meshing.estimate_tets`
      at `web.config.CFG.est_tets_c()`, the function and value that route an
      upload): the uniform reference costs `contrast**3 * n_modes` and nothing
      else moves it -- the outer size cancels (the target edge scales with it)
      and the shell thickness does not enter it, because h_min comes from the
      densest region, not the wall. Contrast 4 at the sibling's ten modes
      predicts 650,465 elements, 3.3x the 200,000 tier-2 mark, so the two
      contrast-4 cases run at 3 modes, the first complete degenerate triplet.
      The non-analytic reference is a uniform solve per level at `h_min/K`,
      costing `K**3`; K = 2.0 is 8x and unaffordable for any case whose outside
      is a 10 mm sphere (1,097,656 predicted for the offset case, 2,601,856 for
      three-region's first 1/2/4 ramp), so `REFINE_FACTORS = (1.25, 1.5)` is ONE
      constant for every case. A per-case factor would be exactly the
      fixture-fitted tuning this row exists to rule out; its cost, stated where
      it is used, is that the reference is only 1.5x finer than the mesh it
      judges and the convergence check between the two levels has to carry the
      trust argument. three-region's ramp is 1/1.5/2 for the same reason, and
      box-sphere carries contrast 3 while offset-sphere carries 2, since that
      case's axis is the SHAPE.

      THE LIST, predicted element counts (the estimator reads ~1.4x high where
      both numbers are known -- 274,415 predicted, 199,062 built -- so these are
      upper bounds). `--list` prints it, plus the production ceilings an upload
      is routed by. Measured output (2026-10-14, tier 1, `.venv-mse`, exit 0):

      ```text
               case    ref  reg modes   h_min    graded   uniform  axes
         c1.5-thick analytic    2    10  1.1923    14,958    34,301  concentric, thick, n=1.5
           c2-thick analytic    2    10  0.8942    20,676    81,308  concentric, thick, n=2.0
           c3-thick analytic    2    10  0.5961    35,314   274,415  concentric, thick, n=3.0
                      ! over the tier-2 mark 200,000: ['uniform'] -- expect UNMEASURED, not a pass
           c4-thick analytic    2     3  0.6679    16,215   195,139  concentric, thick, n=4.0
            c2-thin analytic    2    10  0.8942    75,262    81,308  concentric, thin, n=2.0
            c3-thin analytic    2    10  0.5961   220,160   274,415  concentric, thin, n=3.0
                      ! over the tier-2 mark 200,000: ['graded', 'uniform'] -- expect UNMEASURED, not a pass
            c4-thin analytic    2     3  0.6679   161,389   195,139  concentric, thin, n=4.0
       three-region uniform-refined    3     5  1.1266    11,974    40,654  concentric, thick, n=2.0  refs [79402, 137207]
         box-sphere uniform-refined    2     5  0.7511    18,466    11,564  box, thick, n=3.0  refs [22586, 39028]
      offset-sphere uniform-refined    2     5  1.1266     9,659    40,654  offset, thick, n=2.0  refs [79402, 137207]

      axes covered: contrast=[1.5, 2.0, 3.0, 4.0]; shell=['thick', 'thin'];
      regions=[2, 3]; shape=['box', 'concentric', 'offset'];
      reference=['analytic', 'uniform-refined']
      tier-2 mark is 200,000 elements. production refuses an upload above
      1,200,000 and sends one above 15,000 to the GPU.
      ```
       ^^^ 5.6 CORRECTS THE `h_min` AND `uniform` COLUMNS ABOVE. They were
       measured against the probe baseline; the values, and three prose claims
       in this entry, are superseded -- see 5.6 (2026-10-14).

      WHICH CASES THE ORIN CANNOT SOLVE UNIFORM, plainly: c3-thick's uniform
      reference predicts 274,415 (the row's own case: built 199,062, OOMed once,
      solved once) and c3-thin carries the SAME uniform mesh -- h_min is the
      core's, so the wall's thickness does not change it -- plus a graded side of
      220,160, because the clamp pins a thin wall's edge near 1.0 mm. Both are
      flagged by `--list` and locked by the test's `EXPECTED_OVER_MARK`.
      c4-thick and c4-thin's uniform references predict 195,139, 2.4% under the
      mark: if the estimator's ~1.4x-high reading holds they build near 140k and
      solve, and if it does not they are the next two over -- reporting
      UNMEASURED, never a pass. Everything else predicts at or under 195,139 and
      every reference level at or under 137,207. STEP 3 -- the product rule --
      waits for the lead's go: no product rule is changed by this entry.

- [x] 5.6 STEP 2's diagnosis: the six items the lead returned with 5.5, one
      commit, tier 1 only, no product rule changed (row `600283f7`, 2026-10-14).

      THE ORIN RUN, as the lead reported it: three-region FAIL 0.231% graded vs
      0.106% uniform (2.17x), 7,319 vs 14,074 tets; offset-sphere FAIL 0.091% vs
      0.028% (3.27x), 7,238 vs 19,997; c3-thick FAIL 0.749% vs 0.662% (1.13x),
      16,252 vs 199,062; c4-thin PASS 0.063% vs 0.094% (0.67x), 61,244 vs
      105,369, wall clamped 3.234 -> 0.994 mm; c4-thick UNMEASURED (graded
      solved at 1.465%; the uniform side died on the GPU at 121,518 tets,
      ALLOC_FAILED / NvMap error 12); box-sphere UNMEASURED (graded SOLVED, 5
      modes, 7,350 tets, yet `graded_worst` None; uniform 1.442% at 4,868 tets;
      reference convergence 1.007%).

      ITEM 1 -- box-sphere's missing side. Two different faults, one of them the
      baseline. (a) The report said only that one of the two worst errors was
      missing. `score_multiplets` returns `worst_rel_error=None` exactly when
      nothing matched within its 2% tolerance, so a successful solve can produce
      it -- and the number now says which: `missing_side_reason` names the side,
      how many of its clusters matched (`n_matched` of `n_clusters`), which
      reference multiplets went unmatched, and the nearest cluster's distance
      against that 2% tolerance. Three tier-1 tests pin it, including the arm
      that must stay silent when a side DID produce a number. (b) "grading
      builds MORE tets than uniform" was an artifact of item 3's baseline: on
      the box's own surface the graded build is 7,350 tets (region 0 asks
      1.0105 mm and builds 0.7769 in 4,454 tets; the core asks 0.3368 and
      builds 0.3704 in 2,896 -- the count is dominated by the region that is
      NOT refined), against 105,321 built for the region-aware uniform at the
      same floor: 14x FEWER, not more. It only looked bigger against the
      probe-floor uniform's 4,868. Also checked and excluded as a cause: the
      region tags and the geometry's own eps agree on 100% of tets in every
      region-aware build except the box's graded (96.05%, 290 tets) and its
      uniform (99.66%, 358) -- a box/sphere interface layer, not a materials
      artifact. And the local tier-1 build reproduces the Orin's graded count
      for box-sphere exactly (7,350; also three-region 7,319, offset-sphere
      7,238), so the plan and the achieved mesh ARE the objects the Orin solved.

      ITEM 2 -- the uniform side's per-region edges are now emitted next to the
      graded ones: `_uniform_regions` pins `mse_common.region_edges` to the
      chosen floor, runs the PRODUCT's own `tetrahedralize` call, records
      `LAST_MESH_REPORT` and the tet count, and restores the real rule in a
      `finally` (a test proves both the pin and the lift). It is what showed
      that "asked" and "built" differ in both directions: in graded builds the
      unrefined region comes out FINER than asked (0.769 box-sphere, 0.727
      three-region, 0.735 offset-sphere -- the transition exponent, consistent
      with the 0.71/0.64/0.80 measured in step 1), while in a uniform build a
      region can come out 2.16x COARSER (ramp-outward's middle region). So
      "grading over-refines its coarse region" holds for graded plans, and is
      not a property of a region.

      ITEM 3 -- is the bench's uniform side the mesh the product builds today?
      NO, and every pre-2026-10-14 verdict was scored against another mesh. The
      bench sized the uniform side from a FIXED icosphere probe's floor while
      the graded plan came from the PRODUCT's own base edge for the case's
      surface, so the two sides of one comparison sat on different floors. The
      graded side is what the product builds and is unchanged; the uniform side
      moves. Measured floor pairs (product's own base edge / old probe floor,
      mm): c1.5-thick 1.2169/1.1923, c2-thick 0.9127/0.8942, c3-thick
      0.6084/0.5961, c4-thick 0.6809/0.6679, c2-thin 1.0825/0.8942, c3-thin
      0.7217/0.5961, c4-thin 0.8085/0.6679, three-region 1.2157/1.1266,
      box-sphere 0.3368/0.7511, offset-sphere 1.1416/1.1266, shell-dense
      2.2972/2.2533, ramp-outward 1.6209/1.5022. The product's base edge can
      only equal or exceed the probe's -- `effective_edge_m` caps a case at the
      probe's edge and never shrinks below it (the docstring is right; the
      comment claiming "the SMALLEST h/n_region" is stale) -- so the corrected
      baseline is COARSER for the thick and thin concentric families and for
      three-region/ramp, and MUCH FINER for box-sphere, whose surface edge is
      3x its core's. `run_case(..., baseline=)` now defaults to the product's
      floor, keeps the old one as `--baseline probe`, and reports BOTH floors on
      every run, so no run can be read against the wrong one again. The
      correction's direction per FAIL is stated with item 5.

      ITEM 4 -- the two missing case types, both sized for the Orin. `shell-dense`
      (dense shell, lighter core, 5 modes): the only case where the rule's outer
      `min(h, .)` decides the plan -- the core's raw ask is 2h and is capped to
      h -- so the plan it produces IS uniform, and this case cannot show the
      rule doing anything. Measured: both regions ask 2.2972 mm and the two
      meshes come out identical at 2,753 tets each. `ramp-outward` (n_wall =
      2.0, where every other case has 1.0, 5 modes): the only case that
      exercises the wall term in the denominator. Measured (tier 1, meshing
      only, re-run on the corrected baseline): the wall and the middle region
      both ask 2.4313 mm -- the middle one CAPPED, its material ratio having
      asked for more -- the core asks 1.6209 (2h/3), and the graded mesh is
      6,746 tets against 7,350 for the region-aware uniform at the same floor:
      an 8% saving, which is what a mild ramp buys. The ledger is NOT one-sided
      and that matters for item 6: the wall accounts for the saving (5,494 tets
      against 6,710) while the middle region GAINS 612 (1,172 against 560),
      because in the uniform build that region came out 3.5000 mm against an
      asked 1.6209 (2.16x COARSE -- item 2's other direction) where the graded
      build reached 1.9155. Both cases' estimator readings sit ~1.8-1.9x above
      the built mesh (shell-dense 4,952 estimated for 2,753 built; ramp-outward
      7,167/13,652 for 6,746/7,350), and the notes in `run_grading_matrix.py`
      now carry both instruments so neither reads as the other.

      ITEM 5 -- does the uniform-refined reference favour the uniform side, and
      by how much? Yes, and the bound the lead named is a FLOOR on that
      favouring rather than a ceiling. Both FAILs, at the Orin's own numbers,
      with the uniform side given the reference's WHOLE convergence:
      three-region's converges 0.046%, so 0.152% (0.106 + 0.046) against the
      graded 0.231% -- 1.52x, still FAIL; offset-sphere's converges 0.005%, so
      0.033% against 0.091% -- 2.76x, still FAIL. Read the other way the gaps
      widen (0.060% -> 3.85x, 0.023% -> 3.96x). box-sphere gets no verdict from
      that pair at all: its reference convergence (1.007%) is 70% of the
      uniform side's whole score (1.442%), so nothing in the difference is
      readable -- and on the corrected baseline neither its uniform mesh nor
      its reference fits tier 2.

      THE SCALE, and why the raw bound is the weakest form of the correction.
      The quantity measured as "the reference's convergence" is the
      coarse-vs-finest DIFFERENCE, and for a method of order p refined by
      r = K_f/K_c that difference is (1 - r**-p) of the finest level's own
      error: the finer level's error is a MULTIPLE of it. At this bench's
      1.25 -> 1.5 pair and p = 2 (linear tets, consistent mass) the multiple is
      1/(1.44 - 1) = 2.27, so the correction is ~2.27x the difference and the
      1.52x above is the floor of the uniform side's case. At that scale
      three-region is 0.106 + 2.27 * 0.046 = 0.2105% against 0.231% -- FAIL by
      1.10x, i.e. real but THIN -- and offset-sphere 0.039% against 0.091%,
      FAIL by 2.31x. Three-region's verdict would flip to a tie if the
      reference's own error were more than 2.72x the difference it reports
      (0.231 - 0.106 = 0.125 / 0.046 = 2.72), which is the honest limit of what
      these two runs can decide; offset-sphere's does not flip short of 12.6x
      ((0.091 - 0.028) / 0.005).
      `favourable_bound` computes both readings, the run prints both, and three
      tier-1 tests pin them, including the arm that makes the flip happen.
      Weaker than the bound and still worth saying: both sides are scored
      against the same reference at the same mode count, the graded mesh's fine
      region is at the same floor the reference was built on, and the finer
      reference level is what the convergence was computed from.

      ITEM 3's OTHER HALF, per FAIL, since these are the same two numbers: the
      Orin's uniform floors were the probe's, and the corrected (product) floor
      is COARSER by 7.9% in edge for three-region (1.2157 against 1.1266) and
      1.3% for offset-sphere (1.1416 against 1.1266) -- a coarser uniform mesh
      can only score WORSE, which raises the bound and narrows the FAIL. So
      both corrections (the scale above, the baseline here) push the same way,
      and three-region survives them at ~1.1x while offset-sphere survives at
      2.3x. box-sphere's uniform moves the other way entirely (2.23x FINER in
      edge, 128,225 tets against 11,564), and it has no verdict to correct.

      ITEM 6 -- the candidate and its predictions. The candidate is the rule
      such that only the coarse side moves: keep the densest region at h_min and
      bring the next region's edge toward it instead of leaving it at h. Every
      other shape of change (a transition exponent, a per-region clamp) leaves
      the coarse side at h, and the measured FAILs live exactly there: the
      unrefined region holds 4,454 of box-sphere's 7,350 tets (61%) at contrast
      3, 5,704 of three-region's 7,319 (78%) at contrast 2, and 7,058 of
      offset-sphere's 7,238 (98%) at contrast 2, and it is the region whose edge
      decides the error. PREDICTIONS, one per FAIL: three-region FAILs again
      (the coarse edge would have to come down by roughly 1.4x -- sqrt(2) for a
      second-order method, not the estimator's reading of the same number --
      before that region's error can halve, and it is 78% of the mesh);
      offset-sphere FAILs
      again (same lever, twice the deficit, essentially the whole mesh in the
      coarse region); box-sphere NO PREDICTION -- its reference cannot be built
      on tier 2 and its own convergence is 70% of the uniform's error. What the
      candidate must keep, and cannot be argued from prose: c4-thin's PASS
      (0.063% vs 0.094%, a clamp case whose clamped wall is what makes the
      graded side finer). The decider is the matrix rerun on the corrected
      baseline; this paragraph is a prediction with a falsifier, not a result.

      WHAT 5.5 GOT WRONG, in one place, since its own numbers stay as the record
      of that run: (a) the `h_min` and `uniform` columns and its sizing sentence
      ("650,465, 3.3x", "1,097,656 / 2,601,856") were computed against the probe
      floor; the corrected values are in the `--list` output below and the same
      three claims at the corrected floors are 611,826 (3.1x), 445,159 and
      873,764. (b) "c3-thin carries the SAME uniform mesh" as c3-thick is FALSE:
      the thin wall's clamp raises the case's own base edge (0.7217 against
      0.6084), so the two uniform meshes differ -- 154,673 estimated for
      c3-thin against 258,114 for c3-thick, and c3-thin's graded 220,160 and
      uniform 154,673 are no longer on the same side of the mark as each other.
      (c) "the estimator's ~1.4x-high reading" is now measured across six
      compare points: 1.22x (box-sphere uniform 128,225 estimated against
      105,321 built), 1.30x (c3-thick uniform 258,114 vs 199,062), 1.80x
      (shell-dense 4,952 vs 2,753), 1.86x (ramp-outward's uniform path 13,652
      vs 7,350), 2.0x (c2-thin's uniform path 45,829 against 22,947 built) and
      3.3x (c2-thin's graded path 75,262 against the SAME 22,947, because
      c2-thin's graded plan is its uniform plan -- the estimator's two paths
      differ by 1.64x on one and the same mesh). It is NOT reliably high:
      ramp-outward's graded path reads 7,167 against 6,746 built, 1.06x. So
      `--list` counts flag a case for the Orin and are not verdicts, in either
      direction, and the built mesh is the authority. This is the same seam
      that made c2-thin's uniform look 1.55x the product's mesh before item 3:
      two instruments, one compared against the other's number.

      THE LIST, corrected baseline (tier 1, `.venv-mse/bin/python`, exit 0).
      BOTH floors are printed side by side -- that is item 3's question, and a
      floor printed in a count's place is the defect this row is about -- with
      the length ratio between them; element counts come from the product's own
      estimator. Any case whose two floors differ by more than 5% in LENGTH
      gets a `!` line naming both counts, and one summary line counts them; the
      count ratio is never printed as a length (it is that ratio cubed, which
      is how an earlier version of this line read box-sphere as "0.09x" when
      its floor is 2.23x the product's):

      ```text
               case    ref  reg modes h_min(prod) h_min(probe)    graded   uniform len-rat  axes
         c1.5-thick analytic    2    10      1.2169       1.1923    14,958    32,264    0.98  concentric, thick, n=1.5
           c2-thick analytic    2    10      0.9127       0.8942    20,676    76,478    0.98  concentric, thick, n=2.0
           c3-thick analytic    2    10      0.6084       0.5961    35,314   258,114    0.98  concentric, thick, n=3.0
                      ! over the tier-2 mark 200,000: ['uniform'] -- expect UNMEASURED, not a pass
           c4-thick analytic    2     3      0.6809       0.6679    16,215   184,163    0.98  concentric, thick, n=4.0
            c2-thin analytic    2    10      1.0825       0.8942    75,262    45,829    0.83  concentric, thin, n=2.0
                      ! base edge 0.8942mm is 0.83x the product's 1.0825mm here, so this case's uniform side is 81,308 tets on the old baseline and 45,829 on the corrected one
            c3-thin analytic    2    10      0.7217       0.5961   220,160   154,673    0.83  concentric, thin, n=3.0
                      ! over the tier-2 mark 200,000: ['graded'] -- expect UNMEASURED, not a pass
                      ! base edge 0.5961mm is 0.83x the product's 0.7217mm here, so this case's uniform side is 274,415 tets on the old baseline and 154,673 on the corrected one
            c4-thin analytic    2     3      0.8085       0.6679   161,389   109,990    0.83  concentric, thin, n=4.0
                      ! base edge 0.6679mm is 0.83x the product's 0.8085mm here, so this case's uniform side is 195,139 tets on the old baseline and 109,990 on the corrected one
       three-region uniform-refined    3     5      1.2157       1.1266    11,974    32,361    0.93  concentric, thick, n=2.0  refs [63205, 109218]
                      ! base edge 1.1266mm is 0.93x the product's 1.2157mm here, so this case's uniform side is 40,654 tets on the old baseline and 32,361 on the corrected one
         box-sphere uniform-refined    2     5      0.3368       0.7511    18,466   128,225    2.23  box, thick, n=3.0  refs [250439, 432759]
                      ! over the tier-2 mark 200,000: ['ref1.25', 'ref1.5'] -- expect UNMEASURED, not a pass
                      ! base edge 0.7511mm is 2.23x the product's 0.3368mm here, so this case's uniform side is 11,564 tets on the old baseline and 128,225 on the corrected one
      offset-sphere uniform-refined    2     5      1.1416       1.1266     9,659    39,081    0.99  offset, thick, n=2.0  refs [76330, 131898]
        shell-dense uniform-refined    2     5      2.2972       2.2533     4,952     4,952    0.98  concentric, thick, n=2.0  refs [9672, 16713]
       ramp-outward uniform-refined    3     5      1.6209       1.5022     7,167    13,652    0.93  concentric, thick, n=2.0  refs [26664, 46076]
                      ! base edge 1.5022mm is 0.93x the product's 1.6209mm here, so this case's uniform side is 17,150 tets on the old baseline and 13,652 on the corrected one

      6 of 12 cases: the old probe's base edge differs from the product's by more than 5% (0.83x to 2.23x). Verdicts measured before 2026-10-14 are against that other baseline.

      axes covered: contrast=[1.5, 2.0, 3.0, 4.0]; shell=['thick', 'thin']; regions=[2, 3]; shape=['box', 'concentric', 'offset']; reference=['analytic', 'uniform-refined']
      tier-2 mark is 200,000 elements. production refuses an upload above 1,200,000 and sends one above 15,000 to the GPU
      ```

      ONE PRECISE BLOCKER, for the lead: on the corrected baseline box-sphere's
      REFERENCE cannot be built at tier 2 -- its finer level needs 250,439 and
      432,759 elements against the 200,000 mark, though its uniform mesh
      (128,225) and its graded mesh (18,466) both fit. c3-thick's uniform
      (258,114) and c3-thin's graded (220,160) are also over the mark. So the
      matrix as specified needs either a larger tier-2 budget for those three
      meshes, or box-sphere sized down (a thinner box, or a smaller refine
      factor for its reference), or explicit UNMEASUREDs -- the lead's call, not
      this seat's. Note also that these counts are the estimator's, which reads
      1.2-3.3x high on the meshes we have built, so the Orin may well solve some
      of them; what is certain is that no verdict for one is a pass.

- [x] 5.7 STEP 2b: the five items the lead returned with 5.6 -- bench only, one
      commit, tier 1, no product rule changed, step 3 still NO GO (row
      `600283f7`, 2026-10-14). Two files, `benchmarks/run_grading_matrix.py`
      and `tests/test_grading_matrix.py`. 47 tests green (`PYTEST_EXIT=0`),
      `openspec validate` clean, `--list` and `--estimator` both exit 0. One
      new case, `centred-core` (offset-sphere's A/B control): the lead's
      12-value `h_min` verification of `1935511` is untouched, no existing case
      builder changed (13 cases now).

      ITEM 1 -- a tie is not a verdict. `plan_is_uniform(plan)` is True when
      two or more regions share one `edge_m` (one region alone is not "uniform
      and graded"). When the graded and uniform builds are the same plan the
      verdict is `SAME MESH` -- grading did nothing, and a PASS there would
      report a rule that did nothing as a rule that held -- with exit code 5.
      The real c2-thin pair is the fixture: 22,947 tets and the same plan on
      both sides, worst errors 0.0019135674962923 against 0.0019135674962911,
      i.e. the same mesh with the difference in the last float. Otherwise the
      comparison is `graded <= uniform + tol`, where `tol` is the SUM of both
      sides' measured solver residuals (`mse_eigen.LAST_WORST_RESIDUAL`, the
      same global the solver records as `solver_worst_residual`); a missing
      residual on either side does NOT become a zero -- the runner records
      `tolerance_unavailable` ("the CPU factorisation records none") and says
      the comparison is strict, because a strict comparison presented as a
      justified one is worse than no tolerance. Guard-removed control: with
      `same_plan=False` the same c2-thin pair returns FAIL under the strict
      `<=`, so the test reddens on the guard, not on the fixture.

      ITEM 2 -- signed errors, both sides, in the JSON and the log. Every
      matched mode now carries (mode, l, family, reference GHz, graded %,
      uniform %) with the SIGN kept -- `(centre - reference_kb) / reference_kb`
      -- plus `zero_crossings` (modes whose signed error changes sign between
      the two meshes) and `uniform_level_errors` (each already-built uniform
      rung's worst error, no extra solve). A verdict is flagged `UNRELIABLE`
      (exit 6, `provisional_verdict` kept) when the uniform ladder does not
      fall monotonically, or the case reaches only one multiplet, or a
      crossing is present. **I cannot answer the c4-thin 0.004%
      zero-crossing question from the Orin artifacts**: they carry
      per-multiplet MAGNITUDES only (the raw pairs were filtered out of the
      JSON before this change), and the sign of a 0.004% error is exactly what
      round-off decides. The command that answers it is below. `--ladder` adds
      ONE rung coarser than the comparison (`h_min / 0.8`) because the finest
      rung of a uniform-refined case IS the reference -- its error is 0 by
      construction, which is the blind spot the old Orin JSONs show as
      `uniform_level_correction_from=1.25`: a crossing can be seen from below
      only.

      ITEM 3 -- over-mark on BUILT counts, the 200,000 mark kept, the bases not
      coarsened. `sides_over_mark` is computed from `n_tets_graded` and the
      uniform `1.0` build, never from `--list`'s estimate, so c4-thin (estimate
      161,389, built 61,244) is MEASURED, not refused; c3-thick's uniform is
      199,062 built against the 258,114 estimated, i.e. under the mark on the
      count that judges it. Cases the BUILD puts over stay UNMEASURED.
      box-sphere stays NO VERDICT -- its reference converges only to 1.007% --
      and that is now its own exit code (7), not an UNMEASURED.

      ITEM 4 -- offset-sphere diagnosis, no rule change. Measured at tier 1
      (meshing only, no solves; `item4_regions.py`, `item4_ab.py`):

      ```text
        offset-sphere   7,238 tets | wall asked 2.2831 built 1.6771,   7,058 tets (97.5%)
                                  | core asked 1.1416 built 1.4195,     180 tets (2.5%), bbox centre z=+4.0 mm
                      uniform  20,266 | core 161 @ 1.5375 mm | near wall 2,349 @ 1.4694 mm
        centred-core    8,026 tets | wall asked 2.2831 built 1.6465,   7,835 tets (97.6%)
                                  | core asked 1.1416 built 1.4019,     191 tets (2.4%), centre (0,0,0)
                      uniform  20,323 | core 166 @ 1.4945 mm | near wall 2,462 @ 1.4477 mm
      ```

      So the offset case is NOT under-resolved relative to its centred control
      anywhere measurable: its core carries 180 tets at 1.4650 mm against the
      control's 191 at 1.4381 mm, and its near-wall band 1,701 at 1.6415 mm
      against 1,724 at 1.6411 mm -- the graded side is if anything very
      slightly finer in both. The tuned count ratios match too (graded/uniform
      total 0.36x vs 0.39x, core 1.12x vs 1.15x, near wall 0.72x vs 0.70x). The
      natural story -- "an off-centre core gets pushed coarse" -- is FALSIFIED:
      there is no coarse core to blame, and the two graded meshes are
      near-identical in structure. (The estimator is offset-blind as well:
      both cases read 9,659 / 39,081 -- the offset never enters the estimate.)
      PREDICTION: the 2.27x is not a local resolution deficit. FALSIFIER: solve
      `centred-core` on Orin at the corrected baseline. If it PASSES while
      offset-sphere FAILs, density is ruled out and the deficit is specific to
      the offset partition -- whose only remaining geometric difference is its
      thinner wall (min wall 3.5 mm against 7.5 mm), i.e. where the region
      boundary sits, not how fine the core is. If `centred-core` FAILS at a
      similar ratio, the cause is shared and is not the offset at all.

      ITEM 5 -- the estimator, a PRODUCT finding, reported and not changed. One
      row per case: the live routing path (`materials=`) and the bench's
      uniform path (`min_feature_m=`) against BUILT counts from this bench's
      own `--json` (`--estimator --results /tmp/mse-orin`, exit 0):

      ```text
      live routing (read from config): route to GPU at or above 15,000; refuse an upload above 1,200,000 (= 2x the real cap 600,000); tier-2 mark 200,000 (built counts judge that, not estimates)
               case est(materials=) est(min_feature_m=)  built graded  built uniform built/est  source
         c1.5-thick          14,958              32,264     not built      not built      ----  no built counts here
           c2-thick          20,676              76,478     not built      not built      ----  no built counts here
           c3-thick          35,314             258,114     not built      not built      ----  no built counts here
           c4-thick          16,215             184,163     not built      not built      ----  no built counts here
            c2-thin          75,262              45,829        22,947         22,947     0.30x  c2-thin.json
                      ! the min_feature_m= estimate is 2.00x the BUILT uniform count (45,829 vs 22,947); the bench builds its uniform side by this path
                      ! the materials= estimate is 3.28x the BUILT count (75,262 vs 22,947) -- above the 2x the refuse threshold (web/config.py:248) is documented to allow
            c3-thin         220,160             154,673     not built      not built      ----  no built counts here
            c4-thin         161,389             109,990        61,244         57,872     0.38x  c4-thin.json
                      ! the min_feature_m= estimate is 1.90x the BUILT uniform count (109,990 vs 57,872); the bench builds its uniform side by this path
                      ! the materials= estimate is 2.64x the BUILT count (161,389 vs 61,244) -- above the 2x the refuse threshold (web/config.py:248) is documented to allow
       three-region          11,974              32,361         7,319         11,688     0.61x  three-region.json
                      ! the min_feature_m= estimate is 2.77x the BUILT uniform count (32,361 vs 11,688); the bench builds its uniform side by this path
                      ! the min_feature_m= estimate is 2.77x the BUILT count (32,361 vs 11,688) -- above the 2x the refuse threshold (web/config.py:248) is documented to allow
         box-sphere          18,466             128,225     not built      not built      ----  no built counts here
      offset-sphere           9,659              39,081         7,238         20,266     0.75x  offset-sphere.json
                      ! the min_feature_m= estimate is 1.93x the BUILT uniform count (39,081 vs 20,266); the bench builds its uniform side by this path
       centred-core           9,659              39,081     not built      not built      ----  no built counts here
        shell-dense           4,952               4,952     not built      not built      ----  no built counts here
       ramp-outward           7,167              13,652         6,746          7,350     0.94x  ramp-outward.json
                      ! the min_feature_m= estimate is 1.86x the BUILT uniform count (13,652 vs 7,350); the bench builds its uniform side by this path
      ```

      Reading: the live `materials=` estimate is HIGH on every case we have
      built (built/est 0.30x to 0.94x), so the site errs on the conservative
      side -- it sends to the GPU and refuses uploads more readily than the
      built mesh warrants, not less. But the overshoot is larger than the
      margin the refuse threshold is documented to tolerate (2x,
      `web/config.py:248`, i.e. refuse at 1,200,000 against a real cap of
      600,000): c2-thin 3.28x and c4-thin 2.64x are over that margin, so an
      upload just under the refuse line can be refused at well under 600,000
      built. The bench's own `min_feature_m=` path is worse (1.86x to 2.77x),
      and it is the path this matrix builds its uniform side by -- which is why
      `--list`'s counts are 1.2-3.3x high and are not verdicts in either
      direction. Not changed here; the numbers are the finding.

      THE COMMANDS FOR THE LEAD (tier 2, Orin; I ran none of these -- whole-case
      solves are the lead's). Adapt the path/venv to your Jetson invocation;
      the flags are what matters:

      ```text
      # 1. c4-thin: signed per-mode errors on BOTH sides + one coarser rung,
      #    which is what answers the 0.004% zero-crossing question (item 2).
      run_grading_matrix.py --case c4-thin --baseline product --ladder --backend gpu --json c4-thin-ladder.json

      # 2. the offset-sphere A/B control (item 4's falsifier): if this passes
      #    where offset-sphere failed, local density is ruled out.
      run_grading_matrix.py --case centred-core --baseline product --backend gpu --json centred-core.json

      # 3. optional, for item 2's ladder on the passing thin case:
      run_grading_matrix.py --case c2-thin --baseline product --ladder --backend gpu --json c2-thin-ladder.json
      ```

      Exit codes the runner now returns: 0 PASS, 1 INCOMPLETE/FAIL, 5 SAME
      MESH, 6 UNRELIABLE, 7 NO VERDICT. The `--ladder` rung is COARSER than the
      comparison (fewer elements than the base uniform mesh), so it should
      build where c4-thin's level-1.5 finer rung failed on Orin.

      Bench only. No product rule, estimate or threshold is touched. Step 3
      remains NO GO.

- [x] 5.8 STEP 2c: the four items the lead returned with 5.7 -- bench only, one
      commit, tier 1, no product rule changed, step 3 still NO GO (row
      `600283f7`, 2026-10-14). Two files, `benchmarks/run_grading_matrix.py` and
      `tests/test_grading_matrix.py`. 54 tests green (`PYTEST_EXIT=0`, 141.75s),
      `openspec validate` clean. `--list` still exits 0 (`LIST_EXIT=0`,
      measured with the flag), and now SAYS so when handed `--fine-reference`:
      its refs column comes from `REFINE_FACTORS`, so a silently absorbed flag
      would let a reader size the requested rung off the wrong row, and an
      ESTIMATED count for it would be worse than none (the bench's cube law
      misses k=2.0 by 1.43x, see ITEM 1). The tier-1 run caught
      one real defect of my own before the commit: `self_ladder` called
      `REF._clusters_pairs`, but `_clusters_pairs` is a module-level function of
      the runner (`sphere_dielectric`, which has no such attribute, is
      `score_multiplets`' module). Every `run_case` with two or more rungs raised
      `AttributeError`; 11 tests were red, 8 of them pre-existing. One-line fix
      (`_clusters_pairs(levels[finest])`, the same call the reference builder
      makes), all 54 green. Nothing was committed while red.

      ITEM 1 -- a finer reference rung, as an OPTION not a default.
      `--fine-reference K` adds ONE rung at `h_min/K`, finer than the default
      finest (1.5), to `REFINE_FACTORS`; it is opt-in because it is a real
      build, and `refine_factors` REFUSES `K <= 1.5` (a "finer" rung that is not
      finer would silently REPLACE the finest rung with a coarser one and
      loosen the reference). For a uniform-refined case the reference IS the
      finest rung's build, so a finer rung tightens the reference's own
      convergence -- the knob for the undecidable gaps (centred-core's 0.038%
      under 0.043%; offset-sphere 3.27x). `richardson_scale(factors)` derives
      `1/(r**2 - 1)` from the run's own tuple, so the correction can never come
      from a different ladder (default pair 2.27; with 2.0, 0.64).

      The measured price, tier 1, meshing only (a scratch probe in /tmp
      (`probe_item12.py`); rerun it before quoting -- these are measurements,
      not a spec):

      ```text
        k     centred-core   offset-sphere   c4-thin
        1.0        20,323          20,266      57,872
        1.25       39,448          48,568     189,127
        1.5        81,091         103,623     215,551
        1.75      130,447         142,555     355,400   <- the affordable k
        2.0       232,242   !     278,569   !  595,791
                  ^ over the 200,000 tier-2 mark for BOTH sphere cases
      ```

      So k=2.0 does NOT fit this matrix (232,242 and 278,569 built against the
      200,000 mark) and c4-thin fits neither at any level at or above 1.5; k=1.75
      is the affordable tightening (130,447 / 142,555, both under). The bench's
      own cube-law estimate (`uniform_count * k**3`, the estimator's rule) would
      have said 162,584 for centred-core at k=2.0 -- 1.43x UNDER the 232,242
      actually built, another face of item 5 and a reminder to spend the build,
      not the estimate, when the price is what is being asked.

      ITEM 2 -- why the uniform error rises with refinement, and marking the
      analytic reference unusable there. TWO claims, one falsified first:

      * FALSIFIED: "the mesh is a frozen polyhedron and the reference describes a
        different volume/radius". An icosphere R=4.0 at te=1.0 mm has faceted
        volume 0.991394x the ideal sphere (0.86% low), AND the surface refines:
        distinct near-surface radii grow 18 -> 59 -> 2662 as the edge halves
        1.0 -> 0.5 -> 0.25 mm (`probe_item2.py`). The mesh is not fixed; it
        refines. So the faceting story is not what moves the error.
      * THE MEASURED CAUSE: a THIN region is pinned by its own thinness and the
        achieved edge does not track the requested one. c4-thin's wall
        (thickness 0.9943 mm) achieves an edge far coarser than asked at every
        level (target 0.5390 -> achieved 0.6974 at k=1.5; 0.4043 -> 0.4838 at
        k=2.0), while the core (5.9768 mm) refines -- and the core's own
        achieved edge is NOT monotone in k (0.4636 at 1.25, then 0.5032 at
        1.5). The "ladder" is therefore not one family: each rung is a
        differently-refined mesh, and eigenvalues wander at the 0.1% level,
        which is the scale of the errors being compared (0.034%, 0.004%,
        0.098%, 0.191%). A non-falling ladder here is a statement about the
        FIXTURE, not about grading.

      Fix, exactly as asked: `reliability(..., reference_kind="analytic")` now
      appends, to the case it already refuses as UNRELIABLE (exit 6), why it can
      never be improved: "the reference is ANALYTIC, a closed-form spectrum with
      no finer level of its own to converge into, so it cannot be tightened --
      this case is unusable as a grading fixture until it has a meshed
      reference". A meshed reference gets the same non-falling diagnosis WITHOUT
      that clause (it can be refined, so the case is unjudged, not condemned).
      Note `--fine-reference` cannot rescue an analytic case: it adds a MESH
      rung, and an analytic reference is computed, not built, so the reference
      does not change.

      INSTRUMENT for the next Orin run: `self_ladder(levels, n_modes)` scores
      every uniform rung against the FINEST rung's OWN spectrum (the reference
      taken out), costing no extra solve, because for a FIXED reference the
      change in the error against it between two rungs equals the change between
      those rungs' own spectra. `_self_ladder_note` reads it against the
      reference step and attributes the non-falling ladder either way ("the mesh
      has converged and the REFERENCE is what moved" vs "the mesh itself is
      still moving and no reference at these levels can decide it"). It prints
      as `[matrix] uniform SELF-ladder ...` and travels in the JSON.
      PREDICTION (c4-thin): the 1.25 -> 1.5 self-gap is about 0.093% (the
      difference of the two reference-side errors 0.098% and 0.191%, same sign),
      i.e. at least the 0.093% reference step, so the note reads "the mesh
      itself is still moving" and the analytic reference is not merely mistuned
      but unable to decide this case. FALSIFIER: if the printed SELF-ladder
      shows that gap far BELOW 0.093%, the mesh has converged and the
      reference-geometry story survives after all.

      ITEM 3 -- the duplicate print. WHAT I FOUND: on this code the
      favourable/Richardson pair prints ONCE per case -- `grep -c "most
      favourable"` is 1 in `centred-core.log` and 0 in every other log under
      `~/mse-bench/matrix-f6f0efd`, so I could not reproduce "prints twice" from
      the artifacts. WHAT WAS duplicated and is now fixed: (a) the reference's
      convergence percentage printed on a standalone line AND again inside the
      favourable line's arithmetic; (b) on UNRELIABLE the reliability sentence
      appeared both at the end of the verdict detail and on a separate
      `verdict reliability` line. Both are now single-carrier: the standalone
      convergence line prints only when the favourable line cannot be built (a
      missing worst error leaves the bound with an operand missing), and the
      reliability line only when the verdict is NOT UNRELIABLE (whose detail
      already ends with it). Test, with its own control: the favourable case
      prints the standalone label zero times; the same run with no graded error
      prints it exactly once. If the pair still appears twice on the lead's
      terminal, that is not in the code or the artifacts I can read -- paste it.

      ITEM 4 -- the exact Orin commands (tier 2; I ran none of these -- whole-case
      solves are the lead's). The flags are what matters; adapt the path/venv:

      ```text
      # 1. centred-core with the closest AFFORDABLE reference. k=1.75 built
      #    130,447 (under 200,000); k=2.0 is 232,242 and must NOT be used.
      run_grading_matrix.py --case centred-core --baseline product --fine-reference 1.75 --backend gpu --json centred-core-fine.json

      # 2. offset-sphere, same tightening (142,555 built at k=1.75):
      run_grading_matrix.py --case offset-sphere --baseline product --fine-reference 1.75 --backend gpu --json offset-sphere-fine.json

      # 3. c4-thin: the --ladder run, unchanged. Item 2 needs no new rung -- the
      #    SELF-ladder is computed from rungs already built; this reproduces the
      #    UNRELIABLE and now records WHY, in the JSON and the log.
      run_grading_matrix.py --case c4-thin --baseline product --ladder --backend gpu --json c4-thin-ladder2.json
      ```

      What the new JSON carries: `refine_factors` (the tuple that ran, so the
      Richardson scale can be checked against it), `uniform_self_ladder`
      (per-rung `self_rel_error` against the finest), and, for a finer-reference
      run, a reference built from the newest finest rung. Expectation for command
      1: the reference's convergence at 1.25 vs 1.75 is tighter than 0.043%, and
      if it drops below the 0.038% gap the case becomes decidable; the SELF-ladder
      is the check that the tightening is honest, since it shows whether the
      finest two rungs are still moving.

      Bench only. No product rule, estimate or threshold is touched. Step 3
      remains NO GO.

### 5.9 A per-case scatter band from the case's own uniform ladder (step 2d, notes only)

No code, no runs. Everything below is read from the JSONs the runs already
wrote (`jetson:~/mse-bench/matrix-*/`; the lead copied 1935511, 34ff8ca and
f6f0efd to `/tmp/mse-orin/`, and 4883a55 / 6b97968 were read read-only on the
Jetson).

**Why a band at all.** The comparison this bench exists to make -- graded mesh
vs today's uniform mesh at the product's own `h_min` -- is a ~0.1% effect. The
uniform side's own error moves by a comparable amount when only its resolution
changes. A difference smaller than that movement is not resolvable by this
instrument, whatever the verdict code prints. The band says how small is too
small, per case, without a constant fitted to these cases.

**The proposal.** From the case's JSON take `case.uniform_level_errors`, the
uniform side's error `e_u(k)` at each rung actually run.

1. **Drop the rung that is the reference.** When
   `reference_kind == "uniform-refined"` the finest uniform rung *is* the
   reference, so its error is 0 **by construction** -- it is a tautology, not a
   measurement. Including it injects a jump equal to the whole ladder range.
   Measured on `34ff8ca/centred-core-fine`: the retained (non-reference) rungs
   jump by at most 0.049%, but the last step into the reference rung is 0.139%
   -- 2.8x wider, wide enough to swallow a real 0.113% gap and call it scatter.
   When `reference_kind == "analytic"` no rung is the reference, so keep them
   all.
2. **band(case) = max over adjacent retained rungs of `|e_u(k_{i+1}) - e_u(k_i)|`.**
   Properties: dimensionless; from the case's own numbers only; no constant
   fitted to these cases; computed from the **uniform side alone**, so the
   yardstick does not depend on the grading rule under test; conservative (the
   largest single step, not an average).
3. Rejected alternatives: `(max - min)/2` of the ladder -- also constant-free,
   but it grows with the ladder's spread, i.e. it forgives the most erratic
   ladder the most (0.047% on c4-thin against 0.093% for the max-step rule);
   half the finest step only -- measures residual movement and under-reads.
4. **Reading.** On the raw basis, `d = graded_worst - uniform_worst`:
   clearly worse if `d > band`, clearly better if `d < -band`, within scatter
   otherwise. If the ladder is absent, the case is **`band unmeasured`** --
   *not* "within scatter". (An unmeasured band must not read as agreement.)
5. **Then require the two bases to agree.** The bench's own criterion does not
   use the raw uniform error: it compares `graded_worst` with the uniform error
   Richardson-extrapolated to the graded mesh's *achieved* edge. The two bases
   can disagree in direction, so a case is called clearly better/worse only
   when the raw step and the extrapolated bound agree; otherwise "within
   scatter / direction unresolved".

**What the band covers, and what it does not.** It measures how much the
uniform side moves when only *resolution* changes. It does not cover
style-driven differences at equal resolution, and it is blind to a reference
that is itself wrong (c4-thin's analytic reference, section 5.8 item 2).

**A. The five artifacts that carry a ladder** (all were run with `--ladder`):

| case | revision | ref kind | graded | uniform k=1 | ladder `e_u(k)` | band | `d` | reading |
|---|---|---|---|---|---|---|---|---|
| centred-core | 34ff8ca | uniform-refined | 0.161% | 0.047% | 1.0 0.047 / 1.25 0.096 / 1.5 0.139 / *(1.75 = ref)* | 0.049% | +0.113% | raw 2.3x band -> worse; extrapolated 0.148% -> +0.013% -> within. **unresolved** |
| offset-sphere | 34ff8ca | uniform-refined | 0.097% | 0.055% | 1.0 0.055 / 1.25 0.110 / 1.5 0.091 / *(1.75 = ref)* | 0.054% | +0.042% | raw within; extrapolated 0.169% -> graded 0.57x -> better. **unresolved** |
| c4-thin | 34ff8ca | analytic | 0.063% | 0.004% | 0.8 0.034 / 1.0 0.004 / 1.25 0.098 | 0.093% | +0.059% | within; UNRELIABLE regardless (analytic reference unusable) |
| centred-core | f6f0efd | uniform-refined | 0.130% | 0.092% | 1.0 0.092 / 1.25 0.043 / *(1.5 = ref)* | 0.049% | +0.038% | within (0.79x band), and the extrapolated basis agrees |
| c4-thin | f6f0efd | analytic | 0.063% | 0.004% | 0.8 0.034 / 1.0 0.004 / 1.25 0.098 / 1.5 0.191 | 0.093% | +0.059% | within; UNRELIABLE regardless |

The `34ff8ca` rows are the point of step 2c: with the reference tightened to
1.75, **centred-core and offset-sphere both stop being resolvable effects** --
the raw gap and the extrapolated gap disagree on direction, so neither can be
called. The lead's own numbers say the same (0.161 vs 0.148 = 1.09x; 0.097 vs
0.169 = 0.57x). The `f6f0efd` centred-core row shows the band *confirming* the
existing NO VERDICT (0.038% gap under a 0.049% band).

**B. The fifteen artifacts with no ladder** -- read from the JSONs only, so
`band unmeasured`; raw `d` is shown because it is free, but no scatter reading
is possible:

| case | revision | verdict | graded | uniform | `d` |
|---|---|---|---|---|---|
| box-sphere | 4883a55 | UNMEASURED | -- | 0.882% | -- |
| c1.5-thick | 4883a55 | PASS | 0.379% | 1.113% | -0.734% |
| c2-thick | 4883a55 | PASS | 0.341% | 0.559% | -0.218% |
| c2-thin | 4883a55 | PASS | 0.191% | 0.194% | -0.003% |
| box-sphere | 6b97968 | UNMEASURED | -- | 1.442% | -- |
| c3-thick | 6b97968 | FAIL | 0.749% | 0.662% | +0.087% |
| c4-thick | 6b97968 | UNMEASURED | 1.465% | -- | -- |
| c4-thin | 6b97968 | PASS | 0.063% | 0.094% | -0.031% |
| offset-sphere | 6b97968 | FAIL | 0.092% | 0.028% | +0.064% |
| three-region | 6b97968 | FAIL | 0.231% | 0.106% | +0.125% |
| c2-thin | 1935511 | FAIL | 0.191% | 0.191% | +0.000% |
| c4-thin | 1935511 | FAIL | 0.063% | 0.004% | +0.059% |
| offset-sphere | 1935511 | FAIL | 0.098% | 0.043% | +0.055% |
| ramp-outward | 1935511 | PASS | 0.279% | 0.779% | -0.500% |
| three-region | 1935511 | PASS | 0.221% | 0.274% | -0.053% |

Two things this table says that the per-run messages did not:

- **The revision must be in the row.** `c4-thin` reads PASS at `6b97968`, FAIL
  at `1935511`, UNRELIABLE at `f6f0efd`. The same case name is three different
  claims, so a bare case name carries no verdict.
- **`c2-thin @ 1935511` has `d = 0` to 13 significant digits** -- graded and
  uniform are the same mesh at that `h_min`, so its 0.000% is not a
  measurement of grading (the explicit SAME MESH verdict only arrived in
  `f6f0efd`).

**Cost, and how to band the rest.** The band is free: it is already in the
JSON, and `f6f0efd` added `uniform_self_ladder` for the same rungs. The only
requirement is that the run used `--ladder`; **5 of these 20 artifacts do**.
To band a case in table B, re-run it with `--ladder` (same command shape as
5.8 item 4, e.g.
`run_grading_matrix.py --case <name> --baseline product --ladder --backend gpu
--json <name>.json`). No new run is proposed here -- that is the lead's call.

If the two superseded revisions should be in the canonical set, copy
`jetson:~/mse-bench/matrix-4883a55/*.json` and
`jetson:~/mse-bench/matrix-6b97968/*.json`; the other three are already on the
VM.

### 5.10 Two bench defects the 9118aec ladders exposed (step 2e)

The lead's 2026-10-14 ladder runs (`9118aec`, JSONs in
`~/mse-bench-results/ladders-9118aec/`) left one clear pass, no clear fail, and
an instrument that could not decide the rest. Both items below are **bench
only**: the graded rule, the mesher and the product are untouched, and Step 3
stays NO GO.

#### A. The Richardson line read its flag on the wrong side of the verdict

`favourable_bound` (item 5, 5.8) shifts the uniform side the one way that
**helps grading**: it ADDS the reference's own convergence. On a PASS case that
shift can only widen grading's margin, so `still_fail` is false *by
construction*, and the printer read it as `-- FLIPS: not decidable from these
runs`. Both cases in the ladder log printed that line -- for opposite reasons.
The bound that can overturn a verdict is the one most favourable to that
verdict's **loser**, and the loser changes with the verdict:

- **PASS** loses if the uniform side was *better* than measured, so the
  overturning bound is `uniform - c`, and the PASS flips when graded is **above**
  it. This is the lead's own reading for ramp-outward: *"graded wins even if you
  SUBTRACT the reference's convergence from uniform (0.779 - 0.096 = 0.683;
  Richardson 0.779 - 0.218 = 0.561)"*.
- **FAIL** loses if the uniform side was *worse* than measured -- the direction a
  same-family reference biases it -- so the bound is `uniform + c`, and the FAIL
  flips when graded is **at or under** it.

So the two verdicts sit on **opposite sides** of the measured uniform number,
which is why one printed bound cannot serve both. Implemented as
`adversarial_bounds(bound, verdict)` (one entry per scaling, `[]` for any
verdict with nothing to overturn: UNRELIABLE, NO VERDICT, SAME MESH, UNMEASURED),
printed for both the raw and the Richardson scaling. A lower bound that goes
below zero -- the reference's convergence swallowing the whole uniform number --
is clamped at 0 with `clamped: true`, and the printer says the flip is a limit of
the instrument, not a measurement. The dead `if fb.get("uniform") is None` branch
(the function already returns `None` when an operand is missing) went with it.

Four tests, one per side, plus the regression's own positive control:

| case | verdict | uniform | graded | `c` | bound | reading |
|---|---|---|---|---|---|---|
| ramp-outward | PASS | 0.779% | 0.279% | 0.096% | 0.683% raw / 0.561% Richardson | PASS **holds**; old line printed "FLIPS: not decidable" |
| three-region | PASS | 0.274% | 0.221% | 0.089% | 0.185% raw | PASS **flips** -- matches the lead's "0.274-0.089 = 0.185 < 0.221" |
| three-region, graded forced to 0.500% | FAIL | 0.274% | 0.500% | 0.089% | 0.363% raw | FAIL **holds** (graded far above the de-flattered uniform) |
| three-region, graded forced to 0.290% | FAIL | 0.274% | 0.290% | 0.089% | 0.363% raw | FAIL **flips** to PASS |

#### B. The 5.9 band rule reads convergence as scatter

5.9's band is the largest adjacent jump of the uniform ladder. On
**ramp-outward** the ladder falls monotonically (0.81, 0.78, 0.10, 0 at
k = 0.8, 1, 1.25, 1.5), so its one big step -- 0.78 -> 0.10 = **0.68%**, the
reference rung dropped exactly as 5.9 requires -- *is* its convergence. The real
gap is `uniform - graded` = 0.779 - 0.279 = **0.50%**, so 5.9's rule would print
"within scatter" on the one case in the ladder set the instrument can actually
decide. More refinement makes the convergence step larger, so the rule punishes
the cleanest cases hardest. A monotone drop is convergence, not noise.

**The measure proposed instead (the lead's suggestion, implemented).** Rebuild
the uniform mesh at ONE resolution for a few rigid rotations of the **same
upload** and take the spread of the worst error across them. A rotation changes
nothing physical, so whatever moves is mesh-to-mesh noise; no constant is fitted
to these cases; the uniform side alone is used, so the yardstick does not depend
on the grading rule under test.

- **`--rotations N`** (N in 1..5; `ROTATIONS` is a fixed table of five
  `(degrees, axis)` and the flag is refused outside that range rather than
  silently sliced). Each orientation is a full uniform build+solve at the case's
  own `h_min` and `k = 1.0` -- **the cheap half of the case**.
- The **identity build is a member** and is *not* rebuilt: it is the uniform
  number the verdict itself used, so the band is measured against the gap being
  judged. `N` rotations therefore report `N + 1` members.
- The rotation is applied to the upload's **OBJ `v` lines about the object's own
  centroid**, and the same rigid rotation is handed to the eps assignment
  (`_eps_by_layers(..., rot=, about=)`) so the LAYERS rotate with the object --
  the layers say where the material is *on the object*. Rotating the mesh and
  reading it against the unrotated layer frame would assign the outermost
  material to most of the object and still run without error, so the frame
  direction is pinned by a tier-1 test on an off-centre-core box (a rotated
  centroid must read its core material; the same point unrotated must read the
  outer box; the forward direction must fail).
- **`band = max - min`** of the measured errors, and it is **`None`** -- printed
  `UNMEASURED`, never 0 -- as soon as any member failed or matched nothing: a
  spread over the survivors would shrink toward "clean mesh" exactly when the
  run knew least. The note names which orientation failed and how many of how
  many measured.
- JSON: `uniform_rotations` (`[{degrees, axis, worst_rel_error, n_tets}]`),
  `uniform_rotation_band`, `uniform_rotation_note`; the printer states the band
  in percent and the gap `uniform - graded` as a multiple of it, marked within /
  outside mesh noise.

**Prediction (to be tested on the Orin, tier 2 -- the lead runs it).** For
ramp-outward the band should come out **well under the 0.50% gap** (a rotation
of the same geometry at the same target edge should move the worst error by a
few hundredths of a percent, since grading's advantage is geometric, not
orientation-dependent), so the one clear pass survives. **Falsifier:** a band at
or above 0.50% demotes ramp-outward, and the ladder set would then hold no
decidable case at all. For **c1.5-thick and c2-thick** -- whose ladders do not
fall (1.01/1.09/0.27/0.29 and 0.17/0.60/0.53/0.44) and so read UNRELIABLE -- the
rotation band is the instrument that can say whether that movement is noise or
convergence, which is the question those two cases leave open.

**What it does not cover.** Orientation noise at the uniform rung only; not the
graded side's own noise, and not the reference's error (that is the A line
above). A case whose reference is itself suspect (c4-thick: uniform matched 0 of
64 reference multiplets) is not rescued by a small band -- the band would be
measured against a reference that cannot rank the modes.

**Honest scope.** No `--rotations` run has been made from this seat: meshing and
solves are tier 2 and the lead's. The code is tier-1 tested with the heavy calls
faked (four new tests: the band arithmetic and its unmeasured case, the frame
direction, the rotated-bytes fidelity, and the wiring through `run_case` to the
JSON and the print), so **the band's real magnitude is unmeasured here** and the
prediction above is a prediction, not a result.

**The Orin commands (tier 2, the lead's).** Same shape as the ladder runs, with
`--rotations` in place of `--ladder`:

```
cd ~/mse-bench
(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case ramp-outward --baseline product --rotations 5 --backend gpu \
   --json matrix-9118aec/ramp-outward-rot.json) > ramp-outward-rot.log 2>&1
```

and the same for `three-region`, `c1.5-thick` and `c2-thick` (the four cases
with a usable reference in the ladder set). `--rotations 5` is the maximum and
costs five uniform builds and their solves on top of the case. Add
`--ladder --fine-reference 1.75` to a case only if its own reference needs
tightening.

#### Where this stopped

Item A is implemented and tier-1 green; item B is implemented, tier-1 green, and
**unmeasured on real meshes**. Both are bench-only. One commit reports in the
step-2e message to the lead. Step 3 (the product rule change) remains **NO GO**
and nothing here reopens it.

### 5.11 The reference's error is shared, and BOTH sides get a band (step 2f)

The lead's rotation runs at `8a77aa1` (`jetson:~/mse-bench/matrix-8a77aa1`, six
orientations per case) settled 5.10's question: **mesh noise is small**.
ramp-outward band **0.033%** against a **0.500%** gap (15x), c1.5-thick
**0.017%** against **0.712%** (41x), c2-thick **0.078%** against **0.255%**
(3.3x), three-region **0.034%** against **0.053%** (1.5x, outside the band). So
noise no longer decides these cases; **the reference's own error does**. Three
bench-only items below, in reply to the lead's 2026-10-14 10:45Z message. Step 3
stays **NO GO**.

#### 1. The reference bound is SHARED between the two sides

5.10-A's bound shifts the uniform side alone. But graded and uniform are both
scored **against the same reference**, so a reference error `d` moves both
*signed* errors by the same `d`:

- `e_u` → `e_u + d`, `e_g` → `e_g + d`, and the verdict compares `|e_u + d|`
  against `|e_g + d|`.

The bound that can overturn the verdict therefore is not `uniform ± c` but, per
multiplet, the **worst case over `d ∈ [-c, +c]`** of
`|e_uniform + d| − |e_graded + d|`. Implemented as
`shared_reference_bounds(rows, convergence, richardson=1.0)` — one entry per
scaling (raw and Richardson), built from the **paired** rows of
`per_mode_table` (a row with only one side's signed error is not a comparison,
and the count travels with the bound as `n_paired`/`n_rows`), with
`ends[0]` = `d = -shift` and `ends[1]` = `d = +shift`. A row with nothing paired
has `margin = None` and is **not** a bound.
`shared_verdict_supported(row)` reads the two ends together: `PASS` when both
end margins are positive (uniform the worse side at each end), `FAIL` when both
are negative, `NEITHER` when they disagree, `None` when an end has no margin.

**The property that makes this the right shape.** When `e_u` and `e_g` have the
same sign, `d` **cancels** in the margin, both ends are identical, and the
shared margin equals the raw gap. The shared bound only bites **across a zero
crossing** — which is exactly where the one-sided rule was wrong and where
c2-thick's sign-changing multiplet lives. It is strictly weaker than the
one-sided bound (never overturns a verdict the one-sided bound keeps) and never
stronger in the helpful direction.

**It is printed beside the verdict, not wired into it.** The rule that decides
PASS/FAIL/NO VERDICT (`criterion`) is **unchanged**, including its
worst-multiplet aggregation. The printer prints the shared bound and, when the
shared reading disagrees with the verdict above it, says so: `(the verdict above
is …, from the one-sided rule)`. Folding the shared bound into `criterion` would
change what a verdict *is* — a decision for the lead and the owner, not a bench
edit, and it is deliberately **not** made here.

Four tier-1 tests, on the numbers the lead's message supplies (three-region
u = 0.274%, g = 0.221%, c = 0.089%): the one-sided rule flips on this case while
the shared rule **holds**; a case with a signed error across zero where **both**
flip (`NEITHER` at the two ends, the one-sided lower bound clamped at 0); the
paired-row count; and the no-paired-row case, which must not read as a bound.

#### 2. `--rotations` now covers BOTH sides

The uniform band says how much of the uniform NUMBER is noise; a verdict
compares two numbers, so the graded side needs its own band before a thin gap
can be read — and it is the direct answer to the sign-change worry: if
c2-thick's graded 0.341% sat near a zero crossing by luck, its rotations scatter
widely, while a real error barely moves. `_graded_rotations` rebuilds the graded
mesh (the product's own `solve`) at each orientation, scoring it against the
same reference; each rotation gets **its own store directory**
(`graded-rot<i>`) so the band never overwrites the run's primary graded artifact.
The identity member on each side **reuses the run's own number** (no second
solve), so `--rotations N` costs `N` uniform + `N` graded builds+solves, `N + 1`
members per side.

JSON: `graded_rotations`, `graded_rotation_band`, `graded_rotation_note` (same
`UNMEASURED` discipline as the uniform side: any failed member ⇒ `None`, never a
survivors-only spread). The printer prints each side's band and reads the gap
`uniform − graded` against the **larger of the two measured bands** — a gap
smaller than *either* number's own spread has not been shown to exist.

#### 3. A ladder rise within the case's own band is not a rise

`reliability(..., tolerance=...)` takes the case's own **measured** rotation band
(`uniform_rotation_band`) as the tolerance. A consecutive rise of the uniform
ladder **≤ that band** has not been shown to be a rise: two meshes of the SAME
resolution already differ by that much. The tolerance is the case's own measured
number, never a constant chosen here; `tolerance=None` (no band measured, e.g.
`--rotations` not given) keeps the old strict reading, so a case with no band
**stays UNRELIABLE** — "small enough that it might be noise" is not the same as
"measured to be noise". The message names the band it used and names the excused
rises explicitly (`; a smaller rise 0.100%->0.115% is within that band and is not
counted`), so the comparison is checkable.

**c1.5-thick is BORDERLINE and stays UNRELIABLE — stated plainly, no silent
loosening.** The lead's numbers: ladder rises 0.271 → 0.292% (k 1.25 → 1.5), a
rise of **0.021%**, against a measured band of **0.017%**. `0.021% > 0.017%`, so
under the literal rule this case is still UNRELIABLE. The rule was not bent to
pass it — the band is the case's own measured number and the rise is genuinely
bigger than it. (This is the item where a fixed constant would have been
tempting; it is exactly why the rule takes the measured band instead.)

#### The Orin commands (tier 2, the lead's)

Six cases: the four with a usable reference (`ramp-outward`, `three-region`,
`c1.5-thick`, `c2-thick`) and the two the lead named for the reference check
(`centred-core`, `offset-sphere`, each with `--fine-reference 1.75`, whose
reference needed tightening — see 5.8). Same shape as the rotation runs at
`8a77aa1`, into a fresh directory:

```
cd ~/mse-bench
mkdir -p matrix-2f

(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case ramp-outward --baseline product --rotations 5 --backend gpu \
   --json matrix-2f/ramp-outward.json) > matrix-2f/ramp-outward.log 2>&1

(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case three-region --baseline product --rotations 5 --backend gpu \
   --json matrix-2f/three-region.json) > matrix-2f/three-region.log 2>&1

(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case c1.5-thick --baseline product --rotations 5 --backend gpu \
   --json matrix-2f/c1.5-thick.json) > matrix-2f/c1.5-thick.log 2>&1

(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case c2-thick --baseline product --rotations 5 --backend gpu \
   --json matrix-2f/c2-thick.json) > matrix-2f/c2-thick.log 2>&1

(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case centred-core --baseline product --rotations 5 --fine-reference 1.75 \
   --backend gpu --json matrix-2f/centred-core.json) > matrix-2f/centred-core.log 2>&1

(ulimit -v 5000000; nice -n 10 timeout 30m python run_grading_matrix.py \
   --case offset-sphere --baseline product --rotations 5 --fine-reference 1.75 \
   --backend gpu --json matrix-2f/offset-sphere.json) > matrix-2f/offset-sphere.log 2>&1
```

Six cases, one at a time (tier 2 is one solve at a time). The first four are the
`--rotations`-only runs; the last two add `--fine-reference 1.75` because their
reference is the finest uniform rung and needed tightening (5.8). Every label
above is a real `CASES` label — `grep -oE '"(ramp-outward|three-region|c1\.5-thick|c2-thick|centred-core|offset-sphere)"' benchmarks/run_grading_matrix.py`
returns each exactly once — so a `--case` typo fails loudly rather than silently
running nothing.

**Cost, and the 30-minute cap.** `--rotations 5` is now **10 builds and 10
solves** per case (5 uniform + 5 graded, one identity each side reused): the
graded half is the expensive one, so a case that already sat near the timeout
may now exceed it. If it does, cut to `--rotations 3` first (still 4 members per
side, enough for a spread) rather than dropping a case; a case with no band at
all goes back to the strict `None` reading, which is the outcome 5.11-3 exists to
avoid.

**Predictions, and their falsifiers.** (i) The shared bound should *hold* on
three-region where the one-sided bound flips — because both sides' signed errors
there are on the same side of zero, the shift cancels and the margin is the raw
0.053% gap; **falsifier:** the shared reading flips on a case whose two signed
errors share a sign, which would mean the `d`-cancellation is not implemented
right. (ii) c2-thick's graded band should be **larger** than its uniform band
(0.078%) — its graded number is the one with a sign-changing multiplet, and that
is where orientation luck would show; **falsifier:** graded band ≤ uniform band
on c2-thick would say the "sign change" is not a zero-crossing artifact.
(iii) c1.5-thick's rise (0.021%) should stay above its band (0.017%) — it should
report UNRELIABLE with the band named; **falsifier:** a measured band ≥ 0.021%
excuses the rise and the case becomes decidable.

#### Where this stopped

All three items are implemented and tier-1 green (bench only; the product rule,
the mesher and `criterion` are untouched). **Nothing here is measured on real
meshes from this seat** — meshing and solves are tier 2 and the lead's, so every
number cited above is either the lead's own message or a tier-1 fixture, and the
predictions are predictions. No `--rotations` run was made here. One commit
reports in the step-2f message to the lead with the local sha. Step 3 (the
product rule change) remains **NO GO**.
