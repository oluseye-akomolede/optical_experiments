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
