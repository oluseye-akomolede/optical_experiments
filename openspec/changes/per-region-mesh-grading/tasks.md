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

      (b) BOTH NUMBERS ARE MEDIANS OF MIXED POPULATIONS, and the mixtures are
      made by the SIZING BACKGROUND's ramp between the two region edges. Both
      bands are measured on the wall side, where the field is ramping from
      h_min to h_wall, so no band's median can equal a region target:

        * shell 0.9014 mm = 0.504 x h_wall. The band is an ABSOLUTE
          `SHELL_BAND_MM = 1.0` wide, `r` in `(a_eff, a_eff + 1.0]`. On the
          analogue (b = 6 mm, edge 0.9 mm, n_core = 3, same relative recipe,
          24,887 tets) the bench's OWN band helper gives shell 0.4536 mm =
          0.504 x h_wall -- the same ratio to three digits at 2.7x smaller
          scale, so it is the ramp's shape and not the Orin's fixture. Its tets
          are 14% within 15% of h_min, 0% within 15% of h_wall, 85% in
          between: the band IS the ramp, and its median sits inside a
          0.25-0.70 mm continuum no rule asked for.
        * wall 1.4530 mm = 0.812 x h_wall. `wall` in that helper is `~core` --
          the WHOLE outer region -- so it mixes the ramp with the interior too.
          Analogue, same mask: 0.7685 mm = 0.854 x h_wall (Orin 0.812). The
          wall's own BULK (r > a_eff + 1 mm) is 0.878 mm = 0.975 x h_wall, with
          53% of its tets within 15% of h_wall. So the background DOES deliver
          the wall's asked edge where the band is not contaminated by the
          interface; the 0.81-0.85 figure is the band's placement, not the
          rule's output.

      (c) THE OWNER'S QUESTION, answered. The grading RULE is not fitted to one
      job: `region_plan` reads only the object's own geometry and indices, and
      (a) shows no per-job constant in play at n3. What IS shaped by the job's
      size is the BENCH'S BAND METRIC -- `SHELL_BAND_MM = 1.0` absolute mm is
      10% of this b = 10 mm and would be 50% of a 2 mm upload, so the shell
      number would move with the upload even if the rule did not. A per-region
      report, or a band width relative to a_eff, does not have that failure
      mode. Measuring it across shapes is step 2's matrix, not more arguing.

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
