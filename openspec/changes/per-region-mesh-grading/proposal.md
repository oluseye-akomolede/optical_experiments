# Per-region mesh grading

## Why

A multi-material upload is meshed UNIFORMLY at its finest region's edge. A core
denser than the wall therefore sets the resolution for the wall too, and the
element count is the cavity's whole volume divided by that one fine edge.

The mechanism to do better was believed unavailable. `region_edges` records why:
TetGen's per-region volume, supplied through `add_region(max_vol=...)`, was
measured INERT on a nested PLC -- a nine-order sweep moved the core's own
element count by exactly zero. That measurement stands. What it does not cover
is TetGen's OTHER sizing mechanism: the mesh-sizing FUNCTION (`-m`), a
background mesh whose nodes carry a target edge, which the per-region volume is
not.

Measured on the nested-cube fixture (`benchmarks/mesh_grading_feasibility.py`,
2026-10-12, tetgen 0.8.4, canonical frame, `order=1 mindihedral=20 minratio=2.0`):

| mesh | tets | core median edge |
|---|---|---|
| uniform at `h_core` | 39,454 | 0.0433 |
| graded by the sizing field | 2,813 | 0.0432 |

93% fewer elements at the same core resolution. Both controls pass and both can
fail (reduction < 75%, core edge off its MEASURED uniform target by > 25%).

## What changes

1. **`region_edges` grades.** It stops returning one `h_min` for every region
   and returns `h_r = min(h, h * n_wall / n_r)` per region -- the rule its own
   docstring says is sound but unrealisable. `min` over those values is the
   `h_min` of today, so the finest region's target, and the resolution the
   cavity is solved at, do not move.
2. **The mesher realises the field.** When the plan's edges are not all equal,
   `tetrahedralize` builds a background mesh carrying each region's target and
   passes it as `bgmesh=` (pyvista is already pinned in the solver image); the
   wrapper appends `-m` itself. No `fixedvolume`, no global `maxvolume`: either
   overrides the field and silently meshes uniformly.
3. **The estimate follows the mesh.** `estimate_tets_by_region` becomes
   `c * sum_r V_r / h_r^3`, algebraically the SAME expression as today's
   `c * V_cavity / h_min^3` when the edges are equal, so the uniform case and
   the one-part path keep their numbers exactly. The estimate is what ROUTES the
   job; an estimate that kept counting a uniform mesh would route a graded job
   to the cluster for elements it no longer has.
4. **A coarse target is clamped under its region's thickness.** The field stops
   being honoured when a target approaches the thickness of the region it
   describes: measured on three nested-cube fixtures, the mesh collapses (the
   fine region loses its resolution too) at 1.9-3x the shell's thickness.
   Background-mesh resolution was tested as the cause and REFUTED (0.22 vs
   0.08/0.05/0.04: identical 128-element collapse). The clamp is
   `h_r <= 2 * V_filled / A_boundary` (the region's own thickness proxy, its
   surface plus every child's), it never applies to the finest region, and when
   it bites the job's caveats say so.
5. **The resolution check is made where it means something.** A graded mesh's
   whole-mesh median edge is dominated by the coarse regions; the check is the
   FINEST region's achieved median edge, against the same overshoot allowance
   `MAX_EDGE_OVERSHOOT` that the uniform rule uses today.
6. **The surface is decimated to the coarse edge** for a graded mesh. The
   surface floors element size, so decimating to the finest edge would floor the
   coarse regions and defeat the grading.
7. **The artifact and the page carry it.** The `mesh` block in `modes.json`
   gains `graded`, the per-region targets/achieved edges/volumes, and the
   caveats; the job page shows a graded job's per-region resolution and any
   caveat the mesher recorded.
8. **The acceptance bar is the uniform mesh** (owner's choice, 2026-10-13). The
   Orin at `ebfd570` measured n2 graded worst error 0.341% against the SAME
   object solved uniformly at 0.559% -- a 1.64x improvement -- and n3 graded
   0.749% with its uniform reference UNMEASURED (cuDSS `ALLOC_FAILED` at
   199,062 tets). At `11357f4` the same Orin command DID measure n3's uniform
   reference: n3 graded 0.749% vs uniform 0.662% (1.13x) -- a FAIL under this
   bar, and the run exited 3 -- while n2 was unchanged (0.341% vs 0.559%,
   0.61x, PASS). Median edges at n3: graded core 0.664 / shell 0.9014 / wall
   1.453 mm against uniform core 0.7085 / shell 0.7253 / wall 0.5287 mm. The
   benchmark had been judging the graded mesh against twice
   the EMPTY SPHERE's error, a bar nobody buys; it now judges it against the
   uniform mesh the owner would have got, and a bar that could not be measured
   fails the run rather than passing it.

## Impact

- `dockerfiles/maxwell_solver/mse_common.py` -- `region_edges` (the rule),
  `estimate_tets_by_region` (the sum), a boundary-area helper for the clamp.
- `dockerfiles/maxwell_solver/mse_meshing.py` -- `_region_plan` (carry the
  edges), a background-mesh builder, the graded call and its guards, the
  fine-region check, the decimation target, the mesh report.
- `web/app.py`, `templates/job.html` -- the mesh block's caveats.
- `openspec/specs/maxwell-solver/spec.md`, `openspec/specs/mesh-upload/spec.md`.
- Tests: the graded path's controls (both directions), the guard that no
  `fixedvolume`/global `maxvolume` accompanies the field, the clamp, the
  estimate-equals-the-uniform-number when the edges are equal, and the
  byte-identical one-part/equal-index control.

Single-material jobs are untouched by construction: the one-part path never
calls `region_edges`, and a multi-part upload whose regions share one index
produces equal edges and takes the fixed-volume path unchanged.
