# A probe's legend states its position in the grid's own unit

## Why

Prod td job `aa6f083b` (cube.obj, units nm, scale 50, `td_dx` 20 nm, one probe Ez
at cell [16,16,16]) shows the legend "probe 1: Ez at (0.0000400, -0.000320,
0.000100) fm". The manifest stores `position_m = [4.0e-20, -3.2e-19, 1.0e-19] m`,
which `worker/td_solve.py:573` writes as `origin_m[a] + nodes[a][cell[a]]` -- the
float64 sum of the frame origin and a node coordinate. Three facts in that one
manifest say which:

1. `frames.positions_m[a][16]` equals `probes[0].position_m[a]` on all three axes,
   and 16 x 20 nm = 320 nm = |lo|: the probe's node **is** the object frame's own
   coordinate origin, and the stored value is what is left of `-320 nm + 320 nm`.
2. `positions_m[a][16]` is a node in a list whose neighbours are 20 nm apart, so
   the grid is the nanometre grid the request asked for.
3. The run's own `dt_s` is the 3-D Courant step for `dx` = 20 nm (0.99 x 20 nm /
   (c*sqrt(3))), so no unit was applied twice: a doubled unit would be ~1e9 out,
   not 1e-19.

So it is (a): cancellation residue of a probe at the origin, promoted to
femtometres by a prefix chosen from the residue's own magnitude -- twelve orders
finer than the job's cell. The residue is ~1e-10 of a cell (the grid rounds its
edges to 12 significant digits, `fdtd_grid.py:146`), so there is room to snap.

## What changes

- `web/static/tdplot.js`: `probeLabel(probe, index, manifest)` reads the grid's
  own step -- the finest gap between the sample positions the manifest records --
  snaps a coordinate below a millionth of that step to 0, floors the chosen
  prefix at the step (so the unit is never finer than the job's own cell), and
  prints an exactly-zero coordinate as `0`. A manifest with no sample positions
  (no movie) keeps the previous rule.
- `web/static/viewer.js`: passes the manifest, which is the only place the page
  knows the job's own grid.
- `dockerfiles/rendercheck/tdcheck.mjs`: reads the legend's position back
  through the page's own prefix table and requires it to be the manifest's own
  probe node, within one grid step; and requires an origin coordinate to read
  `0`. The check passed the femtometre legend because it verified the component
  and the colour, not the position.
- `tests/js/test_tdplot.mjs`: four tests, red on the old `probeLabel`, including
  the production job's own three floats.

## Impact

- Page code, the render check and its tests. **No worker change, no manifest
  change, no deploy**: every td job's manifest already records
  `frames.positions_m` (the worker always passes the movie plan to
  `fdtd.simulate`), so the grid step is available where the legend is drawn and
  nothing needs re-solving.
- Jobs solved before the positions were recorded have no step on their manifest;
  for those the legend falls back to the old rule rather than inventing a grid.
