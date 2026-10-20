# A metal face's skin layer stops leaving a sliver in the middle

## Why

Run C's gold mirror is 150 nm of gold, run with the Au-palik model. At that
run's band top -- 900 nm centre, `--band-half 80`, pulse, so
`frequency_hz + bandwidth` = 362.95 THz -- Au-palik's skin depth is 24.466 nm,
so the layout's rule adds a quarter-skin-depth layer (6.116 nm cells) 3 skin
depths deep inside each face. Six skin depths are 146.79 nm, and the two face
layers leave `150.000 - 146.794 = 3.21 nm` of bulk metal between them.
`graded_axis` splits any gap under the coarse cell into `MIN_CELLS` (4) cells,
so that 3.21 nm middle became four cells of 0.801 nm -- 7.6x under the 6.116 nm
skin spacing -- and the timestep follows the finest cell in the grid. A slab
that should be a handful of skin-depth cells is cut into cells under a seventh
of its neighbour's spacing.

The consequence is a step count, not a wrong answer: on Run C's own setup
(`--dx 10 --mirror-pad 600 --pad-nm 320 --pml-nm 200 --band-half 80
--gold-data palik`) the grid needs **378,896 steps** and 15,312,024 cells,
against **121,121** steps and 14,400,594 cells once the strip is gone -- 3.1x
the record for the same physics. Both revisions are over the product's own
ceilings (8,000,000 cells, 20,000 steps, 1,800 s; the web form and the worker
call `fdtd.check_limits` bare); the bench Job raises them on its command line
to 15,000,000 cells, 200,000 steps and 18,000 s, and only the fixed grid fits
inside those -- the unfixed one is over the step ceiling by 1.9x and over the
cell ceiling by 312,024. The same rule bites any metal part whose thickness is
just over 6 skin depths, which is a shape a user will upload: the mirror's own
thickness was chosen to be 6 skin depths, and under the Rakić model at the same
band top (depth 27.928 nm) this slab is 5.37 depths, so the layers would meet
and it was never cut.

**Why a skin depth, not just a coarse cell.** A branch at `middle < dx` alone
leaves a band open. `graded_axis` fills the gap between two fine layers with a
few cells -- four at a quarter of the gap, two and three for two middles in
between -- measured with the layout's own two face layers at dx 10 nm (Rakić,
4e14 Hz, depth 28.981 nm, spacing 7.245 nm): middles of 6.1, 11.1, 16.1 and
26.1 nm came out in cells of 1.528, 5.556, 5.352 and 6.528 nm -- 0.21x to 0.90x
of the spacing. So every middle under one skin depth is finer than the skin
spacing, and one skin depth is where the split meets it (a 28.1 nm middle's
finest cell is 7.03 nm). Above that it is no longer harmful -- with room
around the slab, middles of 31 to 726 nm came out at 0.997x to 1.000x of the
spacing -- so the band the two-layer branch harms ends where this threshold
does. Hence `middle < max(dx_m, depth)`: one layer across
a part in this band takes under `24 + 4*max(dx_m, depth)/depth` cells (28 where
a coarse cell is no coarser than the skin depth) at `depth/4`, never finer than
the cells it replaces. Run C's mirror is in the band under either rule (middle
3.21 nm).

## What Changes

- `fdtd_layout.layout`'s per-axis metal rule adds the two face layers only when
  what they leave between them is at least the wider of one coarse cell and one
  skin depth. Otherwise the part gets **one layer across its whole extent** at
  a quarter of the skin depth -- no strip to subdivide, and the whole part is
  quantised once (Run C's mirror: four cells of 0.801 nm at `df18afc`, none
  finer than 3.000 nm now). The
  existing "thinner than 6 skin depths" case is that rule's limit, so one
  condition replaces the special case.
- Both faces stay grid nodes: the layer spans the part's bounding-box extent
  and is laid on that spacing, so a cell boundary falls on each face.
- `tests/test_fdtd_layout.py`: a unit test on a 180 nm gold slab at dx 50 nm
  (just over 6 skin depths, leaving a 6.1 nm strip) asserting that no cell
  across the slab is under half a quarter-skin-depth and that the slab takes
  no more cells than one uniform layer has. Fails on 2e7ae0e: 36 cells across
  the slab, the smallest 1.53 nm (0.21x the cap).
- No spec'd behaviour is removed: a part thinner than 6 skin depths already got
  one layer across it, and a part thicker than 6 skin depths plus a coarse cell
  still gets two face layers with bulk cells between them.
