# A metal face's skin layer stops leaving a sliver in the middle

## Why

Run C's gold mirror is 150 nm of gold. At 4e14 Hz (80 nm pulse band) gold's
skin depth is 28.95 nm, so the layout's rule adds a quarter-skin-depth layer
(7.25 nm cells) 3 skin depths deep inside each face. The two face layers cover
86.9 nm each and leave a 2.4 nm strip of bulk metal between them. The
thin-layer rule then sees a layer 2.4 nm wide with a spacing a quarter of that
(0.80 nm) applies over it, and `build_grid` takes the smaller spacing where the
two layers overlap -- which is the strip's whole extent. A slab that should be
a handful of skin-depth cells is cut into four 0.6 nm cells in its middle.

The consequence is a step count, not a wrong answer: on Run C's own setup
(`--dx 10 --mirror-pad 600 --pad-nm 320 --pml-nm 200 --band-half 80
--gold-data palik`) the grid needs **378,896 steps** and 15,312,024 cells,
against **121,121** steps and 14,400,594 cells once the strip is gone -- 3.1x
the record for the same physics, and over the app's 200,000-step cap. The same
rule bites any metal part whose thickness is just over 6 skin depths, which is a
shape a user will upload: the mirror's own thickness was chosen to be 6 skin
depths, and under the Rakić model the same slab lands just under 6 and was
never cut.

## What Changes

- `fdtd_layout.layout`'s per-axis metal rule adds the two face layers only when
  what they leave between them is at least one coarse cell. Otherwise the part
  gets **one layer across its whole extent** at a quarter of the skin depth --
  the same 24 cells the two face layers take (150 / 7.245), with no strip to
  subdivide. The existing "thinner than 6 skin depths" case is that rule's
  limit, so one condition replaces the special case.
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
