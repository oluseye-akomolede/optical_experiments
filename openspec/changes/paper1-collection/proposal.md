# paper1-collection

## Why

Paper 1 reports that 74% of the nanowire's emission is collected within a
numerical aperture of 0.8. The pieces to measure that exist: `fdtd.simulate`
records a phasor box (change `phasor-box-monitor`) and `farfield.py` turns its
faces into a far field and a collection fraction (change
`near-field-far-field`). `scripts/paper1_nanowire.py` runs the paper's
geometry but has no arm that reads it. This is task 5 of `phasor-box-monitor`
and task 5b of `near-field-far-field`, and it answers the paper-1 line "NA 0.8
collection waits for the far-field cloud job ... it is not an arm of this
script today".

The open question is how the box meets the mirror. The paper-1 mirror is a
Rakić Lorentz-Drude gold pad of finite width (`--mirror-pad` beyond the wire),
150 nm thick, with the layout's open vacuum pad on every side of it, below
included. It is not a perfect conductor and not an infinite plane, so
`farfield`'s `pec_mirror_z` image is a model of it, not the scene.

## The mirror: options compared, measured

`tests/test_phasor_mirror.py` measures four ways a box can meet a mirror, on
tiny grids, with an x dipole (the paper's polarisation):

- **A, full enclosure**: six faces around the emitter AND the whole mirror,
  every face in vacuum, free-space transform. The equivalence theorem with
  every source of the field inside and vacuum outside: exact for the scene
  the grid holds. Possible only where the mirror is finite with vacuum all
  round it, which is the paper-1 layout.
- **B, PEC image**: five faces, the bottom on the mirror's top surface, each
  face imaged by `pec_mirror_z`. Exact for an infinite perfect conductor.
  Gets wrong: gold is not a perfect conductor (eps -27.6 + 2.45i at 900 nm),
  and the image is an infinite mirror where the scene's pad ends.
- **C, closed box in air**: six faces, the bottom in the air (or spacer)
  between the mirror and the emitter, free space. Gets wrong: the mirror's
  induced currents are outside the box, so the transform drops the
  reflection and reads the emitter as if no mirror were there.
- **D, open bottom**: B's five faces without the image. Gets wrong: only the
  patch of mirror under the box reflects (`farfield.py`'s own docstring
  measures this on analytic fields).

Over a perfect conductor (37 x 37 x 31 nodes, dx 10 nm, 16 cells per
wavelength, f = 1.874 PHz, the dipole lambda/4 above an exact Yee PEC plane
that runs through the absorber), against the closed-form image dipole:

| option | NA 0.8 fraction | pattern off | upward / delivered |
|---|---|---|---|
| closed form | 0.7661 | | 1 |
| B, PEC image | 0.7666 | 0.0062 | 0.993 |
| C, box in air | 0.4996 (the free dipole's 0.4960) | 0.97 | 0.43 |
| C with the image | 0.7660 | 0.0017 | |
| D, open bottom | | 0.32 | 0.951 |

Over a 360 x 360 x 60 nm Rakić gold slab at 900 nm (37 x 37 x 35 nodes, dx
30 nm, CPML 6, the dipole 210 nm above it), where there is no closed form:

| option | NA 0.8 fraction | against A |
|---|---|---|
| A, full enclosure | 0.6121 | |
| A2, every face 2 layers further out | 0.6121 | 0.0000 |
| B, PEC image | 0.7556 | +0.143 (pattern 0.32 off) |
| C, box in air | 0.4964 | -0.116 (upward power 0.48 of A's) |

A closes on itself (full-sphere far field over surface power 0.9999 and
1.0006) and does not move with its box, which is the test that it leaves no
source outside. On an 840 nm slab (53 x 53 x 35 nodes, not in the tests) A
reads 0.7172 and B 0.7346: B's error falls with the pad's width (+0.143 at
0.4 wavelength, +0.017 at 0.93) but stays an error of a mirror model, and its
sign and size on the paper's 1.68 um pad are not known without running A
anyway.

**Choice: A.** The arm's box encloses the wire, the oxide and the whole gold
pad, with every face in the layout's vacuum pad. It reports the scene as
built, with no mirror approximation. What it does not remove is the scene's
own one: the pad is finite (`--mirror-pad`), so if the paper's mirror is
effectively infinite, the pad width is a convergence parameter of the
collection fraction as it already is of F (paper-1 tasks, the 200/400/600
pad run). The fraction is normalised as `farfield.collection_fraction` does,
over the upper hemisphere; the arm also prints the cone over all the radiated
power, since part of a finite pad's emission leaves downward past its edges.

## What Changes

- `scripts/paper1_nanowire.py`: `--collection-na NA` (default off). The CW arm
  then places a phasor box around the whole scene (`collection_box`: in the
  layout's own graded nodes, each face at least 2 layers outside the parts,
  inside `pml + 1 .. n - 2 - pml`, in the middle of its free gap, with the
  faces' slabs checked to be vacuum on the voxelised layout), reads it over
  the CW reading's own tail (`window_periods`, Hann), and prints the NA
  fraction, the upward and total power, and two sanity ratios: the box's
  surface power over the delivered power (below 1 by what the gold absorbs)
  and the full-sphere far field over the surface power (1 for a box in
  vacuum). A `box` line before the run names the layers. Refused with an NA
  outside (0, 1], on a pulse record, with `--tiny` (except `--tiny
  --estimate`), and when a layout has no room for the box. Without the flag
  every call, print and number is what it was.
- `dockerfiles/maxwell_solver/fdtd.py`: `phasor_box_bytes(box, shape, *,
  frequencies, steps)` and `phasor_box` / `phasor_frequencies` /
  `phasor_steps` keywords on `memory_estimate` and `estimate_run`, so a run
  with the box is priced with it. It counts `_PhasorBox`'s own 36
  accumulators and two phase tables (equal byte for byte on a real box) plus
  the larger of a step's transients and the faces' assembly after the run,
  which on numpy is the box's real peak and on the Jetson's unified memory is
  the same RAM. The script's run path (`filled`) and ladder both add it.
  Without the keywords the estimate is unchanged. No phasor code changed: no
  bug in it was found.
- `farfield.py` is not changed.

## Impact

- `scripts/paper1_nanowire.py`, `dockerfiles/maxwell_solver/fdtd.py`
  (`memory_estimate`, `estimate_run`, new `phasor_box_bytes`).
- Tests: `tests/test_phasor_mirror.py` (new, 4 tests, about 14 s),
  `tests/test_paper1_collection.py` (new, 12 tests, about 18 s),
  `tests/test_fdtd_memory.py` (+4), 16 cases in
  `tests/check_discriminating.py`.
- No route, request field, schema, image, manifest or worker change: the
  option is reached only from the bench script.
- Not done here: the full-size run (it is the GPU's); the box's per-step
  folding is not in the seconds estimate (it runs over the tail only).
