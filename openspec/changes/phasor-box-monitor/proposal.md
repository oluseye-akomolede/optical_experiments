# phasor-box-monitor

## Why

`farfield.py` (change `near-field-far-field`) turns a closed box of
tangential E and H phasors into a far-field pattern, a collection fraction and
a power, but nothing in `fdtd.simulate` produces that surface: its tests feed
analytic dipoles through `farfield.sample_box`. Paper 1's 74%-within-NA-0.8
number needs the solver's own field on that box. This is task 5 of
`near-field-far-field`, the monitor half only; the paper-1 run is separate.

## What Changes

- `fdtd.simulate` gains `phasor_box=(centre, half_extent)` (cells, the
  `flux_box` spelling) with `phasor_frequencies` (one or more, Hz),
  `phasor_start` (first step read, default 0) and `phasor_window` (`"hann"`
  default, or `None`). It adds `"phasor_faces"` to the result: per frequency,
  six `farfield.Face` in `DIRECTIONS` order, in the layout the far-field
  module docstring fixes. `None` when not asked for.
- A running DFT `2 sum(w F e^{+i w t}) / sum(w)`, E stamped at `(n+1) dt` and
  H at `(n+1/2) dt`, accumulated in complex128 on the device from a
  precomputed phase table (no per-step sync), transferred once at the end,
  then interpolated from the staggered Yee positions to the face-cell
  centres.
- `fdtd.phasor_box_layers` checks the box: the flux box's rules (live layers
  1..n-2, outside the CPML, refused on a paired Bloch run), with one more cell
  of clearance from the CPML, because a face reads H half a cell outside
  itself.

Requirements added to `time-domain`: the phasor box records the far-field
surface; its phasors are a windowed DFT at each family's own sample time;
without it a run is unchanged; and where it is refused. No existing
requirement changes: without the option every record is bit-identical, and
the new key is `None`.

## Impact

- `dockerfiles/maxwell_solver/fdtd.py` (new option, `_PhasorBox`,
  `phasor_box_layers`); `tests/test_fdtd_phasor_box.py` (9 tests, about 1.5 s
  at `OMP_NUM_THREADS=2`); 9 reversion cases in `tests/check_discriminating.py`.
- No route, request field, schema, image, manifest or worker change: no job
  can ask for the option yet, so `memory_estimate` does not count it.
