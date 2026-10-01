## 1. Solver core (phase 1)

- [ ] 1.1 `dockerfiles/maxwell_solver/fdtd.py`: a Yee update on an array module `xp` (numpy | cupy),
  float32. PEC is applied by zeroing tangential E outside or at the conductor
  mask. A CPML of 10 cells is used on open faces. The time step is
  dt = 0.99·dx / (c·√3).
- [ ] 1.2 Point source: a soft E-component current, with `cw` (smooth ramp) and
  `pulse` (Gaussian) profiles.
- [ ] 1.3 Probes return float32 traces. The spectrum is computed by rfft of a
  windowed trace.
- [ ] 1.4 Tiny tests (≤ 24³ cells, numpy) cover the update's energy conservation
  in a closed PEC box, and that the source profile sits at the requested
  frequency.
- [ ] 1.5 Benchmark V1 (metal box pulse versus the analytic frequencies, and
  versus the cavity solver) and V2 (CW 1/r). These run on the Jetson bench
  (tier 2), and the lead runs them.

## 2. Plane-wave source (phase 2)

- [ ] 2.1 A TF/SF box with a 1-D auxiliary incident-field grid along the axis.
  It is refused with `metal`.
- [ ] 2.2 Benchmark V3.

## 3. Geometry, estimate and caps (phase 3)

- [ ] 3.1 Voxelise the upload's regions to an index map. Cells outside the shape
  are metal (`metal`) or vacuum plus a gap (`open`).
- [ ] 3.2 Estimate cells, steps, memory and run time from the form. Refuse above
  8 M cells or 20 000 steps, naming the limit and the lever to change.
- [ ] 3.3 Every refusal type raised by the form or the geometry is caught by the
  route (no 500 responses).

## 4. Job kind `td` (phase 4)

- [ ] 4.1 DB kind `td` with a `tds` request JSON whose keys are exactly the ones
  the worker reads. A missing key is refused by name.
- [ ] 4.2 Worker routing to the Jetson. Check every path that can change the
  routing field (reroute, reaper and release).
- [ ] 4.3 Result files: `trace.bin`, `spectrum.json` and `frames/<n>.bin`. List
  every reader of each output, and give each its route, its 404 rule and a test.

## 5. Web (phase 5)

- [ ] 5.1 Form: boundary, source kind, profile, f, bandwidth, position,
  polarisation, duration, probes and dx.
- [ ] 5.2 Viewer: field-movie playback in the existing 3-D view, plus trace and
  spectrum plots with axis units.
- [ ] 5.3 Image smoke test, then an in-browser check against a real finished `td`
  job in production.
