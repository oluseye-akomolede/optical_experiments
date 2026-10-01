## 1. Solver core (phase 1)

- [ ] 1.1 `dockerfiles/maxwell_solver/fdtd.py`: a Yee update on an array module `xp` (numpy | cupy),
  float32, on a **graded tensor-product grid**: per-axis primal and dual spacing
  arrays (a uniform grid is just the special case). PEC is applied by zeroing
  tangential E on the conductor mask. A CPML of 10 cells is used on open faces.
  The time step is dt = 0.99 / (c·√(1/dx_min² + 1/dy_min² + 1/dz_min²)).
  The E update carries an **auxiliary polarisation-current hook** (a list of
  per-material ADE terms, empty in phase 1), so dispersive metals add terms
  without changing the core.
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

## 2. Plane-wave source, dispersive metals, drawn waveform (phase 2)

- [x] 2.1 A TF/SF box with a 1-D auxiliary incident-field grid along the axis,
  on the same dt and graded spacings; ≥ 3 cells to objects and to the CPML.
  Refused with `metal` and for a longitudinal polarisation. All profiles are
  zero-mean.
- [x] 2.2 Benchmark V3 (`scripts/fdtd_bench_v3.py`; the Jetson run is the lead's).
- [x] 2.3 Lorentz-Drude ADE terms (Rakić 1998, ε∞ = 1, parameters in eV), stored
  on the metal's bounding box only, plus a material library (Au, Ag, Al, Cu, Cr)
  with the source cited in the code, a refusal outside each fit's range, and a
  table check against a published Rakić n, k dataset at three energies per metal.
  Re-measure the Jetson rate and GPU memory with an open boundary and one gold
  region (lead runs it). [Code and tier-1 tests done; the Jetson re-measure is
  the lead's. The pulse-band refusal is `fdtd.check_metal_band`; the job form
  must call it at queue time.]
- [ ] 2.4 Benchmark V4 (gold slab against analytic Fresnel, same model).
- [ ] 2.5 `drawn` profile: ≤ 10 000 samples plus a duration, resampled to dt,
  then mean removed; refused when > 1% of its spectral energy is above f_res.

## 3. Geometry, estimate and caps (phase 3)

- [ ] 3.0 Graded-grid builder (≥ 4 cells per thin layer or skin depth, neighbour
  ratio ≤ 1.3), sub-cell averaging at cut cells, and coatings (material +
  thickness on a region's surface). A unit check of the builder's own rules.
  Benchmark V5 (parallel-plate lateral modes).
- [ ] 3.1 Voxelise the upload's regions to an index map. Cells outside the shape
  are metal (`metal`) or vacuum plus a gap (`open`).
- [ ] 3.2 Estimate cells, steps, memory and run time from the form. Refuse above
  8 M cells, 20 000 steps, 3.0 GB or 30 minutes, naming the limit, the value and
  the lever to change.
- [ ] 3.3 Every refusal type raised by the form or the geometry is caught by the
  route (no 500 responses).

## 4. Job kind `td` (phase 4)

- [ ] 4.1 DB kind `td` with a `tds` request JSON whose keys are exactly the ones
  the worker reads. A missing key is refused by name.
- [ ] 4.2 Worker routing to the Jetson. Check every path that can change the
  routing field (unclaimed-GPU fallback, abandoned-GPU reroute, reaper and
  release), each tested with a `td` job. A `td` job runs alone on the GPU;
  check free memory first and leave the job queued ("Jetson busy")
  when it is short, for at most 60 minutes; never reroute a `td` job to the
  cluster; fail readably on a device OOM mid-run and after 45 minutes of run
  time.
- [ ] 4.3 Result files: `trace.bin`, `spectrum.json` and `frames/<n>.bin`. List
  every reader of each output, and give each its route, its 404 rule and a test.

## 5. Web (phase 5)

- [ ] 5.1 Form: boundary, source kind, profile, f, bandwidth, position,
  polarisation, duration, probes and dx.
- [ ] 5.2 Viewer: field-movie playback in the existing 3-D view, plus trace and
  spectrum plots with axis units.
- [ ] 5.3 Image smoke test, then an in-browser check against a real finished `td`
  job in production.

## 6. Periodic boundaries (phase 6)

- [ ] 6.1 Periodic faces (wrap the curl differences), plus a Bloch phase for a
  given k: cos and sin fields stepped together through a core face hook.
- [ ] 6.2 Benchmark V6 against the existing pc band solver: an extruded 2-D
  crystal, TM and TE, kz = 0, frequencies at chosen k-points within 1%, every
  band present.
