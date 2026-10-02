## 1. Solver core (phase 1)

- [x] 1.1 `dockerfiles/maxwell_solver/fdtd.py`: a Yee update on an array module `xp` (numpy | cupy),
  float32, on a **graded tensor-product grid**: per-axis primal and dual spacing
  arrays (a uniform grid is just the special case). PEC is applied by zeroing
  tangential E on the conductor mask. A CPML of 10 cells is used on open faces.
  The time step is dt = 0.99 / (c·√(1/dx_min² + 1/dy_min² + 1/dz_min²)).
  The E update carries an **auxiliary polarisation-current hook** (a list of
  per-material ADE terms, empty in phase 1), so dispersive metals add terms
  without changing the core.
- [x] 1.2 Point source: a soft E-component current, with `cw` (smooth ramp) and
  `pulse` (Gaussian) profiles.
- [x] 1.3 Probes return float32 traces. The spectrum is computed by rfft of a
  windowed trace.
- [x] 1.4 Tiny tests (≤ 24³ cells, numpy) cover the update's energy conservation
  in a closed PEC box, and that the source profile sits at the requested
  frequency.
- [x] 1.5 Benchmark V1 (metal box pulse versus the analytic frequencies, and
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
  table check against a published Rakić n, k dataset at 12 wavelengths per metal.
  Re-measure the Jetson rate and GPU memory with an open boundary and one gold
  region (lead runs it). [Code and tier-1 tests done; the Jetson re-measure is
  the lead's. The pulse-band refusal is `fdtd.check_metal_band`; the job form
  must call it at queue time.]
- [x] 2.4 Benchmark V4 (gold slab against analytic Fresnel, same model), on
  periodic transverse faces (6.1a), since a finite plate inside the TF/SF box
  diffracts at its edges and cannot meet the 0.02 gate. [Script
  `scripts/fdtd_bench_v4.py` and its tier-1 tests done; the Jetson run is the
  lead's.]
- [x] 2.5 `drawn` profile: ≤ 10 000 samples plus a duration, resampled to dt,
  then mean removed; refused when > 1% of its spectral energy is above f_res.
  [Solver side: `Source(profile="drawn")`, `fdtd.resolvable_frequency`,
  `fdtd.drawn_energy_above`; `simulate` refuses on the dielectric index as a
  backstop. The queue-time refusal (with the metals' |n|) is the form's, 5.1.]

## 3. Geometry, estimate and caps (phase 3)

- [x] 3.0 Graded-grid builder (≥ 4 cells per thin layer or skin depth, neighbour
  ratio ≤ 1.3), sub-cell averaging at cut cells, and coatings (material +
  thickness on a region's surface). A unit check of the builder's own rules.
  Benchmark V5 (parallel-plate lateral modes): script and its tests; the Jetson
  run is the lead's.
- [x] 3.1 Voxelise the upload's regions to an index map. Cells outside the shape
  are metal (`metal`) or vacuum plus a gap (`open`).
- [x] 3.2 Estimate cells, steps, memory and run time from the form. Refuse above
  8 M cells, 20 000 steps, 3.0 GB or 30 minutes, naming the limit, the value and
  the lever to change.
- 3.3 moved to 5.1a: there is no time-domain route until phase 5, so the
  route-level refusal test lands with the route.

## 4. Job kind `td` (phase 4)

- [x] 4.1 DB kind `td` with a `tds` request JSON whose keys are exactly the ones
  the worker reads. A missing key is refused by name.
- [x] 4.2 Worker routing to the Jetson. Check every path that can change the
  routing field (unclaimed-GPU fallback, abandoned-GPU reroute, reaper and
  release), each tested with a `td` job. A `td` job runs alone on the GPU;
  check free memory first and leave the job queued ("Jetson busy")
  when it is short, for at most 60 minutes; never reroute a `td` job to the
  cluster; fail readably on a device OOM mid-run and after 45 minutes of run
  time.
- [x] 4.3 Result files: `trace.bin`, `spectrum.json` and `frames/<n>.bin`. List
  every reader of each output, and give each its route, its 404 rule and a test.

## 5. Web (phase 5)

- [x] 5.1 Form: boundary, source kind, profile, f, bandwidth, position,
  polarisation, duration, probes and dx.
- [x] 5.1a (was 3.3) Every refusal raised by the form, the geometry or the
  limits comes back as the route's refusal response, never a 500. Counting
  raise sites is not enough (an implicit TypeError from bad input has none):
  the route test feeds each malformed input end to end, namely a null material,
  a list material, an empty mesh, an open/non-manifold surface, a zero-size
  part, NaN coordinates, a zero length scale, a one-node axis, a NaN
  duration, a mesh-library failure, a point source or probe on a conductor
  node, one outside the grid, one inside the CPML, and each limit broken.
- [x] 5.4a Per-part materials (lead, 2026-10-02 08:15Z): a multi-part td upload
  gets one picker per part from `/parts` (index, `pec` or a metal), posting
  `part_index.<name>`; missing, unknown or malformed materials are refused
  naming the part. Until this lands, multi-part td is refused in plain words
  (e9f0cda).
- [x] 5.4c Shared faces and overlap (lead): a shared face goes to the conductor,
  else the higher index; overlapping parts are refused naming both; `/parts`
  uses the td split for td uploads.
- [x] 5.4d Faces on the grid (lead): every flat face that bounds a part's box
  is a grid line, and each node's samples are centred on it, so a film flush
  between conductor plates keeps its drawn thickness wherever the parts sit
  (before: up to 1.5 times it, on most offsets). The form estimate bounds each
  metal by the nodes whose dual cell overlaps its parts, which keeps it above
  the solved run when a node sits on a face.
- [x] 5.4b Per-part coatings end to end: a material plus a thickness on a
  part's surface, painted by the voxeliser on a grid refined by
  `coating_layers`, offered on the page only once it runs. The shell lies
  outside the part (`fdtd_voxel.CoatedIndex`, exact point-to-triangle
  distance): it grows into open space or an enclosing host, never into a
  neighbour or a metal boundary's wall (a coating on a part that encloses the
  cavity is refused). Request key `coatings`; a coating picker per part on the
  page, one-part uploads included; the estimate, the TF/SF box and the
  conductor-point refusal all see the shells.
- [x] 5.4e Follow-up (lead, 2026-10-02 08:52Z; not in phase 5's critical path):
  an upload-time overlap check that fails fast. Probe only the bounding-box
  intersection of each non-nested pair (a 16^3 lattice, ~0.4 s per pair as the
  reviewer measured; flush neighbours have zero-thickness intersections and
  cost nothing). It is sampled and can miss a sliver, so the worker's
  voxeliser refusal stays authoritative. Done as
  `fdtd_voxel.refuse_overlap`, called by `td_form.plan_request`. Measured
  (VM, best of 3): a 3-part flush stack 0.3 ms; overlapping boxes refused in
  20 ms. The cost was the exact nesting test (25 s for two 1280-face spheres
  side by side), now asked only where one bounding box fits in the other:
  42 ms to refuse two overlapping 1280-face spheres.
- [x] 5.2 Viewer: field-movie playback in the existing 3-D view, plus trace and
  spectrum plots with axis units.
- [ ] 5.3 Image smoke test, then an in-browser check against a real finished `td`
  job in production.

## 6. Periodic boundaries (phase 6)

- [x] 6.1a Periodic faces at k = 0 (wrap the curl differences), a boundary per
  axis in the solver API, and a plane wave across periodic transverse faces.
  Built in phase 2 for V4 (lead, 22:15Z). The form option stays in phase 6.
- [ ] 6.1b A Bloch phase for a given k: cos and sin fields stepped together
  through a core face hook.
- [ ] 6.2 Benchmark V6 against the existing pc band solver: an extruded 2-D
  crystal, TM and TE, kz = 0, frequencies at chosen k-points within 1%, every
  band present.
