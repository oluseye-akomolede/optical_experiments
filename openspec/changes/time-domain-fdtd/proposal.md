# Time-domain simulation (FDTD) with user-defined sources

## Why

The owner asked for time-series simulations with sources before the waveguide
solver (Slack, 2026-10-01 15:43Z), and chose a full time-stepping simulation on a
grid (option 1b) with a user-defined source: steady state, pulse, or plane wave
(15:48Z).

Nothing today does this. The legacy solver's "time series"
(`dockerfiles/spark_fem_and_time_series/spark-fem-solver.py:332-352`) replays
mode 0 as `cos(ωt)·φ₀`: no source, no excitation, no propagation. The cavity
and photonic-crystal solvers are eigen-solvers, so they cannot show a field
building up, a pulse travelling, or light scattering off an object.

## What changes

A new job kind, `td`, runs a 3-D finite-difference time-domain (Yee) simulation
of the uploaded object and returns a field movie, probe time traces, and their
spectra.

- **Engine: our own Yee solver on cupy (Jetson GPU), with numpy running the same
  code for tiny local tests.** It was chosen over Meep after a check of the Jetson:
  aarch64, 6 cores, 7 GB of shared RAM, cupy 14.2 already in the worker, and no
  Meep and no conda. Meep would add a conda stack and run on 6 CPU cores. The Yee
  update is about 100 lines and checkable against analytic results and against
  our own cavity solver.
- **Geometry:** the upload is voxelised onto a **graded (non-uniform, tensor-product)
  grid**. Per-axis spacings dx_i, dy_j and dz_k are fine only where the geometry
  needs it (thin coatings and films) and coarse elsewhere, with a neighbour ratio
  of at most 1.3. Each cell takes the material of the region it falls in.
  Interfaces that cut a cell use sub-cell averaging, so layers thinner than one
  cell are not lost.
- **Materials (owner, 15:54Z: "an object coated in gold or a thin film wedged
  between two conducting plates"):**
  - dielectrics by refractive index, as today;
  - perfect conductors;
  - **dispersive, lossy metals** (gold, silver, aluminium, copper and chromium),
    using published Drude-Lorentz fits (Rakić et al. 1998) and stepped by
    auxiliary differential equations.
  - A **coating** is a material plus a thickness applied to a region's surface.
  - A film between plates is three regions with three materials.
- **Boundaries (form choice):**
  - `metal`: the uploaded shape is a metal cavity, as in today's cavity tool.
    This is the default.
  - `open`: the object sits in vacuum, with a gap and a convolutional PML around it.
- **Sources (form choice; the owner's "user-defined"):**
  - **Kind:**
    - `point`: a dipole at a chosen position, defaulting to the shape's centre,
      with polarisation x, y or z.
    - `plane`: a plane wave entering along ±x, ±y or ±z, with a transverse
      polarisation, injected by a total-field/scattered-field box. It is allowed
      only with `open`.
  - **Time profile:**
    - `cw`: steady state at frequency f, with a smooth turn-on over several periods.
    - `pulse`: Gaussian, with centre frequency f and a bandwidth.
    - `drawn`: the owner draws amplitude against time in the browser, or pastes
      samples (at most 10 000, with a duration). It is resampled to dt and its
      mean is removed. Before queueing, the run is refused if more than 1% of
      the waveform's spectral energy lies above the grid's resolvable frequency
      (10 cells per wavelength in the finest material, at the largest cell).
- **Outputs:**
  - the probe time trace at the source and at up to 4 user probe points;
  - the spectrum of each trace, which shows the resonances for a pulse;
  - a field movie of |E| (≤ 64³ samples, ≤ 60 frames), which the 3-D viewer plays.
- **Limits:**
  - The cell size is λ_min / (10·|n|_max) in each material, refined further to resolve each
    thin layer (≥ 4 cells across it, or the metal skin depth where that is thinner).
  - The run is refused, before it is queued, above 8 M cells, 20 000 steps, a
    3.0 GB GPU memory estimate, or a 30-minute run-time estimate. The refusal
    names the limit, the value and the lever to change.
  - The Jetson's memory is shared with other teams' jobs, so the worker checks
    free memory before starting a `td` job and leaves it queued ("Jetson busy")
    when it is short.
  - The estimate (cells, steps, memory, run time) is shown before submit, as the
    cavity tool already does.

- **Scene scale and target range (owner, 15:58Z):** the owner sets the scene's
  physical size with the same `units` + `scale` controls the cavity tool already
  has; the physical size is size × scale × unit. The design range is features of
  10 nm inside objects up to 1 µm, for plasmonic devices and photonic crystals.
  A graded tensor-product grid keeps an axis-aligned film of that ratio (100:1) under the 8 M-cell cap. A curved
  10 nm shell refines nearly every plane (~400 cells per axis, ~64 M cells), so
  the cap is set in phase 1 from the measured Jetson cost per cell-step, not
  guessed. Measured 2026-10-01 for an empty metal box with no CPML and no
  dispersive metal (cupy, 1.73 M cells × 7 088 steps × 3 runs): 1.21e8
  cell-steps/s, so 8 M × 20 000 = 1.6e11 cell-steps ≈ 22 min for that case.
  The CPML and the metal ADE add work and memory, so phase 2 re-measures the
  rate with an open boundary and one gold region; the run-time limit uses the
  rate for the run's own configuration. A 64 M-cell shell stays refused by its
  count. Estimated memory at 8 M cells (not yet measured): ~1.3 GB metal box,
  ~1.7 GB open, ~2.9 GB open with gold if ADE storage were full-grid, which is
  why ADE storage covers only the metal's bounding box. All solver arithmetic is SI, and tests include a nanometre-scale
  grid so float32 coefficients are checked at the scale actually used.

## Validation (each phase ships with its benchmark)

- **V1, metal box with a pulse:** the spectrum peaks are within 1% of the analytic
  rectangular-cavity frequencies, every coupled mode in the band appears, and
  the peaks are within 1% of our cavity solver's eigenvalues for the same box.
- **V2, open vacuum with a CW point source:** in the far field the amplitude falls
  as 1/r, within 5%.
- **V3, open plane-wave pulse through empty space:** three transit times after
  the source ends, the residual energy in the physical region is < 1e-4 of the
  peak, so the PML absorbs, and the scattered-field region stays < 1e-3 of the
  incident amplitude.

- **V4, gold film reflectance:** at normal incidence, a gold slab's reflectance
  (≥ 10 skin depths thick) over the pulse band matches the analytic Fresnel
  result for the same Lorentz-Drude model within 0.02 absolute, and the
  material table matches a published Rakić dataset within 2%.
- **V5, thin film between plates:** the resonances of a dielectric film of index
  n filling the gap between two PEC plates match the parallel-plate lateral modes
  f = (c/2n)·√((m/a)² + (p/b)²) within 1%, on a graded grid whose film is far
  thinner than the coarse cell. These depend on n and not on the thickness, so
  they show the film is seen.

## Phases

1. Solver core: Yee update, PEC, CPML, point source with both profiles, probes.
   Benchmarks V1 and V2.
2. Plane-wave TF/SF source (V3), dispersive metals with the material library (V4),
   and the drawn waveform.
3. Graded-grid builder, sub-cell averaging, coatings, and the voxeliser from the
   upload's regions (V5), plus the estimate and caps.
4. Worker job kind `td`, result files, and routes. Every reader of each output
   must be listed, so that a missing file cannot 404 a page that does not need it.
5. Web form, field-movie playback, trace and spectrum plots, and the in-browser
   check against a real finished `td` job.

6. Periodic boundaries (owner, 16:00Z: "1"): one repeating cell with
   wrap-around faces, including a Bloch phase for a chosen wave vector. The core
   stays real float32. Away from k = 0 and the zone edge the Bloch faces mix the
   real and imaginary parts every step, so the cos and sin fields are stepped
   together with cross-coupled face updates; phase 6 adds a face hook to the
   core for this. Benchmark V6: for TM and TE, band frequencies at chosen
   k-points match the existing photonic-crystal solver for the same 2-D crystal
   within 1%, and every band in the pulse band appears.

## Out of scope

Nonlinear and magnetic materials, flux and transmission monitors, and multi-GPU
runs. The cell count is still capped (8 M cells), so a very large object
with a very thin coating may be refused. The estimate says so before queueing.
