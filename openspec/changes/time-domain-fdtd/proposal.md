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
      samples. It is resampled to dt. Before queueing, the run warns or refuses
      if the waveform's spectrum has significant energy above the grid's
      resolvable frequency (10 cells per wavelength in the finest material).
- **Outputs:**
  - the probe time trace at the source and at up to 4 user probe points;
  - the spectrum of each trace, which shows the resonances for a pulse;
  - a field movie of |E| (≤ 64³ samples, ≤ 60 frames), which the 3-D viewer plays.
- **Limits:**
  - The cell size is λ_min / (10·|n|_max) in each material, refined further to resolve each
    thin layer (≥ 4 cells across it, or the metal skin depth where that is thinner).
  - The run is refused, before it is queued, above 8 M cells or 20 000 steps.
  - The estimate (cells, steps, memory, run time) is shown before submit, as the
    cavity tool already does.

- **Scene scale and target range (owner, 15:58Z):** the owner sets the scene's
  physical size with the same `units` + `scale` controls the cavity tool already
  has; the physical size is size × scale × unit. The design range is features of
  10 nm inside objects up to 1 µm, for plasmonic devices and photonic crystals.
  A graded tensor-product grid keeps an axis-aligned film of that ratio (100:1) under the 8 M-cell cap. A curved
  10 nm shell refines nearly every plane (~400 cells per axis, ~64 M cells), so
  the cap is set in phase 1 from the measured Jetson cost per cell-step, not
  guessed; until then the estimate refuses such a run by its count. All solver arithmetic is SI, and tests include a nanometre-scale
  grid so float32 coefficients are checked at the scale actually used.

## Validation (each phase ships with its benchmark)

- **V1, metal box with a pulse:** the spectrum peaks are within 1% of the analytic
  rectangular-cavity frequencies, and within 1% of our cavity solver's eigenvalues
  for the same box.
- **V2, open vacuum with a CW point source:** in the far field the amplitude falls
  as 1/r, within 5%.
- **V3, open plane-wave pulse through empty space:** after the pulse leaves, the
  residual energy is < 1e-4 of the peak, so the PML absorbs, and the
  scattered-field region stays < 1e-3 of the incident amplitude.

- **V4, gold film reflectance:** at normal incidence, a gold slab's reflectance
  over the pulse band matches the analytic Fresnel result for the same
  Drude-Lorentz model, within 2%.
- **V5, thin film between plates:** the resonances of a dielectric film between
  two metal plates match the analytic planar-cavity frequencies within 1%, on a
  graded grid whose film is far thinner than the coarse cell.

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
   stays real float32, and a Bloch phase uses two real runs in quadrature, so
   phase 1 does not change. Benchmark V6: band frequencies at chosen k-points match
   the existing photonic-crystal solver for the same 2-D crystal, within 1%.

## Out of scope

Nonlinear and magnetic materials, flux and transmission monitors, and multi-GPU
runs. The coarse cell is still capped (8 M cells), so a very large object
with a very thin coating may be refused. The estimate says so before queueing.
