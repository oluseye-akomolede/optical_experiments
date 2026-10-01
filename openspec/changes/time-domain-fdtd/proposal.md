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
- **Geometry:** the upload is voxelised onto a uniform grid, and each cell takes the
  refractive index of the region it falls in (the multi-material regions that
  already exist). Materials are lossless and non-dispersive in this change.
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
  - Arbitrary user-drawn waveforms are out of scope.
- **Outputs:**
  - the probe time trace at the source and at up to 4 user probe points;
  - the spectrum of each trace, which shows the resonances for a pulse;
  - a field movie of |E| (≤ 64³ samples, ≤ 60 frames), which the 3-D viewer plays.
- **Limits:**
  - The cell size is the smaller of the user's value and λ_min / (10·n_max).
  - The run is refused, before it is queued, above 8 M cells or 20 000 steps.
  - The estimate (cells, steps, memory, run time) is shown before submit, as the
    cavity tool already does.

## Validation (each phase ships with its benchmark)

- **V1, metal box with a pulse:** the spectrum peaks are within 1% of the analytic
  rectangular-cavity frequencies, and within 1% of our cavity solver's eigenvalues
  for the same box.
- **V2, open vacuum with a CW point source:** in the far field the amplitude falls
  as 1/r, within 5%.
- **V3, open plane-wave pulse through empty space:** after the pulse leaves, the
  residual energy is < 1e-4 of the peak, so the PML absorbs, and the
  scattered-field region stays < 1e-3 of the incident amplitude.

## Phases

1. Solver core: Yee update, PEC, CPML, point source with both profiles, probes.
   Benchmarks V1 and V2.
2. Plane-wave TF/SF source. Benchmark V3.
3. Voxeliser from the upload's regions, plus the estimate and caps.
4. Worker job kind `td`, result files, and routes. Every reader of each output
   must be listed, so that a missing file cannot 404 a page that does not need it.
5. Web form, field-movie playback, trace and spectrum plots, and the in-browser
   check against a real finished `td` job.

## Out of scope

Dispersive or lossy materials, user-drawn waveforms, flux and transmission
monitors, non-uniform grids, and multi-GPU runs.
