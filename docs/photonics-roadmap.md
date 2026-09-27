# Computational photonics roadmap for the Mode Search Engine

*Research and proposal, 2026-09-27. Requested by the owner: "expand the capabilities
of what they've built into more computational photonics territory ... photonic
crystals, plasmonic devices, topological insulators ... even things like waveguides.
Propose some next features after doing some research."*

This is a plan, not a build. Nothing here has been implemented. Effort figures are
rough ranges for this team (lead + builder + reviewer) and should be read as
"small / medium / large", not as commitments.

## 1. Where we are today

What the solver is, read from `dockerfiles/maxwell_solver/`:

| Aspect | Today |
|---|---|
| Physics | Time-harmonic Maxwell eigenproblem: find E, k² with ∫ curl E · curl v = k² ∫ εᵣ E · v |
| Elements | Lowest-order Nédélec (Whitney) **edge elements** on linear tetrahedra (one unknown per mesh edge). These are the right elements for vector electromagnetics: they avoid the spurious modes that nodal elements produce. |
| Geometry | One closed surface (`.obj`), tetrahedralised with TetGen after decimation to the resolution the requested modes need |
| Material | **One uniform** relative permittivity εᵣ = n² for the whole interior (`mse_fem.py`) |
| Boundary | **PEC** (perfect metal) on the outer surface, by eliminating boundary edges |
| Algebra | Real symmetric generalised eigenproblem; shift-invert Lanczos (`eigsh`) with a projection that removes the gradient (static) null space |
| Linear solver | **In production:** SciPy SuperLU with a symmetric fill-reducing ordering, on the CPU. **Not yet in production:** a GPU path using NVIDIA cuDSS, measured on the Jetson test bench only (2026-09-27, 10x7x5 mm box, 20 modes): slower than the CPU at 10k elements; about 2x faster at 21-31k; 7.6x at 77k; 9.5x at 99k; 266k elements solved in 75 s on the GPU. Wiring it into the solver is in progress (the "dispatch" work). |
| Output | 20 lowest modes: frequencies, 3-D fields sampled on a grid, a rotatable smoke view with component and coordinate choice, ParaView download |
| Compute | Cluster: one bounded Job at a time (4 CPU / 14 GiB) under the project's power cap. Jetson Orin: 8 GB shared CPU/GPU memory, ~9 W under load |

**The honest summary:** we have a correct 3-D vector finite-element eigen-solver for
closed metal cavities. That core (edge elements, eigen-solver, GPU factorisation,
meshing, the viewer) is the hard part of almost every feature below. What we lack is
(a) more than one material, (b) boundaries other than metal, and (c) complex
arithmetic. Most of the owner's list needs one or more of those three.

## 2. The building blocks everything else needs

Four capabilities recur across photonic crystals, waveguides, plasmonics and
topological photonics. Building them once, in this order, unlocks the rest.

### B1. Multiple materials (regions with different refractive index)

- **Why:** every photonic device is a pattern of different materials: rods in air,
  a silicon core in glass, a metal particle in a dielectric. Today the whole interior
  has one index.
- **Maths:** εᵣ becomes piecewise constant per tetrahedron. In the weak form only the
  mass matrix changes: each element's mass matrix is scaled by that element's εᵣ.
- **Code:** assembly is a one-line scaling per element (`element_matrices` already
  works per tet). The real work is meshing: the tetrahedral mesh must **conform to
  material interfaces**, i.e. TetGen with region attributes, from uploads with several
  closed parts (several `.obj` files, or one `.obj` with named groups) plus a
  material per part. Nested and touching parts need care.
- **Web app:** upload several parts, pick a material (or type an index) for each.
  The viewer shows part outlines; the field view is unchanged.
- **Compute:** same as today. Meshes near high-index parts get finer, because the
  wavelength inside them is shorter.
- **Effort:** medium. Most of it is geometry handling and validation, not physics.

### B2. Periodic (Bloch / Floquet) boundaries

- **Why:** photonic crystals and metamaterials are infinite periodic lattices. You
  simulate one unit cell with the rule "the field on one face equals the field on
  the opposite face times a phase e^{i k·a}". Sweeping the Bloch wavevector k gives
  the **band diagram**.
- **Maths:** tie each boundary edge to its partner on the opposite face with the
  phase factor. The matrices become **complex Hermitian**; the eigenvalues stay real
  (lossless materials). The gradient null-space projection must use the same phases.
- **Code:** a **periodic mesh**, meaning the opposite faces must have matching
  triangulations. TetGen does not do this by itself; the usual fix is to build the
  unit cell's boundary surface ourselves with mirrored faces. Complex arithmetic
  through the whole pipeline: `assemble_global` already takes a dtype, but nothing
  passes a complex one today, and both the eigen-solver wrapper and the field sampler
  are real-only (`mse_fields.py` casts eigenvectors to float). So "complex" means
  changes in assembly, the solver, the projection, the sampler and the viewer (a
  Bloch mode has a phase), not just a dtype switch. SciPy's `eigsh` handles Hermitian
  complex problems, and cuDSS supports complex matrices.
- **Web app:** a **band diagram** (frequency vs. k along the standard path through the
  Brillouin zone, e.g. Γ-X-M-Γ) with band gaps shaded. Click any point to see that
  Bloch mode's field in the unit cell in the existing 3-D view.
- **Compute:** one small unit-cell solve per k-point, typically 30-100 k-points. Each
  is independent, which is exactly the batching the owner asked about: a warm GPU
  worker runs them back to back. Estimated seconds per k-point for a typical 3-D
  cell, so minutes per diagram. 2-D crystals (below) take seconds in total.
- **Effort:** medium to large in 3-D (the periodic mesh is the risky part); small to
  medium in 2-D.

### B3. Open boundaries (perfectly matched layers, PML)

- **Why:** real resonators, antennas, nanoparticles and leaky waveguides radiate.
  With metal walls everywhere, nothing can leave, so we cannot compute a quality
  factor (Q) or a radiation loss. A PML is an absorbing layer that imitates empty
  space around the object.
- **Maths:** complex coordinate stretching in the layer. The problem becomes
  **complex symmetric (non-Hermitian)**. The eigenvalues are complex:
  real part = frequency, imaginary part = decay rate, and Q = Re(ω) / (2 Im(ω)).
- **Code:** complex assembly with a per-element anisotropic tensor in the PML region
  (needs B1's per-region materials and B4's tensors); a complex shift; filtering of
  "PML modes" that live in the layer and are not physical. SciPy's `eigs` or ARPACK
  in complex mode on the CPU; cuDSS handles complex factorisations on the GPU.
- **Web app:** each mode shows a Q factor and a decay time; the view marks the PML
  region (hidden by default).
- **Compute:** a larger domain (the object plus air plus the layer), so 2-5x more
  elements than the same object in a metal box. This is where the Jetson GPU earns
  its place for everyday jobs.
- **Effort:** large. It touches assembly, the eigen-solver, mode filtering and
  validation (against analytic sphere resonances, e.g. Mie theory).

### B4. Anisotropic and gyrotropic materials (tensor permittivity)

- **Why:** birefringent crystals, liquid crystals, and, importantly, magneto-optic
  (gyrotropic) materials, which are the classic route to topological photonics
  because they break time-reversal symmetry.
- **Maths:** εᵣ becomes a 3x3 tensor per region. The element mass matrix uses the
  tensor, and Whitney elements handle this exactly. A gyrotropic tensor is Hermitian
  but complex, so the problem becomes complex Hermitian.
- **Code:** small in assembly once B1 exists; the rest is input validation and units.
- **Effort:** small, after B1 and complex assembly.

## 3. The owner's list, feature by feature

### 3.1 Photonic crystals (band diagrams, band gaps)

- **Needs:** B1 (rods or holes of one material in another) + B2 (periodic cell).
- **Fastest valuable version: 2-D photonic crystals.** An infinite array of rods or
  holes, e.g. the textbook square lattice of alumina rods. In 2-D, Maxwell's
  equations split into two scalar polarisations (TE and TM), solved with ordinary
  nodal triangles in seconds. This is the standard teaching and design tool, and it
  has well-known published benchmark band diagrams to validate against (for example
  Joannopoulos et al., *Photonic Crystals: Molding the Flow of Light*).
- **Then 3-D** (slabs with holes, woodpiles) on the existing tetrahedral solver with
  B2.
- **Web app:** a lattice picker (square, triangular), rod/hole radius, the two
  indices, then the band diagram with gaps shaded and the gap size in percent.
  Click a point to see the mode.
- **Effort:** 2-D small to medium; 3-D medium to large.

### 3.2 Waveguides (mode solving on a cross-section)

- **Needs:** B1 in 2-D. Open boundaries are optional at first: a guided mode decays
  outside the core, so a metal box a few wavelengths away is accurate enough for
  guided modes.
- **Maths:** a **2-D cross-section eigenproblem** for the propagation constant β at a
  given wavelength (effective index n_eff = β / k₀). The standard formulation uses
  edge elements for the transverse field and nodal elements for the longitudinal
  one: our existing element technology, in 2-D.
- **Code:** a 2-D mesher (e.g. the `triangle` library or gmsh) and a 2-D assembly
  beside the 3-D one. A parametric input covers most real cases: rectangles for
  strip/rib/slot waveguides, circles for fibres. It needs no `.obj` upload.
- **Web app:** mode profiles as 2-D color maps with arrows, a table of n_eff and
  polarisation, and a **dispersion plot** (n_eff vs. wavelength, one quick solve per
  wavelength).
- **Compute:** tiny: a 2-D mesh of a few thousand triangles solves in well under a
  second.
- **Effort:** medium. Very practical for integrated photonics (silicon photonics:
  n ≈ 3.48 core in n ≈ 1.44 glass), and later it supplies the "port modes" for
  device simulations.

### 3.3 Plasmonic devices

- **Needs:** B1, B3 (plasmons radiate, so PML is essentially required), and
  **dispersive, lossy metals**.
- **Maths:** a metal's permittivity depends on frequency (Drude or Drude-Lorentz
  models) and is negative and complex at optical frequencies. The eigenproblem is
  then **nonlinear** in the frequency. Two standard routes: (a) auxiliary-field
  linearisation (add polarisation unknowns so the problem becomes linear but larger),
  or (b) for spectra, a **driven (scattering) solve**: shine a plane wave in and
  compute absorption and scattering vs. wavelength, one linear solve per wavelength.
  Route (b) is what most plasmonics users actually want (extinction spectra).
- **Code:** complex arithmetic throughout, a plane-wave source, cross-section
  integrals, and very fine meshing at metal surfaces (the fields vary over
  nanometres).
- **Scale:** plasmonic structures are nanometre-sized. The owner asked (2026-09-27)
  for a **global scale** control, meaning the object's overall size and not the mesh
  size. Today the app accepts only mm/cm/m/in; adding um, nm and a free scale factor
  is already queued with the upload-form work. Note that a pure size change of a
  metal-walled (PEC) cavity only rescales its frequencies. Real plasmonic behaviour
  needs the dispersive metals described here.
- **Compute:** heavy. Surface refinement multiplies the element count, and there is
  one factorisation per wavelength, so 50-200 wavelengths per spectrum. The Jetson
  GPU helps. Long spectra may need the batching worker and possibly a cluster GPU
  (owner decision).
- **Effort:** large. Recommended only after B3 is solid.

### 3.4 Topological photonics (edge states, Chern numbers)

- **Needs:** B2 (Bloch modes), usually B4 (gyrotropic materials) or a symmetry-based
  design, and complex eigenvectors.
- **Maths:** from Bloch eigenvectors computed on a grid of k-points across the
  Brillouin zone, compute the **Berry curvature** and the **Chern number** of a band
  with the lattice (Fukui-Hatsugai-Suzuki) method: overlaps of neighbouring
  eigenvectors, which is robust and cheap once the eigenvectors exist. **Edge
  states**: a supercell (a strip periodic along the edge), where the band diagram
  shows states crossing the gap.
- **Code:** a k-grid driver (again batching), overlap computation with the mass
  matrix as the inner product, careful handling of degeneracies.
- **Web app:** band diagram with each band labelled by its Chern number; a supercell
  view showing a mode confined to the edge; later, a demonstration that the edge
  wave flows around a defect without reflecting (needs a driven solve).
- **Compute:** a 2-D k-grid (e.g. 20x20 = 400 small solves) per structure. In 2-D
  that is seconds to a minute; in 3-D minutes to tens of minutes on the GPU.
- **Effort:** medium on top of 2-D photonic crystals; large in 3-D. This is a
  flagship research feature rather than a first step.

### 3.5 Other capabilities worth having

- **Scattering and transmission (driven frequency-domain solves with ports):**
  transmission spectra of devices (bends, splitters, ring filters). Needs 3.2 (port
  modes) + B3. It's the everyday workflow of photonic device design.
- **Symmetry reduction:** use mirror symmetry to solve a half or quarter of the
  object with PEC/PMC walls. It gives 2-8x fewer unknowns for free on symmetric
  shapes, and classifies modes by symmetry. Small effort, big payoff for big uploads.
- **Higher-order (curved, second-order) edge elements:** much better accuracy per
  unknown on curved surfaces (spheres, rings). Medium effort in assembly and
  meshing; it pays off for plasmonics and high-Q resonators.
- **Mode tracking and degeneracy labelling:** group degenerate modes (as in the
  sphere benchmark) and track modes across a parameter sweep. Small.
- **Parameter sweeps:** "vary the rod radius from 0.2 to 0.4 and plot the gap". This
  is the batching queue again, with a plot. Small once batching exists.

## 4. Recommended next features, in order

Chosen for: value to the owner, reuse of what exists, dependency order, and cost
under the current power and quota limits.

1. **Multiple materials (B1).** The foundation for everything below. It also makes
   today's cavity tool more realistic immediately: a dielectric resonator inside a
   metal box, or a layered cavity. *Medium effort, no extra compute.*
2. **2-D photonic crystal band diagrams.** The quickest route into the owner's list:
   a lattice picker, band diagram, band gaps, and click-to-view modes. It's validated
   against textbook results, and it's where the batching worker starts to matter.
   *Small to medium effort, seconds of compute.*
3. **Waveguide mode solver (2-D cross-sections).** n_eff, mode profiles, dispersion
   plots; directly useful for integrated photonics, and it later provides the ports
   for device simulation. *Medium effort, negligible compute.*
4. **Open boundaries (PML) with Q factors.** Turns the cavity tool into a resonator
   tool (ring, disk and photonic crystal cavities with finite Q), and is the gate to
   plasmonics and scattering. *Large effort; this is where the Jetson GPU pays off
   every day.*
5. **Topological photonics on 2-D photonic crystals.** Chern numbers and edge states,
   built on items 1-2 plus gyrotropic materials (B4). **B4 (tensor and gyrotropic
   permittivity) must land between items 2 and 5**; it is small once item 1 and
   complex arithmetic exist. *Medium effort after item 2.*

Plasmonics (dispersive metals, spectra) and full 3-D topological structures come
after item 4. They are the most compute-hungry, and they're where a cluster GPU
might eventually be worth discussing.

## 5. How this fits the current platform

- **Compute limits:** items 1-3 and 5 fit comfortably in today's limits (2-D problems
  are tiny). Item 4 and plasmonics are the first to push memory; the Jetson GPU
  (and the routing/batching design already approved in principle) is the natural
  home for them.
- **Validation:** each feature should ship with a published or analytic benchmark,
  as the cavity solver did with the box and sphere: textbook 2-D band diagrams,
  analytic slab-waveguide modes, Mie resonances for PML.
- **Web app:** the existing job model (upload/configure, queue, view) extends
  naturally with a "problem type" choice: cavity (today), periodic cell, waveguide
  cross-section. Band diagrams and dispersion plots are new 2-D charts next to the
  existing 3-D view.
