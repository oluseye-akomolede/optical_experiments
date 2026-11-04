# Design — electrical-simulation

Revised on 2026-11-01 after the owner's answers (`tasks.md` 0.1), and again on 2026-11-02
after the owner widened it to a general-purpose solver (`tasks.md` 0.3). The superconductor
track moved, unchanged, to the separate change `superconductor-models`. "Loss" ("b yes" on
2026-11-02) is **in this change**: optical absorption by conductor parts in a cavity, its own
`optical-loss` capability, with paper 1 as one fixture and any conductor part in scope.

**The shape, after 2026-11-02: the job is general and paper 1 is a fixture.** The device
below is the first validation bench, not the limit of what the request can describe. Any
number of parts and contacts, materials from the shipped library (task 2.2) with per-part
overrides, electrostatics and drift-diffusion in an axisymmetric cylinder form or in 3-D.
Paper 1 is one entry in the bench set, alongside a parallel-plate capacitor and a
multi-terminal stack (`electrostatics` spec), because generality must be tested by something
that is not the wire.

**Two forms, and why.** A round device can be solved as a 2-D (r, z) mesh of an equal-area
cylinder, each node and edge weighted by 2πr, which is far cheaper than 3-D for the same
resolution. The form is offered only when the uploaded cross-section is round to within 2 %
(circumradius vs inradius), refused at upload otherwise with both figures named. Everything
else is 3-D. A 3-D run is admitted only when its own priced estimate fits the Job (task 2.5):
the estimate comes from the node and edge counts of the mesh that geometry would build, so two
devices that differ only in size get different answers. §5 below prices that mesh; it is the
method the check uses, not a table the check reads.

The rule throughout: start with the simplest model that gives a useful number. Check it
against an answer we did not compute ourselves, and move up a rung only when an owner question
needs it. Every model has an answer key (exact or published) and a mutation that must make
its check fail. Citations used to be marked **(unsure)**: from memory, to be checked against
the source before any code or test quoted a number from them. **Task 0.2 (2026-11-04) cleared
every one of those marks** — each item was either read from its source or replaced by an
explicit statement that no source provides it (e.g. the paper-1 dot's material and Γ₀ are
inputs we choose, not quotations). Any future citation from memory must re-open that practice
rather than quietly inheriting trust. **Every compute figure here is an estimate** until a
bench measures it. The materials library obeys the same rule with no exceptions: a library
value ships only after it is read from its cited source (task 0.2, 2.2).

## 0. The device and what exists

**Device (paper 1 is the validation bench, not the scope).** The job accepts any device
(tasks 0.3, "general ok"); the first thing built and checked against is the paper-1 nanowire
quantum dot (Gangopadhyay, Grayli, Kokilathasan & Reimer, "A Broadband Nanowire Quantum Dot
Cavity Design for the Efficient Extraction of Entangled Photons", arXiv:2601.04440v1
[quant-ph], 7 Jan 2026 — one version, no journal DOI). The scene follows
`scripts/paper1_nanowire.py`: a **wurtzite InP** hexagonal prism of **height H = 1375 nm and
diameter D = 420 nm** ("the optimal cavity dimensions (height of 1375 nm and width of 420 nm)
that we found", p. 3, Results; Fig. 1 caption gives "height (H) of 1375 nm and diameter (D)
of 420 nm"). Take H and D from p. 3, **not** the Fig. 5a caption, which swaps them
("(D=1375 nm and H= 420 nm)", p. 8) and contradicts the rest of the paper. The 1.14 factor is
the paper's own hex-to-cylinder conversion: "A scaling factor of 1.14 was applied to the
diameter as the nanowire has a hexagonal cross-section" (p. 10, S2). `scripts/paper1_nanowire.py`
applies that relation as the paper writes it — `2s = 1.14 D`, s the hexagon's side — giving a
corner-to-corner width of 1.14 × 420 = 478.8 nm, a **circumradius of 239.4 nm** and a
flat-to-flat of 414.6 nm. The bench's 1375–1525 nm scan is the height sweep; the wire height
is the 1375 nm above. It stands on a 12 nm SiO₂ buffer ("a 12 nm thick oxide layer optimizes
the Purcell enhancement and extraction efficiency", p. 4) over a gold mirror.

**The dot is a point dipole in this paper — its material, size and Γ₀ are not quotable from
it.** The paper never names the dot's composition for its own model: "A dipole source was used
to model the QD emitter" (p. 9, S1), and it gives no dot dimensions; the only positional fact
is "a QD emitter placed on-axis at 30 nm below the top" (Fig. 2 caption, p. 5) at "the second
antinode from the top" (p. 3). "InAsP" appears only as cited literature ("In AsP/InP NW
quantum dot sources with a spontaneous emission lifetime of typically ∼1 ns", p. 2) and via a
2012 reference on InAsP/InP dots, never as this paper's system. There is **no Γ₀, decay rate or
dipole moment** anywhere in it; the Purcell factor is read off the simulator ("directly
accessed from the results tab of the dipole", p. 9). So the zero-field dipole or Γ₀ that §3
needs **must be an input we choose, not a quotation** — the ∼1 ns figure above is a literature
value for InAsP/InP NW dots and is the only lifetime the paper offers.

**Permittivities: the paper states refractive indices only, never ε.** InP is modelled at
"an index of refraction n=3.44 ... to simulate the nanowire material (Wurtzite InP)" (p. 9,
S1) — about 11.8 when squared, but that is n², **not** a static ε_s. Gold uses "Palik's
experimental permittivity data" (p. 9) with no number given, and **no SiO₂ index or ε is stated
anywhere**. The paper is therefore no source for any `eps_static` in this design.

- **gate (G):** the gold mirror is the back electrode (it is already a conductor under the
  buffer), plus a top electrode. The wire is undoped. The field at the dot is set by V_G.
  Electrostatics (L1) alone gives the field.
- **p-i-n (P):** a p-doped base, an intrinsic segment holding the dot and an n-doped top. One
  contact reaches the base through the oxide, and one sits on the top. This needs
  drift-diffusion (L5).

The owner's "full electrical analysis" answer (2026-11-01, superseded 2026-11-02 by
tasks 0.3) means both: G for the Stark sweep, and P for I-V and injection timing.

**What exists:**
- `mse_meshing.tetrahedralize` returns `(points_m, tets, region_of_tet)` for a multi-part
  `.obj`, so each part is already a mesh label. It sets ONE global `maxvolume` from the finest
  region's edge (the module says so). A 12 nm oxide therefore forces a 4–6 nm edge everywhere.
  Grading works only through TetGen's `-m` sizing field: a coarse background mesh with a
  per-node `target_size`, passed as `bgmesh=`. `add_region(max_vol=)` does nothing on a
  nested PLC, and a global `maxvolume` overrides the sizing field. All three were measured
  (`benchmarks/mesh_grading_feasibility.py`): 93 % fewer tets at the same core edge.
- `mse_backend.factor` routes SPD sparse factorisations (CPU `splu`, an optional GPU with a
  checked residual).
- `fdtd.delivered_power_spectrum` / `fdtd.purcell_spectrum` give F(ω) across a pulse's band
  from one run, and the paper-1 bench prints F per wavelength bin and saves an `.npz`
  record. F at a Stark-tuned energy is therefore a lookup, not a new run, as long as the
  energy stays inside the band. **No live spec covers the Purcell path** (it is bench code), so
  this change specifies only its own reader. Backfilling a Purcell spec is out of scope and
  noted as a gap.
- The web app has three job kinds: `cavity`, `pc` and `td` (live `time-domain` spec). The
  cluster launcher builds one Job shape, the cavity solver. `pc` and `td` run on the Jetson
  only.
- Compute (`deployment-bounds`, project brief): one Job, ≤ 4 CPU / 16 GiB per container,
  5 CPU / 18 GiB for the whole namespace including the web app, and no GPU. The Jetson Orin
  is tier 2.

## 1. Physics ladder (the paper-1 device)

| rung | model | gives | phase |
|---|---|---|---|
| L1 | 3-D electrostatics ∇·(ε₀ε_s∇φ) = 0, fixed-potential electrodes, no free carriers | field at the dot vs V_G; capacitances | 1 |
| L2 | perturbative Stark ΔE = −pF + βF² | E_X(V) from a fitted p, β | 1 (fit of L3) |
| L3 | 3-D single-band effective mass (electron, heavy hole) in the L1 field + first-order Coulomb | E_X(F), overlap, p, β, dipole orientation | 1 |
| L5 | drift-diffusion + Poisson + SRH/radiative recombination (DEVSIM) | I-V, the field at the dot under bias, carrier arrival after a pulse | 2 |
| C  | coupling: F at E_X(V) from a Purcell spectrum; Γ = F·Γ₀·(overlap ratio); timing = g ⊛ Γe^{−Γt} | radiative rate and photon-timing distribution vs V | 3 |
| L  | contact loss: the same Purcell run with and without the contact metal | ΔQ, ΔF from the contact ("try loss") | 3 |
| L4 | Schrödinger-Poisson | screening by charge in or near the dot | not planned |
| L6 | multi-band k.p + strain | fine structure, light-hole mixing | not planned |

Why these and not more:
- Undoped G needs only L1. The dot's response is, to leading order, a dipole and a
  polarisability (L2). Measured curves are fitted to that form (Fry, Itskevich, Mowbray,
  Skolnick, Finley et al., "Inverted Electron-Hole Alignment in InAs-GaAs Self-Assembled
  Quantum Dots", Phys. Rev. Lett. 84, 733–736 (2000); Finley, Sabathil, Vogl, Abstreiter,
  Oulton, Tartakovskii, Mowbray, Skolnick, Liew, Cullis & Hopkinson, "Quantum-confined Stark
  shifts of charged exciton complexes in quantum dots", Phys. Rev. B 70, 201308(R) (2004)).
  Note the division of labour: Fry 2000 is the InAs-GaAs dot (not InP), and Finley 2004 is a
  **Stark-shift** paper, supporting the L2 polarisability form, not a Purcell figure — neither
  is a paper-1 reference, so neither may be cited as the bench's dot.
- L3 predicts p and β from geometry instead of quoting them. The owner asked for full 3-D
  (2026-11-01, superseded 2026-11-02 by tasks 0.3), and a hexagonal wire is not axisymmetric,
  so L3 is 3-D. It runs on a uniform
  Cartesian finite-difference box around the dot, with the L1 field interpolated onto it.
  At 0.5 nm a 60 × 60 × 40 nm box is about 1.2 × 10⁶ unknowns, about 0.35 GiB (estimate; §5).
  This is the STARK grid arithmetic, which is priced by its own explicit formula in §5's
  Stark table — it is NOT priced by the measured electrostatics model, which covers the
  LOBPCG-free direct and AMG paths only.
  Material parameters come from Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001).
- L5 is the p-i-n half of "full electrical analysis", as in the first electrically driven
  single-photon source (Yuan et al., Science 295, 102 (2002)). Full 3-D L5 does not fit at
  junction-resolving resolution (§5), so L5 has two forms. The **axisymmetric** (r, z) form
  uses an equal-area cylinder for the hexagon and resolves the junctions. The **coarse 3-D**
  form runs on the tet mesh as a cross-check, and is valid only where the Debye length is
  resolved.

  **The 2πr question, settled (2026-11-02, task 0.2).** DEVSIM does the axisymmetric case
  natively: it ships `cylindrical_node_volume`, `cylindrical_edge_couple` and
  `cylindrical_surface_area`, which supply the 2πr weighting of the weak form, selected by
  `raxis_variable` (here `x`, the radial coordinate) and `raxis_zero`. There is no custom
  model to write and no FE assembly to build. The weighted values live in a **separate
  model**: after `cylindrical_edge_couple`, `EdgeCouple` keeps its Cartesian values
  (identical arrays before and after the call) and `CylindricalEdgeCouple` carries the
  cylindrical ones. So the build must name the `Cylindrical*` models in its equations —
  naming `EdgeCouple` there would silently solve the Cartesian problem. Measured on devsim
  2.11.2, not read from the docs: before and after the call `EdgeCouple` reads [0.125,
  0.25, …, 0.125] on a radial strip, unchanged, while `CylindricalEdgeCouple` reads [0.44,
  0.98, …, 0.74] and rises with r (0.44 at r = 0.5, 1.96 at r = 1.0 for the same edge).
  The model names created are `CylindricalNodeVolume`, `CylindricalEdgeCouple` and
  `SurfaceArea` (the last keeps its plain name).
  Two consequences for §5 and for 0.2's follow-on work:
  - The cylindrical volume carries a first-order error, `1 + 1/(9·N_r)` on an
    exactly uniform grid, falling as `1/N_r` (measured at N_r = 10, 20, 40: `N_r × error`
    is 0.1111 at every one). It is a discretisation artefact that converges
    away, not a wrong weight, but it is largest near the axis, so the radial grading should
    put the finest cells at r = 0 (already §5's intent). `scripts/devsim_cylindrical_check.py`
    prints this table (devsim node-volume sum against `π(r_out² − r_in²)h` = 9.424778 for the
    r ∈ [1, 2], h = 1 annulus: 9.529498, 9.477138, 9.450958). The sum includes boundary nodes,
    which carry partial volumes, so it is the region's total volume, not an interior average.
  - The contact/mesh topology is fussier than the solver: a contact only exists once the
    mesh extends past the device into a filler region, and regions whose nodes carry no
    equation make the matrix singular. That is an implementation detail for the build, not a
    limitation of the method, and it is recorded because it cost time to find.
  - **Where the mesh comes from.** `create_gmsh_mesh` + `add_gmsh_region` load both a
    TetGen tetrahedral mesh (3-D path; DEVSIM derives the boundary triangles, lines and
    points from the tets, so no separate surface file) and a 2-D triangle mesh (axisymmetric
    path, cylindrical models on top). The reader requires a positive physical number, and
    the element line is ordinary Gmsh 2.2: `id type ntags physical_number geometric_tag
    nodes…`, with `ntags = 2`, so the physical number is the first tag. (An earlier version
    of this note put a separate element index before the physical number and counted three
    tags; that parses, but the reader takes `ilist[3]` as the physical number, so each
    element lands in the group numbered by its own element index and region selection by
    name breaks silently.)
  - **The Jetson runs it.** `devsim==2.11.2` installs and imports on the Orin (aarch64) with
    no `DEVSIM_MATH_LIBS`, because the aarch64 wheel bundles its BLAS — the x86-64 wheel does
    not, so the VM and the cluster workers need the system-library pointer and the Jetson
    must not be given one. Checked that the cylindrical models work there, not that a solve
    runs there.
- L4 and L6 are left out. One exciton in an undoped segment needs no self-consistency.
  Leading-order selection rules put the heavy-hole dipole in the plane, perpendicular to the
  wire, which is the x/y dipole the paper-1 bench already drives. Reference for later:
  Stier, Grundmann & Bimberg, Phys. Rev. B 59, 5688 (1999).

## 2. Drift-diffusion: why DEVSIM

| option | 3-D | open | fit for us |
|---|---|---|---|
| **DEVSIM** (Sanchez, "DEVSIM: A TCAD Semiconductor Device Simulator", J. Open Source Softw. 7(70), 3898 (2022), doi:10.21105/joss.03898; devsim.org) | yes, tetrahedra | Apache-2.0; wheels for x86_64 **and aarch64** Linux (v2.11.2: `manylinux_2_27_x86_64` and `manylinux_2_26_aarch64.manylinux_2_28_aarch64`, verified on PyPI) | Scharfetter-Gummel, SRH, transient, Python API; **direct documented** (MKL PARDISO / UMFPACK); an iterative path is declared and compiled in but undocumented and unrun by us, plus bundled UMFPACK. On aarch64 there is no MKL, so UMFPACK is the direct solver; **the changelog's note that FPE checking is disabled for direct solves is scoped to Apple M1 (macOS arm64) and says nothing about Linux aarch64** (§2). All open questions settled 2026-11-04 (§2) |
| Sesame (NIST; Gaury, Sun, Bermel & Haney, "Sesame: A 2-dimensional solar cell modeling tool", Sol. Energy Mater. Sol. Cells 198, 53–62 (2019)) | no (1-D/2-D only, from the code: `sesame/builder.py` assigns `dimension` 1 or 2 and nothing else, and the arXiv abstract says "1 and 2-dimensional systems" — read 2026-11-04; the README's "multi-dimensional" overstates the tool) | open | no 3-D cross-check |
| own SG on scikit-fem / FEniCSx | yes | open | we would write and validate the SG stabilisation and Newton ourselves: the most risk |
| Charon (Sandia, Trilinos) | yes | open | heavy C++/MPI build, more than one Job needs |
| nextnano, Sentaurus, Silvaco | yes | commercial | a reference, not a dependency |

DEVSIM is the only open option that does 3-D tets with a Python API and the standard SG
discretisation. **All of its open questions are now answered from the shipped 2.11.2 wheel
(task 0.2, 2026-11-04), so 0.2 no longer gates anything here:**

- **Wheels:** aarch64 Linux wheels exist (`devsim-2.11.2-cp39-abi3-manylinux_2_26_aarch64.manylinux_2_28_aarch64.whl`).
  The Jetson is therefore likely a real fallback, not a maybe — **but that remains unverified**:
  the wheel needs glibc ≥ 2.26 (the `manylinux_2_26` tag), and the Jetson was unreachable
  throughout task 0.2 (2026-11-04), so its glibc was never read. Check it before relying on it.
  **FPE behaviour on that wheel is untested, and the changelog does not cover it.** The one
  FPE statement in `devsim_data/CHANGES.md` sits under the heading `### Apple M1` (line 435):
  "On this platform, the software does not check for floating point exceptions (FPEs) during
  usage of the direct solver", with the follow-on "FPE during testing on macOS M1" (line 447).
  macOS arm64 is a different build from the `manylinux` aarch64 wheel named above, so the note
  does **not** describe what we would run on the Jetson. What is true for Linux aarch64 is
  simply unestablished; if a Jetson bench is ever taken up, check it rather than assume either way.
- **Mesh hand-off:** `create_gmsh_mesh(mesh=..., file=...)` reads a `.msh`, **and can equally
  take `coordinates=`, `elements=`, `physical_names=` as plain Python lists** (element type 3 =
  tetrahedron). So a TetGen mesh can be handed over as arrays with no file on disk; writing a
  `.msh` is a fallback, not a requirement.
- **Linear solvers:** both, though only the direct one is *documented*. `solve(...)` declares
  `solver_type : {'direct','iterative'}` and the wheel bundles UMFPACK; the published solver
  page lists only MKL PARDISO / UMFPACK / SuperLU and never mentions an iterative path. Read
  against the binary itself, `dsMath::IterativeLinearSolver<double>::SolveImpl` — the DC
  solve — **is compiled in and instantiated**, alongside GMRES, `BlockPreconditioner` and
  `MKLPardisoPreconditioner`; the `"... iterative solve not implemented"` messages cover only
  AC, noise and extended precision, none of which this design uses. So iterative DC exists but
  is undocumented and unexercised by us — a thing to prove on a bench, not to assume.
  **So the AMG column of the §5 coupling table is more than a hypothesis** — the memory model
  assumes an iterative path that is present in the tool, though undocumented and unproven here.
- **Cylindrical:** native (`cylindrical_node_volume`, `cylindrical_edge_couple`,
  `cylindrical_surface_area`), settled 2026-11-02, so the axisymmetric form needs no custom
  2πr weighting.

What remains genuinely unknown is mostly convergence — whether the *iterative* solver
converges on coupled drift-diffusion (see §5), a numerical question for a bench — along with
whether the undocumented iterative path behaves as its symbols suggest.

Cluster first. DEVSIM's x86-64 wheels run on the cluster workers. The Jetson is only a
fallback: aarch64 wheels do exist (confirmed independently from PyPI), so it turns on the
host's glibc being ≥ 2.26 — never read, because the host was unreachable throughout 0.2.

## 3. Coupling into the Purcell runs

- **Energy:** L3 (or L2's fit) gives E_X(V). The coupling reads a Purcell spectrum, which is
  pairs of wavelength (nm) and F from a paper-1 bench run, and interpolates F at
  λ_X(V) = hc/E_X(V). A λ_X outside the spectrum's range is refused, not extrapolated. The
  bench gains a small writer for this file (a task, not a spec change to unspecced code).
- **Orientation:** in-plane (x/y) heavy-hole dipole. The Purcell spectrum must come from a
  run with the same dipole orientation, and the file records it.
- **Strength:** Γ₀ = n·ω³·|d|²/(3πε₀ħc³) for a dipole d in a homogeneous medium of index n.
  In the field, the oscillator strength scales by |⟨ψ_e|ψ_h⟩|²(V)/|⟨ψ_e|ψ_h⟩|²(0) (from L3).
  Then Γ(V) = F(λ_X(V))·Γ₀·(overlap ratio). **The zero-field d is an input we choose, not a
  quotation** — paper 1 states no Γ₀, decay rate or dipole moment (task 0.2, read in full;
  see the device note above). The only lifetime it offers is a literature figure for a
  different system, InAsP/InP NW dots at "typically ∼1 ns" (p. 2), which is the natural
  starting value but must be labelled as such, never cited as this bench's dot.
- **Timing:** L5 gives the carrier arrival profile g(t) at the dot after a voltage step or
  pulse. The photon-emission density is (g ⊛ Γe^{−Γt})(t). The electrical RC of the contacts
  (from L1 capacitances and a series resistance from L5's I-V) sets g's rise for the G
  device.
- **The static field and optics:** the gate field does not change the optical ε. The
  electro-optic effect is ignored as far smaller than the Stark shift. L1/L5 and FDTD share
  geometry, not fields.

## 4. "Try loss" (item 3 of the owner round in `proposal.md`)

That question (item 3) asked whether the first device depends more on loss or on
inductance. It was written for superconducting wires. For the paper-1 device, inductance does
not matter (no
superconductor, sub-ns electrical edges on a µm-scale device), and the loss that matters is
the **optical loss a conductor part adds to the cavity**. The owner answered "b yes", so this
is **in scope for this change** (its own `optical-loss` capability). The bench takes any
conductor part of the upload, not only a contact; paper 1's gate is one fixture.

1. **Conductor parts (the main term).** A top contact, gate electrode or any other conductor
   next to a mode with Q ≈ 214 (our clean paper-1 ring-down) absorbs light and lowers Q and F.
   The check runs the Purcell bench twice: once without the absorber, once with it, with the
   absorber as a dispersive metal (the same path as the gold mirror) or as a weakly absorbing
   dielectric. The result is ΔQ and ΔF (or ΔF at λ_X). This decides whether a conductor
   placement is a usable design.
2. **Free-carrier absorption in the doped segments (P device).**
   Q_fca ≈ 2πn_g/(λα) = 2π·3.5/(900 nm · α). The design's working range is α = 10–30 cm⁻¹ at
   10¹⁸ cm⁻³, giving Q_fca ≈ 8 × 10³ – 2.4 × 10⁴. **Sourcing status (task 0.2):** the
   literature exists but no free-carrier coefficient could be read out of it in this session —
   Dumke, Lorenz & Pettit, "Intra- and Interband Free-Carrier Absorption … in n-Type InP",
   Phys. Rev. B 1, 4668–4673 (1970), and the mobility/FCA study in J. Appl. Phys. (1980),
   doi:10.1063/1.327925, both report FCA for n-type InP but only behind paywalls, and the
   latter states FCA in InP is limited to **n ≳ 10¹⁷ cm⁻³** by optical-phonon scattering (the
   design's 10¹⁸ cm⁻³ is inside that valid range). What can be stated without a paywalled
   number is the standard n-type scaling α = C·N·λ²: the design's 10–30 cm⁻¹ at 10¹⁸ cm⁻³ and
   0.9 µm implies C ≈ 1.2–3.7 × 10⁻¹⁷, whereas the coefficient usually quoted for InP FCA is
   order 10⁻¹⁸, which would give α ≈ 1–3 cm⁻¹. **So the quoted loss is probably too pessimistic**:
   the literature coefficient gives a *smaller* α, hence a *larger* Q_fca and a *wider* margin
   than the estimate below. The estimate is safe in the direction that matters — Q_fca is not
   overstated.
   Either way the conclusion is unchanged and is stated as an estimate: Q_fca ≈ 8 × 10³ – 2.4
   × 10⁴ is far above 214, so FCA is negligible next to the cavity's own loss. The margin is
   114× at the low end of the range, and FCA only rivals the cavity at α ≈ 1.1 × 10³ cm⁻¹
   (38× the top of the quoted range). Since every plausible reading of C makes α *smaller*, the
   neglect is safe. A measured α should replace this range when one is available; until then the
   coupling prints this estimate rather than running it, and no downstream figure may quote an
   InP α as measured.
3. **Electrical loss** (series resistance, RC) appears only through timing (§3).

This reading was ours, put to the owner as question 3; the owner answered "b yes" on
2026-11-02, which puts it in this change.

## 5. What fits in 3-D, and at what resolution (MEASURED)

**Status (task 1.7, 2026-11-04): the two solve paths are now MEASURED, on a bounded
cluster Job** — sigma-worker2 (x86_64), 2 cpu / 6 GiB, image
`mse-maxwell-solver-gpu:28997f1` with no GPU, code at `b3bbdc1`, running
`dockerfiles/maxwell_solver/solve_memory_bench.py`. The raw job and output are in
`memory-bench.md` beside this file. The Jetson this was originally planned for was
unreachable for the whole row (DNS resolves; 0/2 pings and every SSH path times out
at the banner exchange), so the figures come from the cluster, which is where solves
actually run — **these are the right device's numbers.**

The estimate is `mse_electrostatics.estimate_memory`, and `solve()` refuses a run
over its `budget=` before assembling. Each path is modelled differently because the
measurements say they behave differently:

- **AMG-CG: a power law, `bytes = 12812 * n_free^0.91`**, worst point 2.0 % off over
  `n_free` 1,859..27,869 (~5 kB/unknown at the top of that range). Above the measured
  range the law is applied with a **1.5× margin** (`AMG_EXTRAPOLATION_MARGIN`) and the
  estimate reports `extrapolated: True`, so a refusal says so.
- **Direct LU: interpolated from a table, not a law.** Measured peak-RSS growth:
  1,859 → 10.9 MiB; 6,137 → 32.2 MiB; 14,375 → 122 MiB; 27,869 → 339 MiB; 47,915 → 723 MiB;
  75,809 → 1,444 MiB; 112,847 → 4,797 MiB (the bench's own column: `VmHWM` is KiB,
  printed as `grow_kb / 1024`, so these are MiB and the table stores
  `MiB × 1_048_576`). **No law fits** (a power law is 27 % off its
  own worst point, linear 277 %), and the per-unknown rate climbs from 6,148 B to
  44,574 B (5.86 → 42.5 kB in the rounded terms the fit is discussed in), with the
  largest single jump in that rate (2.2×) between the last two points as the
  ordering changes strategy. State which growth figure you mean: the rate rises
  2.2× over the last step while the total grows 3.3× (1,444 → 4,797 MiB) as
  unknowns grow 1.49×.
  `estimate_memory` therefore interpolates this table in log-log space and **refuses
  above 112,847 unknowns** (see below) rather than extrapolate.

Inputs: wire circumradius R = 239.4 nm, hex area (3√3/2)R² = 1.49 × 10⁵ nm², height
1500 nm, wire volume 2.23 × 10⁸ nm³. Air 500 nm around and above: box side 2(R + 500)
= 1479 nm, box height 12 + 1500 + 500 = 2012 nm, so air 4.15 × 10⁹ nm³. Oxide
1479² × 12 = 2.62 × 10⁷ nm³. Gold 150 nm, 3.28 × 10⁸ nm³.

Nodes per volume at edge h: a regular tet has volume h³/(6√2), so there are 6√2 ≈ 8.49 tets
per h³, and a tet mesh has about 5.5 tets per node, giving **1.54 nodes per h³**. Memory
(measured per §5 above; the tables below are DERIVED from the measured model, not from
a per-node constant):
- Direct sparse LU: interpolate `DIRECT_MEMORY_TABLE`. **It does not extend above
  112,847 unknowns** — a direct solve of the E1/E2 meshes cannot be priced and is
  refused, with the reason and a recommendation to use AMG.
- AMG-preconditioned CG: `12812 * n_free^0.91` bytes, × 1.5 above 27,869 unknowns.
- The container limit is 16 GiB, and the namespace's 18 GiB includes the web app.

### The model, and why the two paths differ (measured on a cluster Job)

The VM run (task 1.7, before the cluster figures) measured the direct path at four
sizes and found the per-unknown rate was **not constant** — it rose from 6.2 kB to
12.9 kB over a fourfold size range. Neither a power law (33.8 % worst point) nor a
linear fit (116.5 %) described it, and the bench correctly printed `usable model:
NONE` rather than quoting one. That result was the reason to measure larger: at VM
sizes SuperLU's fill had not settled into any regular regime.

At cluster scale the answer split in two. **AMG has a law** — the hierarchy grows
regularly with `n`, so a power law fits to 2.0 % over nearly two decades, and it
could safely be extrapolated (with a margin) beyond the fitted range. **Direct does
not** — seven points spanning 60× in unknowns still show a 2.2× jump between the last
two as the ordering changes strategy, which is a discretely different algorithm, not a
curve. So direct is interpolated and then refused, and AMG is the path a mesh above
~10⁵ unknowns must take.

This is why the original plan to measure only on the VM would have produced a wrong
model: four small points were consistent with several incompatible laws, and only a
size range wide enough to cross SuperLU's strategy changes could tell them apart.


### Electrostatics (1 unknown/node)

Every row is priced by the measured model above (`estimate_memory`), not by the old
fill formula. **Direct is refused for all four**: each is above 112,847 unknowns,
where the direct table stops. So AMG is not a fallback here — it is the only path,
and the verdicts below are AMG verdicts.

| mesh | edges (nm) | nodes | direct | AMG-CG | verdict |
|---|---|---|---|---|---|
| E1 graded | wire 20, oxide 6, air/gold 80 | 2.4 × 10⁵ | refused (> table) | 1.41 GiB (extrap) | **fits** |
| E2 graded | wire 10, dot ball (radius 100) 3, oxide 4, air/gold 50 | 1.27 × 10⁶ | refused (> table) | 6.42 GiB (extrap) | **fits with AMG-CG only** |
| E3 uniform (today's one `maxvolume`) | 10 everywhere | 7.3 × 10⁶ | refused (> table) | 31.5 GiB (extrap) | **does not fit** the 16 GiB container |
| E4 uniform | 4 everywhere (what the oxide asks of one size) | 1.1 × 10⁸ | refused (> table) | 372 GiB (extrap) | **does not fit** |

Worked E2: wire (2.23 × 10⁸ − 4.19 × 10⁶)/10³ × 1.54 = 3.4 × 10⁵; dot ball of radius
100 nm, 4.19 × 10⁶ nm³/27 × 1.54 = 2.4 × 10⁵; oxide 2.62 × 10⁷/64 × 1.54 = 6.3 × 10⁵; air
4.15 × 10⁹/50³ × 1.54 = 5.1 × 10⁴; gold 4 × 10³; total 1.27 × 10⁶. AMG:
12812 × (1.27 × 10⁶)^0.91 × 1.5 = 6.42 GiB, 40 % of a 16 GiB container.

So **3-D electrostatics fits, with the graded mesh and AMG-CG** — but with less
headroom than the pre-measurement estimate suggested. E1 has about 1.4 GiB against
16 GiB; E2 about 6.4 GiB, which still fits a 16 GiB container but would NOT fit the
6 GiB Job this model was measured in. **E3, the uniform mesh the current one-size
path would build, does not fit at all** — 31.5 GiB — so the graded sizing field is
now load-bearing for the product, not just an optimisation.

Note the extrapolation: every row above 27,869 unknowns applies the 1.5× margin, so
these are margin-padded figures, and they are the honest ones to plan against — an
under-estimate here is a pod OOM-kill. The gold mirror and the electrodes are held
nodes, so they cost almost nothing to solve.

That 5 × 10⁵ becomes a named constant (`AMG_NODE_CROSSOVER`, task 1.4). It is still
**not** derived from the memory model, but the measurement has changed what can be
said about it:
- The old justification — "a direct LU at 5 × 10⁵ nodes is ~5.8 GiB, about a third of
  the container, so memory does not set the crossover" — rested on the
  nested-dissection fill formula, which the measurements **do not support**. A direct
  solve at that size has never been measured (the table stops at 112,847) and is now
  REFUSED by `estimate_memory`, so no measured figure exists on either side of the
  crossing.
- What the measurement does establish: at every size measured, AMG fits where direct
  is already impractical, and every graded mesh the product builds (E1, E2) is above
  the direct table. So in practice the crossover is **academic for the product's real
  meshes** — they take AMG regardless.
- The crossover is therefore still a SPEED choice with a *named, untested* basis. It
  sits between E1 (2.4 × 10⁵) and E2 (1.27 × 10⁶), and task 1.4's direct-vs-AMG
  agreement test is the only place it is exercised — and that test runs on tiny
  meshes far below the crossover, so it does not constrain it either.

**This is a known weak point, recorded rather than papered over:** the model cannot
price a direct solve above 112,847 unknowns, so the crossover's claim ("direct is
still feasible here, just slower") is unverified by measurement. Revisit it when
task 2.6 has measurements in that band; until then it is a named estimate.

### Stark (3-D effective mass, finite differences, 7-point, about 10 LOBPCG vectors)

| box | grid | unknowns | memory | verdict |
|---|---|---|---|---|
| 60 × 60 × 40 nm | 0.5 nm | 1.2 × 10⁶ | 0.35 GiB (10 vectors × 3 work copies × 8 B = 0.27 GiB, plus the 7-point matrix at 7 × 12 B/unknown = 0.09 GiB) | **fits** (per carrier, per voltage) |
| 480 × 480 × 40 nm (whole cross-section) | 1 nm | 9.2 × 10⁶ | 2.8 GiB (2.06 GiB of vectors + 0.72 GiB of matrix, same rule) | fits, not needed (the confined states decay within ~10 nm of the dot) |

### Drift-diffusion (3 unknowns/node: ψ, n, p; semiconductor and oxide only)

The LU column uses the rule above with k = 3: 9 × 10⁹ × (nodes/10⁶)^{4/3} × 15.6 B, where
nodes = unknowns/3. Example (D3 5/10): 8.3 × 10⁵ nodes → 9 × 10⁹ × 0.784 × 15.6 B = 103 GiB.
A coupled non-symmetric Jacobian may fill worse than this; the column is a lower bound.

The junction needs about two nodes per Debye length. In InP (ε_s 12.5, 300 K),
L_D = √(ε_s ε₀ k_BT/(q²N)) is 42 nm at 10¹⁶ cm⁻³, 13 nm at 10¹⁷ and 4.2 nm at 10¹⁸. The
mesh here has two 100 nm junction slabs across the hexagon. D2 is the (r, z) half-plane of
an equal-area cylinder (R_eq = √(A/π) = 218 nm), about 1.15/h² triangle nodes per nm².

| form | junction / bulk edge (nm) | unknowns | LU | AMG-CG | verdict |
|---|---|---|---|---|---|
| D2 axisymmetric (r, z) | 2 | ≈ 3 × 10⁵ | < 1 GiB (2-D fill ~n log n) | — | **fits** at any doping up to 10¹⁸ |
| D2 axisymmetric | 1 | ≈ 1.1 × 10⁶ | ~1 GiB | — | **fits** |
| D3 3-D | 20 / 30 | 5.7 × 10⁵ | ≥ 14 GiB | 0.8 GiB | **marginal**: direct is near the 16 GiB container; resolves only N ≲ 10¹⁶ |
| D3 3-D | 10 / 20 | 7.7 × 10⁵ | ≥ 21 GiB | 1.1 GiB | direct does not fit |
| D3 3-D | 5 / 10 | 2.5 × 10⁶ | ≥ 103 GiB | 3.5 GiB | direct does not fit (N ≈ 10¹⁷) |
| D3 3-D | 2 / 10 | 1.86 × 10⁷ | ≥ 1.45 TiB | 26 GiB | **does not fit** (N ≈ 10¹⁸) |

Worked D3 2/10: junctions 2 × 100 × 1.49 × 10⁵ = 2.98 × 10⁷ nm³, /8 × 1.54 = 5.75 × 10⁶
nodes; bulk (2.23 × 10⁸ − 2.98 × 10⁷)/10³ × 1.54 = 3.0 × 10⁵; oxide (a 10 nm side shell,
6R × 10 × 1500 = 2.15 × 10⁷ nm³, plus the buffer under the wire, 1.8 × 10⁶ nm³, at a 6 nm
edge) 2.33 × 10⁷/216 × 1.54 = 1.7 × 10⁵; total
6.2 × 10⁶ nodes, 1.86 × 10⁷ unknowns. LU: 9 × 10⁹ × 6.21^{4/3} × 15.6 B = 1.60 × 10¹² B ≈ 1.45 TiB (the table row uses these unrounded figures).

The AMG column assumes an iterative solver converges on coupled drift-diffusion Newton
steps. **DEVSIM appears to offer one** — `solver_type='iterative'` is declared in the API and
`IterativeLinearSolver<double>` is compiled into the 2.11.2 binary for the DC solve, though
the published docs do not document the path and we have never run it (task 0.2, 2026-11-04).
So the capability is likely present but unproven, and what is *not* established — by a wider
margin now — is that it **converges** on this particular coupled Newton system at these sizes;
that is a numerical question for a bench, not an API question. The LU column is therefore still
the safe reading until such a bench exists, and no downstream figure may claim the AMG column
until one does.

**Verdict.** Electrostatics, Stark and coupling fit in full 3-D. Drift-diffusion fits as
axisymmetric (r, z) with junctions resolved, plus a coarse 3-D check (20/30 nm, ≲ 10¹⁶ cm⁻³)
that is marginal on memory. Junction-resolving 3-D drift-diffusion needs about 100 GiB
(10¹⁷ cm⁻³) to about 1.5 TiB (10¹⁸ cm⁻³) for a direct solve, or a GPU or iterative path that
is not established. That is beyond our quota, so 3-D current flow ships **with** the pre-run
size check that refuses a device this change cannot fit (tasks 0.3, "a ok"); we do not raise
quotas. The FDTD runs for contact loss (§4) are the existing paper-1 bench on the Jetson
(tier 2), at its existing cost.

## 6. Decisions

- Static permittivity is its own per-part input (`eps_static`) and is never derived from the
  optical index. For InP the static value is **ε_s = 12.5** and the high-frequency one is
  **ε∞ = 9.61** (Ioffe/NSM "Semiconductors on NSM", `ioffe.ru/SVA/NSM/Semicond/InP/basic.html`,
  basic-parameters table, 300 K); paper 1 independently gives the optical index as n = 3.44
  for wurtzite InP (p. 9), so n² = 11.83. **All three differ**, which is the point: ε_s = 12.5
  is 5.6 % above n² and 30 % above ε∞, so reading any one as another would be wrong — and a
  silent n² substitution would understate the static field. An electrode part may omit it.
  (Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815–5875 (2001), doi:10.1063/1.1368156,
  is the broader III–V parameter compilation if a different alloy is ever needed.)
- Electrodes are parts held at a potential, and electrode charge comes from the assembled
  residual, so the capacitance matrix is symmetric by construction.
- Grading is an opt-in keyword on `tetrahedralize`, a per-part target edge built into a
  `-m` background sizing mesh. The default (None) is byte-for-byte today's path, so the
  `maxwell-solver` spec gets no delta.
- L3 uses its own Cartesian box, not the tet mesh. A uniform FD grid makes the exact checks
  (harmonic, Airy) clean, and LOBPCG on a 7-point operator is simple. The tet mesh's job is
  the field, which is interpolated onto the box.
- `elec` jobs run on the cluster as a second Job shape (CPU only, ≤ 4 CPU / 16 GiB,
  OMP/MKL/OPENBLAS threads = the CPU limit). The Jetson keeps `pc` and `td`. The launcher's
  "one Job shape" sentence in the live `photonic-bands` spec is modified to match.
- Out of scope: strain, floating electrodes, a Purcell spec backfill, a GPU. (Loss is **in**
  scope — tasks 0.3, "b yes" — as the `optical-loss` capability.)
