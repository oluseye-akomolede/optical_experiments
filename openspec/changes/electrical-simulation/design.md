# Design — electrical-simulation

The rule throughout: start with the simplest model that gives a useful number. Check it
against an answer we did not compute ourselves, and move up a rung only when an owner
question needs it. Citations marked **(unsure)** are from memory and must be checked
against the source before any code or test quotes a number from them. A task that adds a
test against such a number fetches the source first.

## 0. What exists and what it implies

- `mse_meshing.tetrahedralize` returns `(points_m, tets, region_of_tet)` for a multi-part
  `.obj`, so a per-part material is already a mesh label. A P1 (nodal) Poisson solver can
  reuse this mesh unchanged. `mse_fem.py` assembles Nédélec edge elements; nodal P1 is the
  simpler sibling.
- `mse_backend.factor` already routes SPD sparse factorisations (CPU `splu`, optional GPU
  with a checked residual). A Poisson matrix is SPD, so it can go through the same factory.
- `fdtd.py` steps Lorentz-Drude metals with ADEs. The Drude pole uses the exact-decay
  update `J ← e^{-GΔt}J + σ(1-e^{-GΔt})/G·E` (spec: "The Drude pole SHALL be stepped as an
  exact decay"). A London superfluid is a Drude pole with `G = 0` and `ω_p = c/λ_L`, so it
  fits the ADE machinery. But the formula as written divides by `G`, so Phase 2 must use
  the `G → 0` limit `σΔt` (an `expm1` form).
- The Purcell path (`delivered_power_spectrum`, `purcell_spectrum`) gives `F(ω)` across a
  pulse's band from one run. A Stark-tuned emitter therefore needs no new FDTD run, only a
  lookup of `F` at the tuned frequency, as long as the tuning stays inside the band.
- Compute (`deployment-bounds`): one Job, ≤ 4 CPU / 16 GiB per container, 5 CPU / 18 GiB in
  the namespace, GPU 0 unless the owner grants it. The Jetson Orin bench (tier 2) is the
  default for mid-size runs.

## 1. Superconducting wires

### Physics ladder

| rung | model | gives | fits where |
|---|---|---|---|
| S1 | BCS gap Δ(T); London λ_L; two-fluid σ(ω,T) = σ_n(T/T_c)⁴ + i/(ωμ₀λ²(T)) | λ(T), kinetic inductance, crude R_s | anywhere, closed form |
| S2 | Mattis-Bardeen σ₁, σ₂(ω,T), dirty local limit | R_s, X_s and L_k at any ω < and > 2Δ/ħ; BCS R_s(T) | anywhere, 1-D integrals |
| S3 | thin-film impedance Z_s = √(-iωμ₀/σ)·coth(t√(-iωμ₀σ)) (exp(-iωt)) | film R_s, L_s for t ≲ λ | anywhere, closed form |
| S4 | surface impedance in the EM solvers | cavity Q and Δf (FEM perturbation); THz response (FDTD) | existing solvers |
| S5 | 2-D cross-section London/MB current distribution | L_k and L_geo of a strip or CPW with real current crowding | scikit-fem or own P1, small |
| S6 | time-dependent Ginzburg-Landau (TDGL), 2-D film | critical current, vortex entry, phase slips, bends | pyTDGL, short sections only |
| S7 | lumped electrothermal detector model | reset time, latching | ODE, trivial |

Why this order:
- S1–S3 answer the most common design questions: what is L_k, and what are the losses?
  They are also inputs to every later rung.
- Two-fluid is the cheapest model, but its loss term is phenomenological. Mattis-Bardeen is
  the accepted model for dirty films such as NbN, NbTiN and TiN, so S2 replaces two-fluid for
  any number we report. Two-fluid stays as the form FDTD can step (S4).
- S4 reuses the solvers we already trust. The existing FEM eigen solve is real symmetric
  (`eigsh`), so we do not make it complex. Instead we use first-order cavity perturbation:
  `Q_s⁻¹ = R_s∮|H_t|²dA / (ωμ₀∫|H|²dV)`, plus a matching frequency shift from X_s. This is
  valid when the penetration depth is much smaller than the cavity, which holds for any bulk
  superconductor.
- TDGL is the only rung that predicts a critical current with geometry: bends, constrictions
  and vortex entry in strips wider than ~4.4ξ. It is also by far the most expensive (see
  compute).
- Optical note: at the QD's optical photon energies (~1.3 eV) ħω ≫ 2Δ (meV), so a
  superconducting film absorbs light as its normal state. An on-chip detector's optical
  absorption is therefore an FDTD run with a normal-state ε(ω) fit (the existing metal-model
  path), not a superconducting model.

### Tools

| rung | candidates (open source first) | we write |
|---|---|---|
| S1–S3 | none needed (NumPy/SciPy quad); cross-check against `scqubits`? **(unsure if it exposes MB)** | `superconductor.py` (Phase 1) |
| S4 FEM | existing `mse_fem` / `mse_eigen` modes + `mse_fields` sampling | surface-integral post-process |
| S4 FDTD | existing `fdtd.Metal` / `LorentzDrude` | a two-fluid Metal and the G = 0 limit |
| S5 | scikit-fem (Gustafsson & McBain, JOSS 5, 2369, 2020); FEniCSx/DOLFINx (heavier, MPI; Baratta et al. 2023 **(unsure of citation)**); own P1 from Phase 1 | 2-D London equation ∇²A − A/λ² = −μ₀J solve |
| S6 | pyTDGL (L. Bishop-Van Horn, Comput. Phys. Commun. 291, 108799, 2023; py-tdgl.readthedocs.io) | device builder from an `.obj` cross-section, IV driver |
| S7 | none needed | Kerman-style electrothermal ODE |

### Validation (the answer key)

- **Gap.** Δ(0) = 1.764 k_BT_c (weak coupling). Near T_c, Δ ≈ 3.06 k_BT_c√(1−T/T_c).
  Δ(T)/Δ(0) against Mühlschlegel, Z. Phys. 155, 313 (1959), table **(values to fetch)**.
- **Mattis-Bardeen.** Mattis & Bardeen, Phys. Rev. 111, 412 (1958). At T = 0, σ₁ = 0 for
  ħω < 2Δ, and σ₂/σ_n has the closed form in complete elliptic integrals (also in
  Zmuidzinas, Annu. Rev. Condens. Matter Phys. 3, 169 (2012)). For ħω ≪ Δ,
  σ₂/σ_n → (πΔ/ħω)·tanh(Δ/2k_BT). Ferrell-Glover-Tinkham sum rule: the σ₁ weight lost below
  the gap equals the superfluid weight (Tinkham, *Introduction to Superconductivity*, 2nd
  ed., 1996).
- **Kinetic inductance.** L_k,□ = ħR_□/(πΔ)·tanh(Δ/2k_BT), the standard low-frequency MB
  limit, as used for NbN nanowires by Annunziata et al., Nanotechnology 21, 445202 (2010)
  **(unsure of page/article number)**. London film: L_s = μ₀λ·coth(t/λ), which tends to
  μ₀λ²/t for t ≪ λ.
- **S4 cavity.** A pillbox TM₀₁₀ has an analytic geometry factor G (Q = G/R_s). The value
  for a pillbox with length = radius is often quoted as ≈ 257 Ω (Padamsee, Knobloch & Hays,
  *RF Superconductivity for Accelerators*) **(unsure: recompute G analytically rather than
  quote it)**.
- **S5 strip.** Thin-strip kinetic inductance, Meservey & Tedrow, J. Appl. Phys. 40, 2028
  (1969). CPW inductances, Clem, J. Appl. Phys. 113, 013910 (2013).
- **S6 TDGL.** For a strip narrower than ~4.4ξ, the critical current must approach the GL
  depairing current J_d = Φ₀/(3√3πμ₀λ²ξ) (Tinkham §4.4). For dirty-limit corrections at
  T ≪ T_c: Kupriyanov & Lukichev, Sov. J. Low Temp. Phys. 6, 210 (1980) **(unsure)**. Also
  reproduce one example from pyTDGL's own documentation.
- **S7.** Reset time τ = L_k/R_load: Kerman et al., Appl. Phys. Lett. 88, 111116 (2006).
  Latching behaviour: Kerman et al., Phys. Rev. B 79, 100509(R) (2009).

### Compute (estimates, to be measured)

- S1–S3, S7: milliseconds, tier 1.
- S4: the cost of the existing eigen solve plus one surface integral. Fits.
- S5: 2-D, about 10⁴–10⁵ nodes. Fits tier 1–2.
- S6: a 100 nm × 1 µm NbN strip with ξ ≈ 5 nm and edges of ~ξ/2 is about 2×10⁴ sites.
  One bias point is minutes to about an hour on 4 CPU (estimate), and an I-V curve of
  30 points is hours: one Job or the Jetson. **Does not fit:** a full detector meander of
  ~10 µm × 10 µm is ~10⁷ sites, and pyTDGL is 2-D only, so there is no 3-D TDGL. We simulate
  one straight section and one bend and treat the meander as series sections.
- **Does not fit:** FDTD of a microwave resonator. At dx = 10 nm the Courant step is about
  2×10⁻¹⁷ s, and a 5 GHz period is 2×10⁻¹⁰ s, so ~10⁷ steps per period. Microwave work is
  frequency-domain only (S4, S5).

## 2. Electrically triggered quantum dots

### Physics ladder

| rung | model | gives | fits where |
|---|---|---|---|
| Q1 | electrostatics, ∇·(ε₀ε_s∇φ) = −ρ, fixed-potential electrodes, no free carriers | field at the QD vs gate voltage, capacitances | Phase 1, 3-D P1 |
| Q2 | perturbative Stark: ΔE = −pF + βF² with p, β from literature or Q3 | emission energy vs voltage | trivial |
| Q3 | single-band effective-mass Schrödinger, electron and heavy hole separately, plus a Coulomb correction; axisymmetric (r,z) for a nanowire dot | p, β, overlap, emission energy | 2-D, small |
| Q4 | Schrödinger-Poisson, self-consistent | screening when the dot or nearby wells hold charge | Q3 × 10–50 iterations |
| Q5 | drift-diffusion + Poisson (+ SRH/radiative recombination) | p-i-n I-V, carrier arrival at the dot, RC and transit timing | DEVSIM, 1-D/2-D |
| Q6 | multi-band k.p (+ strain) | fine-structure splitting, light-hole mixing, tilted dipoles | only if justified |

Why this order:
- An undoped gated device needs only Q1. The dot's response to a field is, to first order,
  a permanent dipole and a polarisability (Q2). Measured Stark curves are usually fitted to
  that form (Fry et al., Phys. Rev. Lett. 84, 733 (2000); Finley et al., Phys. Rev. B 70,
  201308(R) (2004) **(unsure of article number)**).
- Q3 predicts p and β from geometry instead of quoting them. A nanowire dot is close to a
  disk in a cylinder, so an axisymmetric solve is natural and ~100× cheaper than 3-D.
  Material parameters come from Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815
  (2001).
- Q4 only when free charge matters (a charged dot, a doped region near the dot). One
  exciton in an undoped region does not need it.
- Q5 is needed for p-i-n triggering, as in the first electrically driven single-photon
  source (Yuan et al., Science 295, 102 (2002)). It gives when carriers arrive and how long
  the electrical edge is.
- Q6 is justified only if an owner question needs polarisation or fine-structure detail.
  Leading-order selection rules already say the heavy-hole exciton dipole lies in the
  plane, perpendicular to the wire axis. That is the orientation the paper-1 FDTD bench
  already drives. Strain (InAsP in InP) would add a continuum-elasticity solve. Reference
  for 8-band k.p in dots: Stier, Grundmann & Bimberg, Phys. Rev. B 59, 5688 (1999).

### Coupling into the FDTD/Purcell path

- **Energy:** Q2/Q3 give E_X(V). The paper-1 pulse run already gives F(ω) over its band, so
  F at the tuned energy is a lookup. If the tuning leaves the band, the run is refused, not
  extrapolated.
- **Orientation:** in-plane dipole from heavy-hole selection rules (Q3 or Q6 if mixing
  matters). The existing x/y dipole sources cover this.
- **Field in the cavity:** the gate's static field does not change the optical ε (we ignore
  the electro-optic effect, which is far smaller than the Stark shift). Q1 and FDTD share
  geometry, not fields.
- **Timing:** FDTD gives the radiative rate Γ = F·Γ₀. Q5 gives the injection or capture
  profile g(t). The photon emission time distribution is g convolved with Γe^{−Γt}, so
  jitter follows from the two. This is a rate-equation layer, cheap and our own.
- **The gold mirror as an electrode:** the paper-1 back mirror can be the Q1 ground plane
  (assumption: confirm with the owner's device).

### Tools

| rung | candidates | we write |
|---|---|---|
| Q1 | own P1 on the TetGen mesh (Phase 1); scikit-fem; FEniCSx | `electrostatics.py` |
| Q3/Q4 | scikit-fem (axisymmetric weak form); own finite differences; nextnano is the commercial reference (Birner et al., IEEE TED 54, 2137 (2007) **(unsure)**), not a dependency | effective-mass solver, Coulomb correction |
| Q5 | DEVSIM (Sanchez, J. Open Source Softw. 7, 3898 (2022); devsim.org) **(unsure of aarch64 wheels: check for the Jetson)**; Sesame (Gaury et al., 2019, NIST) **(unsure of venue)** | device builder, timing extraction |
| transport (not planned) | Kwant (Groth et al., New J. Phys. 16, 063065 (2014)). It is for quantum transport and tunnelling spectroscopy, not dot optics, and is listed so we don't reach for it by default | — |

### Validation

- **Q1:** a two-layer parallel-plate capacitor (series formula, exact for P1); concentric
  spheres C = 4πε₀ε_s·ab/(b−a); a dielectric sphere in a uniform field, E_in = 3E₀/(ε_s+2)
  (Jackson, *Classical Electrodynamics*, §4.4).
- **Q3:** an infinite well in a field has an exact Airy-function answer. A harmonic dot
  shifts by exactly −q²F²/(2m*ω²). QCSE in wells: Miller et al., Phys. Rev. B 32, 1043
  (1985); variational well-in-field results, Bastard et al., Phys. Rev. B 28, 3241 (1983).
  Then reproduce a published dot Stark curve (Fry 2000 or Finley 2004) within its stated
  geometry.
- **Q5:** the built-in voltage V_bi = (k_BT/q)·ln(N_AN_D/n_i²), the depletion width and the
  Shockley ideal-diode I-V for an abrupt junction (Sze & Ng, *Physics of Semiconductor
  Devices*, 3rd ed., 2007); DEVSIM's own diode example.

### Compute (estimates, to be measured)

- Q1 3-D: P1 with ≤ ~3×10⁵ nodes factorises through `mse_backend` within a 16 GiB Job.
  About 10⁶ nodes needs CG with an AMG preconditioner (pyamg, a new dependency) instead of
  `splu`. Phase 1 caps nodes and refuses above the cap.
- Q3 axisymmetric: about 10⁵ unknowns, seconds. 3-D single-band at 0.5 nm over
  60 × 60 × 40 nm is ~10⁶ unknowns: LOBPCG or a Job, not shift-invert on the VM.
- **Does not fit:** 3-D 8-band k.p at that resolution (~10⁷ complex unknowns). Use
  axisymmetric k.p or a coarser grid, or the owner grants a GPU.
- Q5: 1-D and 2-D axisymmetric are cheap. 3-D DEVSIM at ~10⁵ nodes × 3 unknowns is a Job.

## 3. Phase 1 scope and its decisions

Phase 1 builds S1–S3 and Q1 only. They are the foundations both tracks need, they are
pure numerics with analytic answers, and none needs the web app or more than tier 1.

- `superconductor.py` uses the `exp(-iωt)` convention of `fdtd.py`, so σ₂ > 0 is
  inductive. It reports R_s and L_s as positive reals (Z_s = R_s − iωL_s) so that callers
  never handle the sign.
- Static permittivity is its own per-part input (`eps_static`) and is never derived from
  the optical index. Static and optical values differ (for GaAs, about 12.9 static vs
  about 11 at 1 µm), so a silent `n²` would be wrong.
- Electrodes are parts held at a potential. Their tets are conductor, so every node of an
  electrode part is fixed. Electrode charge is read from the assembled residual at those
  nodes (the exact discrete charge), so the capacitance matrix is symmetric by
  construction.
- Mesh: reuse `tetrahedralize` for its region labels. Its edge target is driven by Weyl's
  law for modes. If that cannot be replaced by a plain edge-length target without touching
  the eigen path, task 2.1 adds a keyword that defaults to today's behaviour, leaving the
  `maxwell-solver` spec unchanged.
- Out of Phase 1: free charge, doping, floating electrodes, upload/job kinds, the GPU.
