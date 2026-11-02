# Design — electrical-simulation

Revised on 2026-11-01 after the owner's answers (`tasks.md` 0.1), and again on 2026-11-02
after the owner widened it to a general-purpose solver (`tasks.md` 0.3). The superconductor
track moved, unchanged, to the separate change `superconductor-models`; "loss" moved to
`optical-loss`.

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
its check fail. Citations marked **(unsure)** are from memory. They must be checked against
the source before any code or test quotes a number from them, and the task that adds such a
test fetches the source first. **Every compute figure here is an estimate** until a bench
measures it. The materials library obeys the same rule with no exceptions: a library value
ships only after it is read from its cited source (task 0.2, 2.2).

## 0. The device and what exists

**Device (owner Q1 = c, Q5 = paper 1 only).** This is the paper-1 nanowire quantum dot
(arXiv 2601.04440) made electrically triggerable. The scene follows `scripts/paper1_nanowire.py`:
a hexagonal InP prism about 1.5 µm tall (the bench scans 1375 to 1525 nm), with a circumradius
of about 240 nm (420 nm × 1.14 / 2, so about 415 nm across the flats). It stands on a 12 nm
SiO₂ buffer over a gold mirror. The paper-1 dot's material and size **(unsure: InAsP in InP?
fetch from the paper, task 0.2)**. Paper 1 has no electrical contact, so one is added:

- **gate (G):** the gold mirror is the back electrode (it is already a conductor under the
  buffer), plus a top electrode. The wire is undoped. The field at the dot is set by V_G.
  Electrostatics (Q1) alone gives the field.
- **p-i-n (P):** a p-doped base, an intrinsic segment holding the dot and an n-doped top. One
  contact reaches the base through the oxide, and one sits on the top. This needs
  drift-diffusion (Q5).

The owner's Q2 answer, "full electrical analysis", means both: G for the Stark sweep, and P
for I-V and injection timing. Which contact the real device uses is owner question 2.

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

## 1. Physics ladder (device c)

| rung | model | gives | phase |
|---|---|---|---|
| Q1 | 3-D electrostatics ∇·(ε₀ε_s∇φ) = 0, fixed-potential electrodes, no free carriers | field at the dot vs V_G; capacitances | 1 |
| Q2 | perturbative Stark ΔE = −pF + βF² | E_X(V) from a fitted p, β | 1 (fit of Q3) |
| Q3 | 3-D single-band effective mass (electron, heavy hole) in the Q1 field + first-order Coulomb | E_X(F), overlap, p, β, dipole orientation | 1 |
| Q5 | drift-diffusion + Poisson + SRH/radiative recombination (DEVSIM) | I-V, the field at the dot under bias, carrier arrival after a pulse | 2 |
| C  | coupling: F at E_X(V) from a Purcell spectrum; Γ = F·Γ₀·(overlap ratio); timing = g ⊛ Γe^{−Γt} | radiative rate and photon-timing distribution vs V | 3 |
| L  | contact loss: the same Purcell run with and without the contact metal | ΔQ, ΔF from the contact ("try loss") | 3 |
| Q4 | Schrödinger-Poisson | screening by charge in or near the dot | not planned |
| Q6 | multi-band k.p + strain | fine structure, light-hole mixing | not planned |

Why these and not more:
- Undoped G needs only Q1. The dot's response is, to leading order, a dipole and a
  polarisability (Q2). Measured curves are fitted to that form (Fry et al., Phys. Rev. Lett.
  84, 733 (2000); Finley et al., Phys. Rev. B 70, 201308(R) (2004) **(unsure of article
  number)**).
- Q3 predicts p and β from geometry instead of quoting them. The owner asked for full 3-D
  (Q4 = full 3D), and a hexagonal wire is not axisymmetric, so Q3 is 3-D. It runs on a uniform
  Cartesian finite-difference box around the dot, with the Q1 field interpolated onto it.
  At 0.5 nm a 60 × 60 × 40 nm box is about 1.2 × 10⁶ unknowns, about 0.35 GiB (estimate; §5).
  Material parameters come from Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys. 89, 5815 (2001).
- Q5 is the p-i-n half of "full electrical analysis", as in the first electrically driven
  single-photon source (Yuan et al., Science 295, 102 (2002)). Full 3-D Q5 does not fit at
  junction-resolving resolution (§5), so Q5 has two forms. The **axisymmetric** (r, z) form
  uses an equal-area cylinder for the hexagon and resolves the junctions. The **coarse 3-D**
  form runs on the tet mesh as a cross-check, and is valid only where the Debye length is
  resolved.
- Q4 and Q6 are left out. One exciton in an undoped segment needs no self-consistency.
  Leading-order selection rules put the heavy-hole dipole in the plane, perpendicular to the
  wire, which is the x/y dipole the paper-1 bench already drives. Reference for later:
  Stier, Grundmann & Bimberg, Phys. Rev. B 59, 5688 (1999).

## 2. Drift-diffusion: why DEVSIM

| option | 3-D | open | fit for us |
|---|---|---|---|
| **DEVSIM** (Sanchez, J. Open Source Softw. 7, 3898 (2022); devsim.org) | yes, tetrahedra | Apache-2.0, pip wheels for x86-64 Linux | Scharfetter-Gummel, SRH, transient, Python API. **(unsure: aarch64 wheels for the Jetson; how to hand it a TetGen mesh, `create_gmsh_mesh` with arrays or a written `.msh`; whether it has a cylindrical 2-D option; which linear solvers it ships: direct only, or an iterative one)** |
| Sesame (NIST; Gaury et al. 2019 **(unsure of venue)**) | no (1-D/2-D, **unsure**) | open | no 3-D cross-check |
| own SG on scikit-fem / FEniCSx | yes | open | we would write and validate the SG stabilisation and Newton ourselves: the most risk |
| Charon (Sandia, Trilinos) | yes | open | heavy C++/MPI build, more than one Job needs |
| nextnano, Sentaurus, Silvaco | yes | commercial | a reference, not a dependency |

DEVSIM is the only open option that does 3-D tets with a Python API and the standard SG
discretisation. Its open questions (above) are task 0.2 items. If DEVSIM cannot take our tet
mesh, the fallback is to write a Gmsh `.msh` file. If it has no cylindrical option, the
axisymmetric form is a 2-D (r, z) mesh whose node volumes and edge couplings carry the 2πr
weight, which DEVSIM's custom models allow **(unsure)**.

Cluster first. DEVSIM's x86-64 wheels run on the cluster workers. The Jetson is only a
fallback, and only if aarch64 wheels exist.

## 3. Coupling into the Purcell runs

- **Energy:** Q3 (or Q2's fit) gives E_X(V). The coupling reads a Purcell spectrum, which is
  pairs of wavelength (nm) and F from a paper-1 bench run, and interpolates F at
  λ_X(V) = hc/E_X(V). A λ_X outside the spectrum's range is refused, not extrapolated. The
  bench gains a small writer for this file (a task, not a spec change to unspecced code).
- **Orientation:** in-plane (x/y) heavy-hole dipole. The Purcell spectrum must come from a
  run with the same dipole orientation, and the file records it.
- **Strength:** Γ₀ = n·ω³·|d|²/(3πε₀ħc³) for a dipole d in a homogeneous medium of index n.
  In the field, the oscillator strength scales by |⟨ψ_e|ψ_h⟩|²(V)/|⟨ψ_e|ψ_h⟩|²(0) (from Q3).
  Then Γ(V) = F(λ_X(V))·Γ₀·(overlap ratio). The zero-field d is an input (from the paper or
  a measurement) **(unsure: does paper 1 state a lifetime or Γ₀? fetch)**.
- **Timing:** Q5 gives the carrier arrival profile g(t) at the dot after a voltage step or
  pulse. The photon-emission density is (g ⊛ Γe^{−Γt})(t). The electrical RC of the contacts
  (from Q1 capacitances and a series resistance from Q5's I-V) sets g's rise for the G
  device.
- **The static field and optics:** the gate field does not change the optical ε. The
  electro-optic effect is ignored as far smaller than the Stark shift. Q1/Q5 and FDTD share
  geometry, not fields.

## 4. "Try loss" (owner Q6) for this device

Q6 asked whether the first device depends more on loss or on inductance. That question was
written for superconducting wires. For device c, inductance does not matter (no
superconductor, sub-ns electrical edges on a µm-scale device), and the loss that matters is
the **optical loss the electrical contacts add to the cavity**:

1. **Contact metal (the main term).** A top contact or gate electrode next to a mode with
   Q ≈ 214 (our clean paper-1 ring-down) absorbs light and lowers Q and F. The check uses the
   existing paper-1 Purcell bench twice: once without the contact, once with it, with the
   contact as a dispersive metal, the same path as the gold mirror. The result is ΔQ and
   ΔF (or ΔF at λ_X). This decides whether the contact is a usable design.
2. **Free-carrier absorption in the doped segments (P device).**
   Q_fca ≈ 2πn_g/(λα) = 2π·3.5/(900 nm · α). With α = 10–30 cm⁻¹ at 10¹⁸ cm⁻³
   **(unsure: fetch an InP free-carrier absorption value)**, Q_fca ≈ 8 × 10³ – 2.4 × 10⁴.
   That is far above 214, so it is negligible next to the cavity's own loss (estimate). The
   coupling prints this estimate rather than running it.
3. **Electrical loss** (series resistance, RC) appears only through timing (§3).

This reading is ours. The proposal asks the owner to confirm it (owner question 3).

## 5. Pricing "full 3-D" (estimates)

Inputs: wire circumradius R = 239.4 nm, hex area (3√3/2)R² = 1.49 × 10⁵ nm², height
1500 nm, wire volume 2.23 × 10⁸ nm³. Air 500 nm around and above: box side 2(R + 500)
= 1479 nm, box height 12 + 1500 + 500 = 2012 nm, so air 4.15 × 10⁹ nm³. Oxide
1479² × 12 = 2.62 × 10⁷ nm³. Gold 150 nm, 3.28 × 10⁸ nm³.

Nodes per volume at edge h: a regular tet has volume h³/(6√2), so there are 6√2 ≈ 8.49 tets
per h³, and a tet mesh has about 5.5 tets per node, giving **1.54 nodes per h³**. Memory (all
estimates, to be replaced by task 2.6's measured per-node cost):
- Direct sparse LU in 3-D (nested dissection): fill ≈ 10⁹ entries × (n/10⁶)^{4/3}, where n
  is the NODE count, at 12 B/entry × 1.3 overhead (15.6 B/entry). For a k-unknown-per-node
  system, multiply the fill by k² (electrostatics k = 1; drift-diffusion k = 3, so 9 × 10⁹).
- AMG-preconditioned CG: about 1.5 kB per unknown.
- The container limit is 16 GiB, and the namespace's 18 GiB includes the web app.

### Electrostatics (1 unknown/node)

| mesh | edges (nm) | nodes | LU | AMG-CG | verdict |
|---|---|---|---|---|---|
| E1 graded | wire 20, oxide 6, air/gold 80 | 2.4 × 10⁵ | ≥ 2.2 GiB | 0.34 GiB | **fits** (direct) |
| E2 graded | wire 10, dot ball (radius 100) 3, oxide 4, air/gold 50 | 1.27 × 10⁶ | ≥ 20 GiB | 1.8 GiB | **fits with AMG-CG only** |
| E3 uniform (today's one `maxvolume`) | 10 everywhere | 7.3 × 10⁶ | ≥ 205 GiB | 10 GiB | AMG barely, slow; not used |
| E4 uniform | 4 everywhere (what the oxide asks of one size) | 1.1 × 10⁸ | — | 159 GiB | **does not fit** |

Worked E2: wire (2.23 × 10⁸ − 4.19 × 10⁶)/10³ × 1.54 = 3.4 × 10⁵; dot ball of radius
100 nm, 4.19 × 10⁶ nm³/27 × 1.54 = 2.4 × 10⁵; oxide 2.62 × 10⁷/64 × 1.54 = 6.3 × 10⁵; air
4.15 × 10⁹/50³ × 1.54 = 5.1 × 10⁴; gold 4 × 10³; total 1.27 × 10⁶. LU:
10⁹ × 1.27^{4/3} × 15.6 B = 21 GB ≈ 20 GiB. AMG: 1.27 × 10⁶ × 1.5 kB = 1.8 GiB.

So **3-D electrostatics fits**, but only with the graded mesh (the `-m` sizing field). It
needs AMG-CG (`pyamg`) once it is above about 5 × 10⁵ nodes. The gold mirror and the
electrodes are held nodes, so they cost almost nothing to solve.

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
steps. That is not routine, and whether DEVSIM offers one is (unsure). The LU column is the
safer reading.

**Verdict.** Electrostatics, Stark and coupling fit in full 3-D. Drift-diffusion fits as
axisymmetric (r, z) with junctions resolved, plus a coarse 3-D check (20/30 nm, ≲ 10¹⁶ cm⁻³)
that is marginal on memory. Junction-resolving 3-D drift-diffusion needs about 100 GiB
(10¹⁷ cm⁻³) to about 1.5 TiB (10¹⁸ cm⁻³) for a direct solve, or a GPU or iterative path that
is not established. That is beyond our quota and becomes owner question 1; we do not raise
quotas. The FDTD runs for contact loss (§4) are the existing paper-1 bench on the Jetson
(tier 2), at its existing cost.

## 6. Decisions

- Static permittivity is its own per-part input (`eps_static`) and is never derived from the
  optical index. For InP that is about 12.5 static vs n² ≈ 11.8 at 900 nm **(unsure: fetch
  both)**, so a silent n² would be wrong. An electrode part may omit it.
- Electrodes are parts held at a potential, and electrode charge comes from the assembled
  residual, so the capacitance matrix is symmetric by construction.
- Grading is an opt-in keyword on `tetrahedralize`, a per-part target edge built into a
  `-m` background sizing mesh. The default (None) is byte-for-byte today's path, so the
  `maxwell-solver` spec gets no delta.
- Q3 uses its own Cartesian box, not the tet mesh. A uniform FD grid makes the exact checks
  (harmonic, Airy) clean, and LOBPCG on a 7-point operator is simple. The tet mesh's job is
  the field, which is interpolated onto the box.
- `elec` jobs run on the cluster as a second Job shape (CPU only, ≤ 4 CPU / 16 GiB,
  OMP/MKL/OPENBLAS threads = the CPU limit). The Jetson keeps `pc` and `td`. The launcher's
  "one Job shape" sentence in the live `photonic-bands` spec is modified to match.
- Out of scope: Q4, Q6, strain, floating electrodes, a Purcell spec backfill, a GPU.
</content>
</invoke>
