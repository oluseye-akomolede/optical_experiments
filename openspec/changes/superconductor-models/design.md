# Design — superconductor-models

Moved from `electrical-simulation/design.md` §1 (as reviewed at fb4d3ee) on 2026-11-01.
The spec.md is byte-identical to that review. design.md, proposal.md and tasks.md carry
UNREVIEWED edits made in the move: task 1.3 puts surface resistance (loss) first; the S-step
list is renumbered (S4); "What exists" drops the electrostatics half. Diff against
`git show fb4d3ee:openspec/changes/electrical-simulation/design.md` to see them; review them
when this change starts. The rule throughout: start with the simplest model that gives a useful
number, check it against an answer we did not compute ourselves, and move up a rung only when
an owner question needs it. Citations marked **(unsure)** are from memory and must be checked
against the source before any code or test quotes a number from them.

## What exists and what it implies

- `fdtd.py` steps Lorentz-Drude metals with ADEs. The Drude pole uses the exact-decay
  update `J ← e^{-GΔt}J + σ(1-e^{-GΔt})/G·E` (spec: "The Drude pole SHALL be stepped as an
  exact decay"). A London superfluid is a Drude pole with `G = 0` and `ω_p = c/λ_L`, so it
  fits the ADE machinery. But the formula as written divides by `G`, so S4 must use the
  `G → 0` limit `σΔt` (an `expm1` form).
- The existing FEM eigen solve is real symmetric (`eigsh`), so S4 uses perturbation, not a
  complex eigen solve.

## Physics ladder

| rung | model | gives | fits where |
|---|---|---|---|
| S1 | BCS gap Δ(T); London λ_L; two-fluid σ(ω,T) = σ_n(T/T_c)⁴ + i/(ωμ₀λ²(T)) | λ(T), kinetic inductance, crude R_s | anywhere, closed form |
| S2 | Mattis-Bardeen σ₁, σ₂(ω,T), dirty local limit | R_s, X_s and L_k at any ω < and > 2Δ/ħ; BCS R_s(T) | anywhere, 1-D integrals |
| S3 | thin-film impedance Z_s = √(-iωμ₀/σ)·coth(t√(-iωμ₀σ)) (exp(-iωt)) | film R_s, L_s for t ≲ λ | anywhere, closed form |
| S4 | surface impedance in the EM solvers | cavity Q and Δf (FEM perturbation); THz response (FDTD) | existing solvers |
| S5 | 2-D cross-section London/MB current distribution | L_k and L_geo of a strip or CPW with real current crowding | scikit-fem or the electrostatics P1 assembly, small |
| S6 | time-dependent Ginzburg-Landau (TDGL), 2-D film | critical current, vortex entry, phase slips, bends | pyTDGL, short sections only |
| S7 | lumped electrothermal detector model | reset time, latching | ODE, trivial |

Order (owner's Q6 answer, "try loss"): S2 σ₁ → S3 R_s → S4 cavity Q come before the
kinetic-inductance outputs. S1 is an input to all of them.

- Two-fluid is the cheapest model, but its loss term is phenomenological. Mattis-Bardeen is
  the accepted model for dirty films such as NbN, NbTiN and TiN, so S2 replaces two-fluid for
  any number we report. Two-fluid stays as the form FDTD can step (S4).
- S4 uses first-order cavity perturbation: `Q_s⁻¹ = R_s∮|H_t|²dA / (ωμ₀∫|H|²dV)`, plus a
  matching frequency shift from X_s, valid when the penetration depth is much smaller than
  the cavity.
- TDGL is the only rung that predicts a critical current with geometry, and by far the most
  expensive.
- Optical note: at a QD's optical photon energies (~1.3 eV) ħω ≫ 2Δ (meV), so a
  superconducting film absorbs light as its normal state; that is an FDTD run with a
  normal-state ε(ω) fit, not a superconducting model.

## Tools

| rung | candidates (open source first) | we write |
|---|---|---|
| S1–S3 | none needed (NumPy/SciPy quad); cross-check against `scqubits`? **(unsure if it exposes MB)** | `superconductor.py` |
| S4 FEM | existing `mse_fem` / `mse_eigen` modes + `mse_fields` sampling | surface-integral post-process |
| S4 FDTD | existing `fdtd.Metal` / `LorentzDrude` | a two-fluid Metal and the G = 0 limit |
| S5 | scikit-fem (Gustafsson & McBain, JOSS 5, 2369, 2020); FEniCSx/DOLFINx (Baratta et al. 2023 **(unsure of citation)**) | 2-D London equation ∇²A − A/λ² = −μ₀J solve |
| S6 | pyTDGL (L. Bishop-Van Horn, Comput. Phys. Commun. 291, 108799, 2023) | device builder from an `.obj` cross-section, IV driver |
| S7 | none needed | Kerman-style electrothermal ODE |
| Majorana (not planned here) | Kwant (Groth et al., New J. Phys. 16, 063065 (2014)) | re-scope after task 0.1 |

## Validation (the answer key)

- **Gap.** Δ(0) = 1.764 k_BT_c (weak coupling). Near T_c, Δ ≈ 3.06 k_BT_c√(1−T/T_c).
  Δ(T)/Δ(0) against Mühlschlegel, Z. Phys. 155, 313 (1959), table **(values to fetch)**.
- **Mattis-Bardeen.** Mattis & Bardeen, Phys. Rev. 111, 412 (1958). At T = 0, σ₁ = 0 for
  ħω < 2Δ, and σ₂/σ_n has the closed form in complete elliptic integrals (also in
  Zmuidzinas, Annu. Rev. Condens. Matter Phys. 3, 169 (2012)). For ħω ≪ Δ,
  σ₂/σ_n → (πΔ/ħω)·tanh(Δ/2k_BT). Ferrell-Glover-Tinkham sum rule (Tinkham, *Introduction
  to Superconductivity*, 2nd ed., 1996).
- **Kinetic inductance.** L_k,□ = ħR_□/(πΔ)·tanh(Δ/2k_BT); Annunziata et al.,
  Nanotechnology 21, 445202 (2010) **(unsure of article number)**. London film:
  L_s = μ₀λ·coth(t/λ) → μ₀λ²/t for t ≪ λ.
- **S4 cavity.** Pillbox TM₀₁₀ geometry factor G (Q = G/R_s), often quoted ≈ 257 Ω for
  length = radius (Padamsee, Knobloch & Hays) **(unsure: recompute G analytically rather than
  quote it)**.
- **S5 strip.** Meservey & Tedrow, J. Appl. Phys. 40, 2028 (1969); Clem, J. Appl. Phys. 113,
  013910 (2013).
- **S6 TDGL.** Depairing current J_d = Φ₀/(3√3πμ₀λ²ξ) (Tinkham §4.4); Kupriyanov &
  Lukichev, Sov. J. Low Temp. Phys. 6, 210 (1980) **(unsure)**; one pyTDGL documentation
  example.
- **S7.** τ = L_k/R_load: Kerman et al., Appl. Phys. Lett. 88, 111116 (2006); latching,
  Kerman et al., Phys. Rev. B 79, 100509(R) (2009).

## Compute (estimates, to be measured)

- S1–S3, S7: milliseconds, tier 1. S4: the existing eigen solve plus one surface integral.
  S5: 2-D, about 10⁴–10⁵ nodes, tier 1–2.
- S6: a 100 nm × 1 µm NbN strip (ξ ≈ 5 nm, edges ~ξ/2) is about 2×10⁴ sites; one bias point
  minutes to an hour on 4 CPU, an I-V of 30 points hours (estimate). **Does not fit:** a full
  ~10 µm × 10 µm meander (~10⁷ sites); pyTDGL is 2-D only.
- **Does not fit:** FDTD of a microwave resonator (dx = 10 nm → Δt ≈ 2×10⁻¹⁷ s; a 5 GHz
  period is ~10⁷ steps). Microwave work is frequency-domain only (S4, S5).

## Decisions

- `superconductor.py` uses the `exp(-iωt)` convention of `fdtd.py`, so σ₂ > 0 is
  inductive. It reports R_s and L_s as positive reals (Z_s = R_s − iωL_s).
