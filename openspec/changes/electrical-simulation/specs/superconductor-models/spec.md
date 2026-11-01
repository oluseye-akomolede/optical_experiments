## ADDED Requirements

### Requirement: The BCS gap SHALL be solved from the weak-coupling gap equation
`superconductor.bcs_gap(T, Tc)` SHALL return Δ(T) in joules by solving the BCS
weak-coupling gap equation numerically, not by an interpolation formula. It SHALL return
Δ(0) = 1.764 k_B·Tc at T = 0, return exactly 0 for T ≥ Tc, and refuse Tc ≤ 0 or T < 0 with
`ValueError`.

#### Scenario: The zero-temperature gap
- **WHEN** `bcs_gap(0, Tc)` is called for any Tc > 0
- **THEN** it returns 1.764 k_B·Tc within 0.1 %

#### Scenario: The gap closes with the Ginzburg-Landau slope
- **WHEN** `bcs_gap(T, Tc)` is called at T/Tc = 0.99
- **THEN** it returns 3.06 k_B·Tc·√(1 − T/Tc) within 2 %

#### Scenario: Above Tc there is no gap
- **WHEN** `bcs_gap(T, Tc)` is called with T ≥ Tc
- **THEN** it returns 0

### Requirement: Complex conductivity SHALL follow Mattis-Bardeen, with two-fluid as a separate model
`superconductor.mattis_bardeen(omega, T, Tc, sigma_n)` SHALL return σ = σ₁ + iσ₂ in the
`exp(-iωt)` convention (σ₂ > 0 inductive). It SHALL evaluate the dirty-limit, local
Mattis-Bardeen integrals (Mattis & Bardeen, Phys. Rev. 111, 412, 1958), including the
pair-breaking term for ħω > 2Δ. `superconductor.two_fluid(omega, T, Tc, sigma_n, lambda0)`
SHALL return σ_n(T/Tc)⁴ + i/(ωμ₀λ(T)²) with λ(T) = λ₀/√(1 − (T/Tc)⁴). Neither function
SHALL silently clamp an input; ω ≤ 0 SHALL raise `ValueError`.

#### Scenario: No absorption below the gap at zero temperature
- **WHEN** `mattis_bardeen` is evaluated at T = 0 for ħω < 2Δ(0)
- **THEN** σ₁ is 0

#### Scenario: The low-frequency superfluid limit
- **WHEN** `mattis_bardeen` is evaluated for ħω ≤ 0.01·Δ(T) at T/Tc ∈ {0.1, 0.5}
- **THEN** σ₂/σ_n equals (πΔ/ħω)·tanh(Δ/2k_BT) within 1 %

#### Scenario: The sum rule holds
- **WHEN** σ₁(ω) − σ_n is integrated over ω at T = 0 from 0 to well above 2Δ/ħ
- **THEN** the missing weight equals (π/2)·σ_n·(πΔ/ħ) (the superfluid δ-function weight,
  Ferrell-Glover-Tinkham) within 2 %

#### Scenario: Normal state above Tc
- **WHEN** either model is evaluated at T ≥ Tc
- **THEN** it returns σ_n with σ₂ = 0

### Requirement: Thin-film surface impedance and kinetic inductance SHALL come from one conductivity
`superconductor.film_impedance(sigma, omega, thickness)` SHALL return the surface impedance
of a film of the given thickness, Z_s = √(−iωμ₀/σ)·coth(t·√(−iωμ₀σ)), taking the square-root
branch with positive real part. It SHALL report it as positive reals R_s and L_s with
Z_s = R_s − iωL_s. `superconductor.kinetic_inductance_sheet(T, Tc, R_sheet)` SHALL return
L_k,□ = ħR_□/(πΔ)·tanh(Δ/2k_BT).

#### Scenario: The London film limit
- **WHEN** `film_impedance` is given σ = i/(ωμ₀λ²) (σ₁ = 0)
- **THEN** R_s is 0 and L_s equals μ₀λ·coth(t/λ) within 1e-9 relative, for t/λ ∈ {0.1, 1, 10}

#### Scenario: Thin dirty film agrees with the sheet formula
- **WHEN** `film_impedance(mattis_bardeen(ω, T, Tc, σ_n), ω, t)` is evaluated with t ≪ λ and ħω ≪ Δ
- **THEN** L_s matches `kinetic_inductance_sheet(T, Tc, 1/(σ_n·t))` within 2 %

#### Scenario: Thick-film limit
- **WHEN** t ≥ 20λ
- **THEN** Z_s equals the bulk √(−iωμ₀/σ) within 1e-6 relative

### Requirement: The superconductor bench SHALL print its checks and exit nonzero on a failure
`scripts/sc_film_bench.py` SHALL run every scenario of this capability at tier 1 (under one
minute, no network, under a 4 GiB memory cap), print one line per check with the computed
value, the reference value, its source and the tolerance, and exit 1 if any check fails.

#### Scenario: A broken model fails the bench
- **WHEN** the 1.764 factor or the tanh term is altered in the implementation
- **THEN** the bench exits 1 and names the failed check
