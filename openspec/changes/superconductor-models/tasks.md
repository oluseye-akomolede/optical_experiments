# Tasks — superconductor-models (not started)

Every local run uses `nice -n 19` and `(ulimit -v 4194304; ...)` with tiny inputs, per the
rig's tier-1 rule. Each numbered group is one reviewable commit.

## 0. Before any code
- [ ] 0.1 Read the "Majorana 2" paper the owner names. Re-scope this change against it:
      which superconductor quantities it actually needs (shell Δ(T), film loss, kinetic
      inductance, or a Bogoliubov-de Gennes model this change does not cover). Revise the
      specs before any code.
- [ ] 0.2 Fetch every source marked **(unsure)** in `design.md` that a test will cite, and
      correct `design.md` where it is wrong.

## 1. superconductor-models (loss first, per the owner's "try loss")
- [ ] 1.1 `superconductor.bcs_gap`: gap-equation solve (brentq over Δ with quad over the
      Debye-cut integral, cutoff ≫ Δ), plus tests for the three gap scenarios.
- [ ] 1.2 `two_fluid` and `mattis_bardeen` (σ₁ with the pair-breaking term, σ₂). Handle the
      integrable endpoint singularities by substitution, not by a tiny epsilon. Add tests
      for below-gap σ₁ = 0, the low-frequency σ₂ limit, the sum rule and normal state.
- [ ] 1.3 `film_impedance` (R_s first) and then `kinetic_inductance_sheet`. Pin the branch
      rule with a test. Add the London-film, thin-dirty-film and thick-film scenarios.
- [ ] 1.4 `scripts/sc_film_bench.py`. Prove the mutation scenario: alter 1.764, then the
      tanh term, confirm exit 1 each time, and revert.

## 2. Close
- [ ] 2.1 Reviewer checks each scenario against its test and runs the bench.
- [ ] 2.2 Archive this change. Write the S4+ proposals from what the Majorana work needs.
