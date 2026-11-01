# Tasks — electrical-simulation (Phase 1 only)

Every local run uses `nice -n 19` and `(ulimit -v 4194304; ...)` with tiny inputs, per the
rig's tier-1 rule. Each numbered group is one reviewable commit.

## 0. Owner decisions
- [ ] 0.1 Send the owner the questions at the end of `proposal.md`. Record the answers in
      this change. If an answer moves Phase 1 (for example "THz FDTD first"), revise the
      specs before any code.
- [ ] 0.2 Fetch every source marked **(unsure)** that a Phase 1 test will cite, and correct
      `design.md` where it is wrong.

## 1. superconductor-models
- [ ] 1.1 `superconductor.bcs_gap`: gap-equation solve (brentq over Δ with quad over the
      Debye-cut integral, cutoff ≫ Δ), plus tests for the three gap scenarios.
- [ ] 1.2 `two_fluid` and `mattis_bardeen` (σ₁ with the pair-breaking term, σ₂). Handle the
      integrable endpoint singularities by substitution, not by a tiny epsilon. Add tests
      for below-gap σ₁ = 0, the low-frequency σ₂ limit, the sum rule and normal state.
- [ ] 1.3 `film_impedance` and `kinetic_inductance_sheet`. Pin the branch rule with a test.
      Add the London-film, thin-dirty-film and thick-film scenarios.
- [ ] 1.4 `scripts/sc_film_bench.py`. Prove the mutation scenario: alter 1.764, then the
      tanh term, confirm exit 1 each time, and revert.

## 2. electrostatics
- [ ] 2.1 Read `mse_meshing.tetrahedralize`. If its size target is mode-driven, add a
      keyword size override that defaults to today's behaviour. The maxwell-solver tests
      stay green, and that spec gets no delta.
- [ ] 2.2 P1 assembly with per-part ε_s. Unit test: one tetrahedron's element matrix against
      the closed form, and row sums equal to 0.
- [ ] 2.3 Dirichlet reduction for electrode parts, solve through `mse_backend.factor`, plus
      the input refusals. Test the maximum principle.
- [ ] 2.4 `field_at` and `capacitance_matrix` from the residual charge. Test symmetry with
      three electrodes.
- [ ] 2.5 Bench geometries as small `.obj` fixtures: two-layer slab, concentric spheres,
      sphere between plates. Add `scripts/electrostatics_bench.py` with the peak-RSS line.
      Prove the mutation (ε_s ignored → exit 1).
- [ ] 2.6 Measure the per-node memory cost from the bench RSS and record it in the module. Add
      `budget=None` (defaulting to `admission_budget()`) and a test that a passed small budget
      is refused before factorisation.

## 3. Close Phase 1
- [ ] 3.1 Reviewer checks each scenario against its test and runs both benches.
- [ ] 3.2 Archive this change. Write the Phase 2 and Phase 3 proposals from the owner's
      answers.
