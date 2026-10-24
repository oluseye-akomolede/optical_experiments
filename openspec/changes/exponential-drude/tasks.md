# exponential-drude tasks

## 1. Spec

- [x] 1.1 `openspec/changes/exponential-drude/specs/time-domain/spec.md`:
      add the requirement that the Drude pole steps as an exact decay
- [x] 1.2 `proposal.md`

## 2. Solver

- [x] 2.1 `lorentz_drude_coefficients`: tagged records; the `w0 = 0` pole
      returns `("drude", exp(-G dt), sigma (1 - a)/G)`
- [x] 2.2 `LorentzDrude.prepare`: step the Drude current in place and add
      `dt J^{n+1}` to the E update; Lorentz poles unchanged
- [x] 2.3 `lorentz_drude_growth`: model the Drude record as a first-order
      state; the pole block's root is `a < 1`
- [x] 2.4 `stable_dt` contract unchanged (accept the step or return it)
- [x] 2.5 rework: the E row subtracts the step contribution once
      (`(d^{n+1} - d^n)`), not its second difference; Rakic gold passed
      1.0379 / 1.0634 under the old row and passes now

## 3. Proof

- [x] 3.1 `test_the_drude_pole_steps_as_an_exact_decay`: seven library metals
      x five steps; `a == exp(-G dt)`, `0 < a < 1`, falling with dt,
      `b == sigma (1 - a)/G`
- [x] 3.2 `test_the_growth_check_passes_the_runs_own_steps`: Run I's and Run
      K's own dt and finest spacings, read as at most `1 + GROWTH_TOLERANCE`
      (the exact answer at S = 0 is 1), not as strictly below 1
- [x] 3.3 `test_rakic_gold_passes_the_growth_check_on_the_paper_grids`:
      Rakic Au, Au-jc and Au-palik on both grids; fails on 1880992 (Rakic
      1.0379 / 1.0634), passes after the E-row fix
- [x] 3.4 Correct `test_a_metal_shrinks_dt_only_as_far_as_it_must`'s comment:
      the tolerance covers the S = 0 neutral root's rounding (at most 1.4e-11
      measured), not 6e-8 of structural residue
- [x] 3.5 `tests/test_fdtd_metals.py` 37 passed
- [x] 3.6 `tests/test_fdtd_source_eps.py`, `tests/test_phasor_mirror.py`,
      `tests/test_fdtd_bench_v4.py` 28 passed

## 4. Open

- [x] 4.1 Characterise the fixture's metal + open-boundary growth: it was a
      small-box artefact. At 24 cells of lateral interior against a 7-cell PML,
      Au-jc grew +1.31e-3 per step in amplitude; at 40 and 56 cells, same
      spacings and metal, it decays -9.8e-5 in both. No reproduction of Run K
      with this fixture (proposal, "What this does not establish")
- [ ] 4.2 Find a fixture that reproduces Run K's growth. Needs Run K's own
      geometry/plan (the lead has it); the metal fit is not sufficient by
      itself -- both golds behave the same in every box tried here
