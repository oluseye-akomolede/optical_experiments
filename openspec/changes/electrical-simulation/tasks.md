# Tasks — electrical-simulation (device c: the electrically triggered paper-1 QD)

Every local run uses `nice -n 19` and `(ulimit -v 4194304; ...)` with tiny inputs, per the
rig's tier-1 rule. Mid-size runs go to the Jetson (tier 2) one at a time; cluster runs are
bounded Jobs in `mode-search-engine` (tier 3). Each numbered item is one reviewable commit.
Every bench check prints its value, reference and tolerance, and each mutation scenario is
proven red once (mutate, confirm exit 1, revert) before the check counts.

## 0. Owner decisions
- [x] 0.1 Questions sent; the owner answered in Slack on 2026-11-01:
      1 = **c**: the first device is the paper-1 nanowire QD with a gate or p-i-n contact.
      2 = **full electrical analysis**: electrostatics, the Stark shift vs voltage and p-i-n
      injection / I-V (drift-diffusion), with emission energy, dipole and timing feeding the
      existing Purcell runs.
      3 = **show results in the web app**.
      4 = **full 3D**.
      5 = **just paper 1**: "We'll be investigating the Majorana 2 paper next."
      6 = **"I'm not sure, try loss"**.
      These moved Phase 1, so the specs were revised before any code (this revision).
      Superconductors moved to the not-started change `superconductor-models`.
- [ ] 0.2 Fetch every source marked **(unsure)** before a test quotes it, and correct
      `design.md` where it is wrong: paper 1's dot material, size and lifetime/Γ₀; InP and
      dot effective masses and ε_s (Vurgaftman 2001); InP n at the emission wavelength; an
      InP free-carrier absorption value; the Finley 2004 article number; DEVSIM's aarch64
      wheels, mesh import route, cylindrical option and linear solvers; Sesame's venue and
      dimensionality.
- [ ] 0.3 Ask the owner the three questions at the end of `proposal.md` (memory for
      junction-resolving 3-D drift-diffusion; which contact; our reading of "loss"). Record
      the answers here. Answer 2 decides whether Phase 2 is built at all.

## 1. Phase 1: electrostatics and the Stark shift
- [ ] 1.1 `tetrahedralize(..., sizing=None)`: a per-part target edge built into a `-m`
      background mesh. The default path stays byte-for-byte today's; the maxwell-solver tests
      stay green. Test that the graded mesh's edge in each part is within its target.
- [ ] 1.2 P1 assembly with per-part ε_s; one-tet element matrix against the closed form;
      row sums 0.
- [ ] 1.3 Electrode Dirichlet reduction, direct solve through `mse_backend.factor`, input
      refusals, the maximum-principle test.
- [ ] 1.4 AMG-CG path (`pyamg`) with the checked residual; the direct-vs-AMG agreement test.
- [ ] 1.5 `field_at` and `capacitance_matrix` from the residual charge; three-electrode
      symmetry test.
- [ ] 1.6 Bench fixtures (two-layer slab, concentric spheres, sphere between plates, the
      paper-1 wire with a top electrode) and `scripts/electrostatics_bench.py`. Prove the
      ε_s-ignored mutation.
- [ ] 1.7 Measure the per-node memory of both solve paths from the bench RSS, replace the
      `design.md` §5 estimates with the measured figures, and wire `budget=` refusals.
- [ ] 1.8 `stark.py`: the 7-point effective-mass operator on a Cartesian box, LOBPCG, the
      field interpolated from the tet solve. Harmonic-oscillator and Airy checks.
- [ ] 1.9 First-order Coulomb term, E_X(F), the p/β fit and `scripts/stark_bench.py`. Prove
      its mutations.
- [ ] 1.10 The paper-1 gate sweep (tier 2): E_X(V) for the wire, recorded with its mesh
      counts and RSS.

## 2. Phase 2: p-i-n drift-diffusion (only if owner answer 0.3-2 includes p-i-n)
- [ ] 2.1 Install DEVSIM in the solver image; a smoke that imports it and records its
      version. Confirm the mesh route (task 0.2).
- [ ] 2.2 Axisymmetric (r, z) form with the 2πr weight; the abrupt-junction V_bi, depletion
      width and cylinder-area checks.
- [ ] 2.3 Ideal-diode slope, current conservation and the Debye-length refusal.
- [ ] 2.4 Coarse 3-D form on the tet mesh; the axisymmetric-vs-3-D agreement at
      ≤ 10¹⁶ cm⁻³ (tier 2 or 3, measure RSS).
- [ ] 2.5 Transient: charge balance and junction capacitance; g(t) at the dot.
- [ ] 2.6 `scripts/dd_bench.py`. Prove the central-difference and dropped-2πr mutations.

## 3. Phase 3: coupling into the Purcell runs, and loss
- [ ] 3.1 The paper-1 bench writes `purcell_spectrum.json` from the F it already prints.
- [ ] 3.2 `emitter_coupling.purcell_at`, `rate` and `timing` with their checks.
- [ ] 3.3 `scripts/contact_loss_bench.py`: the weak-film inverse-Q check and its mutation,
      the far-contact check, then the paper-1 wire with and without the contact (tier 2,
      one run at a time). Report ΔQ, ΔF and the Q_fca estimate.

## 4. Phase 4: web app
- [ ] 4.1 `elecs` column, the `elec` kind in the app's and worker's kind lists, the request
      keys and refusals.
- [ ] 4.2 The upload arm and the posted-equals-read test.
- [ ] 4.3 The launcher's electrical Job shape, the pricing refusal, the `pc` refusal test
      updated for the second shape.
- [ ] 4.4 Result files and the result page; the plotted-points test and its wrong-key
      mutation; a render check of a real finished `elec` job.

## 5. Close
- [ ] 5.1 Reviewer checks each scenario against its test and runs the benches.
- [ ] 5.2 Archive this change.
