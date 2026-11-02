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
- [ ] 0.2 Fetch every source marked **(unsure)** before a test quotes it, read every value
      the materials library will ship from its cited source, and correct `design.md` where
      it is wrong: paper 1's dot material, size and lifetime/Γ₀; the effective masses and
      ε_s of every library semiconductor (Vurgaftman 2001); InP n and free-carrier
      absorption at the emission wavelength; the Finley 2004 article number; DEVSIM's
      aarch64 wheels, mesh import route, cylindrical option and linear solvers; Sesame's
      venue and dimensionality. No library number ships from memory.
- [x] 0.3 Questions answered by the owner on 2026-11-02 (Slack), which this revision
      implements:
      1 = **general purpose**: a library of common electrical materials (Si, GaAs, InP,
      SiC, SiO₂, Al₂O₃, HfO₂, Au, Ag, air), any number of parts and contacts, and 1-D,
      2-D and 3-D.
      2 = **all design options**: axisymmetric for round devices (a cylinder form) and
      3-D for everything else.
      3 = **3-D size is our problem**: the solver must refuse a run that does not fit,
      before it starts, and never retry it at a smaller resolution.
      "Loss" was not needed: it is a separate change (`optical-loss`).
      Junction-resolving 3-D memory is therefore NOT an owner question; the solver's own
      array-priced estimate decides (task 2.4).

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

## 2. Phase 2: drift-diffusion (general parts, any contacts, both forms)
- [ ] 2.1 Install DEVSIM in the solver image; a smoke that imports it and records its
      version. Confirm the mesh route (task 0.2).
- [ ] 2.2 The materials library with cited sources per parameter, and the per-part override
      path. Its test checks a sampled subset against the cited sources and refuses an
      uncited entry.
- [ ] 2.3 Axisymmetric (r, z) form with the 2πr weight; the abrupt-junction V_bi, depletion
      width and cylinder-area checks.
- [ ] 2.4 Ideal-diode slope, current conservation, the Debye-length refusal, and the
      multi-terminal check (three doped regions, four ohmic contacts).
- [ ] 2.5 The size check: price the 3-D run from the mesh it would build — node and edge
      counts of the actual geometry through the allocator's array sizes — and refuse
      terminally against the Job's 16 GiB, naming the estimate, the limit and the
      axisymmetric form when the device is round. Two-geometry test proves the estimate is
      the device's own. A native solve is never attempted to find out.
- [ ] 2.6 The roundness check: circumradius/inradius within 2 % for the axisymmetric form,
      measured from the uploaded geometry, refused at upload naming both figures.
- [ ] 2.7 Coarse 3-D form on the tet mesh, admitted only when 2.5's estimate fits; the
      axisymmetric-vs-3-D agreement at ≤ 10¹⁶ cm⁻³ (tier 2 or 3, measure RSS).
- [ ] 2.8 Transient: charge balance and junction capacitance; g(t) at the dot.
- [ ] 2.9 `scripts/dd_bench.py`. Prove the central-difference, dropped-2πr and
      table-priced refusal mutations.

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
- [ ] 4.3 The launcher's electrical Job shape (both forms), the pricing refusal, the `pc`
      refusal test updated for the second shape, and the roundness refusal surfaced at
      upload.
- [ ] 4.4 Result files and the result page; the plotted-points test and its wrong-key
      mutation; a render check of a real finished `elec` job.

## 5. Close
- [ ] 5.1 Reviewer checks each scenario against its test and runs the benches.
- [ ] 5.2 Archive this change.
