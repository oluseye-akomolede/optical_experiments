# Tasks — electrical-simulation (general-purpose electrical solver; paper 1 is a bench)

Every local run uses `nice -n 19` and `(ulimit -v 4194304; ...)` with tiny inputs, per the
rig's tier-1 rule. Mid-size runs go to the Jetson (tier 2) one at a time; cluster runs are
bounded Jobs in `mode-search-engine` (tier 3). Each numbered item is one reviewable commit.
Every bench check prints its value, reference and tolerance, and each mutation scenario is
proven red once (mutate, confirm exit 1, revert) before the check counts.

## 0. Owner decisions
- [x] 0.1 **SUPERSEDED 2026-11-02 by 0.3 (owner chose a general solver; loss in scope).
      NOT the owner's current answers; kept verbatim only as the record rev2 was built on.**
      Questions sent; the owner answered in Slack on 2026-11-01:
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
      **Resolve this FIRST — it decides whether "a ok" has a cheap path at all:** can
      DEVSIM's custom models carry the 2πr weight on node volumes and edge couplings in a
      cylindrical (r, z) mesh (`design.md` §2)? If yes, record the call with a pointer to
      its docs and release. If no, say so here and in `design.md`, and re-price the
      fallback before any axisymmetric code is written: a junction-resolving 3-D only (about
      100 GiB, §5 — does not fit our quota), or a handwritten 2-D solver (scope this
      change currently does not carry). This is the one (unsure) that a headline owner
      deliverable rests on, so it is not left until the other fetches are done.
      **ANSWERED 2026-11-02: yes, and it does not need a custom model.** DEVSIM 2.11.2
      (PyPI, verified on this VM with `DEVSIM_MATH_LIBS=libblas.so.3:liblapack.so.3`)
      ships `cylindrical_node_volume`, `cylindrical_edge_couple` and
      `cylindrical_surface_area`, which apply the 2πr weight directly, with the radial
      axis set by `raxis_variable` / `raxis_zero`. Measured, not read from the docs: on a
      radial strip `EdgeCouple` keeps its Cartesian values after the call (identical arrays)
      while `CylindricalEdgeCouple` scales with r, so the *coupling* — the term that would
      have been wrong if only volumes were weighted — is genuinely cylindrical, and the
      equations must name the `Cylindrical*` models or they silently solve Cartesian. The cylindrical
      volume differs from the exact annulus by exactly `1 + 1/(9·N_r)` on a uniform grid
      and converges at first order (checked at N_r = 10, 20, 40), so refine toward r = 0.
      Evidence and the residual caveat (topology quirks; a full L5 not yet assembled)
      are in `design.md` §2. The 3-D fallback is therefore not needed for the axisymmetric path.
      - **aarch64 (Jetson): answered, yes.** `pip install devsim==2.11.2` builds and imports
        on the Orin (aarch64, Python 3.10.12) in a venv under `~/mse-bench`, and needs **no**
        `DEVSIM_MATH_LIBS` — the aarch64 wheel bundles its own BLAS ("Skipping libblas.so",
        UMFPACK 5.1 loaded). The cylindrical API then works there unchanged (a 6-node
        axisymmetric strip returned cylindrical node volumes summing to 9.948377). This is
        the opposite of the x86-64 wheel, which does need the system BLAS pointer.
      - **mesh import: answered, yes.** `create_gmsh_mesh(file=…, mesh=…)` then
        `add_gmsh_region(mesh=…, gmsh_name=<physical name>, region=…, material=…)`, then
        `finalize_mesh` / `create_device`. A TetGen tetrahedral mesh loads directly: DEVSIM
        derives its boundary triangles, lines and points from the tets, so no separate
        surface file is needed. A 2-D triangle mesh loads the same way for the axisymmetric
        path, with the cylindrical models applying on top of it. Two reader rules cost time
        and are worth knowing: the physical number must be positive (tag 0 is rejected), and
        the element line is Gmsh 2.2 `id type ntags physical_number geometric_tag nodes…`
        with `ntags = 2`. The reader takes `ilist[3]` as the physical number, so inserting a
        separate element index before it puts each element in the group numbered by its own
        element index and breaks region selection by name without an error.
- [x] 0.3 The owner's answers, in their own words (Slack, 2026-11-02). Only two messages
      were sent: 13:19Z, *"Your questions about p-I-n diodes make me think that you're not
      creating a general-purpose electrical simulation ability for our solver. If you are,
      then please enlighten me."* and 13:33Z, *"General ok a ok b yes"*, answering the
      lead's three options:
      - **general ok** = a general electrical job: any parts and materials, optional
        doping, any number of contacts; the QD analysis is an optional add-on and paper 1
        is validation.
      - **a ok** = 3-D current flow ships with a pre-run size check that refuses runs that
        will not fit the quota, plus a faster 2-D option for round shapes; no quota increase
        is asked for.
      - **b yes** = "try loss" means optical absorption by metal contacts — so loss is
        **in scope for this change**, not deferred.
      Design choices made in service of those answers, and NOT owner statements: the
      shipped materials library and its contents; the axisymmetric form being a cylinder;
      the 3-D estimate being priced from the device's own arrays and never retried coarser
      (task 2.4). These are recorded as design in `design.md`.

## 1. Phase 1: electrostatics and the Stark shift
- [x] 1.1 `tetrahedralize(..., sizing=None)`: a per-part target edge built into a `-m`
      background mesh. The default path stays byte-for-byte today's; the maxwell-solver tests
      stay green. Test that the graded mesh's edge in each part is within its target.
      **Fix (reviewer): the override test never reached the guard.** The first fixture
      (n=1.5 core, 4 modes) gave the core a 0.617 mm optical allowance against a 0.273 mm
      achieved edge, so deleting the override changed nothing and the test still passed 10/10.
      Rebuilt on a fine-optics fixture (n=3.5 core, 16 modes): deleting the override makes the
      guard REFUSE, naming the core at 1.642× its optical allowance (needs 0.1665 mm, achieved
      0.2734 mm); with the override present the same build is ACCEPTED at 1.094× its caller
      allowance (0.25 mm). The test now asserts that inequality against the plan the build
      used, so it fails if the fixture stops reaching the line.
- [x] 1.2 P1 assembly with per-part ε_s; one-tet element matrix against the closed form;
      row sums 0. **Done** (`mse_electrostatics.py`: `check_tet_regions`, `eps_by_tet`,
      `element_stiffness`, `assemble_stiffness`). The material is the TET's, from the mesh's
      `region_of_tet`, not its nodes' — a node-derived part cannot represent a boundary that
      runs through a shared face. `mse_fem.tet_geometry` was extracted to module level and is
      now shared, so the `inv(J).T` trap has one home. Tests use relative tolerance: every
      matrix entry is ~1e-11 (ε₀), i.e. far below `np.allclose`'s default `atol=1e-8`, so an
      absolute floor that large would make every comparison vacuous — both the closed-form and
      assembled-matrix assertions pin `atol=0.0`. (An earlier note here claimed the default
      masked a live "3.5% difference"; that figure was never reproduced and is withdrawn — the
      default is a trap for future mismatches, not a present error.)
      Mutations checked: dropping `eps_tet` fails (a)+(d); dropping the transpose in
      `tet_geometry` fails (a) only, which is why (a) has to exist.
      **Interface physics (lead ruling).** The "solved 2× wrong" finding was a
      layers-swapped exact column, not an assembly bug. Added the test that pins the closed
      form: a 1:2 layered block, interface on a node plane, φ = ε_b/(ε_a+ε_b) = 0.6667 exact
      to 2.1e-15 at every node; parametrised over 1:2, 2:1, 1:4, 4:1 so a swapped fixture lands
      on 1/3 or 1/5. Asserted that no tet spans the interface — that conforming property is
      what makes it exact. Mutations (cache cleared first): swap layers → 0.3333; one ε
      everywhere → 0.5000; both red 4/4, restore green 18/18. The docstring no longer claims
      "the whole of the multi-material physics": exact for a conforming interface, not handled
      for a tet spanning two materials, and `check_tet_regions` does not detect one.
- [x] 1.3 Electrode Dirichlet reduction, direct solve through `mse_backend.factor`, input
      refusals, the maximum-principle test.
      **Done.** `mse_electrostatics.solve(points_m, tets, parts, region_of_tet, eps_static,
      electrodes)` returns `phi`/`free`/`electrodes`/`backend`/`n_free`. A contact is a PART:
      every node of every tet in it is held, including nodes shared with the driven material,
      which is what puts the value at the contact rather than one element away. The reduced
      system is `K_ff phi_f = -K_fe phi_e` solved through `BACKEND.factor(..., spd_lu=SPD_LU)`
      (never `spsolve`), so the backend seam and its refusal behaviour are shared with the
      eigen path. Refusals all precede factorising: no electrode; an electrode that is not a
      part; a non-finite potential; two electrodes sharing a node at different volts; an
      assembled part with no eps_static; eps_static < 1 (the shape of an optical index in a
      permittivity slot). A floating dielectric is solved, not refused. Measured on the
      two-layer block: interface potential exact to 2.2e-16 on a 2-D free-node set; maximum
      principle holds at -2 V / +5 V.
      **`skip` is a refusal contract, not a physics knob.** Measured: assembling the held
      parts' tets versus excluding them changes the matrix by 8.4e-12 over 352 entries and
      changes `phi` by exactly 0.0 — the fixed rows and columns are dropped from `K_ff` and
      the fixed values overwritten either way. The M2 mutation (do not exclude) therefore
      SURVIVES, correctly: it removes nothing the answer can see. `skip` is kept because an
      electrode part has no permittivity to supply, and a test pins the 0.0 agreement so a
      later reader neither deletes it nor defends it with a test that cannot fail.
      Five mutations, `__pycache__` cleared between mutate and restore each time:
      eps_tet dropped -> red; skip dropped -> survives (above); shared-node refusal dropped
      -> red; eps<1 refusal dropped -> red; electrode exemption dropped -> red (8 failures).
      The first attempt at the interface test had the fixture wrong (electrodes at slabs
      0.083 wide, so the linear answer spans the GAP not the box) and the code was right.
- [ ] 1.4 AMG-CG path (`pyamg`) with the checked residual; the direct-vs-AMG agreement test.
      The switch is `AMG_NODE_CROSSOVER` (5 × 10⁵ nodes, `design.md` §5), a named estimate
      with its arithmetic beside it, not a derived limit — the task proves both paths agree
      either side of it, and names it as the line to revisit when 2.6 measures the real
      per-node cost.
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
- [ ] 3.1 The paper-1 bench writes `purcell_spectrum.npz` (via `np.savez`, as the bench already
      does for its other records) from the F it already prints.
- [ ] 3.2 `emitter_coupling.purcell_at`, `rate` and `timing` with their checks, reading back the
      `.npz` 3.1 writes.
- [ ] 3.3 `scripts/contact_loss_bench.py` (`optical-loss`, in scope per "b yes"): it takes any
      named absorber part, not only paper 1's gate. The weak-film inverse-Q check and its
      mutation, the far-conductor check, the two-part check, then the paper-1 wire with and
      without its contact as one fixture (tier 2, one run at a time). Report ΔQ, ΔF and the
      Q_fca estimate; skip Q_fca when the device has no doped segment.

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
