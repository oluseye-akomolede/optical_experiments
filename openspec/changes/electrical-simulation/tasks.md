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
      **STATUS 2026-11-05 -- NOT TICKED, deliberately.** Every dot value that
      ships is now either cited with table/page or marked ASSUMED (see
      `dot_material.py`), which is what 1.9c step 4 asked to confirm. The one
      remaining item is the InAs_xP_1-x band gap and bowing parameter: the
      bounded retrieval came back EMPTY (AIP returns 403; four searches found
      nothing), so that number is recorded as NOT SOURCED rather than filled in.
      Ticking 0.2 would assert a fetch that did not happen. It stays open until
      someone with table-level access to a III-V compilation supplies the gap,
      the offset split and the bowing parameter with section and table.
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
      **Count correction (reviewer).** The 1.3 handoff quoted "30 tests / 77
      passed"; those were read before 1.4 added tests to the same file and can no
      longer be reproduced. On the 1.4 tree the file runs 37 and the six affected
      suites 138 passed / 6 skipped. The measurements below are unchanged; only
      the totals moved. (Re-measured at 1.4's completion, after two more tests
      joined the file: `pytest tests/test_electrostatics.py` 39, and
      `pytest tests/test_electrostatics.py tests/test_solver_sizing.py
      tests/test_maxwell_solver.py` 78 passed / 6 skipped. Suite composition
      differs from the reviewer's six, which is why the second figure is not
      140; the three named above are the ones this branch can run locally in
      under a minute.)
      Five mutations, `__pycache__` cleared between mutate and restore each time:
      eps_tet dropped -> red; skip dropped -> survives (above); shared-node refusal dropped
      -> red; eps<1 refusal dropped -> red; electrode exemption dropped -> red (8 failures).
      The first attempt at the interface test had the fixture wrong (electrodes at slabs
      0.083 wide, so the linear answer spans the GAP not the box) and the code was right.
- [x] 1.4 AMG-CG path (`pyamg`) with the checked residual; the direct-vs-AMG agreement test.
      The switch is `AMG_NODE_CROSSOVER` (5 × 10⁵ nodes, `design.md` §5), a named estimate
      with its arithmetic beside it, not a derived limit — the task proves both paths agree
      either side of it, and names it as the line to revisit when 2.6 measures the real
      per-node cost.
      **Done.** `amg_solve` (smoothed aggregation + CG) and `solve(..., method="auto" |
      "direct" | "amg")`. `solve` returns `method` and `residual` too.
      - *The residual is recomputed against `A` and `b`, never read from pyamg's
        `residuals` list.* That list is relative to pyamg's own initial residual, so it
        answers "how much did I improve" while the caller needs "is the answer right".
        Measured consequence: pyamg warns "Indefinite matrix detected in CG, aborting" on a
        reduced operator it dislikes and returns **x = 0 with relative residual 1.0** — a
        silent wrong answer. The checked residual turns that into a refusal, pinned by a test.
      - *`AMG_RTOL` is 1e-12, and it was measured, not chosen.* 1e-10 was the first value;
        the direct path is exact to 1.2e-15 on the contact-chain fixture, so all the
        difference is AMG's, and at 1e-10 AMG's error against a known-linear answer reached
        6.4e-11 — too close to the bound to assert against. At 1e-12 it is 1.9e-13. The
        agreement test's tolerance is `AMG_RTOL` and its docstring says where it came from.
      - *`method="direct"` exists so the crossover can be proven either side.* Tier 1 cannot
        build a 5 × 10⁵-node mesh, so without it the agreement test could only compare AMG
        with direct above the line. An unknown `method` is REFUSED rather than treated as
        `auto` — a silent fallthrough is the same failure as `MSE_SOLVER_BACKEND=cuda`.
      - *Crossover routing tested in both directions.* Below: `auto` is direct. Above:
        `auto` is AMG (by lowering the constant under a small fixture, since the real one is
        unreachable in tier 1). Mutating `>=` to `>` fails the at-the-line test; deleting the
        auto clause fails it; neither red before these tests existed.
      - *Five mutations, `__pycache__` cleared between mutate and restore:* crossover `>=` ->
        `>` red; auto clause deleted red; method validation deleted red; AMG setup replaced
        by plain scipy CG red; checked residual loosened to `rtol * 1e12` red (2 tests).
        The plain-CG one is the one that mattered: it left EVERY other AMG test green,
        because plain CG converges on these fixtures too, so a test spying on
        `smoothed_aggregation_solver` was added to see the preconditioner at all.
      - Reviewer note 1 (1.3): the `skip` guard cannot be reached through `solve`, so it is
        tested by calling `assemble_stiffness` directly rather than dropped — the assembler
        is a public entry point.
- [x] 1.5 `field_at` and `capacitance_matrix` from the residual charge; three-electrode
      symmetry test.
      - Done. Three-electrode stack with two gaps 3x apart, checked against the exact
        series capacitances, plus a two-plate plate value pinned as measured (1.700e-12 F)
        and its mesh independence. `field_at` checked against -V/d with its SIGN asserted,
        and refused both outside the mesh and inside a held electrode.
      - The three-electrode test caught two real bugs and one process failure:
        `_electrode_charge` negated `K phi` and its docstring asserted that negation was
        the charge (a global sign error every symmetric fixture hides), and the
        `_barycentric` constant `c` contracted against a TRANSPOSED gradient, which
        broadcast a tet-by-tet matrix instead of testing any point.
      - The process failure is the one worth recording: the task said "symmetry test", and
        a transposed `C` IS symmetric, so writing `C[j, i]` instead of `C[i, j]` passed
        the whole 45-test suite FOUR times. No mesh in these fixtures can break reciprocity
        (measured max |C - C^T| = 0.0), so the orientation is unpinnable through `solve`.
        Fixed by splitting `assemble_capacitance` out and pinning the convention on
        deliberately non-symmetric charge maps. The task's "symmetry test" wording would
        have shipped the bug; a symmetry property cannot test an index convention.
- [x] 1.6 Bench fixtures (two-layer slab, concentric spheres, sphere between plates) and
      `scripts/electrostatics_bench.py`. Prove the ε_s-ignored mutation.
      **Lead decision 2026-11-04:** the paper-1 wire fixture moved to 1.6b. Its geometry and
      both ε_s values are still marked unsure, and 0.2 says no test quotes an unsure value.
- [x] 1.6b The paper-1 wire with a top electrode, as a bench fixture labelled UNVALIDATED
      until compared with paper 1. Built at `scripts/paper1_wire_electrode.py` with
      `tests/test_paper1_wire_electrode.py` (15 tests). Same placement and shape as its
      sibling `scripts/paper1_nanowire.py`, which is also a standalone bench script, not
      solver-package code: the "not `scripts/`" note in 1.6 history below is about the
      ELECTROSTATICS fixtures (slab/spheres), which are imported by the solver package and
      run in the container. This one is a bench that stands on its own.
      **The fixture's headline result is a NEGATIVE, and it is measured.** The 12 nm oxide
      inside a 1567 nm stack cannot be resolved by any mesh the current one-part path can
      build: tet z-span falls only as ~n^(-1/3), and the SPREAD is wide -- at n_modes 200
      (153k tets) the MEDIAN element spans 20.4 nm in z and the WORST spans 51.7 nm (at
      n_modes 20 the worst is 110.4 nm), so even the median is 1.7x the oxide. |E| at the
      probe moves **40.1 %** and NOT monotonically across n_modes 20/60/120 (557, 780, 690
      kV/m per volt). Tets STRADDLE the layer boundaries -- measured at n_modes 20, 1064 of
      9775 tets (10.9 %), with 555 crossing z=120 nm and 544 crossing z=132 nm -- so the
      per-centroid layer labels are applied to elements whose
      permittivity is not uniform -- that, not the physics, is what moves. **No field value
      from this fixture is quotable**, and the script and its tests say so (`--convergence`
      prints the spread; `test_the_field_is_not_converged_in_the_mesh` pins it, and
      `test_tets_straddle_layer_boundaries` pins the straddle count so the claim cannot
      drift back to "the label is exact").
      **Reviewer correction 2026-11-04 (`qitem-20261104052338`):** two figures here were
      MAX tet z-span presented as if typical, and one docstring in the function claimed
      TetGen puts Steiner points on the facets so a tet "never straddles a boundary" --
      false, and directly contradicting this entry. Both corrected in `c1819ea`; the
      correction is recorded rather than silently edited because the false comment was the
      shape of error the 0.2 round kept finding: a true-sounding statement about the wrong
      thing, in the code a maintainer would read when deciding the label is safe.
      A fifth control (field falls when the top electrode is withdrawn) is **GONE, not
      merely disabled**: it needed two-part input, which wall (a) does not permit. Do not
      rebuild it here. It becomes worth filing when `mse_meshing` gains internal facets or a
      graded electrical upload -- the same condition that unblocks walls (a) and (b).
      Two walls, both measured, neither fixable in this file: (a) `PartSet` refuses touching
      parts and requires one part to enclose all others, so the four layers cannot be four
      parts (a 12 nm oxide cannot enclose a 1375 nm wire); (b) `sizing=` is REFUSED on a
      one-part upload, so there is no graded path for an electrical mesh. A gate needs a
      change to `mse_meshing` (internal facets at the layer planes, or a graded
      electrical upload), not to the fixture.
      **What IS established:** the solver runs end to end on the paper's geometry; the prism
      volume matches the closed form exactly (rel < 1e-9); the solid is watertight, Euler 2;
      the result is exactly linear in the applied volts (max |dE| = 0 at 2V/2 vs 1V and at
      -1V vs 1V); and direct and AMG agree to 1e-5 on this fixture.
      **Geometry departure, reported not hidden:** every layer carries the wire's hexagonal
      outline, so the electrode is a hexagon of the wire's size rather than the square pad
      plus `ELECTRODE_PAD_NM`; `footprint_error()` returns the 2.408x area ratio that says so.
      `--no-top` and `--heights-run` (the two controls in the earlier draft) were REMOVED when
      the geometry became one solid: a single prism has no withdrawable electrode, and
      leaving flags that do nothing would have been a control that is not a control.
      **1.6 history:** Built and proven: two-layer slab,
      concentric spheres, sphere between plates; the bench and its fixtures live in
      `dockerfiles/maxwell_solver/` beside the solver they exercise (not `scripts/`, so they
      import with the rest of the package and run from a container). Fixtures are in
      `mse_bench_fixtures.py`, the runner in `electrostatics_bench.py`, tests in
      `tests/test_electrostatics_bench.py`. NOT done: the paper-1 wire with a top electrode.

      **The ε_s-ignored mutation is proven.** In `element_stiffness`, dropping the
      `eps_tet` factor turns all three scored quantities red (slab 5.86e-01, shells
      capacitance 6.02e-01, shells interface 9.97e-01), and a 1% error in one shell's eps
      scores 8.98e-03 against a 2e-3 tolerance. Both are now tests, not one-off runs.

      **Review round (1.6 fixes).** The reviewer found that `SPHERE_N_R` was chosen
      because it passed, not because it converged, and that the comments were prose
      rather than measurements. Both correct. `--sweep` now runs three tables; reproduce
      them with `python electrostatics_bench.py --sweep` (they are printed, not stored).
      The honest order estimate is the JOINT sweep (both resolutions refined together):
      capacitance error 6.75e-04 (n_r 16, theta/phi 8/16) -> 1.69e-05 (96, 24/48) at an
      observed order of 2.06-2.08, and the scored mesh (n_r=48, 16/32) sits at 7.05e-05.
      Neither one-dimensional table supports an order claim. Radial-only goes 7.05e-05
      (48) -> 4.15e-07 (128), ~170x for a 2.7x change in h: not order 2, almost certainly
      radial against angular error cancelling, so 4e-7 is not the mesh's accuracy.
      Angular-only RISES, 6.09e-06 at 6/12 -> 7.80e-05 at 24/48; 6/12 is an accidental
      cancellation, not a floor, and at the scored 16/32 the angular error is comparable
      to the radial one and of opposite sign. `rel_ideal` is 7.53e-02 at n_r=8 and
      7.22e-02 for n_r >= 16, and never converges: the conductor shell faces are pinned
      at `r_in+t_shell` with `t_shell` independent of n_r, so that gap is geometry, not
      discretisation. It is reported, never scored.
      `TOL_SHELL` is 5e-4 (7x the floor, 3.3x below the weakest 1% perturbation).
      `TOL_PHI_SHELL` is 5e-3 and is a SANITY CHECK, not an eps detector -- all four
      perturbations move phi by only 0.2x-0.4x of it, and a test freezes that so it
      cannot later be claimed as coverage. `perturbation_table()` scores all four
      against references rebuilt from UNPERTURBED eps, closing the circularity that
      made the first bench blind to a wrong eps.
- [x] 1.7 Measure the per-node memory of both solve paths from the bench RSS, replace the
      `design.md` §5 estimates with the measured figures, and wire `budget=` refusals.
      **DONE, measured on the cluster.** The Jetson never answered (DNS resolves
      100.74.195.113; 0/2 pings, ssh dies at the banner exchange), so the lead ran
      `solve_memory_bench.py` as a bounded Job on `sigma-worker2` (2 cpu / 6 GiB, no GPU,
      code at `b3bbdc1`) instead — and the cluster is where solves actually run, so these
      are the right device's numbers. Full provenance in `memory-bench.md`.
      - *Done:* `estimate_memory()`, `check_budget()` and `solve_memory()`, wired into
        `solve(..., budget=)`. The check runs BEFORE `assemble_stiffness`, not before the
        factorisation — assembly is the first big allocation, so a later check would spend
        the memory it protects. `EstimateError` subclasses `MemoryError` (that is the failure
        it stands in for) and the message names the estimate, the budget, the path and the
        "coarser mesh" lever. `budget=None` reads THIS process's admission ceiling
        (`admission_budget()`, at call time), and `budget=NO_DEVICE` skips the check — the
        same three meanings `fdtd.check_limits` uses, re-exported from `fdtd` rather than
        reinvented, because a module running on the control plane while the solve runs
        elsewhere must be able to say "I am not on the run's device" explicitly.
        `web/td_form.py` already passes `NO_DEVICE` for exactly this.
      - *Measured AMG, power law fits:* `bytes = 12812 * n_free^0.91`, worst point 2.0 %
        off over `n_free` 1,859..27,869 (~5 kB/unknown at the top). Above that range the
        law is scaled by `AMG_EXTRAPOLATION_MARGIN` (1.5) and the estimate reports
        `extrapolated: True`, which the refusal message renders as "extrapolated, with a
        1.5x margin applied".
      - *Measured DIRECT, no law fits:* seven points from 1,859 to 112,847 unknowns
        (10.9 MB → 4,797 MB). Power law 27 % off its own worst point, linear 277 %.
        Bytes/unknown climb 5.86k → 42.5k with a **2.2× jump** between the last two
        points — SuperLU switching ordering strategy, an algorithm change rather than a
        curve.
        So direct interpolates `DIRECT_MEMORY_TABLE` in log-log space and **refuses above
        112,847 unknowns** with an `EstimateError` recommending AMG.
      - *Mutations, three, each caught by a different test:* margin 1.5→1.0 (caught by the
        margin test); direct refusal pushed beyond reach (caught by the refusal test); and
        **log-log → linear interpolation, which SURVIVED the first suite** — the
        midpoint-in-the-gap test passes for a straight line too. A curvature test was added
        that compares the log-log value against the linear one at the geometric mean of each
        gap, and now kills it. Recorded because a surviving mutation is the finding, not a
        footnote.
      - *Product consequence, and it changed the story:* every graded mesh the product
        builds (E1 2.4e5, E2 1.27e6) is above the direct table, so **all of them take AMG**;
        direct is refused for all four §5 meshes. E2 prices at 6.42 GiB (40 % of a 16 GiB
        container) — it fits, but does not fit the 6 GiB Job it was measured in. E3 (the
        uniform mesh today's one-size path would build) is **31.5 GiB and does not fit**,
        which makes the graded sizing field load-bearing rather than an optimisation.
      - *A bug the measurement exposed, now fixed:* `AMG_NODE_CROSSOVER` (5e5) sits **above**
        the direct table's top (112,847), so `auto` in that band chose direct and then
        could not price it — refusing a solve AMG could do, for the product's *normal* mesh
        size. `_direct_is_viable()` now consults both limits (the speed crossover AND the
        knowledge ceiling) so they cannot drift, and `auto` takes AMG in the band while
        `method="direct"` still refuses there. A new test covers the band, proven by
        mutating the fix away. The old test sampled at `crossover + 1`, past both limits,
        which is why it passed while the bug lived.
      - *Known weak point:* the crossover itself is still a speed choice with no measurement
        on either side. What changed is that it can no longer cause a false refusal.
      - *Corrected:* the old §5 claim that "a direct LU at 5e5 nodes is ~5.8 GiB, so memory
        does not set the crossover" rested on the nested-dissection fill formula, which the
        measurements do not support. Removed. The VM four-point run that found no usable
        model is kept in §5 as the reason a wider range was needed.

- [x] 1.8 `stark.py`: the 7-point effective-mass operator on a Cartesian box, LOBPCG, the
      field interpolated from the tet solve. Harmonic-oscillator and Airy checks.
      - *Both analytic cases proven by MEASURED CONVERGENCE ORDER, not tolerance:*
        harmonic 2.00 / 1.99 across 160->320->640 cells, Airy 2.00 / 2.00 across
        240->480->960. A tolerance cannot tell a correct second-order operator from a
        wrong one that happens to be close; the order can. `test_stark.py` is 16 tests.
      - *Sign of the Laplacian was wrong on the first attempt* (diagonal `-2t` instead of
        `+2t`), giving a spectrum from -2t to 0: every energy negative, which a kinetic
        operator alone cannot produce. Reproduced against this commit: `main = -2.0 * t` in
        `laplacian_1d_eV_nm2` (the 1-D operator the analytic tests use, NOT the 3-D
        `build_laplacian_eV_nm2`) fails 7 of 16, all four analytic tests among them.
        `test_laplacian_is_positive_semidefinite` catches it in one line.
        *Corrected 2026-11-04:* an earlier version of this note claimed "the analytic
        tests passed anyway -- their tolerances were loose enough". That is false at
        this commit and the reviewer showed it by running the mutation. What I had
        actually observed was a looser intermediate `harmonic` on a coarse box, not the
        committed tests; I generalised a transient observation into a property of the
        record. The committed tests are tight enough to catch the sign error.
      - *Particle-in-a-box formula:* the correct value is `K(pi/L)^2`, where `K` is already
        `hbar^2/2m`, NOT `pi^2 K / (2 L^2)`. The wrong formula gives exactly half, a ratio
        of 2.0000 that does not move with the grid -- which is how it was distinguished
        from a discretisation error (that would converge away). Measured ratio 0.99998.
      - *Parallel-plate field is NOT hit exactly, and that is correct physics.* On the axis
        at 4 um x 4 um x 0.6 um (the gap is 0.6 um; an earlier note said 0.4 um, which was
        wrong) the error is 14.28% / 9.96% / 8.79% at n_modes 8/16/30, converging toward
        -1/d. The evidence for fringing rather than a solver defect is that the error falls
        monotonically under refinement at fixed geometry; a bad operator would not improve.
        *Corrected 2026-11-04:* an earlier note claimed "a 1.2 um box is WORSE (21.85%)" and
        offered that as the decisive signature. It does not reproduce and is withdrawn, not
        explained.
        *Corrected again 2026-11-04:* the replacement pair (1.2 um 8.98% vs 4 um 10.54%, then
        labelled "a fixed 0.2 um gap") was also wrong and is deleted rather than replaced. The
        reviewer found their own helper holds the top and bottom 0.6 um at fixed potential, so
        the gap is extent - 1.2 um and CHANGES with the box. Measured gaps: 1.2 um box ->
        1.08 um, 4 um box -> 3.88 um. A cross-box comparison at "fixed gap" is comparing two
        different geometries.
        *Corrected a third time, same day:* the note that replaced it stood on a refusal, and
        the refusal was for the wrong box. The helper refuses extent_m=1.2e-7 (0.12 um) with
        n_modes 8/16, and that refusal is gone by 0.2 um (solved, gap 80 nm) and at 1.2 um
        (solved, 112/167 electrode tets). "1.2e-7 m is the 1.2 um box" was a factor-of-ten
        error of the same kind as the pair it replaced, and the 4 um gap is 3.88 um, not 2.8.
        Resolved by dropping the refusal sentence entirely, not by re-quoting a boundary.
        Nothing is now claimed here about box size in either direction: the error falls
        monotonically under refinement at fixed geometry, and that is all.
        The test asserts the convergence direction and a bound, not a single value.
      - *V/m -> eV/nm is `* 1e-9` and nothing else.* A volt across a nanometre is 1 eV/nm
        because the electron charge cancels between the volt and the electron-volt.
        Dividing by `e` as well (the obvious-looking mistake) gives a spectrum in the
        millions of eV. End-to-end measured error 1.3e-4.
      - *The field sign is asserted, not discarded.* A sign flip in `field_on_grid` was
        caught by only ONE test while the end-to-end check took `abs()`; it now asserts
        `E_z < 0` before entering the Airy case, and two tests catch the mutation.
      - *A `Box` takes CELLS per axis (`n`), so it has `n + 1` nodes;* `lowest_states_1d`
        takes NODES. Passing one where the other belongs shifts the walls and changes the
        answer without an error.
- [ ] 1.9 First-order Coulomb term, E_X(F), the p/β fit and `scripts/stark_bench.py`. Prove
      its mutations.
      - **Ruling 2026-11-04 (`qitem-20261104102757`, "2 + 3"): cited endpoints, an
        ASSUMED interpolation, and a sweep of the well depth. Docstrings first.**
      - Docstrings fixed: `sawtooth_potential` now states plainly that it adds NO well,
        so the old "+ well" case was a free particle in the box. (The physics defect:
        `<z^2>` came back as 95.27 nm^2, the bare-box `L^2(1/12 - 1/(2 pi^2))`, while
        the docstrings called the carrier a quantum dot.)
      - `well_depth_eV` / `well_height_nm` threaded through `_potential_grid`,
        `_ground_state` and `exciton_energy_eV`; `None` keeps today's behaviour.
      - **Confinement is now measured, not asserted.** `test_a_deep_well_actually_confines_the_state`
        pins that a bare box reproduces `L^2(1/12 - 1/(2 pi^2))` (52.2762 measured vs
        52.2764 analytic for L = 40 nm; 4.7049 vs 4.7049 for L = 12 nm) and that a 0.6 eV
        well narrows `<z^2>` by more than 20x. Mutation: dropping the well makes it fail
        with `<z^2>` equal to the bare-box value, which is the exact trap.
      - **Validation on a known answer (finite square well).** The ground state solves
        `u tan u = sqrt(v0 - u^2)`, `E = u^2 K/(m a^2)`; `test_finite_square_well_recovers_the_bound_state`
        pins it. Measured errors at 127/255/511 cells: 1.13e-2, 5.30e-3, 2.55e-3.
      - **Measured convergence order, which is the point of the proof.** O(h^2) on any
        smooth case -- harmonic 2.04 / 2.14 across 40->80->160->320 cells, reproducing the
        existing pin; transverse box 1.91 / 1.98 / 1.99 / 2.00. The finite square well is
        **O(h)**: 1.16 / 1.09 / 1.05 / 1.04. That is correct behaviour, not a defect: a
        node-sampled step lies between grid points, so the effective well width carries an
        O(h) error. Recorded here so a later reader does not "fix" it.
      - *Trap worth keeping:* a barrier mask whose half-width equals the box half-width
        puts the step on the Dirichlet walls, where the potential value is unused. The
        solver then silently returns the INFINITE-well energy: measured 0.0734434 against
        the true 0.0471547, stable at every resolution, so it looks converged. It is not.
        Any well must be strictly narrower than its box.
      - *Instrument lesson:* three separate alarms during this proof ("E is 12x too
        large", "the potential is ignored", "transverse couplings are missing") were all
        wrong, and each came from comparing numbers produced by DIFFERENT boxes -- a 1-D
        formula applied to a 3-D box, an analytic transverse term from a different z
        extent, an elementwise diff on a degenerate pair. Cross-box subtraction is the
        failure mode here; subtract only within one box family.
      - **`scripts/stark_bench.py`, the depth sweep (ruling item 3).** Prints, for each
        well depth, `E_X(0)`, the electron and hole `<z^2>`, `beta`, and the Stark shift
        at each field. Defaults: depths 0/0.1/0.2/0.3/0.5/0.8 eV, fields 0/0.1/0.2/0.4
        eV/nm, well height 4 nm, box 12x12x16 nm at 6x6x40 cells. Measured:
        `beta` = -10.008, -8.601, -6.989, -5.306, -1.985, +0.882 -- it **changes sign**.
        `<z^2>` for the hole falls 8.364 (bare box, exactly the analytic value) -> 1.377
        -> 1.086 -> 0.987 -> 0.904 -> 0.853 nm^2; the electron 8.364 -> 4.936 -> 3.244
        -> 2.439 -> 1.759 -> 1.391 nm^2.
      - **CONFINEMENT CRITERION, corrected 2026-11-04 (reviewer finding 1).** The first
        version compared `<z^2>` to half the bare-box value, which reads BACKWARDS: at
        depth 0 a free particle sits exactly at the box ceiling, and that rule called it
        confined. Confinement is a LENGTH, so the criterion is now the tight-well ceiling
        `h^2/12` (1.333 nm^2 for the 4 nm well), which depends on the well, not the box.
        Under the corrected rule the HOLE is unconfined at 0 and 0.1 eV only, but the
        ELECTRON is unconfined at EVERY swept depth: rows 0.2-0.8 eV are
        electron-limited, so the exciton extent is set by the box and they are not a
        dot's either. This STRENGTHENS the negative result below.
      - **NON-PERTURBATIVE FIELDS, added 2026-11-04 (reviewer finding 5).** The default
        sweep's largest shift (4.89 eV) exceeds the smallest `E_X(0)` (0.081 eV) by 60x.
        Both carriers are driven up the ramp past the well, which is outside the regime
        the quadratic p/beta fit assumes; the bench now prints a warning saying the
        large-field rows are a model probe. Fields are also reported in kV/cm alongside
        eV/nm (1 eV/nm = 1e4 kV/cm), so `EV_PER_NM_TO_KV_PER_CM` is no longer an unused
        export.
      - **What the sweep concludes, stated plainly in the bench output:** the p/beta fit
        is well-defined at every depth, but `beta` is a property of FOUR assumed numbers,
        not one: the well DEPTH, the well HEIGHT (4 nm assumed; the cited TEM range is
        3-7 nm), the alloy masses (linear interpolation, ASSUMED) and `eps_s` (the InP
        host's, ASSUMED for an InAsP dot). A single beta needs four citations. This is a
        NEGATIVE result about the deliverable's headline number, and it is the honest one.
      - The bench refuses `--height >= --z-extent` with an error naming the
        infinite-well trap, and refuses a field sweep with no zero point.
      - `tests/test_stark_bench.py` (11 tests) pins the sweep's discrimination, the
        bare-box calibration, the refusals, the well-ceiling criterion, the
        non-perturbative warning, and the beta-cell marker. Mutations caught: dropping
        the well from `_potential_grid` fails 3; reverting the criterion to
        `0.5 * bare` fails the ceiling test; never flagging an electron-limited row
        fails the bare-number test; flagging every row fails the held-electron test.
      - **THE BETA COLUMN CARRIES ITS OWN CAVEAT, 2026-11-05 (reviewer CHANGES-NEEDED
        @8c531ad, on the lead's stricter bar).** The emitted table printed `beta` as a
        bare number on every row, including the four where `<z2>e` was above the
        tight-well ceiling the same script computes. The prose saying those rows are
        "not a dot's either" sat sixteen lines BELOW the table -- and the table is what
        gets screenshotted and the number lifted from. Electron-limited rows now print
        `blank*` with a one-line key directly under the table naming the depths. In the
        default sweep every row is electron-limited, so the honest output is that beta
        is blank on all six; the test pins depth 0.2 specifically so a later edit cannot
        restore a bare number there.
      - **1.9c, 2026-11-05: THE OFFSET CITATION CAME BACK EMPTY. Recorded, not
        filled in.** A bounded one-pass retrieval (as the lead specified) for the
        standard III-V compilation -- Vurgaftman, Meyer & Ram-Mohan, J. Appl. Phys.
        89, 5815 (2001) -- returned nothing usable: the AIP landing page answers
        HTTP 403 to automated retrieval, and four searches (the paper by title and
        author; the InP/InAs gap table; the InAsP bowing parameter; a narrowed
        value-based query) returned no hits. **Nothing was written from memory.**
        A control query ("indium phosphide band gap") did return a result, quoting
        ~1.34 eV with no source -- which is exactly the kind of unsourced number
        this change refuses to use, so it was discarded. What is needed to close
        this: table-level access to that paper (or an equivalent compilation), and
        a citation of section and table for each of the InP gap, the InAs gap, the
        InP/InAs offset split, and the InAsP bowing parameter. The sweep remains
        the answer until then, as the lead's step 3 provides.
      - **1.9c: HOW FAR SHORT THE ELECTRON FALLS, measured (the part that does not
        need the missing citations).** The lead's rough figure checks: the 4 nm
        infinite-well electron level at m_e = 0.08 is **0.2938 eV**. Sweeping a
        z-only square well, the electron first satisfies the h^2/12 = 1.333 nm^2
        criterion at a depth just above **0.6 eV** (0.30 and 0.50 eV: loose, 2.139
        and 1.464 nm^2; 0.60 and 0.80 eV: confined, 1.316 and 1.137 nm^2). So a
        realistic conduction offset sits well short of what this model needs, and
        the reviewer's "unbound at every swept depth" is not merely a range
        artifact -- sweeping deeper would not rescue it until ~0.6 eV. Pinned by
        `test_the_electron_needs_an_unrealistically_deep_well_to_bind`, which
        fails if the threshold is shrunk to make the dot bind.
      - **1.9c: NAMING THE LIKELY CAUSE -- measured zero-points, hypothesis
        flagged as hypothesis.** The lead asked that if the electron does not
        bind, the plausible cause be named: the 1-D z model leaves out the 18 nm
        lateral confinement. Measuring the two zero-points at m_e = 0.08 gives
        **0.2938 eV (z, 4 nm) against 0.0145 eV (lateral, 18 nm)** -- the lateral
        term is twenty times smaller. That is a measurement, and it does NOT by
        itself prove the 1-D model overstates the needed depth. **The mechanism
        is a hypothesis, not established here:** it would be that the 1-D model
        assigns the electron's entire zero-point to the z axis and ignores the
        lateral degree of freedom, so z well depth is being asked to supply an
        energy that a real 3-D state partly supplies laterally. What would test
        it: a 3-D solve on this box including the lateral level in the energy
        reference, then comparing the z-depth at which the electron binds.
        Building that model is deliberately OUT OF SCOPE for this row (the lead's
        instruction), and neither the offset nor the mass was tuned to make the
        dot bind.
      - **SILENT LOBPCG FAILURE, fixed 2026-11-04 (reviewer finding 3, made REQUIRED by
        the lead).** `scipy`'s `lobpcg` reports a missed tolerance only through a
        `UserWarning` and then returns whatever it has, so `lowest_states` was handing
        back unconverged eigenvalues with nothing to distinguish them. It happened on
        this change's own sweep (12 of 159 electron solves). `lowest_states` now measures
        the true residual `||Hv - Ev||/(|E|||v||)` per pair and raises `StarkError`
        unless it is below `residual_tol` (default 1e-6), or returns a `converged` array
        with `return_converged=True`. `residual_tol` is deliberately NOT `tol`: `tol` is
        LOBPCG's internal estimate, while the measured residual floors near 5e-10 and
        does not improve with more iterations, so binding the check to `tol` would
        reject correct solves. Tests force failure with `maxiter=3` (raises) and check a
        good solve still passes (does not fire); mutations both ways are caught.
      - **Test counts, corrected 2026-11-04 (reviewer finding 2).** The round's message
        reported "113 passed" as if it were this change's; that was the WHOLE electrical
        suite. This change touches these files: `test_stark.py` 21, `test_stark_bench.py`
        9, `test_exciton.py` 22, `test_paper1_wire_electrode.py` 15 = 67 passing. Both
        numbers are counted from the run output, not transcribed.
      - **The E_C sign, documented 2026-11-04 (reviewer finding 4).** `e_c` is negative
        and correct. `e_e` and `e_h` are energies from a band edge at zero, and the field
        pulls both carriers up the ramp, so each can sit below zero; `e_x` is a SPACING
        between levels, not an absolute energy, and `e_c` is the attractive correction to
        it. Comment added at the call site because the next reader would otherwise read a
        negative attraction as a sign bug.
      - **Finite-square-well convergence order, stated precisely 2026-11-04.** Measured
        1.09 / 1.05 / 1.04 -- FIRST order, and that is correct, not a defect: the
        potential step is sampled at nodes, so the well width the discrete problem solves
        is uncertain by up to one cell and `dE/da` is nonzero. Smooth (harmonic 2.04 /
        2.14) and hard-wall (transverse 1.91-2.00) cases pin the difference. The old
        assertion `min(orders) > 0.9` would have passed at 0.91; it now asserts
        `1.0 < min < max < 1.5`, which pins the order the test documents.
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
