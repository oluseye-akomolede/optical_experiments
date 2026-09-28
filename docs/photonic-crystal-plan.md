# Photonics feature 2: 2-D photonic crystal band diagrams — plan

*Row `qitem-20260928175549-cd34cc4b`. Source: `docs/photonics-roadmap.md` §3.1.
Feature 1 (multiple materials) is live. **PLAN ONLY** — no code until the lead
approves.*

**Revision 2.** The first version had three errors, all caught by the lead's
review: the TM/TE operator assignment was **reversed**, the ε-placement control
was one **that cannot fail**, and the Bloch problem was described as real
non-symmetric when it is **complex Hermitian**. Each is corrected below, and the
correction is stated where it applies rather than quietly applied.

**Revision 3.** Three further corrections, all from the lead's second review,
and all measured rather than argued here:

1. **The s/p labels were crossed again** (revision 2 called the `E_z` field "s
   (TE)"). The mapping is **TM ↔ s, TE ↔ p**, derived in §4 from `ẑ · x̂ = 0`.
2. **The p admittance is `k_x/ε`, not `k_x/ε²`.** The oracle now **validates
   itself** against an independent ODE shooting solve before any mesh is
   compared to it, at **ε₂ = 4** — never ε = 1, where the two forms coincide and
   the check cannot fail. Measured: **6.86e-12** (`rtol = 1e-13`) against a
   1e-10 bar, with the `1/ε²` form rejected at **3.71**.
3. **1e-6 is not achievable on a P1 mesh.** Only `λ = 0` at Γ is exact; the rest
   is stated as `O(h²)` convergence with a pre-stated bound and an observed rate.
   The 1e-10 sits on the oracle, not the mesh.

## Scope of v1

An infinite 2-D lattice of circular rods or holes: square and triangular, index
`n_rod` in `n_bg`, scalar TE and TM polarisations, Bloch-periodic unit cell,
band diagram along the irreducible-zone path (Γ–X–M–Γ square; Γ–M–K–Γ
triangular), gaps shaded, gap-midgap ratio in %. Input is a **form** — lattice,
`r/a`, the two indices, number of bands. No `.obj` upload.

---

## 1. Formulation

In 2-D, with the lattice in the x–y plane and invariance along z, Maxwell
splits into two scalar problems. Write the Bloch state `ψ(r) = e^{i k·r} u_k(r)`
with `u_k` cell-periodic. The two polarisations are **named for the field
component that is out of plane**:

**TM — `E = E_z ẑ`** (so `ψ = E_z`). With ε scalar and μ = 1,
`curl curl E = (ω/c)² ε E` and `curl curl E = −∇²E_z ẑ`, giving
```
−∇²ψ = (ω/c)² ε ψ
```
so TM has a **plain stiffness** and **ε on the mass**.

**TE — `H = H_z ẑ`** (so `ψ = H_z`; the in-plane E is what is usually plotted).
The H-field equation `∇·((1/ε)∇H_z) = −(ω/c)² H_z` gives
```
−∇·( (1/ε) ∇ψ ) = (ω/c)² ψ
```
so TE has **1/ε on the stiffness** and a **plain mass**.

| | operator `A` | mass `B` |
|---|---|---|
| **TM** (`E_z`) | `∫ ∇u·∇v` (plain) | `∫ ε_r u v` |
| **TE** (`H_z`) | `∫ (1/ε_r) ∇u·∇v` | `∫ u v` (identity) |

> **This table is the reverse of revision 1's, and the reversal was the review's
> first finding.** The physical signature: dielectric rods in air show their
> large 1–2 gap under the **TM** label, and MPB's own tutorial says the
> triangular rod lattice has "a gap for TM light, but not for TE light"
> (§3 below). Revision 1 would have drawn that gap under the TE label.

**Boundary treatment: periodic node pairing with a phase.** Pair the boundary
nodes and impose `u(r + L) = e^{i k·L} u(r)`, eliminating the slave nodes.

> **Corrected from revision 1.** The phase is **complex at almost every point of
> the band path**, not only inside the zone: on Γ→X, `k = (tπ/a, 0)` and the
> phase is `e^{i t π}`, which is real only at `t = 0` and `t = 1`. A previous
> draft called this a real non-symmetric problem; it is not — it is **complex
> Hermitian**. Splitting into real and imaginary parts gives a real **symmetric**
> problem of twice the size with **every eigenvalue doubled** (verified
> numerically in this session), so that is not a cheaper route either.

The linear algebra is therefore: assemble the complex Hermitian generalised pair
`(A, B)` per k-point and solve with **`scipy.linalg.eigh`**. The cells are small
(§3), so a dense complex Hermitian solver is the simplest correct choice, and it
is exactly what revision 1's real-symmetric framing ruled out. Nothing in v1
needs the 3-D solver's real shift-invert Lanczos path.

---

## 2. Mesh

**Library: `scipy.spatial.Delaunay` on a structured point set.** The unit cell is
a square (`a1 = (a,0)`, `a2 = (0,a)`) or a rhombus for the triangular lattice.

- **Orientation matters for the benchmark, not the physics.** For the triangular
  lattice MPB sets `basis1 = (√3/2, ½)`, `basis2 = (√3/2, −½)` (60° apart, and its
  comment explains the 30° tilt). A cell built as `a1 = (1,0)`,
  `a2 = (½, √3/2)` has the **same Gram matrix** — the same lattice, rotated 30°.
  Band frequencies are rotation-invariant, so the two agree; this was checked
  here rather than assumed, because a grid that silently differed by a rotation
  would move every high-symmetry point.
- **Conforming circle**: nodes placed on the circle at the same angles as the
  cell's boundary divisions, so no triangle straddles the ε discontinuity. This
  is the 2-D analogue of feature 1's region-conforming mesh; Delaunay on that
  point set conforms to the convex hole with no TetGen.
- **Periodic node matching**: match boundary nodes pairwise by `r → r + a_i`,
  found by rounding coordinates to a tolerance and hashing the pair — **not** by
  index arithmetic, which breaks at the cell corners.

**Which image.** Pure numpy/scipy on a 2-D triangulation of at most a few
thousand nodes. It runs **inline in the web pod**, does **not** enter
`dockerfiles/maxwell_solver/`, and adds **no dependency to the Orin worker venv**.

---

## 3. Compute tier

At ~40 divisions per lattice vector: ~1,600 triangles, ~1,700 nodal unknowns;
after eliminating the pairing slaves, a complex Hermitian problem of order ~1,700
for 10–20 bands.

- **Per k-point**: a dense complex `eigh` at n ≈ 1,700 is well under a second.
  A 60-point path is **seconds**. For scale, the 3-D solver's 100k-element solves
  take minutes on the Jetson.
- **Where it runs: web pod inline, synchronously.** It does **not** go to the
  Jetson and does **not** launch a k8s Job — those exist for the heavy 3-D vector
  solves, and routing a 2-second scalar problem through a lease, a Job and a
  scheduling tick would make it slower and less reliable.
- **Bound, stated before running**: hard caps of 20 s wall clock and 20,000
  nodes. Above either, the form is **refused with the limit named**, never run to
  a timeout with no page (the could-not-ask guard).

---

## 4. Validation

**Control A — the empty lattice (analytic, exact, can fail).**
`n_rod = n_bg = 1` must give the folded free-space light lines,
`ωa/(2πc) = |k+G|a/(2πc)`. Landmark values **computed here**:

| point | lowest `ωa/(2πc)` | exact |
|---|---|---|
| Γ | 0.00000 | 0 |
| X | 0.50000 | 1/2 |
| M | 0.70711 | √2/2 = 0.70710678 |

plus the four-fold degeneracy at Γ at 1.0.

> **Tolerance, corrected in revision 3 (the lead's third finding).** `1e-6` is
> **not achievable** against a P1 finite-element mesh, and asserting it would
> have produced a test that is either failing or quietly weakened. The folded
> light lines are exact only in the continuum; every **non-zero** band carries
> `O(h²)` discretisation error. Exactly one landmark is exact on the mesh:
> **λ = 0 at Γ** (the constant field is in the P1 space and the stiffness
> annihilates it), and that is asserted **tight, 1e-10**. Everything else is
> stated as **convergence**, not equality:
> - **relative error ≤ 1e-3 at the production mesh** (a bound stated *before*
>   running, at the mesh §3 specifies), and
> - **an observed rate near 2** across two refinements (`err(2h)/err(h) ∈
>   [3.5, 4.5]`, i.e. order in [1.8, 2.2]), which is the assertion that actually
>   distinguishes a correct second-order assembly from one with an optimised-away
>   boundary term.
>
> The 1e-10 belongs on the **oracle**, not the mesh: the closed-form-vs-shooting
> self-check stated in Control B. A wrong band ordering, a broken pairing
> constraint, or a bad stiffness still moves these numbers well outside 1e-3.

**Control B — a 1-D layered cell against the exact transfer-matrix bands
(analytic, can fail, and catches the ε swap).**
> **This replaces revision 1's control 3, which could not fail** (the lead's
> second finding, and correct). With uniform ε, `(1/ε)K u = λ M u` and
> `K u = λ ε M u` both give `λ = λ₀/ε` — verified here — so a uniform-`n` check
> passes with ε in **either** slot.

The replacement uses a cell that is a **single stripe of ε₂, varying only in x**,
at a k **with a nonzero component parallel to the layers** (the stripe normal is
x; the conserved in-plane component is `k_y`).

**The polarisation mapping, written explicitly** (revision 3 — the labels were
crossed twice before, so it is stated as a mapping and not as prose):

| PhC polarisation | field | relation to the layer normal x | 1-D problem |
|---|---|---|---|
| **TM** (`E_z`) | `E` along z | **tangential** to every interface | **s** |
| **TE** (`H_z`) | `H` along z | `E` in-plane, normal to the interfaces | **p** |

`ẑ · x̂ = 0`, so `E_z` is tangential — that is the whole content of the mapping,
and it is a one-line geometric fact rather than a convention. Revision 2 said
"s (TE)"; it is **TM ↔ s, TE ↔ p**.

**The two closed forms** (1-D transfer matrix over the stripe, `k_x,j = √(λ ε_j − k_y²)`,
`η` the `ψ'/ψ` admittance, `ψ` the out-of-plane field in each case):

- **s (TM, `ψ = E_z`)**: `(ψ, ψ')` is continuous; `η_s = k_x,j`.
- **p (TE, `ψ = H_z`)**: `ψ` and `(1/ε)∂ψ/∂x` are continuous, so the state that
  propagates is `(ψ, (1/ε)∂ψ/∂x)` and **`η_p = k_x,j / ε_j`** — *not*
  `k_x,j/ε_j²`. (My message to the lead said `ε²`; that was wrong. Integrate
  `(ψ, w)` with `w = (1/ε)∂ψ/∂x` directly: `ψ' = εw`, `w' = (k_y²/ε − λ)ψ`.)

The oracle is a **single script** used by both this control and Control D's
generator, and that script **validates itself on its own** before any mesh is
compared to it:

> **Oracle self-check (must pass first).** Compare the closed form's monodromy
> trace against an **independent ODE shooting solve** of the same two systems
> with `scipy.integrate.solve_ivp`, imposing the interface conditions by
> integrating the *continuous* state directly, at **ε₂ = 4** (never ε = 1, where
> `k_x/ε` and `k_x/ε²` coincide and the check cannot fail).
> **Measured in this session:** max `|tr(closed) − tr(shooting)|` over 56 points
> (7 values of `k_y` from 0.05 to 1.2 × 4 values of λ × both polarisations) =
> **6.86e-12 at `rtol = 1e-13`**, against a required **1e-10**.
> The error is the *integrator's*, not the closed form's: it is 1.1e-8 at
> `rtol = 1e-10`, 6.8e-11 at 1e-12, 6.9e-12 at 1e-13. So the plan **pins
> `rtol = 1e-13`** for the shooting solve; at `rtol = 1e-12` the 1e-10 bar is met
> by only 1.5× and is too tight to be a stable test.
> **Control that this check can fail:** the `1/ε²` admittance (the error I made)
> diverges from the same shooting oracle by **3.71** — seven orders above the bar.
> **Reproducible:** `benchmarks/layered_oracle_check.py` runs the whole check and
> prints these numbers; it emits no mesh and is dev-only.

Then the mesh is compared: for each polarisation, the FEM dispersion is compared
to **its own** closed-form branch, at a `k` with `k_y ≠ 0`.

*Why this one can fail:* the swap puts ε on the wrong side of the generalised
problem, which changes which polarisation gets the stripe-weighted operator — so
at least one of the two comparisons must move. Confirmed numerically here: at
ε₂ = 4, `k_y = 0.3`, the two traces differ by **3.3e-2** (and by **3.1e-1** at
`k_y = 0.9`), while at `k_y = 0` they agree to **2.2e-16** — which is precisely
why the parallel component is required. **A uniform-ε check cannot see the swap;
this one cannot miss it.**

**Tolerance on the mesh comparison: the same `O(h²)` convergence structure as
Control A above — not 1e-6.**

**Control C — qualitative, and the one a human can read at a glance.**
Dielectric rods in air must show a **TM 1–2 gap and no TE 1–2 gap**. This is
independent of any number in a table and is what revision 1 would have drawn
backwards.

**Control D — the published benchmark (numeric source, so a documented
tolerance).**
Source: **MPB's own Scheme tutorial**, `Bands of a Triangular Lattice`,
<https://mpb.readthedocs.io/en/latest/Scheme_Tutorial/> — **fetched and read in
this session** (HTTP 200, 47,394 bytes). The exact sentences and parameters:

- the geometry is a dielectric cylinder, and the square-lattice subsection sets
  **`(radius 0.2)` with `(epsilon 12)`**, cylinders in air. The triangular
  subsection says only the lattice changes — *"we only need to change the
  lattice… All of the other parameters (geometry, num-bands, and grid-size) can
  remain the same"* — so **r = 0.2a and ε = 12 for the triangular case too**;
- *"As it turns out, this structure has an even larger TM gap than the square
  lattice: Gap from band 1 (0.275065617068082) to band 2 (0.446289918847647),
  47.4729292989213%"*;
- resolution: *"(set! resolution 32)"* → *"a 32x32 computational grid"*.

So the benchmark is: **triangular lattice, TM, r = 0.2a, ε = 12, gap
0.275066 → 0.446290 (47.47%)**, in units of `ωa/(2πc)`.

**Tolerance, stated before running: the computed gap edges must agree to within
1% of the midgap frequency.** MPB at resolution 32 is itself a numerical
computation with its own discretisation error, so this is a numeric-vs-numeric
comparison and 1% of midgap is the right bar — much looser than the 1e-10 the
**oracle** earns, because the oracle is analytic-vs-analytic and this is
numeric-vs-numeric. **If the numbers fall outside 1%, the difference is to be
explained, not tuned away.**

> **Dropped from revision 1:** the square-lattice alumina (ε = 8.9, 0.302–0.443)
> figures. `ab-initio.mit.edu` is unreachable from this VM (the lead confirms the
> same), I never read it, and the lead's figure was from memory. Those numbers
> are gone. Note the tutorial's own square-lattice range, 0.2812–0.4174, appears
> in a *later* subsection about a defect cavity, not as the headline gap, so it
> is **not** used as a benchmark either.

---

## 5. Data model and UI

**Storage.** Every field is already on `mse.jobs`, and the diagram is small
(~60 k-points × 20 bands × 2 polarisations), stored the same way the mode list
is: a `pcs jsonb` column holding the request (`lattice`, `r_over_a`, `n_rod`,
`n_bg`, `n_bands`) and the computed bands. Additive, via the **exact
`IF NOT EXISTS (... information_schema.columns ...)` idiom `materials` used**
(`db/001_schema.sql:247-257`) — quoted, not restated:

```sql
IF NOT EXISTS (
    SELECT 1 FROM information_schema.columns
    WHERE table_schema = 'mse' AND table_name = 'jobs' AND column_name = 'pcs'
) THEN
    ALTER TABLE mse.jobs ADD COLUMN pcs jsonb;
END IF;
```

NULL means "not a photonic-crystal job", exactly as `materials` NULL means "the
one-part path".

**Routing.** A `kind` discriminator (proposed `jobs.kind text NOT NULL DEFAULT
'cavity'`, values `cavity` | `pc`) so the 3-D pages and the band-diagram page are
two views of one table. The default keeps every existing row and reader on the
cavity path unchanged.

**UI, without breaking 3-D jobs.**
- The upload form gains a mode selector; the `pc` branch renders a lattice /
  `r/a` / indices / bands form and **accepts no `.obj`**.
- The job page gains a band-diagram panel **only when `kind = 'pc'`**: an inline
  canvas/SVG plot of ω versus the k-path, gaps shaded, gap-midgap ratio printed.
  **No new JS library** — the pages already ship three.js for the 3-D viewer, and
  a line chart with shaded bands does not justify another vendor file. The
  existing `viewer.js` panel is untouched and simply not rendered for a `pc` job.
- `/jobs/<id>/status` grows a `kind` field and, for `pc`, the diagram payload.
  Adding a key is backward-compatible.

---

## 6. Explicitly out of scope for v1

- **3-D slabs, woodpiles, and anything on the tetrahedral solver** (roadmap
  §3.1 "then 3-D"); those need Bloch-periodic *vector* elements.
- **Dispersion and loss**: ε real, no frequency dependence, no absorption.
- **Defects and waveguides** (a line defect is a supercell problem).
- **Non-circular rods**, and **radius or lattice optimisation** (MPB's own
  "Maximizing the First TM Gap" section is a separate feature).
- **Complex-ω / quality factors**, and **finite-height effective-index** slabs.
- `r/a` beyond where the circles still fit without touching (a refusal, not a
  clamp); `r/a = 0` is Control A's empty lattice.

---

## Files (proposed)

| file | change |
|---|---|
| `web/pc.py` (new) | lattice vectors, unit-cell meshing + periodic node matching, the two scalar operators, `scipy.linalg.eigh` on the complex Hermitian pair, k-path, band assembly |
| `web/app.py` | `kind` selector on upload; `pc` form; refuse `.obj` there; band-diagram panel |
| `web/db.py`, `db/001_schema.sql` | `kind` and `pcs` columns (additive idiom) |
| `templates/`, `web/static/` | the form and the canvas chart (no new library) |
| `tests/test_pc.py` (new) | Controls A–C; refusals naming the limit |
| `benchmarks/pc_bands.py` (new, dev-only) | Control D against the MPB figures |

## Verification

Tier 1, the capped command, **read the COUNTS**:
`ulimit -v 4194304; nice -n 19 .venv-mse/bin/python -B -m pytest tests/ -q`

0. **Oracle self-check first** — closed form vs ODE shooting (`solve_ivp`,
   `rtol = 1e-13`) at ε₂ = 4 to **1e-10**, with the `1/ε²` admittance shown to
   fail it (measured 3.7). No mesh comparison runs until this passes.
1. Control A (empty lattice): **λ = 0 at Γ tight (1e-10)**; the other landmarks
   as `O(h²)` **convergence** — rel. err. ≤ 1e-3 at the production mesh and an
   observed rate near 2 across two refinements. **Not** 1e-6.
2. **Control B (1-D layered vs transfer-matrix), per polarisation**, against each
   polarisation's own closed form at `k_y ≠ 0` — the check that can see the
   ε swap. Mesh tolerance as Control A (O(h²)), not 1e-6.
3. Control C: dielectric rods give a TM 1–2 gap and no TE 1–2 gap.
4. Periodic pairing: a **mutation** dropping the phase factor reddens the k≠0
   bands and leaves Γ green.
5. Refusals: node cap and `r/a` too large, each **naming the limit**.
6. Control D (MPB) at **1% of midgap** (unchanged).
