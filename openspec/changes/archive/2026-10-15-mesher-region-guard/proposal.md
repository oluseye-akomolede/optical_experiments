# mesher-region-guard

**Product change.** The resolution guard becomes PER REGION: each region's
achieved median edge is compared against `MAX_EDGE_OVERSHOOT` (1.6) times that
region's OWN required edge, instead of the whole mesh's median against 1.6 times
the WALL's edge. The diagnosis below is the evidence for it — the accuracy hole
is demonstrated with real meshes, so this is no longer diagnosis-only. Files
touched: `dockerfiles/maxwell_solver/mse_meshing.py`,
`dockerfiles/maxwell_solver/mse_common.py`, `tests/test_meshing_guard.py`. No
web/ file, no new constant, no message constant.

## Why

Row `qitem-20261015050324-adc16144`, from the lead:

> can a live multi-part job ship frequencies that come from a high-index core
> meshed coarser than the resolution rule allows, with no refusal?

The suspicion was concrete: the too-coarse guard compares the achieved edge
against **1.6x the WALL's edge** (`mse_meshing.py:568`, `allowed_edge`), while
the `maxvolume` it is trying to achieve comes from **`h_min`**, the finest
region edge (`:507`, `:509`). For a core at index `n_r` against a wall at
`n_wall`, the guard's tolerance expressed in the core's OWN required edge is

    allowed_edge / h_r  =  MAX_EDGE_OVERSHOOT * n_r / n_wall

which is 1.6x for the wall, 3.2x for an n=2 core, 4.8x for an n=3 core and 5.6x
for the n=3.5 core of the dense-core family. The question is whether that slack
is ever *used*.

**Answer, in one line:** on the nine families at the three mode scales, at the
product's own 600,000 cap, **no** — the worst region anywhere is 1.243x its own
rule and the guard closes with 0.29-0.78 of its allowance unused (a). But the
slack is exercised the moment the element cap pushes a mesh through the
coarsening retry: at a fixture cap just **4%** below the count the same mesh
would otherwise have had, the core of a 2-region upload is delivered at
**1.155x** its own required edge and the guard passes it (b, d). The retry is reachable in the
product — it fires for any accepted upload whose real mesh exceeds 600,000 — so
the accurate statement is "not at the product cap *on these families with no
retry*", not "not in the product".

## The code, as it stands

Read at `dockerfiles/maxwell_solver/mse_meshing.py`:

- `effective_edge_m(...)` (`mse_common`) returns the **wall's** edge, by its own
  docstring ("this value is the wall's edge by construction ... and the
  multi-part branch below does not shrink it") and by its own recorded history
  ("an earlier version took `min(base/n_r)` here ... over-predicted the count by
  more than 10x"). The index enters *downstream*, as a ratio, in
  `region_edges`: `h_r = h * n_wall / n_r`.
- The plan is **uniform at the finest edge**: `edge_units = h_min` for the whole
  cavity (`:507`), `maxvolume = target_maxvolume(surface, edge_units)` (`:509`).
  Every region is meshed at the *finest* region's edge — the wall is
  deliberately over-resolved, and the delivered per-region target is `h_min`
  everywhere, not each region's own `h_r`.
- The attempt loop (`for attempt in range(MAX_MESH_ATTEMPTS)`, `:573`; branches
  `:702`-`:781`), in order:
  1. not over cap and not too coarse -> accept;
  2. too coarse and `enforce_resolution` off -> warn and accept (bench only);
  3. **too coarse AND over cap -> RAISE** (`_coarse_message`);
  4. too coarse, within cap -> `maxvolume /= (achieved/allowed_edge)**3`, retry;
  5. else (over cap, NOT too coarse) -> the **cap-coarsening** path:
     `maxvolume *= (n_tets/max_tets)*1.15`, retry; raise instead if the previous
     coarsening moved the count by less than `CAP_RETRY_RESPONSE` (0.20), or on
     the last attempt.
- `enforce_resolution` defaults to `True`, so the product runs the strict guard.

So the guard's reference is the wall's edge while `maxvolume` is the finest
region's, and the cap path can raise `maxvolume` (making the delivered edge
larger) *between* two guard readings.

## (a) Per-region achieved edge vs each region's OWN required edge, at the product cap

`benchmarks/mesher_region_guard.py`, the product's own mesher and cap
(600,000), region labels from TetGen's returned per-tet attribute array (not an
absolute-mm mask), the probe's medians measured with the guard's own
`achieved_edge_length` applied to that region's tets. `h_r = h_wall * n_wall /
n_r`. Ratio > 1 means the region shipped coarser than its own rule needs.

Nine families at scale 1.0 (one-part families at their declared `n_modes`; the
three scales are one-part n=10/5/20 and multi-region n=4/2/8, since the mesher
takes `n_modes` and it changes the mesh). Every one of the 27 runs reports
`OUTCOME accepted`, `refine 0, coarsen 0` — the counts are far below 600,000, so
there is no cap pressure and this is the mesher's own spread, not the retry.
(The probes count retries by the strings the mesher actually prints —
`too coarse; refining` and `N tets is over the M cap; retrying` — after an
earlier version counted a string that is never printed and so could not fail.)

| family | region (n) | h_r mm | achieved mm | ach/h_r | guard slack |
| --- | --- | --- | --- | --- | --- |
| one-part 6mm cube | 1 body (1) | 0.6734 | 0.7558 | **1.122** | 1.60x |
| one-part plate 8x4x2 | 1 body (1) | 0.4489 | 0.5091 | **1.134** | 1.60x |
| one-part sphere r3 | 1 body (1) | 0.5412 | 0.4935 | 0.912 | 1.60x |
| one-part bumpy sphere r3 | 1 body (1) | 0.5430 | 0.5409 | 0.996 | 1.60x |
| 2-region core 1.5mm | wall (1) | 0.9186 | 0.4002 | 0.436 | 1.60x |
| | core (2) | 0.4593 | 0.3633 | 0.791 | 3.20x |
| 2-region dense core | wall (1) | 0.9186 | 0.2285 | 0.249 | 1.60x |
| | core (3.5) | 0.2625 | 0.2306 | 0.879 | **5.60x** |
| 3-region shell+core | wall (1) | 1.0059 | 0.1259 | 0.125 | 1.60x |
| | shell (2) | 0.5029 | 0.1047 | 0.208 | 3.20x |
| | core (3) | 0.3353 | 0.0769 | 0.229 | 4.80x |
| 2-region off-centre | wall (1) | 0.9250 | 0.4301 | 0.465 | 1.60x |
| | core (2) | 0.4625 | 0.4212 | 0.911 | 3.20x |
| 2-region thin wall 0.4mm | wall (1) | 1.0801 | 0.6282 | 0.582 | 1.60x |
| | core (2) | 0.5401 | 0.6137 | **1.136** | 3.20x |

Worst region per family at all three mode scales (n x1.0 / x0.5 / x2.0):

| family | x1.0 | x0.5 | x2.0 |
| --- | --- | --- | --- |
| one-part 6mm cube | 1.122 | **1.243** | 1.028 |
| one-part plate 8x4x2 | 1.134 | 1.214 | 1.030 |
| one-part sphere r3 | 0.912 | 1.039 | 1.008 |
| one-part bumpy sphere r3 | 0.996 | 0.798 | 1.143 |
| 2-region core 1.5mm | 0.791 | 1.096 | 0.816 |
| 2-region dense core | 0.879 | 0.754 | 0.953 |
| 3-region shell+core | 0.229 | 0.790 | 0.460 |
| 2-region off-centre | 0.911 | 0.818 | 0.853 |
| 2-region thin wall 0.4mm | 1.136 | 1.057 | 0.960 |

Maximum over all 27 runs: **1.243** (one-part cube, half mode count, whose own
slack is only 1.60x). Across all 27 runs there were **0 guard failures, 0 refine
retries, 0 coarsen retries, 0 refusals**.

The structural hole is visible in the same table and is *inverted* in practice:
the regions with the largest allowance are the most over-resolved. The dense
core has a 5.60x allowance and ships at 0.879-0.953x its own rule; the 3-region
core has 4.80x and ships at 0.229-0.790x.

## (b) Is any region shipped above 1.6x its OWN required edge while the guard passes?

**At the product cap: no.** Worst 1.243x (above), at scale 1.0, 0.5 and 2.0
alike. Nothing above 1.6x occurs, so nothing exercises even the wall's 1.6x, let
alone the 3.2x/4.8x/5.6x the cores are allowed. Stated positively: on these nine
families at three mode scales the answer to the row's question is that a
high-index core is never shipped coarser than its own rule — every core in the
sweep came out **fine**r than its own rule, by 0.12x to 0.96x.

**Under the cap retry: yes.** See (d): at fixture caps the finest region reaches
1.155x, 1.662x and 1.915x its own required edge while the guard reports PASS,
and in the hard-bite case 2.309x. So the per-region blindness is real and
exercised — it is the *cap*, not the guard, that keeps it out of the product-cap
sweep.

## (c) The over-delivery cliff: cost-only, or can it reach the cap retry and then the guard?

Not cost-only. The cap is enforced on the count TetGen actually returns
(`over_cap = n_tets > max_tets`), not on the estimate, so **any accepted upload
whose real mesh exceeds 600,000 enters the coarsening path** — and that band is
wide. This change's estimator is accepted up to
`upload_refuse_tets() = UPLOAD_REFUSE_FACTOR * max_tets() = 2 * 600,000 =
1,200,000`, i.e. `est = 1.35 * law <= 1,200,000` -> `law <= 888,889`; measured
`built/law` runs 0.64-1.91 across the families, so at that extreme an accepted
upload with `law` above about 319,000 can already build over the cap, and the
reachable band is `law` in roughly (319,000, 888,889]. (Main's `C = 14` refuses above 727k law-tets, so main additionally
runs every upload in 600k-727k through this same path.) The path then does one
thing — `maxvolume *= (n_tets/max_tets) * 1.15`, rebuild, re-read the guard —
and the guard it re-reads is the per-region-blind one from (a)-(b). Measured on
four families at fixture caps (d), a bite of **1.03-1.10x** (the product's own
regime: a few percent over the cap) delivered the finest region at 0.67-1.16x
its own rule with the guard passing, and a bite of 2.19-8.38x reached
0.53-2.31x. In **8 of the 14** accepted runs some region came out coarser than
its own required edge, and across all 14 the guard passed and no further
correction was attempted: exactly one coarsening each, `refine 0`. One family
refused instead — the facet-dense sphere, where the coarsening moved the count
by less than the 20% floor (`CAP_RETRY_RESPONSE`), i.e. a hard error rather than
a coarse mesh.

The case the row names — the 3-region family at scale 0.5, building 30,144
elements — is **5% of the cap**, so at the product cap it does not enter this
path at all: on its own, over-delivery is a cost matter (in this sweep that mesh
came out 0.888x the requested edge, i.e. finer, not coarser). The accuracy risk
begins only where a count *exceeds* the cap, and (d) measures what happens then.

The retry also **overshoots its own arithmetic**. `tets` is treated as
`1/maxvolume`, so a bite of `b` should raise `maxvolume` by `1.15b` and land the
count near `cap/1.15`. It lands much lower: the 3-region family at a 1.05x bite
went 419,176 -> 78,053 elements on a 1.20x `maxvolume` bump (a 5.4x count drop),
and the 2-region core at a 1.04x bite went 24,376 -> 13,132 on a 1.20x bump
(1.9x). Raising the ceiling far enough lets TetGen's quality criteria — not the
bound — set the element size, which is the same "a bound is not a target" effect
the estimator change records in `maxvolume-is-a-ceiling`. So one coarsening step
costs more resolution than the multiplier suggests, and the delivered edge is
not plan-derivable.

## (d) The cap retry, measured: how much coarser, does the guard catch it, is the user told

`benchmarks/mesher_cap_retry.py`, tier 1 on the VM, the product's mesher, the
product's rule. **Fixture caps are used and every one is printed and labelled
`FIXTURE CAP ... (NOT the product's 600,000)`** — the retry is only reachable
when the unconstrained count exceeds the cap, and a tier-1 fixture cannot reach
600,000 elements. The transferable quantities are the ratios. `bite` is the
uncapped count over the cap (the mesher's own `N tets is over the M cap` line).

| family | uncapped n | fixture cap | bite | delivered n | finest region | ach/h_r | guard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2-region core (core 0.791 uncapped) | 24,376 | 23,500 | 1.04x | 13,132 | core | **1.155** | PASS |
| | | 22,500 | 1.08x | 16,148 | core | 1.130 | PASS |
| | | 20,000 | 1.22x | 12,297 | core | 0.817 | PASS |
| | | 15,000 | 1.63x | 8,992 | core | 1.230 | PASS |
| | | 8,000 | 3.05x | 4,098 | core | 1.381 | PASS |
| | | 4,000 | 6.09x | 1,956 | core | **2.309** | PASS |
| dense core (core 0.879 uncapped) | 131,631 | 128,000 | 1.03x | 90,510 | core | 0.683 | PASS |
| | | 120,000 | 1.10x | 90,612 | core | 0.848 | PASS |
| | | 60,000 | 2.19x | 44,920 | core | 1.093 | PASS |
| | | 20,000 | 6.58x | 12,459 | core | **1.662** | PASS |
| 3-region (core 0.229 uncapped) | 419,176 | 400,000 | 1.05x | 78,053 | core | 0.670 | PASS |
| | | 380,000 | 1.10x | 72,286 | core | 0.787 | PASS |
| | | 100,000 | 4.19x | 50,826 | core | 0.531 | PASS |
| | | 50,000 | 8.38x | 6,486 | core | **1.915** | PASS |
| one-part sphere r3 | 7,586 | 5,000 | 1.52x | refused | — | — | MeshingError |
| | | 2,500 | 3.03x | refused | — | — | MeshingError |
| | | 1,200 | 6.32x | refused | — | — | MeshingError |

("uncapped" column = the same family's `ach/h_r` with no retry at all, from (a),
for comparison.)

1. **How much coarser than the rule.** One coarsening is enough to push the
   finest region past its own rule even at a 4% over-cap bite: 1.155x at 1.04x,
   and 1.13-1.38x over the 1.08-3.05x bites, up to 2.31x at 6.09x. The
   comparison that matters is against the *uncapped* build of the same family:
   the 2-region core's own rule is exceeded (0.791 -> 1.155x) by a step that
   only had to shed 4% of the elements.
2. **Does the guard catch it.** Never. `refine 0, coarsen 1` in all seventeen
   runs (fourteen accepted, three refused); the guard re-read after the rebuild
   passed every accepted mesh. It
   cannot catch it: its limit is 1.6x the *wall's* edge, which for these
   families is 3.20x/4.80x/5.60x the finest region's own rule, and the coarsened
   meshes sit between 0.53x and 2.31x that rule. There is also no oscillation — the
   refine arm never fires, because a coarsened mesh inside the cap is `not
   over_cap`, so branch 5 is never re-entered and branch 4 is not reached.
   **(Pre-fix reading only — see "after the fix" below: with the per-region
   verdict the refine arm does fire, and branch 4 is reached.)**
3. **Is the user told.** Not in the product UI. The retry prints one stdout line
   — `[mse] N tets is over the M cap; retrying with maxvolume V` — which reaches
   the pod log only: the app launches the Job with `--job-id --units --n-index
   --n-modes` and no `--quiet` (`web/jobs.py:74-77`), so the solver's
   `verbose = not args.quiet` is True (`mse_maxwell_solver.py:797`) and the line
   *is* in `kubectl logs`; there is no log route on the app (`/jobs/<id>` and its
   five siblings only), so nothing renders it. What the owner does see is
   `_resolution_caption` (`web/app.py:1386`): the **global** achieved median
   against the declared smallest feature, with `within_allowance` computed as
   `achieved <= MAX_EDGE_OVERSHOOT * request` (`web/app.py:1432`), and the job
   template's wording that a median up to that factor is "the allowance being
   used, not a fault" (`web/templates/job.html:31-45`). That caption (a) exists
   only when a feature was declared — `value is None` returns `None` — (b) is
   one global median, so a single coarse core among a fine wall cannot show in
   it, and (c) says nothing about a coarsening having happened or what
   `maxvolume` was used. So the honest summary is: the user is told the
   *aggregate* achieved edge, is framed to read up to 1.6x as normal, and is not
   told that a cap retry coarsened the mesh, nor given any per-region number.

### (d, after the fix) The same seventeen runs, judged by the per-region guard

Re-run of `benchmarks/mesher_cap_retry.py` against the fixed mesher — same
fixtures, same caps, so the before/after is one table. The script prints both
readings per row: the global one the pre-fix guard applied, and the per-region
one the mesher now applies. Verdicts against the table above:

| fixture cap (family) | pre-fix | post-fix |
| --- | --- | --- |
| 2-region core, 4,000 (6.09x bite) | accepted, core **2.309x** | **REFUSED, naming the core** |
| dense core, 20,000 (6.58x bite) | accepted, core **1.662x** | **REFUSED** — the cap arm's "would need about 25,715 elements" |
| 3-region, 50,000 (8.38x bite) | accepted, core **1.915x**, 6,486 tets | accepted, core **1.068x**, 16,781 tets — the refine arm used its lever |
| the other eleven accepted rows | accepted | accepted, **same delivered count** |
| the three sphere rows | refused (cap arm) | refused (cap arm), unchanged |

The fix removes every accepted row that was past 1.6x a region's own rule
(1.662, 1.915, 2.309) and leaves the eleven rows inside it untouched. No accepted
row in the table is now above 1.6x its own region's rule; the worst is the
2-region core at an 8,000 cap at 1.381x — the same row and the same number as
before the fix.

Two things the pre-fix measurement could not show:

* **The refine arm is reachable from the cap retry after all.** "The refine arm
  never fires" in (d) item 2 was a consequence of the *blind* guard: a coarsened
  mesh that was too coarse but inside the cap looked fine, so branch 4 never ran.
  With the per-region verdict the 3-region row at a 50,000 cap takes exactly that
  path and ends compliant — the coarsened mesh is judged too coarse,
  `maxvolume /= ratio**3` refines it to 16,781 elements (inside the cap, 1.068x
  the rule) and it is accepted. Cost: 2.6x the elements the blind guard would
  have shipped for a mesh 1.9x past the rule.
* **The refusal is not always the region one.** Where refinement cannot fit under
  the cap the mesh is refused, with the region-naming message when the final
  attempt is both too coarse and over cap (the 2-region 4,000 row), and with the
  pre-existing cap-arm message when refinement *succeeded* in resolution but not
  in size (the dense-core 20,000 row). That second message is right for that
  state — the mesh is fine enough, it is simply too big for the cap — so it was
  left alone. Both are refusals rather than a coarse answer, which is the point.

At the PRODUCT cap these fixture rows are not reachable: the nine families in (a)
build at most 131,631 elements (the dense core) — 22% of 600,000 — so a real
upload's cap retry is the few-percent overshoot the estimate under-reads by, not
a 6x bite. The rows above show the mechanism on fixtures small enough for tier 1.

## Documentation defect found in passing (fixed here)

`mse_meshing.py:459-463` still said "`parts`/`materials` go in so a multi-part
upload's edge is the SMALLEST `h/n_region`" — the behaviour that
`effective_edge_m`'s own docstring records as replaced ("an earlier version took
`min(base/n_r)` here ... over-predicted the count by more than 10x"). The code
returns the wall's edge; the comment described the withdrawn rule. It is the kind
of stale comment that has already propagated once into a spec
(`a-scale-claim-in-a-comment-propagated-into-the-spec`). This change corrects
it: the comment now says the wall's edge is what this function derives, and
points at `region_edges` for the per-region requirement it is NOT returning.

## What is not claimed

- The fixture caps in (d) are **not** the product's cap and the elements of the
  table are the RATIOS. The real numbers at 600,000 are the lead's on the Orin.
- "Not exercised at the product cap" is scoped to the nine families at three
  mode scales on the VM. It is not a claim about every upload: (c) shows the
  band of accepted uploads whose real count exceeds 600,000, and every one of
  them runs this path.
- The new guard **does not newly refuse any of the 27 swept runs** (worst region
  1.243x < 1.6), so the change is behaviour-preserving at the product cap on
  those families. It changes the verdict only where a region is genuinely past
  its OWN rule — the cap-retry regime in (d), which is what it is for. This is
  an expectation from (a), not a claim that no real upload anywhere is newly
  refused: any upload that today ships a region above its own 1.6x will now be
  refined (then refused, if refining cannot help).

## The change (step 1)

Per the lead's decision on row `qitem-20261015050324-adc16144`.

- **Per region.** `mse_common.region_edges` now returns each region's OWN
  required edge `h_r = h * n_wall / n_r` (unchanged arithmetic, no floor).
  `region_plan` keeps `edge_units` = `h_min` — the MESHING edge the mesher builds
  and the estimator counts at, so the estimate, the app and every existing reader
  are untouched — and adds `own_edge_units` per region for the guard. The guard's
  new helper `_resolution_verdict` takes each region's tets (TetGen's per-tet
  attribute array), measures that region's median with the SAME
  `achieved_edge_length` the global check used, and returns the region furthest
  past `MAX_EDGE_OVERSHOOT * own_edge_units`.
- **No new constant.** 1.6 stays `MAX_EDGE_OVERSHOOT`, and the message
  INTERPOLATES it rather than retyping it, so the sentence cannot drift from the
  number the guard used. Both existing arms stay: refine within the cap, refuse
  over it (or out of attempts).
- **The refusal names the region** when there is one: "The 'core' region needs
  elements about 0.2 mm across to resolve 20 modes at its 3x refractive index,
  but they came out about 1.414 mm -- past the 1.6x the resolution rule
  allows. Try fewer modes, or a lower refractive index for that region." The
  one-part wording is unchanged, because there is no region to name and because a
  one-part upload has no per-region diagnostic to offer.
- **The refine ratio is the worst region's**, not the whole-mesh median's:
  refining to the median can leave the region that is actually failing short.
- **The global median is still measured and reported.** `achieved` continues to
  feed the verbose log line and `achieved_median_edge_m` in `modes.json`, so the
  app's resolution caption is unchanged. The guard simply no longer reads it.
- **A one-part upload is unchanged by construction.** With no `region_plan` the
  verdict is `achieved / allowed_edge > 1` — literally the old expression — and
  with a one-region plan the region's own edge is the wall's.

## Verification

Tier 1 on this VM, bounded (4 GiB, `nice -n 19`, 1 BLAS thread,
`.venv-mse/bin/python`):

- `tests/test_meshing_guard.py` — three new tests:
  `test_a_coarse_core_is_refused_by_its_own_rule` (a fake two-region mesh whose
  fine wall passes at 0.147x its allowance while its coarse core is 4.42x past
  its own; the old whole-mesh median, 0.587 mm, sat inside the wall's 0.96 mm
  allowance at 0.61x and so ACCEPTED it), `test_a_fine_core_is_accepted` (the
  control: the same path must pass when the core is fine, so the test cannot be
  satisfied by a guard that refuses every multi-region upload), and
  `test_the_per_region_rule_reduces_to_the_global_one_for_one_part` (one region,
  two code paths, one answer — both a too-coarse and a fine mesh).
- **Mutation-checked**: reverting `_resolution_verdict` to the global comparison
  makes the first test FAIL with "DID NOT RAISE" — the defect, reproduced — and
  the other two stay green. The red is the test's own, not a harness error.

## Impact

- Behaviour changes only where a region is built past `1.6 * h_r` — i.e. the
  cap-retry regime in (d). On the 27-run product-cap sweep it is a no-op, and
  the one-part path is provably identical.
- An upload that can only be resolved by coarsening past a region's rule is now
  refused (after the refine arm fails) rather than silently shipping frequencies
  from a mesh that cannot resolve them. The user gets a message naming the region
  and the two levers that help (fewer modes, lower index) — not "a smaller
  object", which cannot help, since scaling the object scales the region's
  requirement with it.
- Not addressed here: the user is STILL not told that a cap retry coarsened a
  mesh that stayed inside every region's rule (d, item 3). That is a display
  question (there is no log route), not a guard question.

## Commands

Tier 1, this VM, bounded (`nice -n 19`, 1 BLAS thread, 4 GiB virtual-memory cap,
`.venv-mse/bin/python`), tier-1 fixtures only (6 mm cubes, one 8x4x2 plate, r=3
spheres — nothing larger):

```
(ulimit -v 4194304; OMP_NUM_THREADS=1 nice -n 19 .venv-mse/bin/python \
    benchmarks/mesher_region_guard.py 1.0 0.5 2.0)          # (a)/(b), ~3 min
(ulimit -v 4194304; OMP_NUM_THREADS=1 nice -n 19 .venv-mse/bin/python \
    benchmarks/mesher_cap_retry.py)                         # (c)/(d), ~3 min
```

Tier 2 (the lead, the Orin, real meshes). The two probes build their own tier-1
fixtures, so running them on the Orin would just produce the same tiny numbers.
What needs the Orin is a real upload whose mesh exceeds 600,000 elements — the
only way to see this path at the product cap on real geometry — which is the
bench the lead already runs, `benchmarks/estimate_vs_built.py --build` at the
product cap. (Passing `--max-tets 600000` explicitly now prints no override
label; that was fixed in `c5d7e44`.)

Step 3 of the row's decision is one Orin command, run by the lead: the same
`benchmarks/estimate_vs_built.py --build` at the product cap, whose per-region
`ach/h_r` column now reads against each region's own rule. Nothing in this change
needs the Orin to be *verified* — the guard's verdict is a pure function of the
per-tet attribute array and the plan, and the tier-1 tests pin both arms.

To re-run just the guard tests (seconds, no meshing):

```
(ulimit -v 4194304; OMP_NUM_THREADS=1 nice -n 19 .venv-mse/bin/python -m pytest \
    tests/test_meshing_guard.py -q)
```
