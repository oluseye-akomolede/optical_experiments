# Graded meshing is the product default

## Why

`per-region-mesh-grading` built the graded mesher -- each region of a multi-part
upload at its own edge, from TetGen's sizing field -- and then parked it behind
`GRADED_MESHING = False`, because turning it on (that change's step 3) was never
approved. It is approved now: the owner answered "4 yes" on 2026-11-02 to making
per-region grading the default for a multi-part upload.

What that buys: the coarse region stops being meshed at the finest region's
edge. On the parent change's nested-cube fixture the graded path builds FEWER
tets than the uniform one while the finest region keeps its resolution (its
task 4.1, pinned in both directions) -- the saving is in the coarse region, and
it is what makes a large part with one small feature fit the same element cap.

## What changes

1. **The switch is on, and it is one switch.** `mse_common.graded_meshing_enabled`
   reads `MSE_GRADED_MESHING` from the environment: unset (and any value nobody
   meant) means ON; the explicit spellings `0`/`false`/`no`/`off` mean OFF. The
   solver's `mse_meshing.graded_meshing()` delegates to it, so a product route
   never sets a module flag; the flag remains only as a tier-1 test seam.
2. **The estimate prices the mesh the mesher builds.** `web.meshing.estimate_tets`
   calls `estimate_tets_graded` for a multi-part upload when the switch is on and
   `estimate_tets_by_region` when it is off -- the same accessor the mesher reads,
   through the `mse_common` module the web image already copies. The two rules are
   materially different on any nested upload, and pricing a graded run with the
   uniform sum is the failure this change must not ship: the count is what routes
   the job, so it must be the count the chosen mesher produces. Measured on the
   `three-region-shell-core` fixture the graded estimate reads 15,791 where the
   uniform sum reads 63,214 -- the uniform plan meshes the coarse wall at the
   core's fine edge, so its sum prices ~4x a mesh nobody builds. On
   `two-region-core` the graded figure is 4,907 against 18,755. The estimate can
   move either way with the fixture; what must hold is that it moves WITH the
   mesh, not that it moves in one direction. One-part uploads are untouched on
   both sides.
3. **The rollback is the environment variable, and the app forwards it.** 
   `MSE_GRADED_MESHING=0` gives back the uniform plan, the uniform estimate and
   the pre-feature `modes.json` mesh block, with no code change. Setting it on
   the web app is enough for a multi-part upload: the app builds the solver Job
   manifest and now sends the setting it priced with as `MSE_GRADED_MESHING` on
   the Job's container (`web/jobs.py`'s `build_job_manifest`, fed by the
   `K8sJobLauncher.manifest` wrapper). Before this was written the Job pod
   started from the image default, so a rollback would have reached only the web
   pod -- the app pricing one mesh while the solver built another, which is
   exactly the failure the switch exists to prevent. `test_routing` pins both
   directions. A deploy that wants a rollback beyond that sets it on both.

## What does not change

- A one-part upload: same call, same mesh, same estimate (`test_meshing_guard`'s
  byte-identity control still pins it).
- A multi-part upload whose regions all land on one edge: it never enters the
  sizing-field branch, so it meshes as before.
- The graded path's own guards: `fixedvolume` and a global `maxvolume` still
  raise beside the sizing field, and the per-region thickness clamp still
  applies, with its caveat.

## What this does not establish

That every existing multi-part job still fits its element cap after the switch.
The estimate and the mesh move together and the per-region verdict refuses a
region outside its allowance, but a real grading run on a real upload is tier-2/
tier-3 work and has not been done for this change. The change is reversible with
one environment variable on both deployments.
