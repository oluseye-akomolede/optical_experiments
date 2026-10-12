# The field movie draws the object it solved

## Why

The owner, watching a time-domain job's movie: *"make sure they show the
original object during the movies."* A frame today is the field in a box --
literally the solver's domain box drawn as a wireframe. Nothing in the picture
is the thing they uploaded, so a mode's shape has no object to read it against.

The cavity viewer already draws an uploaded surface (`surface.glb`). Time-domain
jobs write no such file and carry no `surface_url`, which is why they show a box
instead of an object.

## What changes

- `web/outline.py` (new): builds the uploaded object's outline from the job's
  own stored upload, every part of it, in metres, by the job's own `units` and
  `scale` -- `unit_scale(units) x scale`, nothing else. No translation, no
  re-scaling, no centring: the stored frames are in exactly that frame (the
  worker stores `origin_m + node`, whose origin is the parts' combined minimum).
- `web/app.py`: a finished td job's payload carries `outline_url`, and
  `/jobs/<id>/outline.glb` serves that outline from the stored upload. So a job
  solved long before this change shows its object with no re-run, and the solver
  is not touched.
- `web/static/viewer.js`: the movie draws the outline, every part, over the
  field on every frame, and names in the caption any outline it could not draw
  rather than showing a box with no object and no explanation. It reuses the
  cavity's `addSurface` drawing (`glbMeshParts` + `outlineGroup`).
- The render check's td mirror carries the job's outline and checks it: the box
  the page says it drew against the box the manifest's own positions span
  (within one grid cell), and, on every sampled frame including one reached by
  playing, a substantial share of the frame's pixels gained from the outline by
  comparing the same page with the outline route answering 404.

## Why not the cavity's `surface.glb`

That file is written only by the cavity solver, in the *user's* units and
centred on its own origin, and the cavity view recovers its scale by ratio and
places it at `bounds.centre`. The movie's frames are not in those coordinates,
and the ratio trick is only right for a mesh centred on its own box. Deriving on
request is also what lets jobs already solved show their object.

## Scope

- The td job page and movie only. The cavity view is unchanged.
- No solver change, no re-run of any job, no schema change.
- The check's own fixture uploads TWO parts, the second at negative x, so the
  parts' combined minimum is not the upload's origin: an outline drawn in the
  grid's frame, or shifted to the box's centre, cannot pass its placement check.
  `tests/test_web_routes.py`'s outline box is likewise neither centred nor
  cubic, and the mirrored real job (the owner's film job, whose plates are
  offset inside their domain) is the same case on production's own artifacts --
  that mirror can only be fetched after this change is deployed, because the
  outline URL it needs is served by the app this change adds.

## Verified

- Tier 1: `tests/test_web_routes.py` (the outline's place, all parts, the box
  compared in metres), `tests/js/test_viewer_surface.mjs` (the movie draws the
  group it built, and the loader's happy path is asserted, not only its
  failures), `tests/test_run_tdcheck.py` (a mirror without the outline is
  refused before docker runs).
- Browser: `scripts/run_tdcheck.sh` against the non-centred fixture mirror, and
  against a mirror of a real job, sampling the first, middle, last and
  Play-reached frames.
