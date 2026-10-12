# Tasks

## The outline itself

- [x] `web/outline.py`: the upload's every part, in metres, by the job's own
      `units` x `scale`; a refusal is an `OutlineError`, never a traceback.
- [x] `web/app.py`: `outline_url` on a finished td job's payload, and the
      `/jobs/<id>/outline.glb` route over the stored `input.obj`.
- [x] `tests/test_web_routes.py`: the outline's place in metres (mm x scale=2),
      every part present, and the two-parts-same-name guard.
- [x] Mutation-proved: dropping `length_scale` reddens the placement test;
      keeping only the largest part reddens the multi-part test; dropping the
      name-uniqueness reddens its direct guard.

## The movie draws it

- [x] `web/static/viewer.js`: the movie draws the outline over the field on
      every frame, reusing `glbMeshParts` + `outlineGroup`, and captions any
      outline it could not draw.
- [x] `tests/js/test_viewer_surface.mjs`: the loader's happy path is asserted
      (a healthy surface reaches `scene.add`), so the absent-outline branches
      cannot pass at a loader that draws nothing.

## The check

- [x] `scripts/tdcheck_fixture.py`: the mirror's own upload (a two-part object,
      the second part at negative x so the parts' minimum is not the upload's
      origin) is written as the job's stored `input.obj`, and the real
      `/jobs/<id>/outline.glb` route is fetched into the mirror as
      `outline.glb` -- a 404 there stops the fixture rather than shipping a
      mirror that cannot show an outline.
- [x] `scripts/tdcheck_fetch.py`: the outline is fetched into a mirror of a real
      job, and a mirror whose job predates the outline is refused.
- [x] `scripts/run_tdcheck.sh`: `outline.glb` is in the pre-check list, and
      `tests/test_run_tdcheck.py` derives that list from what tdcheck reads.
- [x] `dockerfiles/rendercheck/tdcheck.mjs`: the outline route arm (with a 404
      arm as the control), the stated box against the manifest's own positions
      within one grid cell, and the same frame drawn with and without the
      outline on every sampled frame including one Play reached.
- [x] Measured green on the fixture mirror (51 checks, ALL PASS, exit 0): the
      stated box (-2..4 x 0..3 x 0..2 mm) and the manifest's own positions agree
      exactly, well inside one cell (0.25 mm), on an object whose parts do not
      start at the upload's origin; and each of the first frame, frame 61, the
      last frame and the frame Play reached differs by 44934 of its 834x331
      pixels when the outline is not drawn, against a control that still draws
      1622..20748 lit pixels of field (the floor is 1/64 of those, 26..325).
- [x] Measured green on a mirror of a REAL job (the owner's film job or another
      finished td job whose parts are offset inside their domain) -- the fixture
      cannot see a translation to its own box's centre, that mirror can.
      Lead, 2026-10-12 13:1xZ, main 2fb19da against prod mse-web:5aaf9f0: a
      fresh mirror of the owner's job 180cb65f (60 frames, two-digit names, open
      boundary) through `scripts/tdcheck_fetch.py` and
      `scripts/run_tdcheck.sh --dir`: 53 PASS, 0 FAIL, ALL PASS, exit 0, under
      the movie-derived cap. The placement check held within one cell
      (9.09e-9 m): page x 0..1.0e-6, y 0..8.0e-7, z 0..2.5e-7 m, the outline
      file's own box, inside the sampled field (x -3.2e-7..1.32e-6 m).
- [x] Mutation-proved in the browser against that same mirror, one mutant at a
      time, source confirmed present in each run before it was judged:
      `scene.add(group)` commented out -- the four pixel checks FAIL, each
      naming its frame (`0 pixels ... differ`, the control still drawing
      1622..20748 lit px), while the placement check stays green, which is why
      both halves are needed; a `group.position.x = 8e-3` translation -- exactly
      one FAIL, the placement check naming `x min` and `x max` with an 8 mm
      error against the one-cell 0.25 mm tolerance.

## Spec and hand-off

- [x] `openspec validate --all --strict` (16 items passed, 0 failed). Note for
      the archive (corrected by the lead, 2026-10-12): `time-domain-fdtd` is the
      change that creates `openspec/specs/time-domain/` and archives first; this
      change's `time-domain` delta is ADDED-only and goes next; `td-movie-window`
      follows, since its MODIFIED delta needs the requirements `time-domain-fdtd`
      adds.
- [x] Tier 1 green: `tests/test_td_solve.py tests/test_web_routes.py
      tests/test_run_tdcheck.py` 200 passed, and all 14 `tests/js/*.mjs`
      147 pass / 0 fail (each file run on its own: `node --test tests/js/`
      discovers nothing).
- [x] Commit locally, report the sha to the lead. The lead pushes, deploys and
      re-mirrors a real job for the browser proof (the mirror needs prod).
      Shipped as 8e4c659 + 5aaf9f0, live as mse-web:5aaf9f0 (2026-10-03).
