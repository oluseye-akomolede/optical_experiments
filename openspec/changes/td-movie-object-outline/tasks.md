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
- [ ] Measured green on a mirror of a REAL job (the owner's film job or another
      finished td job whose parts are offset inside their domain) -- the fixture
      cannot see a translation to its own box's centre, that mirror can.
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
      the archive: this change's `time-domain` delta is ADDED-only, so it is the
      one that creates `openspec/specs/time-domain/`; `td-movie-window` is still
      unarchived and its MODIFIED delta against that same (not yet existing)
      spec is refused if it is archived first.
- [x] Tier 1 green: `tests/test_td_solve.py tests/test_web_routes.py
      tests/test_run_tdcheck.py` 200 passed, and all 14 `tests/js/*.mjs`
      147 pass / 0 fail (each file run on its own: `node --test tests/js/`
      discovers nothing).
- [ ] Commit locally, report the sha to the lead. The lead pushes, deploys and
      re-mirrors a real job for the browser proof (the mirror needs prod).
