# Tasks

## 1. Spec and predicates

- [x] 1.1 Write `specs/render-check/spec.md` (ADDED requirements: capture method,
      what counts as PASS, the timeout, and one scenario per FAIL reason).
      Verify: `openspec validate fix-rendercheck-capture --strict` passes.
- [x] 1.2 Add `outlineOnlyProblem(field, outline)` to
      `dockerfiles/rendercheck/pixels.mjs`: returns a failure when a mode frame
      carries no more lobe hue (`warm`/`cool`) than the outline-only frame, or
      when the outline frame itself is empty (an empty control cannot be a
      control). It SHALL judge hue, not the lit-pixel count or the digest — the
      count is the object's box and is identical with the field on and off.
      Verify: unit test in 2.1.

## 2. Tests (write them red first)

- [x] 2.1 Extend `tests/js/test_rendercheck_predicates.mjs` with cases that call
      an outline-only frame PASS before the fix and FAIL after; include the
      negative control (a real field frame differs from the outline and passes).
      Verify: the predicate test reddens on the pre-fix predicate; run
      `node --test tests/js/*.mjs` (the glob) and quote the counts.
- [x] 2.2 Add a case that an all-black control frame is reported as a blank
      control and NOT as "the frame is the outline only".
      Verify: removing the `sum === 0` guard from `outlineOnlyProblem` reddens
      this test and only this test.

## 3. Capture and the control frame

- [x] 3.1 Add the `?field=off` hook to `web/static/viewer.js` so the volume
      stays hidden when the query param is set; document it.
      Verify: a manual load of `/jobs/<id>?field=off` shows the outline and no
      field (checked in the local proof run).
- [x] 3.2 Replace every `page.screenshot` in `check.mjs` with the canvas
      `toDataURL('image/png')` read (the `preserveDrawingBuffer` addInitScript
      patch from the lead's `grab.mjs`); decode the data URL to a PNG buffer.
      Verify: local Docker run (3.4) completes with no timeout.
- [x] 3.3 Capture the outline-only frame (`?field=off`) and judge every mode
      frame against it with `outlineOnlyProblem`; keep the existing
      component-vs-component and rotate-vs-static comparisons.
      Verify: the local run's PASS/FAIL output (3.4).
- [x] 3.4 Force `preserveDrawingBuffer` on EVERY page the check reads, including
      the control page. Discovered while proving 4.2: the control page was
      created without the init script, so its canvas read returned a cleared
      buffer (`coloured` == the whole frame, `sum` == 0) and the check failed
      every mode frame against a control made of nothing.
      Verify: unit test in 2.2 and the control's summary in the 4.2 run.
- [x] 3.5 Make `outlineOnlyProblem` catch a blank (all-black) control, which the
      `coloured === 0` empty-control guard cannot see.
      Verify: unit test in 2.2.

## 3b. The diverging-ramp predicate, and when the control is captured

- [x] 3b.1 Fix `divergingProblem`'s denominator. It divided the frame's hue by
      the frame's own `coloured`, which is the object's box (91637) and not the
      field, so every real signed frame failed it -- measured: `|E|` 23205/91637
      = 25%, `Ez` 37768/91637 = 41%, `Erho` mode 2 23810/91637 = 26%, rotated
      9428/81268 = 12%, all below the old 50% floor. It SHALL judge hue against
      the outline-only control's own area (`share * outline.coloured`), which is
      stable under the camera rotation the rotated frame applies, and SHALL fail
      closed with no usable control.
      Verify: unit test in 3b.3; the pre-fix predicate reddens it.
- [x] 3b.2 State the limit honestly: a floor on hue/area CANNOT catch a
      half-black lobe, because the measured correct rotated frame sits at 10.3%
      and a half-black lobe still clears any lower floor. The check SHALL NOT
      claim otherwise; the black-lobe case is carried by `outlineOnlyProblem`
      and the component-vs-component comparisons.
      Verify: the "WHAT IS NOT CLAIMED" test in 3b.3.
- [x] 3b.3 Replace the invented `divergingProblem` fixtures (which used small
      `coloured` values that were mostly hue -- 65291/49027, 20000/18500 -- and
      so matched a predicate that failed every real frame) with the measured
      frames above, and add a case that every measured correct frame passes.
      Verify: `node --test tests/js/*.mjs` green; reverting `divergingProblem`
      to the old denominator reddens the correct-frame case.
- [x] 3b.4 Capture the outline-only control AFTER the main page is closed, so it
      is the only WebGL context in the 2-CPU container. The failed run's four
      "the control frame has no lit pixels" failures came from the second
      context's fetch being starved (~75 s contended vs ~6 s alone). Defer the
      outline-dependent verdicts (`fieldVsOutline`, `diverging`,
      `outlineOnlyProblem`) to after `page.close()`.
      Verify: the control's summary is non-empty in the 4.1 run.
- [x] 3b.5 When the control cannot be captured at all, FAIL and name every
      judgment that could not be made, so a missing control is visible, not a
      silent skip.
- [x] 3b.6 Do NOT route a below-floor frame to the lobe-hue predicate. A signed
      component whose share of the mode's peak |E| is under the viewer's own
      drawing floor is drawn almost empty BY DESIGN and correctly carries no
      hue. Measured: mode 1's Erho has a share of 0.0826 against the 0.12 floor,
      so routing it to `divergingProblem` reported two false failures on an
      otherwise correct run.
      Verify: the call-site test in `tests/js/test_rendercheck_callsites.mjs`
      ("the below-the-floor frame is NOT routed to the lobe-hue judgement",
      green; it reddens if a `diverging(erho, ...)` call is restored), and
      the 4.1 run no longer reporting diverging failures for mode 1's Erho.

## 4. Local proof and validation

- [x] 4.1 Run the `mse-rendercheck` image with `docker run --cpus=2 --memory=3g`
      against a local web server with a finished job: show a PASS with no
      timeout. (2g is NOT enough: at 2g the renderer died on frame 2 with
      "Target page, context or browser has been closed" in `readCanvas`;
      the same image at 3g completed all five frames. The cap is raised in
      the proof, not in the check.)
      Verify: EXIT=0, five PNGs, no timeout, `report.json` with zero problems.
- [x] 4.2 Show an outline-only page (field hidden) FAILs the new predicate while
      the real page passes.
      Verify: the run's stdout names the outline-only frame.
- [x] 4.3 `openspec validate fix-rendercheck-capture --strict` passes; commit
      locally and report the sha (no push).
