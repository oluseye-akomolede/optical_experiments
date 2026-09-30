# Fix the render check's capture and its judgement of the field

## Why

The render check has one job: prove a finished job's 3-D field plot actually
draws. It fails at that job in two independent ways, and both were measured
against the live service.

1. **The capture hangs.** `check.mjs` takes every frame with
   `page.screenshot({ clip })`. In the cluster pod that call times out after
   30 s — `page.screenshot: Timeout 30000ms exceeded ... at shoot
   (check.mjs:231)` — and it times out on the loaded VM too, at 120 s and 400 s,
   while the *same image* passes in Docker on an idle VM. So the check cannot
   run where it is needed, and a check that cannot run is not a check.

2. **The judgement measures the box, not the field.** Every frame reports the
   same `91637` coloured pixels and a `400x337` box, because `summarise()`
   counts any pixel that differs from the background — and the object's
   wireframe outline is such a pixel. The predicates therefore PASS on an
   outline-only frame: a page whose field never drew reports green.

The owner's standing rule is that a check which cannot fail is worse than no
check, because it is trusted. Both defects are that shape.

## What changes

- **Capture.** Replace every `page.screenshot` with a canvas read:
  `addInitScript` forces `preserveDrawingBuffer: true` on
  `HTMLCanvasElement.prototype.getContext`, and the frame is read with
  `document.querySelector('#viewer canvas').toDataURL('image/png')`. This path
  never waits on a compositor frame, and it is the one the lead measured as
  never timing out where `page.screenshot` did.

- **Judgement.** The check now captures an **outline-only control frame** — the
  same page with the field hidden — and every mode frame is judged *against
  it*: a mode frame must carry the field's hue (`warm`/`cool` lobe pixels) where
  the control carries none, and the per-component frames must differ from each
  other. A frame whose hue matches the control is the failure, and it is now
  detectable because there is a frame to compare it to.

  The comparison is on hue and not on the lit-pixel count, because the count is
  the object's box: hiding the field mesh changes `warm`/`cool` from
  `5809`/`17396` to `0`/`0` and leaves the `91637` count untouched. Comparing
  counts would therefore agree by construction — the very defect being fixed.

- To capture the outline-only frame the viewer gains a small, documented hook:
  `?field=off` keeps the volume hidden. It is also a real debugging aid (show
  me my outline without the field), and it makes the control behavioural rather
  than a guessed brightness threshold.

- **A third defect, found while proving the second.** The signed-component
  predicate (`divergingProblem`) divided a frame's lobe hue by the frame's own
  `coloured` count — the object's box, `91637` — and required 50% of it. That is
  the same box-versus-field mistake, in the opposite direction: measured against
  the live frames, `|E|` reaches `23205/91637` = 25%, `Ez` `41%`, `Erho` mode 2
  `26%` and the rotated frame `9428/81268` = 12%, so **every correct frame
  failed** — a check that cannot pass, which is as untrustworthy as one that
  cannot fail. It now judges hue against the outline-only *control's* own area,
  which does not move when the camera rotates the frame, and fails closed when
  there is no usable control. It states plainly, in the code, the one thing it
  cannot catch (a single lobe clamped to black still clears any floor that a
  genuinely faint correct frame clears), rather than over-claiming.

- **The control is now captured after the main page closes.** It was captured
  while the main page's WebGL context was still live; in the 2-CPU container the
  second context's data fetch starved (~75 s contended vs ~6 s alone) and the
  control came back empty or cleared, failing every mode frame against it. The
  outline-dependent verdicts are deferred until after `page.close()`.

- A new capability spec, `render-check`, states what the check asserts, what
  counts as FAIL, and the timeout.

## Impact

- `dockerfiles/rendercheck/check.mjs` — capture and the new control frame.
- `dockerfiles/rendercheck/pixels.mjs` — a predicate that judges a field frame
  against an outline frame.
- `web/static/viewer.js` — the `?field=off` hook.
- `tests/js/test_rendercheck_predicates.mjs` — must redden when an outline-only
  frame is called PASS.
- Image: `mse-rendercheck`. The cluster Job is re-run by the lead after review;
  nothing here is pushed.

## Out of scope

- The thread-count hypothesis (os.cpu_count in a CPU-limited pod). It is not
  needed for the fix and is not tested here.
- Any change to what the viewer draws, beyond the `?field=off` hook.
