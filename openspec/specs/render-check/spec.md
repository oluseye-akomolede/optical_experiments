# render-check Specification

## Purpose
The render check loads a real job page in a headless browser inside the cluster and judges every mode frame against a field-off control, so a viewer that draws nothing, or only the outline, fails the run instead of passing on HTTP 200. Accepted defects are named as known issues, reported on every run, and do not fail it.

## Requirements

### Requirement: The check grants no exemption for an empty field frame

The render check SHALL NOT excuse a field frame that matches the outline-only
control. A field frame whose lobe-coloured pixels do not differ from the control
SHALL FAIL, for every component and every mode, unless that frame is named
individually as a known issue (see the next requirement). No caption, share
value, or predicate relaxation shall be accepted as a reason to pass. Source:
`dockerfiles/rendercheck/check.mjs`.

An `accountForEmptyFrame` exemption previously allowed this: when the page
captioned a component as drawn below its floor, a frame matching the outline
control was accepted as "correct here". That gate was the defect: it made the
check's verdict depend on the page's own prose, so it would also forgive any
component that drew nothing while printing a plausible caption. The frame it
forgave was a real defect -- mode 1's `Erho` drew as a bare outline because the
viewer scaled the component by its maximum sample, and the caption's half-truth
(the field is faint) was taken as the whole story (drawing as nothing is not the
physics). A caption is not a control. That exemption is removed; the viewer fix
(see the `mode-viewer` change) removes the max-normalisation cause. Source:
`dockerfiles/rendercheck/pixels.mjs`.

#### Scenario: a field frame that matches the outline control fails
- **WHEN** a captured field frame's lobe-coloured pixels do not differ from the
  outline-only control frame
- **AND** that frame is not listed as a known issue
- **THEN** the check FAILS that frame
- **AND** it does not accept a caption or a share value as a reason to pass

#### Scenario: an absent control is a failure, not a skip
- **WHEN** no `MSE_OUTLINE_URL` is set, so the control frame cannot be captured
- **THEN** the field-versus-outline verdict FAILS rather than being skipped
- **AND** the failure names that the control is missing

### Requirement: A known issue is waived by name, and reported as KNOWN

The check SHALL be able to waive an accepted defect for ONE named frame, and
only that frame. The waiver SHALL be keyed on the check's own label for the
frame (the same string the verdict prints), SHALL be listed in source, and SHALL
be evaluated in a single place so that no other frame can inherit it. A waived
frame SHALL be reported with the word `KNOWN`, distinct from `ok`, SHALL record
its one-line reason, SHALL NOT appear as a passing `ok`, and SHALL NOT set the
run's exit code. Every other frame SHALL be judged by the unrelaxed predicate.
Source: `dockerfiles/rendercheck/pixels.mjs` (`KNOWN_ISSUE_FRAMES`, `knownIssue`)
and `dockerfiles/rendercheck/check.mjs`.

The one entry at present is mode 1's `Erho`. Its field is sign-mixed, so along a
view ray the two lobes blend to a near-neutral haze: after the p99-scale fix the
frame draws (lit and framed) but paints almost no hue, and the lobe-hue predicate
reads it as the outline. Mode 1 is `z`-polarised, so `Eρ` is about 8% of the
mode's peak `|E|` -- the faint frame the caption names. The owner accepted this
as a minor cosmetic nit on 2026-09-30. Because the waiver is keyed on the frame's
own name, a second empty frame -- a different component, or the same component in
another mode -- is judged normally and still FAILS.

#### Scenario: the named frame is waived and reported as KNOWN
- **WHEN** the frame labelled `Erho in mode 1` matches the outline-only control
- **THEN** the check prints a `KNOWN` line naming the frame and its reason
- **AND** it does not print an `ok` line for that frame
- **AND** the run's exit code stays 0 for that frame

#### Scenario: the waiver covers only the named frame
- **WHEN** any other frame matches the outline-only control
- **THEN** the check FAILS that frame, exactly as if no waiver existed
- **AND** a near-miss label that is not in the list gets no waiver

#### Scenario: a non-frame failure is never waived
- **WHEN** any failure other than a waived frame's lobe verdict occurs (a missing
  control, a blank control, a selector that does not change the picture, an
  empty caption, a framing failure, a console error)
- **THEN** the check FAILS the run
- **AND** the known-issue list does not reach it

### Requirement: The page MUST keep presenting frames for the check to read

The viewer SHALL NOT pace its render loop on the consumer end of a
double-buffered WebGL context. Measured, in the 2-CPU container against a
finished job: the check's read backpressures the render loop drive the page's
`requestAnimationFrame` callback to 0.4-0.7 Hz, against a 60 Hz ceiling measured
on a blank page in the same container, and a canvas read that takes 104-129 s
(plus four frames per 10 s on a `?field=off` page). The canvas read is therefore
a very expensive task in this container regardless of backpressure, and a task
that awaits one cannot itself be driven by `requestAnimationFrame`.

Equivalently, the check's own reads SHALL NOT be issued from inside a
`requestAnimationFrame` callback, and SHALL NOT await any work that depends on
one. By the time a read resolves, the buffer it reads has been presented and
cleared.

#### Scenario: a check read resolves without waiting on the page's render loop
- **WHEN** the check captures a frame inside the 2-CPU container
- **THEN** the read does not await a `requestAnimationFrame` callback on the
  page
- **AND** the read's decoded frame is the canvas backing store's dimensions

#### Scenario: a short wait is enough once nothing backpressures the loop
- **WHEN** the check settles for a fixed short delay after the canvas exists
- **THEN** the page is presenting frames by then, because nothing in the check
  is holding the loop

### Requirement: The check captures frames by reading the viewer canvas

The render check SHALL capture each frame by reading the viewer's own canvas
with `document.querySelector('#viewer canvas').toDataURL('image/png')`, after
forcing `preserveDrawingBuffer: true` on every canvas context via
`addInitScript`. It SHALL NOT use `page.screenshot` or `locator.screenshot` for
the pixel path. The reason is measured, not stylistic: `page.screenshot({clip})`
times out in the CPU-limited pod (`Timeout 30000ms exceeded`, reported from
`shoot`) and on the loaded VM (at 120 s and 400 s), while the canvas read never
has; the canvas read never waits on a compositor frame.

The decoded data-URL frame is the canvas pixels only. It therefore excludes the
page chrome, so every pixel predicate operates on the plot alone.

#### Scenario: a finished job's frame is captured without timing out
- **WHEN** the check loads a job page whose plot is drawing, under a 2-CPU
  container limit
- **THEN** each of the check's frames is captured by the canvas read
- **AND** no frame reports a `page.screenshot: Timeout` failure.

#### Scenario: the capture path leaves the page chrome out of the frame
- **WHEN** the canvas read returns
- **THEN** the frame's dimensions equal the canvas element's backing-store
  `width` and `height`, not the viewport's

#### Scenario: the control frame is read on the same path as every other frame
- **WHEN** the outline-only control frame is captured
- **THEN** its pixels are read by the shared capture helper, not by a second
  implementation
- **AND** the control reports the outline's lit-pixel extent, and not zero,
  which is what reading a cleared buffer returns

#### Scenario: every page the check reads has the drawing buffer preserved
- **WHEN** a second page (the outline-only control) is created after the first
- **THEN** the `preserveDrawingBuffer` init script is registered on that page
  before it navigates, exactly as on the first
- **AND** this holds for every page the check reads, because a page that reads
  its canvas without the flag returns a cleared buffer

  Measured: the control page was created without the init script once, and its
  canvas read returned `{coloured: 294402, sum: 0}` — every pixel black, which
  is a cleared buffer and not an outline. The empty-control guard tests
  `coloured === 0` and cannot see it, so the check failed every mode frame
  against a control made of nothing, reporting "the frame is the outline only"
  for a field that had drawn.

#### Scenario: an all-black control frame is reported as a broken control
- **WHEN** the control frame's pixels sum to zero (every pixel black, so the
  drawing buffer was cleared before the read)
- **THEN** the check reports the control as a blank image that is not an outline
- **AND** it does not report "the frame is the outline only", which would blame
  the mode frame for a broken control

#### Scenario: the control capture waits on the data the page needs, not only on a selector
- **WHEN** the control page is a second WebGL context created while the main
  page's context is live, under a 2-CPU limit
- **THEN** the capture waits for the job's `index.json` to arrive before
  awaiting the viewer's visibility
- **AND** if the page fails to become visible it reports which of the job's
  resources did arrive, rather than reporting only that the control had no
  pixels

  Measured, in a run capped at 2 CPUs: the control page's three data fetches
  (`index.json`, `files/surface.glb`, `field/1.bin`) took ~75 s from load to
  arrive under contention, against ~6 s for the same page when it was the only
  container racing the gate. The 120 s selector wait expired before the first of
  them landed, on a page that had loaded every byte of its code — so
  `#viewer:not([hidden])` could never match and the wait expired. The selector
  was a proxy for readiness that this page can reach too late or not at all
  under a shared CPU cap, and the failure surfaced as "the control frame has no
  lit pixels" — could-not-ask reported as nothing-there.

### Requirement: The viewer SHALL expose a field-visibility switch for the control

The mode viewer SHALL accept a `?field=off` query parameter that keeps the field
volume hidden while leaving the object's outline and every other element drawn.
The check uses it to capture its outline-only control from the SAME page and
code path as the frames it judges.

This is a production-viewer change and is stated here rather than left implicit.
It is required because the alternative — a control page whose field merely failed
to load — is not a control: a fetch that hangs and a fetch that was told not to
run are indistinguishable in the result, so a control built from a failing fetch
cannot tell "the field is absent" from "the field is broken". A switch makes the
control's state a decision rather than an accident. It is also usable by a human
("show me the outline without the field") and is inert when the parameter is
absent.

#### Scenario: the field-visibility switch is inert without the parameter
- **WHEN** the viewer is loaded without `?field=off`
- **THEN** the field draws exactly as it did before the parameter existed

#### Scenario: the switch hides the field and nothing else
- **WHEN** the viewer is loaded with `?field=off`
- **THEN** the field volume is not drawn
- **AND** the object's outline is still drawn, so the frame is a usable control

### Requirement: A mode frame MUST carry field hue the outline-only frame lacks

The check SHALL capture an outline-only control frame — the same page with the
field hidden (`?field=off`) — and SHALL require each mode frame to carry hue
(`warm`/`cool` pixels, the two lobes of the diverging ramp) that the control
does not. A frame whose hue matches the control's is a FAIL, because it is a
page whose field did not draw. The control frame SHALL itself be non-empty; an
empty control cannot serve as a control and is a FAIL.

The comparison SHALL be on hue and SHALL NOT be on the lit-pixel count or the
pixel digest. This is measured, not stylistic: putting the field mesh's
visibility on and off with everything else fixed leaves the lit count at
`91637` in both states and changes only `warm`/`cool` (`5809`/`17396` on,
`0`/`0` off). The lit count is the object's box and appears in every frame, so a
count-based comparison agrees by construction and discriminates nothing; the
field's whole contribution is the hue it paints over that box.

This exists because the previous judgement counted any pixel differing from the
background, and the outline is such a pixel: every frame reported the same
`91637` coloured pixels and a `400x337` box, so an outline-only page passed.

#### Scenario: an outline-only frame is reported as FAIL
- **WHEN** the field never draws and the frame contains only the object's
  outline, carrying no lobe hue
- **THEN** the check reports that the frame does not differ from the outline-only
  frame
- **AND** the run exits non-zero

#### Scenario: a real field frame differs from the outline and passes
- **WHEN** the field draws over the outline
- **THEN** the frame carries more lobe-coloured pixels than the control
- **AND** the check does not report a field-versus-outline failure

#### Scenario: a frame with the control's lit count but no hue is a failure
- **WHEN** a frame reports the same lit-pixel count as the control and its
  pixels are the outline's, carrying no lobe hue
- **THEN** the check reports the frame as the outline only, even though its
  pixel digest differs from the control's

#### Scenario: an empty control frame is itself a failure
- **WHEN** the outline-only frame has no lit pixels (the outline did not draw
  either)
- **THEN** the check reports the control frame as empty rather than passing
  every mode frame against it.

#### Scenario: a blank control frame is a failure even though it is not empty
- **WHEN** every pixel of the control frame is black, so the frame is all-"lit"
  by the background-distance test but sums to zero
- **THEN** the check reports the control as a blank image rather than using it
  as the floor for every mode frame.

#### Scenario: a missing control frame is a failure, not a skipped check
- **WHEN** `MSE_OUTLINE_URL` is unset, or the control page cannot be read
- **THEN** the check reports the field-versus-outline comparison as failed
- **AND** it does not silently pass every mode frame

### Requirement: A frame the check has decided drew MUST carry lobe hue

For a frame the check has already accepted as drawing (it cleared the draw
floor), the check SHALL require its lobe-coloured pixel count (`warm` + `cool`)
to reach a floor measured against the CONTROL's own lit count — the object's
area — not against the frame's own count. The floor SHALL be a small share of
the control's count, the same instrument the field-versus-outline predicate
already uses, because a black-clamped pixel is a lit pixel carrying neither lobe
colour and so lowers this count.

The check SHALL apply this predicate ONLY to frames it has established are meant
to be drawn. It SHALL NOT route a frame to this predicate merely because that
frame carries no hue: a signed component whose measured share of the mode's peak
|E| sits under the viewer's own drawing floor is drawn as an almost empty frame
BY DESIGN, and the hue-less frame is its correct picture. Measured on the live
job: mode 1's `Erho` has a share of `0.0826` against the viewer's `0.12` floor,
so the shader discards every sample and the frame legitimately carries zero hue
while its lit count still reports the outline's `91637`. Routing that frame here
reports "the component is drawn in neither end of the diverging ramp", which is
false, and it produced two of the three failures on an otherwise correct run.
The question that applies to such a frame is "did it draw at all", which the
field-versus-outline comparison answers; the check SHALL keep that comparison on
it.

The floor SHALL be stated with its limit: a floor on the lobe count catches the
case where the hue is gone, but it does NOT by itself catch a signed component
whose NEGATIVE lobe is drawn black, because the positive lobe's hue remains
above any floor low enough to pass the faintest legitimate frame. Measured: a
correct frame that reaches this predicate carries between 11% and 41% of the
object's area in lobe hue depending on the component and the camera, so a floor
that passes the faintest correct frame leaves the black-lobe case to the other
predicates (the per-component digests, and the field-versus-outline comparison).
The check SHALL NOT claim a discrimination it does not make.

#### Scenario: a frame whose hue is gone is a failure
- **WHEN** a frame that was meant to draw carries no lobe-coloured pixels at all
- **THEN** the check reports that too little of the object is drawn in a ramp
  colour, against the object-area floor
- **AND** the run exits non-zero

#### Scenario: the faintest legitimate frame still passes
- **WHEN** a real signed component carries the fewest lobe-coloured pixels any
  measured correct frame shows
- **THEN** the check does not report a diverging failure for it

#### Scenario: a frame drawn below the viewer's floor is not judged for hue
- **WHEN** a signed component's measured share of the mode's peak |E| is under
  the viewer's drawing floor, so the frame is drawn almost empty and carries no
  lobe hue
- **THEN** the check does not report a diverging failure for it
- **AND** the check still requires the frame to differ from the outline-only
  control, so a field that never drew is still a FAIL

#### Scenario: a black lobe alone is not claimed to be caught
- **WHEN** a signed component's negative lobe is drawn black while its positive
  lobe keeps its hue
- **THEN** the check does not report this as the diverging failure, because that
  case is left to the per-component and field-versus-outline predicates

### Requirement: The outline-only control SHALL be captured after the main page closes

The check SHALL capture the outline-only control frame while it is the only
WebGL context in the container: it SHALL close the main page before opening the
control page, so the two contexts do not contend for the CPU the container is
limited to. The verdicts that depend on the control SHALL be evaluated after the
control is captured.

This is required because two live WebGL contexts under a 2-CPU cap make the
control page's data fetches take far longer than they do alone — measured at
about 75 s contended against about 6 s uncontended — so the control's selector
wait expires and the control reports no pixels. Waiting longer does not fix
contention; removing the contention does.

#### Scenario: the control is captured with no other context live
- **WHEN** the check captures the outline-only control frame
- **THEN** the main page's context has already been closed
- **AND** the control's data fetches complete within the timeout

#### Scenario: the deferred verdicts still run
- **WHEN** the control is captured at the end of the run
- **THEN** the field-versus-outline and diverging verdicts are still evaluated
- **AND** their results appear in the report

### Requirement: The per-component frames MUST differ from each other

The check SHALL require the frame for one component (for example `Ez`) to differ
in its pixel digest from the frame for another (`Erho`), and from `|E|`. Two
byte-identical frames mean the selector reached nothing.

#### Scenario: two components draw identical frames
- **WHEN** changing the component selector leaves a frame with the same digest
  as the previous component
- **THEN** the check reports that the selector is not reaching the texture.

### Requirement: The check is bounded by MSE_TIMEOUT_MS

Each wait the check performs SHALL use `MSE_TIMEOUT_MS` (default `120000`, from
the environment) or a value below it. The check SHALL exit `0` only when no
predicate reported a failure, and non-zero otherwise, and the exit code SHALL be
set in exactly one place.

#### Scenario: the run's exit status is the result
- **WHEN** any predicate fails
- **THEN** the process exits non-zero
- **AND** the reason is printed to stderr.
