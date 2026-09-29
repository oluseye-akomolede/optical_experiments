# render-check

## ADDED Requirements

### Requirement: The check reads the canvas, never a screenshot

The check SHALL capture each frame by reading the viewer's canvas
(`document.querySelector('#viewer canvas').toDataURL('image/png')`) inside a
single `page.evaluate`, after an init script forces `preserveDrawingBuffer: true`
on every `getContext` call. It SHALL NOT call `page.screenshot`.

A compositor screenshot waits on a frame the CPU-limited pod may never present;
measured, `page.screenshot` timed out at 30 s in the pod and at 120-400 s under
load on the VM. The canvas read returns the last presented frame without waiting
on the compositor. Source: `dockerfiles/rendercheck/check.mjs`.

#### Scenario: a frame is captured without a compositor round trip
- **WHEN** the check captures any frame of a finished job
- **THEN** it reads the canvas through `toDataURL('image/png')`
- **AND** no `page.screenshot` call exists in the check's code

#### Scenario: a screenshot has crept back in
- **WHEN** the check's code contains a `page.screenshot(` call
- **THEN** the callsite test fails, naming the pod timeout the read replaced

### Requirement: A frame is judged by the field it draws, not by the object's box

A frame SHALL be judged by the HUE it paints in a lobe colour (`warm` + `cool`,
where hue is assigned only to lit pixels), not by its lit-pixel count. The
lit-pixel count (`coloured`) is the object's silhouette and is measured constant
(91637 px on the reference job) with the field on and off, so it carries no
signal about the field. Source: `dockerfiles/rendercheck/check.mjs`.

#### Scenario: the object's box is constant across frames
- **WHEN** the same job is captured with the field on and with the field off
- **THEN** both frames report the same `coloured` count
- **AND** the check does not treat that count as evidence the field drew

### Requirement: A frame that drew MUST differ from an outline-only control

For each frame the check has accepted as drawing, its lobe-hue count SHALL differ
from the outline-only control frame's count by reaching a floor of
`ceil(outline.coloured / 16)` -- a share of the CONTROL's own lit area, stable
under rotation because the control does not move. Source:
`dockerfiles/rendercheck/check.mjs`, `dockerfiles/rendercheck/pixels.mjs`.

#### Scenario: an outline-only page is caught
- **WHEN** the check runs against a page whose field never draws (every frame is
  the outline)
- **THEN** each frame that was meant to draw FAILS as outline-only
- **AND** the run exits non-zero

#### Scenario: a missing or unreadable control is a FAIL, not a skip
- **WHEN** the outline-only control frame cannot be captured, or is empty, or is
  an all-black cleared buffer
- **THEN** the check FAILS and names every judgment the missing control prevents
- **AND** it does not silently drop the comparison

### Requirement: A frame drawn below the viewer's floor is not judged for hue

A frame SHALL NOT be routed to the lobe-hue judgment or the outline comparison
merely because it carries no hue. A signed component whose share of the mode's
peak `|E|` is under the viewer's own drawing floor is drawn as an almost empty
frame BY DESIGN. Source: `dockerfiles/rendercheck/check.mjs`.

Measured: mode 1's `Erho` has a share of 0.0826 against the viewer's 0.12 floor
(`web/static/viewer.js` `THRESHOLD`), so the shader discards every sample and a
hue-less frame is the correct picture.

#### Scenario: the below-floor frame passes when the page explains it
- **WHEN** mode 1's `Erho` carries no lobe hue AND its caption names the
  component's measured share and calls the emptiness physics, not a fault
- **THEN** the check accepts the near-empty frame as correct
- **AND** it reports this explicitly, not as a silent skip

#### Scenario: an unexplained empty frame still fails
- **WHEN** a frame carries no lobe hue and the page does NOT caption it as
  below-floor
- **THEN** the check FAILS it as outline-only

### Requirement: The check SHALL NOT claim a discrimination it does not make

The lobe-hue floor SHALL NOT be stated as catching a half-black lobe. Measured, a
half-black lobe keeps 13-21% of the object's area, above any floor that also
admits the faintest correct frame (rotated frame, 10.3%), so the populations
overlap. Source: `dockerfiles/rendercheck/pixels.mjs`.

#### Scenario: a half-black lobe is left to the other predicates
- **WHEN** a signed component's negative lobe is drawn black while its positive
  lobe keeps its hue
- **THEN** the check does not report this as the lobe-hue failure
- **AND** the case is left to the per-component digests and the outline
  comparison

### Requirement: The outline-only control SHALL be captured after the main page closes

The check SHALL capture the outline-only control frame only after the main page
has closed, so the control is the only live WebGL context. Two live contexts
under a 2-CPU limit starve the control's fetches; measured, ~75 s contended vs
~6 s alone. Source: `dockerfiles/rendercheck/check.mjs`.

#### Scenario: the control is captured alone
- **WHEN** the check captures the outline-only control
- **THEN** the main page has already been closed
- **AND** the control's summary is non-empty
