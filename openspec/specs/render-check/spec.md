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
