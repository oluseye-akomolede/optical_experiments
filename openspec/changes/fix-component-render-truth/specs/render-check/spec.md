# render-check

## ADDED Requirements

### Requirement: The check grants no exemption for an empty field frame

The render check SHALL NOT excuse a field frame that matches the outline-only
control. A field frame whose lobe-coloured pixels do not differ from the control
SHALL FAIL, for every component and every mode, with no caption-based or
share-based exemption. Source: `dockerfiles/rendercheck/check.mjs`.

An `accountForEmptyFrame` exemption previously allowed this: when the page
captioned a component as drawn below its floor, a frame matching the outline
control was accepted as "correct here". The frame it forgave was a real defect --
mode 1's `Erho` drew as a bare outline because the viewer scaled the component by
its maximum sample, and the caption's half-truth (the field is faint) was taken
as the whole story (drawing as nothing is not the physics). A caption is not a
control. With the viewer fixed (see the `mode-viewer` change), the component
draws and the frame is judged like any other.

#### Scenario: a field frame that matches the outline control fails
- **WHEN** a captured field frame's lobe-coloured pixels do not differ from the
  outline-only control frame
- **THEN** the check FAILS that frame
- **AND** it does not accept a caption or a share value as a reason to pass

#### Scenario: an absent control is a failure, not a skip
- **WHEN** no `MSE_OUTLINE_URL` is set, so the control frame cannot be captured
- **THEN** the field-versus-outline verdict FAILS rather than being skipped
- **AND** the failure names that the control is missing
