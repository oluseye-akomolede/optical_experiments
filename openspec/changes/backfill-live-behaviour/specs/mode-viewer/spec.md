# mode-viewer

## ADDED Requirements

### Requirement: A faint component is captioned, never auto-scaled

The viewer SHALL draw every mode frame on ONE shared colour scale, so a
component's brightness is comparable to the mode's other components. It SHALL
NOT auto-scale a component up to fill the plot. A component faint on the shared
scale SHALL be captioned in plain words, naming its measured share of the mode's
peak `|E|`. Source: `web/static/viewer.js`.

Measured on the reference job: in mode 1 (z-polarised) `Eρ` reaches only ~8% of
the mode's peak `|E|`; in mode 2 it reaches ~96%. Sharing one scale is what makes
that difference visible.

#### Scenario: a component is faint
- **WHEN** a component's share of the mode's peak is below the viewer's threshold
- **THEN** the frame is drawn on the shared scale (not auto-scaled up)
- **AND** the caption names the share and says the faintness is the mode's
  physics, not a fault

### Requirement: A signed component is drawn with a diverging scale

A signed component SHALL be drawn with a diverging colour scale centred on zero:
a blue lobe for one sign, a red lobe for the other, and a distinct near-neutral
pale tone near zero. Source: `web/static/viewer.js`.

#### Scenario: a signed field is drawn
- **WHEN** the viewer draws a signed component
- **THEN** positive and negative regions take different lobe colours
- **AND** near-zero regions take the pale near-neutral tone, not the background

### Requirement: A mode label and a per-component selector

The viewer SHALL let the user choose a mode and a component, and SHALL label
what is shown, so a faint or empty frame is not mistaken for a broken one.
Source: `web/static/viewer.js`.

#### Scenario: a component is selected
- **WHEN** the user selects a component
- **THEN** the viewer draws it and names it in the caption
