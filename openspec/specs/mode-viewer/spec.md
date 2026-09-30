# mode-viewer Specification

## Purpose
How a job's page shows its results: the object, each mode's frequency, and its field components drawn as a volume plot with captions that say how strong each component is.

## Requirements

### Requirement: A component is normalised by its own peak and captioned with its share

The viewer SHALL normalise each rendered component by ITS OWN peak before upload,
so the value the shader sees spans `-1..1` for a signed component (`0..1` for the
magnitude |E|) and the drawing floor is a contrast
floor ("within a fixed ratio of this component's strongest sample"), not an
amplitude floor that varies with the mode. The viewer SHALL caption the component
with its measured share of the mode's peak `|E|`, computed as a high percentile
of the component against `|E|`, so the caption reports how small the component is
*in this mode* rather than how it was scaled. Source: `web/static/viewer.js`,
`web/static/components.js`.

Measured on the reference job: in mode 1 (z-polarised) `Eρ` reaches only ~8% of
the mode's peak `|E|` (its own peak is ~0.42 of `|E|`); in mode 2 it reaches ~96%
(own peak ~0.99). Normalising by the component's own peak is what makes the two
frames comparable; the caption's share is what makes the difference legible.

#### Scenario: a component is small in the mode
- **WHEN** a component's share of the mode's peak is below the viewer's drawing
  threshold
- **THEN** the viewer still normalises the component by its own peak, so its
  strongest sample is drawn
- **AND** the caption names the share and says the smallness is the mode's
  physics, not a fault

#### Scenario: a component is normalised by its own peak
- **WHEN** the viewer builds the texture for a component
- **THEN** the values uploaded span `-1..1` for a signed component (`0..1` for
  |E|), with the component's own peak at
  `1.0`
- **AND** the drawing floor is applied to that own-peak-normalised value

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
