# mode-viewer

## MODIFIED Requirements

### Requirement: A component is normalised by its own peak and captioned with its share

The viewer SHALL normalise each rendered component by a robust percentile of ITS
OWN samples -- the 99th percentile of `|component|`, not the maximum -- before
upload, so the value the shader sees spans `-1..1` for a signed component (`0..1`
for the magnitude |E|) and the drawing floor is a contrast floor, not an
amplitude floor that varies with the mode. A signed component SHALL be clamped to
`-1..1` after the division; the magnitude |E| SHALL NOT be clamped. The viewer
SHALL caption the component with its measured share of the mode's peak `|E|`,
computed as a high percentile of the component against `|E|`, so the caption
reports how strong the component is *in this mode* rather than how it was scaled.
Source: `web/static/viewer.js`, `web/static/components.js`.

The scale is a percentile, not the maximum, because a component's single largest
sample can be a local spike on an otherwise small field. Measured on the
reference job, mode 1's `Erho` has an absolute maximum of 0.4236 but a 99th
percentile of 0.081 and a median of 0.014. Dividing by the maximum left 94% of
the field under the 0.12 floor and drew essentially nothing (the captured frame
matched the field-off control to 0.0004%); dividing by the p99 puts 62.8% of the
samples over the floor. In mode 1 (z-polarised) `Eρ` reaches only ~8% of the
mode's peak `|E|`; in mode 2 it reaches ~96%. Normalising by the component's own
robust scale is what makes the two frames comparable; the caption's share is what
makes the difference legible.

#### Scenario: a component is small in the mode
- **WHEN** a component's share of the mode's peak is below the viewer's drawing
  threshold
- **THEN** the viewer still normalises the component by its own robust scale, so
  its strongest samples are drawn
- **AND** the caption names the share and says the component is drawn at its own
  scale, so its smallness is the mode's physics and not a rendering fault

#### Scenario: a component is normalised by its own peak
- **WHEN** the viewer builds the texture for a component
- **THEN** the values uploaded span `-1..1` for a signed component (`0..1` for
  |E|), with the component's 99th percentile at `1.0`
- **AND** the drawing floor is applied to that normalised value

#### Scenario: a spiked component is not suppressed by its own spike
- **WHEN** a component has one sample far above the rest of its field (a spike of
  five times its 99th percentile)
- **THEN** the viewer scales by the p99, so that spike saturates at `1.0`
- **AND** the rest of the field is drawn, not pushed under the floor by the spike
- **AND** the caption still reports the component's share of the mode's `|E|`

#### Scenario: a signed component is clamped and the magnitude is not
- **WHEN** a signed component has samples beyond its 99th percentile
- **THEN** those samples are clamped to `-1..1` before upload
- **AND** the magnitude |E| is not clamped, because it has one sign

#### Scenario: a component with no finite samples is not scaled
- **WHEN** a component's field has no finite samples, or all its samples are zero
- **THEN** the viewer does not divide by a zero or a NaN scale
- **AND** the component draws no volume rather than a fabricated one
