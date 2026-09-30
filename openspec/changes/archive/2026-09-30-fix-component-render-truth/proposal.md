# fix-component-render-truth

## Why

The component view does not draw what it claims. On the reference job (mode 1,
`Erho`), the page showed a bare outline — the volume was absent to the pixel —
while the caption said the component was merely faint.

The cause was measured, not inferred. The viewer normalised each component by
its **maximum** sample (`peak = max|scalar|`, then `scalar[i] /= peak`). Mode 1's
`Erho` is a spiked field: absolute max 0.4236, 99th percentile 0.081, median
0.014, and — measured from the page's own uploaded array — **94% of its 77,400
samples sit under the 0.12 drawing floor** after that division. Only the spike
clears the floor, and along the view diagonal through the flattened box the
accumulated opacity sits on the shader's discard, so the frame draws nothing.
The captured `Erho` frame was `sum` 22,033,282 against the field-off outline
control's 22,033,186 — identical to 0.0004%. `Ez` and `|E|` on the same page drew
normally (37,768 and 23,205 lobe-coloured pixels), so the shader, the ramp, the
signed path, the texture and the payload were all sound.

The code also contradicted its own docstring, which said a component's single
largest sample can be "a local spike on an otherwise small field" and "the p99 is
what the eye sees" — while the normaliser divided by that very spike.

The render check then **forgave** the empty frame: an `accountForEmptyFrame`
exemption accepted an `Erho` frame that matched the outline control, because the
page captioned the component as below its floor. The caption was half right (the
field is weak in this mode) and half wrong (drawing as nothing is the defect). A
caption is not a control, and the exemption hid a real bug.

## What changes

1. **The viewer scales each component by its own robust percentile (p99), not its
   max, and clamps only the signed components.** The scale moves the sharpest
   0.5% of samples to the ends; everything at or beyond the p99 saturates. This
   is the owner's chosen behaviour: a component is drawn on its **own** scale so
   its shape is legible, and the caption carries how strong it is against the
   mode's `|E|` peak. Source: `web/static/viewer.js`, `web/static/components.js`.
   The scale and the normalisation are extracted into `robustScale` /
   `normaliseComponent` in `components.js` so they can be tested without a browser.

2. **The render check's `accountForEmptyFrame` exemption is removed.** An empty
   lobe frame must FAIL. With the fix, mode 1's `Erho` draws and is judged like
   every other frame: it must differ from the outline-only control's hue.

3. **The captions and comments are made true.** All three viewer caption branches
   and the `components.js` docstring claimed "one shared scale". The view draws
   each component at its own scale and captions its share of the mode's `|E|`
   peak; the text now says that.

4. **The render-check Job manifest carries the control URL and a bounded
   container.** `MSE_OUTLINE_URL` (the same job URL plus `?field=off`) is added —
   without it the check has no control and fails, not skips. Memory is raised to
   3Gi for both request and limit (2Gi OOM-kills the renderer on the second
   frame), and `OMP_NUM_THREADS` / `MKL_NUM_THREADS` / `OPENBLAS_NUM_THREADS` are
   pinned to the 2-CPU limit so SwiftShader's pool does not start one thread per
   host core.

## Out of scope

- Changing which colour ramp a component uses, or the drawing floor's value.
- The photonics work (row `cd34cc4b`).
