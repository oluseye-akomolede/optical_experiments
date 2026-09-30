# tasks: fix-component-render-truth

## 1. Diagnose the empty Erho frame

- [x] 1.1 Reproduce the viewer's exact `Erho` pipeline on the page's real
  payload; confirm the array that reaches the shader and its distribution.
- [x] 1.2 Measure the captured frames under `docker --cpus=2 --memory=3g`:
  `Erho` matches the field-off outline control to 0.0004% (`sum` 22,033,282 vs
  22,033,186), while `Ez` (37,768) and `|E|` (23,205) draw.
- [x] 1.3 Rule out the shader, the signed ramp, the alpha/threshold, the
  texture build/binding, the payload and the decode — all sound, since `Ez` is
  signed and draws.
- [x] 1.4 Name the mechanism by effect, not by inference: a local, uncommitted
  p99 normaliser makes `Erho` draw hue where it drew none, with `Ez` and `|E|`
  unchanged. (A/B, reported to the lead.)
- [x] 1.5 Report where the over-floor samples sit — spread across the object's
  projection, not a corner.

## 2. Fix the normaliser

- [x] 2.1 Scale each component by its own p99 (`robustScale`), clamp only signed
  components to `-1..1`; guard the all-zero and all-NaN cases.
- [x] 2.2 Extract `robustScale` / `normaliseComponent` into `components.js` so
  they are testable without a browser; `viewer.js` imports them.
- [x] 2.3 Add a test that a single spike 5x the p99 does not suppress the rest
  of the field; confirm it FAILS when max-normalisation is restored.

## 3. Make the text true

- [x] 3.1 Rewrite the three `viewer.js` caption branches to say each component
  is drawn at its own scale, and report its share of the mode's `|E|` peak.
- [x] 3.2 Rewrite the `components.js` docstring (was "one shared scale").
- [x] 3.3 Fix the legend text that claimed every mode is normalised to `max|E|=1`.

## 4. Remove the exemption from the check

- [x] 4.1 Delete `accountForEmptyFrame` from `check.mjs`; an empty lobe frame
  now FAILS. Update its callsite so `Erho` is judged unconditionally.
- [x] 4.2 Update the render-check callsite tests to the new contract.

## 5. Manifest

- [x] 5.1 `manifests/mse/rendercheck-job.yaml`: add `MSE_OUTLINE_URL`
  (`...?field=off`), set memory to 3Gi (request and limit), add
  `OMP/MKL/OPENBLAS_NUM_THREADS=2`, and say the image tag must equal the
  deployed web sha.

## 6. File mode 1's Erho as a named known issue (owner-accepted 2026-09-30)

- [x] 6.1 Add `KNOWN_ISSUE_FRAMES` + `knownIssue(what)` to `pixels.mjs`,
  keyed on the check's own frame label, with the acceptance date and the
  mechanism in the reason line.
- [x] 6.2 Route every lobe verdict through one `judgeFrame(what, problemsOf)`
  in `check.mjs`; a waived frame prints `KNOWN` (never `ok`), is recorded in
  `knowns`, and does not set the exit code.
- [x] 6.3 Tests: the waived frame is still judged by the unrelaxed predicate;
  the waiver covers only the named frame (a second empty frame still fails);
  the callsite is unconditional, so a caption-gated form cannot creep back.
- [x] 6.4 Revert-check: restore the caption gate around the mode-1 Erho calls
  and confirm test 38 reddens; make `knownIssue` unconditional and confirm
  test 68 reddens. Both restored clean afterwards.

## 7. Verify

- [x] 7.1 `node --test tests/js/*.mjs` — all pass (not `tests/js/`, which runs
  nothing and exits 0).
- [ ] 7.2 Run the render check against a page with the fixed viewer and confirm
  the `Erho` mode-1 frame now differs from the outline control.
- [x] 7.3 `openspec validate fix-component-render-truth --strict`.
