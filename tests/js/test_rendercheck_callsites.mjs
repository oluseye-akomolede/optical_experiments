// The render check's JUDGEMENT CALLS, tested against the source.
//
// Why this file exists, and why reading the source is the right instrument
// here. The predicates in pixels.mjs were correct and green, and the check
// still failed a correct page -- twice, on the same run:
//
//   FAIL Erho has no colour at all: 0 warm and 0 cool pixels ...
//   FAIL Erho in mode 1: the frame is the outline only: 0 pixels in either
//        lobe colour ... so it did not draw
//
// Both came from ONE call site. Mode 1's Erho has a measured p99 share of
// 8.26% against the viewer's own 12% drawing floor, so every sample is
// discarded and a hue-less frame is the DESIGNED picture -- the page's caption
// says so. The driver routed that frame to `diverging(...)` anyway -- on the
// reasoning "the frame drew enough that drawProblem is happy, so judge its
// hue", which is true of the frame's pixel count and false of the product.
//
// No unit test of `divergingProblem` could have caught that: the predicate was
// asked a question about a frame that had drawn nothing, and it answered
// correctly. The defect is in WHICH frames the driver judges, so that is what
// is asserted here. It cannot be driven through a fake frame -- the call sites
// live in a 700-line browser script -- so the source is read and the routing
// is pinned. This is the weaker instrument than a behavioural test and is used
// because the alternative was no test at all, which is how it shipped.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

const source = readFileSync(
  new URL('../../dockerfiles/rendercheck/check.mjs', import.meta.url), 'utf8');

// The body of the mode-1 Erho block: from the caption read to the mode-2 click
// that ends it. Scoped rather than searching the whole file, so a `diverging(`
// added elsewhere cannot satisfy or break these assertions by accident.
const between = (startMarker, endMarker) => {
  const start = source.indexOf(startMarker);
  assert.ok(start >= 0, `the source no longer contains ${JSON.stringify(startMarker)}`);
  const end = source.indexOf(endMarker, start);
  assert.ok(end > start, `the source no longer contains ${JSON.stringify(endMarker)}`);
  return source.slice(start, end);
};

test('the Erho frame IS routed to the lobe-hue judgement, like every component frame', () => {
  // The fix's whole point. Mode 1's Erho used to draw empty because the viewer
  // scaled by the component's MAX; at a p99 scale it draws, so it must carry
  // hue and the check must ask. The exemption that used to skip this is gone --
  // it forgave a real defect (the frame was pixel-identical to no field).
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  assert.match(block, /diverging\(\s*erho\s*,/,
    'the mode-1 Erho frame is no longer judged for its hue. After the p99 fix ' +
    'it draws, so a frame with no lobe hue is a defect -- this is the predicate ' +
    'that catches it');
});

test('the below-the-floor frame is still required to differ from the outline', () => {
  // The question that applies however the component is scaled: the frame must
  // carry the field's hue somewhere, or it is the outline. This is what stands
  // between a dead field and a green run.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  assert.match(block, /fieldVsOutline\(\s*erho\s*,/,
    'the mode-1 Erho frame is no longer compared against the outline-only ' +
    'control, so a page whose field never drew would pass this block on the ' +
    'strength of its caption alone');
});

test('the empty-frame exemption is GONE, not merely gated', () => {
  // The exemption let a frame matching the outline-only control pass when the
  // caption explained it. It was the wrong shape: a caption is not a control,
  // and the frame it forgave was a real defect. It must not creep back, in any
  // gated form.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  assert.doesNotMatch(block, /accountForEmptyFrame/,
    'accountForEmptyFrame is back in the mode-1 Erho block: a frame whose ' +
    'field never drew would pass again on the strength of a caption');
  // Strip comments first: the check's own comments explain WHY the exemption
  // was removed, and a rule that forbade the mention would force the next
  // reader to delete the reason (same trap as page.screenshot above).
  const code = source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '');
  assert.doesNotMatch(code, /accountForEmptyFrame/,
    'accountForEmptyFrame is back in the check\'s CODE -- the exemption was ' +
    'removed because it forgave a real defect (mode 1\'s Erho was pixel-' +
    'identical to no field at all)');
  // What REPLACED it is the owner-accepted, name-keyed known issue. It is not a
  // caption gate: the frame is passed to the same verdict function as any other,
  // and the waiver is looked up by the frame's OWN LABEL inside that verdict.
  assert.match(block, /diverging\(\s*erho\s*,\s*'Erho in mode 1'\s*\)/,
    'the mode-1 Erho frame is no longer routed to the lobe-hue verdict under ' +
    'its own label -- the known-issue waiver is keyed on that exact label, so ' +
    'the two must not drift apart');
  assert.match(block, /fieldVsOutline\(\s*erho\s*,\s*'Erho in mode 1'\s*\)/,
    'the mode-1 Erho frame is no longer compared against the outline under its ' +
    'own label, so the known-issue waiver would never be reached for it');
  // And the gate must NOT read the caption: the verdict closures stop reading
  // `erhoCaptionText`, which is the caption-gate shape the removed exemption had.
  assert.doesNotMatch(block, /knownIssue\s*\(\s*erhoCaptionText/,
    'the known-issue waiver is being read from the page caption -- that is the ' +
    'caption gate again, and it would waive any frame the page explains');
  // The routing must be UNCONDITIONAL. The regex matches above would still match
  // a call nested inside a guard -- `if (captionSays...) { diverging(erho, ...) }`
  // matches `/diverging\(\s*erho\s*,/` just as well -- so a caption gate could
  // creep back under these assertions and never redden them. That gap is real
  // (measured: a bracketed `if (...) { diverging(erho, ...); ... }` around the
  // two calls left all 96 tests green). What closes it is requiring the two
  // calls to be reached from the block body, with no statement between the
  // closing brace of the caption `else` and the first verdict call.
  const verdictStart = block.search(/\b(?:diverging|fieldVsOutline)\s*\(\s*erho\s*,/);
  assert.ok(verdictStart > 0, 'the mode-1 Erho verdict calls are missing');
  const before = block.slice(0, verdictStart);
  const lastClose = before.lastIndexOf('}');
  const leadIn = before.slice(lastClose + 1);
  assert.equal(/\bif\b|\belse\b|\?/.test(leadIn), false,
    'a statement between the caption check and the mode-1 Erho verdict calls ' +
    'can only be a gate -- the routing must be unconditional, or an exemption ' +
    'has crept back in a form these assertions would otherwise miss: ' +
    JSON.stringify(leadIn.trim()));
  const afterCalls = block.slice(verdictStart);
  assert.equal((afterCalls.match(/\bif\b|\belse\b|\?/g) || []).length, 0,
    'the statements following the mode-1 Erho verdict calls include a branch -- ' +
    'the calls are inside a conditional, which is the gate shape');
});

test('the frame that DOES draw is still judged for its hue', () => {
  // The control case: mode 2's Erho (share 0.98) and the rotated frame must
  // keep their hue judgement.
  assert.match(source, /diverging\(\s*erho2\s*,/,
    'mode 2\'s Erho is no longer judged for its hue -- that frame DOES draw ' +
    '(measured share 0.9824) and the black-lobe defect is caught here');
  assert.match(source, /diverging\(\s*rotated\s*,/,
    'the rotated frame is no longer judged for its hue -- it is mode 2\'s ' +
    'picture from a new camera and DOES draw');
});

test('the caption assertion requires the share AND the own-scale wording', () => {
  // After the fix the caption states two different things: the component's
  // STRENGTH (its share of |E|) and the fact that the picture shows SHAPE at
  // the component's own scale. The predicate must require both, or a caption
  // that names only the number would pass while leaving the shape/strength
  // distinction to the reader.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  const naming = /const erhoNamesTheTest =([\s\S]*?);/.exec(block);
  assert.ok(naming, 'erhoNamesTheTest has gone -- the caption is no longer named');
  assert.match(naming[1], /own scale/,
    'the caption assertion no longer requires the "own scale" wording, so a ' +
    'caption that only states the share would pass -- the shape/strength ' +
    'distinction is the point of the caption after the fix');
  assert.match(naming[1], /8\\?s\*%/,
    'the caption assertion no longer requires the component\'s real share, so ' +
    'any caption mentioning its own scale would pass');
});

test('a missing control frame fails the run instead of skipping the comparison', () => {
  // Fail-closed, the rig's recurring shape: the deferred verdicts need the
  // control, and a control that could not be captured must not read as "no
  // problem found". The three predicates that use it all report a broken check
  // when it is absent, and the driver must surface that.
  assert.match(source, /if \(!outline\)/,
    'the driver no longer checks whether the control was captured, so a failed ' +
    'control capture would silently drop every field-versus-outline verdict');
  assert.match(source, /cannot be made, and each is reported as a failure/,
    'the missing-control path no longer reports the dropped verdicts as failures');
});

test('the deferred verdicts are evaluated after the main page closes', () => {
  // The control must be the only WebGL context when it is captured; two live
  // contexts under a 2-CPU cap starve the control's fetches (measured ~75 s
  // contended vs ~6 s alone). The order is the fix, so the order is asserted.
  const closeAt = source.indexOf('await page.close()');
  const captureAt = source.indexOf('outline = await captureOutlineFrame()');
  assert.ok(closeAt > 0 && captureAt > 0,
    'the page close or the control capture has gone from the source');
  assert.ok(closeAt < captureAt,
    'the control is captured before the main page closes -- that is the ' +
    'contended-fetch bug this fixes, and it made the control read as empty');
});

test('the capture path is the canvas read, not page.screenshot', () => {
  // The other half of the change, pinned so it cannot silently come back: a
  // clip screenshot waits on a compositor frame and times out in the pod.
  //
  // The comments in the file discuss page.screenshot at length to explain why
  // it is not used, so this strips comments before looking -- a check that
  // forbade the MENTION would force the next reader to delete the reason.
  const code = source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/^\s*\/\/.*$/gm, '');
  assert.doesNotMatch(code, /page\.screenshot\s*\(/,
    'page.screenshot is back in the check\'s CODE -- it times out in the ' +
    'CPU-limited pod (measured, 30 s in the pod and 120-400 s on the loaded VM)');
  assert.match(source, /toDataURL\('image\/png'\)/,
    'the canvas read has gone from the source');
});
