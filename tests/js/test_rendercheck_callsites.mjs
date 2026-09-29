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

test('the below-the-floor frame is NOT routed to the lobe-hue judgement', () => {
  // The measured case: mode 1's Erho, share 0.0826 < THRESHOLD 0.12. The check
  // must not ask a frame the viewer drew empty on purpose for its hue.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  assert.doesNotMatch(block, /diverging\(\s*erho\s*,/,
    'the mode-1 Erho frame is routed to diverging(): that frame is below the ' +
    'viewer\'s own drawing floor (measured share 0.0826 vs 0.12), so it ' +
    'carries no hue BY DESIGN and this call fails a correct page -- it ' +
    'produced two of the three failures on the last run');
});

test('the below-the-floor frame is still required to differ from the outline', () => {
  // The question that DOES apply to that frame, and the reason dropping the
  // hue judgement does not make the check blind to a field that never drew:
  // the frame must still carry the field's hue somewhere, or it is the outline.
  // Without this, removing the diverging call would remove the only thing
  // standing between a dead field and a green run.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  assert.match(block, /fieldVsOutline\(\s*erho\s*,/,
    'the mode-1 Erho frame is no longer compared against the outline-only ' +
    'control, so a page whose field never drew would pass this block on the ' +
    'strength of its caption alone');
});

test('the empty-frame exemption is gated on the caption, not unconditional', () => {
  // The final form of the fix, and the part that must not silently widen. The
  // below-floor frame IS allowed to match the outline-only control -- that is
  // its designed picture -- but ONLY when the page's own caption has accounted
  // for it (`erhoNamesTheTest`: names the measured 8% share AND calls the
  // emptiness physics). An unconditional exemption would make the check blind
  // to a page whose field never drew; a gate on the caption keeps that case
  // failing, because such a page prints no caption. Pinned here because the
  // argument is one word long and easy to delete.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  const call = /fieldVsOutline\(\s*erho\s*,\s*[^)]*\)/.exec(block);
  assert.ok(call, 'the mode-1 Erho field-versus-outline call has gone');
  assert.match(call[0], /accountForEmptyFrame\s*:\s*erhoNamesTheTest/,
    'the empty-frame exemption is no longer gated on erhoNamesTheTest -- if ' +
    'it is unconditional the check accepts a page whose field never drew, and ' +
    'if the gate is dropped entirely the check fails a correct below-floor ' +
    'frame again (measured: the last of three failures on a correct run)');
});

test('with the exemption absent, an empty frame FAILS rather than passes', () => {
  // The else-branch of the gate, and the one that must stay a FAIL. The gate
  // (`if (problem && opts.accountForEmptyFrame)`) is false in two cases: the
  // call passes no exemption (every frame but mode 1's Erho), or it passes
  // `false` (mode 1's Erho on a page whose caption did NOT name the share).
  // Both must reach `else if (problem) fail(...)`. Pinned by reading the
  // branch ORDER in the source: if `fail` moved above the exemption, or the
  // exemption became the fallthrough, an unexplained empty frame would pass.
  const start = source.indexOf('const fieldVsOutline = ');
  assert.ok(start >= 0, 'the fieldVsOutline call site has gone');
  const end = source.indexOf('});', start) + 3;
  const body = source.slice(start, end);
  const gateAt = body.indexOf('opts.accountForEmptyFrame');
  const exemptOkAt = body.indexOf('below its own floor');
  const failAt = body.indexOf('${what}: ${problem}');
  assert.ok(gateAt > 0 && exemptOkAt > 0 && failAt > 0,
    'one of the three branches of fieldVsOutline has gone');
  assert.ok(gateAt < exemptOkAt,
    'the exemption is no longer guarded by accountForEmptyFrame -- an ' +
    'unexplained empty frame (a page whose field never drew) would pass');
  assert.ok(failAt > exemptOkAt,
    'the failure branch no longer follows the exemption branch, so an ' +
    'empty frame now reports ok() instead of fail()');
  // Positions alone are not enough: swapping fail() for ok() keeps every
  // position above and turns the empty frame into a pass, which is exactly the
  // bug this test exists for. Assert the VERB, reading the text between the
  // failure branch and the problem it reports.
  const failBranch = body.slice(exemptOkAt, failAt);
  assert.match(failBranch, /\}\s*else\s+if\s*\(problem\)\s*\{\s*fail\(/,
    'the empty-frame branch no longer calls fail() -- it reports the frame as ' +
    'ok, so an unexplained empty frame (the "field never drew" defect) passes');
});

test('the frame that DOES draw is still judged for its hue', () => {
  // The control case. Removing a judgement for the below-floor frame must not
  // quietly remove it for the frames that carry the field's real colour --
  // mode 2's Erho (share 0.98) and the rotated frame. A fix that disabled the
  // hue judgement everywhere would pass the test above and disable the check.
  assert.match(source, /diverging\(\s*erho2\s*,/,
    'mode 2\'s Erho is no longer judged for its hue -- that frame DOES draw ' +
    '(measured share 0.9824) and the black-lobe defect is caught here');
  assert.match(source, /diverging\(\s*rotated\s*,/,
    'the rotated frame is no longer judged for its hue -- it is mode 2\'s ' +
    'picture from a new camera and DOES draw');
});

test('the caption assertion accepts either true caption, not just "small"', () => {
  // Mode 1's Erho sits in the viewer's BELOW-FLOOR branch, whose wording is
  // "almost empty ... not a fault". The middle branch says "is small". Both are
  // true of a component at 8%; requiring the word "small" fails the live page.
  const block = between('const erhoCaptionText = await erhoCaption(page);',
                        'await page.click(\'#modes-body tr:nth-child(2)\')');
  const naming = /const erhoNamesTheTest =([\s\S]*?);/.exec(block);
  assert.ok(naming, 'erhoNamesTheTest has gone -- the caption is no longer named');
  assert.match(naming[1], /not a fault/,
    'the caption assertion does not accept the below-floor wording ("not a ' +
    'fault"), which is the caption the live page actually shows for mode 1\'s ' +
    'Erho -- requiring "small" fails a correct page');
  assert.match(naming[1], /small/,
    'the caption assertion no longer accepts the "small" wording, so a caption ' +
    'in the middle band would now fail');
  assert.match(naming[1], /8\\?s\*%/,
    'the caption assertion no longer requires the component\'s real share, so ' +
    'any caption mentioning a fault would pass');
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
