// The render check's judgement, tested without a browser.
//
// These predicates are what the render check concludes from a frame, and the
// failure mode of a wrong one is asymmetric: a predicate that is too strict
// fails correct data and gets disabled, and one that is too lax reports green on
// a broken plot. The second is worse, because the check is then trusted. So each
// predicate here is pinned on the bug it exists for -- the assertion must fail
// on the broken frame and pass on the correct one.
//
// The broken frames are not invented. They are the shapes the real defects
// produced, measured on the live job:
//   black half-disc: a large `coloured` with almost nothing in either hue
//   the dot:         a small bounding box in a large frame
//   nothing drawn:   a blank canvas
// and the *correct* one-sided frame is real too: mode 1's Ez is 92% negative
// and 0% positive, so an all-cool frame must pass.

import test from 'node:test';
import assert from 'node:assert/strict';
import { drawProblem, framingProblem, divergingProblem, divergingNote,
         freshClip }
  from '../../dockerfiles/rendercheck/pixels.mjs';

// A frame summary shaped like the one check.mjs builds.
const frame = (fields) => ({
  coloured: 0, warm: 0, cool: 0, box: null, width: 834, height: 353,
  total: 834 * 353, ...fields,
});

test('a blank canvas is reported as not drawing', () => {
  const blank = frame({});
  assert.ok(drawProblem(blank, '|E|'),
    'a frame with no coloured pixels must report a draw failure');
});

test('a thin-but-real component is not failed for being thin', () => {
  // The floor is about kind, not amount: 600 field pixels is a real plot of a
  // sparse component and must pass, or the check fails correct data.
  const thin = frame({ coloured: 600 });
  assert.equal(drawProblem(thin, 'Erho'), null,
    'a sparse component must not be failed for covering little of the frame');
});

test('the dot is reported as a framing failure', () => {
  // 5% of the frame, which is what the too-far camera produced.
  const dot = frame({ coloured: 900, box: { minX: 400, minY: 170, maxX: 445,
                                            maxY: 200, w: 46, h: 31 } });
  const problem = framingProblem(dot);
  assert.ok(problem && /dot/.test(problem),
    'a 46x31 picture in an 834x353 frame must be reported as a dot');
});

test('a well-framed plot passes, and framing is not measurable without pixels', () => {
  const good = frame({ coloured: 59610, box: { minX: 300, minY: 22, maxX: 596,
                                               maxY: 296, w: 297, h: 275 } });
  assert.equal(framingProblem(good), null, 'a 297x275 picture is framed');

  // No box means nothing was drawn. That must NOT read as "framed fine" -- it
  // is the vacuous pass that an `if (box)` guard produces.
  const blank = frame({});
  assert.ok(framingProblem(blank),
    'an unmeasurable frame must not silently pass the framing check');
});

test('the black half-disc is caught', () => {
  // The bug: negative lobe clamped to black. The pixels are far from the
  // background so they count as coloured, but almost none carry a hue.
  const blackLobe = frame({ coloured: 40000, warm: 15000, cool: 40 });
  const problem = divergingProblem(blackLobe, 'Ez');
  assert.ok(problem && /black/.test(problem),
    'a frame that is mostly black-with-one-hue must be reported: the negative ' +
    'lobe is being drawn in black');
});

test('a genuinely one-sided component passes', () => {
  // mode 1 Ez, measured: 65291 coloured, 0 warm, 49027 cool. This is correct
  // data and the predicate must not fail it -- requiring both lobes here was
  // the first version's error.
  const oneSided = frame({ coloured: 65291, warm: 0, cool: 49027 });
  assert.equal(divergingProblem(oneSided, 'Ez'), null,
    'an all-cool component is a legitimate one-sided field, not a defect');
  // ... and its log line must name the lobe that IS there.
  assert.match(divergingNote(oneSided, 'Ez'), /49027 cool/);
});

test('a two-lobed component passes and reports both', () => {
  const two = frame({ coloured: 20000, warm: 9000, cool: 9500 });
  assert.equal(divergingProblem(two, 'Er'), null);
  const note = divergingNote(two, 'Er');
  assert.match(note, /9000 warm/);
  assert.match(note, /9500 cool/);
});

test('a frame with no hue at all is caught', () => {
  const noHue = frame({ coloured: 5000, warm: 0, cool: 0 });
  assert.ok(divergingProblem(noHue, 'Ez'),
    'a signed component drawn in neither lobe colour must fail');
});

// ---- the screenshot clip -------------------------------------------------
//
// A separate defect, from the same run: the clip that frames each screenshot
// went stale. Clicking a mode row scrolls the page, so a rect captured before
// the click photographs where the canvas USED to be -- page chrome. Two saved
// PNGs (`05-...-mode2.png`, `04-rotated.png`) were text and selector widgets.
//
// The clip's contract is small and worth pinning, because the tempting repair
// is wrong in a way that looks like care. `boundingBox()` and
// `page.screenshot({clip})` are BOTH viewport-relative -- the screenshot
// scrolls internally to satisfy a clip -- so the rect needs no scroll
// correction, and ADDING `window.scrollY` to it double-counts the scroll and
// pushes the clip off the canvas by exactly the scroll amount.
//
// So the test drives a fake canvas whose viewport rect moves when the page
// scrolls, and asserts the two properties that matter: the clip tracks the
// canvas, and it is NOT shifted by the scroll. A version that added scrollY
// passes the first and fails the second, which is the whole point.

test('the screenshot clip tracks the canvas and is not scroll-corrected', async (t) => {
  t.after(() => { delete globalThis.scrollY; });
  // The canvas sits at viewport y = 194.78 at scrollY 0, and the page then
  // scrolls by 284 (both measured), putting it at -89.22.
  const canvasRectAt = (scrollY) => ({
    x: 120, y: 194.78 - scrollY, width: 834, height: 353,
  });

  // The fake page puts the scroll where the REAL code would read it. That
  // matters: a version that corrects the clip reads `window.scrollY`, so a
  // fake that keeps the scroll in a closure it alone can see would let the
  // buggy function pass this test. The scroll must be on a global the code
  // under test can reach, exactly as the browser exposes it.
  const makePage = () => ({
    scrollTo: (y) => { globalThis.scrollY = y; },
    boundingBox: async () => canvasRectAt(globalThis.scrollY),
  });

  // The real function the check uses, called the way check.mjs calls it.
  const good = makePage();
  const before = await freshClip(() => good.boundingBox());
  good.scrollTo(284);
  const after = await freshClip(() => good.boundingBox());
  assert.equal(after.y, -89.22,
    'the clip must follow the canvas to where it now is; a stale clip ' +
    'photographs page chrome');
  assert.notEqual(after.y, before.y,
    'the clip went stale -- it did not move with the canvas');

  // The repair that looks like care, computed here exactly as it would be
  // written in check.mjs: the rect is already viewport-relative, so adding the
  // scroll shifts it by the full scroll amount -- 194.78 - 284 + 284 = 194.78,
  // i.e. back to where the canvas was and no longer is. That is the stale
  // position, which is what a stale clip photographs: page chrome.
  const wrong = { ...after, y: after.y + 284 };
  assert.equal(wrong.y, 194.78,
    'adding scrollY to a viewport-relative rect cancels the canvas moving: it ' +
    'lands back on the stale position, off by the full scroll amount');
  assert.notEqual(wrong.y, after.y,
    'the doubled correction must land somewhere other than the true canvas');
});
