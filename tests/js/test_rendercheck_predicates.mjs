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
         freshClip, judgeFrameDecision, knownIssue, outlineOnlyProblem }
  from '../../dockerfiles/rendercheck/pixels.mjs';

// A frame summary shaped like the one check.mjs builds. `sum` defaults to a
// positive value because a real frame is never all-black: the page background
// is 0x0b0e12, so even a frame with nothing drawn sums above zero. A control
// with `sum === 0` is a cleared buffer, and there is a test for that below.
//
// The numbers in these tests are MEASURED, not invented. `coloured` is the
// same 91637 with the field on and off -- it is the object's box -- and the
// field's whole contribution is `warm`/`cool`. An earlier version of these
// fixtures used small invented `coloured` values (65291, 20000) that were
// mostly hue, so they matched a predicate whose denominator was the frame and
// hid the fact that it failed every real one.
const frame = (fields) => ({
  coloured: 0, warm: 0, cool: 0, sum: 834 * 353 * 28, box: null,
  width: 834, height: 353, total: 834 * 353, ...fields,
});

// The outline-only control, MEASURED: the same box, no hue. Every diverging
// assertion needs it, and it is a module constant so a test that forgets it is
// a failure rather than a silent pass.
const OUTLINE = { coloured: 91637, warm: 0, cool: 0, sum: 22458710 };

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

// Every number below is MEASURED on the live job. The four correct frames that
// reach this predicate carry these hued counts over the same 91637-px object:
//   rotated Erho mode 2   9428   (10.3%, the worst correct frame)
//   |E|                  23205   (25.3%)
//   Erho mode 2          23810   (26.0%)
//   Ez mode 1            37768   (41.2%)
// The floor is 1/16 of the object = 5727, so all four pass with margin.
const CORRECT = {
  rotatedErhoM2: { coloured: 81268, warm: 0, cool: 9428 },
  magnitude:     { coloured: 91637, warm: 5809, cool: 17396 },
  erhoM2:        { coloured: 91637, warm: 0, cool: 23810 },
  ezM1:          { coloured: 91637, warm: 0, cool: 37768 },
};

test('every measured correct frame passes, including the faint rotated one', () => {
  // This is the test that failed twice before: the first floor divided by the
  // FRAME's `coloured` (the object box, 91637), so |E| at 25% failed; the second
  // subtracted the control and went negative under rotation. Both are pinned
  // here by the worst case, the rotated frame whose box is 81268 merely because
  // the camera moved.
  for (const [name, f] of Object.entries(CORRECT)) {
    assert.equal(divergingProblem(frame(f), name, OUTLINE), null,
      `${name} is correct data and must pass: ${f.warm + f.cool} hued px`);
  }
});

test('a frame that drew too little hue fails against the object floor', () => {
  // The claim this predicate CAN make: a frame the check has decided drew must
  // carry a real share of the object in a ramp colour. 40 hued px of a 91637-px
  // object is 0.04% -- far under the floor -- and is what a component whose
  // colour is wiped out reads like.
  const starved = frame({ coloured: 91637, warm: 0, cool: 40 });
  const problem = divergingProblem(starved, 'Ez', OUTLINE);
  assert.ok(problem, 'a frame with 0.04% of the object hued must fail');
  assert.match(problem, /floor/);
});

test('a frame with no hue at all is caught against the object floor', () => {
  // Zero hue crosses the same floor as 40 px does -- there is ONE message for
  // both, deliberately. An earlier cut gave zero its own louder message ("a
  // signed component is being drawn in neither end of the diverging ramp"),
  // which asserted a CAUSE the pixel count cannot establish: zero hue is equally
  // what a frame drawn beneath the viewer's own drawing floor looks like, and on
  // the live page that is the CORRECT picture for mode 1's Erho (share 0.0826
  // against a 0.12 floor). The predicate states the measurement; naming the
  // cause is the caller's job, and on that frame the caller must not call it
  // at all.
  const noHue = frame({ coloured: 91637, warm: 0, cool: 0 });
  const problem = divergingProblem(noHue, 'Ez', OUTLINE);
  assert.ok(problem, 'a frame carrying no hue must fail against the object floor');
  assert.match(problem, /floor/,
    'the message must state the floor it failed against, not a cause it cannot see');
  assert.doesNotMatch(problem, /being drawn in neither end/,
    'the message must not name a cause -- zero hue is also a correct ' +
    'below-the-floor frame, and that wording is what misread one as a defect');
});

test('a two-lobed component passes and reports both lobes', () => {
  const two = frame({ coloured: 91637, warm: 12000, cool: 13000 });
  assert.equal(divergingProblem(two, 'Er', OUTLINE), null);
  const note = divergingNote(two, 'Er', OUTLINE);
  assert.match(note, /12000 warm/);
  assert.match(note, /13000 cool/);
});

test('the verdict does not move when the control does, for a fixed frame', () => {
  // The control sets only the denominator. A predicate that changed its verdict
  // on a fixed frame as the control moved would be a check on the control, not
  // on the picture. Both controls here are honest (non-empty, non-black).
  const drawn = frame(CORRECT.ezM1);
  const starved = frame({ coloured: 91637, warm: 0, cool: 40 });
  for (const ctrl of [{ coloured: 91637, sum: 22458710 },
                      { coloured: 60000, sum: 15000000 }]) {
    assert.equal(divergingProblem(drawn, 'Ez', ctrl), null,
      'correct data must pass against any honest control');
    assert.ok(divergingProblem(starved, 'Ez', ctrl),
      'starved hue must fail against any honest control');
  }
});

test('WHAT IS NOT CLAIMED: a half-black lobe is NOT caught here', () => {
  // Measured, not assumed. A frame that loses one lobe of a two-lobe component
  // keeps the other lobe: losing Erho mode 2's cool lobe leaves it 0 hued, but
  // losing HALF leaves ~11905 (13% of the object), and losing half of Ez mode 1
  // leaves ~18884 (21%). Both are ABOVE the 10.3% that the faint rotated frame
  // legitimately shows, so no floor separates them. This test pins that the
  // predicate passes those frames, so nobody re-tightens the share expecting it
  // to catch a half-black lobe -- the black-lobe defect is caught by the mode
  // frames differing from each other and by fieldVsOutline instead.
  const halfBlackErhoM2 = frame({ coloured: 104842, warm: 0, cool: 11905 });
  const halfBlackEzM1 = frame({ coloured: 104842, warm: 0, cool: 18884 });
  assert.equal(divergingProblem(halfBlackErhoM2, 'Erho', OUTLINE), null,
    'a half-black lobe is above the floor -- this predicate does not catch it');
  assert.equal(divergingProblem(halfBlackEzM1, 'Ez', OUTLINE), null,
    'and neither does the other half-black shape -- the populations overlap');
});

test('a missing or broken control fails the frame rather than passing it', () => {
  // With no control the field's extent is unknown, and the honest answer is
  // "cannot judge" -- NOT a pass. An `if (!outline) return null` here would be
  // the could-not-ask-reported-as-nothing-there defect: every frame would go
  // green the moment the control page failed to load.
  const drawn = frame({ coloured: 91637, warm: 0, cool: 37768 });
  assert.ok(divergingProblem(drawn, 'Ez', null),
    'no control must fail closed');
  assert.ok(divergingProblem(drawn, 'Ez', { coloured: 0, sum: 1234 }),
    'an empty control must fail closed');
  assert.ok(divergingProblem(drawn, 'Ez', { coloured: 294402, sum: 0 }),
    'an all-black cleared-buffer control must fail closed');
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

// ---- the field vs the outline -------------------------------------------
//
// The second defect, from the same run: the check measured the BOX, not the
// field. `summarise()` counted any pixel differing from the background, and the
// object's wireframe outline is such a pixel -- so every frame reported the
// same 91637 coloured pixels and a 400x337 box, and an outline-only page (a
// page whose field never drew) passed.
//
// The fix is a CONTROL FRAME: capture the same page with the field hidden
// (`?field=off`) and require a mode frame to differ from it. The numbers below
// are the measured ones, from toggling the live field mesh's visibility with
// everything else fixed:
//
//     field ON    { coloured: 91637, warm: 5809, cool: 17396 }
//     field OFF   { coloured: 91637, warm: 0,    cool: 0     }
//
// `coloured` is IDENTICAL in both. The field mesh never changes the object's
// silhouette -- the outline draws regardless -- so the count is the box in every
// frame and carries no signal about the field at all. The field's entire
// contribution is the HUE it paints over that box. So the predicate compares
// `warm`/`cool` against the control, and the test values below are built from
// those same measured frames.

test('an outline-only frame is caught by the field-versus-outline predicate', () => {
  // The "field" frame IS the control frame: the field never drew, so what is
  // measured is the outline twice, hue and all. This is the frame the old
  // predicate called a pass -- the control numbers are the measured ones.
  const outline = frame({ coloured: 91637, warm: 0, cool: 0, hash: 111 });
  const field = frame({ coloured: 91637, warm: 0, cool: 0, hash: 111 });
  const problem = outlineOnlyProblem(field, outline);
  assert.ok(problem, 'a frame with the control\'s zero hue must fail: the ' +
    'field did not draw');
  assert.match(problem, /outline/i);
});

test('a real field frame differs from the outline and passes', () => {
  // The negative control: the same predicate on a frame that DID draw the field
  // must be silent, or the check fails correct data and gets ignored. These are
  // the measured `field ON` numbers over the measured control.
  const outline = frame({ coloured: 91637, warm: 0, cool: 0, hash: 111 });
  const field = frame({ coloured: 91637, warm: 5809, cool: 17396, hash: 222 });
  assert.equal(outlineOnlyProblem(field, outline), null,
    'a frame that differs from the outline must not be reported as outline-only');
});

test('the same coloured count with no hue is the outline, not a field', () => {
  // THE DISCRIMINATING CASE, and the regression guard for the defect itself.
  // The field frame and the control frame have the SAME `coloured` count
  // (91637) and a DIFFERENT digest -- so a predicate that judged `coloured`, or
  // that accepted a differing digest, would call this outline-only frame a
  // real field. Only the hue separates them: the real frame carries hue, the
  // outline carries none. This is the measured shape of the defect (count
  // constant, field in the hue), and it reddens if the predicate ever goes back
  // to judging the count or the digest.
  const outline = frame({ coloured: 91637, warm: 0, cool: 0, hash: 111 });
  const outlineAgain = frame({ coloured: 91637, warm: 0, cool: 0, hash: 444 });
  const problem = outlineOnlyProblem(outlineAgain, outline);
  assert.ok(problem, 'a frame whose hue matches the control is the outline, ' +
    'even when its digest differs and its coloured count is identical');
  assert.match(problem, /outline/i);
});

test('a faint but real field passes against a large outline', () => {
  // The floor is a share of the control's own area, not an absolute count, so
  // it does not have to be retuned when the model or viewport changes. A real
  // mode frame paints ~20% of the outline's pixels in hue; a blank one paints
  // none. This asserts the predicate does not fail the real one.
  const outline = frame({ coloured: 91637, warm: 0, cool: 0, hash: 111 });
  const faintField = frame({ coloured: 91637, warm: 900, cool: 8000, hash: 222 });
  assert.equal(outlineOnlyProblem(faintField, outline), null,
    'a frame with less hue than a bright mode, but clearly more than none, is ' +
    'not the outline');
});

test('an empty outline frame is itself a failure, not a free pass', () => {
  // The vacuous pass: if the control frame has no lit pixels, "the field
  // differs from the outline" is true for every frame and the comparison says
  // nothing. That must be reported, not silently accepted.
  const empty = frame({ coloured: 0, warm: 0, cool: 0, hash: 0 });
  const field = frame({ coloured: 91637, warm: 5809, cool: 17396, hash: 222 });
  const problem = outlineOnlyProblem(field, empty);
  assert.ok(problem, 'an empty control frame cannot serve as a control');
  assert.match(problem, /control|outline/i);
});

test('an all-black control is caught, not mistaken for a drawn outline', () => {
  // THE REGRESSION THAT PROMPTED THIS GUARD, and the reason `coloured === 0`
  // is not enough. A canvas read from a context without `preserveDrawingBuffer`
  // returns a cleared buffer: every pixel (0,0,0,0). Decoded, that is
  // `coloured === total` (every pixel differs from the non-black background)
  // and `sum === 0`. The empty-control guard tests `coloured === 0`, so it does
  // NOT fire, and the predicate goes on to fail every mode frame against a
  // floor of `ceil(total/64)`. These numbers are the measured ones from that
  // bug: control `{coloured: 294402, sum: 0}` on an 834x353 frame.
  const allBlack = frame({ coloured: 834 * 353, warm: 0, cool: 0, sum: 0,
                           hash: 0 });
  const field = frame({ coloured: 91637, warm: 5809, cool: 17396, sum: 22033186,
                        hash: 222 });
  const problem = outlineOnlyProblem(field, allBlack);
  assert.ok(problem, 'a blank (all-black) control cannot serve as a control');
  assert.match(problem, /blank|black|control/i);
  // And specifically NOT the "the frame is the outline only" verdict, which is
  // what the bug produced: that blames the mode frame for a broken control.
  assert.doesNotMatch(problem, /the frame is the outline only/,
    'a broken control must be reported as a broken control, not as the mode ' +
    'frame being outline-only');
});

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

// --- the known-issue waiver --------------------------------------------------
//
// The owner accepted mode 1's Erho haze on 2026-09-30, so the check stops
// FAILING THE RUN on that frame. The danger of any exemption is the shape the
// removed `accountForEmptyFrame` had: a gate that widened into silence, so a
// real defect passed. These tests pin the waiver as narrow -- it names one frame
// and only that frame -- and pin that it is keyed on the check's own label, not
// on anything the page says.
test('the accepted known issue is named by the frame label the check uses', () => {
  // The string here is deliberately the EXACT one the verdict prints
  // (`diverging(erho, 'Erho in mode 1')`), because a waiver keyed on anything
  // else -- a substring, a component name, a regex over the whole log -- could
  // widen to frames it was not meant for.
  const reason = knownIssue('Erho in mode 1');
  assert.ok(reason, 'mode 1 Erho is the owner-accepted known issue');
  assert.match(reason, /sign-mixed|neutral haze/i,
    'the reason names the mechanism that produced the frame');
  assert.match(reason, /2026-09-30/,
    'the reason records when the owner accepted it');
});

test('the waiver covers only that frame: a second empty frame still fails', () => {
  // A second empty frame is the thing the waiver must never swallow. Each of
  // these has a DIFFERENT label from the waived one, and a different label is
  // the whole of the gate.
  const otherLabels = [
    'Ez',                       // another component, mode 1
    '|E|',                      // the magnitude frame
    'Erho in mode 2',           // the same component, another mode
    'the rotated frame',        // the same component, another camera
  ];
  for (const what of otherLabels) {
    assert.equal(knownIssue(what), null,
      `"${what}" is not the waived frame -- it must be judged normally`);
  }
  // And a near-miss spelling gets no waiver either: the match is exact, so a
  // label that merely resembles the waived one is not excused.
  assert.equal(knownIssue('Erho in mode 1 '), null,
    'a trailing space is a different label and must not inherit the waiver');
  assert.equal(knownIssue('erho in mode 1'), null,
    'the waiver is case-sensitive -- a lookalike label is not the named frame');
  assert.equal(knownIssue('Erho'), null,
    'the bare component name is NOT the waived frame; using it would waive the '
    + 'component everywhere it appears, including the modes where it draws');
});

test('the waived frame is still judged by the unrelaxed predicate', () => {
  // The waiver lives at the CALL SITE, not in the predicate. So the predicate
  // must still report the empty frame at full strength: the measurement is
  // printed and filed, and no threshold was moved to make the frame pass.
  const outline = frame({ coloured: 91637, warm: 0, cool: 0, sum: 22458710 });
  const erhoHaze = frame({ coloured: 91637, warm: 0, cool: 0, sum: 22458710 });
  assert.ok(outlineOnlyProblem(erhoHaze, outline),
    'the raw predicate still reports the haze frame -- the waiver does not '
    + 'weaken it, it files an exception');
  assert.ok(divergingProblem(erhoHaze, 'Erho in mode 1', outline),
    'the lobe-hue predicate is likewise unrelaxed for the waived frame');
});

test('judgeFrameDecision: the waived label turns a failing frame into KNOWN', () => {
  // THE DECISION ITSELF, not the table. The three tests above call knownIssue()
  // and the predicates; none of them calls the function that APPLIES the
  // waiver. A mutation that made the waiver always apply (M2) or never apply
  // (M3) left every one of them green. This test calls it.
  const found = ['the frame is the outline only', 'no lobe hue reached the frame'];
  const r = judgeFrameDecision('Erho in mode 1', found);
  assert.equal(r.verdict, 'known',
    'the waived label is accepted, so the run does not fail on this frame');
  assert.ok(r.reason, 'the verdict carries the reason for the log');
  assert.match(r.reason, /sign-mixed|neutral haze/i, 'and it is the real reason');
  assert.deepEqual(r.problems, [],
    'a waived frame contributes NOTHING to the problem list -- that is what '
    + 'lets the run continue');
});

test('judgeFrameDecision: another label appends its problems and fails', () => {
  // The other half, and the half M2 breaks. A frame with a different label is
  // judged normally even when the measurements are identical to the waived
  // frame's.
  const found = ['the frame is the outline only', 'no lobe hue reached the frame'];
  for (const what of ['Ez', '|E|', 'Erho in mode 2', 'the rotated frame',
                      'erho in mode 1', 'Erho in mode 1 ']) {
    const r = judgeFrameDecision(what, found);
    assert.equal(r.verdict, 'failed', `"${what}" must fail -- it is not waived`);
    assert.deepEqual(r.problems, found,
      `"${what}" keeps ALL of its problems; nothing is swallowed`);
  }
  // The count matters: a mutation that appended only the first problem would
  // still return 'failed' and pass a truthiness check, so pin the list.
  const r = judgeFrameDecision('Ez', ['a', 'b', 'c']);
  assert.equal(r.problems.length, 3, 'every problem is carried, not just one');
});

test('judgeFrameDecision: a clean frame passes, and a waiver cannot invent one', () => {
  // The no-problems branch. A mutation that always waives (M2) would also have
  // to be careful here: 'passed' and 'known' must stay distinct, or the log
  // would print KNOWN on frames that are simply correct.
  const clean = judgeFrameDecision('Erho in mode 1', []);
  assert.equal(clean.verdict, 'passed',
    'a frame with no problems is passed even when its label is waived -- the '
    + 'waiver only ever downgrades a FAILURE, it never upgrades a pass to KNOWN');
  assert.equal(clean.reason, undefined, 'and it carries no reason');
  const cleanOther = judgeFrameDecision('Ez', []);
  assert.equal(cleanOther.verdict, 'passed', 'a clean unwaived frame passes');
});

test('judgeFrameDecision: undefined/non-array problems is treated as clean', () => {
  // Defensive: the caller always passes an array, but the decision must not
  // read undefined as "a problem" -- that would fail every frame whose
  // predicate returned nothing.
  assert.equal(judgeFrameDecision('Ez', undefined).verdict, 'passed');
});
