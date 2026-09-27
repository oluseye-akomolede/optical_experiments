// The pixel predicates the render check asserts on, separated from the browser
// driving so they can be tested without a browser.
//
// Why they are split out: these predicates ARE the render check's judgement --
// "did it draw", "is the plot framed", "is a signed component drawn in its own
// hues rather than black". When they lived inline in check.mjs the only way to
// exercise them was to run a headless Chromium against a live page, which is
// slow, needs a server, and cannot be run on the VM at all. So they went
// untested, and a predicate that is wrong in the direction of *passing* is the
// worst kind of check: it reports green on a broken plot and is trusted.
//
// They are pure functions of a summary object, which is what makes them
// testable. check.mjs produces the summary from a real frame and calls these.
//
// The predicate that matters most is `divergingProblem`. It exists because of a
// specific bug: the first cut drew a signed component with `colormap(|v|) *
// sign(v)`, and multiplying an RGB colour by -1 clamps it to black -- so the
// negative half of every signed component rendered as a black half-disc, and a
// rotated frame showed the object with half of itself silently missing.
//
// The first version of the predicate demanded BOTH lobes on every signed
// component, which is also wrong: on the real job mode 1's Ez runs -1.000 to
// +0.030, so it is 92% negative and 0% positive and an all-cool frame is the
// *correct* picture. Demanding red there fails correct data. What can be
// asserted without knowing the field is that the pixels are not being painted
// black -- which is exactly what the bug did -- so the predicate asks what
// fraction of the field-coloured pixels are in either lobe's hue.

/** A pixel is "field-coloured" if it is distinguishable from the background. */
export const BACKGROUND = [0x0b, 0x0e, 0x12];
export const LIT_DISTANCE = 24;

/**
 * How far a pixel is from the page background, as a sum of channel differences.
 * This is the channel metric both the summary and the tests use, so a change
 * here changes what "lit" means everywhere rather than in one place.
 */
export function distanceFromBackground(r, g, b) {
  return Math.abs(r - BACKGROUND[0]) + Math.abs(g - BACKGROUND[1]) +
         Math.abs(b - BACKGROUND[2]);
}

/**
 * Did this frame draw the field at all?
 *
 * The floor is deliberately about *kind*, not amount: a component that is
 * genuinely thin over the volume covers far less of the frame than |E| does, and
 * failing it would be failing correct data. A count of a handful of pixels is
 * the "nothing was drawn" signal; anything more is judged by the comparisons
 * and the framing and hue predicates, which carry the rest.
 */
export function drawProblem(sample, what, floor = 500) {
  if (sample.coloured < floor) {
    return `${what} drew ${sample.coloured} field pixels of ${sample.total} ` +
           '-- the plot is not drawing this component';
  }
  return null;
}

/**
 * Is the object filling the frame, rather than sitting in it as a dot?
 *
 * The first cut put the camera 2.87 box-widths out, so the whole mode landed
 * inside 5% of the canvas and read as a speck. The object fills ~84% of the
 * grid box, so a correct camera puts its picture across most of the frame;
 * 0.35 of the frame is far below anything correct and far above the dot.
 *
 * Returns null when there is nothing to measure, which is NOT a pass: the
 * caller reports the draw failure instead, so a blank frame is never silently
 * "framed correctly".
 */
export function framingProblem(sample, floor = 0.35) {
  if (!sample.box) return 'framing is not measurable -- nothing was drawn';
  const frame = Math.max(sample.width, sample.height);
  const across = Math.max(sample.box.w, sample.box.h);
  if (across < floor * frame) {
    return `the plot fills only ${(across / frame * 100).toFixed(1)}% of the ` +
           `frame (${sample.box.w}x${sample.box.h} of ` +
           `${sample.width}x${sample.height}) -- the camera is too far back, ` +
           'so the object reads as a dot';
  }
  return null;
}

/**
 * Is a signed component being drawn in its own hues, rather than in black?
 *
 * A signed component has two lobes and the diverging ramp gives them blue and
 * red. The bug this catches painted the negative lobe black, which the
 * "coloured" count still counts -- black is a long way from the background --
 * while neither hue counter sees it. So the signature is a large `coloured`
 * with almost nothing in either hue, and that is what is asserted.
 *
 * It deliberately does NOT require both lobes: a component can be genuinely
 * one-sided (mode 1's Ez is 92% negative), and demanding the absent lobe would
 * fail correct data.
 */
export function divergingProblem(sample, what, share = 0.5) {
  const hued = sample.warm + sample.cool;
  if (!hued) {
    return `${what} has no colour at all: ${sample.warm} warm and ` +
           `${sample.cool} cool pixels -- a signed component is being drawn in ` +
           'neither end of the diverging ramp';
  }
  if (hued < share * sample.coloured) {
    return `${what} draws ${sample.coloured} pixels but only ${hued} of them ` +
           'are in either lobe colour -- the rest are being drawn in black, ' +
           'which is what multiplying a colour by its sign does';
  }
  return null;
}

/** How a passing diverging frame reads in the log. */
export function divergingNote(sample, what) {
  const lobes = [];
  if (sample.warm > 0) lobes.push(`${sample.warm} warm`);
  if (sample.cool > 0) lobes.push(`${sample.cool} cool`);
  const hued = sample.warm + sample.cool;
  return `${what} is diverging: ${lobes.join(' / ')} px, ` +
         `${(100 * hued / sample.coloured).toFixed(0)}% of its field pixels`;
}
