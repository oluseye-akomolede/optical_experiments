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
 * Decode a PNG buffer to raw RGBA bytes, in NODE.
 *
 * Why node-side and not `getImageData` in the page: the decode is a pure
 * function and it belongs on the side that can be unit-tested without a
 * browser, which is the same reason the rest of this file exists. It is also
 * the side that costs the page nothing -- an in-page decode is one more task
 * the page's render loop has to schedule, and the loop is the thing being
 * starved (see readCanvas in check.mjs for the measurement).
 *
 * Reachability, because the reachable set is small: Chromium's toDataURL emits
 * 8-bit truecolour PNGs -- colour type 6 (RGBA) or 2 (RGB), bit depth 8, no
 * palette, no interlace, one IDAT of zlib-compressed bytes carried in one or
 * more chunks. Anything else throws rather than being silently mis-decoded;
 * guessing would turn an unexpected encoder into wrong pixel counts, and the
 * whole point of this check is to not report green on a frame it did not read.
 *
 * Only the five filters PNG defines are handled (0 none, 1 sub, 2 up,
 * 3 average, 4 paeth), each undone with the bpp offset the format's spec
 * gives.
 */
export async function decodePng(buffer) {
  const { inflateSync } = await import('node:zlib');

  const SIGNATURE = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
  for (let i = 0; i < SIGNATURE.length; i++) {
    if (buffer[i] !== SIGNATURE[i]) {
      throw new Error('decodePng: not a PNG (bad signature)');
    }
  }

  let width = 0;
  let height = 0;
  let bitDepth = 0;
  let colourType = 0;
  let interlace = 0;
  const idat = [];
  let offset = 8;

  while (offset + 8 <= buffer.length) {
    const length = buffer.readUInt32BE(offset);
    const type = buffer.toString('latin1', offset + 4, offset + 8);
    const data = buffer.subarray(offset + 8, offset + 8 + length);
    offset += 12 + length;
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      bitDepth = data[8];
      colourType = data[9];
      interlace = data[12];
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
  }

  if (interlace !== 0) throw new Error('decodePng: interlaced PNG is not supported');
  if (bitDepth !== 8) throw new Error(`decodePng: bit depth ${bitDepth} is not supported`);
  const channels = { 2: 3, 6: 4 }[colourType];
  if (!channels) {
    throw new Error(`decodePng: colour type ${colourType} is not supported`);
  }
  if (!width || !height) throw new Error('decodePng: no IHDR');

  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(height * stride);

  for (let y = 0; y < height; y++) {
    const filter = raw[y * (stride + 1)];
    const line = raw.subarray(y * (stride + 1) + 1, (y + 1) * (stride + 1));
    const row = out.subarray(y * stride, (y + 1) * stride);
    const prev = y === 0 ? null : out.subarray((y - 1) * stride, y * stride);
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? row[x - channels] : 0;
      const b = prev ? prev[x] : 0;
      const c = prev && x >= channels ? prev[x - channels] : 0;
      const v = line[x];
      switch (filter) {
        case 0: row[x] = v; break;
        case 1: row[x] = (v + a) & 0xff; break;
        case 2: row[x] = (v + b) & 0xff; break;
        case 3: row[x] = (v + ((a + b) >> 1)) & 0xff; break;
        case 4: {
          const p = a + b - c;
          const pa = Math.abs(p - a);
          const pb = Math.abs(p - b);
          const pc = Math.abs(p - c);
          const pred = pa <= pb && pa <= pc ? a : (pb <= pc ? b : c);
          row[x] = (v + pred) & 0xff;
          break;
        }
        default: throw new Error(`decodePng: unknown filter ${filter} on row ${y}`);
      }
    }
  }

  // The summary counts RGBA quads, so RGB is widened here rather than at every
  // caller. Alpha is 255: PNG's RGB means opaque.
  if (channels === 3) {
    const rgba = Buffer.alloc(width * height * 4);
    for (let i = 0, j = 0; i < out.length; i += 3, j += 4) {
      rgba[j] = out[i];
      rgba[j + 1] = out[i + 1];
      rgba[j + 2] = out[i + 2];
      rgba[j + 3] = 255;
    }
    return { data: rgba, width, height };
  }
  return { data: out, width, height };
}

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
 *
 * THE DENOMINATOR IS THE CONTROL'S OWN AREA -- the object, not the frame and not
 * the frame minus the control. `coloured` counts the object's BOX, and it is the
 * same number with the field on and off (measured, 91637 both ways -- see
 * outlineOnlyProblem), so dividing the field's hue by it asks "is more than half
 * of this object painted in one hue", which can never hold: measured against a
 * real run, |E| carries 23205 hued pixels, 25% of that box. The first cut of
 * this predicate did exactly that and failed every frame. A predicate that cannot
 * pass is as useless as one that cannot fail, and it passed its unit tests only
 * because they used invented frames whose `coloured` was mostly hue.
 *
 * Subtracting the control does NOT fix it either. `coloured - outline.coloured`
 * is zero for every frame that drew a field (the field mesh does not change the
 * silhouette), so it passes vacuously; and it goes NEGATIVE for the rotated
 * frame, whose box is 81268 merely because the camera moved. Both were tried and
 * both were wrong; the measurement is kept here so neither is retried.
 *
 * The control's own area is the one quantity that is stable across all those
 * frames, and the share is set against it. `share` sits above the 1/64 of
 * `outlineOnlyProblem` -- which asks the weaker question "did the field draw at
 * all" -- and below the faintest correct frame that reaches here. Measured, the
 * four correct frames that pass through this predicate carry 9428 (rotated
 * Erho mode 2, the worst), 23205 (|E|), 23810 (Erho mode 2) and 37768 (Ez mode
 * 1) hued pixels of the 91637 box: 10.3% is the floor for a correct frame, so
 * 1/16 (6.25%, 5727 px) passes all four with margin.
 *
 * WHAT THIS DOES NOT PROVE, stated so it is not over-claimed: it does NOT catch
 * a HALF-black lobe. Measured, a frame that loses one lobe of a two-lobe
 * component still keeps 11905-18884 hued pixels -- 13-21% of the box -- which is
 * above any floor that also admits the legitimately faint 10.3% rotated frame.
 * The two populations overlap, so no threshold on this quantity can separate
 * them. The black-lobe defect is caught by the OTHER predicates -- the mode
 * frames differing from each other, and `fieldVsOutline` -- not by this one.
 * This ends the search: a share that failed a half-black lobe was looked for and
 * does not exist, because 0.103 (worst correct) > 0.13 needs no arithmetic to
 * see is impossible.
 *
 * AND IT IS NOT ARMED BY AN ABSENCE. The caller must not route a frame here
 * merely because that frame carried no hue. A signed component whose p99 sits
 * under the viewer's own drawing floor is drawn as an almost empty frame by
 * DESIGN -- measured on the live job, mode 1's Erho has a share of 0.0826
 * against a floor of 0.12, so the shader discards every sample and the frame
 * legitimately carries zero hue while `coloured` still reports the outline's
 * 91637. Judging that frame here reads 0 hue against the 5727 floor and reports
 * "a signed component is being drawn in neither end of the diverging ramp",
 * which is false: nothing was drawn, on purpose, and the page says so in its
 * caption. Measured, that single mis-routing -- `diverging(erho)` on mode 1 --
 * produced two of the three failures on an otherwise correct run. "The frame
 * has no hue" is answered by `outlineOnlyProblem` and by the caption, not here.
 * This predicate is for a frame the caller already knows DREW.
 *
 * Also NOT claimed: that the hue is two-lobed. Three of the four correct frames
 * above are single-lobed (Ez mode 1 and both Erho frames are drawn almost
 * entirely at one end of the ramp), so requiring both lobes would fail three
 * correct frames -- the over-tight predicate this file has been burned by.
 */
export function divergingProblem(sample, what, outline, share = 1 / 16) {
  if (!outline || outline.coloured === 0 || outline.sum === 0) {
    return `${what} cannot be judged: there is no usable outline-only control ` +
           'frame, so the object area the hue is measured against is unknown ' +
           '-- this is a broken check, not a broken frame';
  }
  const hued = sample.warm + sample.cool;
  const floor = share * outline.coloured;
  if (hued < floor) {
    // ONE message, not two. An earlier cut returned a louder message for
    // `hued === 0` ("no colour at all: a signed component is being drawn in
    // neither end of the diverging ramp") and the softer one above; the two are
    // the same quantity crossing the same floor, and the loud wording asserted
    // a CAUSE the pixel count cannot establish -- zero hue is equally what a
    // frame drawn beneath the viewer's floor looks like, which is correct. One
    // message that states the measurement is honest; two that guess at why are
    // the "a mechanism is claimed, not noticed" shape.
    return `${what} draws ${sample.coloured} pixels but only ${hued} of them ` +
           `are in either lobe colour, against a floor of ${Math.round(floor)} ` +
           `for this object -- too little of it is being drawn in a ramp colour`;
  }
  return null;
}

/** How a passing diverging frame reads in the log. */
// ---- the screenshot clip ---------------------------------------------------
//
// The rect that frames a screenshot, kept here with the predicates because it
// has the same failure mode: wrong in the direction of passing quietly. It is
// pure -- it takes a bounding-box getter and returns a rect -- so it can be
// tested without a browser.
//
// Why it takes a getter rather than a rect: clicking a mode row scrolls the
// page (measured, scrollY 0 -> 284), and the canvas then sits at viewport
// y = -89 while a rect captured at the top still says 194.78. A rect captured
// once and reused photographs where the canvas USED to be -- page chrome.
// Re-reading the box at every shot is the fix.
//
// Why there is no scroll correction: `boundingBox()` and
// `page.screenshot({clip})` are BOTH viewport-relative -- the screenshot scrolls
// internally to satisfy a clip. So the rect wants no correction at all, and
// adding `window.scrollY` to it double-counts the scroll and lands back on the
// stale position, off by the full scroll amount. That mistake was made and
// measured: the same rect captured the field with no scroll added and the page
// chrome with it added.
export async function freshClip(boundingBox) {
  return boundingBox();
}

/**
 * Does this mode frame actually differ from the object's outline alone?
 *
 * This exists because of a specific defect: the check measured the BOX, not the
 * field. Every pixel that is not the page background counts as "lit", and the
 * wireframe outline is such a pixel -- so an outline-only page (the field never
 * drew) passed every draw predicate with the same 91637 coloured pixels and
 * 400x337 box as a real plot.
 *
 * WHAT "DIFFERS" MEANS IS THE HUE, NOT THE COUNT. That is measured, not chosen.
 * Toggling the field mesh's visibility on the live page, with everything else
 * fixed, reads:
 *
 *     field ON    { coloured: 91637, warm: 5809, cool: 17396 }
 *     field OFF   { coloured: 91637, warm: 0,    cool: 0     }
 *     field ON    { coloured: 91637, warm: 5809, cool: 17396 }
 *
 * `coloured` does not move at all. The field mesh does not change the object's
 * silhouette -- the outline and the surface are drawn whether or not the
 * volume's colour is on -- so the count is the box in every frame, and the
 * field's entire contribution is the hue it paints over the box. An
 * outline-only frame is therefore one whose hue matches the control's, and the
 * comparison has to be on `warm`/`cool`. Comparing `coloured` here (the first
 * cut of this predicate did) is exactly the defect it was written to catch: the
 * count agrees by construction, so the only thing left discriminating was the
 * pixel digest, which moves on any one-pixel jitter.
 *
 * The digest is deliberately NOT part of the verdict either. A control frame
 * that is one antialiased pixel different from the field frame is not a field;
 * `warm`/`cool` are what a field is made of. The digest is still reported,
 * because a frame must also differ from the *other* mode frames (see the
 * per-component check), where it is the right instrument.
 *
 * An EMPTY control frame is itself a failure: if the outline has no hue, every
 * frame "differs from" it and the comparison proves nothing. That is the
 * vacuous pass an `if (outline.coloured)` guard would produce.
 *
 * The floor is a fraction of the control's *own* area rather than an absolute
 * pixel count, so it does not need retuning when the viewport or the model
 * changes shape: a frame must carry at least `minShare` of the outline's pixels'
 * worth of hue to count as having drawn field over it. `minShare` of 1/64 is
 * far below the 20% a real mode frame shows and far above the 0 a blank one
 * shows.
 */
export function outlineOnlyProblem(field, outline, minShare = 1 / 64) {
  if (!outline || outline.coloured === 0) {
    return 'the outline-only control frame has no lit pixels, so no mode ' +
           'frame can be compared against it -- the field-versus-outline ' +
           'check proves nothing';
  }
  // A frame whose pixels sum to zero is black everywhere, which is not the
  // page background (0x0b0e12) and cannot be an outline. It is what
  // `toDataURL` returns for a context made WITHOUT `preserveDrawingBuffer`: the
  // buffer is cleared before the read, so every pixel is (0,0,0,0). Decoded,
  // that is `coloured === total` -- every pixel differs from the background --
  // and `sum === 0`, which no real image can be. The guard above tests
  // `coloured === 0` and so cannot see it: an all-black control passes it and
  // then fails every mode frame against a floor computed from the whole canvas.
  // Measured once, from exactly that bug: control `{coloured: 294402, sum: 0}`,
  // every mode frame reported as "the outline only" for a field that had drawn.
  if (outline.sum === 0) {
    return 'the outline-only control frame is a blank image: every pixel is ' +
           'black, so it is a cleared drawing buffer and not an outline, and ' +
           'no mode frame can be compared against it -- the field-versus-' +
           'outline check proves nothing';
  }
  const hued = field.warm + field.cool;
  const floor = Math.ceil(minShare * outline.coloured);
  if (hued < floor) {
    return `the frame is the outline only: ${hued} pixels in either lobe ` +
           'colour, against a drawing floor of ' + floor + ' for a control ' +
           `frame of ${outline.coloured} lit pixels -- the field painted no ` +
           'hue over the object, so it did not draw';
  }
  return null;
}

// Frames the check KNOWS are wrong and has been told not to fail on, named one
// by one. The owner accepted mode 1's `Erho` haze on 2026-09-30 as a minor
// cosmetic nit, so the check stops failing THE RUN on it -- while still saying
// out loud that the frame is wrong.
//
// This is deliberately not the `accountForEmptyFrame` exemption that used to
// live at the call site. That one was GATED ON THE PAGE'S OWN CAPTION: a frame
// that drew nothing was forgiven whenever the caption explained it. A caption is
// not a control, and the gate made the check's verdict depend on the page's
// prose -- so any other component that drew nothing while printing a plausible
// caption would be forgiven too. That is why it was removed (see the spec delta).
//
// A waiver here is the opposite shape: it is keyed on the check's OWN name for
// the frame (`'Erho in mode 1'`, the very string the verdict prints), it is
// listed in source, and it excuses exactly the frame it names. A second empty
// frame -- a different component, or the same component in mode 2 -- carries a
// different name and is judged normally, so it still FAILS.
export const KNOWN_ISSUE_FRAMES = new Map([
  ['Erho in mode 1',
   'the field is sign-mixed, so along a view ray the two lobes blend to a ' +
   'near-neutral haze: the frame draws (lit and framed) but paints almost no ' +
   'hue, and the lobe-hue predicate reads it as the outline. Mode 1 is ' +
   "z-polarised, so Eρ is ~8% of the mode's peak -- the caption's faint " +
   'frame. Owner accepted as a cosmetic nit, 2026-09-30.'],
]);

/**
 * The known-issue reason for `what`, or null.
 *
 * `what` is the check's own label for the frame, so a verdict that reaches the
 * log and a waiver that excuses it are keyed on the same string. A near-miss
 * label -- one that is not in the table -- gets no waiver.
 */
export function knownIssue(what) {
  return KNOWN_ISSUE_FRAMES.has(what) ? KNOWN_ISSUE_FRAMES.get(what) : null;
}

/**
 * Decide a frame's verdict from the problems its measurements found.
 *
 * This is the ONLY place a waiver is granted, and it is granted on the frame's
 * own label alone -- `problems` is never inspected for a caption, a share, or a
 * size. It returns:
 *   - `{ verdict: 'passed',  problems: [] }`      nothing was wrong
 *   - `{ verdict: 'known',   problems: [] , reason }` named waiver, run continues
 *   - `{ verdict: 'failed',  problems }`          real problems, run fails
 *
 * `check.mjs` does the logging; the DECISION lives here so it can be tested
 * without a browser. Splitting it out matters: tests that only called
 * `knownIssue()` pinned the table, not the decision, so a mutation making the
 * waiver always apply (M2) or never apply (M3) left every test green.
 */
export function judgeFrameDecision(what, problems) {
  const found = problems || [];
  if (!found.length) return { verdict: 'passed', problems: [] };
  const reason = knownIssue(what);
  if (reason) return { verdict: 'known', problems: [], reason };
  return { verdict: 'failed', problems: found };
}

export function divergingNote(sample, what, outline) {
  const lobes = [];
  if (sample.warm > 0) lobes.push(`${sample.warm} warm`);
  if (sample.cool > 0) lobes.push(`${sample.cool} cool`);
  const hued = sample.warm + sample.cool;
  // The same denominator the predicate judges against -- the control's own
  // area -- so the log reads in the units the floor is set in.
  const area = outline ? outline.coloured : sample.coloured;
  return `${what} is diverging: ${lobes.join(' / ')} px, ` +
         `${(100 * hued / area).toFixed(1)}% of the ${area} px object`;
}
