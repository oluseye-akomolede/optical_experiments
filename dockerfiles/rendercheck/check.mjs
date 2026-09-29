// Load the live viewer in headless Chromium and prove the volume plot draws.
//
// What this is for: every other test in the repository can pass while the page
// shows nothing. The shader compiles only inside a browser, the 3-D texture is
// only uploaded inside a WebGL context, and the selectors only do anything
// inside a DOM. A picture is the one artifact none of them can inspect.
//
// The assertion is therefore not "the page loaded" and not "no errors" -- a
// blank canvas is a page that loaded with no errors. It is that the canvas has
// field-coloured pixels, that those pixels *change* when a different component
// is selected, and that they change again when the camera moves. Each of those
// three is a different way for the plot to be silently dead:
//
//   no colours at all        -> the shader or the texture never came up
//   same colours on two components -> the selector is wired to nothing
//   same picture after a drag      -> the camera is not the one rendering
//
// Environment:
//   MSE_URL         the job page to load (required)
//   MSE_OUTLINE_URL the same page with the field hidden (`?field=off`), used as
//                   the control frame the mode frames must differ from. When it
//                   is UNSET the check FAILS rather than skipping the
//                   comparison: a check that quietly drops its strongest
//                   predicate is indistinguishable from one that passed it, and
//                   this predicate is the one that catches a plot that never
//                   drew.
//   OUT_DIR         where the screenshots go (default /out)
//
// Running it by hand: OUT_DIR must be writable by uid 10002 (the image's USER,
// `rendercheck`). The Job is fine -- Kubernetes mounts its emptyDir writable --
// but a host bind mount is not: `mkdir -p /some/dir` gives you 0755 owned by
// you, uid 10002 gets the "other" bits (r-x) and every screenshot dies with
// "EACCES: permission denied, open '/out/01-magnitude.png'". Measured: 0755,
// 0775 and 2775 all fail; 0777 works. So `chmod 777` the host directory, or use
// a named volume. This costs 20 minutes if you meet it without the note.
//
// Exit status is the result: 0 only if every check passed.

import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { decodePng, divergingNote, divergingProblem, drawProblem,
         framingProblem, outlineOnlyProblem }
  from './pixels.mjs';

const url = process.env.MSE_URL;
const outDir = process.env.OUT_DIR || '/out';
if (!url) {
  console.error('MSE_URL is not set: there is no page to check.');
  process.exit(2);
}
mkdirSync(outDir, { recursive: true });

// A timeout that is generous on purpose: this runs on SwiftShader, which is a
// CPU rasteriser. A raymarch that is instant on a GPU takes seconds per frame
// here, and a check that flakes is a check that gets ignored.
const TIMEOUT_MS = Number(process.env.MSE_TIMEOUT_MS || 120000);

const problems = [];
const notes = [];

function fail(message) {
  problems.push(message);
  console.error(`FAIL ${message}`);
}

function ok(message) {
  notes.push(message);
  console.log(`ok   ${message}`);
}

// A pixel summary that can be compared between two frames without keeping the
// whole image, and that is insensitive to a one-pixel shift.
function summarise(buffer, width, height) {
  // Chromium hands back RGBA. The background is 0x0b0e12, so "field-coloured"
  // means a pixel that is not that, with a little tolerance for the shader's
  // own blending.
  const bg = [0x0b, 0x0e, 0x12];
  let coloured = 0;
  let sum = 0;
  let hash = 0;
  // Sign-of-colour counters. The diverging ramp puts blue on the negative lobe
  // and red on the positive one, so a frame that has field in it and a frame
  // that has the *right* field in it can be told apart by counting how much of
  // each hue is present. A single sequential ramp, or the old broken
  // `colormap(|v|) * sign(v)`, produces a frame with one hue and a great deal
  // of black -- which is exactly the defect these counters exist to catch.
  let warm = 0;    // r clearly above b: the + lobe
  let cool = 0;    // b clearly above r: the - lobe
  // Bounding box of the coloured pixels, which is how "the object fills the
  // frame" becomes an assertion rather than an opinion.
  let minX = width; let minY = height; let maxX = -1; let maxY = -1;
  const total = width * height;
  for (let i = 0; i < total; i += 1) {
    const o = i * 4;
    const r = buffer[o];
    const g = buffer[o + 1];
    const b = buffer[o + 2];
    const distance = Math.abs(r - bg[0]) + Math.abs(g - bg[1]) + Math.abs(b - bg[2]);
    const lit = distance > 24;
    if (lit) {
      coloured += 1;
      const x = i % width;
      const y = (i - x) / width;
      if (x < minX) minX = x;
      if (y < minY) minY = y;
      if (x > maxX) maxX = x;
      if (y > maxY) maxY = y;
      // The thresholds are slack on purpose: they ask which side of the ramp a
      // pixel is on, not what exact hue it is, so an antialiased edge pixel
      // part-way between the lobes is simply not counted either way.
      if (r > b + 32) warm += 1;
      else if (b > r + 32) cool += 1;
    }
    sum += r + g + b;
    // A cheap order-sensitive digest: two frames that differ anywhere in their
    // leading bytes differ here. Collisions are not a concern because the frames
    // being compared differ by whole regions, not by a bit.
    hash = (hash * 31 + r + 7 * g + 13 * b) >>> 0;
  }
  const box = maxX < 0 ? null : { minX, minY, maxX, maxY,
                                  w: maxX - minX + 1, h: maxY - minY + 1 };
  // width/height are returned rather than left for the caller to guess: an
  // earlier version of the framing check read `sample.width` and rendered as
  // "undefinedxundefined", so the assertion compared NaN and passed without
  // measuring anything.
  return { coloured, fraction: coloured / total, sum, hash, total,
           warm, cool, box, width, height };
}

const browser = await chromium.launch({
  // SwiftShader: the cluster has no GPU, and the point of the check is the
  // render path, not its speed. These flags are what make WebGL2 exist at all
  // in a headless container.
  args: [
    '--use-gl=angle',
    '--use-angle=swiftshader',
    '--enable-unsafe-swiftshader',
    '--no-sandbox',
    '--disable-dev-shm-usage',
  ],
});

try {
  const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });

  // Force `preserveDrawingBuffer: true` on every canvas context the page makes.
  //
  // Why this and not page.screenshot: a clipped page screenshot waits on a
  // COMPOSITOR frame, and in a CPU-limited container that wait is what times
  // out (measured: `page.screenshot: Timeout 30000ms exceeded ... at shoot
  // (check.mjs:231)` in the pod; the same at 120 s and 400 s on the loaded VM,
  // while the same image passed in Docker on an idle VM). Reading the canvas
  // with toDataURL never waits on the compositor.
  //
  // The grab needs `preserveDrawingBuffer`, because without it the drawing
  // buffer is cleared after the compositor presents it and toDataURL returns a
  // blank image. The viewer itself does not set the flag (it does not need to
  // for on-screen display), so it is forced here by wrapping getContext before
  // any page script runs.
  //
  // The release build sets the flag itself, and that is also why the page's
  // render loop runs at the display rate here: Chromium disables its
  // requestAnimationFrame throttling for a canvas whose drawing buffer is
  // preserved. Measured on the same container and the same page: ~60 Hz on a
  // blank page, and 0.4-0.7 Hz only once this check started awaiting a canvas
  // read (see readCanvas below for why that was, and for what removed it).
  //
  // This is a named function and not an inline literal because EVERY page the
  // check reads must get it, and one page did not. The control frame's canvas
  // came back all-zero -- not background, zero: `coloured` == the whole frame
  // (294402), `sum` == 0 -- because its context was made without the flag and
  // `toDataURL` returned the cleared buffer. An all-zero frame reads as
  // "everything is lit", so the empty-control guard (which tests
  // `coloured === 0`) never fired and the predicate failed every mode frame
  // against a control made of nothing. The control must be read on the same
  // path as every other frame, and that includes having the flag forced.
  const forcePreservedDrawingBuffer = () => {
    const orig = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function getContext(type, attrs) {
      return orig.call(this, type, Object.assign({}, attrs || {},
                                                  { preserveDrawingBuffer: true }));
    };
  };
  await page.addInitScript(forcePreservedDrawingBuffer);

  const consoleErrors = [];
  page.on('console', (message) => {
    if (message.type() !== 'error') return;
    // The text of a failed subresource is generic -- "Failed to load resource:
    // the server responded with a status of 404" -- and names no URL, so
    // filtering on the message alone cannot tell the optional surface.glb from
    // a real failure. The location carries the URL, so it is captured with the
    // text and the pair is what gets classified below.
    const where = message.location();
    const at = where && where.url ? where.url : '';
    consoleErrors.push(at ? `${message.text()} [${at}]` : message.text());
  });
  page.on('pageerror', (error) => consoleErrors.push(`uncaught: ${error.message}`));

  await page.goto(url, { waitUntil: 'load', timeout: TIMEOUT_MS });

  // The viewer only reveals the canvas once the job is done and the index has
  // landed, so the selector is the readiness signal rather than a fixed sleep.
  await page.waitForSelector('#viewer:not([hidden])', { timeout: TIMEOUT_MS });
  await page.waitForSelector('#modes-body tr', { timeout: TIMEOUT_MS });

  const fallback = await page.locator('#viewer-fallback').isVisible();
  if (fallback) {
    fail('the page says WebGL2 is unavailable -- the check cannot see a plot');
  }

  // Are we actually on WebGL2, or did the browser quietly give us WebGL1 and a
  // blank canvas? The viewer checks this too; checking it here separates "the
  // browser cannot" from "the viewer did not".
  const gl = await page.evaluate(() => {
    const canvas = document.getElementById('scene');
    const context = canvas && canvas.getContext('webgl2');
    if (!context) return { webgl2: false };
    const dbg = context.getExtension('WEBGL_debug_renderer_info');
    return {
      webgl2: true,
      renderer: dbg ? context.getParameter(dbg.UNMASKED_RENDERER_WEBGL) : 'unknown',
      floatBuffer: !!context.getExtension('EXT_color_buffer_float'),
    };
  });
  if (!gl.webgl2) fail('no WebGL2 context on the canvas');
  else ok(`WebGL2 up: ${gl.renderer}`);

  const canvas = page.locator('#scene');

  // The canvas box. The frames no longer come from a clipped page screenshot --
  // they come from the canvas's own drawing buffer (see readCanvas below) -- so
  // the box is needed only by the drag, which drives real mouse events on the
  // canvas and must aim at it.
  //
  // The clip this used to feed is gone, and with it the two stalls that made the
  // clip necessary: element screenshots run a geometry-stability poll
  // (playwright-core's `_checkElementIsStable`) that reads getBoundingClientRect()
  // each frame and only proceeds once two consecutive reads are exactly equal, no
  // epsilon -- against the live service that timed out at 30 s, because the frame
  // rate under this container is not what the poll assumes; and page.screenshot
  // waits on a compositor frame. Neither is a property of the box, which is all
  // this still computes.
  //
  // Two different cases, and only one of them is this guard's:
  //
  //   #scene missing entirely   -> boundingBox() does not return null, it times
  //     out after 30 s and throws (measured). That, not this guard, is what
  //     catches a vanished canvas.
  //   #scene present, no box    -> boundingBox() returns null (measured:
  //     display:none gives null; a zero-size box still gives an object, so the
  //     guard is not about zero size). This is the case below.
  //
  // Failing closed: if #scene is not on the page this check must not go on to
  // drive a drag at nothing.
  const box = await canvas.boundingBox();
  if (!box) {
    fail('#scene has no bounding box -- the canvas is on the page but has no ' +
         'box, so there is nothing to photograph');
    throw new Error('no canvas box; stopping rather than screenshotting nothing');
  }

  // Put the canvas back at the top of the viewport before every shot.
  //
  // Clicking a mode row scrolls the page (the mode table is below the fold),
  // which moves the canvas UP and PARTLY OUT of the viewport -- measured at
  // y = -89.22 after the mode-2 click, so the top of the canvas is 89 px above
  // the viewport. That mattered most when each frame was a clipped page
  // screenshot taken at the canvas's coordinates: a clip captured at the top and
  // reused after a scroll photographed where the canvas USED to be, and two
  // saved PNGs (`04-rotated.png`, `05-...-mode2.png`) came back as page chrome. A
  // clip extending outside the viewport also wedged the GPU process.
  //
  // The frames no longer come from a clipped screenshot, so neither of those is
  // the reason this stays. It stays because the canvas is still being driven --
  // the drag sends real mouse events at `box`, which is a viewport rect -- and
  // because a canvas scrolled out of the viewport is a canvas Chromium may stop
  // compositing. Scrolling to the top costs one wait and removes the question.
  const bringCanvasIntoView = async () => {
    await page.evaluate(() => window.scrollTo(0, 0));
    // The scroll is async in the compositor; wait for the canvas to be back at
    // a non-negative y rather than assuming a fixed delay covers it.
    await page.waitForFunction(() => {
      const el = document.getElementById('scene');
      if (!el) return false;
      return el.getBoundingClientRect().top >= 0;
    }, null, { timeout: 30000 });
  };

  // Read the canvas's own drawing buffer, as a PNG data URL.
  //
  // The read uses toDataURL and NOT readPixels, and it does not wait for a frame.
  // Both of those were measured, and both were previously assumed the other way
  // round. The measurement is in the 2-CPU container against a finished job, with
  // the machine at load ~6:
  //
  //   * requestAnimationFrame on a blank page in the same container  60 Hz
  //     (900 frames in 15 s)
  //   * requestAnimationFrame on this page, with the check reading     0.4-0.7 Hz
  //     (31 frames in the first 12 s, then +2 per 10 s)
  //   * toDataURL of the 834x368 canvas                                104-129 s
  //     (8214-char PNG, 6143 bytes decoded)
  //
  // The page's render loop is driven by `requestAnimationFrame`. Awaiting
  // toDataURL keeps two CDP round trips and the read itself outstanding while the
  // page keeps drawing, and each outstanding frame drives the next rAF toward the
  // next vsync: the loop collapses to the rate above. This helper's own
  // `requestAnimationFrame` then sits in a queue drained a few times a minute --
  // 104 s to settle, and 129 s when the rAF was armed before the toDataURL, which
  // is the whole 30 s / 120 s timeout range the pod and the VM reported. That is
  // the stall this session was hunting; it is a livelock between the check and the
  // page, not a slow machine and not the thread count.
  //
  // So: one CDP round trip, no page-side callback the page's loop can starve.
  //
  // Why toDataURL and not readPixels: toDataURL returns the drawing buffer the
  // compositor last presented, which is the frame on screen. readPixels is a
  // synchronous GPU stall whose result depends on WHEN it is called, and every
  // timing of it measured here is wrong in one direction or the other -- from an
  // arbitrary task it lands after the buffer was presented and cleared and comes
  // back all-lit (834*353 = 294402, an empty frame passing every predicate); read
  // from inside a frame it competes with the page's own draw for the same
  // SwiftShader pipe and its cost makes the starvation above worse. toDataURL has
  // neither dependency. It does need `preserveDrawingBuffer: true` -- without it
  // there is no buffer to read after presentation, which is why the init script
  // above forces the flag.
  //
  // Every read in this file goes through this one helper so the callers cannot
  // drift -- the control-frame capture was once a second, slower implementation,
  // and the two disagreed.
  const readCanvas = (target) => target.evaluate(() => {
    const el = document.querySelector('#viewer canvas') ||
               document.getElementById('scene');
    if (!el) return null;
    return el.toDataURL('image/png');
  });

  const shoot = async (name) => {
    const path = join(outDir, name);
    await bringCanvasIntoView();
    // The frame comes from the VIEWER CANVAS, not from page.screenshot.
    //
    // This is the capture fix. `page.screenshot({ clip })` waits on a
    // compositor frame, and in a CPU-limited container that wait is what times
    // out, in the pod (30 s) and on the loaded VM (120 s, 400 s). The canvas
    // read never waits on the compositor -- it reads the buffer the compositor
    // already presented.
    //
    // The PNG is written for a human; the predicates read the decoded pixels.
    // The decode is node-side on purpose: it costs the page nothing, and every
    // page-side callback the check awaits is one the page's render loop can
    // starve (see readCanvas above).
    const url = await readCanvas(page);
    if (!url) {
      throw new Error('no #viewer canvas to read -- the frame cannot be captured');
    }
    const png = Buffer.from(url.split(',')[1], 'base64');
    writeFileSync(path, png);
    const frame = await decodePng(png);
    return { path, ...summarise(frame.data, frame.width, frame.height) };
  };

  // Capture the outline-only CONTROL frame: the same page with the field
  // hidden (`?field=off`). Every mode frame is judged against this, so a frame
  // that is only the outline -- the field never drew -- is caught. Without a
  // control frame the check measured the box, not the field: every frame
  // reported the same 91637 coloured pixels and 400x337 box, and an
  // outline-only page passed.
  //
  // Returns null when no MSE_OUTLINE_URL is set or the control cannot be read,
  // in which case the field-versus-outline predicate reports that it has no
  // control rather than silently passing every frame.
  //
  // Why this waits on the DATA and not only on the selectors. The control page
  // is a SECOND WebGL context, and it used to be created 13 s in, while the
  // main page's context was live and drawing. Under a 2-CPU cap that page's
  // three data fetches -- index.json, files/surface.glb and field/1.bin -- took
  // far longer than the static assets: measured at
  // ~75 s from load in a contended run, against ~6 s for the same page when it
  // was the only container racing the gate. The 120 s selector wait expired
  // before the first of them landed, on a page that had loaded every byte of
  // its code, and the failure surfaced as "the control frame has no lit pixels":
  // could-not-ask reported as nothing-there.
  //
  // So the wait is on the resource the selector actually depends on, not on the
  // selector alone. It names what has to happen, and when the fetch does arrive
  // and the viewer still does not unhide, the catch reports which resources did
  // arrive instead of blaming the pixels.
  const captureOutlineFrame = async () => {
    const outlineUrl = process.env.MSE_OUTLINE_URL;
    if (!outlineUrl) return null;
    const p = await browser.newPage({ viewport: { width: 1024, height: 768 } });
    // The same flag the main page gets, for the same reason: without it the
    // control's context has no buffer to read after presentation and its canvas
    // read returns an all-zero image, which summarises as "everything lit" and
    // is indistinguishable from a real control by the empty-control guard.
    await p.addInitScript(forcePreservedDrawingBuffer);
    const seen = new Set();
    p.on('response', (r) => {
      const u = r.url();
      if (u.includes('/index.json')) seen.add('index.json');
      if (u.includes('/files/surface.glb')) seen.add('surface.glb');
      if (/\/field\/\d+\.bin/.test(u)) seen.add('field');
    });
    try {
      await p.goto(outlineUrl, { waitUntil: 'load', timeout: TIMEOUT_MS });
      await p.waitForResponse((r) => r.url().includes('/index.json'),
        { timeout: TIMEOUT_MS });
      await p.waitForSelector('#viewer:not([hidden])', { timeout: TIMEOUT_MS });
      await p.waitForSelector('#modes-body tr', { timeout: TIMEOUT_MS });
      // Give the outline's fetch and the render loop the same settle the main
      // page gets, so the control and the frames share conditions. The render
      // loop here runs at the display rate (measured 60 Hz on a blank page in
      // this same container, and the release build sets preserveDrawingBuffer,
      // which is what disables Chromium's rAF throttling), so a few seconds is
      // many frames and not a race.
      await p.waitForTimeout(4000);
      const url = await readCanvas(p);
      if (!url) throw new Error('no canvas on the outline-only page');
      const frame = await decodePng(Buffer.from(url.split(',')[1], 'base64'));
      return summarise(frame.data, frame.width, frame.height);
    } catch (error) {
      console.error(
        `note: could not capture the outline-only control frame: ${error}` +
        ` (fetches the page completed: ${[...seen].join(', ') || 'none'})`);
      return null;
    } finally {
      await p.close().catch(() => {});
    }
  };

  // Give the render loop a few frames; the lazy field fetch has to land first.
  await page.waitForTimeout(4000);

  // The predicates live in pixels.mjs so they can be tested without a browser.
  // They used to be inline here, which meant the only way to exercise them was
  // to run this whole check against a live page -- so they went untested, and a
  // predicate that is wrong in the direction of *passing* would have reported
  // green forever.
  function drew(sample, what) {
    const problem = drawProblem(sample, what);
    if (problem) { fail(problem); return false; }
    ok(`${what} draws: ${(sample.fraction * 100).toFixed(2)}% of the canvas ` +
       `(${sample.coloured} px)`);
    return true;
  }

  // The outline-dependent verdicts are DEFERRED to the end of the run.
  //
  // They need the outline-only CONTROL frame, and that control is a SECOND
  // WebGL context in a container already running one. Captured mid-run, while
  // the main page's context is live and drawing, its three data fetches take
  // far longer than the static assets under a 2-CPU cap -- measured at ~75 s
  // from load in a contended run against ~6 s when it was the only context
  // racing the gate (see captureOutlineFrame). That is what produced the failed
  // run's four "the outline-only control frame has no lit pixels" verdicts: a
  // control that could not finish loading reads as a control with nothing in
  // it, and every frame judged against it is then failed. Waiting longer does
  // not fix contention; removing the contention does.
  //
  // So the control is captured LAST, after the main page has closed and its
  // context is gone, and the verdicts that use it are collected here and
  // flushed then. The frame summaries they judge are retained in their own
  // variables, so nothing about the pictures is lost by the move -- only the
  // ORDER of the log lines, which is why the deferred verdicts are emitted
  // under a heading that says so.
  let outline = null;
  const pendingOutlineVerdicts = [];
  const fieldVsOutline = (sample, what, opts = {}) => pendingOutlineVerdicts.push(() => {
    const problem = outlineOnlyProblem(sample, outline);
    if (problem && opts.accountForEmptyFrame) {
      // The frame matches the outline-only control AND the page has already
      // explained why, in the caption, naming the component's measured share.
      // That is the correct picture for a below-floor component, so it is
      // reported as such rather than as a failure -- but stated, never silent,
      // because an unexplained empty frame is exactly the defect this predicate
      // exists to catch.
      ok(`${what} matches the outline-only control ` +
         `(${sample.warm + sample.cool} lobe-coloured px), which is correct ` +
         `here: the page captions this component as drawn below its own floor`);
    } else if (problem) {
      fail(`${what}: ${problem}`);
    } else if (outline) {
      // The hue, not the lit count: the count is the object's box and is the
      // same in every frame (measured, 91637 with the field on and off); the
      // field's whole contribution is the hue it paints over that box.
      ok(`${what} differs from the outline-only control ` +
         `(${sample.warm + sample.cool} lobe-coloured px vs ` +
         `${outline.warm + outline.cool} in the control)`);
    }
    // With no control frame at all, `outline` is null, `outlineOnlyProblem`
    // returns its "the control is missing or unreadable" message, and this
    // FAILS -- a missing control is a FAIL, never a silent skip, and it is NOT
    // forgiven by `accountForEmptyFrame` (a null control is not an explained
    // empty frame). That is deliberate: a check that quietly drops its
    // strongest predicate is the "could not ask looks like nothing there"
    // shape, and this predicate is the one that catches a plot that never drew.
  });

  const magnitude = await shoot('01-magnitude.png');
  const magnitudeDrew = drew(magnitude, '|E|');

  // Switch coordinate system and component the way a user would, and require
  // the picture to change. A selector wired to nothing leaves an identical
  // frame, which is invisible without this comparison.
  await page.selectOption('#component-system', 'cartesian');
  await page.selectOption('#component-name', 'Ez');
  await page.waitForTimeout(3000);
  const ez = await shoot('02-cartesian-ez.png');
  if (ez.hash === magnitude.hash) {
    fail('selecting Ez gave a byte-identical frame to |E| -- the component ' +
         'selector is not reaching the texture');
  } else {
    ok('the component selector changes the picture');
  }
  drew(ez, 'Ez');
  fieldVsOutline(magnitude, '|E|');
  fieldVsOutline(ez, 'Ez');

  {
    const problem = framingProblem(magnitude);
    if (problem) {
      fail(problem);
    } else {
      const frame = Math.max(magnitude.width, magnitude.height);
      const across = Math.max(magnitude.box.w, magnitude.box.h);
      ok(`the plot is framed: ${magnitude.box.w}x${magnitude.box.h} px, ` +
         `which is ${(across / frame * 100).toFixed(0)}% of the ` +
         `${magnitude.width}x${magnitude.height} frame`);
    }
  }

  const diverging = (sample, what) => pendingOutlineVerdicts.push(() => {
    const problem = divergingProblem(sample, what, outline);
    if (problem) fail(problem); else ok(divergingNote(sample, what, outline));
  });

  // The component caption's text, or '' when it is hidden or absent. Both are
  // "no caption", which is the failure the caller reports.
  const erhoCaption = async (page) => page.evaluate(() => {
    const el = document.getElementById('component-note');
    if (!el || el.hidden) return '';
    return el.textContent.trim();
  });
  diverging(ez, 'Ez');

  await page.selectOption('#component-system', 'cylindrical');
  await page.selectOption('#component-name', 'Erho');
  await page.waitForTimeout(3000);
  const erho = await shoot('03-cylindrical-erho.png');
  if (erho.hash === ez.hash) {
    fail('selecting Erho gave a byte-identical frame to Ez -- the system ' +
         'selector is not reaching the transform');
  } else {
    ok('the coordinate-system selector changes the picture');
  }
  // The faint-component caption, asserted where it is faint and where it is
  // not. This is the check for a misreading that actually happened: mode 1's
  // Erho draws as an almost empty frame because the component is genuinely
  // small in this mode (8% of the mode's peak |E|), and an almost empty frame
  // is exactly what a render bug looks like. The picture cannot say which it
  // is; the caption has to.
  //
  // The caption is read BEFORE the draw is judged, because the two are one
  // claim between them: "this component is small here, and the page says so".
  // Judging the draw alone would fail correct data -- mode 1's Erho really does
  // draw 31 pixels -- and a check that fails on correct data is worse than no
  // check, because it will be ignored and then it is not there for the failures
  // that matter.
  //
  // Asserting only the faint case would be half a check: a caption that is
  // hard-coded to "this component is small" would pass it while telling every
  // user a lie. The strong case below is the control.
  const erhoCaptionText = await erhoCaption(page);
  // TWO captions are honest for mode 1's Erho, and the check must not demand
  // one of them. The viewer has three branches, keyed on where the component's
  // p99 share sits relative to its own constants (viewer.js):
  //
  //   share < THRESHOLD (0.12)      "almost empty ... under the 12% drawing
  //                                   floor ... not a fault"
  //   share < SMALL_SHARE (0.2)     "is small in this mode ... faint cloud"
  //   otherwise                     "reaches N% ... should read as strongly"
  //
  // Mode 1's Erho is measured at 8.26%, so the LIVE caption is the below-floor
  // one, and it is true. An earlier cut of this check required the word
  // "small", which only appears in the middle branch, and so FAILED a correct
  // page -- the exact over-tight predicate this file keeps being burned by.
  // What the check may require is that the caption is PRESENT, that it names
  // the component's real share, and that it says the emptiness is physical
  // rather than a fault; which of the two true branches it chose is the
  // viewer's business.
  const erhoNamesTheTest =
    /\b8\s*%/.test(erhoCaptionText) &&
    (/\bsmall\b/i.test(erhoCaptionText) || /\bnot a fault\b/i.test(erhoCaptionText));

  if (!erhoCaptionText) {
    fail('the component caption is empty on a component that draws as an ' +
         'almost empty frame -- a viewer has no way to tell a small component ' +
         'from a broken one');
  } else if (!erhoNamesTheTest) {
    fail(`the component caption does not name Erho's real share in mode 1, ` +
         `where it reaches 8% of the mode's peak |E|, and does not say the ` +
         `emptiness is physics rather than a fault: ` +
         `${JSON.stringify(erhoCaptionText)}`);
  } else {
    ok(`the faint component is captioned: ${JSON.stringify(erhoCaptionText)}`);
  }

  // A faint frame is a failure ONLY when nothing explains it. With the caption
  // naming the component's real share, the near-empty picture is the honest
  // one and the check says so; silence there is the defect.
  const erhoDrawProblem = drawProblem(erho, 'Erho');
  if (erhoDrawProblem && !erhoNamesTheTest) {
    fail(erhoDrawProblem);
  } else if (erhoDrawProblem) {
    ok(`Erho draws ${erho.coloured} px in mode 1, which is correct here and ` +
       'is what the caption above explains');
  } else {
    drew(erho, 'Erho');
  }
  // NO `diverging(erho)` HERE -- and its absence is the fix, not an oversight.
  //
  // This frame is mode 1's Erho at a measured share of 8.26% against the
  // viewer's own 12% drawing floor: the shader discards every sample, so a
  // frame with NO hue is the designed, correct picture, and the caption above
  // says so. Routing it to the lobe-hue predicate asks a frame that drew
  // nothing to carry hue, which is a demand the product does not meet and does
  // not intend to. Measured on a correct run, that one call produced two of the
  // three failures ("no colour at all" and "the frame is the outline only").
  //
  // The question "did this frame draw at all" is the one that applies here, and
  // `fieldVsOutline` below is where it is asked: the frame must still differ
  // from the outline-only control, so a page whose field never drew is still
  // caught even though this frame is too empty to judge by hue.
  // The field-versus-outline comparison is what the near-empty case needs most:
  // the caption is text, and text that says "almost empty" is true of a frame
  // whose field never drew either. Without some comparison the caption check
  // above can pass on a page whose field never drew at all.
  //
  // BUT on THIS frame the comparison's expected answer is "no difference". Mode
  // 1's Erho is below the viewer's own drawing floor, so the field paints no hue
  // and the frame is supposed to look exactly like the outline-only control --
  // that is what "below the level the picture draws at all" means. Running it
  // reads the correct picture as "the frame is the outline only" and fails a
  // good page (measured: this was the last of the three failures on the correct
  // run).
  //
  // So the two frames are compared, and the empty frame is forgiven ONLY when
  // the page's own caption has already accounted for it -- `erhoNamesTheTest`,
  // which requires the caption to name the component's measured 8% share AND to
  // say the emptiness is physics, not a fault. A page whose field never drew
  // cannot satisfy that (it does not know the number, and a hard-coded caption
  // fails the mode-2 control), so the guarantee the comparison existed to give
  // is kept; what is dropped is only its false alarm on the correct page.
  fieldVsOutline(erho, 'Erho in mode 1', { accountForEmptyFrame: erhoNamesTheTest });

  // Mode 2 is the control case: the SAME component and coordinate system, in a
  // mode where Erho reaches 98% of |E|. It must draw strongly and must not be
  // captioned as small. Together the two say the caption tracks the field
  // rather than the component's name.
  //
  // NOTE: this switches the page to mode 2 and leaves it there, so the rotation
  // below operates on MODE 2's picture. Any assertion about the rotated frame
  // must therefore compare it with erho2, not erho.
  await page.click('#modes-body tr:nth-child(2)');
  await page.waitForTimeout(4000);
  const erho2 = await shoot('05-cylindrical-erho-mode2.png');
  drew(erho2, 'Erho in mode 2');
  diverging(erho2, 'Erho in mode 2');
  fieldVsOutline(erho2, 'Erho in mode 2');
  {
    const text = await erhoCaption(page);
    if (!text) {
      fail('the component caption is empty on mode 2, where Erho reaches 98% ' +
           'of the peak |E|');
    } else if (/\bsmall\b/i.test(text)) {
      fail(`the component caption calls Erho small in mode 2, where it reaches ` +
           `98% of the mode's peak |E| -- the caption is not tracking the ` +
           `field: ${JSON.stringify(text)}`);
    } else {
      ok(`the strong component is captioned as strong: ${JSON.stringify(text)}`);
    }
  }
  // And the two modes must not be the same picture: if switching modes left the
  // texture untouched, both captions and both frames above would be describing
  // one field.
  if (erho2.hash === erho.hash) {
    fail('mode 2 drew a byte-identical frame to mode 1 -- the mode selection is ' +
         'not reaching the texture, so the caption comparison above is ' +
         'comparing one field with itself');
  } else {
    ok('selecting a mode changes the picture');
  }

  // Drag the canvas and require the picture to move. This is the "rotatable"
  // half of what the owner asked for, and a camera that is not the one drawing
  // the scene is the failure it catches.
  //
  // A FRESH viewport rect for the drag. The `box` taken at the top is stale
  // here: the mode click above scrolled the page by 284 px, so the canvas has
  // moved and mouse events (viewport coordinates) must follow it. Taking it
  // once and reusing it is what the previous version did, and it is what made
  // the drag land 284 px away from the canvas -- which still "rotated" something
  // often enough to look like a pass.
  const dragBox = await canvas.boundingBox();
  await page.mouse.move(dragBox.x + dragBox.width / 2,
                        dragBox.y + dragBox.height / 2);
  await page.mouse.down();
  await page.mouse.move(dragBox.x + dragBox.width / 2 + 220,
                        dragBox.y + dragBox.height / 2 + 60,
                        { steps: 20 });
  await page.mouse.up();
  await page.waitForTimeout(3000);
  const rotated = await shoot('04-rotated.png');
  if (rotated.hash === erho.hash) {
    fail('dragging the canvas did not change the frame -- the plot does not ' +
         'rotate');
  } else {
    ok('dragging rotates the plot');
  }
  // A rotation can only be *seen* on a frame that has something in it. If the
  // preceding component drew nothing, the two frames are the same background
  // and the comparison above cannot tell "did not rotate" from "nothing there
  // to rotate" -- so the two are checked together rather than either alone.
  //
  // What is asserted is that rotating did not leave the plot GONE -- not that
  // it kept most of its pixels. A rotation genuinely changes how much of the
  // volume projects onto the screen, so the count can drop a long way and be
  // entirely correct; measured on the real job, mode 2's Erho went from 58101
  // field pixels to 19116 after the drag below and that is the right picture
  // from another angle. An earlier version of this guard required the rotated
  // frame to keep half its pixels and would have failed that correct frame,
  // which is the over-tight predicate this check has been burned by before.
  //
  // The real failure it exists for is "rotation lost the plot", so that is what
  // it tests: something must still be there, and the frame-by-frame comparison
  // above already established the camera moved.
  if (magnitudeDrew && rotated.coloured < 500) {
    fail(`the rotated frame has only ${rotated.coloured} field pixels ` +
         `(against ${erho2.coloured} before the drag) -- rotating lost the plot`);
  }
  // And the rotated frame must still be drawn in its own hues. The defect this
  // catches is the black half-disc, seen on a rotated Erho: the negative lobe
  // clamped to black, so the frame changed when dragged (the test above passed)
  // while half the object had silently stopped being drawn. "The picture moved"
  // and "the picture is still there" are different claims.
  //
  // THIS FRAME IS MODE 2's PICTURE FROM A NEW CAMERA -- see the NOTE at the mode
  // click above: the page is left on mode 2 and the drag happens after. So it is
  // the same component (Erho) as the frame at `erho2`, and its hue is judged on
  // the same grounds. That matters, because the caption sits on the PAGE rather
  // than on the frame: the mode-1 caption ("under the 12% drawing floor") is
  // stale here and describes a different mode's picture, so it is not evidence
  // about this frame either way.
  //
  // It applies only when there is enough in the frame to judge. `drawProblem`
  // asks the weaker question -- did this frame draw anything at all -- and a
  // frame it rejects has no lobes to find, so calling that a colour failure
  // would be the same error as failing the near-empty mode-1 frame above.
  //
  // It is a weaker question than the one this predicate asks, deliberately: a
  // frame too empty to clear `drawProblem` is judged by `fieldVsOutline` below
  // instead, which catches a plot that never drew without demanding hue from a
  // frame the viewer left empty on purpose.
  if (magnitudeDrew && !drawProblem(rotated, 'the rotated frame')) {
    diverging(rotated, 'the rotated frame');
  } else if (magnitudeDrew) {
    ok('the rotated frame carries ' + rotated.coloured + ' field pixels -- ' +
       'too few to judge its hue');
  }
  // The rotated frame must still carry the FIELD, not just the outline. This is
  // the comparison the drag test cannot make on its own: a frame that rotated
  // to show only the wireframe has changed, so "dragging changes the picture"
  // passes, while the plot itself is gone.
  if (magnitudeDrew) fieldVsOutline(rotated, 'the rotated frame');

  if (consoleErrors.length) {
    // Console errors are reported but are only fatal when they are the viewer's
    // own: a 404 for a surface a job does not have is survivable and the code
    // says so. The URL is part of the message (see the console listener), and
    // the match is against the whole line so it can only excuse that one file.
    const real = consoleErrors.filter(
      (line) => !/surface\.glb|favicon\.ico/.test(line));
    if (real.length) real.forEach((line) => fail(`console error: ${line}`));
    else ok('console errors are limited to the optional surface.glb');
  } else {
    ok('no console errors');
  }

  // Now every frame is shot and every page interaction is done, so the main
  // page can be closed. Only then is the control captured, as the sole WebGL
  // context in the container -- which is the fix for the contended fetch that
  // made the failed run report "the control frame has no lit pixels".
  await page.close().catch(() => {});
  outline = await captureOutlineFrame();
  if (!outline) {
    // Fail-closed, like outlineOnlyProblem: a check that could not take its own
    // control frame must say so, not report the frames it compares against that
    // control as if the comparison had run. "Could not ask" is not "all clear".
    //
    // What is dropped is the SET of diverging/field-versus-outline judgments,
    // and it is dropped visibly: each is replaced by its own failure naming the
    // quantity it could not weigh. The other frame judgments (the draw, the
    // caption, the selectors, the rotation) still run below.
    fail(`the check could not capture its own control frame from ` +
         `${process.env.MSE_OUTLINE_URL || '(no MSE_OUTLINE_URL set)'} -- ` +
         `${pendingOutlineVerdicts.length} field-versus-outline and lobe-hue ` +
         `judgment(s) cannot be made, and each is reported as a failure below`);
  }
  if (pendingOutlineVerdicts.length) {
    console.log('\n-- field-versus-outline verdicts (control captured after ' +
                'the main page closed) --');
    pendingOutlineVerdicts.forEach((verdict) => verdict());
  }

  writeFileSync(join(outDir, 'report.json'), JSON.stringify({
    url, gl, magnitude, ez, erho, erho2, rotated, outline, problems, notes,
    consoleErrors,
  }, null, 2));
} catch (error) {
  fail(`the check itself threw: ${error && error.stack ? error.stack : error}`);
} finally {
  await browser.close().catch(() => {});
}

// The exit code is set here and nowhere else. An earlier version called
// process.exit(1) inside a try/finally whose own exit then won, so a run that
// had found problems still reported success -- a check that cannot fail is
// worse than no check, because it is trusted. `process.exitCode` rather than
// `process.exit()` also lets stdout flush, so the reason reaches the log.
if (problems.length) {
  console.error(`\n${problems.length} problem(s):`);
  problems.forEach((p) => console.error(`  - ${p}`));
  process.exitCode = 1;
} else {
  console.log(`\nall render checks passed; screenshots in ${outDir}`);
}
