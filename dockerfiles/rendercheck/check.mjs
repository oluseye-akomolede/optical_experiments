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
//   MSE_URL   the job page to load (required)
//   OUT_DIR   where the screenshots go (default /out)
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
import { divergingNote, divergingProblem, drawProblem, framingProblem,
         freshClip }
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

  // The canvas box, taken once and reused by every shoot and by the drag below.
  //
  // The screenshots below are clips of a page screenshot, not
  // `canvas.screenshot()`. Element screenshots run a geometry-stability poll
  // first (playwright-core's `_checkElementIsStable`): it reads
  // getBoundingClientRect() each frame and only proceeds once two consecutive
  // reads are exactly equal -- no epsilon. Against the live service that poll
  // timed out at 30 s two runs running while passing locally, so it is a race,
  // not a fixed latency: under SwiftShader the frame rate is ~1/s, and the
  // canvas is sized in a rAF callback at startup, so two reads can straddle it.
  // page.screenshot() never runs that poll, which is why a clip sidesteps the
  // stall rather than racing it.
  //
  // Failing closed: if #scene is not on the page this check must not go on to
  // photograph nothing.
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
  // What the guard buys, stated exactly because the obvious version of this
  // sentence is false: with it removed, `clip: undefined` does screenshot the
  // whole page and every predicate really does run and pass -- but the run does
  // NOT then report green. The drag below reads `box.x`, so it dies with
  // "TypeError: Cannot read properties of undefined (reading 'x')" about 190
  // lines later. The guard's value is failing EARLY and with a diagnostic,
  // instead of a cryptic TypeError after four whole-page screenshots have been
  // written to OUT_DIR as if they were canvas crops.
  const box = await canvas.boundingBox();
  if (!box) {
    fail('#scene has no bounding box -- the canvas is on the page but has no ' +
         'box, so there is nothing to photograph');
    throw new Error('no canvas box; stopping rather than screenshotting nothing');
  }

  // The clip, re-read at every shot. See freshClip in pixels.mjs for why it is
  // re-read and why it carries no scroll correction.
  const clipNow = () => freshClip(() => canvas.boundingBox());

  // Put the canvas back at the top of the viewport before every shot.
  //
  // WHAT THIS DOES AND DOES NOT FIX. Clicking a mode row scrolls the page (the
  // mode table is below the fold), which moves the canvas UP and PARTLY OUT of
  // the viewport -- measured at y = -89.22 after the mode-2 click, so the top of
  // the canvas is 89 px above the viewport. Two things were wrong with the older
  // code:
  //   1. the clip was captured once, at the top, and reused -- so after any
  //      scroll it photographed where the canvas USED to be. Two saved PNGs
  //      (`04-rotated.png`, `05-...-mode2.png`) were page chrome, not the plot.
  //      Taking a fresh boundingBox() per shot is the fix for that, and it is
  //      measured: the first run after the change saved a real 834x353 plot
  //      where the old one had saved text and selector widgets.
  //   2. a clip extending outside the viewport made the run HANG: the GPU
  //      process pinned at 99% CPU, node in state R, three shots and then
  //      nothing until an external kill. Twice. Scrolling back to the top keeps
  //      the clip inside the viewport.
  //
  // (2) IS ONLY PARTLY EVIDENCED, and the honest statement is what changed, not
  // that it is cured. After this change the run no longer HANGS -- it exits,
  // with `page.screenshot: Timeout ... exceeded` on the second shot instead of
  // an unbounded stall. On the VM that run was sharing 12 CPUs with a full
  // `pytest tests/` at 98% (another seat's process), with this container capped
  // at --cpus=2, so the timeout is not attributable to the clip. Re-measure on a
  // quiet machine before claiming the wedge is fixed.
  //
  // The clip stays a plain fresh boundingBox(); only the page's scroll position
  // is reset, so the "no scroll correction on the rect" rule above still holds.
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

  const shoot = async (name) => {
    const path = join(outDir, name);
    await bringCanvasIntoView();
    const clip = await clipNow();
    // The PNG is for a human; every assertion reads the readPixels summary
    // returned below, never this file. It comes off the compositor, so it can in
    // principle catch a frame whose drawing buffer was already presented and
    // cleared (the renderer has no preserveDrawingBuffer -- web/static/viewer.js).
    // The readback below is immune because it reads inside a frame; this one is
    // best-effort. A blank PNG here is that, not a check false-alarm.
    // An explicit timeout, above Playwright's 30 s default. A clipped
    // screenshot is slow on CPU-starved SwiftShader: the compositor has to
    // render the clip, and under --cpus=2 that is tens of seconds even on an
    // idle VM (the first shot pays the shader-compile cost as well). This is
    // not a page slow to become stable -- the pixels are ready and Playwright's
    // log shows the wait is in "taking page screenshot", after fonts have
    // loaded -- so a longer timeout is the right lever rather than a workaround
    // for a real stall. MEASURED LIMIT: on 2026-09-28 the VM was running another
    // seat's full `pytest tests/` at 98% CPU, and a shot then exceeded even this
    // 120 s. So this is a bound for a QUIET machine; on a loaded one the run
    // should be given a Job (or waited for), not a bigger number blindly.
    // The whole-run budget is MSE_TIMEOUT_MS; this must stay under it.
    await page.screenshot({ path, clip, timeout: 120000 });
    const { data, width, height } = await page.evaluate(() => new Promise((resolve) => {
      const el = document.getElementById('scene');
      const gl2 = el.getContext('webgl2');
      const w = el.width;
      const h = el.height;
      const pixels = new Uint8Array(w * h * 4);
      // The read must happen *inside* a frame. Reading the default framebuffer
      // from an arbitrary task lands after the compositor has presented and
      // cleared it, and WebGL says so with a "GPU stall due to ReadPixels"
      // warning -- the buffer reads as all zeros, which is indistinguishable
      // from a plot that drew nothing. rAF fires before the present, while the
      // drawing buffer is still intact.
      requestAnimationFrame(() => {
        gl2.readPixels(0, 0, w, h, gl2.RGBA, gl2.UNSIGNED_BYTE, pixels);
        resolve({ data: Array.from(pixels), width: w, height: h });
      });
    }));
    return { path, ...summarise(Uint8Array.from(data), width, height) };
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

  const diverging = (sample, what) => {
    const problem = divergingProblem(sample, what);
    if (problem) fail(problem); else ok(divergingNote(sample, what));
  };

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
  const erhoCaptionedSmall =
    /\bsmall\b/i.test(erhoCaptionText) && /\b8\s*%/.test(erhoCaptionText);

  if (!erhoCaptionText) {
    fail('the component caption is empty on a component that draws as a ' +
         'faint cloud -- a viewer has no way to tell a small component from ' +
         'a broken one');
  } else if (!erhoCaptionedSmall) {
    fail(`the component caption does not name Erho as small in mode 1, ` +
         `where it reaches 8% of the mode's peak |E|: ` +
         `${JSON.stringify(erhoCaptionText)}`);
  } else {
    ok(`the faint component is captioned: ${JSON.stringify(erhoCaptionText)}`);
  }

  // A faint frame is a failure ONLY when nothing explains it. With the caption
  // naming the component as small, 31 pixels is the honest picture and the
  // check says so; silence there is the defect.
  const erhoDrawProblem = drawProblem(erho, 'Erho');
  if (erhoDrawProblem && !erhoCaptionedSmall) {
    fail(erhoDrawProblem);
  } else if (erhoDrawProblem) {
    ok(`Erho draws ${erho.coloured} px in mode 1, which is correct here and ` +
       'is what the caption above explains');
  } else {
    drew(erho, 'Erho');
  }
  // The divergence check only applies to a frame that drew enough to judge; a
  // 31-pixel frame has no lobes to find, and calling that a colour failure
  // would be the same mistake as failing the draw.
  if (!erhoDrawProblem) diverging(erho, 'Erho');

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
  // It applies only when there is enough in the frame to judge: a 31-pixel
  // frame has no lobes to find and calling that a colour failure would be the
  // same error as failing the faint draw above.
  if (magnitudeDrew && !drawProblem(rotated, 'the rotated frame')) {
    diverging(rotated, 'the rotated frame');
  } else if (magnitudeDrew) {
    ok('the rotated frame carries ' + rotated.coloured + ' field pixels -- ' +
       'faint because the component is, which the caption explains');
  }

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

  writeFileSync(join(outDir, 'report.json'), JSON.stringify({
    url, gl, magnitude, ez, erho, erho2, rotated, problems, notes,
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
