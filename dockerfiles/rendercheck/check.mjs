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
// Exit status is the result: 0 only if every check passed.

import { chromium } from 'playwright-core';
import { mkdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { divergingNote, divergingProblem, drawProblem, framingProblem }
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
  const shoot = async (name) => {
    const path = join(outDir, name);
    await canvas.screenshot({ path });
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
  drew(erho, 'Erho');
  diverging(erho, 'Erho');

  // Drag the canvas and require the picture to move. This is the "rotatable"
  // half of what the owner asked for, and a camera that is not the one drawing
  // the scene is the failure it catches.
  const box = await canvas.boundingBox();
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.down();
  await page.mouse.move(box.x + box.width / 2 + 220, box.y + box.height / 2 + 60,
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
  if (magnitudeDrew && drawProblem(rotated, 'the rotated frame')) {
    fail('the rotated frame is empty -- rotating lost the plot');
  }
  // And the rotated frame must still be drawn in its own hues. The defect this
  // catches is the black half-disc, seen on a rotated Erho: the negative lobe
  // clamped to black, so the frame changed when dragged (the test above passed)
  // while half the object had silently stopped being drawn. "The picture moved"
  // and "the picture is still there" are different claims.
  if (magnitudeDrew) diverging(rotated, 'the rotated frame');

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
    url, gl, magnitude, ez, erho, rotated, problems, notes, consoleErrors,
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
