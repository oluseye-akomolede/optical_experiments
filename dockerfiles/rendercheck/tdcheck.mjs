// In-browser check of the time-domain result card on the REAL job page
// (job.html from the Flask app, viewer.js + tdplot.js + tdmovie.js from
// web/static) fed the files a real solve wrote: the trace and spectrum plots
// and the field movie, as written and in each failure case (caveats, trace
// 502, a frame 502, no movie, an all-zero movie, a throw after every frame has
// loaded). The uploaded object's outline is checked the same way: the box the
// page says it drew against the box the outline artifact's own vertices span
// (its glTF accessor min/max, so a viewer that moved or dropped something
// disagrees), that box INSIDE the box the manifest's sample positions span (on
// an open boundary the samples carry pad cells past the object, so equality
// there would fail a correct job), and every sampled frame drawn twice -- with
// the outline and with the outline route answering 404 -- so the pixels it
// adds are the outline's and nothing else. Nothing is fetched from the network.
//
// TD_DIR holds a job whose data a real solve wrote, not a job this check
// imagines: scripts/tdcheck_fixture.py writes a tiny solve, and
// scripts/tdcheck_fetch.py mirrors a real production job into the same layout.
// Everything expected of the plots is derived from the manifest this check is
// served and from the page's own drawing rules -- the module imported below is
// the very tdplot.js the page was served -- so the probe count, the unit each
// axis should pick for this run's span and band, and the tick factor the
// trace's own values need all follow the run. A one-probe, femtosecond,
// terahertz job is checked as strictly as a multi-probe one.
//
// A movie caption still reading "Loading" after 60 s is recorded as a FAIL and
// ends the run with exit 1: that is the a6127d3 defect (a post-load throw left
// the caption stuck), and the wait is what bounds it.
//
// The page draws inside requestAnimationFrame, and a real movie's per-frame
// work starves Playwright's actionability waits and element screenshots, which
// both wait on animation frames (measured: a real 58x53x48 movie hung the
// run at a slider fill and again at a canvas screenshot). So the slider is set
// and fired inside one evaluate, and the canvas is read with toDataURL against
// a context forced to preserveDrawingBuffer, after the page's own loop has
// drawn a further frame -- counted below, so the read never races the loop.
// And the pages are read one at a time, bringing each to the front first: a
// backgrounded page has its animation frames throttled, so its canvas can be
// several moves stale (measured: two pages open at once made one frame's
// with-outline and without-outline shots identical, because the read was of a
// canvas from before that frame).
//
// Waits on the page's progress are PROGRESS-BASED, for that same reason: the
// movie's frames cost on the order of a second each on the open-boundary
// fixture in this 2-CPU container (software WebGL, no GPU) and this VM is
// shared with every other rig, so a wall-clock budget on "the movie has played
// on" is really a budget on how busy the machine is -- which is how this
// check's first open-boundary run failed (review row 2026-10-03: the same
// mirror ran the play through in three seconds when the wait was taken out,
// with no page errors). A wait FAILs on a STALL -- no progress for stallMs(),
// 20x the slowest frame measured here, floor 90 s -- and every timed-out wait
// records a FAIL naming what the page showed and threw, instead of ending the
// run with no verdict at all.
//
// Inputs: TD_DIR (default /work). scripts/run_tdcheck.sh builds the fixture
// inputs and runs this, bounded.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';
const require = createRequire('/opt/rendercheck/check.mjs');
const { chromium } = require('playwright-core');

const DIR = process.env.TD_DIR || '/work';
const fails = [];
const ok = (c, msg) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) fails.push(msg); };
const TYPES = { '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json' };
// The page's own plotting rules: the axis units, the tick factor, the trace
// decode and the spectrum's scale claim are this module's, not a copy of them
// written out again here. Imported from TD_DIR, so it is the file the page was
// served with the job.
const { decodeTraces, siPrefix, axisExponent, exponentText, SPECTRUM_SCALE } =
  await import(pathToFileURL(path.join(DIR, 'static/tdplot.js')).href).catch((e) => {
    console.log(`FAIL  TD_DIR (${DIR}) has no readable static/tdplot.js: ${e.message}`);
    return process.exit(1);
  });
// The movie's own two drawing constants, for the same reason: the caption's
// percentile and cut are these numbers, so the caption contract has one source.
const { SCALE_PERCENTILE, MOVIE_THRESHOLD } =
  await import(pathToFileURL(path.join(DIR, 'static/tdmovie.js')).href).catch((e) => {
    console.log(`FAIL  TD_DIR (${DIR}) has no readable static/tdmovie.js: ${e.message}`);
    return process.exit(1);
  });
// The pixel reading is the rendercheck image's own pixels.mjs (the Dockerfile
// COPYs it beside this file, which run_tdcheck.sh mounts), so "a lit pixel" and
// "a changed pixel" mean here exactly what they mean in the other render
// checks instead of a second definition written out again.
const { decodePng, distanceFromBackground, LIT_DISTANCE } =
  await import('./pixels.mjs').catch((e) => {
    console.log(`FAIL  no readable pixels.mjs beside tdcheck.mjs: ${e.message}`);
    return process.exit(1);
  });

const manifest = JSON.parse(fs.readFileSync(path.join(DIR, 'files/spectrum.json'), 'utf8'));
const PROBES = manifest.probes;
const NP = PROBES.length;
// The trace the page decodes, decoded here with the page's own decoder: the
// factor its y axis must state follows from these values.
const raw = fs.readFileSync(path.join(DIR, 'files/trace.bin'));
const rows = decodeTraces(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength),
                          manifest.trace.shape);
// The page pads each drawn range 5% each side before choosing the tick factor
// (tdplot.range, which the page does not export). The padding can raise the
// trace's peak by at most 10% of it, so the factor the page picks is the one
// axisExponent gives at either end of that; both are accepted, and nothing
// else is. The page's one other case is a trace with no span at all, which it
// draws as [v-1, v+1] -- a factor of 0 -- and which the swing check below
// already fails.
const lo = rows.reduce((m, r) => r.reduce((n, v) => Math.min(n, v), m), Infinity);
const hi = rows.reduce((m, r) => r.reduce((n, v) => Math.max(n, v), m), -Infinity);
const peak = Math.max(Math.abs(lo), Math.abs(hi));
const traceFactors = [...new Set([axisExponent(peak), axisExponent(peak * 1.1),
  ...(hi > lo ? [] : [0])])];
const traceTitle = 'field component (arb. units)';
const traceTitles = traceFactors.map((e) => traceTitle + (e ? ` ${exponentText(e)}` : ''));
// The units the page's own axis rule picks for this run: the trace's span, and
// the top of the frequency band.
const spanUnit = siPrefix(manifest.steps * manifest.dt_s);
const axisUnit = siPrefix(Math.max(manifest.steps - 1, 1) * manifest.dt_s);
const bandUnit = siPrefix(Math.max(Math.abs(manifest.band_hz[0]), Math.abs(manifest.band_hz[1])));
const CAVEATS = ['thin curved features are resolved only at the coarse cell'];
// Everything this check knows about the movie comes from the manifest it is
// served, not from constants here: whether the run recorded one at all, the
// frame count, the name width, and the spacing the caption is expected to
// report. A fixture may be any window of any run.
// How many pixels a frame must gain when the outline is drawn: a share of that
// frame's own lit field pixels, because the outline's edges cross the object,
// which is where the field is -- so a larger frame demands a longer line rather
// than the same count. The floor is a *kind* test (an object-sized mark, not a
// stray antialiased pixel) and is deliberately far below what a drawn outline
// measures: the measured count is printed in the check's own message.
const OUTLINE_SHARE = 1 / 64;
const FR = manifest.frames || null;
const DIGITS = (FR && FR.digits) || 2;
const frameName = (n) => `frame_${String(n).padStart(DIGITS, '0')}.bin`;
const frameTimes = FR ? (FR.times_s || FR.steps.map((s) => s * manifest.dt_s)) : [];
const spacingS = frameTimes.length > 1
  ? (frameTimes[frameTimes.length - 1] - frameTimes[0]) / (frameTimes.length - 1) : 0;
const cycles = manifest.source_frequency_hz
  ? spacingS * manifest.source_frequency_hz : null;

// --- reading the legend's own position back ---------------------------------
// The prefix table is the page's own: siPrefix over every decade, so a unit the
// page cannot write is a unit this check cannot read.
const FACTOR = new Map();
for (let e = -24; e <= 24; e += 1) {
  const u = siPrefix(10 ** e);
  FACTOR.set(u.prefix, u.factor);
}
/** The finest gap between the sample positions a manifest lists: the job's own
 *  cell in metres, or null when it lists none. Read here rather than imported:
 *  the page served in TD_DIR may predate that export. */
function fineStep(positions) {
  if (!Array.isArray(positions)) return null;
  let s = Infinity;
  for (const axis of positions) {
    for (let i = 1; i < (axis || []).length; i += 1) {
      const d = Math.abs(Number(axis[i]) - Number(axis[i - 1]));
      if (d > 0 && d < s) s = d;
    }
  }
  return Number.isFinite(s) ? s : null;
}
/** The movie's own frame, as the manifest's positions bound it: [[min,max] per
 *  axis] in metres, or null when the manifest lists no positions. A reduce, not
 *  a spread: a real job's axis holds tens of thousands of samples and a spread
 *  of that length is a call-stack overflow, not a measurement. */
function positionsBox(positions) {
  if (!Array.isArray(positions) || !positions.length) return null;
  const box = positions.map((axis) => {
    let lo = Infinity, hi = -Infinity;
    for (const v of axis || []) {
      const n = Number(v);
      if (!Number.isFinite(n)) continue;
      if (n < lo) lo = n;
      if (n > hi) hi = n;
    }
    return Number.isFinite(lo) ? [lo, hi] : null;
  });
  return box.every(Boolean) ? box : null;
}
/** Where the page says it drew the outline: ``minx,miny,minz;maxx,maxy,maxz``
 *  in metres, the dataset entry the movie renderer writes. Null (and a reason)
 *  when it states something else -- including 'none', which is what a page
 *  that could not draw the outline says. */
function outlineBox(text) {
  const m = /^([-\d.eE,]+);([-\d.eE,]+)$/.exec(String(text || ''));
  if (!m) return null;
  const nums = (t) => t.split(',').map(Number);
  const box = { min: nums(m[1]), max: nums(m[2]) };
  if (box.min.length !== 3 || box.max.length !== 3
      || [...box.min, ...box.max].some((v) => !Number.isFinite(v))) return null;
  return box;
}
const fmtBox = (box) => box.map(([lo, hi], a) => `${'xyz'[a]} ${lo}..${hi}`).join(', ');

/** The vertex box the outline FILE itself carries -- every mesh primitive's
 *  POSITION accessor min/max unioned -- in the file's own units (this job's
 *  metres). glTF requires min/max on an accessor and this file's nodes carry no
 *  transforms, so no matrix walk is needed; a node that grows one, or an
 *  accessor without min/max, is a *reason* rather than a measurement -- this
 *  check refuses to place what it cannot read.
 *
 *  Read from the outline artifact rather than derived from the manifest,
 *  because the manifest records the SAMPLED positions, which on the product's
 *  default open boundary carry OPEN_PAD_CELLS*dx of padding past the object on
 *  every side (fdtd_layout.py:204). Treating those as the object's box is what
 *  a review row rejected on 2026-10-03, and it failed a correct open job.
 *  Returns {box} or {why}. */
function glbBox(bytes) {
  const buf = Buffer.from(bytes);
  if (buf.length < 20 || buf.readUInt32LE(0) !== 0x46546c67) return { why: 'it is not a glb' };
  let at = 12, json = null;
  while (at + 8 <= buf.length && !json) {
    const len = buf.readUInt32LE(at), type = buf.readUInt32LE(at + 4);
    if (type === 0x4e4f534a) {
      try {
        json = JSON.parse(buf.subarray(at + 8, at + 8 + len).toString('utf8'));
      } catch (e) {
        return { why: `its JSON chunk will not parse (${e.message})` };
      }
    }
    at += 8 + len;
  }
  if (!json) return { why: 'it has no JSON chunk' };
  const moved = (json.nodes || []).find((n) => n
    && (n.matrix || n.translation || n.rotation || n.scale));
  if (moved) {
    return { why: `its node ${JSON.stringify(moved.name === undefined ? '?' : moved.name)} `
                  + 'carries a transform this check cannot place' };
  }
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  let prims = 0;
  for (const mesh of json.meshes || []) {
    for (const prim of mesh.primitives || []) {
      const index = prim.attributes && prim.attributes.POSITION;
      const acc = index == null ? null : (json.accessors || [])[index];
      if (!acc || !Array.isArray(acc.min) || !Array.isArray(acc.max)
          || acc.min.length !== 3 || acc.max.length !== 3) {
        return { why: 'a primitive has no POSITION accessor min/max' };
      }
      prims += 1;
      for (let a = 0; a < 3; a += 1) {
        lo[a] = Math.min(lo[a], Number(acc.min[a]));
        hi[a] = Math.max(hi[a], Number(acc.max[a]));
      }
    }
  }
  const box = lo.map((v, a) => [v, hi[a]]);
  if (!prims || !box.flat().every(Number.isFinite)) {
    return { why: 'it states no POSITION accessor min/max' };
  }
  if (!box.every(([a, b]) => b > a)) return { why: `its own box is degenerate: ${fmtBox(box)}` };
  return { box };
}

/** A canvas shot decoded to RGBA pixels (pixels.mjs widens RGB to RGBA). */
async function shotPixels(shot) {
  const { data, width, height } = await decodePng(Buffer.from(shot.url.split(',')[1], 'base64'));
  return { data, width, height };
}
/** Pixels two shots of the same size disagree on anywhere in RGB, or -1 when
 *  they are not the same size (which is itself a difference, not a pass). */
function differingPixels(a, b) {
  if (a.width !== b.width || a.height !== b.height) return -1;
  let n = 0;
  for (let i = 0; i < a.width * a.height; i += 1) {
    const o = i * 4;
    if (a.data[o] !== b.data[o] || a.data[o + 1] !== b.data[o + 1]
        || a.data[o + 2] !== b.data[o + 2]) n += 1;
  }
  return n;
}
/** A short fingerprint of a shot, so a failing comparison says WHAT the two
 *  pictures were and not only that they agreed. A digest, not a dump: the
 *  images cannot be written out of the container (the mirror is mounted
 *  read-only), and "identical" is only half a diagnosis. */
function imageDigest(img) {
  let h = 2166136261;
  for (let i = 0; i < img.data.length; i += 4) {
    h = Math.imul(h ^ (img.data[i] + img.data[i + 1] * 3 + img.data[i + 2] * 7), 16777619);
  }
  const at = (x, y) => {
    const k = (y * img.width + x) * 4;
    return `${img.data[k]},${img.data[k + 1]},${img.data[k + 2]}`;
  };
  return `${img.width}x${img.height} lit ${litPixels(img)} centre rgb(${at(img.width >> 1, img.height >> 1)})`
    + ` corner rgb(${at(0, 0)}) hash ${(h >>> 0).toString(16)}`;
}

/** Pixels clear of the page's background by more than the lit threshold. */
function litPixels(img) {
  let n = 0;
  for (let i = 0; i < img.width * img.height; i += 1) {
    const o = i * 4;
    if (distanceFromBackground(img.data[o], img.data[o + 1], img.data[o + 2]) > LIT_DISTANCE) n += 1;
  }
  return n;
}

/** A legend line's position in metres, read through the page's own prefixes:
 *  {metres, tokens, prefix}, or {error} naming what the line does not state. */
function legendPosition(text) {
  const m = /^■ probe (\d+): E(.+) at \(([^()]*)\)\s*(\S*)m$/.exec(text);
  if (!m) return { error: `${JSON.stringify(text)} states no "probe N: Ec at (x, y, z) <unit>m"` };
  const factor = FACTOR.get(m[4]);
  if (factor === undefined) return { error: `${JSON.stringify(m[4] + 'm')} is not a unit the page writes` };
  const tokens = m[3].split(',').map((t) => t.trim());
  if (tokens.length !== 3 || tokens.some((t) => !/^-?\d+(\.\d+)?$/.test(t))) {
    return { error: `${JSON.stringify(m[3])} is not three plain numbers` };
  }
  return { metres: tokens.map((t) => Number(t) * factor), tokens, prefix: m[4] };
}

// --- moving the page without waiting on an animation frame ------------------
// Setting the slider and firing its event in one evaluate is what the page's
// own input listener does; Playwright's fill() would first wait for the
// element to be stable across animation frames, which a loaded movie starves.
const setFrame = (page, i) => page.evaluate((n) => {
  const s = document.getElementById('td-movie-frame');
  s.value = String(n);
  s.dispatchEvent(new Event('input', { bubbles: true }));
}, i);

/** Wait until a number read from the page gets to `want`, FAILing only if it
 *  STANDS STILL for stallMs() -- not if the run as a whole is slow.
 *
 *  Why this shape: the page is a software-rendered point cloud (no GPU in this
 *  container), driven by requestAnimationFrame, and it is measured drawing on
 *  the order of a second per frame on the open-boundary mirror while this VM is
 *  shared with every other rig. A wall-clock budget on "the movie has played
 *  three frames" is therefore a budget on how busy the machine is: this check's
 *  first open-boundary run was reported as "the check never finishes on an open
 *  mirror" (review row 2026-10-03), while a run of the same mirror with the
 *  wait taken out showed the page playing frame 0 -> 1 -> 3 in three seconds
 *  with no page errors -- the page was alive and the constant was wrong. The
 *  same mirror, at the same point, passes this wait where that one aborted.
 *
 *  Progress, not elapsed time, is what a stalled page cannot fake: the counter
 *  has to move before the deadline counts again, and a page that stops is
 *  FAILed by name with the value it stopped at and any error it threw.
 *  stallMs() is derived from the page's own slowest measured frame rather than
 *  tuned to one fixture: 20 frames of slack, floor 90 s. */
let frameCostMs = 0;                          // the slowest drawn frame yet seen
const stallMs = () => Math.max(90000, 20 * frameCostMs);

async function waitAdvance(page, read, from, want, what, errors) {
  let last = from, movedAt = Date.now();
  for (;;) {
    let now;
    try {
      now = await page.evaluate(read);
    } catch (e) {
      ok(false, `the page stopped answering while waiting for ${what}: ${String(e.message).split('\n')[0]}`);
      return null;
    }
    if (Number(now) > last) { last = Number(now); movedAt = Date.now(); }
    if (last >= want) return last;
    if (Date.now() - movedAt > stallMs()) {
      const errs = errors && errors.length ? `; page errors: ${JSON.stringify(errors)}` : '';
      ok(false, `${what} stood still for ${Math.round(stallMs() / 1000)} s at ${last} of ${want}${errs}`);
      return null;
    }
    await new Promise((r) => setTimeout(r, 250));
  }
}

/** The movie canvas after the page's own loop has drawn one further frame.
 *  Counted through requestAnimationFrame (the loop renders in the same
 *  callback that advances the count), then read with toDataURL: with
 *  preserveDrawingBuffer forced from the page's first context, that returns
 *  the frame the loop last drew, and needs no animation frame of its own.
 *  Null (and a recorded FAIL) if no frame arrives, so a stalled loop is named
 *  instead of looking like two identical pictures.
 *
 *  Read the page that is on top. A backgrounded page has its animation frames
 *  throttled, so a shot of one is a shot of whatever it last drew: an older
 *  canvas, indistinguishable from "the thing is not drawn". Measured on this
 *  check: with the outline page left in the background while a second page was
 *  open, one read came back a canvas from before the last frame change and the
 *  two runs were identical on that frame. The outline samples below are taken
 *  one page at a time for the same reason; this keeps a read honest whichever
 *  page opened last. */
async function drawnShot(page, what, errors) {
  await page.bringToFront();
  const n = await page.evaluate(() => window.__tdFrames || 0);
  const t0 = Date.now();
  const got = await waitAdvance(page, () => window.__tdFrames || 0, n, n + 1,
                               `the render loop to draw ${what}`, errors);
  if (got === null) return null;
  // This frame's cost is the page's own, not the harness's assumption: the next
  // wait's floor scales with it.
  frameCostMs = Math.max(frameCostMs, Date.now() - t0);
  return page.evaluate(() => {
    const c = document.getElementById('td-scene');
    return { w: c.width, h: c.height, url: c.toDataURL('image/png') };
  });
}

async function open(browser, { caveats = null, traceStatus = 200, frameStatus = 200, noMovie = false,
                               zeroFrames = false, noPositions = false,
                               outlineStatus = 200 } = {}) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  await page.addInitScript(() => {
    // The movie canvas keeps its last drawn frame, so this check can read it
    // without racing the page's loop, and the loop's own frames are counted so
    // the wait for one needs no requestAnimationFrame of its own.
    const getContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function (type, attrs) {
      if (String(type).startsWith('webgl')) attrs = { ...(attrs || {}), preserveDrawingBuffer: true };
      return getContext.call(this, type, attrs);
    };
    const raf = window.requestAnimationFrame.bind(window);
    window.requestAnimationFrame = (cb) => raf((t) => {
      window.__tdFrames = (window.__tdFrames || 0) + 1;
      return cb(t);
    });
  });
  await page.route('**/*', (route) => {
    const p = new URL(route.request().url()).pathname;
    if (p.startsWith('/static/')) {
      const f = path.join(DIR, p);
      return route.fulfill({ body: fs.readFileSync(f),
                             contentType: TYPES[path.extname(f)] || 'application/octet-stream' });
    }
    if (p.endsWith('/status')) {
      return route.fulfill({ body: fs.readFileSync(path.join(DIR, 'status.json')),
                             contentType: 'application/json' });
    }
    if (p.endsWith('/files/spectrum.json')) {
      let m = caveats ? { ...manifest, caveats } : manifest;
      if (noMovie) m = { ...m, frames: null };
      // frames load, then the renderer throws (movieLattice reads positions_m)
      if (noPositions) { const { positions_m, ...fr } = m.frames; m = { ...m, frames: fr }; }
      return route.fulfill({ body: JSON.stringify(m), contentType: 'application/json' });
    }
    if (p.endsWith('/files/trace.bin')) {
      if (traceStatus !== 200) return route.fulfill({ status: traceStatus, body: 'storage down' });
      return route.fulfill({ body: fs.readFileSync(path.join(DIR, 'files/trace.bin')),
                             contentType: 'application/octet-stream' });
    }
    // The object outline the page draws over the field. The 404 arm is the
    // control the outline checks below are measured against: with it, the only
    // difference between two runs of the same page is whether the outline was
    // drawn at all.
    if (p.endsWith('/outline.glb')) {
      if (outlineStatus !== 200) return route.fulfill({ status: outlineStatus, body: 'no outline' });
      return route.fulfill({ body: fs.readFileSync(path.join(DIR, 'outline.glb')),
                             contentType: 'model/gltf-binary' });
    }
    const fm = p.match(/\/files\/(frame_\d{2,3}\.bin)$/);
    if (fm) {
      if (frameStatus !== 200 && fm[1] === frameName(7)) return route.fulfill({ status: frameStatus, body: 'x' });
      const onDisk = path.join(DIR, 'files', fm[1]);
      // A page that asks for a name the manifest did not write gets the 404 the
      // real route gives it, so a wrong name shows up as a FAIL naming the
      // page's own error, not as a crash in this check.
      if (!fs.existsSync(onDisk)) return route.fulfill({ status: 404, body: 'no such frame' });
      const bytes = fs.readFileSync(onDisk);
      return route.fulfill({ body: zeroFrames ? Buffer.alloc(bytes.length) : bytes,
                             contentType: 'application/octet-stream' });
    }
    return route.fulfill({ body: fs.readFileSync(path.join(DIR, 'job.html')), contentType: 'text/html' });
  });
  await page.goto('http://mse.check/jobs/x');
  await page.waitForFunction(() => {
    const c = document.getElementById('td-caption');
    return c && c.textContent && !c.textContent.startsWith('Loading');
  }, null, { polling: 100, timeout: 30000 }).catch(async () => {
    // A wait that times out is a FAIL that names what the page showed and threw,
    // never a bare abort that ends the run without a verdict. (This check's
    // first open-boundary run ended exactly that way, and the review row had to
    // ask "page or harness?" -- review row 2026-10-03.)
    const cap = await page.evaluate(() => (document.getElementById('td-caption') || {}).textContent);
    ok(false, `the field caption is still ${JSON.stringify(cap)} after 30 s; page errors: ${JSON.stringify(errors)}`);
    await browser.close();
    console.log(`${fails.length} FAILED`);
    process.exit(1);
  });
  if (traceStatus === 200) {
    await page.waitForFunction(() => {
      const c = document.getElementById('td-movie-caption');
      return c && c.textContent && !c.textContent.startsWith('Loading');
    }, null, { polling: 100, timeout: 60000 }).catch(async () => {
      const cap = await page.evaluate(() => (document.getElementById('td-movie-caption') || {}).textContent);
      ok(false, `movie caption still ${JSON.stringify(cap)} after 60 s; page errors: ${JSON.stringify(errors)}`);
      await browser.close();
      console.log(`${fails.length} FAILED`);
      process.exit(1);
    });
  }
  return { page, errors };
}

const browser = await chromium.launch();

// --- as written ------------------------------------------------------------
{
  const { page, errors } = await open(browser);
  ok(errors.length === 0, `no page errors (${errors.join('; ')})`);
  const s = await page.evaluate(() => {
    const vis = (id) => { const e = document.getElementById(id); return !!e && !e.hidden; };
    const lines = (id) => [...document.querySelectorAll(`#${id} polyline`)]
      .map((l) => l.getAttribute('points').trim().split(/\s+/).map((pt) => pt.split(',').map(Number)));
    const texts = (id) => [...document.querySelectorAll(`#${id} text`)].map((t) => t.textContent);
    return { td: vis('td-card'), viewer: vis('viewer'), modes: vis('modes-card'), bands: vis('bands-card'),
             caption: document.getElementById('td-caption').textContent,
             legend: [...document.querySelectorAll('#td-legend li')].map((l) => l.textContent),
             caveats: document.querySelectorAll('#td-caveats li').length,
             trace: lines('td-trace-svg'), spec: lines('td-spectrum-svg'),
             traceText: texts('td-trace-svg'), specText: texts('td-spectrum-svg'),
             swatch: [...document.querySelectorAll('#td-legend .td-swatch')].map((e) => getComputedStyle(e).color),
             stroke: [...document.querySelectorAll('#td-trace-svg polyline')].map((e) => {
               const probe = document.createElement('span'); probe.style.color = e.getAttribute('stroke');
               document.body.append(probe); const c = getComputedStyle(probe).color; probe.remove(); return c; }) };
  });
  ok(s.td, 'the td card is shown');
  ok(!s.viewer && !s.modes && !s.bands, 'the cavity viewer, modes and band cards stay hidden');
  ok(s.caption.includes(`${NP} probe(s), ${manifest.steps} time steps`)
     && s.caption.includes(`over ${(manifest.steps * manifest.dt_s / spanUnit.factor).toPrecision(3)} ${spanUnit.prefix}s`)
     && s.caption.includes(`spectrum from ${(manifest.band_hz[0] / bandUnit.factor).toPrecision(3)}`)
     && s.caption.includes(`${(manifest.band_hz[1] / bandUnit.factor).toPrecision(3)} ${bandUnit.prefix}Hz.`),
     `caption names the probes, the steps and this run's span and band: ${s.caption}`);
  const wantLegend = PROBES.map((p, i) => `■ probe ${i + 1}: E${p.component}`);
  ok(s.legend.length === NP
     && s.legend.every((t, i) => t.startsWith(wantLegend[i]))
     && new Set(wantLegend).size === wantLegend.length,
     `legend per probe: ${JSON.stringify(s.legend)}`);
  // The legend's position is the probe's own node: worker/td_solve.py stores
  // origin_m + the coordinate of the grid node the probe sits on, which is what
  // the manifest's probes[].position_m holds, so the two must agree on every
  // axis. A node AT the frame's own origin arrives as the float64 remainder of
  // two grid-sized numbers -- on this job ~1e-19 m, a *millionth of a cell or
  // less* -- and printing that as a position rather than as the origin is what
  // put "0.0000400 fm" on prod job aa6f083b, twelve orders below the cell.
  const step = fineStep(FR && FR.positions_m);
  const stated = s.legend.map(legendPosition);
  ok(stated.length === NP && stated.every((p) => p.metres),
     `every legend states its position as three numbers in a unit the page writes: ` +
     JSON.stringify(stated.map((p) => p.error || p.tokens.join(', '))));
  if (stated.every((p) => p.metres)) {
    // The page prints three significant digits, so a coordinate is its node to
    // within half a percent of itself -- but never further than one cell, which
    // is the whole of what a node's coordinate means on this grid.
    const okAxis = (v, got) => Math.abs(got - v) <= Math.max(step || 0, Math.abs(v) * 5e-3);
    ok(PROBES.every((p, i) => p.position_m.every((v, a) => okAxis(v, stated[i].metres[a]))),
       `every legend position is the manifest's own node, within ` +
       `${step === null ? 'the printed three digits' : `one cell (${step} m)`}: ` +
       `${JSON.stringify(stated.map((p) => p.tokens.join(', ')))} vs ` +
       `${JSON.stringify(PROBES.map((p) => p.position_m))}`);
    if (step === null) {
      console.log('      the manifest lists no sample positions: the origin rule is not checked here');
    } else {
      const origin = [];
      stated.forEach((p, i) => PROBES[i].position_m.forEach((v, a) => {
        if (Math.abs(v) < 1e-6 * step) origin.push(`probe ${i + 1} ${'xyz'[a]}=${v} -> ${p.tokens[a]}`);
      }));
      ok(origin.every((c) => c.endsWith(' -> 0')),
         `a coordinate below a millionth of the grid's own cell (${step} m) reads as 0, ` +
         `and a probe at the origin is a case of it: ${JSON.stringify(origin)}`);
    }
  }
  ok(s.caveats === 0, 'no caveat lines when the run has none');
  ok(s.trace.length === NP && s.spec.length === NP,
     `one trace and one spectrum line per probe (${NP}): ${s.trace.length} and ${s.spec.length}`);
  const finite = (ls) => ls.every((l) => l.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
  ok(finite(s.trace) && finite(s.spec), 'every plotted point is finite');
  // A flat line is the "probe saw nothing" picture; the real trace rings.
  const ySpan = (l) => Math.max(...l.map((p) => p[1])) - Math.min(...l.map((p) => p[1]));
  ok(s.trace.every((l) => ySpan(l) > 20), `each trace has visible swing (${s.trace.map(ySpan).map(Math.round)})`);
  ok(s.spec.some((l) => ySpan(l) > 20), 'a spectrum has a visible peak');
  ok(s.swatch.length === NP && JSON.stringify(s.swatch) === JSON.stringify(s.stroke)
     && new Set(s.swatch).size === NP,
     `one legend swatch per probe, in its line's colour: ${s.swatch} vs ${s.stroke}`);
  const ticks = [...s.traceText, ...s.specText].filter((t) => /^-?[\d.]/.test(t));
  ok(ticks.length > 8 && ticks.every((t) => !/e/i.test(t)), `no e-notation tick labels: ${ticks}`);
  // The factor must be the one these values need: the page reads it from the
  // trace it was served, so a factor from any other run fails here.
  const yTitle = s.traceText.find((t) => t.startsWith(traceTitle)) || '';
  ok(traceTitles.includes(yTitle),
     `trace y title states the factor its own values need (peak ${peak}): ` +
     `${traceTitles.join(' or ')} vs ${JSON.stringify(yTitle)}`);
  ok(s.traceText.includes(`time (${axisUnit.prefix}s)`),
     `time axis in ${axisUnit.prefix}s, this run's span: ${s.traceText.filter((t) => t.startsWith('time'))}`);
  ok(s.specText.includes(`spectral amplitude (${SPECTRUM_SCALE})`),
     `spectrum y title states the scale: ${s.specText.filter((t) => t.includes('spectral'))}`);
  ok(Math.max(...PROBES.flatMap((p) => p.spectrum.amplitude)) === 1, 'the manifest spectra peak at 1');
  ok(s.specText.includes(`frequency (${bandUnit.prefix}Hz)`),
     `frequency axis in ${bandUnit.prefix}Hz, this run's band: ${s.specText.filter((t) => t.startsWith('freq'))}`);
  await page.close();
}

// --- the field movie --------------------------------------------------------
let zeroShot = null;
if (!FR || !FR.count) {
  // A job can hold no movie at all. The page must say so, and the movie cases
  // below (which need frames) are then not applicable to it: scripts/
  // run_tdcheck.sh's fixture covers them.
  const { page } = await open(browser);
  const c = await page.textContent('#td-movie-caption');
  ok(c === 'This run recorded no field movie.', `a job with no movie says so: ${c}`);
  await page.close();
  console.log('      this job recorded no movie: the movie cases are not applicable here');
} else {
  const { page, errors } = await open(browser);
  ok(errors.length === 0, `movie: no page errors (${errors.join('; ')})`);
  const st = await page.evaluate(() => ({
    shown: !document.getElementById('td-movie').hidden,
    caption: document.getElementById('td-movie-caption').textContent,
    max: document.getElementById('td-movie-frame').max,
    frame: document.getElementById('td-movie').dataset.frame,
    outline: document.getElementById('td-movie').dataset.outline }));
  ok(st.shown, 'the movie panel is shown');
  ok(st.max === String(FR.count - 1), `slider spans every frame: max ${st.max}`);
  // --- the object outline sits on the object, inside the field -------------
  // The owner's ask: the uploaded object drawn over the field. Two comparisons,
  // because they catch different things and neither alone is enough:
  //
  //   1. the box the PAGE states it drew, against the box the outline FILE's
  //      own vertices span (decoded from the mirror's outline.glb -- the very
  //      bytes the page fetched). The page built its box by walking the file it
  //      loaded, so a viewer that translated, rescaled or dropped a part
  //      disagrees here, and nothing server-side can hide it.
  //   2. the outline file's box, inside the box the MANIFEST's sample positions
  //      span. Containment, not equality, and deliberately so: the samples are
  //      where the field was computed, which on the default open boundary is
  //      the object plus OPEN_PAD_CELLS*dx of padding on every side (16 cells
  //      today; fdtd_layout.py:204), and on metal is the object's own box. The
  //      manifest records no boundary, so this check cannot tell which of the
  //      two it is holding and must not assume either -- an earlier version of
  //      this check compared the drawn box to the samples directly and so
  //      FAILED a correct open job (the review row of 2026-10-03). Containment
  //      is the one relation that is true on both, and it still catches a unit,
  //      a scale or a whole-frame translation: those put the object outside the
  //      field, or the field's 1000th.
  //
  // One cell of slack is the grid's own step, read from the manifest's
  // positions: the outermost samples need not land exactly on the upload's
  // face. A real displacement is a large fraction of the object, not a cell.
  //
  // WHY THE FIXTURE'S UPLOAD IS TWO PARTS, the second at negative x: a single
  // block starting at its own origin has the same numbers in the object's frame
  // and in the grid's (the offset is zero), so comparing the two was an
  // identity that could not fail -- the trap this row names. With the offset
  // part, the grid's frame is 2 mm from the object's and the object's box is
  // smaller than the sampled box, so an outline left in the mesh's frame, an
  // outline centred on the domain instead of on its own coordinates, and an
  // outline scaled by the wrong unit all land outside one cell -- and neither
  // comparison above is an identity.
  const wantBox = positionsBox(FR.positions_m);
  const drewBox = outlineBox(st.outline);
  const file = glbBox(fs.readFileSync(path.join(DIR, 'outline.glb')));
  ok(!file.why, `the mirror's outline.glb states the object's own box (${file.why || fmtBox(file.box)})`);
  ok(drewBox, `the movie says where it drew the outline (${JSON.stringify(st.outline)})`);
  const cell = fineStep(FR.positions_m);
  if (drewBox && file.box) {
    const drew = [0, 1, 2].map((a) => [drewBox.min[a], drewBox.max[a]]);
    const wrong = [];
    for (let a = 0; a < 3; a += 1) {
      for (const end of [0, 1]) {
        const d = Math.abs(drew[a][end] - file.box[a][end]);
        if (!(d <= cell)) {
          wrong.push(`${'xyz'[a]} ${end ? 'max' : 'min'} ${drew[a][end]} vs ${file.box[a][end]} `
                     + `(${d} > one cell ${cell})`);
        }
      }
    }
    ok(wrong.length === 0,
       'the page drew the outline where the outline file puts the object, within '
       + `one cell (${cell} m): page ${fmtBox(drew)} against the file ${fmtBox(file.box)}`
       + (wrong.length ? ` -- ${wrong.join('; ')}` : ''));
  }
  if (file.box && wantBox) {
    const outside = [];
    for (let a = 0; a < 3; a += 1) {
      if (!(file.box[a][0] >= wantBox[a][0] - cell) || !(file.box[a][1] <= wantBox[a][1] + cell)) {
        outside.push(`${'xyz'[a]} ${file.box[a][0]}..${file.box[a][1]} not inside `
                     + `${wantBox[a][0]}..${wantBox[a][1]}`);
      }
    }
    ok(outside.length === 0,
       'the object sits inside the field the manifest sampled, within one cell '
       + `(${cell} m): outline ${fmtBox(file.box)} against the samples ${fmtBox(wantBox)}`
       + ' (the samples carry the open boundary\'s pad cells past the object; on metal they '
       + 'are the object)'
       + (outside.length ? ` -- ${outside.join('; ')}` : ''));
  }
  // The frames on disk must be exactly the ones the manifest names: a mirror
  // with a hole in it would otherwise reach the page as a 404 the page names,
  // reading as a page defect rather than as an incomplete mirror.
  const framesOnDisk = fs.readdirSync(path.join(DIR, 'files'))
    .filter((n) => /^frame_.*\.bin$/.test(n)).sort();
  ok(framesOnDisk.length === FR.count && framesOnDisk.every((n, i) => n === frameName(i + 1)),
     `the inputs hold every frame the manifest names (${framesOnDisk.length}/${FR.count}, `
     + `${frameName(1)}..${frameName(FR.count)})`);
  const head = (n) => `|E|, frame ${n} of ${FR.count}, t = `;
  const wantTime = (i) => `t = ${(FR.steps[i] * manifest.dt_s / spanUnit.factor).toPrecision(3)} `
    + `${spanUnit.prefix}s of the run.`;
  const spacing = (() => {
    if (!(spacingS > 0)) return 'this run has one frame, so it states no spacing';
    const u = siPrefix(spacingS);
    let text = `Frames are ${(spacingS / u.factor).toPrecision(3)} ${u.prefix}s apart`;
    if (!manifest.source_frequency_hz) return text + ', and this job\'s manifest states no source frequency';
    const f = siPrefix(manifest.source_frequency_hz);
    return text + ` (${cycles.toFixed(2)} cycles of the `
      + `${(manifest.source_frequency_hz / f.factor).toPrecision(3)} ${f.prefix}Hz source)`;
  })();
  ok(st.caption.startsWith(head(1)) && st.caption.includes(spacing)
     && st.caption.includes(`One colour scale for every frame `
                            + `(the ${Math.round(SCALE_PERCENTILE * 100)}th percentile`)
     && st.caption.includes(`below ${MOVIE_THRESHOLD} of it is not drawn`),
     `movie caption: ${st.caption}`);
  // The caption's honest verdict, checked against the manifest's own numbers:
  // more than half a cycle between frames must be said to be the envelope, and
  // a spacing under that must NOT claim it.
  if (cycles !== null) {
    const says = st.caption.includes('shows the envelope, not the oscillation');
    ok((cycles > 0.5) === says,
       `the envelope verdict follows the spacing (${cycles} cycles/frame): ${st.caption}`);
  }
  // The manifest's own frame times, which the spacing above is read from, must
  // agree with its steps; otherwise the page is being checked against a
  // manifest that contradicts itself.
  const last = FR.count - 1;
  ok(Math.abs(frameTimes[last] - FR.steps[last] * manifest.dt_s) / (frameTimes[last] || 1) < 0.01,
     `the manifest's recorded frame times agree with its steps (${frameTimes[last]} vs `
     + `${FR.steps[last] * manifest.dt_s})`);
  // Frame i sits at its own recorded step, read from the manifest.
  const mid = Math.floor(FR.count / 2);   // the slider is 0-based
  await setFrame(page, mid);
  const capMid = await page.textContent('#td-movie-caption');
  ok(capMid.startsWith(head(mid + 1)) && capMid.includes(wantTime(mid)),
     `frame ${mid + 1} caption at its step's time (${FR.steps[mid] * manifest.dt_s}): ${capMid}`);
  // The last frame is at the window's end, and the caption says where that is.
  await setFrame(page, last);
  const capLast = await page.textContent('#td-movie-caption');
  ok(capLast.startsWith(head(last + 1)) && capLast.includes(wantTime(last)),
     `the last frame is at the window's end (${frameTimes[last]}): ${capLast}`);
  await setFrame(page, mid);
  const midShot = await drawnShot(page, `frame ${mid + 1}`, errors);
  await setFrame(page, 0);
  const firstShot = await drawnShot(page, 'frame 1', errors);
  if (midShot && firstShot) {
    ok(midShot.w > 8 && midShot.h > 8,
       `the movie canvas is drawn at a real size (${midShot.w}x${midShot.h})`);
    ok(midShot.url !== firstShot.url, 'two frames draw two different pictures');
  }
  await page.evaluate(() => document.getElementById('td-movie-play').click());
  // Two more frames than the page is on now, read at the click: the play is
  // asserted as MOVEMENT, so it cannot pass on a page that was already past 3.
  const playFrom = Number(await page.evaluate(() => document.getElementById('td-movie').dataset.frame));
  const reached = await waitAdvance(page, () => Number(document.getElementById('td-movie').dataset.frame),
                                    playFrom, playFrom + 2, 'the movie to play on', errors);
  if (reached !== null) {
    ok(await page.textContent('#td-movie-play') === 'Pause', 'play advances the frames and offers Pause');
  }
  // Stop the play where it got to: the frame it reached is one of the frames
  // the outline is checked on below, because "while the movie plays" is what
  // the owner asked for. The stop is asserted, not assumed -- a timer still
  // running would move the frame under the pixel read and make it a race.
  await page.evaluate(() => document.getElementById('td-movie-play').click());
  ok(await page.textContent('#td-movie-play') === 'Play', 'the second click stops the play');
  const playedFrame = Number(await page.evaluate(
    () => document.getElementById('td-movie').dataset.frame));
  // --- the outline is DRAWN on the field, on every frame it is asked for ----
  // The same page is opened again with the outline route answering 404: same
  // manifest, same code, same browser, deterministic drawing, so the pixels the
  // two runs disagree on are the outline's and nothing else. Both halves are
  // needed. The control alone would be satisfied by a field that merely stopped
  // being drawn; the outline run alone says nothing about what the pixels
  // changed from. Play while the movie plays too, not only while scrubbing.
  //
  // THE TWO RUNS ARE SERIAL, one page open at a time. Measured: with both pages
  // open at once, the page in the background had its animation frames throttled
  // -- one read came back a canvas from before the last move, and the two runs
  // were identical on that frame -- which reads as a FAIL that says nothing
  // about the page. One page at a time leaves every read with the browser to
  // itself. The frames played through are the same either way, so the runs are
  // still the same picture with and without the outline.
  const sampleFrames = [[0, 'the first frame'], [mid, `frame ${mid + 1} of ${FR.count}`],
                        [last, 'the last frame'],
                        [playedFrame, `the frame playing reached (${playedFrame + 1} of ${FR.count})`]];
  const withIt = [];
  for (const [i, what] of sampleFrames) {
    await setFrame(page, i);
    const shot = await drawnShot(page, what, errors);
    // What the page says about itself at the moment of its own read: a canvas
    // that disagrees with the page's stated frame is a rendering staleness, not
    // an outline that was never drawn, and the two want different fixes.
    const said = await page.evaluate(() => {
      const p = document.getElementById('td-movie');
      return `frame ${Number(p.dataset.frame) + 1}, outline ${p.dataset.outline === 'none'
        ? 'none' : p.dataset.outline ? 'stated' : 'unstated'}`;
    });
    withIt.push({ shot, said });
  }
  await page.close();
  const off = await open(browser, { outlineStatus: 404 });
  const offState = await off.page.evaluate(() => ({
    outline: document.getElementById('td-movie').dataset.outline,
    caption: document.getElementById('td-movie-caption').textContent }));
  ok(offState.outline === 'none' && /outline could not be drawn/.test(offState.caption),
     'the control run names the outline it did not draw, so a pixel difference below is the '
     + `outline's: ${offState.outline} -- ${offState.caption}`);
  const withoutIt = [];
  for (const [i, what] of sampleFrames) {
    await setFrame(off.page, i);
    withoutIt.push(await drawnShot(off.page, `${what}, with no outline to draw`, off.errors));
  }
  await off.page.close();
  for (let k = 0; k < sampleFrames.length; k += 1) {
    const [, what] = sampleFrames[k];
    if (!withIt[k].shot || !withoutIt[k]) continue;   // drawnShot named the failure
    const [a, b] = [await shotPixels(withIt[k].shot), await shotPixels(withoutIt[k])];
    const fieldPixels = litPixels(b);
    ok(fieldPixels > 0, `the control run still draws the field on ${what} (${fieldPixels} lit px), `
       + 'so what changes below is the outline and not a frame that went blank');
    const changed = differingPixels(a, b);
    const need = Math.max(1, Math.ceil(OUTLINE_SHARE * fieldPixels));
    if (!(changed >= need)) {
      // Only on the way to a FAIL: what the two pictures actually were, so the
      // next reader is not left with "identical" and a question. In a passing
      // run these lines are noise.
      console.log(`      with the outline: ${imageDigest(a)}`);
      console.log(`      without it:       ${imageDigest(b)}`);
      console.log(`      the page said:    ${withIt[k].said}`);
    }
    ok(changed >= need, `the outline is drawn on ${what}: ${changed} pixels of that frame differ `
       + `when no outline is drawn (of ${a.width}x${a.height}; ${fieldPixels} lit px in the `
       + `control, ${need} needed)`);
  }
  const z = await open(browser, { zeroFrames: true });
  const zc = await z.page.textContent('#td-movie-caption');
  ok(zc.endsWith('Every frame is zero: there is no field to draw.'), `an all-zero movie says so: ${zc}`);
  await setFrame(z.page, mid);
  zeroShot = await drawnShot(z.page, 'the all-zero frame', z.errors);
  ok(zeroShot && midShot && zeroShot.url !== midShot.url,
     `the real frame ${mid + 1} draws field the all-zero movie does not`);
  await z.page.close();
}
if (FR && FR.count) {
  const { page } = await open(browser, { frameStatus: 502 });
  const c = await page.textContent('#td-movie-caption');
  ok(c === 'The field movie could not be loaded: frame 7 returned HTTP 502.', `a failed frame is named: ${c}`);
  await page.close();
}
if (FR && FR.count) {
  const { page } = await open(browser, { noMovie: true });
  const c = await page.textContent('#td-movie-caption');
  ok(c === 'This run recorded no field movie.', `a run without a movie says so: ${c}`);
  await page.close();
}
if (FR && FR.count) {
  const { page } = await open(browser, { noPositions: true });
  const c = await page.textContent('#td-movie-caption');
  ok(c.startsWith('The field movie could not be shown: '), `a throw after loading is captioned: ${c}`);
  await page.close();
}

// --- caveats reach the page -------------------------------------------------
{
  const { page } = await open(browser, { caveats: CAVEATS });
  const got = await page.$$eval('#td-caveats li', (ls) => ls.map((l) => l.textContent));
  ok(JSON.stringify(got) === JSON.stringify(CAVEATS), `caveats listed: ${JSON.stringify(got)}`);
  await page.close();
}

// --- a failed trace is named, not an empty card -------------------------------
{
  const { page } = await open(browser, { traceStatus: 502 });
  const caption = await page.textContent('#td-caption');
  ok(/could not be loaded: the trace returned HTTP 502/.test(caption), `failure named: ${caption}`);
  await page.close();
}

await browser.close();
console.log(fails.length ? `${fails.length} FAILED` : 'ALL PASS');
process.exit(fails.length ? 1 : 0);
