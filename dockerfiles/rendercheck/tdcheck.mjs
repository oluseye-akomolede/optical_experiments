// In-browser check of the time-domain result card on the REAL job page
// (job.html from the Flask app, viewer.js + tdplot.js + tdmovie.js from
// web/static) fed the files a real solve wrote: the trace and spectrum plots
// and the field movie, as written and in each failure case (caveats, trace
// 502, a frame 502, no movie, an all-zero movie, a throw after every frame has
// loaded). Nothing is fetched from the network.
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

/** The movie canvas after the page's own loop has drawn one further frame.
 *  Counted through requestAnimationFrame (the loop renders in the same
 *  callback that advances the count), then read with toDataURL: with
 *  preserveDrawingBuffer forced from the page's first context, that returns
 *  the frame the loop last drew, and needs no animation frame of its own.
 *  Null (and a recorded FAIL) if no frame arrives, so a stalled loop is named
 *  instead of looking like two identical pictures. */
async function drawnShot(page, what) {
  const n = await page.evaluate(() => window.__tdFrames || 0);
  try {
    await page.waitForFunction((k) => (window.__tdFrames || 0) > k, n,
                               { polling: 100, timeout: 90000 });
  } catch {
    ok(false, `the page's render loop drew no frame for ${what} in 90 s`);
    return null;
  }
  return page.evaluate(() => {
    const c = document.getElementById('td-scene');
    return { w: c.width, h: c.height, url: c.toDataURL('image/png') };
  });
}

async function open(browser, { caveats = null, traceStatus = 200, frameStatus = 200, noMovie = false,
                               zeroFrames = false, noPositions = false } = {}) {
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
  }, null, { polling: 100, timeout: 30000 });
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
    frame: document.getElementById('td-movie').dataset.frame }));
  ok(st.shown, 'the movie panel is shown');
  ok(st.max === String(FR.count - 1), `slider spans every frame: max ${st.max}`);
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
  const midShot = await drawnShot(page, `frame ${mid + 1}`);
  await setFrame(page, 0);
  const firstShot = await drawnShot(page, 'frame 1');
  if (midShot && firstShot) {
    ok(midShot.w > 8 && midShot.h > 8,
       `the movie canvas is drawn at a real size (${midShot.w}x${midShot.h})`);
    ok(midShot.url !== firstShot.url, 'two frames draw two different pictures');
  }
  await page.evaluate(() => document.getElementById('td-movie-play').click());
  await page.waitForFunction(() => Number(document.getElementById('td-movie').dataset.frame) >= 3, null,
                             { polling: 100, timeout: 30000 });
  ok(await page.textContent('#td-movie-play') === 'Pause', 'play advances the frames and offers Pause');
  await page.close();
  const z = await open(browser, { zeroFrames: true });
  const zc = await z.page.textContent('#td-movie-caption');
  ok(zc.endsWith('Every frame is zero: there is no field to draw.'), `an all-zero movie says so: ${zc}`);
  await setFrame(z.page, mid);
  zeroShot = await drawnShot(z.page, 'the all-zero frame');
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
