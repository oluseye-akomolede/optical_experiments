// In-browser check of the time-domain result card on the REAL job page
// (job.html from the Flask app, viewer.js + tdplot.js + tdmovie.js from
// web/static) fed the files a real tiny solve wrote: the trace and spectrum
// plots and the field movie, as written and in each failure case (caveats,
// trace 502, a frame 502, no movie, an all-zero movie, a throw after every
// frame has loaded). Nothing is fetched from the network.
//
// A movie caption still reading "Loading" after 60 s is recorded as a FAIL
// and ends the run with exit 1: that is the a6127d3 defect (a post-load
// throw left the caption stuck), and the wait is what bounds it.
//
// Inputs: TD_DIR (default /work) as written by scripts/tdcheck_fixture.py.
// scripts/run_tdcheck.sh builds the inputs and runs this, bounded.
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire('/opt/rendercheck/check.mjs');
const { chromium } = require('playwright-core');

const DIR = process.env.TD_DIR || '/work';
const fails = [];
const ok = (c, msg) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) fails.push(msg); };
const TYPES = { '.js': 'application/javascript', '.css': 'text/css', '.json': 'application/json' };
const manifest = JSON.parse(fs.readFileSync(path.join(DIR, 'files/spectrum.json'), 'utf8'));
const CAVEATS = ['thin curved features are resolved only at the coarse cell'];
// Everything this check knows about the movie comes from the manifest it is
// served, not from constants here: the frame count, the name width, and the
// spacing the caption is expected to report. A fixture may be any window of
// any run -- the current one is a 120-frame window, which the old two-digit
// name rule would have checked wrong (and the old page would have failed).
const FR = manifest.frames;
const DIGITS = FR.digits || 2;
const frameName = (n) => `frame_${String(n).padStart(DIGITS, '0')}.bin`;
const UNITS = { P: 1e15, T: 1e12, G: 1e9, M: 1e6, k: 1e3, '': 1, m: 1e-3,
                'µ': 1e-6, u: 1e-6, n: 1e-9, p: 1e-12, f: 1e-15 };
const frameTimes = FR.times_s || FR.steps.map((s) => s * manifest.dt_s);
const spacingS = frameTimes.length > 1
  ? (frameTimes[frameTimes.length - 1] - frameTimes[0]) / (frameTimes.length - 1) : 0;
const cycles = manifest.source_frequency_hz
  ? spacingS * manifest.source_frequency_hz : null;
// The caption's time for frame ``i``, in seconds, or NaN if it does not
// state one in a unit we know.
function captionTime(text) {
  const m = text.match(/t = ([\d.]+) (P|T|G|M|k|m|µ|u|n|p|f)?s /);
  return m ? Number(m[1]) * UNITS[m[2] || ''] : NaN;
}

async function open(browser, { caveats = null, traceStatus = 200, frameStatus = 200, noMovie = false,
                               zeroFrames = false, noPositions = false } = {}) {
  const page = await browser.newPage();
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
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
  }, null, { timeout: 30000 });
  if (traceStatus === 200) {
    await page.waitForFunction(() => {
      const c = document.getElementById('td-movie-caption');
      return c && c.textContent && !c.textContent.startsWith('Loading');
    }, null, { timeout: 60000 }).catch(async () => {
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
  ok(s.caption.includes(`${manifest.probes.length} probe(s), ${manifest.steps} time steps`),
     `caption names probes and steps: ${s.caption}`);
  const wantLegend = manifest.probes.map((p, i) => `■ probe ${i + 1}: E${p.component}`);
  ok(s.legend.length === manifest.probes.length
     && s.legend.every((t, i) => t.startsWith(wantLegend[i]))
     && new Set(wantLegend).size === wantLegend.length,
     `legend per probe: ${JSON.stringify(s.legend)}`);
  ok(s.caveats === 0, 'no caveat lines when the run has none');
  ok(s.trace.length === 2 && s.spec.length === 2, 'one trace and one spectrum line per probe');
  const finite = (ls) => ls.every((l) => l.every(([x, y]) => Number.isFinite(x) && Number.isFinite(y)));
  ok(finite(s.trace) && finite(s.spec), 'every plotted point is finite');
  // A flat line is the "probe saw nothing" picture; the real trace rings.
  const ySpan = (l) => Math.max(...l.map((p) => p[1])) - Math.min(...l.map((p) => p[1]));
  ok(s.trace.every((l) => ySpan(l) > 20), `each trace has visible swing (${s.trace.map(ySpan).map(Math.round)})`);
  ok(s.spec.some((l) => ySpan(l) > 20), 'a spectrum has a visible peak');
  ok(s.swatch.length === 2 && JSON.stringify(s.swatch) === JSON.stringify(s.stroke)
     && s.swatch[0] !== s.swatch[1], `legend swatches are the line colours: ${s.swatch} vs ${s.stroke}`);
  const ticks = [...s.traceText, ...s.specText].filter((t) => /^-?[\d.]/.test(t));
  ok(ticks.length > 8 && ticks.every((t) => !/e/i.test(t)), `no e-notation tick labels: ${ticks}`);
  ok(s.traceText.some((t) => t.endsWith('(arb. units) ×10⁻¹³')), `trace y title carries the factor: ${s.traceText.filter((t) => t.includes('field'))}`);
  ok(s.traceText.some((t) => /^time \((p|n)s\)$/.test(t)), `time axis in ps/ns: ${s.traceText.filter((t) => t.startsWith('time'))}`);
  ok(s.specText.includes('spectral amplitude (largest peak = 1)'), `spectrum y title states the scale: ${s.specText.filter((t) => t.includes('spectral'))}`);
  ok(Math.max(...manifest.probes.flatMap((p) => p.spectrum.amplitude)) === 1, 'the manifest spectra peak at 1');
  ok(s.specText.some((t) => /^frequency \(GHz\)$/.test(t)), `frequency axis in GHz: ${s.specText.filter((t) => t.startsWith('freq'))}`);
  await page.close();
}

// --- the field movie --------------------------------------------------------
const settle = (page) => page.evaluate(() => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r))));
const shot = async (page) => (await page.locator('#td-scene').screenshot()).toString('base64');
let zeroShot = null;
{
  const { page, errors } = await open(browser);
  ok(errors.length === 0, `movie: no page errors (${errors.join('; ')})`);
  const st = await page.evaluate(() => ({
    shown: !document.getElementById('td-movie').hidden,
    caption: document.getElementById('td-movie-caption').textContent,
    max: document.getElementById('td-movie-frame').max,
    frame: document.getElementById('td-movie').dataset.frame }));
  ok(st.shown, 'the movie panel is shown');
  ok(st.max === String(manifest.frames.count - 1), `slider spans every frame: max ${st.max}`);
  ok(manifest.frames.count === 120 && manifest.frames.digits === 3,
     `fixture: a 120-frame, three-digit movie (${manifest.frames.count}, ${manifest.frames.digits})`);
  const head = (n) => `|E|, frame ${n} of ${manifest.frames.count}, t = `;
  ok(st.caption.startsWith(head(1))
     && /Frames are [\d.]+ (P|T|G|M|k|m|µ|u|n|p|f)?s apart \([\d.]+ cycles of the /.test(st.caption)
     && st.caption.includes('One colour scale for every frame (the 99th percentile')
     && st.caption.includes('below 0.12 of it is not drawn'), `movie caption: ${st.caption}`);
  // The caption's honest verdict, checked against the manifest's own numbers:
  // more than half a cycle between frames must be said to be the envelope, and
  // a spacing under that must NOT claim it.
  const says = st.caption.includes('shows the envelope, not the oscillation');
  ok(cycles !== null && ((cycles > 0.5) === says),
     `the envelope verdict follows the spacing (${cycles} cycles/frame): ${st.caption}`);
  // Frame i sits at its own recorded step, read from the manifest.
  const mid = Math.floor(manifest.frames.count / 2);   // the slider is 0-based
  await page.fill('#td-movie-frame', String(mid));
  await page.dispatchEvent('#td-movie-frame', 'input');
  const capMid = await page.textContent('#td-movie-caption');
  const wantMid = manifest.frames.steps[mid] * manifest.dt_s;
  const gotMid = captionTime(capMid);
  ok(capMid.startsWith(head(mid + 1)) && Math.abs(gotMid - wantMid) / wantMid < 0.01,
     `frame ${mid + 1} caption at its step's time (${wantMid}): ${capMid}`);
  // The last frame is at the window's end, and the caption says where that is.
  const last = manifest.frames.count - 1;
  await page.fill('#td-movie-frame', String(last));
  await page.dispatchEvent('#td-movie-frame', 'input');
  const capLast = await page.textContent('#td-movie-caption');
  const wantLast = frameTimes[last];
  const gotLast = captionTime(capLast);
  ok(capLast.startsWith(head(last + 1)) && Math.abs(gotLast - wantLast) / wantLast < 0.01,
     `the last frame is at the window's end (${wantLast}): ${capLast}`);
  await page.fill('#td-movie-frame', String(mid));
  await page.dispatchEvent('#td-movie-frame', 'input');
  await settle(page); const realMidShot = await shot(page);
  await page.fill('#td-movie-frame', '0'); await page.dispatchEvent('#td-movie-frame', 'input');
  await settle(page); const realFirstShot = await shot(page);
  ok(realMidShot !== realFirstShot, 'two frames draw two different pictures');
  await page.click('#td-movie-play');
  await page.waitForFunction(() => Number(document.getElementById('td-movie').dataset.frame) >= 3, null, { timeout: 10000 });
  ok(await page.textContent('#td-movie-play') === 'Pause', 'play advances the frames and offers Pause');
  await page.close();
  const z = await open(browser, { zeroFrames: true });
  const zc = await z.page.textContent('#td-movie-caption');
  ok(zc.endsWith('Every frame is zero: there is no field to draw.'), `an all-zero movie says so: ${zc}`);
  await z.page.fill('#td-movie-frame', String(mid)); await z.page.dispatchEvent('#td-movie-frame', 'input');
  await settle(z.page); zeroShot = await shot(z.page);
  ok(zeroShot !== realMidShot,
     `the real frame ${mid + 1} draws field the all-zero movie does not`);
  await z.page.close();
}
{
  const { page } = await open(browser, { frameStatus: 502 });
  const c = await page.textContent('#td-movie-caption');
  ok(c === 'The field movie could not be loaded: frame 7 returned HTTP 502.', `a failed frame is named: ${c}`);
  await page.close();
}
{
  const { page } = await open(browser, { noMovie: true });
  const c = await page.textContent('#td-movie-caption');
  ok(c === 'This run recorded no field movie.', `a run without a movie says so: ${c}`);
  await page.close();
}

{
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
