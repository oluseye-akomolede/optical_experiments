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
    const fm = p.match(/\/files\/(frame_\d\d\.bin)$/);
    if (fm) {
      if (frameStatus !== 200 && fm[1] === 'frame_07.bin') return route.fulfill({ status: frameStatus, body: 'x' });
      const bytes = fs.readFileSync(path.join(DIR, 'files', fm[1]));
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
  ok(s.legend.length === manifest.probes.length && s.legend[0].startsWith('■ probe 1: Ez')
     && s.legend[1].startsWith('■ probe 2: Ex'), `legend per probe: ${JSON.stringify(s.legend)}`);
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
  ok(/^\|E\|, frame 1 of 60, t = [\d.]+ (p|n)s\. One colour scale for every frame \(the 99th percentile/.test(st.caption)
     && st.caption.includes('below 0.12 of it is not drawn'), `movie caption: ${st.caption}`);
  // Frame 31 sits at its own recorded step, read from the manifest.
  await page.fill('#td-movie-frame', '30');
  await page.dispatchEvent('#td-movie-frame', 'input');
  const cap31 = await page.textContent('#td-movie-caption');
  const want = manifest.frames.steps[30] * manifest.dt_s;
  const got = Number(cap31.match(/t = ([\d.]+) (p|n)s/)[1]) * (cap31.includes(' ps.') ? 1e-12 : 1e-9);
  ok(cap31.startsWith('|E|, frame 31 of 60') && Math.abs(got - want) / want < 0.01,
     `frame 31 caption at its step's time (${want}): ${cap31}`);
  await settle(page); const mid = await shot(page);
  await page.fill('#td-movie-frame', '0'); await page.dispatchEvent('#td-movie-frame', 'input');
  await settle(page); const first = await shot(page);
  ok(mid !== first, 'two frames draw two different pictures');
  await page.click('#td-movie-play');
  await page.waitForFunction(() => Number(document.getElementById('td-movie').dataset.frame) >= 3, null, { timeout: 10000 });
  ok(await page.textContent('#td-movie-play') === 'Pause', 'play advances the frames and offers Pause');
  await page.close();
  const z = await open(browser, { zeroFrames: true });
  const zc = await z.page.textContent('#td-movie-caption');
  ok(zc.endsWith('Every frame is zero: there is no field to draw.'), `an all-zero movie says so: ${zc}`);
  await z.page.fill('#td-movie-frame', '30'); await z.page.dispatchEvent('#td-movie-frame', 'input');
  await settle(z.page); zeroShot = await shot(z.page);
  ok(zeroShot !== mid, 'the real frame 31 draws field the all-zero movie does not');
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
