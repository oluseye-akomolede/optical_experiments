// In-browser check of the upload form's show/hide script (index.html).
//
// The Python tests read the page's markup; none of them can run its script,
// and the script is what decides which controls are required, disabled and
// shown. So this loads the rendered page in headless Chromium and asks the
// browser itself: is the form valid as the owner would submit it, in each
// mode, and is every control the chosen arm does not use out of the way?
//
// Inputs: FORM_DIR holds index.html (the rendered GET /), style.css and
// probe.obj, three.obj, slow.obj and bad.obj (contents unused; /parts is scripted). Nothing is fetched from the network; every request is answered
// from FORM_DIR. Run it like the other checks, bounded:
//   docker run --rm --cpus=2 --memory=2g -v "$DIR:/work:ro" -e FORM_DIR=/work \
//     --entrypoint node <rendercheck image> /work/formcheck.mjs
import fs from 'node:fs';
import path from 'node:path';
import { createRequire } from 'node:module';
const require = createRequire('/opt/rendercheck/check.mjs');
const { chromium } = require('playwright-core');

const DIR = process.env.FORM_DIR || '/work';
const fails = [];
const ok = (c, msg) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) fails.push(msg); };

const browser = await chromium.launch();
const page = await browser.newPage();
page.on('pageerror', (e) => ok(false, `page error: ${e.message}`));
// /parts is answered by file name, the way the app's parser would: probe.obj
// is one part, three.obj three, bad.obj a refusal; slow.obj is three parts
// answered late, to show a reply for an earlier file is dropped.
const PARTS = { 'probe.obj': [200, { parts: ['probe'] }],
                'three.obj': [200, { parts: ['plate_top', 'film', 'plate_bottom'] }],
                'slow.obj': [200, { parts: ['a', 'b'] }],
                'bad.obj': [400, { error: "part 'lid' is not closed" }] };
await page.route('**/parts', async (route) => {
  const body = route.request().postDataBuffer().toString('latin1');
  const name = Object.keys(PARTS).find((n) => body.includes(`filename="${n}"`));
  if (name === 'slow.obj') await new Promise((r) => setTimeout(r, 800));
  const [status, json] = PARTS[name] || [500, { error: 'unscripted file' }];
  route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(json) });
});
await page.route(/^(?!.*\/parts$).*/, (route) => {
  const url = new URL(route.request().url());
  const file = url.pathname.endsWith('.css') ? 'style.css' : 'index.html';
  route.fulfill({ body: fs.readFileSync(path.join(DIR, file)),
                  contentType: file.endsWith('.css') ? 'text/css' : 'text/html' });
});
await page.goto('http://mse.check/');

// One snapshot of what the browser thinks of a named control.
const state = (name) => page.evaluate((n) => {
  const el = document.querySelector(`[name="${n}"]`);
  return { required: el.required, disabled: el.disabled,
           shown: el.offsetParent !== null };
}, name);
const valid = () => page.evaluate(() => document.getElementById('upload-form').checkValidity());
const label = () => page.textContent('#submit-button');

// --- cavity, as first shown ------------------------------------------------
await page.setInputFiles('input[name="meshfile"]', path.join(DIR, 'probe.obj'));
ok((await state('meshfile')).required, 'cavity: the file is required');
ok(!(await state('scale')).required, 'cavity: the optional scale is not required');
ok(!(await state('detail')).required, 'cavity: the optional feature size is not required');
ok(await valid(), 'cavity: the defaults plus a file are a valid submission');
ok(!(await state('td_probes')).shown && (await state('td_probes')).disabled,
   'cavity: the td arm is hidden and disabled');
ok((await label()) === 'Compute modes', 'cavity: button label');

// --- photonic crystal ------------------------------------------------------
await page.selectOption('#mode-select', 'pc');
ok(await valid(), 'pc: the defaults are a valid submission');
ok((await state('meshfile')).disabled && !(await state('meshfile')).shown,
   'pc: the chosen file is hidden and disabled, so it is not posted');
ok((await state('r_over_a')).required, 'pc: its fields are required');
ok((await label()) === 'Compute the band diagram', 'pc: button label');

// --- time domain: point source, pulse ---------------------------------------
await page.selectOption('#mode-select', 'td');
ok((await state('meshfile')).shown && !(await state('meshfile')).disabled,
   'td: the shared upload is shown');
ok(!(await state('n_index')).shown, 'td: the cavity-only fields are hidden');
const pos = await state('td_position');
ok(pos.shown && pos.required, 'td/point: the source position is shown and required');
const dir = await state('td_direction');
ok(!dir.shown && dir.disabled && !dir.required, 'td/point: the direction is out of the way');
ok((await state('td_bandwidth_thz')).shown, 'td/pulse: the bandwidth is shown');
ok(!(await state('td_samples')).shown, 'td/pulse: the drawn waveform is hidden');
ok(!(await valid()), 'td: empty position and probes block the submission');
await page.fill('[name="td_position"]', '0.5 0.4 0.3');
await page.fill('[name="td_probes"]', '0.5 0.4 0.5 z');
ok(await valid(), 'td: defaults plus position and probes are valid');
ok((await label()) === 'Run the time-domain simulation', 'td: button label');

// --- per-part materials ------------------------------------------------------
const pickers = () => page.$$eval('#td-parts label:not(.td-coating)', (ls) => ls.map((l) => l.textContent));
const coatings = () => page.$$eval('#td-parts label.td-coating', (ls) => ls.map((l) => l.textContent));
const coatingKeys = async () => Object.keys(await posted()).filter((k) => k.startsWith('part_coating'));
const posted = () => page.evaluate(() => {
  const out = {};
  new FormData(document.getElementById('upload-form')).forEach((v, k) => {
    if (typeof v === 'string') out[k] = v;
  });
  return out;
});
const settle = () => page.waitForTimeout(300);
await page.waitForSelector('#td-parts label.td-coating');
ok((await pickers()).length === 0 && (await state('td_n_index')).shown,
   'td/one part: no material pickers; the single index is shown');
ok((await coatings()).length === 1 && (await coatings())[0].includes('probe'),
   `td/one part: one coating picker, named from /parts (${await coatings()})`);
ok((await coatingKeys()).length === 0, 'td/one part: "no coating" posts no coating field');
const coatKind = await page.$('#td-parts label.td-coating select');
await coatKind.selectOption('Au');
let sentCoat = await posted();
ok(sentCoat['part_coating.probe'] === 'Au' && sentCoat['part_coating_thickness.probe'] === '0.01',
   `td/one part: a gold coating posts its material and thickness (${JSON.stringify(sentCoat).slice(0, 300)})`);
ok(await valid(), 'td/one part: a coated part is a valid submission');
await page.fill('[name="part_coating_thickness.probe"]', '');
ok(!(await valid()), 'td/one part: a coating with no thickness blocks the submission');
await coatKind.selectOption('index');
await page.fill('[name="part_coating_thickness.probe"]', '0.02');
await page.fill('#td-parts label.td-coating input[type="number"]:not([name])', '2');
sentCoat = await posted();
ok(sentCoat['part_coating.probe'] === '2' && sentCoat['part_coating_thickness.probe'] === '0.02',
   `td/one part: an index coating posts its index (${JSON.stringify(sentCoat).slice(0, 300)})`);
await coatKind.selectOption('');
ok((await coatingKeys()).length === 0 && await valid(),
   'td/one part: back to "no coating" posts nothing and is valid');
await page.setInputFiles('input[name="meshfile"]', path.join(DIR, 'three.obj'));
await page.waitForSelector('#td-parts label');
ok((await pickers()).length === 3, `td/three parts: three pickers (${(await pickers()).length})`);
ok((await pickers())[1].includes('film'), 'td/three parts: pickers named from /parts');
ok((await coatings()).length === 3, `td/three parts: a coating picker per part (${(await coatings()).length})`);
ok(!(await state('td_n_index')).shown && (await state('td_n_index')).disabled,
   'td/three parts: the single index is hidden and not posted');
const kinds = await page.$$('#td-parts label:not(.td-coating) select');
await kinds[0].selectOption('Au');
await kinds[2].selectOption('pec');
const boxes = await page.$$('#td-parts label:not(.td-coating) input[type="number"]');
ok(!(await boxes[0].evaluate((e) => e.required || e.offsetParent !== null)),
   'td/three parts: a metal part has no index box in the way');
ok(await boxes[1].evaluate((e) => e.required && e.offsetParent !== null),
   'td/three parts: an index part has a required index box');
await boxes[1].fill('2.25');
const sent = await posted();
ok(sent['part_index.plate_top'] === 'Au' && sent['part_index.film'] === '2.25'
   && sent['part_index.plate_bottom'] === 'pec',
   `td/three parts: posts each part's material (${JSON.stringify(sent).slice(0, 200)})`);
ok(!('td_n_index' in sent), 'td/three parts: the single index is not posted');
ok(await valid(), 'td/three parts: a valid submission');
await boxes[1].fill('');
ok(!(await valid()), 'td/three parts: an empty index box blocks the submission');
await page.setInputFiles('input[name="meshfile"]', path.join(DIR, 'probe.obj'));
await settle();
ok((await pickers()).length === 0 && (await state('td_n_index')).shown,
   'td/back to one part: pickers gone, single index back');
ok(!Object.keys(await posted()).some((k) => k.startsWith('part_index.')),
   'td/back to one part: no stale part_index field is posted');
await page.setInputFiles('input[name="meshfile"]', path.join(DIR, 'bad.obj'));
await settle();
ok((await page.textContent('#td-parts-note')).includes("part 'lid' is not closed")
   && (await page.isVisible('#td-parts-note')),
   'td/refused file: the parser\'s refusal is shown, not read as one part');
await page.setInputFiles('input[name="meshfile"]', path.join(DIR, 'slow.obj'));
await page.setInputFiles('input[name="meshfile"]', path.join(DIR, 'probe.obj'));
await page.waitForTimeout(1200);
ok((await pickers()).length === 0, 'td/stale reply: a late answer for an earlier file is dropped');
ok(!(await page.isVisible('#td-parts-note')), 'td/one part again: no leftover note');

// --- plane wave, then a drawn waveform ---------------------------------------
await page.selectOption('[name="td_source_kind"]', 'plane');
ok(!(await state('td_position')).shown && (await state('td_position')).disabled,
   'td/plane: the position is out of the way');
ok((await state('td_direction')).shown, 'td/plane: the direction is shown');
await page.selectOption('[name="td_profile"]', 'drawn');
const samples = await state('td_samples');
ok(samples.shown && samples.required, 'td/drawn: the samples are shown and required');
ok(!(await state('td_bandwidth_thz')).shown, 'td/drawn: the bandwidth is hidden');
ok(!(await valid()), 'td/drawn: an empty waveform blocks the submission');

// --- back to cavity: nothing td leaks into it --------------------------------
await page.selectOption('#mode-select', 'cavity');
ok(await valid(), 'cavity again: valid, nothing hidden blocks it');

await browser.close();
console.log(fails.length ? `FAILED ${fails.length}` : 'ALL PASS');
process.exit(fails.length ? 1 : 0);
