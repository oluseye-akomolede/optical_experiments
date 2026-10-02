// In-browser check of the upload form's show/hide script (index.html).
//
// The Python tests read the page's markup; none of them can run its script,
// and the script is what decides which controls are required, disabled and
// shown. So this loads the rendered page in headless Chromium and asks the
// browser itself: is the form valid as the owner would submit it, in each
// mode, and is every control the chosen arm does not use out of the way?
//
// Inputs: FORM_DIR holds index.html (the rendered GET /), style.css and
// probe.obj. Nothing is fetched from the network; every request is answered
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
await page.route('**/*', (route) => {
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
