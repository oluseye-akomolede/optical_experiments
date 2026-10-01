// One-off in-browser check of the pc band card on a finished job.
import fs from 'node:fs';
import { createRequire } from 'node:module';
const require = createRequire('/opt/rendercheck/check.mjs');
const { chromium } = require('playwright-core');

const URL_ = process.env.MSE_URL;
const fails = [];
const ok = (c, msg) => { console.log(`${c ? 'PASS' : 'FAIL'}  ${msg}`); if (!c) fails.push(msg); };

// Oracle from the wire, computed here, not by the page's own code.
const st = await (await fetch(URL_ + '/status')).json();
const diag = Array.isArray(st.bands) ? st.bands[0] : st.bands;
const B = diag.bands;                        // [k][band]
const nk = B.length, nb = B[0].length;
console.log(`wire: status=${st.status} kind=${st.kind} nk=${nk} nb=${nb}`);
const oracleGaps = [];
for (let i = 0; i + 1 < nb; i++) {
  const lo = Math.max(...B.map((r) => r[i])), hi = Math.min(...B.map((r) => r[i + 1]));
  if (hi > lo) oracleGaps.push([i + 1, lo, hi]);
}
console.log('oracle gaps:', JSON.stringify(oracleGaps));

const browser = await chromium.launch({ args: ['--use-gl=angle', '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader', '--no-sandbox', '--disable-dev-shm-usage'] });
const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });
const errors = [];
page.on('console', (m) => { if (m.type() === 'error') errors.push(m.text()); });
page.on('pageerror', (e) => errors.push(String(e)));
await page.goto(URL_, { waitUntil: 'load', timeout: 120000 });
await page.waitForSelector('#bands-card:not([hidden])', { timeout: 120000 }).catch(() => {});
await page.waitForTimeout(2000);

const r = await page.evaluate(() => {
  const card = document.getElementById('bands-card');
  const paths = [...document.querySelectorAll('#bands-svg path')].map((p) => p.getAttribute('d') || '');
  const rows = [...document.querySelectorAll('#bands-gaps-body tr')].map((tr) =>
    [...tr.querySelectorAll('td')].map((td) => td.textContent.trim()));
  const box = card ? card.getBoundingClientRect() : null;
  return { hidden: card ? card.hidden : null, w: box && box.width, h: box && box.height,
    caption: (document.getElementById('bands-caption') || {}).textContent || '',
    paths, rows };
});
console.log('caption:', r.caption);
console.log('gap rows:', JSON.stringify(r.rows));
ok(r.hidden === false && r.w > 0 && r.h > 0, `bands card visible (${r.w}x${r.h})`);
ok(r.caption.startsWith(`${nb} bands`), `caption names ${nb} bands, not ${nk}`);
ok(r.paths.length === nb, `svg has ${r.paths.length} band paths, expected ${nb}`);
r.paths.forEach((d, i) => {
  const nums = (d.match(/-?\d+(\.\d+)?/g) || []).length;
  ok(!/NaN|Infinity|undefined/.test(d) && nums === 2 * nk, `path ${i}: ${nums} coords (expect ${2 * nk}), finite`);
});
if (oracleGaps.length) {
  ok(r.rows.length === oracleGaps.length, `gap table rows ${r.rows.length} == oracle ${oracleGaps.length}`);
  oracleGaps.forEach(([i, lo, hi], j) => {
    const row = r.rows[j] || [];
    ok(row[0] === `${i}–${i + 1}` && row[1] === lo.toFixed(4) && row[2] === hi.toFixed(4),
      `gap ${i}-${i + 1} row ${JSON.stringify(row)} matches ${lo.toFixed(4)}..${hi.toFixed(4)}`);
  });
} else {
  ok(r.rows.length === 1 && /No complete gap/.test(r.rows[0].join(' ')), 'no-gap row shown');
}
ok(errors.length === 0, `console errors: ${JSON.stringify(errors)}`);
// The image is the artifact a human actually looks at, so whether it exists
// is a finding, not a shrug: the old `.catch(() => {})` let a run print
// RESULT PASS with no /out/bands.png at all. Report it either way.
try {
  await page.screenshot({ path: '/out/bands.png' });
  const bytes = fs.statSync('/out/bands.png').size;
  console.log(`PASS  screenshot /out/bands.png written, ${bytes} bytes`);
} catch (err) {
  ok(false, `screenshot /out/bands.png was NOT written (${err && err.message}) ` +
    '-- if /out is not mounted for this run, that is the cause, not the page');
}
await browser.close();
console.log(fails.length ? `RESULT FAIL (${fails.length})` : 'RESULT PASS');
process.exit(fails.length ? 1 : 0);
