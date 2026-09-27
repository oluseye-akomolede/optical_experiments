// Positive control for the render check's screenshot fix.
//
// The claim to test is NOT "the check passes" -- it passed locally before the
// fix too, because the stall is a race. The claim is stronger and is the one
// that actually matters: the bug is reproducible on demand, and the fix removes
// it while the old code still fails.
//
// The mechanism, read out of playwright-core 1.49: locator.screenshot() calls
// screenshotElement -> handle._waitAndScrollIntoViewIfNeeded, which polls
// injected _checkElementIsStable; that reads element.getBoundingClientRect() on
// each animation frame and only proceeds once two consecutive reads are EQUAL --
// no epsilon. page.screenshot() never runs that poll at all.
//
// So: give #scene a sub-pixel size jitter. Whole-pixel movement would be caught
// by anyone's probe (the lead measured the box as constant at 834x414, and it
// is); the poll is defeated by what a whole-pixel probe cannot see. Then run
// both screenshot routes and show old-stalls / new-passes.
//
// Usage: node jitter_probe.mjs <url> <outdir>
//
// Result when this was run (2026-09-27, against the real app over
// scratch/live-08f832dd, inside the slim image):
//
//   rect over 6 frames: 6 distinct rects, every one of them whole-pixel 834x353
//   OLD  locator.screenshot() : STALLED after 12009ms
//   NEW  page.screenshot(clip): PASSED in 10592ms
//
// The "6 distinct but all 834x353" line is the whole point and is why an
// earlier probe missed this: a whole-pixel measurement of the box is correct
// and constant, and the poll that was timing out compares exact floats.

import { chromium } from 'playwright-core';
import { setTimeout as sleep } from 'node:timers/promises';

const url = process.argv[2];
const outDir = process.argv[3] || '/tmp';
if (!url) { console.error('usage: jitter_probe.mjs <url> [outdir]'); process.exit(2); }

const browser = await chromium.launch({
  args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader',
         '--no-sandbox', '--disable-dev-shm-usage'],
});
const page = await browser.newPage({ viewport: { width: 1024, height: 768 } });
await page.goto(url, { waitUntil: 'load', timeout: 120000 });
await page.waitForSelector('#viewer:not([hidden])', { timeout: 120000 });
await page.waitForTimeout(4000);

// Sub-pixel jitter on the canvas box: a fraction of a pixel, changing every
// frame, well below anything a human or an 8-second whole-pixel probe notices.
await page.evaluate(() => {
  const el = document.getElementById('scene');
  let t = 0;
  const step = () => {
    t += 0.03;
    el.style.transform = `translateX(${(Math.sin(t) * 0.4).toFixed(3)}px)`;
    requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
});
await page.waitForTimeout(1500);

// What the jitter actually did to the box, over several frames.
const seen = await page.evaluate(() => new Promise((resolve) => {
  const el = document.getElementById('scene');
  const out = [];
  let n = 0;
  const step = () => {
    const r = el.getBoundingClientRect();
    out.push({ top: r.top, left: r.left, w: r.width, h: r.height });
    if (++n < 6) requestAnimationFrame(step); else resolve(out);
  };
  requestAnimationFrame(step);
}));
console.log('rect over 6 frames:');
for (const r of seen) {
  console.log(`  top=${r.top} left=${r.left} w=${r.w} h=${r.h}`);
}
const distinct = new Set(seen.map((r) => `${r.top}|${r.left}|${r.w}|${r.h}`)).size;
console.log(`distinct rects across those frames: ${distinct} of ${seen.length}`);
console.log(`whole-pixel box: ${JSON.stringify(seen.map((r) => `${Math.round(r.w)}x${Math.round(r.h)}`))}`);

// OLD route: element screenshot. Expected to stall on the exact-equality poll.
let oldResult;
const t0 = Date.now();
try {
  await page.locator('#scene').screenshot({ path: `${outDir}/jitter-old.png`, timeout: 12000 });
  oldResult = `PASSED in ${Date.now() - t0}ms`;
} catch (e) {
  oldResult = `STALLED after ${Date.now() - t0}ms: ${e.message.split('\n')[0]}`;
}
console.log(`\nOLD  locator.screenshot() : ${oldResult}`);

// NEW route: clip of a page screenshot. Expected to be unaffected.
const box = await page.locator('#scene').boundingBox();
let newResult;
const t1 = Date.now();
try {
  await page.screenshot({ path: `${outDir}/jitter-new.png`, clip: box, timeout: 12000 });
  newResult = `PASSED in ${Date.now() - t1}ms`;
} catch (e) {
  newResult = `STALLED after ${Date.now() - t1}ms: ${e.message.split('\n')[0]}`;
}
console.log(`NEW  page.screenshot(clip): ${newResult}`);

console.log(`\nverdict: old=${oldResult.startsWith('STALLED') ? 'STALLED' : 'passed'} ` +
            `new=${newResult.startsWith('PASSED') ? 'PASSED' : 'stalled'} ` +
            `${oldResult.startsWith('STALLED') && newResult.startsWith('PASSED')
                ? '-> the fix is demonstrated, not assumed'
                : '-> INCONCLUSIVE: the control did not reproduce the stall'}`);

await browser.close();
