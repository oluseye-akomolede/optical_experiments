// The end of a finished job, driven through the real viewer: does the mode
// table get built, does the viewer un-hide, and does a throw inside render
// surface instead of vanishing?
//
// WHY THIS FILE EXISTS. A production regression shipped: buildModeTable (the
// last function in web/static/viewer.js) was deleted by an edit, finish() at
// :101 still called it, and the throw landed in tick()'s `.catch(() => {})`.
// The result was invisible in the worst way -- on EVERY finished job the
// viewer never appeared and the mode table was empty, with no console error,
// because the catch swallowed the ReferenceError. The post-deploy render check
// caught it only by timing out on `#viewer:not([hidden])`; a browser probe
// read status 200 "done", 0 rows, viewer hidden for 85 s.
//
// So this file tests two things, both of them BEHAVIOUR rather than a name:
//   1. driving render({status:'done', modes:[...]}) actually builds the rows
//      and un-hides the viewer;
//   2. a throw inside render/finish is REPORTED (console.error + the page's
//      failure text), not swallowed -- the could-not-ask pattern.
//
// The real viewer.js is loaded, not a double. It is an ES module with no
// exports whose last line is `if (root) init(root);`, so it cannot simply be
// imported and inspected. It is evaluated in a vm context whose document
// deliberately has NO `#job` element, which makes that call a no-op; the test
// then reaches `init`/`render` by running the page's own init against its own
// `#job`. WebGL (loadFieldData) is stubbed at the boundary, because the claim
// here is about the table and the failure path, not about three.js -- and the
// test asserts loadFieldData was reached, so stubbing it cannot hide a
// failure to get there.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import vm from 'node:vm';

const here = dirname(fileURLToPath(import.meta.url));
const viewerSrc = readFileSync(join(here, '..', '..', 'web', 'static', 'viewer.js'), 'utf8');

// The REAL helpers viewer.js imports from bands.js. The harness strips imports,
// so inside the vm these are free identifiers resolved against the context --
// injecting the real ones means the caption test is not measuring a double.
import { bandCount, bandGaps } from '../../web/static/bands.js';
const FIXTURES = JSON.parse(
  readFileSync(join(here, 'fixtures', 'band_diagrams.json'), 'utf8'),
);

// --- A DOM just large enough for the finished-job path ---------------------
class El {
  constructor(tag = 'div', id = '') {
    this.tagName = tag.toUpperCase();
    this.id = id;
    this.children = [];
    this.hidden = false;
    this.textContent = '';
    this.innerHTML = '';
    this.title = '';
    this.style = {};
    this.dataset = {};
    this.tabIndex = -1;
    this._attrs = {};
    this._listeners = {};
    // buildModeTable marks an unreadable mode with classList.add('unavailable'),
    // so the stub needs the method or that row throws and the test would be
    // measuring the stub rather than the page.
    this.className = '';
    this.classList = {
      _o: this,
      add: (c) => { this.className = (this.className + ' ' + c).trim(); },
      contains: (c) => this.className.split(/\s+/).includes(c),
    };
  }
  setAttribute(k, v) { this._attrs[k] = String(v); }
  getAttribute(k) { return this._attrs[k] ?? null; }
  addEventListener(t, fn) { (this._listeners[t] ||= []).push(fn); }
  appendChild(c) { this.children.push(c); return c; }
  querySelector() { return null; }
  querySelectorAll() { return []; }
}

function makeDocument(withJob) {
  const ids = ['job', 'status-text', 'stage-text', 'bar', 'elapsed', 'queue-line',
               'progress', 'failure', 'failure-text', 'viewer', 'modes-card',
               'modes-body', 'scene', 'viewer-fallback',
               // The band panel, reached only by a pc job. Named here rather
               // than left out because a missing element makes renderBands
               // return early at `if (!svg)` -- the caption test would then
               // assert on a stub that never gave the page its panel, and
               // read as "the caption is empty" for a harness reason.
               'bands-card', 'bands-svg', 'bands-caption', 'bands-gaps-body'];
  const byId = {};
  for (const id of ids) byId[id] = new El('div', id);
  // The template's own starting states (web/templates/job.html): the viewer,
  // the modes card, the band card and the failure box start hidden, and the
  // progress block starts visible. A stub that had them all visible would make
  // "the viewer un-hid" and "the failure box stayed hidden" both vacuous.
  for (const id of ['viewer', 'modes-card', 'bands-card', 'failure', 'queue-line']) {
    byId[id].hidden = true;
  }
  // The auto-init guard: `const root = document.getElementById('job');
  // if (root) init(root);`. Present so init runs; the test grabs the root and
  // calls init itself only when it wants a second run.
  if (!withJob) byId.job = null;
  return {
    getElementById: (id) => byId[id] ?? null,
    createElement: (t) => new El(t),
    addEventListener: () => {},
    _byId: byId,
  };
}

/**
 * Remove one anchor from the source, and FAIL if it did not match exactly once.
 *
 * A bare `.replace(re, '')` that matches zero times is silent, and the harness
 * then tests a file it only *thinks* it edited: the guard is armed against text
 * that is not there. Reformatting viewer.js (quote style, spacing) must not be
 * able to disarm the harness with no signal. Use this for single-line,
 * once-only anchors; a multi-occurrence strip (the imports) asserts its own
 * postcondition instead.
 */
function stripOnce(src, re, what) {
  const count = (src.match(new RegExp(re.source, re.flags.includes('g') ? re.flags : re.flags + 'g')) || []).length;
  if (count !== 1) {
    throw new Error(
      `viewer.js strip "${what}" matched ${count} times, expected exactly 1: ` +
      `the harness would test text it did not edit`);
  }
  return src.replace(re, '');
}

/**
 * Remove every import statement, and FAIL if any survived.
 *
 * The imports span several lines, so a single-line pattern would leave `export`
 * behind and the module would fail to parse -- a harness error that proves
 * nothing about the page. There are several, so this is not a "matched once"
 * strip: assert the postcondition instead.
 */
function stripImports(src) {
  const out = src.replace(/^\s*import\b[\s\S]*?;\s*$/gm, '');
  if (/^\s*import\b/m.test(out)) {
    throw new Error('viewer.js strip "imports" left an import behind: the ' +
      'harness would test text it did not edit');
  }
  return out;
}

/** Load viewer.js with no #job, so nothing auto-runs, and return its globals. */
function loadViewer() {
  const document = makeDocument(true);
  const ctx = {
    document,
    window: { devicePixelRatio: 1, addEventListener: () => {} },
    console,
    Date, Math, Number, JSON, Object, Array, String, Promise, Error,
    // A retry scheduled with the real setTimeout would keep the test process
    // alive for 3 s per still-live tick. Record the schedule and do not run
    // it: the tests assert the SCHEDULE (a retry happened) and the absence of
    // a reported error, not the second fetch.
    setTimeout: (fn) => { (ctx.__timeouts ||= []).push(fn); return 0; },
    requestAnimationFrame: (fn) => { (ctx.__timeouts ||= []).push(fn); return 0; },
    cancelAnimationFrame: () => {},
    fetch: async () => { throw new Error('fetch not stubbed in this test'); },
    WebGL2RenderingContext: class {},
    globalThis: {},
  };
  ctx.globalThis = ctx;
  vm.createContext(ctx);
  let stripped = stripImports(viewerSrc);
  // The module also exports formatFrequency/formatWavelength (they are tested
  // elsewhere). A vm script has no module record, so the `export` keyword must
  // go -- but the functions stay, because buildModeTable calls them and that
  // call is what the test exercises. Two occurrences, so assert the
  // postcondition rather than a single match.
  stripped = stripped.replace(/^export function /gm, 'function ');
  if (/^export\b/m.test(stripped)) {
    throw new Error('viewer.js strip "export keyword" left an export behind: ' +
      'the harness would test text it did not edit');
  }
  // The auto-init guard. Removed so nothing runs at load. NOTE: the tests call
  // init() themselves, so these two lines are not what is under test -- a
  // mutation to them cannot redden anything, which is why what matters is that
  // the ANCHORS still match (stripOnce throws otherwise), not the lines.
  stripped = stripOnce(stripped, /^const root = document\.getElementById\('job'\);$/m,
                       'root binding');
  stripped = stripOnce(stripped, /^if \(root\) init\(root\);$/m, 'init call');
  // The WebGL boundary. loadFieldData builds a THREE.WebGLRenderer on a real
  // canvas; a vm stub has no GL, and on the FIXED code the finished-job path
  // now REACHES it (on the broken code it threw first), so leaving it live
  // hangs rather than failing. Replace the body's first step with a recording
  // stub -- the claims here are about the table and the failure path, not
  // about three.js.
  const loadFieldCall = /^function loadFieldData\(jobId, statusData\) \{$/m;
  if ((stripped.match(new RegExp(loadFieldCall.source, 'gm')) || []).length !== 1) {
    throw new Error('viewer.js strip "loadFieldData body" did not match exactly once');
  }
  stripped = stripped.replace(
    loadFieldCall,
    'function loadFieldData(jobId, statusData) {\n' +
    '  (globalThis.__loadFieldCalls ||= []).push(jobId);\n' +
    '  return;');
  vm.runInContext(stripped, ctx, { filename: 'viewer.js' });
  return ctx;
}

// --- The tests -------------------------------------------------------------

test('a finished job builds the mode rows and un-hides the viewer', async () => {
  const ctx = loadViewer();
  const doc = ctx.document;
  const root = new El('div', 'job');
  root.dataset.jobId = 'JOB-1';
  doc._byId.job = root;

  const data = {
    status: 'done',
    modes: [
      { index: 1, freq_hz: 12345.0, vacuum_wavelength_m: 2.4e-5, available: true },
      { index: 2, freq_hz: 23456.0, vacuum_wavelength_m: 1.2e-5, available: false },
    ],
  };

  // Drive the page's own poll: one status fetch that resolves to `done`.
  ctx.fetch = async () => ({ ok: true, status: 200, json: async () => data });

  // init() is not exported; run it by evaluating the module's init binding.
  vm.runInContext('init(document.getElementById("job"))', ctx);
  await new Promise((r) => setTimeout(r, 30));

  const body = doc._byId['modes-body'];
  assert.equal(body.children.length, 2,
    'the finished-job path must build one row per mode; 0 rows means ' +
    'buildModeTable was not reached (or was swallowed before it ran)');
  assert.equal(doc._byId.viewer.hidden, false,
    '#viewer must be un-hidden once the job is done; if it is still hidden ' +
    'the page shows nothing and the user has no way to see the modes');
  assert.equal(doc._byId['modes-card'].hidden, false,
    'the modes card must be shown alongside the viewer');
  assert.equal(Array.from(ctx.__loadFieldCalls || []).join(','), 'JOB-1',
    'the finished-job path must reach loadFieldData with the job id -- this ' +
    'is what the stubbed WebGL boundary cannot hide, so the table assertions ' +
    'above are on a path that actually got that far');
  // The unreadable mode stays visible but not selectable -- the numbering has
  // to match the physics even when a mode has no field.
  const unreadable = body.children[1];
  assert.ok(unreadable.classList.contains('unavailable'),
    'a mode with available:false must be marked unavailable');
  assert.ok(unreadable._attrs.role === undefined,
    'a mode with available:false must not be a selectable button');
  // The readable mode IS selectable, so the guard above is not vacuous.
  assert.equal(body.children[0]._attrs.role, 'button',
    'a mode with available:true must be a selectable button');
});

test('a throw inside render is reported, not swallowed', async () => {
  const ctx = loadViewer();
  const doc = ctx.document;
  const root = new El('div', 'job');
  root.dataset.jobId = 'JOB-2';
  doc._byId.job = root;

  // A payload that makes render() throw: modes is not iterable, so
  // buildModeTable blows up. THIS is the shape of the production bug -- a
  // throw on the done path that the old `.catch(() => {})` ate.
  const bad = { status: 'done', modes: { not: 'an array' } };
  ctx.fetch = async () => ({ ok: true, status: 200, json: async () => bad });

  const errors = [];
  const realError = ctx.console.error;
  ctx.console.error = (...a) => errors.push(a.join(' '));

  vm.runInContext('init(document.getElementById("job"))', ctx);
  await new Promise((r) => setTimeout(r, 30));

  ctx.console.error = realError;

  assert.ok(errors.length >= 1,
    'a throw inside render/finish must reach console.error; silence is the ' +
    'exact production failure (a blank page with no way to tell why)');
  assert.equal(doc._byId.failure.hidden, false,
    'the page must show its failure text when the viewer cannot draw');
  assert.match(String(doc._byId['failure-text'].textContent), /modes/i,
    'the failure text must say something about the modes');
});

test('a status fetch failure retries rather than reporting a render bug', async () => {
  const ctx = loadViewer();
  const doc = ctx.document;
  const root = new El('div', 'job');
  root.dataset.jobId = 'JOB-3';
  doc._byId.job = root;

  let calls = 0;
  ctx.fetch = async () => { calls += 1; throw new Error('network down'); };

  const errors = [];
  const realError = ctx.console.error;
  ctx.console.error = (...a) => errors.push(a.join(' '));

  vm.runInContext('init(document.getElementById("job"))', ctx);
  await new Promise((r) => setTimeout(r, 30));

  ctx.console.error = realError;

  assert.equal(calls, 1, 'the first tick fires once');
  assert.equal(errors.length, 0,
    'a transport failure is transient and must NOT be reported as a bug');
  assert.equal(doc._byId.failure.hidden, true,
    'a transient fetch failure must not put the page into a failed state');
});

test('a running-status payload schedules a re-poll and leaves the viewer hidden', async () => {
  const ctx = loadViewer();
  const doc = ctx.document;
  const root = new El('div', 'job');
  root.dataset.jobId = 'JOB-4';
  doc._byId.job = root;

  // A payload that is NOT done: the job is still computing, so the page must
  // poll again rather than draw or fail. Deleting render()'s
  // `setTimeout(tick, 3000)` would stop the page ever noticing the job
  // finished -- the viewer would stay blank forever -- and no other test
  // drives a non-done payload, so nothing else can catch it.
  const data = { status: 'running', progress: 0.4 };
  let calls = 0;
  ctx.fetch = async () => { calls += 1; return { ok: true, status: 200, json: async () => data }; };

  const errors = [];
  const realError = ctx.console.error;
  ctx.console.error = (...a) => errors.push(a.join(' '));

  vm.runInContext('init(document.getElementById("job"))', ctx);
  await new Promise((r) => setTimeout(r, 30));

  ctx.console.error = realError;

  assert.equal(calls, 1, 'the first tick fires once');
  const scheduled = ctx.__timeouts || [];
  assert.ok(scheduled.length >= 1,
    'a running job must schedule a re-poll; without it the page never sees the ' +
    'job finish and the viewer stays blank forever');
  assert.equal(doc._byId.viewer.hidden, true,
    'the viewer must stay hidden while the job is still running');
  assert.equal(doc._byId['modes-card'].hidden, true,
    'the modes card must stay hidden while the job is still running');
  assert.equal(doc._byId.failure.hidden, true,
    'a running job is not a failure and must not show the failure text');
  assert.equal(errors.length, 0, 'a running job must not report an error');
});

test('a pc job captions the BAND count, not the k-point count', async () => {
  // The caption is the app's headline sentence for a finished pc job, and it
  // read ``diagrams[0].bands.length`` -- the OUTER array, which on the wire is
  // the K-POINT count. On a real 30x4 diagram it said "30 bands". Reverting
  // ``bandCount(diagrams[0])`` back to that leaves every other test green: the
  // whole suite runs with the imports stripped, so bandCount is a free
  // identifier no test supplied, and only a test that DRIVES a pc payload
  // through the real renderBands can see the caption at all.
  //
  // The fixture is 3 k-points x 2 bands, so the two counts differ and a
  // transposed caption cannot accidentally agree.
  const ctx = loadViewer();
  const doc = ctx.document;
  const root = new El('div', 'job');
  root.dataset.jobId = 'JOB-PC';
  doc._byId.job = root;

  const diagram = FIXTURES.jetson_tm_slice;
  assert.equal(diagram.bands.length, 3, 'the fixture is 3 k-points ...');
  assert.equal(diagram.bands[0].length, 2, '... by 2 bands');

  // The stripped imports, supplied as the real functions.
  ctx.bandCount = bandCount;
  ctx.bandGaps = bandGaps;
  // The plot itself is tested in test_bandplot.mjs; stub it at the boundary but
  // RECORD the call, so "the caption was set" cannot be satisfied by a
  // renderBands that never reached the plot.
  const drawn = [];
  ctx.drawBandDiagram = (svg, diagrams) => { drawn.push(diagrams); };

  const data = { status: 'done', kind: 'pc', bands: diagram };
  ctx.fetch = async () => ({ ok: true, status: 200, json: async () => data });

  vm.runInContext('init(document.getElementById("job"))', ctx);
  await new Promise((r) => setTimeout(r, 30));

  assert.equal(doc._byId['bands-card'].hidden, false,
    'a finished pc job must un-hide the band panel');
  assert.equal(drawn.length, 1, 'renderBands must reach the plot');
  assert.equal(drawn[0].length, 1, 'one diagram in, one diagram drawn');
  assert.equal(doc._byId.viewer.hidden, true,
    'a pc job has no 3-D field, so the cavity viewer must stay hidden');

  const caption = String(doc._byId['bands-caption'].textContent);
  assert.match(caption, /^2 bands/,
    `the caption must count BANDS (2), not k-points (3): "${caption}"`);
  assert.match(caption, /TM polarisation/,
    `the caption must name the polarisation it drew: "${caption}"`);

  // The gap table is the second real reader of the same payload, so it is
  // asserted here too: band 1 tops at 0.281, band 2 bottoms at 0.421, so the
  // panel must show one gap row rather than "no complete gap".
  const rows = doc._byId['bands-gaps-body'].children;
  assert.equal(rows.length, 1,
    `the panel must show the one gap the diagram has, got ${rows.length} rows`);
  assert.match(String(rows[0].innerHTML), /0\.2810[\s\S]*0\.4210/,
    `the gap row must carry the measured bounds: ${rows[0].innerHTML}`);
});
