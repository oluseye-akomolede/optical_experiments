// The SVG renderer for a band diagram. Run with:
//   node --test tests/js/*.mjs        <-- the glob matters; see below
//
// DO NOT run `node --test tests/js/`. That directory form discovers ZERO tests,
// prints "pass 0", and EXITS 0 -- so it looks green while having checked
// nothing. Read the test COUNT, not the exit code.
//
// bandplot.js touches the DOM, so this file installs a minimal `document` with
// only the two methods the renderer uses (createElementNS, and the setAttribute
// / appendChild / textContent surface of an element). The stub RECORDS what was
// built rather than rendering it -- the thing under test is which elements a
// diagram produces and with what coordinates, and a real DOM would answer that
// through a browser instead of through this process. The node shapes here are
// deliberately dull: they exist so a renderer that produced no elements, or the
// wrong number, fails rather than passing on an empty document.
//
// ORIENTATION. The wire shape is bands[k][band]; see test_bands.mjs for the
// full note. Every fixture comes from ./fixtures/band_diagrams.json and is
// non-square, so the path count (one per BAND) and the point count (one per
// k-point) are different numbers and a transposed renderer cannot agree.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

function makeElement(tag) {
  return {
    tagName: tag,
    attributes: {},
    children: [],
    textContent: '',
    setAttribute(k, v) { this.attributes[k] = v; },
    appendChild(child) { this.children.push(child); return child; },
    get firstChild() { return this.children[0] || null; },
    removeChild(child) {
      const i = this.children.indexOf(child);
      if (i >= 0) this.children.splice(i, 1);
      return child;
    },
    // The renderer walks nothing and queries nothing, so the stub needs no more
    // than this; adding a querySelector here would be inventing an interface
    // the module does not use.
  };
}

const svg = makeElement('svg');
globalThis.document = { createElementNS: (_ns, tag) => makeElement(tag) };

const { drawBandDiagram } = await import('../../web/static/bandplot.js');

const FIXTURES = JSON.parse(
  readFileSync(new URL('./fixtures/band_diagrams.json', import.meta.url), 'utf8'),
);
const f = (name) => {
  if (!(name in FIXTURES) || name === '_about') {
    throw new Error(`no such band fixture: ${name}`);
  }
  return FIXTURES[name];
};

function diagram(over = {}) {
  return { ...f('gap_between_1_and_2'), ...over };
}

test('a diagram draws one path per BAND, over the k-points', () => {
  const d = diagram();
  drawBandDiagram(svg, [d]);
  const paths = svg.children.filter((c) => c.tagName === 'path');
  // Two bands -> two paths, and three k-points -> three points per path. The
  // fixture is 3x2 so those are different numbers: a renderer that drew one
  // path per ROW would produce three two-point paths. That is the transposition
  // the shipped renderer had, and it drew 30 four-point paths on a real job.
  assert.equal(d.bands.length, 3, 'the fixture is 3 k-points ...');
  assert.equal(d.bands[0].length, 2, '... by 2 bands');
  assert.equal(paths.length, 2, 'one path per band, not per k-point');
  // Each path is a real polyline: three k-points -> M plus two L commands.
  for (const p of paths) {
    assert.match(p.attributes.d, /^M[-\d.]+,[-\d.]+ L[-\d.]+,[-\d.]+ L[-\d.]+,[-\d.]+$/);
  }
});

test('a band path visits the k-points left to right', () => {
  // The x of the points along one path must INCREASE: the path is sampled
  // along k, so a renderer that walked the other axis would draw a two-point
  // stub (and, before the fix, did).
  drawBandDiagram(svg, [diagram()]);
  const first = svg.children.filter((c) => c.tagName === 'path')[0];
  const xs = first.attributes.d.split(' ').map((cmd) => Number(cmd.slice(1).split(',')[0]));
  assert.equal(xs.length, 3);
  assert.ok(xs[0] < xs[1] && xs[1] < xs[2], `x not increasing along k: ${xs}`);
});

test('the tick labels are the high-symmetry points, in path order', () => {
  drawBandDiagram(svg, [diagram()]);
  const texts = svg.children.filter((c) => c.tagName === 'text')
    .map((t) => t.textContent);
  for (const label of ['Γ', 'X', 'M']) {
    assert.ok(texts.includes(label), `tick ${label} missing; got ${texts}`);
  }
});

test('a gap between bands is shaded exactly once', () => {
  // Band 1 tops at 2, band 2 bottoms at 3: one gap. The shaded rect is the
  // physical claim of the feature, so its presence is asserted directly rather
  // than left to the picture -- and its height must be positive, since a rect
  // of zero height is an invisible gap that reads as "no gap".
  drawBandDiagram(svg, [diagram()]);
  const shaded = svg.children.filter((c) => c.tagName === 'rect'
    && c.attributes['fill-opacity'] === '0.12');
  assert.equal(shaded.length, 1);
  assert.ok(Number(shaded[0].attributes.height) > 0,
    'the gap rect has no height');
});

test('bands that touch produce no shaded rect', () => {
  drawBandDiagram(svg, [f('touching_bands')]);
  const shaded = svg.children.filter((c) => c.tagName === 'rect'
    && c.attributes['fill-opacity'] === '0.12');
  assert.equal(shaded.length, 0,
    'a shaded band appeared where the bands overlap');
});

test('an empty diagram draws nothing and does not throw', () => {
  drawBandDiagram(svg, []);
  assert.equal(svg.children.length, 0, 'a stale frame survived an empty draw');
  // And the negative control for the control: the same svg, given a real
  // diagram afterwards, must populate -- proving the emptiness above was the
  // input's doing and not a renderer that never draws anything. Compared
  // against ZERO, not against the previous test's leftover element count:
  // that count came from a different fixture, so it varied with whether the
  // previous fixture happened to have a shaded gap.
  drawBandDiagram(svg, [diagram()]);
  assert.ok(svg.children.length > 0, 'the renderer drew nothing for a real diagram');
});

test('a ragged diagram draws no NaN into its path data', () => {
  // A RAGGED diagram -- rows of different lengths -- is a shape the server's
  // own validator (_band_payload) REJECTS, so it cannot arrive from production.
  // It is here because the renderer is a reader of whatever it is handed, and
  // the band-wise loop reaches ACROSS rows: 8e2d60e's reader drew one path per
  // row and never looked past a row's end, so it emitted no NaN, and the
  // band-wise one that replaced it hit `undefined`. scale.y(undefined) is NaN,
  // and a NaN in an SVG path `d` is not an error the browser reports -- it
  // silently drops the whole subpath, so the plot would lose a curve while the
  // gap table beside it (bandGaps skips non-finite values) still counted that
  // band. Plot and table must agree about which values exist.
  //
  // Deliberately NOT in fixtures/band_diagrams.json: that file's contract is
  // that every entry passes _band_payload, which this one must not.
  const ragged = {
    polarisation: 'TM',
    labels: ['Γ', 'X', 'M'],
    k_points: [[0, 0], [1, 0], [1, 1]],
    bands: [[0, 1, 2], [3], [4, 5]],
  };
  drawBandDiagram(svg, [ragged]);
  const paths = svg.children.filter((c) => c.tagName === 'path');
  assert.equal(paths.length, 3, 'one path per band in the widest row');
  for (const p of paths) {
    assert.ok(!/NaN/.test(p.attributes.d),
      `a ragged diagram put NaN in path data: ${p.attributes.d}`);
  }
});

test('the shared frame is fixed once, so ticks and marks cannot disagree', () => {
  // The viewBox is set from the SAME width/height the scale used. If it were
  // set independently, a caller passing a size would get a frame the marks were
  // not mapped into -- plausible-looking and wrong.
  drawBandDiagram(svg, [diagram()], { width: 500, height: 300 });
  assert.equal(svg.attributes.viewBox, '0 0 500 300');
});
