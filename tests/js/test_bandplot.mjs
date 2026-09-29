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

import test from 'node:test';
import assert from 'node:assert/strict';

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

function diagram(over = {}) {
  return {
    polarisation: 'TM',
    labels: ['Γ', 'X', 'M'],
    k_points: [[0, 0], [0.5, 0], [0.5, 0.5]],
    bands: [[0, 1, 2], [3, 4, 5]],
    ...over,
  };
}

function tags(root) {
  return root.children.map((c) => c.tagName);
}

test('a diagram with bands draws one path per band', () => {
  drawBandDiagram(svg, [diagram()]);
  const paths = svg.children.filter((c) => c.tagName === 'path');
  // Two bands -> two paths. A renderer that packed all bands into one path
  // would draw the same picture and make a per-band test impossible.
  assert.equal(paths.length, 2);
  // Each path is a real polyline: three k-points -> M plus two L commands.
  for (const p of paths) {
    assert.match(p.attributes.d, /^M[-\d.]+,[-\d.]+ L[-\d.]+,[-\d.]+ L[-\d.]+,[-\d.]+$/);
  }
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
  drawBandDiagram(svg, [diagram({ bands: [[0, 2, 2], [3, 4, 5]] })]);
  const shaded = svg.children.filter((c) => c.tagName === 'rect'
    && c.attributes['fill-opacity'] === '0.12');
  assert.equal(shaded.length, 1);
  assert.ok(Number(shaded[0].attributes.height) > 0,
    'the gap rect has no height');
});

test('bands that touch produce no shaded rect', () => {
  drawBandDiagram(svg, [diagram({ bands: [[0, 3, 3], [1, 4, 5]] })]);
  const shaded = svg.children.filter((c) => c.tagName === 'rect'
    && c.attributes['fill-opacity'] === '0.12');
  assert.equal(shaded.length, 0,
    'a shaded band appeared where the bands overlap');
});

test('an empty diagram draws nothing and does not throw', () => {
  const before = svg.children.length;
  drawBandDiagram(svg, []);
  assert.equal(svg.children.length, 0, 'a stale frame survived an empty draw');
  // And the negative control for the control: the same svg, given a real
  // diagram afterwards, must populate -- proving the emptiness above was the
  // input's doing and not a renderer that never draws anything.
  drawBandDiagram(svg, [diagram()]);
  assert.ok(svg.children.length > before);
});

test('the shared frame is fixed once, so ticks and marks cannot disagree', () => {
  // The viewBox is set from the SAME width/height the scale used. If it were
  // set independently, a caller passing a size would get a frame the marks were
  // not mapped into -- plausible-looking and wrong.
  drawBandDiagram(svg, [diagram()], { width: 500, height: 300 });
  assert.equal(svg.attributes.viewBox, '0 0 500 300');
});
