// Tests for the band-diagram geometry in bands.js. Run with:
//   node --test tests/js/*.mjs        <-- the glob matters; see below
//
// DO NOT run `node --test tests/js/`. That directory form discovers ZERO tests,
// prints "pass 0", and EXITS 0 -- so it looks green while having checked
// nothing. Read the test COUNT, not the exit code.
//
// The two things here that can be silently wrong are the ones a picture cannot
// reveal: the k-axis distance (plot against the k-INDEX and the short segments
// stretch, so the labels land in the wrong places and the gaps shift), and the
// gap arithmetic (off-by-one on the band pair still draws a plausible shaded
// rectangle). Both are checked against numbers written down by hand.

import test from 'node:test';
import assert from 'node:assert/strict';

import { kPathAxis, bandRange, bandGaps, plotScale } from '../../web/static/bands.js';

test('the k-axis is distance along the path, not the k-index', () => {
  // Γ=(0,0), X=(0.5,0), M=(0.5,0.5): the first leg is 0.5 long, the second
  // 0.5, so the cumulative lengths are 0, 0.5, 1.0. A path whose segments had
  // DIFFERENT lengths is the case that tells the two apart, so the third point
  // is placed to make the second leg 0.25, not 0.5.
  const k = [[0, 0], [0.5, 0], [0.5, 0.25]];
  const { lengths, total, ticks } = kPathAxis(k, ['Γ', 'X', 'M']);
  assert.deepEqual(lengths, [0, 0.5, 0.75]);
  assert.equal(total, 0.75);
  // Ticks at the cumulative distances, NOT at 0, 1, 2.
  assert.deepEqual(ticks, [
    { at: 0, label: 'Γ' }, { at: 0.5, label: 'X' }, { at: 0.75, label: 'M' },
  ]);
});

test('a repeated label gets one tick, at its first occurrence', () => {
  // A path that returns to Γ carries the symbol twice. Two ticks at the same
  // place would stack the text; the first occurrence is where the path enters.
  const k = [[0, 0], [1, 0], [0, 0]];
  const { ticks } = kPathAxis(k, ['Γ', 'X', 'Γ']);
  assert.deepEqual(ticks, [{ at: 0, label: 'Γ' }, { at: 1, label: 'X' }]);
});

test('an empty path is an empty axis, not a throw', () => {
  assert.deepEqual(kPathAxis([], []), { lengths: [], total: 0, ticks: [] });
  assert.deepEqual(kPathAxis(null, null), { lengths: [], total: 0, ticks: [] });
});

test('the y-range spans every band of every diagram passed in', () => {
  // Two diagrams: the range must be their UNION, so two polarisations share one
  // scale. Taken separately the max would be 3 and a comparison between the
  // panels would be off by the difference the union hides.
  const a = { bands: [[0, 1], [2, 3]] };
  const b = { bands: [[-1, 0], [1, 5]] };
  assert.deepEqual(bandRange([a, b]), { lo: -1, hi: 5 });
});

test('a flat diagram gets a non-zero range so the mapping cannot divide by zero', () => {
  // The empty lattice: every frequency is 0. A zero-height range is an infinite
  // scale factor, and the plot would render NaN paths that look like nothing.
  const flat = { bands: [[0, 0], [0, 0]] };
  const range = bandRange([flat]);
  assert.ok(range.hi > range.lo, `flat range collapsed: ${JSON.stringify(range)}`);
});

test('a gap is the window between one band max and the next band min', () => {
  // Band 1 tops at 2 (over k), band 2 bottoms at 3: a gap of width 1 centered
  // at 2.5, ratio 1/2.5 = 0.4. The numbers are chosen so a max/min SWAP gives a
  // different, still-plausible ratio -- which is the bug this catches.
  const diagram = { bands: [[0, 2], [3, 4]] };
  const gaps = bandGaps(diagram);
  assert.equal(gaps.length, 1);
  assert.deepEqual(gaps[0], { lower: 2, upper: 3, midgapRatio: 0.4, index: 1 });
});

test('bands that touch everywhere have no gap', () => {
  // Band 1 goes 0->3, band 2 goes 1->4: their ranges overlap, so no frequency
  // is free of states. A per-k-point check would find "gaps" that are not gaps.
  const touching = { bands: [[0, 3], [1, 4]] };
  assert.deepEqual(bandGaps(touching), []);
});

test('every adjacent pair is checked, and the index is 1-based on the lower band', () => {
  // Three bands with two gaps: between 1-2 and between 2-3. The `index` is the
  // LOWER band's number, so the pair 2-3 is index 2 -- an off-by-one here is
  // invisible in the plot and wrong in the caption.
  const diagram = { bands: [[0, 1], [2, 3], [5, 6]] };
  const gaps = bandGaps(diagram);
  assert.deepEqual(gaps.map((g) => g.index), [1, 2]);
  assert.deepEqual(gaps.map((g) => [g.lower, g.upper]),
                   [[1, 2], [3, 5]]);
});

test('a single band has no gap to report', () => {
  assert.deepEqual(bandGaps({ bands: [[1, 2, 3]] }), []);
  assert.deepEqual(bandGaps(null), []);
});

test('the plot scale maps the axis ends to the frame and inverts y', () => {
  // ω grows UP the page, SVG y grows down, so y(hi) must be the TOP of the
  // frame and y(lo) the bottom. Getting this wrong flips the diagram -- which
  // still looks like a band diagram.
  const s = plotScale({ width: 200, height: 100, pad: { top: 10, right: 10, bottom: 10, left: 10 },
                        xMax: 2, yLo: 0, yHi: 1 });
  assert.equal(s.x(0), 10);
  assert.equal(s.x(2), 190);
  assert.equal(s.y(0), 90);   // the bottom of the 80px inner box
  assert.equal(s.y(1), 10);   // the top
});
