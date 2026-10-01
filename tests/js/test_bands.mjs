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
//
// ORIENTATION. The wire shape is bands[k][band] -- the outer index is the
// k-point, the inner index is the band number (pc_solve builds one vector of
// n_bands per k-point; _band_payload pins len(k_points) == len(bands)). Both
// readers shipped transposed, reading bands[band][k], and their fixtures were
// written in the readers' orientation, so they agreed with each other and not
// with the server. Every fixture now comes from ./fixtures/band_diagrams.json
// and is validated against _band_payload by tests/test_band_fixtures.py, and
// every one is NON-SQUARE so a transposed reader cannot agree by accident.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';

// A namespace import, not named imports: on the pre-fix reader the helpers
// below do not exist yet, and a missing named export aborts the whole module
// before a single test runs. Through the namespace the absence is one failing
// test, and the orientation tests still report their own reason.
import * as bands from '../../web/static/bands.js';

const FIXTURES = JSON.parse(
  readFileSync(new URL('./fixtures/band_diagrams.json', import.meta.url), 'utf8'),
);
const f = (name) => {
  if (!(name in FIXTURES) || name === '_about') {
    throw new Error(`no such band fixture: ${name}`);
  }
  return FIXTURES[name];
};

test('the k-axis is distance along the path, not the k-index', () => {
  // Γ=(0,0), X=(0.5,0), M=(0.5,0.5): the first leg is 0.5 long, the second
  // 0.5, so the cumulative lengths are 0, 0.5, 1.0. A path whose segments had
  // DIFFERENT lengths is the case that tells the two apart, so the third point
  // is placed to make the second leg 0.25, not 0.5.
  const k = [[0, 0], [0.5, 0], [0.5, 0.25]];
  const { lengths, total, ticks } = bands.kPathAxis(k, ['Γ', 'X', 'M']);
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
  const { ticks } = bands.kPathAxis(k, ['Γ', 'X', 'Γ']);
  assert.deepEqual(ticks, [{ at: 0, label: 'Γ' }, { at: 1, label: 'X' }]);
});

test('an empty path is an empty axis, not a throw', () => {
  assert.deepEqual(bands.kPathAxis([], []), { lengths: [], total: 0, ticks: [] });
  assert.deepEqual(bands.kPathAxis(null, null), { lengths: [], total: 0, ticks: [] });
});

test('the shape of a diagram is [k-points][bands], not [bands][k-points]', () => {
  // The orientation, named once and pinned. The fixture is 3 k-points by 2
  // bands, so the two counts differ and a reader that swapped them cannot
  // agree with this by luck.
  const d = f('gap_between_1_and_2');
  assert.equal(d.bands.length, 3, 'three k-point rows');
  assert.equal(d.bands[0].length, 2, 'two bands per row');
  assert.equal(bands.kPointCount(d), 3);
  assert.equal(bands.bandCount(d), 2);
});

test('the counts are guarded against an empty or absent diagram', () => {
  assert.equal(bands.kPointCount(null), 0);
  assert.equal(bands.bandCount(null), 0);
  assert.equal(bands.kPointCount({ bands: [] }), 0);
  assert.equal(bands.bandCount({ bands: [] }), 0);
});

test('the y-range spans every band of every diagram passed in', () => {
  // Two diagrams: the range must be their UNION, so two polarisations share one
  // scale. One fixture reaches -1 and the other 5; taken separately neither
  // range would hold both, so a reader that ignored one diagram fails here.
  const { lo, hi } = bands.bandRange([f('gap_between_1_and_2'), f('negative_band')]);
  assert.deepEqual({ lo, hi }, { lo: -1, hi: 5 });
});

test('a flat diagram gets a non-zero range so the mapping cannot divide by zero', () => {
  // The empty lattice: every frequency is 0. A zero-height range is an infinite
  // scale factor, and the plot would render NaN paths that look like nothing.
  const range = bands.bandRange([f('flat_empty_lattice')]);
  assert.ok(range.hi > range.lo, `flat range collapsed: ${JSON.stringify(range)}`);
});

test('a gap is the window between one band max and the next band min', () => {
  // Band 1 rises to 2 and stops; band 2 starts at 3. Over the whole path the
  // free window is [2, 3], centred at 2.5, ratio 1/2.5 = 0.4. The values are
  // chosen so that reading the array TRANSPOSED gives a different, still
  // plausible answer (-0.4 or no gap), which is the bug this catches.
  const gaps = bands.bandGaps(f('gap_between_1_and_2'));
  assert.equal(gaps.length, 1);
  assert.deepEqual(gaps[0], { lower: 2, upper: 3, midgapRatio: 0.4, index: 1 });
});

test('bands that touch everywhere have no gap', () => {
  // Band 1 goes 0->3, band 2 goes 1->4.5: their ranges overlap, so no
  // frequency is free of states. A per-k-point check would find "gaps" that are
  // not gaps -- and the transposed read of this fixture invents one ([1, 3]).
  assert.deepEqual(bands.bandGaps(f('touching_bands')), []);
});

test('every adjacent pair is checked, and the index is 1-based on the lower band', () => {
  // Three bands with two gaps: between 1-2 and between 2-3. The `index` is the
  // LOWER band's number, so the pair 2-3 is index 2 -- an off-by-one here is
  // invisible in the plot and wrong in the caption.
  const gaps = bands.bandGaps(f('three_bands_two_gaps'));
  assert.deepEqual(gaps.map((g) => g.index), [1, 2]);
  assert.deepEqual(gaps.map((g) => [g.lower, g.upper]), [[1, 2], [3, 5]]);
});

test('a single band has no gap to report', () => {
  // One band, three k-points. Read [k][band] there is one band column and
  // nothing to pair it with; read transposed there are three "bands", each of
  // one value, and two invented gaps. The counts are what tell them apart.
  assert.deepEqual(bands.bandGaps(f('single_band')), []);
  assert.deepEqual(bands.bandGaps(null), []);
});

test('the real Jetson diagram reports its gap, not a transposed silence', () => {
  // The measured extreme of the real 30-point TM diagram (Jetson bench,
  // 2026-10-01): band 1 tops at 0.281 at M, band 2 bottoms at 0.421 at X, so a
  // gap [0.281, 0.421] exists. Laid out on the Γ-X-M slice of that path.
  //
  // This is the failing case the reviewer measured: the shipped reader read
  // bands[band][k], so band "1" was the k=Γ row [0.0, 0.44] and band "2" the
  // k=X row [0.25, 0.421]; max(0, 0.44) = 0.44 is not below min(0.25, 0.421) =
  // 0.25, and the panel reported NO gap for a diagram whose whole point is the
  // gap.
  const gaps = bands.bandGaps(f('jetson_tm_slice'));
  assert.equal(gaps.length, 1, 'the real diagram has a complete gap');
  const g = gaps[0];
  assert.ok(Math.abs(g.lower - 0.281) < 1e-12, `lower ${g.lower}`);
  assert.ok(Math.abs(g.upper - 0.421) < 1e-12, `upper ${g.upper}`);
  assert.equal(g.index, 1);
  assert.ok(Math.abs(g.midgapRatio - 0.14 / 0.351) < 1e-12,
            `midgapRatio ${g.midgapRatio}`);
});

test('the plot scale maps the axis ends to the frame and inverts y', () => {
  // ω grows UP the page, SVG y grows down, so y(hi) must be the TOP of the
  // frame and y(lo) the bottom. Getting this wrong flips the diagram -- which
  // still looks like a band diagram.
  const s = bands.plotScale({ width: 200, height: 100, pad: { top: 10, right: 10, bottom: 10, left: 10 },
                              xMax: 2, yLo: 0, yHi: 1 });
  assert.equal(s.x(0), 10);
  assert.equal(s.x(2), 190);
  assert.equal(s.y(0), 90);   // the bottom of the 80px inner box
  assert.equal(s.y(1), 10);   // the top
});
