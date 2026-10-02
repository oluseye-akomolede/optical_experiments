// Tests for the time-domain plot helpers in tdplot.js. Run with:
//   node --test tests/js/*.mjs        <-- the glob matters (see test_bands.mjs)
//
// The trace fixture is NON-SQUARE (2 probes x 5 steps) so a transposed read
// (steps x probes) cannot agree by accident; values are written down by hand.

import test from 'node:test';
import assert from 'node:assert/strict';
import * as tdplot from '../../web/static/tdplot.js';

test('siPrefix picks the prefix that puts the span in 1..1000', () => {
  assert.deepEqual(tdplot.siPrefix(2e-10), { factor: 1e-12, prefix: 'p' });
  assert.deepEqual(tdplot.siPrefix(3.5e14), { factor: 1e12, prefix: 'T' });
  assert.deepEqual(tdplot.siPrefix(1e-9), { factor: 1e-9, prefix: 'n' });
  assert.deepEqual(tdplot.siPrefix(999e-9), { factor: 1e-9, prefix: 'n' });
  assert.deepEqual(tdplot.siPrefix(0), { factor: 1, prefix: '' });
});

test('niceTicks steps on 1/2/5 and stays inside the range', () => {
  assert.deepEqual(tdplot.niceTicks(0, 10, 5), [0, 2, 4, 6, 8, 10]);
  assert.deepEqual(tdplot.niceTicks(0.3, 1.2, 4), [0.4, 0.6, 0.8, 1, 1.2]);
  assert.deepEqual(tdplot.niceTicks(-1, 1, 4), [-1, -0.5, 0, 0.5, 1]);
});

function traceBuffer(rows) {
  const steps = rows[0].length;
  const buf = new ArrayBuffer(rows.length * steps * 4);
  const view = new DataView(buf);
  rows.forEach((row, p) => row.forEach((v, n) => view.setFloat32((p * steps + n) * 4, v, true)));
  return buf;
}

test('decodeTraces reads probes x steps in C order, little endian', () => {
  const rows = [[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]];
  const got = tdplot.decodeTraces(traceBuffer(rows), [2, 5]).map((r) => Array.from(r));
  assert.deepEqual(got, rows);
});

test('decodeTraces refuses a buffer the manifest shape does not fit', () => {
  const buf = traceBuffer([[1, 2, 3, 4, 5], [10, 20, 30, 40, 50]]);
  assert.throws(() => tdplot.decodeTraces(buf, [2, 6]), /40 bytes.*48/);
});

test('envelope keeps each bucket min and max, and covers every sample once', () => {
  // A CW trace of period 2 samples: plain decimation to 4 columns of 40
  // samples would land on the same phase every time and draw a flat line.
  const values = Array.from({ length: 160 }, (_, i) => (i % 2 ? -1 : 1));
  const env = tdplot.envelope(values, 4);
  assert.equal(env.length, 4);
  for (const b of env) assert.deepEqual([b.min, b.max], [-1, 1]);
  assert.equal(env[0].i0, 0);
  assert.equal(env.at(-1).i1, 160);
  for (let c = 1; c < env.length; c += 1) assert.equal(env[c].i0, env[c - 1].i1);
});

test('envelope with more columns than samples gives one bucket per sample', () => {
  const env = tdplot.envelope([3, -2, 7], 10);
  assert.deepEqual(env.map((b) => [b.min, b.max]), [[3, 3], [-2, -2], [7, 7]]);
});

test('probeLabel names the component and the position in a suitable unit', () => {
  const label = tdplot.probeLabel({ component: 'z', position_m: [1.5e-6, 0, -2.25e-6] }, 0);
  assert.equal(label, 'probe 1: Ez at (1.50, 0.00, -2.25) µm');
});

test('axisExponent pulls out a power of ten only when plain ticks would not read', () => {
  assert.equal(tdplot.axisExponent(2.4e-13), -13);
  assert.equal(tdplot.axisExponent(9.9e-12), -12);
  assert.equal(tdplot.axisExponent(3.2e5), 5);
  assert.equal(tdplot.axisExponent(0.5), 0);
  assert.equal(tdplot.axisExponent(250), 0);
  assert.equal(tdplot.axisExponent(0), 0);
});

test('exponentText writes the factor with superscripts', () => {
  assert.equal(tdplot.exponentText(-13), '×10⁻¹³');
  assert.equal(tdplot.exponentText(5), '×10⁵');
  assert.equal(tdplot.exponentText(0), '');
});
