// Tests for the field-movie helpers in tdmovie.js. Run with:
//   node --test tests/js/*.mjs
//
// Frames are NON-SQUARE (2 x 3 x 1, 2 x 1 x 3) so a transposed read cannot
// agree by accident, and every expected number is worked out by hand.

import test from 'node:test';
import assert from 'node:assert/strict';
import * as movie from '../../web/static/tdmovie.js';

function frameBuffer(values) {
  const buf = new ArrayBuffer(values.length * 4);
  const view = new DataView(buf);
  values.forEach((v, i) => view.setFloat32(i * 4, v, true));
  return buf;
}

const close = (got, want) => {
  assert.equal(got.length, want.length);
  want.forEach((w, i) => assert.ok(Math.abs(got[i] - w) < 1e-6, `[${i}] ${got[i]} != ${w}`));
};

test('decodeFrame reads little-endian float32 in C order, z fastest', () => {
  // (ix, iz) -> 10*ix + iz at index ix*3 + iz
  const f = movie.decodeFrame(frameBuffer([0, 1, 2, 10, 11, 12]), [2, 1, 3]);
  assert.deepEqual(Array.from(f), [0, 1, 2, 10, 11, 12]);
  assert.throws(() => movie.decodeFrame(frameBuffer([0, 1, 2]), [2, 1, 3]),
    /12 bytes, expected 24/);
});

test('latticeAxis keeps the finest gap, never drops samples, and caps', () => {
  assert.deepEqual(movie.latticeAxis([0, 1, 1.5, 3]), { lo: 0, step: 0.5, count: 7 });
  assert.deepEqual(movie.latticeAxis([0, 1, 1.5, 3], 4), { lo: 0, step: 1, count: 4 });
  assert.deepEqual(movie.latticeAxis([2e-6]), { lo: 2e-6, step: 0, count: 1 });
});

test('axisTable names the graded sample below each lattice point and its weight', () => {
  const positions = [0, 1, 1.5, 3];
  const t = movie.axisTable(positions, movie.latticeAxis(positions));
  assert.deepEqual(Array.from(t.below), [0, 0, 1, 2, 2, 2, 2]);
  close(t.weight, [0, 0.5, 0, 0, 1 / 3, 2 / 3, 1]);
});

test('resampleFrame writes texture order, x fastest, from a C-order frame', () => {
  // (ix, iy) -> 10*ix + iy; frame index ix*3 + iy (sz = 1)
  const lat = movie.movieLattice([[0, 1], [0, 1, 2], [0]]);
  assert.deepEqual([lat.grid.nx, lat.grid.ny, lat.grid.nz], [2, 3, 1]);
  const out = movie.resampleFrame(Float32Array.from([0, 1, 2, 10, 11, 12]), lat);
  assert.deepEqual(Array.from(out), [0, 10, 1, 11, 2, 12]);
});

test('resampleFrame places graded samples at their true positions', () => {
  // x samples at 0, 1, 3 hold 0, 4, 8: the even lattice 0, 1, 2, 3 reads
  // 0, 4, 6, 8 -- an index-spaced draw would read 0, 4, 8 at 0, 1.5, 3.
  const lat = movie.movieLattice([[0, 1, 3], [5e-7], [0]]);
  assert.deepEqual(lat.grid.spacing, [1, 0, 0]);
  close(movie.resampleFrame(Float32Array.from([0, 4, 8]), lat), [0, 4, 6, 8]);
});

test('movieScale is one high percentile shared by every frame', () => {
  const ramp = Float32Array.from({ length: 100 }, (_, i) => i + 1);   // 1..100
  assert.equal(movie.movieScale([ramp]), 99);
  // a spike in one frame does not set the scale, and the quiet frame shares it
  const quiet = Float32Array.from({ length: 100 }, (_, i) => (i + 1) / 100);
  const loud = Float32Array.from(ramp); loud[0] = 1e6;
  const s = movie.movieScale([quiet, loud]);
  assert.equal(s, 99);
  assert.equal(movie.movieScale([new Float32Array(4)]), null);
});

test('scaledFrame divides by the shared scale and clamps at 1', () => {
  close(movie.scaledFrame(Float32Array.from([0, 5, 10, 40]), 10), [0, 0.5, 1, 1]);
  close(movie.scaledFrame(Float32Array.from([3]), null), [0]);
});
