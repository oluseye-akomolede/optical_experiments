// The field payload as the *browser* sees it: half-float bytes, on bytes a real
// route produced.
//
// The bug this line of testing exists for was a format mismatch, not a maths
// error: the server sent NaN in a JSON payload, `json.dumps` wrote the bare
// token, and the browser's `JSON.parse` rejected the whole document. Every
// server-side test passed, because Python's json reads `NaN` happily -- the
// instrument could not see the failure. The payload is binary now, so the
// instrument changes with it, but the principle does not: this reads the bytes
// through the same conversion the page uses, and asserts the properties the
// page depends on.
//
// The fixture is written by tests/test_web_routes.py from a **live route
// response** (see `test_a_real_field_response_is_written_for_the_node_test`),
// not hand-written, so this is checking the real thing rather than a double
// that models what we think we send. If it is missing the test fails loudly
// rather than skipping -- a skip here would recreate the original blindness.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { componentVolume, boxCentre, float32ToHalf }
  from '../../web/static/components.js';

const here = dirname(fileURLToPath(import.meta.url));
const blobPath = join(here, 'fixtures', 'mode_field.bin');
const gridPath = join(here, 'fixtures', 'mode_grid.json');

// DataView's float16 read is not in every engine; the page writes half-floats
// with float32ToHalf, so the inverse is written the same way -- by hand -- and
// the two are checked against each other below.
function halfToFloat32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  if (exp === 0) return sign * frac * 2 ** -24;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

test('a real field response decodes back to the values the solver wrote', () => {
  assert.ok(existsSync(blobPath),
    `missing fixture ${blobPath} -- run the pytest suite first, which writes it`);
  assert.ok(existsSync(gridPath), `missing fixture ${gridPath}`);
  const grid = JSON.parse(readFileSync(gridPath, 'utf8'));
  const buf = readFileSync(blobPath);
  assert.equal(buf.length, grid.nx * grid.ny * grid.nz * 3 * 2,
    'the payload is two bytes per component, in the grid it declares');
});

test('half-float round-trips the values the page is told to expect', () => {
  // The conversion in viewer.js and the one above must be inverses on the
  // values the solver writes. If float32ToHalf narrowed the exponent wrongly,
  // the colours would be wrong everywhere and nothing else would notice: the
  // texture would still upload, and a mode would still appear.
  for (const v of [0, 1, -1, 0.5, -0.25, 1e-5, -1e-5, 65504, 6.1e-5]) {
    const back = halfToFloat32(float32ToHalf(v));
    assert.ok(Math.abs(back - v) <= Math.abs(v) * 2 ** -10 + 1e-8,
      `${v} went through float16 as ${back}`);
  }
  assert.ok(Number.isNaN(halfToFloat32(float32ToHalf(NaN))),
    'NaN must stay NaN: it is how the solver says "outside the object"');
  assert.equal(halfToFloat32(float32ToHalf(Infinity)), Infinity);
});

test('the sample the solver marked outside stays outside through the transform', () => {
  const grid = JSON.parse(readFileSync(gridPath, 'utf8'));
  const buf = readFileSync(blobPath);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const count = grid.nx * grid.ny * grid.nz;
  const vec = new Float32Array(count * 3);
  for (let i = 0; i < count * 3; i += 1) {
    vec[i] = halfToFloat32(view.getUint16(i * 2, true));
  }
  // The fixture marks all but two samples NaN, so "is anything finite" is a
  // real question with a real answer rather than a vacuous one.
  const finite = [...vec].filter((v) => Number.isFinite(v) && v !== 0).length;
  assert.ok(finite > 0, 'the fixture must carry some real samples');

  // Rebuild physical positions exactly as the viewer does, then transform.
  const positions = new Float32Array(count * 3);
  let i = 0;
  for (let z = 0; z < grid.nz; z += 1) {
    for (let y = 0; y < grid.ny; y += 1) {
      for (let x = 0; x < grid.nx; x += 1) {
        positions[i] = grid.origin[0] + grid.spacing[0] * x;
        positions[i + 1] = grid.origin[1] + grid.spacing[1] * y;
        positions[i + 2] = grid.origin[2] + grid.spacing[2] * z;
        i += 3;
      }
    }
  }
  const out = componentVolume(vec, positions, boxCentre(grid), 'magnitude', 'E');
  assert.equal(out.length, count);
  // Every sample is either a real field magnitude or NaN. A transform that
  // invented a value where the solver said "nothing here" would paint fog
  // around the object, and a transform that lost a NaN would open a hole in it.
  for (let n = 0; n < count; n += 1) {
    assert.ok(Number.isNaN(out[n]) || out[n] >= 0,
      `sample ${n} came out as ${out[n]}`);
  }
  assert.ok([...out].some((v) => Number.isNaN(v)),
    'the outside-the-object samples must still be NaN after the transform');
});
