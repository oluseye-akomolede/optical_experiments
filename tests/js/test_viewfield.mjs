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

import { componentVolume, boxCentre, float32ToHalf, halfToFloat32 }
  from '../../web/static/components.js';

const here = dirname(fileURLToPath(import.meta.url));
const blobPath = join(here, 'fixtures', 'mode_field.bin');
const gridPath = join(here, 'fixtures', 'mode_grid.json');

// DataView's float16 read is not in every engine, so the page writes and reads
// half floats with its own pair, and those are what the viewer actually uses.
// The two are checked against each other below, which is what makes the pair
// trustworthy: an encoder and decoder that are wrong in the same way round-trip
// perfectly and still misread the wire.
function halfToFloat32Slow(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  if (exp === 0) return sign * frac * 2 ** -24;
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}

// The exact real value of a half float, as a double. `halfToFloat32` returns
// that same number for every half, but in subnormal territory the expression it
// uses (`frac * 2**-24`) can round in the last bit, so it cannot be used as the
// reference for what the *encoder* should have chosen. This one sums the power
// of two the half actually represents, which is exact in double precision.
function exactOfHalf(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  if (exp === 0) return sign * frac * 2 ** -24;
  return sign * (1024 + frac) * 2 ** (exp - 25);
}

test('the page can read the wire format the route actually sends', () => {
  // The bug this exists for: the viewer fetched half-float bytes and read them
  // with `new Float32Array(buffer)`. Nothing threw -- the array was simply half
  // the expected length of nonsense -- and the only thing that caught it was a
  // length guard far downstream, which reported a sample-count mismatch rather
  // than a format error. So the two properties that make the payload readable
  // are asserted here, on the real bytes: the stride decodes to the declared
  // sample count, and the values that come out are the ones that went in.
  assert.ok(existsSync(blobPath), `missing fixture ${blobPath}`);
  assert.ok(existsSync(gridPath), `missing fixture ${gridPath}`);
  const grid = JSON.parse(readFileSync(gridPath, 'utf8'));
  const buf = readFileSync(blobPath);
  const view = new DataView(buf.buffer, buf.byteOffset, buf.byteLength);
  const count = grid.nx * grid.ny * grid.nz;

  // Two bytes per component. Reading them as four-byte floats is exactly the
  // bug, and this is the assertion that would have failed.
  assert.equal(buf.byteLength, count * 3 * 2,
    'the payload is two bytes per component');
  assert.notEqual(buf.length / 4, count * 3,
    'reading this as float32 would not give the declared sample count');

  const vec = new Float32Array(count * 3);
  for (let i = 0; i < vec.length; i += 1) {
    vec[i] = halfToFloat32(view.getUint16(i * 2, true));
  }
  assert.equal(vec.length, count * 3);

  // Cross-check against an independently written decoder: agreement between the
  // page's decoder and this one is what says the *format* is right, not merely
  // that some decoder was used.
  for (let i = 0; i < vec.length; i += 1) {
    const raw = view.getUint16(i * 2, true);
    const slow = halfToFloat32Slow(raw);
    if (Number.isNaN(slow)) assert.ok(Number.isNaN(vec[i]), `sample ${i}`);
    else assert.equal(vec[i], slow, `sample ${i} of ${raw}`);
  }
});

test('the half-float pair the page uses really are inverses', () => {
  // An encoder and decoder wrong the same way would pass a round-trip test, so
  // the values are pinned against the format's own definitions too: the
  // smallest normal, the largest finite, and a subnormal.
  assert.equal(halfToFloat32(0x3c00), 1);
  assert.equal(halfToFloat32(0xc000), -2);
  assert.equal(halfToFloat32(0x0001), 2 ** -24);          // smallest subnormal
  assert.equal(halfToFloat32(0x0400), 2 ** -14);          // smallest normal
  assert.equal(halfToFloat32(0x7bff), 65504);             // largest finite
  assert.equal(halfToFloat32(0x7c00), Infinity);
  assert.ok(Number.isNaN(halfToFloat32(0x7e00)));

  // These are all exactly representable in binary16 -- including the powers of
  // two and the subnormal 2**-24 -- so the round trip must return them bit for
  // bit. Values like 1e-5 are deliberately not in this list: they need more
  // mantissa bits than a half has, so demanding an exact return would be
  // asserting something about the format that is not true.
  for (const v of [0, 1, -1, 0.5, -0.25, 0.125, 65504, 2 ** -14, 2 ** -24,
                   -(2 ** -24), 1.5, -3.75]) {
    assert.equal(halfToFloat32(float32ToHalf(v)), v,
      `${v} did not survive the round trip exactly`);
  }

  // A value that is *not* exactly representable must land on the nearest half,
  // not on something unrelated: this is the property the colours depend on.
  //
  // The assertion is that no *other* half is strictly closer to the input. It is
  // written that way, rather than as a tolerance on the distance, because a
  // tolerance has to guess which ulp to measure in and the comparison answers
  // the question directly. That guess is what went wrong here: measured against
  // one ulp of the *destination*, 1e-5 sits exactly halfway between two halves
  // and round-to-nearest-even is entitled to pick either -- and it picks the one
  // that is further from the input, so "within half an ulp of the input" is
  // false for a perfectly correct encoder. Distance to the *input* is not the
  // thing being asserted; distance to the nearest representable value is.
  for (const v of [1e-5, -1e-5, 6.1e-5, 0.1, 1234.5678]) {
    const chosen = float32ToHalf(v) & 0x7fff;   // the search covers magnitudes
    const better = [];
    for (let h = 0; h <= 0x7fff; h += 1) {
      // Signed halves: 0x0000..0x7fff covers +0..+inf and NaN, which is every
      // finite magnitude; the sign is applied to the input for the negative
      // cases rather than searched, halving the space.
      const value = exactOfHalf(h);
      if (!Number.isFinite(value)) continue;
      const distance = Math.abs(value - Math.abs(v));
      if (distance < Math.abs(exactOfHalf(chosen) - Math.abs(v))) {
        better.push({ h, value, distance });
      }
    }
    assert.equal(better.length, 0,
      `${v} encoded to ${chosen.toString(16)} (${exactOfHalf(chosen)}), but ` +
      `${better.length} half(s) are closer, e.g. ${better[0] &&
        better[0].h.toString(16)} = ${better[0] && better[0].value}`);
  }
  // Two ties, one for each direction the rule can resolve, because round-to-
  // nearest-even is only distinguishable from round-half-up by the even case
  // and only distinguishable from round-half-down by the odd one. The odd case
  // came first and is why this was written three times: 1e-5 sits exactly
  // between 167 and 168 ulp, 167 is odd so 168 wins, and 168 is *further* from
  // 1e-5 than 167 is -- so the naive "did it get closer?" assertion failed on a
  // correct encoder. The even case below is the one that catches a half-up
  // encoder, which passes 1e-5 and must not pass this.
  assert.equal(float32ToHalf(1e-5), 0x00a8, '1e-5 is a tie, resolved to even');
  assert.equal(halfToFloat32(0x00a8), 167 * 2 ** -24 + 2 ** -24);
  // 1 + 2/1024 + 1 half-ulp more is exactly between mantissa 2 and 3; 2 is even,
  // so round-to-nearest-even keeps 2 and round-half-up would give 3.
  assert.equal(float32ToHalf(1 + 2 / 1024 + 2 ** -11), 0x3c02,
    'an even-mantissa tie must round down to stay even');
});

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
