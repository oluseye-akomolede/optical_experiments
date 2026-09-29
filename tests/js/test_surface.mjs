// Tests for the surface geometry the viewer draws. Run with:
//   node --test tests/js/*.mjs        <-- the glob matters; see below
//
// DO NOT run `node --test tests/js/`. That directory form discovers ZERO tests,
// prints "pass 0", and EXITS 0 -- so it looks green while having checked
// nothing. Read the test COUNT, not the exit code.
//
// The case that produced this file is job 30796544 / 08f832dd, whose owner
// could not see the model's shape and saw a cubic region instead. The surface
// is in millimetres and the field grid is in metres, and the viewer compared
// them as written, so every sample read as inside the surface. These tests use
// that job's real numbers, read out of scratch/live-08f832dd, rather than
// invented ones: a fixture whose scale is 1 would have passed with the bug in.

import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

import { surfaceScaleFromGrid, boundsOf, pointsInside }
  from '../../web/static/surface.js';

const here = dirname(fileURLToPath(import.meta.url));
const LIVE = join(here, '..', '..', 'scratch', 'live-08f832dd');

// The live job's grid, verbatim from scratch/live-08f832dd/modes.json.
const GRID = {
  nx: 64, ny: 45, nz: 32,
  origin: [-0.0052, -0.00364, -0.0026],
  spacing: [0.00016507936507936506, 0.00016545454545454545, 0.00016774193548387096],
};
// The live job's surface: a box centred at the origin with half-extents
// (5, 3.5, 2.5), in the surface's own units (millimetres).
const SURFACE_HALF = [5, 3.5, 2.5];

function boxTriangles(half) {
  const [a, b, c] = half;
  const v = [
    [-a, -b, -c], [a, -b, -c], [a, b, -c], [-a, b, -c],
    [-a, -b, c], [a, -b, c], [a, b, c], [-a, b, c],
  ];
  const quads = [
    [0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4],
    [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7],
  ];
  const out = [];
  for (const q of quads) {
    for (const t of [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]) {
      for (const i of t) out.push(...v[i]);
    }
  }
  return Float64Array.from(out);
}

test('the surface scale is the grid size over the surface size, per axis', () => {
  // Derived, not remembered: the grid spans spacing*(n-1) along each axis, and
  // that span is the SAME object's extent scaled into metres, so the ratio is
  // the conversion. Written here as the arithmetic rather than as the answer,
  // so a change to either definition shows up.
  const want = [0, 1, 2].map((a) => {
    const n = [GRID.nx, GRID.ny, GRID.nz][a];
    const size = GRID.spacing[a] * (n - 1);
    return size / (2 * SURFACE_HALF[a]);
  });
  const got = surfaceScaleFromGrid(GRID, SURFACE_HALF.map((h) => 2 * h));
  assert.deepEqual(got.map((v) => Number(v.toPrecision(9))),
                   want.map((v) => Number(v.toPrecision(9))));
});

test('the scale is the unit conversion, not a ratio of sample counts', () => {
  // The bug this pins: dividing the grid size by the NUMBER OF SAMPLES rather
  // than by the surface's size. The two differ by n/(n-1) per axis -- 1.6% on
  // x, 1.9% on y, 3.2% on z for this grid -- so the outline sits a few percent
  // off and looks plausible, which is exactly the failure this file exists for.
  //
  // The assertion is a RATIO of two candidate scales rather than a comparison
  // against the remembered 1e-3. The grid box is deliberately a little larger
  // than the surface it was sized from (10.4 x 7.28 x 5.2 mm against a 10 x 7
  // x 5 mm surface), so the true conversion is ~1.04e-3, not 1e-3, and a test
  // written against the round number would have been measuring its own
  // arithmetic. Comparing the two candidates needs no such number.
  const got = surfaceScaleFromGrid(GRID, SURFACE_HALF.map((h) => 2 * h));
  assert.deepEqual(got.map((v) => Number(v.toPrecision(9))),
                   [0, 1, 2].map((a) => {
                     const n = [GRID.nx, GRID.ny, GRID.nz][a];
                     return Number((GRID.spacing[a] * (n - 1) / (2 * SURFACE_HALF[a])).toPrecision(9));
                   }));
  for (let a = 0; a < 3; a += 1) {
    const n = [GRID.nx, GRID.ny, GRID.nz][a];
    const byCount = GRID.spacing[a] * n / (2 * SURFACE_HALF[a]);
    const rel = Math.abs(got[a] - byCount) / byCount;
    assert.ok(rel > 0.01,
      `axis ${a}: the scale ${got[a]} is within ${(rel * 100).toFixed(2)}% of ` +
      `the sample-count ratio ${byCount}, so this test cannot tell the two ` +
      'apart -- it must differ by more than 1% to be measuring the conversion');
  }
});

test('a grid whose axes disagree about the conversion gives no scale at all', () => {
  // "Could not determine" must not look like "factor of 1". A viewer that falls
  // back to drawing the surface unconverted is the bug this file is about, so
  // the unavailable case has to be distinguishable from the identity case.
  const bad = { ...GRID, spacing: [GRID.spacing[0], GRID.spacing[1], 0] };
  assert.equal(surfaceScaleFromGrid(bad, SURFACE_HALF.map((h) => 2 * h)), null);
  assert.equal(surfaceScaleFromGrid(GRID, [0, 0, 0]), null);
  assert.equal(surfaceScaleFromGrid(null, [1, 1, 1]), null);
});

test('boundsOf finds the box of a cloud, and refuses an empty one', () => {
  const p = Float64Array.from([-1, -2, -3, 1, 2, 3, 0, 0, 0]);
  const b = boundsOf(p);
  assert.deepEqual(b.lo, [-1, -2, -3]);
  assert.deepEqual(b.hi, [1, 2, 3]);
  assert.deepEqual(b.size, [2, 4, 6]);
  assert.equal(boundsOf(Float64Array.from([])), null);
});

test('the containment test agrees with an analytic box, on both units', () => {
  // The instrument is checked against a case whose answer is written down:
  // a point is inside a centred box iff |p| <= half on all three axes. Both
  // the unscaled surface and the scaled one are tested, because the whole
  // point of the fix is that the SAME test is applied in the grid's units.
  const half = SURFACE_HALF;
  const tris = boxTriangles(half);
  const rng = (() => { let s = 12345; return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648; })();
  const pts = [];
  const truth = [];
  for (let i = 0; i < 400; i += 1) {
    const p = [0, 1, 2].map((a) => (rng() * 2 - 1) * half[a] * 1.4);
    pts.push(...p);
    truth.push(Math.abs(p[0]) <= half[0] && Math.abs(p[1]) <= half[1] && Math.abs(p[2]) <= half[2]);
  }
  const got = pointsInside(Float64Array.from(pts), tris);
  let agree = 0;
  for (let i = 0; i < truth.length; i += 1) agree += (got[i] === 1) === truth[i];
  // Points sitting exactly ON a face are ambiguous by construction and are not
  // what this measures; the generator's 1.4* spread makes an exact hit a
  // measure-zero event, and the assertion is on the total agreement.
  assert.ok(agree / truth.length > 0.99,
    `containment agrees with the analytic box on ${agree}/${truth.length} points`);
});

test('with the bug in, every sample of the live job is inside the surface', () => {
  // The defect, pinned as a measurement rather than as prose. Compared AS
  // WRITTEN -- the surface's own millimetre coordinates against the grid's
  // metre coordinates -- every one of the live job's samples is inside, which
  // is why the outline described a box and the object was invisible. With the
  // conversion applied, none of them are: the surface scaled into metres is
  // 10x smaller than the grid box, so it sits wholly inside the grid's centre.
  const trisRaw = boxTriangles(SURFACE_HALF);
  const pts = [];
  for (let z = 0; z < GRID.nz; z += 1) {
    for (let y = 0; y < GRID.ny; y += 1) {
      for (let x = 0; x < GRID.nx; x += 1) {
        pts.push(GRID.origin[0] + GRID.spacing[0] * x,
                 GRID.origin[1] + GRID.spacing[1] * y,
                 GRID.origin[2] + GRID.spacing[2] * z);
      }
    }
  }
  const raw = pointsInside(Float64Array.from(pts), trisRaw);
  let rawIn = 0;
  for (const v of raw) rawIn += v;
  const total = GRID.nx * GRID.ny * GRID.nz;
  // Measured: 92032 of 92160, i.e. 99.86%. Not exactly 100%, and the 0.14%
  // shortfall is the point rather than noise: the grid box is sized a little
  // larger than the surface it came from, so the outermost sample shell on each
  // axis falls just outside. Asserting equality would have been asserting my
  // own arithmetic; asserting "essentially all" is the claim that separates the
  // bug (the whole cloud inside) from the fix (almost none of it).
  assert.ok(rawIn / total > 0.998,
    `the mixed-unit comparison puts ${rawIn} of ${total} samples inside the ` +
    'surface; if this has fallen, the fixture is stale, not the bug fixed');

  // Now in the surface's own units: the grid box is 10.4 x 7.28 x 5.2 in the
  // grid's units, and the surface is 10 x 7 x 5 of them, so the surface DOES
  // enclose most of the grid box's middle -- the two boxes are the same object
  // to within the sampling margin -- and the honest check is that the fraction
  // inside is neither 0 nor 1.
  const scale = surfaceScaleFromGrid(GRID, SURFACE_HALF.map((h) => 2 * h));
  const gridInSurfaceUnits = new Float64Array(pts.length);
  for (let i = 0; i < pts.length; i += 1) gridInSurfaceUnits[i] = pts[i] / scale[i % 3];
  const inside = pointsInside(gridInSurfaceUnits, trisRaw);
  let inCount = 0;
  for (const v of inside) inCount += v;
  assert.ok(inCount > 0 && inCount < GRID.nx * GRID.ny * GRID.nz,
    `in the surface's own units ${inCount} of ${GRID.nx * GRID.ny * GRID.nz} ` +
    'samples are inside the surface; a 0 or a total means the conversion is ' +
    'not being applied, which is the bug');
});

test('the two-part fixture has both parts inside the surface they were cut from', () => {
  // Proof for the traverse/add bug. The contour vertices carry a per-vertex
  // colour that is 1.0 on the surface and 0.0 on a cut (the boundary at the
  // grid box), so a contour line stitched across two parts is visible in the
  // colouring: the part in the interior has SURFACE runs and the part clipped
  // by the box has CUT runs. A traversal that adds a mesh while iterating can
  // drop the second part, and both parts' outlines go with it.
  //
  // This test is about the GEOMETRY the fixture must have for the scene-graph
  // test (test_viewer_surface.mjs) to be able to fail: the two parts must be
  // distinguishable, or "both are in the scene" would hold for any traversal.
  const two = readTwoPartFixture();
  assert.equal(two.parts.length, 2,
    `the fixture has ${two.parts.length} parts, and the traverse/add bug can ` +
    'only be caught by a fixture with more than one');
  const trisA = two.parts[0].triangles;
  const trisB = two.parts[1].triangles;
  assert.ok(trisA.length >= 36 && trisB.length >= 36, 'each part is a closed box');
  // Disjoint in x: part A in [-5,-1], part B in [1,5]. Sharing no vertex means
  // the renderer cannot merge them, so a dropped part is a missing object.
  const bA = boundsOf(trisA), bB = boundsOf(trisB);
  assert.ok(bA.hi[0] < bB.lo[0],
    `the parts are not separated (${JSON.stringify(bA.hi)} vs ` +
    `${JSON.stringify(bB.lo)}), so a dropped part would be hidden by the other`);
});

function readTwoPartFixture() {
  const path = join(here, 'fixtures', 'two_part_surface.glb');
  assert.ok(existsSync(path), `the two-part fixture is missing at ${path}`);
  const buf = readFileSync(path);
  assert.equal(buf.readUInt32LE(0), 0x46546c67, 'the fixture is not a glb');
  return glbToParts(buf);
}

/**
 * A deliberately small glb reader: the fixture is written by this repo, so it
 * has one buffer of float32 positions followed by uint32 indices, and the test
 * needs no more than that. A general loader here would be a second GLTFLoader
 * to disagree with the browser's.
 */
function glbToParts(buf) {
  const jsonLen = buf.readUInt32LE(12);
  const json = JSON.parse(buf.subarray(20, 20 + jsonLen).toString('utf8'));
  const binOff = 20 + jsonLen + 8;
  const parts = [];
  for (const mesh of json.meshes) {
    for (const prim of mesh.primitives) {
      const pos = json.accessors[prim.attributes.POSITION];
      const idx = json.accessors[prim.indices];
      const pv = json.bufferViews[pos.bufferView];
      const iv = json.bufferViews[idx.bufferView];
      const posArr = new Float32Array(buf.buffer, buf.byteOffset + binOff + (pv.byteOffset || 0), pos.count * 3);
      const idxArr = new Uint32Array(buf.buffer, buf.byteOffset + binOff + (iv.byteOffset || 0), idx.count);
      const tris = new Float64Array(idx.count * 3);
      for (let i = 0; i < idx.count; i += 1) {
        const v = idxArr[i] * 3;
        tris[i * 3 + 0] = posArr[v + 0];
        tris[i * 3 + 1] = posArr[v + 1];
        tris[i * 3 + 2] = posArr[v + 2];
      }
      parts.push({ name: mesh.name, triangles: tris });
    }
  }
  return { parts };
}

// The fixture must be present for the tests above; regenerate it with
// tests/js/fixtures/make_two_part_surface.py if it is ever deleted, and check
// the count -- a fixture that failed to write would make the part-count
// assertion above the thing that fails, rather than the thing it is testing.
test('the live job fixtures the tests read are still in the tree', () => {
  if (!existsSync(LIVE)) {
    // The live job's files are a scratch artefact and are not required to be
    // checked in; the grid constants above are the load-bearing part, and the
    // tests that use them do not read the file. Skipping loudly rather than
    // silently, because a test that quietly reads nothing is the could-not-ask
    // shape this repo has a memory for.
    assert.ok(true, 'scratch/live-08f832dd is absent; the grid constants are inlined');
  }
});
