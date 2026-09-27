// Tests for the vector-to-scalar transform in components.js. Run with:
//   node --test tests/js/
//
// The cases are fields whose answer can be written down by hand, because the
// two ways this can be wrong are both silent: reading a grid position off the
// wrong axis, and letting the transform invent a NaN where the physics has
// none. A colour plot of a wrong-but-plausible field still looks like a picture
// of a mode, so nothing downstream would catch either.

import test from 'node:test';
import assert from 'node:assert/strict';

import { componentOf, componentVolume, boxCentre, SYSTEMS, SYSTEM_NAMES,
         COMPONENT_NAMES, SYSTEM_LABELS, COMPONENT_LABELS,
         DEFAULT_SYSTEM, DEFAULT_COMPONENT }
  from '../../web/static/components.js';

const C = [0, 0, 0];

test('the box centre is half a box up from the origin, per axis', () => {
  const grid = { nx: 5, ny: 3, nz: 3, origin: [1, 2, 3], spacing: [2, 4, 6] };
  // x: 1 + 2*(5-1)/2 = 5 ; y: 2 + 4*(3-1)/2 = 6 ; z: 3 + 6*(3-1)/2 = 9
  assert.deepEqual(boxCentre(grid), [5, 6, 9]);
});

test('cartesian components are the vector itself', () => {
  const v = [0.25, -0.5, 0.75];
  assert.equal(componentOf(v, [0, 0, 0], C, 'cartesian', 'Ex'), 0.25);
  assert.equal(componentOf(v, [0, 0, 0], C, 'cartesian', 'Ey'), -0.5);
  assert.equal(componentOf(v, [0, 0, 0], C, 'cartesian', 'Ez'), 0.75);
});

test('|E| is the vector norm and is never negative', () => {
  assert.equal(componentOf([3, 4, 0], [0, 0, 0], C, 'magnitude', 'E'), 5);
  assert.equal(componentOf([-3, -4, 0], [0, 0, 0], C, 'magnitude', 'E'), 5);
});

test('a purely radial field has Er = |E| and no transverse part', () => {
  // A field pointing straight out from the centre: at (2,0,0) it is (5,0,0).
  const pos = [2, 0, 0];
  const v = [5, 0, 0];
  assert.equal(componentOf(v, pos, C, 'spherical', 'Er'), 5);
  assert.equal(componentOf(v, pos, C, 'spherical', 'Etheta'), 0);
  assert.equal(componentOf(v, pos, C, 'spherical', 'Ephi'), 0);
});

test('a field along +z projects onto Er and Etheta as cos and -sin of theta', () => {
  // At 60 degrees from +z: cos(60)=0.5 along r, and e_theta points away from
  // +z, so the projection is -sin(60) = -0.8660.
  const r = 2;
  const th = Math.PI / 3;
  const pos = [r * Math.sin(th), 0, r * Math.cos(th)];
  const v = [0, 0, 1];
  const Er = componentOf(v, pos, C, 'spherical', 'Er');
  const Eth = componentOf(v, pos, C, 'spherical', 'Etheta');
  assert.ok(Math.abs(Er - Math.cos(th)) < 1e-6, `Er ${Er} != cos(theta)`);
  assert.ok(Math.abs(Eth - -Math.sin(th)) < 1e-6, `Etheta ${Eth} != -sin(theta)`);
});

test('on the polar axis the transverse components are 0, not NaN', () => {
  // rho = 0 here. A division would give NaN and punch a hole in the plot that
  // is not in the physics -- the object would appear to have a thread of
  // nothing through it.
  const pos = [0, 0, 1];
  const v = [0, 0, 0.5];
  for (const name of ['Er', 'Etheta', 'Ephi']) {
    const got = componentOf(v, pos, C, 'spherical', name);
    assert.ok(Number.isFinite(got), `spherical ${name} on axis gave ${got}`);
  }
  for (const name of ['Erho', 'Ephi']) {
    const got = componentOf(v, pos, C, 'cylindrical', name);
    assert.ok(Number.isFinite(got), `cylindrical ${name} on axis gave ${got}`);
  }
});

test('cylindrical Ez is the cartesian Ez, wherever the sample sits', () => {
  const v = [1, 1, 0.42];
  assert.equal(componentOf(v, [3, -2, 7], C, 'cylindrical', 'Ez'), 0.42);
});

test('a purely azimuthal field reads as Ephi with no radial part', () => {
  // At (1,0,0), the phi direction is +y. A field of (0, 3, 0) there is purely
  // azimuthal: Ephi = 3, Erho = 0.
  const pos = [1, 0, 0];
  const v = [0, 3, 0];
  assert.ok(Math.abs(componentOf(v, pos, C, 'cylindrical', 'Ephi') - 3) < 1e-6);
  assert.ok(Math.abs(componentOf(v, pos, C, 'cylindrical', 'Erho')) < 1e-6);
});

test('the transform never turns a good sample into NaN', () => {
  // Sweep a sample grid, including the exact centre and the axes, over every
  // offered component. Only the solver may produce NaN.
  const centre = [0, 0, 0];
  const values = [-1, -0.5, 0, 0.5, 1];
  let checked = 0;
  for (const x of values) {
    for (const y of values) {
      for (const z of values) {
        const pos = [x, y, z];
        const v = [x + 0.1, y - 0.2, z + 0.3];
        for (const [system, names] of Object.entries(SYSTEMS)) {
          for (const name of names) {
            const got = componentOf(v, pos, centre, system, name);
            assert.ok(Number.isFinite(got),
                      `${system}.${name} at ${pos} gave ${got}`);
            checked += 1;
          }
        }
      }
    }
  }
  assert.ok(checked > 100, `only ${checked} cases checked`);
});

test('the solver\'s NaN survives every component', () => {
  // NaN outside the object is the solver saying "no sample here". The plot
  // renders it transparent, so losing it would paint field where there is none.
  const v = [1, NaN, 0.5];
  for (const [system, names] of Object.entries(SYSTEMS)) {
    for (const name of names) {
      const got = componentOf(v, [1, 2, 3], C, system, name);
      assert.ok(Number.isNaN(got), `${system}.${name} lost the NaN`);
    }
  }
});

test('componentVolume reads positions and vectors on the same sample', () => {
  // Two samples: one pointing along +x at (1,0,0), one along +y at (0,1,0).
  // Erho is 1 for both -- but only if position and vector stay paired. If the
  // stride were wrong, the second would be measured with the first's vector.
  const vec = [1, 0, 0, 0, 1, 0];
  const pos = [1, 0, 0, 0, 1, 0];
  const out = componentVolume(vec, pos, C, 'cylindrical', 'Erho');
  assert.equal(out.length, 2);
  assert.ok(Math.abs(out[0] - 1) < 1e-6);
  assert.ok(Math.abs(out[1] - 1) < 1e-6);
});

test('componentVolume keeps NaN where the solver put it', () => {
  const vec = [1, 0, 0, NaN, NaN, NaN];
  const pos = [1, 0, 0, 2, 0, 0];
  const out = componentVolume(vec, pos, C, 'cartesian', 'Ex');
  assert.equal(out[0], 1);
  assert.ok(Number.isNaN(out[1]));
});

test('the offered component list is complete and has no duplicates', () => {
  assert.deepEqual(Object.keys(SYSTEMS),
                   ['cartesian', 'cylindrical', 'spherical', 'magnitude']);
  for (const [system, names] of Object.entries(SYSTEMS)) {
    assert.equal(new Set(names).size, names.length, `${system} repeats a name`);
    assert.ok(names.length >= 1);
  }
});

test('the picker lists every system the transform can answer for', () => {
  // The UI is built from these lists, so a system the transform supports but
  // the picker omits would be unreachable; one the picker offers but the
  // transform cannot answer would draw a blank plot. Both are silent.
  assert.deepEqual(SYSTEM_NAMES.slice().sort(), Object.keys(SYSTEMS).sort());
  for (const system of SYSTEM_NAMES) {
    assert.ok(SYSTEM_LABELS[system], `${system} has no label`);
    const names = COMPONENT_NAMES(system);
    assert.deepEqual(names, SYSTEMS[system]);
    for (const name of names) {
      const got = componentOf([1, 2, 3], [1, 1, 1], [0, 0, 0], system, name);
      assert.ok(Number.isFinite(got), `${system}.${name} is not answerable`);
      assert.ok(COMPONENT_LABELS[name], `${name} has no label`);
    }
  }
  // COMPONENT_NAMES must not hand out the module's own array, or a caller
  // sorting it for display would reorder the transform's table underneath it.
  const first = COMPONENT_NAMES('cartesian');
  first.reverse();
  assert.deepEqual(COMPONENT_NAMES('cartesian'), ['Ex', 'Ey', 'Ez']);
});

test('the default selection is one the transform can actually answer', () => {
  assert.ok(SYSTEMS[DEFAULT_SYSTEM], 'default system is not offered');
  assert.ok(SYSTEMS[DEFAULT_SYSTEM].includes(DEFAULT_COMPONENT),
            'default component is not in the default system');
});
