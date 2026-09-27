// Turning a mode's vector field into the one scalar field the volume plot
// colours. Kept apart from the viewer -- no DOM, no three.js -- so the maths
// can be tested against fields whose answer is known by hand.
//
// The owner asked to choose a field *component* and a *coordinate system*:
// Cartesian Ex/Ey/Ez, cylindrical E-rho/E-phi/Ez, spherical Er/E-theta/E-phi,
// plus the magnitude |E|. The solver's bytes are always Cartesian: the last
// axis of a (nz, ny, nx, 3) array is (Ex, Ey, Ez) at each grid point, and the
// transformation happens here, in the browser, once per texture.
//
// Two ways to get this wrong, both silent, which is why the tests pin them:
//   - the wrong axis for a grid position (the field array is (nz, ny, nx) while
//     origin/spacing are (x, y, z) -- see web/fields.py, where that exact swap
//     shipped and had to be fixed);
//   - a NaN produced by the *transform* rather than by the solver. The solver
//     marks everything outside the object NaN, and only that NaN may stay NaN.
//     A division on the polar axis would invent NaN inside the object, which
//     would punch a hole in the plot that is not in the physics.

/**
 * Physical centre of the grid box, in model coordinates.
 *
 * This is the origin for the radial systems. It is the box centre, not the
 * object's centroid: the box is what the solver sampled, and the two agree
 * only when the object happens to sit symmetrically in its own bounding box.
 */
export function boxCentre(grid) {
  const lo = grid.origin;
  const n = [grid.nx, grid.ny, grid.nz];
  const out = [0, 0, 0];
  for (let a = 0; a < 3; a += 1) {
    out[a] = lo[a] + (grid.spacing[a] * (n[a] - 1)) / 2;
  }
  return out;
}

/** The scalar fields a user can pick, grouped by coordinate system. */
export const SYSTEMS = {
  cartesian: ['Ex', 'Ey', 'Ez'],
  cylindrical: ['Erho', 'Ephi', 'Ez'],
  spherical: ['Er', 'Etheta', 'Ephi'],
  magnitude: ['E'],
};

/** Display order of the coordinate systems in the picker. */
export const SYSTEM_NAMES = ['cartesian', 'cylindrical', 'spherical', 'magnitude'];

/** Human labels. Kept beside the names so the two cannot drift apart. */
export const SYSTEM_LABELS = {
  cartesian: 'Cartesian',
  cylindrical: 'Cylindrical',
  spherical: 'Spherical',
  magnitude: 'Magnitude',
};

export const COMPONENT_LABELS = {
  Ex: 'Eₓ', Ey: 'Eᵧ', Ez: 'Eᵥ',
  Erho: 'Eᵨ', Ephi: 'Eᵩ',
  Er: 'Eᵣ', Etheta: 'Eθ',
  E: '|E|',
};

/** The components of one system, as the picker should list them. */
export function COMPONENT_NAMES(system) {
  return SYSTEMS[system] ? SYSTEMS[system].slice() : [];
}

/** What the plot shows before the user touches anything. */
export const DEFAULT_SYSTEM = 'magnitude';
export const DEFAULT_COMPONENT = 'E';

/**
 * The scalar value at one sample, for the chosen component.
 *
 * `v` is the (x, y, z) vector at that sample; `pos` is its physical position
 * and `centre` the box centre, both needed by the radial systems.
 *
 * NaN in any component of `v` means the solver marked the sample outside the
 * object, and the result is NaN -- the caller renders that as transparent. The
 * radial transforms add their own guard: on the polar axis rho (or r) is zero
 * and the direction is undefined, so the transverse components are reported as
 * 0 rather than as a division by zero. That is not a physical claim -- "no
 * azimuthal component on the axis" is what a smooth field does, but a field
 * with a genuine on-axis azimuthal component would be misreported here. It is
 * the choice that keeps an honest hole (outside) distinguishable from a
 * fabricated one (a NaN thread through the object); if on-axis phi components
 * ever matter, this is the line to revisit.
 */
export function componentOf(v, pos, centre, system, name) {
  const [ex, ey, ez] = v;
  if (Number.isNaN(ex) || Number.isNaN(ey) || Number.isNaN(ez)) return NaN;

  if (system === 'magnitude' || name === 'E') {
    return Math.hypot(ex, ey, ez);
  }
  if (system === 'cartesian') {
    return { Ex: ex, Ey: ey, Ez: ez }[name];
  }

  // Both radial systems measure from the box centre. dx/dy are the in-plane
  // offsets; the z axis is the polar axis for both, which is the convention the
  // lead confirmed with the owner (z as the axis).
  const dx = pos[0] - centre[0];
  const dy = pos[1] - centre[1];
  const dz = pos[2] - centre[2];

  if (system === 'cylindrical') {
    const rho = Math.hypot(dx, dy);
    if (name === 'Ez') return ez;
    if (rho === 0) {
      // On the axis the radial/azimuthal directions are undefined; report the
      // transverse part as 0 (see the docstring).
      return 0;
    }
    const erho = (dx * ex + dy * ey) / rho;
    const ephi = (-dy * ex + dx * ey) / rho;
    return { Erho: erho, Ephi: ephi }[name];
  }

  if (system === 'spherical') {
    const r = Math.hypot(dx, dy, dz);
    if (name === 'Er') {
      return r === 0 ? 0 : (dx * ex + dy * ey + dz * ez) / r;
    }
    if (r === 0) return 0;
    // theta is the polar angle from +z; its unit vector points along increasing
    // theta, i.e. away from +z. phi is measured in the x-y plane from +x.
    const rho = Math.hypot(dx, dy);
    if (name === 'Etheta') {
      if (rho === 0) return 0;   // on the polar axis
      const costh = dz / r;
      const sinth = rho / r;
      // e_theta = (cos(theta)cos(phi), cos(theta)sin(phi), -sin(theta))
      const et = (costh / rho) * (dx * ex + dy * ey) - sinth * ez;
      return et;
    }
    if (name === 'Ephi') {
      if (rho === 0) return 0;
      return (-dy * ex + dx * ey) / rho;
    }
  }
  return NaN;
}

/**
 * Turn a whole vector volume into the chosen scalar volume.
 *
 * `vec` is a flat Float32Array in (nz, ny, nx, 3) order -- the solver's own
 * layout, so no reshaping is needed. `positions` supplies each sample's
 * physical (x, y, z); it is passed in rather than derived here so the one place
 * that knows the two axis orders stays in the viewer, next to the grid it came
 * from.
 *
 * Returns a Float32Array in (nz, ny, nx) order, ready to hand to a Data3DTexture.
 */
export function componentVolume(vec, positions, centre, system, name) {
  const n = positions.length / 3;
  const out = new Float32Array(n);
  const v = [0, 0, 0];
  const p = [0, 0, 0];
  for (let i = 0; i < n; i += 1) {
    const o = i * 3;
    v[0] = vec[o]; v[1] = vec[o + 1]; v[2] = vec[o + 2];
    p[0] = positions[o]; p[1] = positions[o + 1]; p[2] = positions[o + 2];
    out[i] = componentOf(v, p, centre, system, name);
  }
  return out;
}

/**
 * How strong the chosen component is in this mode, relative to the mode's own
 * peak |E|, as a fraction in 0..1.
 *
 * The plot draws every component on ONE shared scale per mode -- the mode's
 * max |E| -- so that a component is directly comparable with the same
 * component in the next mode, and with |E| in this one. That is the right
 * scale, but it has a consequence the picture alone cannot explain: a
 * component that is genuinely small in this mode draws as a very faint cloud,
 * and a faint cloud is exactly what a rendering bug looks like.
 *
 * It is not a rendering bug, and the two must be told apart. Measured on the
 * live job: mode 1 is the z-polarised (1,1,0) mode, whose Erho reaches only 8%
 * of its own peak |E| and draws as almost nothing, while mode 2's Erho reaches
 * 98% and draws strongly. The same control, two opposite pictures, both
 * correct.
 *
 * The comparison is a high percentile rather than the maximum because a
 * component's single largest sample can be a local spike on an otherwise small
 * field: mode 1's Erho has a maximum of 42% of |E| contributed by one corner
 * sample, while its 99th percentile is 8%. The p99 is what the eye sees.
 *
 * Returns null when the answer would be meaningless -- no finite samples, no
 * non-zero peak to compare against -- rather than a fabricated 0 or 1. A
 * caller that cannot measure must say so, not print a number that reads like a
 * measurement (the could-not-ask rule).
 */
export function componentShare(vec, positions, centre, system, name) {
  if (system === 'magnitude' || name === 'E') return 1;
  const n = positions.length / 3;
  const v = [0, 0, 0];
  const p = [0, 0, 0];
  const mags = [];
  const comps = [];
  for (let i = 0; i < n; i += 1) {
    const o = i * 3;
    v[0] = vec[o]; v[1] = vec[o + 1]; v[2] = vec[o + 2];
    if (Number.isNaN(v[0]) || Number.isNaN(v[1]) || Number.isNaN(v[2])) continue;
    p[0] = positions[o]; p[1] = positions[o + 1]; p[2] = positions[o + 2];
    const m = Math.hypot(v[0], v[1], v[2]);
    const c = componentOf(v, p, centre, system, name);
    if (!Number.isFinite(m) || !Number.isFinite(c)) continue;
    mags.push(m);
    comps.push(Math.abs(c));
  }
  if (!mags.length) return null;
  const peak = percentile(mags, 0.99);
  if (!peak) return null;
  return percentile(comps, 0.99) / peak;
}

/** Linear-interpolated percentile of an unsorted array. 0..1, inclusive. */
function percentile(values, q) {
  const sorted = values.slice().sort((a, b) => a - b);
  if (!sorted.length) return null;
  const at = q * (sorted.length - 1);
  const lo = Math.floor(at);
  const hi = Math.ceil(at);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (at - lo);
}

// --------------------------------------------------------------------------
// float32 -> float16 (IEEE 754 binary16), round-to-nearest-even
// --------------------------------------------------------------------------
// Written out rather than built on Float16Array, which Safari did not ship
// until 2025 and which would turn an older-but-fine browser into a page that
// loads and then shows nothing. NaN maps to a NaN half (the exponent is all
// ones and the mantissa is non-zero), which is what keeps the solver's
// outside-the-object samples transparent through the upload.
export function float32ToHalf(value) {
  const buffer = new ArrayBuffer(4);
  const f32 = new Float32Array(buffer);
  const u32 = new Uint32Array(buffer);
  f32[0] = value;
  const x = u32[0];
  const sign = (x >>> 16) & 0x8000;
  const exp = (x >>> 23) & 0xff;
  const mantissa = x & 0x7fffff;

  if (exp === 0xff) {
    // Infinity or NaN. A non-zero mantissa is what makes a NaN a NaN, so it is
    // carried over (any non-zero value will do, but keeping it costs nothing).
    return sign | 0x7c00 | (mantissa ? 0x0200 : 0);
  }
  const half = exp - 127 + 15;
  if (half >= 0x1f) return sign | 0x7c00;              // overflow -> infinity
  if (half <= 0) {
    if (half < -10) return sign;                        // underflow -> zero
    // Subnormal: shift the implicit leading 1 back in and round.
    const m = mantissa | 0x800000;
    const shift = 14 - half;
    const r = m >> shift;
    const rem = m & ((1 << shift) - 1);
    const halfway = 1 << (shift - 1);
    const roundUp = rem > halfway || (rem === halfway && (r & 1) === 1);
    return sign | (r + (roundUp ? 1 : 0));
  }
  let m = mantissa >> 13;
  const rem = mantissa & 0x1fff;
  if (rem > 0x1000 || (rem === 0x1000 && (m & 1) === 1)) {
    m += 1;
    if (m === 0x400) return sign | ((half + 1) << 10);   // mantissa carried
  }
  return sign | (half << 10) | m;
}

// --------------------------------------------------------------------------
// float16 -> float32
// --------------------------------------------------------------------------
// The inverse of the above, and the same reasoning for writing it by hand:
// DataView's half-float read is not in every engine, and Float16Array is the
// same recent addition. The field route sends half floats, so without this the
// viewer has no way to read the payload it fetches at all -- and reading the
// bytes as float32 instead does not fail loudly, it produces a plausible array
// of the wrong length whose numbers are nonsense. That is the bug this exists
// to close: the viewer decoded the wire format with the wrong assumption and
// the only thing that caught it was the payload-length guard downstream.
//
// NaN decodes back to NaN rather than to Infinity, because the solver uses NaN
// to mean "outside the object" and the shader tests for it.
export function halfToFloat32(h) {
  const sign = h & 0x8000 ? -1 : 1;
  const exp = (h >> 10) & 0x1f;
  const frac = h & 0x3ff;
  if (exp === 0x1f) return frac ? NaN : sign * Infinity;
  if (exp === 0) return sign * frac * 2 ** -24;   // subnormal (and zero)
  return sign * (1 + frac / 1024) * 2 ** (exp - 15);
}
