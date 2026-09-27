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
