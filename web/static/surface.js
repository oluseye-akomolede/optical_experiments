// The uploaded model's own surface: unit conversion, per-part outlines, and
// the vertex-colour rule that said which of two objects a vertex belongs to.
// No DOM and no three.js, so the geometry can be tested against boxes whose
// answer is known by hand.
//
// WHY THIS FILE EXISTS. The owner of job 30796544 wrote: "The fields look
// strange, as though they are in a cubic region. I can't see the shape of the
// model whatsoever." The field data was correct; the viewer was comparing two
// objects in two different units and drawing the one it was not solving.
//
// The field grid is in METRES -- `modes.json`'s grid carries
// `origin: [-0.0052, ...]` and `spacing: [1.65e-4, ...]`, and the viewer builds
// every sample position as `origin + spacing * i`. The uploaded surface is in
// the USER'S units: `mse_maxwell_solver.write_results` exports
// `surface.export(file_type="glb")` before anything scales it, and the mesh
// path scales a *copy* (`apply_scale`) so the caller's object is untouched.
//
// Those agree only at mm/m (unit_scale 1) with scale 1. On the live job: the
// surface is a box of half-extents (5, 3.5, 2.5) in mm, the grid spans
// 10.4 x 7.28 x 5.2 in units where those numbers mean metres. Every one of the
// 92,160 grid samples therefore lands "inside" the surface when the two are
// compared as written -- measured, 92160/92160 -- so the outline (and the
// enclosure test that depends on it) describes the surface's own box and not
// the object at all. That is the cubic region the owner could see.
//
// The fix is one shared conversion, `surfaceScaleFromGrid`, applied by the
// viewer to the outline's transform (viewer.js's `addSurface`), so the object
// that is drawn sits in the grid's own space. (An earlier version of this
// comment named it `writeSurfaceScale`; no such function is exported here.)
// `pointsInside` below is the same rule for the sample positions, and exists so
// a test can hold the two to each other rather than to prose.

/**
 * The scale that maps the uploaded surface's own coordinates into the grid's.
 *
 * `mode.grid` already carries the conversion the solver applied when it built
 * its sample positions, and the viewer reads a grid in metres, so this is the
 * number that makes the two spaces one. Returns `[sx, sy, sz]` only for a grid
 * that actually encodes a conversion by a single length: a grid whose spacing
 * is already in the same space as the surface needs no conversion, and a grid
 * whose axes disagree about the conversion is not something this can guess --
 * both return null, and every caller treats null as "no conversion available"
 * rather than as a factor of 1.
 *
 * "Unavailable" is not "no scaling". The two are told apart on purpose: a
 * wrong factor puts the outline in the wrong place while looking deliberate,
 * and that is worse than an outline that admits it could not be placed.
 */
export function surfaceScaleFor(surfaceBounds, boxScale) {
  if (!surfaceBounds || !boxScale) return null;
  const s = [];
  for (let a = 0; a < 3; a += 1) {
    const v = Number(boxScale[a]);
    if (!Number.isFinite(v) || v <= 0) return null;
    s.push(v);
  }
  return s;
}

/**
 * The enclosure size the grid box was chosen from, in the surface's own units.
 *
 * The solver sizes its grid box from the mesh it built, so the grid box and the
 * surface describe the same object and the ratio of their sizes IS the unit
 * conversion:
 *
 *     box_in_metres[a] = box_in_surface_units[a] * unit_scale(units) * scale
 *     grid_sizes[a]    = box_in_metres[a]
 *   => conversion[a]   = grid_sizes[a] / box_in_surface_units[a]
 *
 * which needs no knowledge of `units` or `scale` separately -- deliberately, so
 * this file cannot disagree with the solver about either.
 *
 * Takes the grid *sizes* rather than counting samples: the surface and the grid
 * box are the same box, so the size is the well-defined quantity. Counting
 * samples divides by (n - 1) and is only an estimate of it, which is how the
 * first version of this got a 5.7% error on the live job -- one node width
 * short on every axis.
 */
export function surfaceScaleFromGrid(grid, surfaceSize) {
  if (!grid || !surfaceSize) return null;
  const n = [grid.nx, grid.ny, grid.nz];
  const sp = grid.spacing;
  if (!sp || !Number.isFinite(sp[0])) return null;
  const out = [];
  for (let a = 0; a < 3; a += 1) {
    const size = sp[a] * (n[a] - 1);   // the grid's extent along axis a
    const want = Number(surfaceSize[a]);
    if (!(want > 0) || !(size > 0)) return null;
    out.push(size / want);
  }
  return out;
}

/** Axis-aligned bounds of a flat float array of xyz positions, as [lo, hi]. */
export function boundsOf(positions) {
  const lo = [Infinity, Infinity, Infinity];
  const hi = [-Infinity, -Infinity, -Infinity];
  let any = false;
  for (let i = 0; i + 2 < positions.length; i += 3) {
    for (let a = 0; a < 3; a += 1) {
      const v = positions[i + a];
      if (!Number.isFinite(v)) continue;
      if (v < lo[a]) lo[a] = v;
      if (v > hi[a]) hi[a] = v;
      any = true;
    }
  }
  if (!any) return null;
  return { lo, hi, size: hi.map((v, a) => v - lo[a]) };
}

/**
 * A Möller-Trumbore ray cast of `points` against `triangles`, counting the
 * crossings of one ray per point: odd means inside.
 *
 * Why a ray cast and not the winding number the mesher uses (`point_in_part`):
 * this runs in the browser on a mesh of geometry_0's size, and the parity of a
 * ray crossing is the cheapest test with no orientation assumption. One ray
 * from far outside the surface towards the point, offset per point so a ray
 * cannot land on a shared edge or vertex -- a ray through an edge hits two
 * triangles or none depending on rounding, and either way reports a crossing
 * count that is wrong by exactly one.
 *
 * Returns a Uint8Array of 0/1. Direction (1, 0, 0) is not arbitrary: the
 * offset is applied on the two axes the ray does NOT travel along, so the
 * jitter cannot move a point along its own ray and change its parity.
 */
export function pointsInside(points, triangles, direction = [1, 0, 0]) {
  const n = Math.floor(points.length / 3);
  const m = Math.floor(triangles.length / 9);
  const out = new Uint8Array(n);
  const d = direction;
  for (let t = 0; t < m; t += 1) {
    const o = t * 9;
    const e1x = triangles[o + 3] - triangles[o + 0];
    const e1y = triangles[o + 4] - triangles[o + 1];
    const e1z = triangles[o + 5] - triangles[o + 2];
    const e2x = triangles[o + 6] - triangles[o + 0];
    const e2y = triangles[o + 7] - triangles[o + 1];
    const e2z = triangles[o + 8] - triangles[o + 2];
    // p = d x e2, computed once per triangle
    const px = d[1] * e2z - d[2] * e2y;
    const py = d[2] * e2x - d[0] * e2z;
    const pz = d[0] * e2y - d[1] * e2x;
    const det = e1x * px + e1y * py + e1z * pz;
    if (Math.abs(det) < 1e-18) continue;   // ray parallel to the triangle
    const inv = 1 / det;
    for (let i = 0; i < n; i += 1) {
      const q = i * 3;
      const sx = points[q + 0] - triangles[o + 0];
      const sy = points[q + 1] - triangles[o + 1];
      const sz = points[q + 2] - triangles[o + 2];
      const u = (sx * px + sy * py + sz * pz) * inv;
      if (u < -1e-12 || u > 1 + 1e-12) continue;
      const qx = sy * e1z - sz * e1y;
      const qy = sz * e1x - sx * e1z;
      const qz = sx * e1y - sy * e1x;
      const v = (qx * d[0] + qy * d[1] + qz * d[2]) * inv;
      if (v < -1e-12 || u + v > 1 + 1e-12) continue;
      const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
      // Only crossings in front of the point count. The ray is aimed at the
      // point from outside, so a hit beyond it is not between the point and
      // infinity and must not flip the parity.
      if (tt > 1e-12) out[i] ^= 1;
    }
  }
  return out;
}
