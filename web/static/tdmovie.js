// The time-domain field movie, as the 3-D viewer needs it.
//
// The worker writes each frame as |E| float32, little-endian, C order, shaped
// (sx, sy, sz) in the grid's x, y, z order -- z varies fastest. The viewer's
// volume texture wants x fastest, on an EVENLY spaced lattice (texture space is
// uniform). The run's grid is graded, so its samples are not evenly spaced:
// drawing them as if they were would stretch a thin film to the width of a
// coarse cell. So each frame is resampled, by linear interpolation along each
// axis, onto an even lattice spanning the same box; `spectrum.json` gives the
// true sample positions (`frames.positions_m`).
//
// Pure functions only, so the node tests can pin them.

// The largest even lattice drawn per axis. 96^3 is ~0.9 M samples: rebuilt on
// every frame change, it stays responsive in a browser.
export const LATTICE_CAP = 96;

// The movie's scale is ONE number for every frame, so the field visibly grows
// and fades as it really does. It is a high percentile, not the maximum: one
// hot sample (a source node) would otherwise put the whole movie under the
// drawing floor.
export const SCALE_PERCENTILE = 0.99;

// The drawing floor, as a share of the scale above: the same contrast floor the
// cavity view uses (its THRESHOLD). Read by both the shader uniform and the
// caption, so the caption cannot state a different floor from the picture.
export const MOVIE_THRESHOLD = 0.12;

export function decodeFrame(buffer, shape) {
  const [sx, sy, sz] = shape;
  const count = sx * sy * sz;
  if (buffer.byteLength !== count * 4) {
    throw new Error(`a movie frame is ${buffer.byteLength} bytes, expected ${count * 4} `
      + `for ${sx} × ${sy} × ${sz} float32 samples`);
  }
  const view = new DataView(buffer);
  const out = new Float32Array(count);
  for (let i = 0; i < count; i += 1) out[i] = view.getFloat32(i * 4, true);
  return out;
}

// One axis of the even lattice: as many points as the finest gap between
// samples needs to keep every sample's spacing, never fewer than the samples
// themselves, at most LATTICE_CAP. A single-sample axis stays a single point.
export function latticeAxis(positions, cap = LATTICE_CAP) {
  const n = positions.length;
  if (n < 2) return { lo: positions[0] || 0, step: 0, count: 1 };
  const lo = positions[0];
  const hi = positions[n - 1];
  let gap = Infinity;
  for (let i = 1; i < n; i += 1) gap = Math.min(gap, positions[i] - positions[i - 1]);
  const needed = Math.ceil((hi - lo) / gap - 1e-9) + 1;
  const count = Math.max(Math.min(needed, cap), Math.min(n, cap));
  return { lo, step: (hi - lo) / (count - 1), count };
}

// For each lattice point on one axis, the sample below it and the weight of
// the sample above it, for linear interpolation between graded samples.
export function axisTable(positions, axis) {
  const below = new Int32Array(axis.count);
  const weight = new Float32Array(axis.count);
  const n = positions.length;
  let j = 0;
  for (let i = 0; i < axis.count; i += 1) {
    if (n < 2) { below[i] = 0; weight[i] = 0; continue; }
    const p = i === axis.count - 1 ? positions[n - 1] : axis.lo + i * axis.step;
    while (j < n - 2 && positions[j + 1] <= p) j += 1;
    const t = (p - positions[j]) / (positions[j + 1] - positions[j]);
    below[i] = j;
    weight[i] = Math.min(Math.max(t, 0), 1);
  }
  return { below, weight };
}

// The even lattice and its interpolation tables, built once per movie.
export function movieLattice(positionsM, cap = LATTICE_CAP) {
  const axes = positionsM.map((p) => latticeAxis(p, cap));
  const tables = positionsM.map((p, a) => axisTable(p, axes[a]));
  const grid = {
    nx: axes[0].count, ny: axes[1].count, nz: axes[2].count,
    origin: axes.map((a) => a.lo),
    spacing: axes.map((a) => a.step),
  };
  return { grid, tables, shape: positionsM.map((p) => p.length) };
}

// One frame, (sx, sy, sz) C order, onto the even lattice in texture order:
// index x + nx * (y + ny * z).
export function resampleFrame(frame, lattice) {
  const { grid, tables, shape } = lattice;
  const [, sy, sz] = shape;
  const [tx, ty, tz] = tables;
  const at = (ix, iy, iz) => frame[(ix * sy + iy) * sz + iz];
  const out = new Float32Array(grid.nx * grid.ny * grid.nz);
  // A single-sample axis has no sample above; its weight is 0, so reuse it.
  const up = (n) => (n > 1 ? 1 : 0);
  let o = 0;
  for (let z = 0; z < grid.nz; z += 1) {
    const z0 = tz.below[z]; const wz = tz.weight[z]; const z1 = z0 + up(sz);
    for (let y = 0; y < grid.ny; y += 1) {
      const y0 = ty.below[y]; const wy = ty.weight[y]; const y1 = y0 + up(sy);
      for (let x = 0; x < grid.nx; x += 1) {
        const x0 = tx.below[x]; const wx = tx.weight[x];
        const x1 = x0 + up(shape[0]);
        const c00 = at(x0, y0, z0) * (1 - wx) + at(x1, y0, z0) * wx;
        const c10 = at(x0, y1, z0) * (1 - wx) + at(x1, y1, z0) * wx;
        const c01 = at(x0, y0, z1) * (1 - wx) + at(x1, y0, z1) * wx;
        const c11 = at(x0, y1, z1) * (1 - wx) + at(x1, y1, z1) * wx;
        const c0 = c00 * (1 - wy) + c10 * wy;
        const c1 = c01 * (1 - wy) + c11 * wy;
        out[o] = c0 * (1 - wz) + c1 * wz;
        o += 1;
      }
    }
  }
  return out;
}

// The shared scale: the SCALE_PERCENTILE of every positive sample in every
// frame. Null when no frame has a positive sample (an all-zero movie has no
// scale, and is drawn empty, correctly).
export function movieScale(frames) {
  let total = 0;
  for (const f of frames) for (const v of f) if (v > 0) total += 1;
  if (!total) return null;
  const all = new Float32Array(total);
  let k = 0;
  for (const f of frames) for (const v of f) if (v > 0) { all[k] = v; k += 1; }
  all.sort();
  return all[Math.min(total - 1, Math.floor(SCALE_PERCENTILE * (total - 1)))];
}

// A frame divided by the shared scale and clamped to 1, ready for the shader.
export function scaledFrame(values, scale) {
  const out = new Float32Array(values.length);
  if (!scale) return out;
  for (let i = 0; i < values.length; i += 1) out[i] = Math.min(values[i] / scale, 1);
  return out;
}
