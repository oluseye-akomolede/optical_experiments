// The time-domain result plots: each probe's trace against time and its
// amplitude spectrum against frequency, drawn into inline SVG like the band
// diagram (no charting library, labels as text).
//
// The numbers a picture cannot check live in the pure helpers at the top and
// are tested in tests/js/test_tdplot.mjs: the unit prefix (a 0.2 ns run read
// as "0.0000000002 s" is unreadable, and a wrong prefix is off by 1000x), the
// ticks, the trace decode (probes x steps, C order, float32 LE -- a transposed
// read still draws plausible wiggles), and the envelope. A trace can be
// 20 000 steps; plain decimation to the plot's width aliases a CW trace into a
// slow false beat, so each pixel column keeps its min AND max instead.

const SVG_NS = 'http://www.w3.org/2000/svg';
const PAD = { top: 14, right: 16, bottom: 42, left: 62 };
const SERIES = ['--accent', '--warn', '--ok', '--muted', '--text'];
const PREFIXES = [[1e15, 'P'], [1e12, 'T'], [1e9, 'G'], [1e6, 'M'], [1e3, 'k'],
                  [1, ''], [1e-3, 'm'], [1e-6, 'µ'], [1e-9, 'n'],
                  [1e-12, 'p'], [1e-15, 'f']];

/** The SI prefix for values up to ``span``: {factor, prefix}, where
 * ``value / factor`` lies in 1..1000 for the largest value. */
export function siPrefix(span) {
  const s = Math.abs(Number(span));
  if (!Number.isFinite(s) || s === 0) return { factor: 1, prefix: '' };
  for (const [factor, prefix] of PREFIXES) {
    if (s >= factor * 0.9995) return { factor, prefix };
  }
  return { factor: 1e-15, prefix: 'f' };
}

/** Round tick positions covering [lo, hi], about ``count`` of them, on
 * 1/2/5 x 10^n steps. */
export function niceTicks(lo, hi, count = 5) {
  if (!(Number.isFinite(lo) && Number.isFinite(hi)) || hi <= lo) return [lo];
  const raw = (hi - lo) / Math.max(count, 1);
  const mag = 10 ** Math.floor(Math.log10(raw));
  // Round to the NEAREST 1/2/5 step, not the first one >= raw: "first >=" turns
  // 0.3..1.2 into ticks 0.5 and 1 only.
  const r = raw / mag;
  const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag;
  const ticks = [];
  for (let k = Math.ceil(lo / step - 1e-9); k * step <= hi + step * 1e-9; k += 1) {
    ticks.push(Number((k * step).toPrecision(12)));
  }
  return ticks;
}

/** The power of ten to pull out of a y axis whose largest magnitude is
 * ``peak``: 0 when the ticks read fine as plain numbers (1e-2..1e3), else the
 * exponent that puts the peak in 1..10. Fields come out near 1e-13, and
 * "2.00e-13" tick labels ran into the axis title. */
export function axisExponent(peak) {
  const p = Math.abs(Number(peak));
  if (!Number.isFinite(p) || p === 0 || (p >= 1e-2 && p < 1e3)) return 0;
  return Math.floor(Math.log10(p));
}

const SUPERSCRIPT = { '-': '⁻', 0: '⁰', 1: '¹', 2: '²', 3: '³', 4: '⁴',
                      5: '⁵', 6: '⁶', 7: '⁷', 8: '⁸', 9: '⁹' };

/** "×10⁻¹³" for -13; "" for 0. */
export function exponentText(e) {
  if (!e) return '';
  return `×10${String(e).split('').map((ch) => SUPERSCRIPT[ch]).join('')}`;
}

/** The colour of series ``i``, the same one its line is drawn in. */
export function seriesColour(i) {
  return colour(i);
}

/** ``trace.bin`` as one Float32Array per probe. ``shape`` is
 * [probes, steps] from the manifest; the buffer is C order, float32 LE. */
export function decodeTraces(buffer, shape) {
  const [probes, steps] = shape.map(Number);
  if (buffer.byteLength !== probes * steps * 4) {
    throw new Error(`trace.bin holds ${buffer.byteLength} bytes; the manifest's `
      + `${probes} x ${steps} float32 trace needs ${probes * steps * 4}`);
  }
  const view = new DataView(buffer);
  const rows = [];
  for (let p = 0; p < probes; p += 1) {
    const row = new Float32Array(steps);
    for (let n = 0; n < steps; n += 1) row[n] = view.getFloat32((p * steps + n) * 4, true);
    rows.push(row);
  }
  return rows;
}

/** Min and max of ``values`` in each of ``columns`` equal buckets of index:
 * [{i0, i1, min, max}]; every sample falls in exactly one bucket. */
export function envelope(values, columns) {
  const n = values.length;
  const cols = Math.max(1, Math.min(columns, n));
  const out = [];
  for (let c = 0; c < cols; c += 1) {
    const i0 = Math.floor((c * n) / cols);
    const i1 = Math.floor(((c + 1) * n) / cols);
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = i0; i < i1; i += 1) {
      if (values[i] < lo) lo = values[i];
      if (values[i] > hi) hi = values[i];
    }
    out.push({ i0, i1, min: lo, max: hi });
  }
  return out;
}

/** The grid's own step in metres: the smallest positive gap between the sample
 * positions the manifest lists, or null when it lists none (a job with no
 * movie, or a manifest written before the positions were recorded). */
export function gridStep(manifest) {
  const pos = manifest && manifest.frames && manifest.frames.positions_m;
  if (!Array.isArray(pos)) return null;
  let step = Infinity;
  for (const axis of pos) {
    for (let i = 1; i < (axis || []).length; i += 1) {
      const d = Math.abs(Number(axis[i]) - Number(axis[i - 1]));
      if (d > 0 && d < step) step = d;
    }
  }
  return Number.isFinite(step) ? step : null;
}

/** A probe's legend text: its component and position, in the length prefix of
 * the job's own grid. */
export function probeLabel(probe, index, manifest = null) {
  const pos = probe.position_m || [];
  // A stored position is ``origin_m + node``: two numbers the size of the grid
  // whose sum is the coordinate. A probe sitting AT the frame's own origin
  // therefore arrives as their float64 remainder -- here ~1e-19 m, a nanometre
  // grid's own rounding -- and printing that in the unit its own magnitude
  // asks for reads "0.0000400 fm": residue promoted to a measurement, twelve
  // orders below the grid's cell. Anything under a millionth of the cell IS
  // that origin; the stored residue is ~1e-10 of a cell at worst, so nothing
  // real is within four orders of the line.
  const step = gridStep(manifest);
  const snap = step === null ? 0 : 1e-6 * step;
  const axes = pos.map((v) => (Math.abs(Number(v)) < snap ? 0 : Number(v)));
  // The prefix is the grid's, never the residue's: an exactly-zero coordinate
  // reads "0" there. With no step (no movie in the manifest) neither the snap
  // nor the floor exists, and the largest coordinate picks the unit as before.
  const { factor, prefix } = siPrefix(Math.max(...axes.map((v) => Math.abs(v)), step || 0));
  const xyz = axes.map((v) => (v === 0 ? '0' : (v / factor).toPrecision(3))).join(', ');
  return `probe ${index + 1}: E${probe.component} at (${xyz}) ${prefix}m`;
}

function el(name, attrs, text) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, String(value));
  if (text !== undefined) node.textContent = text;
  return node;
}

function colour(i) {
  const css = getComputedStyle(document.documentElement)
    .getPropertyValue(SERIES[i % SERIES.length]).trim();
  return css || ['#4fa3ff', '#ffb347', '#6fdc8c', '#999', '#ddd'][i % SERIES.length];
}

// One framed plot: axes with ticks and labelled units, then ``series`` as
// polylines of [x, y] in data units. Returns nothing; a bad range draws the
// frame and labels only, so the caption (set by the caller) explains it.
function drawFrame(svg, { width, height, xLo, xHi, yLo, yHi, xUnit, yLabel, series }) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  const w = width - PAD.left - PAD.right;
  const h = height - PAD.top - PAD.bottom;
  const sx = (x) => PAD.left + ((x - xLo) / (xHi - xLo || 1)) * w;
  const sy = (y) => PAD.top + h - ((y - yLo) / (yHi - yLo || 1)) * h;
  const fg = colour(4);
  svg.appendChild(el('rect', { x: PAD.left, y: PAD.top, width: w, height: h,
                               fill: 'none', stroke: fg, 'stroke-opacity': 0.4 }));
  const { factor, prefix } = siPrefix(Math.max(Math.abs(xLo), Math.abs(xHi)));
  for (const t of niceTicks(xLo / factor, xHi / factor, 6)) {
    const x = sx(t * factor);
    svg.appendChild(el('line', { x1: x, x2: x, y1: PAD.top + h, y2: PAD.top + h + 4, stroke: fg }));
    svg.appendChild(el('text', { x, y: PAD.top + h + 16, 'text-anchor': 'middle',
                                 'font-size': 11, fill: fg }, String(t)));
  }
  const e = axisExponent(Math.max(Math.abs(yLo), Math.abs(yHi)));
  for (const t of niceTicks(yLo / 10 ** e, yHi / 10 ** e, 4)) {
    const y = sy(t * 10 ** e);
    svg.appendChild(el('line', { x1: PAD.left - 4, x2: PAD.left, y1: y, y2: y, stroke: fg }));
    svg.appendChild(el('text', { x: PAD.left - 6, y: y + 4, 'text-anchor': 'end',
                                 'font-size': 11, fill: fg }, String(t)));
  }
  if (e) yLabel = `${yLabel} ${exponentText(e)}`;
  svg.appendChild(el('text', { x: PAD.left + w / 2, y: height - 6, 'text-anchor': 'middle',
                               'font-size': 12, fill: fg }, `${xUnit.name} (${prefix}${xUnit.unit})`));
  svg.appendChild(el('text', { x: 14, y: PAD.top + h / 2, 'text-anchor': 'middle',
                               'font-size': 12, fill: fg,
                               transform: `rotate(-90 14 ${PAD.top + h / 2})` }, yLabel));
  series.forEach((points, i) => {
    const d = points.map(([x, y]) => `${sx(x).toFixed(1)},${sy(y).toFixed(1)}`).join(' ');
    svg.appendChild(el('polyline', { points: d, fill: 'none', stroke: colour(i),
                                     'stroke-width': 1.2, 'data-series': i }));
  });
}

function range(arrays) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const a of arrays) {
    for (const v of a) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!(hi > lo)) {
    const m = Number.isFinite(lo) ? lo : 0;
    return [m - 1, m + 1];
  }
  const pad = 0.05 * (hi - lo);
  return [lo - pad, hi + pad];
}

/** Each probe's trace against time; ``rows`` from decodeTraces. */
export function drawTraces(svg, rows, dt, { width = 640, height = 300 } = {}) {
  const steps = rows.length ? rows[0].length : 0;
  const cols = Math.max(1, width - PAD.left - PAD.right);
  const series = rows.map((row) => envelope(row, cols).flatMap((b) => {
    const t0 = b.i0 * dt;
    const t1 = Math.max(b.i1 - 1, b.i0) * dt;
    return [[t0, b.min], [t1, b.max]];
  }));
  const [yLo, yHi] = range(rows);
  drawFrame(svg, { width, height, xLo: 0, xHi: Math.max(steps - 1, 1) * dt, yLo, yHi,
                   xUnit: { name: 'time', unit: 's' },
                   yLabel: 'field component (arb. units)', series });
}

/** How the worker scales the spectra (td_solve.SPECTRUM_SCALE). */
export const SPECTRUM_SCALE = 'largest peak = 1';

/** Each probe's amplitude spectrum over the manifest's band, already scaled
 *  by the worker so the largest peak across all probes is 1. */
export function drawSpectra(svg, probes, band, { width = 640, height = 300 } = {}) {
  const series = probes.map((p) => p.spectrum.frequency_hz.map((f, i) => [f, p.spectrum.amplitude[i]]));
  const [, yHi] = range(probes.map((p) => p.spectrum.amplitude));
  drawFrame(svg, { width, height, xLo: band[0], xHi: band[1], yLo: 0, yHi,
                   xUnit: { name: 'frequency', unit: 'Hz' },
                   yLabel: `spectral amplitude (${SPECTRUM_SCALE})`, series });
}
