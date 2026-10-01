// Draws a solved band diagram into an inline SVG.
//
// SVG, not canvas, and not a charting library: the plan says no new vendor file
// for a line chart, and SVG gets the labels as selectable text and the figure
// scalable for free. The geometry (axis, gaps, the flat-range floor) lives in
// bands.js and is tested there; this file only turns those numbers into
// elements, so a bug here is a wrong picture rather than a wrong number.
//
// The x-axis is drawn in k-path DISTANCE with a tick per high-symmetry point,
// and each polarisation gets its own row of lines inside the shared frame --
// TM and TE are different eigenvalue problems and overlaying them would invite
// reading a crossing as a degeneracy.

import { kPathAxis, bandRange, bandGaps, plotScale,
         kPointCount, bandCount } from './bands.js';

const SVG_NS = 'http://www.w3.org/2000/svg';
const PAD = { top: 18, right: 16, bottom: 40, left: 52 };
const SERIES = ['--accent', '--warn'];

function el(name, attrs) {
  const node = document.createElementNS(SVG_NS, name);
  for (const [key, value] of Object.entries(attrs)) {
    node.setAttribute(key, String(value));
  }
  return node;
}

/**
 * Render one or more diagrams (one per polarisation) into ``svg``.
 *
 * ``diagrams`` is [{polarisation, labels, k_points, bands}, ...] as stored in
 * ``pcs.bands``. The frame is shared: one x-axis, one y-axis, one set of gap
 * bands, so the two polarisations are directly comparable.
 */
export function drawBandDiagram(svg, diagrams, { width = 640, height = 380 } = {}) {
  const list = (diagrams || []).filter((d) => d && Array.isArray(d.k_points)
    && Array.isArray(d.bands) && d.bands.length);
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  if (!list.length) return;

  const axes = list.map((d) => kPathAxis(d.k_points, d.labels));
  const xMax = Math.max(...axes.map((a) => a.total));
  const { lo, hi } = bandRange(list);
  const scale = plotScale({ width, height, pad: PAD, xMax, yLo: lo, yHi: hi });
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.setAttribute('role', 'img');

  // Frame, drawn first so the curves sit on top of it.
  svg.appendChild(el('rect', {
    x: scale.left, y: scale.top, width: scale.innerW, height: scale.innerH,
    fill: 'none', stroke: 'currentColor', 'stroke-opacity': '0.35',
  }));

  // The gaps of the FIRST series shade the whole frame: a gap is the app's
  // headline result, and if the two polarisations disagree about one there must
  // be two panels, not one frame with two shadings arguing. The caller passes
  // one series when that is the case.
  const gaps = bandGaps(list[0]);
  for (const gap of gaps) {
    svg.appendChild(el('rect', {
      x: scale.left, y: scale.y(gap.upper),
      width: scale.innerW, height: Math.max(0, scale.y(gap.lower) - scale.y(gap.upper)),
      fill: 'var(--accent)', 'fill-opacity': '0.12',
    }));
  }

  // k-path ticks: a vertical rule at each high-symmetry point, labelled on the
  // bottom. The rule is faint so it reads as a gridline behind the bands.
  for (const tick of axes[0].ticks) {
    const x = scale.x(tick.at);
    svg.appendChild(el('line', {
      x1: x, x2: x, y1: scale.top, y2: scale.top + scale.innerH,
      stroke: 'currentColor', 'stroke-opacity': '0.18',
    }));
    const text = el('text', {
      x, y: scale.top + scale.innerH + 18, 'text-anchor': 'middle',
      'font-size': '12', fill: 'currentColor',
    });
    text.textContent = tick.label;
    svg.appendChild(text);
  }

  // The two or three y-gridlines, labelled. Chosen at round fractions of the
  // range rather than from a tick algorithm: the range is a unitless ω and the
  // reader only needs the scale, not a measured axis.
  for (let i = 0; i <= 4; i += 1) {
    const w = lo + ((hi - lo) * i) / 4;
    const y = scale.y(w);
    svg.appendChild(el('line', {
      x1: scale.left, x2: scale.left + scale.innerW, y1: y, y2: y,
      stroke: 'currentColor', 'stroke-opacity': '0.08',
    }));
    const text = el('text', {
      x: scale.left - 8, y: y + 4, 'text-anchor': 'end', 'font-size': '11',
      fill: 'currentColor', 'fill-opacity': '0.7',
    });
    text.textContent = w.toFixed(2);
    svg.appendChild(text);
  }

  // The bands, one <path> per band per series. A single path per band keeps the
  // element count proportional to bands, not to k-points, which matters because
  // the production path is ~30 k-points x 20 bands x 2 polarisations.
  //
  // Orientation: ``bands[k][band]``, so a path walks the OUTER index (the
  // k-points, one point per entry of ``k_points``) and holds the band number
  // fixed. Iterating the outer array instead draws one path per k-point -- on a
  // real 30x4 diagram, 30 two-to-four-point stubs that look like a plot.
  list.forEach((diagram, series) => {
    const axis = axes[series];
    const colour = `var(${SERIES[series % SERIES.length]})`;
    const nK = kPointCount(diagram);
    const nB = bandCount(diagram);
    for (let band = 0; band < nB; band += 1) {
      let d = '';
      // Is there a subpath the next point could extend? A ragged row -- one that
      // is shorter than the widest -- has no value for this band, and
      // ``scale.y(undefined)`` is NaN. NaN in an SVG path ``d`` is not an error
      // any browser reports: it silently drops the whole subpath, so the curve
      // vanishes while the gap table beside it (bandGaps skips non-finite
      // values) still counts that band. So a missing value SKIPS the point and
      // starts a new subpath, which is the same claim the table makes: this
      // k-point has no value for this band.
      let pen = false;
      for (let i = 0; i < nK; i += 1) {
        const value = diagram.bands[i][band];
        if (!Number.isFinite(value)) {
          pen = false;
          continue;
        }
        const x = scale.x(axis.lengths[i] ?? 0);
        const y = scale.y(value);
        d += `${pen ? 'L' : 'M'}${x.toFixed(2)},${y.toFixed(2)} `;
        pen = true;
      }
      // A band with no finite value anywhere draws nothing at all, rather than
      // an empty path element that reads as a curve of zero length.
      if (!d) continue;
      svg.appendChild(el('path', {
        d: d.trim(), fill: 'none', stroke: colour, 'stroke-width': '1.6',
        'stroke-linejoin': 'round',
      }));
    }
  });

  svg.appendChild(axisLabel('k-path', width / 2, height - 6));
  svg.appendChild(rotatedAxisLabel('ω a / (2πc)', 14, height / 2));
}

function axisLabel(text, x, y) {
  const node = el('text', { x, y, 'text-anchor': 'middle', 'font-size': '12',
                            fill: 'currentColor' });
  node.textContent = text;
  return node;
}

function rotatedAxisLabel(text, x, y) {
  const node = el('text', {
    x, y, 'text-anchor': 'middle', 'font-size': '12', fill: 'currentColor',
    transform: `rotate(-90 ${x} ${y})`,
  });
  node.textContent = text;
  return node;
}
