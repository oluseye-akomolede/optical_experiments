// Band diagram for a `kind='pc'` job: the geometry of the plot, and the gaps.
//
// PURE. Every function here takes the diagram payload and returns numbers or
// plain objects -- no DOM, no canvas, no fetch. That split is what lets the
// arithmetic that decides a gap be tested without a browser, because the gap is
// the part of this feature that can be quietly wrong: a plot that draws the
// wrong lines looks fine, and a gap number that is off by one band index reads
// as physics. The rendering below it is a thin loop over what these return.
//
// ω is in units of `w a / (2 pi c)`, which the solver returns and which is
// already the conventional normalisation for a PhC band diagram. The plot is
// therefore dimensionless and has no axis units to get wrong -- but that also
// means it cannot borrow the 3-D viewer's THz formatting, so the axis label
// says the unit rather than a number carrying one.

/**
 * The k-path as a distance axis: cumulative length along the polyline through
 * k-space, plus where each high-symmetry label sits.
 *
 * The x-axis of a band diagram is PATH LENGTH, not the k-index. They coincide
 * only when every segment is sampled at the same spacing AND has the same
 * length, which is not true here: the irreducible-zone path has segments of
 * different lengths (Γ→X and M→Γ differ), so plotting against the index would
 * stretch the short segments and put the labels at the wrong places. Distances
 * come from the actual 2-D k-points the solver returned.
 *
 * Returns {lengths, total, ticks} where ``ticks`` is [{at, label}] for the
 * high-symmetry points. A label is placed where it first appears, and a
 * repeated endpoint (the path ends where it began) does not get a second tick:
 * `labels` from the solver is one label per k-point, so the last point carries
 * the closing symbol and belongs at its own distance.
 */
export function kPathAxis(kPoints, labels) {
  if (!Array.isArray(kPoints) || kPoints.length === 0) {
    return { lengths: [], total: 0, ticks: [] };
  }
  const lengths = [0];
  for (let i = 1; i < kPoints.length; i += 1) {
    const [x0, y0] = kPoints[i - 1];
    const [x1, y1] = kPoints[i];
    lengths.push(lengths[i - 1] + Math.hypot(x1 - x0, y1 - y0));
  }
  const total = lengths[lengths.length - 1];

  // One tick per DISTINCT label, at the FIRST k-point carrying it. Walking the
  // labels rather than the k-points means a path that revisits a symbol in the
  // middle (some paths do) still yields one tick for it, and taking the first
  // occurrence is the choice that matches where the path enters that point.
  const ticks = [];
  const seen = new Set();
  if (Array.isArray(labels)) {
    for (let i = 0; i < labels.length && i < kPoints.length; i += 1) {
      const label = labels[i];
      if (label === undefined || label === null || seen.has(label)) continue;
      seen.add(label);
      ticks.push({ at: lengths[i], label: String(label) });
    }
  }
  return { lengths, total, ticks };
}

/**
 * The y-range of the plot: the extremes over every band at every k-point.
 *
 * The scale is SHARED across both polarisations when two are plotted -- the
 * caller passes the union -- because two panels drawn on separate automatic
 * scales invite a comparison between them that the axes do not support. The
 * 3-D viewer learned the same lesson about component scales; the answer is one
 * honest scale, not two flattering ones.
 *
 * A flat band (min === max, which the empty lattice produces: every frequency
 * is zero) would give a zero-height range and divide by zero downstream, so the
 * span is floored to 1 and the callers get a finite axis.
 */
export function bandRange(diagrams) {
  let lo = Infinity;
  let hi = -Infinity;
  for (const diagram of diagrams) {
    if (!diagram || !Array.isArray(diagram.bands)) continue;
    for (const row of diagram.bands) {
      for (const value of row) {
        if (!Number.isFinite(value)) continue;
        if (value < lo) lo = value;
        if (value > hi) hi = value;
      }
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return { lo: 0, hi: 1 };
  if (hi - lo < 1e-9) return { lo, hi: lo + 1 };
  return { lo, hi };
}

/**
 * Band gaps: for each pair of adjacent bands, the frequency window no state
 * occupies anywhere along the path.
 *
 * A gap is a property of the WHOLE path, not of one k-point: band n's maximum
 * over all k must sit below band n+1's minimum over all k for a window to exist
 * at all. Computed per band pair from the arrays rather than read off the plot,
 * because the plot is a picture of this number and not its source.
 *
 * Returns [{lower, upper, midgapRatio, index}] where ``midgapRatio`` is
 * Δω/ω_midgap, the standard figure of merit, and ``index`` is the lower band's
 * 1-based number (the gap "between band 1 and band 2" is index 1). An empty
 * array means no gap -- an honest absence, and the caller must say so rather
 * than leave the panel blank, since a blank panel reads as a loading state.
 */
export function bandGaps(diagram) {
  if (!diagram || !Array.isArray(diagram.bands) || diagram.bands.length < 2) {
    return [];
  }
  const bands = diagram.bands;
  const nK = bands[0].length;
  // Column-wise extremes: the max of band n over k, the min of band n+1 over k.
  // Guarded per value so a short or ragged row cannot make Math.max return NaN
  // and silently erase every gap.
  const gaps = [];
  for (let n = 0; n + 1 < bands.length; n += 1) {
    let lower = -Infinity;
    let upper = Infinity;
    for (let i = 0; i < nK; i += 1) {
      const a = bands[n][i];
      const b = bands[n + 1][i];
      if (Number.isFinite(a) && a > lower) lower = a;
      if (Number.isFinite(b) && b < upper) upper = b;
    }
    if (!(upper > lower)) continue;
    const mid = (upper + lower) / 2;
    // midgapRatio is Δω/ω_mid. A near-zero mid (the empty lattice) makes the
    // ratio meaningless rather than large, so it is only reported when the mid
    // is a real positive frequency.
    const ratio = mid > 0 ? (upper - lower) / mid : null;
    gaps.push({ lower, upper, midgapRatio: ratio, index: n + 1 });
  }
  return gaps;
}

/**
 * The plot's box: where a (k-distance, ω) pair lands in pixels.
 *
 * A factory rather than four loose functions so the axis limits are fixed once
 * and the marks and the ticks cannot disagree about them -- the failure mode
 * being a label drawn against a range the points were not mapped with. The
 * y-axis is inverted (SVG and canvas both grow downward, ω grows upward).
 */
export function plotScale({ width, height, pad, xMax, yLo, yHi }) {
  const left = pad.left;
  const top = pad.top;
  const innerW = Math.max(1, width - pad.left - pad.right);
  const innerH = Math.max(1, height - pad.top - pad.bottom);
  const xSpan = xMax > 0 ? xMax : 1;
  const ySpan = yHi - yLo || 1;
  return {
    left, top, innerW, innerH,
    x: (k) => left + (k / xSpan) * innerW,
    y: (w) => top + innerH - ((w - yLo) / ySpan) * innerH,
  };
}
