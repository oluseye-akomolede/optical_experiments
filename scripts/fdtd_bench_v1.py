#!/usr/bin/env python3
"""V1 bench: the time-domain solver against an analytic rectangular PEC box.

WHAT THIS IS FOR. Phase 1 asks for one thing the unit tests cannot give: that a
real run of the engine reproduces a spectrum we already know independently. An
empty rectangular box has the textbook answer, so the test is not "does it look
right" but "where does each peak land".

WHAT IT DOES. Picks a grid, drives the box from one off-axis cell with a
broadband pulse, records E at a different off-axis cell, takes the spectrum of
the window AFTER the source has stopped, and prints every resolvable peak next
to the nearest analytic box frequency.

THREE RUNS, ONE PER E COMPONENT. A point current on z only drives modes whose Ez
is non-zero. In a box those are the TM-to-z modes, m and n both non-zero; the
TE-to-z modes (m or n zero) have Ez identically zero and are invisible to it.
Measured on the default box, a z source alone excites 2 of the 6 chased modes.
So each component gets its own run and the peaks are pooled. It costs three
times a single run, which is seconds.

THE PEAK IS READ OFF THE POWER SPECTRUM, NOT OFF |E|. |E| of a ringing field is
a rectified envelope: its spectrum is a large DC step plus lines at 2f, 4f and
so on, and the box's own lines are buried under the leakage of that step. With
the default box the |E| sweep reports six peaks with errors up to 17% and three
of them matched to the SAME mode, all of it leakage; the same run read per
component lands within 0.2%. Summing |Ex|^2 + |Ey|^2 + |Ez|^2 keeps it free of
the rectification and stops a mode hiding behind a polarisation the probe does
not record.

THE PEAK FREQUENCY IS NOT QUANTISED TO THE FFT BIN. At 120 recorded periods a
bin is 30 MHz, which at 4.8 GHz is 0.6% -- the size of the thing being measured.
Each peak is therefore placed at the vertex of a parabola through the three
log-amplitudes around the tallest bin, which is good to a small fraction of a
bin for a smooth peak.

THE ANALYTIC ORACLE IS IMPORTED, NOT RETYPED: ``analytic_box_k`` from
tests/bench_box.py, the same function the cavity-solver bench is judged by. The
integer triple next to each frequency comes from a local scan of the same
formula, and that scan is checked against the imported oracle at startup -- if
they ever disagree the run says so and fails, rather than printing a label that
belongs to a different number.

WHAT IT WILL NOT SHOW. A perfect match, or every mode. The Yee grid propagates a
wave slightly slower than c and the error grows as (dx/lambda)^2, so the peaks
land a little low; the run prints that estimate next to the measured errors. And
a point source excites only the modes it overlaps, so the run reports how many
of the analytic modes were seen rather than pretending the rest were checked.

THE GATE IS THE SPEC'S 1%, AND EVERY EXCITABLE MODE MUST BE FOUND. A peak more
than 1% from its mode fails the run, and so does an analytic mode the source
and probe both overlap that no peak was matched to: a dropped family is a
failure, not a smaller count. Peaks are matched to modes one-to-one (closest
pair first), because nearest-mode matching let two peaks claim the same mode --
measured on the default box, the 012 line at -0.30% was labelled 111 at +0.26%
and the run reported 4 of 5 modes when all 5 were there. The tolerance widens
only to 2 x the grid's own dispersion floor, and says so when it does; a
widened run reports NOT A SPEC PASS and exits 2 even if every peak is inside
the wider gate (FAIL exits 1, PASS 0). The default 20 cells per wavelength
keeps that floor (0.40%) inside 1%.

MEASURED. Orin at the old default of 16 cells per wavelength: 5.7 s wall, 44 MB
peak RSS. The VM at 20 (numpy, 2 threads): 9.7 s, 7500 cells x 6847 steps x 3
runs, all 5 distinct modes within 0.30%. The cost line prints the cell-step
rate a cap can be sized from.

THE CAVITY SOLVER IS NOT RUN HERE. It is the other leg of the comparison and it
is not cheap -- it tetrahedralizes the box with TetGen and solves a sparse
3-D eigenproblem -- so it runs only with ``--compare-cavity``, which shells out
to tests/bench_box.py on the box the grid describes and gates each matched
peak at --cavity-tolerance (1%) of the cavity solver's frequency for the same
mode. By default this script only does the analytic comparison.

RUN IT ON THE JETSON (mid-size solves are tier 2, one at a time). bench_box.py
imports the production stack, so ship the checkout, not the one file:

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v1'
    tar -C <repo> -cf - scripts/fdtd_bench_v1.py tests/bench_box.py \
        dockerfiles/maxwell_solver | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v1'
    ssh jetson 'cd ~/mse-bench/fdtd-v1 && \
      (ulimit -v 5000000; nice -n 10 timeout 30m python3 scripts/fdtd_bench_v1.py)'
"""

from __future__ import annotations

import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

# Cap threads before numpy loads, matching the container settings, so the bench
# measures the same thing a Job does.
_THREADS = os.environ.get("MSE_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))
sys.path.insert(0, str(ROOT / "tests"))

import fdtd  # noqa: E402
from bench_box import analytic_box_k  # noqa: E402

UNITS_TO_M = {"m": 1.0, "cm": 1e-2, "mm": 1e-3}
COMPONENTS = ("x", "y", "z")


def labelled_modes(a, b, d, n_max=8):
    """``{k: [triples]}`` for each PEC-box mode, labels from the same formula.

    The numbers used for the comparison come from ``analytic_box_k``; this scan
    exists to NAME the mode a peak matched, and main() checks the two agree.
    """
    modes = []
    for m in range(n_max):
        for p in range(n_max):
            for q in range(n_max):
                if (m > 0) + (p > 0) + (q > 0) < 2:
                    continue
                k = math.pi * math.sqrt((m / a) ** 2 + (p / b) ** 2 + (q / d) ** 2)
                modes.append((k, (m, p, q)))
    modes.sort(key=lambda item: item[0])
    grouped = {}
    for k, triple in modes:
        grouped.setdefault(round(k, 9), []).append(triple)
    return grouped


def dispersion_error(dx, wavelength):
    """The Yee phase-velocity error at ``dx`` per ``wavelength``, roughly.

    The standard estimate for a 3-D Yee cell: the maximum phase error is about
    ``(1/24) * (2*pi*dx/lambda)^2`` for a wave along a grid axis. It is printed,
    not asserted on -- the point is to say whether a residual is the grid.
    """
    return (1.0 / 24.0) * (2.0 * math.pi * dx / wavelength) ** 2


def refine_peak(frequencies, amplitude, index):
    """Sub-bin peak frequency, from a parabola through the three log-amplitudes.

    The tallest bin sits up to half a bin from the true line, and the bin is the
    size of the effect being measured, so the vertex offset is worth having.
    """
    if index <= 0 or index >= len(amplitude) - 1:
        return float(frequencies[index])
    y0, y1, y2 = (math.log(max(float(v), 1e-300))
                  for v in amplitude[index - 1:index + 2])
    denominator = y0 - 2.0 * y1 + y2
    if denominator == 0.0:
        return float(frequencies[index])
    offset = 0.5 * (y0 - y2) / denominator
    offset = max(-0.5, min(0.5, offset))
    step = float(frequencies[index + 1] - frequencies[index])
    return float(frequencies[index]) + offset * step


def find_peaks(frequencies, amplitude, f_lo, f_hi, n_peaks, floor=0.08):
    """Local maxima of the spectrum inside the band, tallest first."""
    amplitude = np.asarray(amplitude)
    inside = (frequencies >= f_lo) & (frequencies <= f_hi)
    peak_bins = [i for i in range(1, len(amplitude) - 1)
                 if inside[i] and amplitude[i] > amplitude[i - 1]
                 and amplitude[i] >= amplitude[i + 1]]
    if not peak_bins:
        return []
    tallest = max(float(amplitude[i]) for i in peak_bins)
    kept = [i for i in peak_bins if float(amplitude[i]) >= floor * tallest]
    kept.sort(key=lambda i: -float(amplitude[i]))
    kept = kept[:n_peaks]
    return sorted((refine_peak(frequencies, amplitude, i), float(amplitude[i]))
                  for i in kept)


#: The smallest |source shape x probe shape| at which a mode counts as one the
#: run must find. The shape factors are each at most 1; a mode under this is
#: one the run cannot be expected to see above the peak floor, and it is listed
#: as not gated rather than silently dropped.
EXCITABLE = 0.05


def coupling(triple, box, dx, cell, probe):
    """How strongly a point source at ``cell`` and a probe at ``probe`` see a mode.

    For each E component, the product of the analytic mode shape at the two
    Yee sample points (the component's own axis is offset by half a cell, as
    on the grid): cos along its own axis, sin along the other two. The best
    component wins, since the run drives and records all three. A degenerate
    TE/TM pair spans the plane normal to k, so each component that the factors
    allow is reachable by one of the pair.
    """
    best = 0.0
    for own in range(3):
        factor = 1.0
        for axis in range(3):
            shift = 0.5 if axis == own else 0.0
            wave = math.cos if axis == own else math.sin
            for index in (cell[axis], probe[axis]):
                factor *= wave(triple[axis] * math.pi * (index + shift) * dx
                               / box[axis])
        best = max(best, abs(factor))
    return best


def excitable_modes(labels, box, dx, cell, probe):
    """Distinct mode keys (as in ``labels``) the run must find."""
    return sorted(key for key, triples in labels.items()
                  if max(coupling(t, box, dx, cell, probe) for t in triples)
                  >= EXCITABLE)


def v1_verdict(failures, widened, *, worst, n_peaks, found, excitable,
               ungated, tolerance, dispersion):
    """``(verdict, line, exit code)`` for the run's summary.

    PASS / exit 0 only for a run that met the spec's own gate. A run whose
    tolerance was widened to the grid's dispersion floor did not test that
    gate, so even with every peak inside the wider one it is NOT A SPEC PASS
    and exits 2 (FAIL exits 1). The modes under the coupling threshold are
    counted in the same line, so a run that found 5 of 5 also says how many it
    never asked about.
    """
    if failures:
        verdict, code = "FAIL", 1
    elif widened:
        verdict, code = "NOT A SPEC PASS", 2
    else:
        verdict, code = "PASS", 0
    line = (f"V1 {verdict}: worst error {worst:.2f}% over {n_peaks} peak(s), "
            f"{found} of {excitable} excitable mode(s) found, "
            f"{ungated} mode(s) under the {EXCITABLE} coupling threshold, not "
            f"gated; tolerance {tolerance:.3g}% (dispersion floor "
            f"{dispersion:.3%})"
            + (" -- WIDENED past the spec's gate, so the spec's gate was not "
               "tested" if widened else ""))
    return verdict, line, code


def match_peaks(peaks, keys):
    """``{peak index: mode key}``, one-to-one, closest relative pair first.

    Nearest-mode matching lets two peaks claim one mode and hides the mode the
    second peak really is; here each mode takes at most one peak.
    """
    pairs = sorted((abs(value - fdtd.C0 * key / (2 * math.pi))
                    / (fdtd.C0 * key / (2 * math.pi)), index, key)
                   for index, value in enumerate(peaks) for key in keys)
    taken_peaks, taken_keys, assigned = set(), set(), {}
    for _error, index, key in pairs:
        if index in taken_peaks or key in taken_keys:
            continue
        assigned[index] = key
        taken_peaks.add(index)
        taken_keys.add(key)
    return assigned


def compare_with_cavity(peaks, assigned, modes, cavity, tolerance, v1_tolerance):
    """Gate the FDTD peaks against the cavity solver, mode by mode.

    ``modes`` (analytic k, ascending, degenerate modes repeated) and ``cavity``
    (the cavity solver's frequencies, ascending) are the same list of modes, so
    index i of one is index i of the other; a degenerate set is compared with
    the mean of its cavity frequencies. Only peaks that V1 matched to a chased
    mode within its own tolerance are compared, and every such mode must have a
    cavity frequency: a short cavity list is a failure, not fewer rows.

    Returns ``(rows, problems)``: rows are (peak Hz, cavity Hz, diff %, key).
    """
    problems = []
    if len(cavity) < len(modes):
        problems.append(f"the cavity solver returned {len(cavity)} mode(s) for "
                        f"{len(modes)} chased")
    by_key = {}
    for index, k in enumerate(modes[:len(cavity)]):
        by_key.setdefault(round(float(k), 9), []).append(float(cavity[index]))
    rows = []
    for index, key in sorted(assigned.items()):
        analytic = fdtd.C0 * key / (2.0 * math.pi)
        value = peaks[index]
        if abs(value - analytic) / analytic * 100.0 > v1_tolerance:
            continue
        if key not in by_key:
            if any(round(float(k), 9) == key for k in modes):
                problems.append(f"no cavity frequency for the mode at "
                                f"{analytic/1e9:.4f} GHz")
            continue
        reference = sum(by_key[key]) / len(by_key[key])
        diff = 100.0 * (value - reference) / reference
        rows.append((value, reference, diff, key))
        if abs(diff) > tolerance:
            problems.append(f"{value/1e9:.4f} GHz is {diff:+.2f}% from the cavity "
                            f"solver's {reference/1e9:.4f} GHz")
    if not rows and not problems:
        problems.append("no peak was compared")
    return rows, problems


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=float, default=40.0, help="box side along x")
    parser.add_argument("--b", type=float, default=53.0, help="box side along y")
    parser.add_argument("--d", type=float, default=67.0, help="box side along z")
    parser.add_argument("--units", default="mm", choices=sorted(UNITS_TO_M))
    parser.add_argument("--n-modes", type=int, default=6,
                        help="how many analytic modes to chase")
    parser.add_argument("--cells-per-wavelength", type=float, default=20.0,
                        help="grid resolution at the HIGHEST mode chased")
    parser.add_argument("--record-periods", type=float, default=120.0,
                        help="periods of the fundamental in the recorded window; "
                             "a FLOOR -- the window is lengthened to resolve the "
                             "closest analytic pair in the band")
    parser.add_argument("--max-record-periods", type=float, default=4000.0,
                        help="refuse rather than record longer than this")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="percent; the spec's gate (spec.md, benchmark V1)")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"),
                        help="cupy is the Jetson path; the same update code runs")
    parser.add_argument("--compare-cavity", action="store_true",
                        help="also run the production cavity solver on the same "
                             "box and gate against it (NOT cheap)")
    parser.add_argument("--cavity-tolerance", type=float, default=1.0,
                        help="percent; each matched peak against the cavity "
                             "solver's frequency for the same mode (proposal V1)")
    parser.add_argument("--seed-cell", default=None,
                        help="override the source cell, 'i,j,k'")
    args = parser.parse_args(argv)

    scale = UNITS_TO_M[args.units]
    xp = np
    if args.xp == "cupy":
        import cupy as xp  # noqa: F811 -- only here, never at module level

    # -- the grid, and the box the grid actually describes -------------------
    # The PEC faces sit ON the first and last E samples, so the wall separation
    # is (n - 1) * dx, not n * dx. The analytic frequencies must be taken for
    # the box the run really has, or the comparison is against another box.
    wanted = [args.a * scale, args.b * scale, args.d * scale]
    dx = min(wanted) / 8.0
    counts = None
    for _ in range(8):
        counts = [int(round(side / dx)) + 1 for side in wanted]
        box = [(n - 1) * dx for n in counts]
        modes = analytic_box_k(box[0], box[1], box[2], args.n_modes)
        f_top = float(fdtd.C0 * modes[-1] / (2.0 * math.pi))
        need = fdtd.C0 / (f_top * args.cells_per_wavelength)
        if dx <= need * 1.0000001:
            break
        dx = need
    frequencies = fdtd.C0 * np.asarray(modes) / (2.0 * math.pi)
    f_freq = list(map(float, frequencies))
    f_band_lo, f_band_hi = f_freq[0], f_freq[-1]

    labels = labelled_modes(box[0], box[1], box[2])
    oracle = np.asarray(modes)
    for k in oracle:
        if round(float(k), 9) not in labels:
            print(f"ORACLE MISMATCH: {k!r} from analytic_box_k is not in the "
                  f"local scan; refusing to print labels that may be wrong")
            return 2

    # -- source and probe: off-centre and off-axis, so modes are not skipped --
    n_a, n_b, n_c = counts
    if args.seed_cell:
        cell = tuple(int(v) for v in args.seed_cell.split(","))
    else:
        cell = (max(1, n_a // 4), max(1, n_b // 3), max(1, n_c // 5))
    for axis, index in enumerate(cell):
        if not 1 <= index <= counts[axis] - 2:
            print(f"source cell {cell} is on or outside a wall; move it inward")
            return 2
    probe = (max(1, (2 * n_a) // 3), max(1, n_b // 4 + 1), max(1, n_c // 3 + 2))
    probe = tuple(min(max(1, probe[i]), counts[i] - 2) for i in range(3))

    carrier = 0.5 * (f_band_lo + f_band_hi)
    bandwidth = 0.6 * (f_band_hi - f_band_lo)
    dt = fdtd.courant_dt(dx)
    source_steps = int(math.ceil(
        fdtd.Source(cell, frequency=carrier, profile="pulse",
                    bandwidth=bandwidth).end_time() / dt))
    # The record must resolve the closest pair of modes the peak search can
    # see, or two lines merge into one peak between them and the error is the
    # record length, not the solver: measured on the Orin at 128 cells per
    # wavelength, 012 sits 84 MHz from 111, and a 10-period record (bin
    # ~360 MHz) read 1.29% where a 40-period one read 0.32%. A Hann window's
    # main lobe is 4 bins wide, so a gap of 4 bins (record >= 4 / gap)
    # separates two lines. Modes just outside the chased set count too: the
    # search window is 0.8 x f_lo .. 1.2 x f_hi and the pulse reaches them.
    span_lo, span_hi = 0.8 * f_band_lo, 1.2 * f_band_hi
    visible = sorted(f for f in (fdtd.C0 * k / (2.0 * math.pi) for k in labels)
                     if span_lo <= f <= span_hi)
    gaps = [(b - a, a, b) for a, b in zip(visible, visible[1:])]
    gap, gap_lo, gap_hi = min(gaps) if gaps else (math.inf, None, None)
    record = max(args.record_periods / f_band_lo, 4.0 / gap)
    if record * f_band_lo > args.max_record_periods:
        print(f"CANNOT RESOLVE: modes at {gap_lo/1e9:.4f} and {gap_hi/1e9:.4f} GHz "
              f"are {gap/1e6:.4g} MHz apart, so separating them needs "
              f"{record * f_band_lo:.0f} periods of the fundamental, over "
              f"--max-record-periods {args.max_record_periods:g}; a 1% gate on "
              f"merged peaks would test the record length, not the solver")
        return 2
    window_steps = int(math.ceil(record / dt))
    n_steps = source_steps + window_steps

    cells = n_a * n_b * n_c
    field_mb = 6 * cells * 4 / 1e6
    dispersion = dispersion_error(dx, fdtd.C0 / f_band_hi)
    # The grid has its own error floor, so a tolerance tighter than twice that
    # is a test of the resolution, not of the solver. ``dispersion`` is a
    # fraction and the tolerance is in percent, so twice the floor is
    # 100 * 2 * dispersion. Widening is printed rather than silent.
    floor = 100.0 * 2.0 * dispersion
    tolerance = max(args.tolerance, floor)
    widened = tolerance > args.tolerance
    print(f"box        {args.a:g} x {args.b:g} x {args.d:g} {args.units} requested "
          f"-> {n_a} x {n_b} x {n_c} cells")
    print(f"walls      {box[0]*1e3:.4g} x {box[1]*1e3:.4g} x {box[2]*1e3:.4g} mm apart "
          f"(the PEC faces are ON the first and last E samples)")
    print(f"grid       dx = {dx*1e3:.4g} mm, {cells} cells, "
          f"fields {field_mb:.3g} MB float32 ({args.xp})")
    print(f"modes      {len(set(round(float(k), 9) for k in modes))} distinct analytic "
          f"of {args.n_modes} chased, {f_band_lo/1e9:.4g} to {f_band_hi/1e9:.4g} GHz")
    print(f"source     pulse carrier {carrier/1e9:.4g} GHz, bandwidth "
          f"{bandwidth/1e9:.4g} GHz, at cell {cell}; one run per E component")
    print(f"probe      Ex, Ey, Ez at cell {probe}; peaks from |Ex|^2+|Ey|^2+|Ez|^2")
    if gap_lo is not None:
        lengthened = record > args.record_periods / f_band_lo * 1.0000001
        print(f"record     {record * f_band_lo:.0f} periods of the fundamental: the "
              f"closest pair in the search band is {gap_lo/1e9:.4f} / "
              f"{gap_hi/1e9:.4f} GHz, {gap/1e6:.4g} MHz apart, which needs "
              f">= {4.0 / gap * 1e9:.4g} ns"
              + (f" (lengthened from --record-periods {args.record_periods:g})"
                 if lengthened else ""))
    print(f"steps      {n_steps} ({source_steps} of source + {window_steps} recorded) "
          f"x {len(COMPONENTS)} runs, dt = {dt*1e12:.4g} ps")
    print(f"dispersion at the top mode: {dispersion:.3%} (the Yee phase error, so "
          f"this is the floor on the comparison)")
    if widened:
        print(f"TOLERANCE WIDENED from {args.tolerance:g}% to {tolerance:.3g}% "
              f"(2 x the dispersion floor): this run does not test the spec's "
              f"{args.tolerance:g}%; raise --cells-per-wavelength for one that does")
    chased = sorted(set(round(float(v), 9) for v in modes))
    excitable = excitable_modes({k: labels[k] for k in chased}, box, dx, cell, probe)
    weak = [k for k in chased if k not in excitable]
    print(f"excitable  {len(excitable)} distinct mode(s) overlap both the source "
          f"and the probe (coupling >= {EXCITABLE})"
          + (f"; {len(weak)} do not and are not gated" if weak else ""))

    grid = xp.ones((n_a, n_b, n_c), dtype=xp.float32)
    probes = [tuple(probe) + (component,) for component in COMPONENTS]
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    pooled = {}
    elapsed_total = 0.0
    loudest = 0.0
    for component in COMPONENTS:
        source = fdtd.Source(cell, frequency=carrier, profile="pulse",
                             bandwidth=bandwidth, component=component)
        start = time.perf_counter()
        run = fdtd.simulate(grid, source, n_steps, probes, dx=dx,
                            boundary="metal", xp=xp)
        elapsed = time.perf_counter() - start
        elapsed_total += elapsed
        tails = [np.asarray(trace)[source_steps:] for trace in run["traces"]]
        span = float(tails[0].shape[0]) * dt
        freqs = None
        power = None
        for trace in tails:
            freqs, amplitude = fdtd.spectrum(trace, dt)
            freqs, amplitude = np.asarray(freqs), np.asarray(amplitude)
            power = amplitude * amplitude if power is None else power + amplitude * amplitude
        band = (freqs >= f_band_lo * 0.8) & (freqs <= f_band_hi * 1.2)
        loudest = max(loudest, float(power[band].max()))
        # Peak heights come back in the raw power units of this run, so the
        # merge below can compare a mode seen in two runs by either component.
        found = find_peaks(freqs, power, f_band_lo * 0.8, f_band_hi * 1.2, args.n_modes)
        bin_hz = 1.0 / span
        for value, height in found:
            key = round(value / bin_hz)
            if key not in pooled or height > pooled[key][1]:
                pooled[key] = (value, height, component)
        print(f"run {component}      {elapsed:.2f} s, {len(found)} peak(s) above "
              f"the floor")
    elapsed = elapsed_total
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"run        {elapsed:.2f} s total, peak RSS {peak_mb/1e3:.3g} MB "
          f"({(peak_mb-before)/1e3:+.3g} MB)")
    # The cost line is what a cell cap can be set from: cells x steps is the work
    # the runs actually did, and cell-steps/s is the rate this machine does it at.
    cell_steps = cells * n_steps * len(COMPONENTS)
    print(f"cost       {cells} cells x {n_steps} steps x {len(COMPONENTS)} runs = "
          f"{cell_steps:.4g} cell-steps in {elapsed:.2f} s -> "
          f"{cell_steps/elapsed:.4g} cell-steps/s")

    peaks = sorted(pooled.values(), key=lambda item: item[0])
    if not peaks:
        print("no peaks above the floor: the probe saw nothing to compare")
        return 1

    span = float(window_steps) * dt
    print(f"\nrecorded window {span*1e9:.4g} ns, bin {1.0/span/1e6:.4g} MHz "
          f"(peaks placed by parabola, so not quantised to the bin)")
    print(f"{'peak (GHz)':>12} {'height':>8} {'analytic (GHz)':>15} {'error':>9}  mode")
    # Every mode the search band can see is a candidate, chased or not, so a
    # peak from a mode just above the chased set is named rather than stray;
    # only the chased, excitable ones are required to be found.
    candidates = sorted(k for k in labels
                        if span_lo <= fdtd.C0 * k / (2.0 * math.pi) <= span_hi)
    assigned = match_peaks([value for value, _h, _c in peaks], candidates)
    worst = 0.0
    matched = set()
    stray = 0
    for index, (value, height, component) in enumerate(peaks):
        key = assigned.get(index)
        if key is None:
            stray += 1
            print(f"{value/1e9:12.4f} {100*height/loudest:7.1f}% {'-':>15} "
                  f"{'-':>9}  no analytic mode left to match")
            continue
        analytic = fdtd.C0 * key / (2.0 * math.pi)
        error = 100.0 * (value - analytic) / analytic
        worst = max(worst, abs(error))
        if abs(error) <= tolerance:
            matched.add(key)
        names = ", ".join("".join(str(i) for i in t) for t in labels[key])
        print(f"{value/1e9:12.4f} {100*height/loudest:7.1f}% {analytic/1e9:15.4f} "
              f"{error:+8.2f}%  {names}")

    missing = sorted(k for k in excitable if k not in matched)
    failures = []
    if worst > tolerance:
        failures.append(f"worst error {worst:.2f}% is over {tolerance:.3g}%")
    if missing:
        failures.append(
            f"{len(missing)} excitable mode(s) had no peak within tolerance: "
            + ", ".join(f"{fdtd.C0 * k / (2 * math.pi) / 1e9:.4f} GHz ("
                        + ", ".join("".join(map(str, t)) for t in labels[k]) + ")"
                        for k in missing))
    if stray:
        failures.append(f"{stray} peak(s) matched no analytic mode")
    verdict, line, code = v1_verdict(
        failures, widened, worst=worst, n_peaks=len(peaks),
        found=len(matched & set(excitable)), excitable=len(excitable),
        ungated=len(weak), tolerance=tolerance, dispersion=dispersion)
    print("\n" + line)
    for failure in failures:
        print(f"  {failure}")
    if worst > tolerance:
        print("A residual the size of the dispersion line above is the grid; "
              "anything larger is not.")

    cavity_ok = True
    if args.compare_cavity:
        # The box the GRID describes, not the one requested: the walls sit
        # (n - 1) * dx apart, and comparing against the requested box would
        # build its difference into every row.
        sides = [f"{side / scale:.9g}" for side in box]
        out = Path(os.environ.get("TMPDIR", "/tmp")) / f"v1-cavity-{os.getpid()}.json"
        command = [sys.executable, str(ROOT / "tests" / "bench_box.py"),
                   "--shape", "box", "--a", sides[0], "--b", sides[1],
                   "--d", sides[2], "--units", args.units,
                   "--n-modes", str(args.n_modes), "--json-out", str(out)]
        print("\n" + " ".join(command), "(the production path, NOT cheap)")
        done = subprocess.run(command, check=False)
        try:
            cavity = json.loads(out.read_text())["freq_hz"]
        except (OSError, ValueError, KeyError) as exc:
            cavity = None
            print(f"CAVITY FAIL: the cavity solver gave no modes to compare "
                  f"(exit {done.returncode}, {type(exc).__name__}); that is not "
                  f"agreement")
        if cavity is not None:
            rows, problems = compare_with_cavity(
                [value for value, _h, _c in peaks], assigned, modes, cavity,
                args.cavity_tolerance, tolerance)
            print(f"\n{'peak (GHz)':>12} {'cavity (GHz)':>13} {'diff':>9}  mode")
            for value, reference, diff, key in rows:
                names = ", ".join("".join(map(str, t)) for t in labels[key])
                print(f"{value/1e9:12.4f} {reference/1e9:13.4f} {diff:+8.2f}%  {names}")
            if done.returncode != 0:
                problems.append(f"the cavity solver's own check failed "
                                f"(exit {done.returncode}); its modes are not a "
                                f"reference")
            cavity_ok = not problems
            print(f"CAVITY {'PASS' if cavity_ok else 'FAIL'}: {len(rows)} peak(s) "
                  f"against the cavity solver, tolerance {args.cavity_tolerance:g}%")
            for problem in problems:
                print(f"  {problem}")
        else:
            cavity_ok = False

    return code if cavity_ok else 1


if __name__ == "__main__":
    sys.exit(main())
