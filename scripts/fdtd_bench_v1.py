#!/usr/bin/env python3
"""V1 bench: the time-domain solver against an analytic rectangular PEC box.

WHAT THIS IS FOR. Phase 1 asks for one thing the unit tests cannot give: that a
real run of the engine reproduces a spectrum we already know independently. An
empty rectangular box has the textbook answer, so the test is not "does it look
right" but "where does each peak land".

WHAT IT DOES. Picks a grid, drives the box from one off-axis cell with a
broadband pulse, records |E| at a different off-axis cell, takes the spectrum of
the window AFTER the source has stopped, and prints every resolvable peak next
to the nearest analytic box frequency.

THE ANALYTIC ORACLE IS IMPORTED, NOT RETYPED: ``analytic_box_k`` from
tests/bench_box.py, the same function the cavity-solver bench is judged by. The
integer triple next to each frequency comes from a local scan of the same
formula, and that scan is checked against the imported oracle at startup -- if
they ever disagree the run says so and fails, rather than printing a label that
belongs to a different number.

WHAT IT WILL NOT SHOW. A perfect match. The Yee grid propagates a wave slightly
slower than c, and the error grows as (dx/lambda)^2, so the peaks land a little
low. The run prints that estimate next to the measured errors: if the residual
is the size of the dispersion, it is the grid, and if it is larger something
else is wrong.

THE CAVITY SOLVER IS NOT RUN HERE. It is the other leg of the comparison and it
is not cheap -- it tetrahedralizes the box with TetGen and solves a sparse
3-D eigenproblem -- so it runs as its own step (``--compare-cavity`` shells out
to tests/bench_box.py and prints its table). By default this script only does
the analytic comparison.

RUN IT ON THE JETSON (mid-size solves are tier 2, one at a time):

    ssh jetson 'cd ~/mse-bench && mkdir -p fdtd-v1 && cd fdtd-v1 && \
      git -C <repo> show <sha>:scripts/fdtd_bench_v1.py > fdtd_bench_v1.py && \
      (ulimit -v 5000000; nice -n 10 timeout 30m python fdtd_bench_v1.py)'
"""

from __future__ import annotations

import argparse
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


def labelled_modes(a, b, d, n_max=8):
    """``(k, triples)`` for each PEC-box mode, labels from the same formula.

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
        key = round(k, 9)
        grouped.setdefault(key, []).append(triple)
    return grouped


def dispersion_error(dx, wavelength):
    """The Yee phase-velocity error at ``dx`` per ``wavelength``, roughly.

    The standard estimate for a 3-D Yee cell: the maximum phase error is about
    ``(1/24) * (2*pi*dx/lambda)^2`` for a wave along a grid axis. It is printed,
    not asserted on -- the point is to say whether a residual is the grid.
    """
    return (1.0 / 24.0) * (2.0 * math.pi * dx / wavelength) ** 2


def find_peaks(frequencies, amplitude, f_lo, f_hi, n_peaks, floor=0.08):
    """Local maxima of the spectrum inside the band, tallest first."""
    amplitude = np.asarray(amplitude)
    inside = np.zeros(amplitude.shape, dtype=bool)
    inside[(frequencies >= f_lo) & (frequencies <= f_hi)] = True
    peak_bins = []
    for i in range(1, len(amplitude) - 1):
        if not inside[i]:
            continue
        if amplitude[i] > amplitude[i - 1] and amplitude[i] >= amplitude[i + 1]:
            peak_bins.append(i)
    if not peak_bins:
        return []
    tallest = max(amplitude[i] for i in peak_bins)
    kept = [i for i in peak_bins if amplitude[i] >= floor * tallest]
    kept.sort(key=lambda i: -amplitude[i])
    kept = kept[:n_peaks]
    return sorted((float(frequencies[i]), float(amplitude[i])) for i in kept)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=float, default=40.0, help="box side along x")
    parser.add_argument("--b", type=float, default=53.0, help="box side along y")
    parser.add_argument("--d", type=float, default=67.0, help="box side along z")
    parser.add_argument("--units", default="mm", choices=sorted(UNITS_TO_M))
    parser.add_argument("--n-modes", type=int, default=6,
                        help="how many analytic modes to chase")
    parser.add_argument("--cells-per-wavelength", type=float, default=16.0,
                        help="grid resolution at the HIGHEST mode chased")
    parser.add_argument("--record-periods", type=float, default=120.0,
                        help="periods of the fundamental in the recorded window")
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="percent; every resolvable peak must be within this")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"),
                        help="cupy is the Jetson path; the same update code runs")
    parser.add_argument("--compare-cavity", action="store_true",
                        help="also run the production cavity solver (NOT cheap)")
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
    source = fdtd.Source(cell, frequency=carrier, profile="pulse",
                         bandwidth=bandwidth, component="z")

    dt = fdtd.courant_dt(dx)
    source_steps = int(math.ceil(source.end_time() / dt))
    window_steps = int(math.ceil(args.record_periods / f_band_lo / dt))
    n_steps = source_steps + window_steps

    cells = n_a * n_b * n_c
    field_mb = 6 * cells * 4 / 1e6
    print(f"box        {args.a:g} x {args.b:g} x {args.d:g} {args.units} requested "
          f"-> {n_a} x {n_b} x {n_c} cells")
    print(f"walls      {box[0]*1e3:.4g} x {box[1]*1e3:.4g} x {box[2]*1e3:.4g} mm apart "
          f"(the PEC faces are ON the first and last E samples)")
    print(f"grid       dx = {dx*1e3:.4g} mm, {cells} cells, "
          f"fields {field_mb:.3g} MB float32 ({args.xp})")
    print(f"modes      {args.n_modes} analytic, {f_band_lo/1e9:.4g} to "
          f"{f_band_hi/1e9:.4g} GHz")
    print(f"source     pulse carrier {carrier/1e9:.4g} GHz, bandwidth "
          f"{bandwidth/1e9:.4g} GHz, component z at cell {cell}")
    print(f"probe      |E| at cell {probe}")
    print(f"steps      {n_steps} ({source_steps} of source + {window_steps} recorded), "
          f"dt = {dt*1e12:.4g} ps")
    print(f"dispersion at the top mode: {dispersion_error(dx, fdtd.C0/f_band_hi):.3%} "
          f"(the Yee phase error, so this is the floor on the comparison)")

    grid = xp.ones((n_a, n_b, n_c), dtype=xp.float32)
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    run = fdtd.simulate(grid, source, n_steps, [(probe[0], probe[1], probe[2])],
                        dx=dx, boundary="metal", xp=xp)
    elapsed = time.perf_counter() - start
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"run        {elapsed:.2f} s, peak RSS {peak_mb/1e3:.3g} MB "
          f"({(peak_mb-before)/1e3:+.3g} MB)")

    trace = run["traces"][0][source_steps:]
    span = float(trace.shape[0]) * dt
    freqs, amplitude = fdtd.spectrum(trace, dt)
    if xp is not np:
        freqs, amplitude = np.asarray(freqs), np.asarray(amplitude)

    peaks = find_peaks(freqs, amplitude, f_band_lo * 0.8, f_band_hi * 1.2,
                       args.n_modes)
    if not peaks:
        print("no peaks above the floor: the probe saw nothing to compare")
        return 1

    print(f"\nrecorded window {span*1e9:.4g} ns, bin {1.0/span/1e6:.4g} MHz")
    print(f"{'peak (GHz)':>12} {'analytic (GHz)':>15} {'error':>9}  mode")
    worst = 0.0
    for value, _height in peaks:
        nearest = min(oracle, key=lambda k: abs(fdtd.C0 * k / (2 * math.pi) - value))
        analytic = fdtd.C0 * nearest / (2.0 * math.pi)
        error = 100.0 * (value - analytic) / analytic
        worst = max(worst, abs(error))
        names = ", ".join("".join(str(i) for i in t)
                          for t in labels[round(float(nearest), 9)])
        print(f"{value/1e9:12.4f} {analytic/1e9:15.4f} {error:+8.2f}%  {names}")

    verdict = "PASS" if worst <= args.tolerance else "FAIL"
    print(f"\nV1 {verdict}: worst error {worst:.2f}% over {len(peaks)} peak(s), "
          f"tolerance {args.tolerance:g}%")
    if worst > args.tolerance:
        print("A residual the size of the dispersion line above is the grid; "
              "anything larger is not.")

    if args.compare_cavity:
        command = [sys.executable, str(ROOT / "tests" / "bench_box.py"),
                   "--shape", "box", "--a", str(args.a), "--b", str(args.b),
                   "--d", str(args.d), "--units", args.units,
                   "--n-modes", str(args.n_modes)]
        print("\n" + " ".join(command), "(the production path, NOT cheap)")
        subprocess.run(command, check=False)

    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
