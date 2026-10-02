#!/usr/bin/env python3
"""Benchmark V5: a thin dielectric film between two PEC plates.

The spec's gate: a film of index n, far thinner than the coarse cell, fills
the gap between two PEC plates of lateral size a x b with PEC side walls.
Driven with E normal to the plates, its resonances are within 1% of the
parallel-plate lateral modes

    f = (c / 2n) * sqrt((m/a)^2 + (p/b)^2),   m, p >= 1,

and every such mode in the pulse band appears. They depend on n and not on
the film thickness, so they show the film is seen.

SETUP. The cavity IS the grid: ``metal`` walls on all six faces, so the plates
are the z walls and the side walls the x and y walls. (Plates inside a larger
graded domain do not work with the solver's conductor mask: it zeroes every E
component that starts at a flagged node, so a bottom plate at node k also
zeroes Ez in the film cell k..k+1 just above it.) x and y are uniform at the
coarse cell; z comes from the grid builder, which puts at least 4 cells across
the film because it is thinner than the coarse cell. dt then follows the thin
z cells, which is what makes a thin film expensive.

The source is an Ez point current at an off-centre node and the probe records
Ez at another; both overlap every mode of the family (m, p >= 1 have Ez =
sin(m pi x/a) sin(p pi y/b), uniform in z). Modes with m or p = 0 have no Ez
and are not driven; modes varying across the film are cut off far above the
band (c / 2nd). A mode whose source x probe overlap is under EXCITABLE is
listed as not gated, as in V1.

TWO THICKNESSES. The film is run at --thickness-ratio 1 and 0.5 of its default
thickness (a tenth of the coarse cell). The analytic modes are the same for
both, so both runs are gated against them, and the run prints how far each
mode moved between the two. --single runs only the first.

THE GATE. Peaks are matched to modes one-to-one, closest first (V1's lesson:
nearest-mode matching let two peaks claim one mode). A run fails if a matched
peak is more than 1% from its mode, if an excitable mode in the band has no
peak, or if a peak in the band is more than 1% from EVERY analytic mode of the
family (a line no mode explains). The setup is not the spec's (exit 2) if the
film is not at most 1/FAR_THINNER of the coarse cell, if the z axis breaks the
grid builder's own rules, or if the grid's dispersion floor at the top of the
band is over half the tolerance.

DEFAULTS. n = 2, a x b = 40 x 27 coarse cells, the coarse cell 1/20 of the
film wavelength at the top of the band, film 1/10 of the coarse cell (4 cells
of 1/40). Five modes in band: (1,1) (2,1) (3,1) (1,2) (2,2). About 6e3 cells;
dt is ~1/40 of the coarse Courant step, so ~5e5 steps for 120 periods of the
lowest mode: ~3e9 cell-steps for the first run and twice that for the half
thickness. It is a solver run: on the Jetson.

RUN IT ON THE JETSON (tier 2):

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v5'
    git -C <repo> archive <sha> scripts tests/bench_box.py \\
        dockerfiles/maxwell_solver | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v5'
    ssh jetson 'cd ~/mse-bench/fdtd-v5 && \\
      (ulimit -v 5000000; nice -n 10 timeout 30m python3 scripts/fdtd_bench_v5.py)'

Exit 0 PASS, 1 FAIL, 2 the setup is not the spec's, bad arguments, or the
bench did not run (an import, setup or solver crash measured nothing, so it
is never reported as a FAIL). The whole solver directory is archived because
V1's bench_box imports the mse_* modules beside fdtd.
"""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

# Cap threads before numpy loads, as the other benches do.
_THREADS = os.environ.get("MSE_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

# A missing file on the bench host must read as "did not run" (exit 2), not
# as a FAIL: hold the import error and let main() report it.
_IMPORT_ERROR = None
try:
    import fdtd  # noqa: E402
    import fdtd_grid  # noqa: E402

    _spec = importlib.util.spec_from_file_location(
        "fdtd_bench_v1", ROOT / "scripts" / "fdtd_bench_v1.py")
    _v1 = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_v1)
except Exception as _error:  # noqa: BLE001 -- any import failure is a no-run
    _IMPORT_ERROR = _error

#: The spec's numbers.
TOLERANCE = 0.01                    # relative, per mode
#: The bench's own.
FAR_THINNER = 4                     # film at most coarse / FAR_THINNER
EXCITABLE = 0.05                    # source x probe overlap a gated mode needs
DEGENERATE = 1e-3                   # modes closer than this are one line
PEAK_FLOOR = 0.02                   # of the tallest peak in band


def lateral_modes(a: float, b: float, n: float, f_max: float, f_min=0.0):
    """``[(f, [(m, p), ...]), ...]``: the family's lines in f_min..f_max.

    Pairs whose frequencies agree within DEGENERATE are one line (a square
    plate's (1,2) and (2,1)), since no spectrum can tell them apart.
    """
    pairs = []
    m = 1
    while fdtd.C0 / (2 * n) * m / a <= f_max:
        p = 1
        while True:
            f = fdtd.C0 / (2 * n) * math.hypot(m / a, p / b)
            if f > f_max:
                break
            if f >= f_min:
                pairs.append((f, (m, p)))
            p += 1
        m += 1
    pairs.sort()
    lines = []
    for f, pair in pairs:
        if lines and f - lines[-1][0] <= DEGENERATE * lines[-1][0]:
            lines[-1][1].append(pair)
        else:
            lines.append((f, [pair]))
    return lines


def overlap(pair, a, b, source_xy, probe_xy):
    """|Ez shape at the source x Ez shape at the probe| for mode (m, p)."""
    m, p = pair
    value = 1.0
    for x, y in (source_xy, probe_xy):
        value *= math.sin(m * math.pi * x / a) * math.sin(p * math.pi * y / b)
    return abs(value)


def match(peaks, lines):
    """``{peak index: line index}``, one-to-one, closest relative pair first."""
    pairs = sorted((abs(peak - line[0]) / line[0], i, j)
                   for i, peak in enumerate(peaks)
                   for j, line in enumerate(lines))
    taken_peaks, taken_lines, assigned = set(), set(), {}
    for _error, i, j in pairs:
        if i in taken_peaks or j in taken_lines:
            continue
        assigned[i] = j
        taken_peaks.add(i)
        taken_lines.add(j)
    return assigned


def judge(peaks, lines, gated, tolerance: float = TOLERANCE):
    """``(rows, failures)`` for one run.

    ``peaks``: measured frequencies in the band. ``lines``: every analytic line
    of the family in the band. ``gated``: the line indices the run must find.
    Each row is (line index or None, peak or None, relative error or None).
    """
    assigned = match(peaks, lines)
    failures = []
    rows = []
    found = {}
    for i, peak in enumerate(peaks):
        j = assigned.get(i)
        error = None if j is None else (peak - lines[j][0]) / lines[j][0]
        if j is None or abs(error) > tolerance:
            nearest = min(abs(peak - f) / f for f, _ in lines) if lines else math.inf
            if nearest > tolerance:
                failures.append(f"a peak at {peak:.6g} Hz is "
                                f"{nearest:.2%} from every mode of the family")
            elif j is not None:
                failures.append(f"the peak matched to {lines[j][1]} is "
                                f"{error:+.2%} from it")
            if j is None:
                rows.append((None, peak, None))
                continue
        found[j] = peak
        rows.append((j, peak, error))
    for j in gated:
        if j not in found:
            failures.append(f"no peak for mode {lines[j][1]} at "
                            f"{lines[j][0]:.6g} Hz")
            rows.append((j, None, None))
    return rows, failures


def gated_modes(best_overlaps, threshold: float = EXCITABLE):
    """Indices of the lines whose best source x probe overlap reaches the gate."""
    return [j for j, best in enumerate(best_overlaps) if best >= threshold]


def film_problem(film: float, dx: float):
    """The quarter-cell rule: why ``film`` is not "far thinner", or None."""
    if film > dx / FAR_THINNER * (1 + 1e-9):
        return (f"the film {film:.4g} m is not at most 1/{FAR_THINNER} of the "
                "coarse cell")
    return None


def floor_problem(floor: float, tolerance: float = TOLERANCE):
    """The dispersion-floor rule: why the grid is too coarse, or None."""
    if floor > tolerance / 2:
        return (f"the dispersion floor {floor:.3%} is over half the "
                f"{tolerance:.0%} tolerance")
    return None


def v5_verdict(failures, setup_problems):
    """``(verdict, exit code)``: FAIL 1 beats NOT A SPEC SETUP 2 beats PASS 0."""
    if failures:
        return "FAIL", 1
    if setup_problems:
        return "NOT A SPEC SETUP", 2
    return "PASS", 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--index", type=float, default=2.0, help="film index n")
    parser.add_argument("--a-cells", type=int, default=40,
                        help="plate length along x, in coarse cells")
    parser.add_argument("--b-cells", type=int, default=27,
                        help="plate length along y, in coarse cells")
    parser.add_argument("--coarse", type=float, default=1e-3,
                        help="coarse cell, m (the frequencies scale with it)")
    parser.add_argument("--cells-per-wavelength", type=float, default=20.0,
                        help="coarse cells per film wavelength at the top of "
                             "the band")
    parser.add_argument("--film-fraction", type=float, default=0.1,
                        help="film thickness as a fraction of the coarse cell")
    parser.add_argument("--thickness-ratio", type=float, nargs="+",
                        default=[1.0, 0.5],
                        help="runs at these multiples of the film thickness")
    parser.add_argument("--single", action="store_true",
                        help="only the first thickness")
    parser.add_argument("--periods", type=float, default=120.0,
                        help="recorded periods of the lowest mode after the "
                             "source ends")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"))
    return parser


def main(argv=None) -> int:
    if _IMPORT_ERROR is not None:
        print(f"V5 DID NOT RUN: import failed: {_IMPORT_ERROR!r} (copy the "
              f"whole dockerfiles/maxwell_solver directory; see the docstring)")
        return 2
    args = build_parser().parse_args(argv)
    xp = np
    if args.xp == "cupy":
        import cupy as xp  # noqa: F811 -- only here, never at module level
    if args.a_cells < 4 or args.b_cells < 4 or args.index <= 0:
        print("--a-cells and --b-cells need at least 4, --index > 0")
        return 2
    if args.a_cells == args.b_cells:
        print("note: a square plate makes (m, p) and (p, m) one line")

    dx = args.coarse
    a, b = args.a_cells * dx, args.b_cells * dx
    n = args.index
    f_max = fdtd.C0 / (n * args.cells_per_wavelength * dx)
    lines = lateral_modes(a, b, n, f_max)
    if not lines:
        print("no mode of the family below the top of the band; widen it")
        return 2
    f_min = 0.9 * lines[0][0]
    # Gaussian amplitude spectrum exp(-((f - fc)/bw)^2): >= 0.1 at both ends
    centre = 0.5 * (f_min + f_max)
    bandwidth = 0.5 * (f_max - f_min) / math.sqrt(math.log(10.0))
    source_ij = (round(0.37 * args.a_cells), round(0.29 * args.b_cells))
    probe_ij = (round(0.71 * args.a_cells), round(0.63 * args.b_cells))
    source_xy = (source_ij[0] * dx, source_ij[1] * dx)
    probe_xy = (probe_ij[0] * dx, probe_ij[1] * dx)
    best_overlaps = [max(overlap(pair, a, b, source_xy, probe_xy)
                         for pair in pairs) for _f, pairs in lines]
    gated = gated_modes(best_overlaps)
    floor = _v1.dispersion_error(dx, fdtd.C0 / (n * f_max))

    print(f"plates     {args.a_cells} x {args.b_cells} coarse cells of "
          f"{dx:.4g} m, film n = {n:g}")
    print(f"band       {f_min:.6g}-{f_max:.6g} Hz; {len(lines)} line(s), "
          f"{len(gated)} gated (overlap >= {EXCITABLE}); dispersion floor "
          f"{floor:.3%} at the top")
    for j, (f, pairs) in enumerate(lines):
        print(f"  mode {pairs} {f:.6g} Hz overlap {best_overlaps[j]:.3f}"
              + ("" if j in gated else " (not gated)"))

    ratios = args.thickness_ratio[:1] if args.single else args.thickness_ratio
    setup_problems = [p for p in [floor_problem(floor)] if p]
    all_failures = []
    found_by_run = []
    eps_r = n * n
    for ratio in ratios:
        film = args.film_fraction * ratio * dx
        if film_problem(film, dx):
            setup_problems.append(film_problem(film, dx))
        try:
            z = fdtd_grid.graded_axis(film, dx, [fdtd_grid.Layer(0.0, film)])
        except fdtd_grid.GridError as error:
            print(f"refused: {error}")
            return 2
        problems = fdtd_grid.check_axis(z, dx, [fdtd_grid.Layer(0.0, film)])
        setup_problems += [f"z axis: {p}" for p in problems]
        shape = (args.a_cells + 1, args.b_cells + 1, z.size)
        spacings = (np.full(args.a_cells, dx), np.full(args.b_cells, dx),
                    np.diff(z))
        nodes = (np.linspace(0, a, shape[0]), np.linspace(0, b, shape[1]), z)
        eps, _ = fdtd_grid.average_eps(nodes, lambda p: np.zeros(len(p), int),
                                       [eps_r], samples=1)
        k_mid = (z.size - 1) // 2
        source = fdtd.Source((source_ij[0], source_ij[1], k_mid), component="z",
                             frequency=centre, bandwidth=bandwidth)
        probe = (probe_ij[0], probe_ij[1], max(k_mid - 1, 0), "z")
        dt = fdtd.courant_dt(dx, dx, float(np.diff(z).min()))
        record = args.periods / lines[0][0]
        start_step = int(math.ceil(source.end_time() / dt))
        n_steps = start_step + int(math.ceil(record / dt))
        cells = int(np.prod(shape))
        print(f"run        film {film:.4g} m ({z.size - 1} cells of "
              f"{np.diff(z).min():.4g} m), grid {shape[0]} x {shape[1]} x "
              f"{shape[2]} = {cells}, {n_steps} steps = "
              f"{cells * n_steps:.3g} cell-steps", flush=True)
        clock = time.perf_counter()
        run = fdtd.simulate(eps.astype(np.float32), source, n_steps, [probe],
                            spacings=spacings, boundary="metal", xp=xp)
        print(f"           {time.perf_counter() - clock:.1f} s", flush=True)
        trace = np.asarray(run["traces"][0], dtype=np.float64)[start_step:]
        frequencies, amplitude = fdtd.spectrum(trace, run["dt"])
        amplitude = np.asarray(amplitude)
        peaks = [f for f, _amp in _v1.find_peaks(
            frequencies, amplitude, f_min, f_max, len(lines) + 4,
            floor=PEAK_FLOOR)]
        rows, failures = judge(peaks, lines, gated)
        found = {}
        for j, peak, error in rows:
            if j is None:
                print(f"  peak {peak:.6g} Hz matches no mode")
            elif peak is None:
                print(f"  mode {lines[j][1]} {lines[j][0]:.6g} Hz: NOT FOUND")
            else:
                found[j] = peak
                print(f"  mode {lines[j][1]} {lines[j][0]:.6g} Hz: peak "
                      f"{peak:.6g} Hz, {error:+.3%}")
        for failure in failures:
            print(f"  FAIL {failure}")
        all_failures += [f"film x{ratio:g}: {f}" for f in failures]
        found_by_run.append(found)

    if len(found_by_run) == 2:
        both = sorted(set(found_by_run[0]) & set(found_by_run[1]))
        shifts = [abs(found_by_run[1][j] - found_by_run[0][j]) / lines[j][0]
                  for j in both]
        if shifts:
            print(f"thickness  the {len(both)} mode(s) found in both runs moved "
                  f"at most {max(shifts):.3%} when the film thickness changed")

    verdict, code = v5_verdict(all_failures, setup_problems)
    for problem in setup_problems:
        print(f"  setup: {problem}")
    print(f"V5 {verdict}: {len(ratios)} run(s), {len(gated)} gated mode(s) each, "
          f"tolerance {TOLERANCE:.0%}, {len(all_failures)} failure(s)")
    return code


def run(argv=None) -> int:
    """main(), with any crash reported as "did not run" (exit 2): a crash
    measured nothing, and exit 1 would read as a FAIL of the solver."""
    try:
        return main(argv)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("V5 DID NOT RUN: the bench crashed (exit 2, not a FAIL)")
        return 2


if __name__ == "__main__":
    sys.exit(run())
