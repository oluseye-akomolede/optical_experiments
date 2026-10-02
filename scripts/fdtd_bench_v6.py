#!/usr/bin/env python3
"""Benchmark V6: Bloch-periodic bands against exact lines and the pc solver.

The spec's gate: a pulse run in one cell of a 2-D crystal extruded along z,
with Bloch-periodic x and y faces at a chosen k and periodic z faces at
kz = 0, gives spectrum peaks within 1% of the photonic-crystal band solver's
frequencies at that k, for TM (E along z) and TE (H along z), and every band
in the pulse band appears.

TWO PARTS.

1. HOMOGENEOUS (the k-to-phase check). The cell holds one medium of index n,
   so the bands are known exactly: the folded light lines

       f = c |k + G| / (n a),   G = (p, q) 2 pi / a,

   at k = (kx, ky) 2 pi / a. A wrong mapping from k to the face phase (a
   period off by a cell, a factor of 2 pi, a dropped axis) moves every line
   but G = 0 by far more than 1%; the symmetry checks of 6.1b (a full turn,
   +pi / -pi, k / -k) cannot see it. Run at several non-zero k below the
   zone edge (the paired cos/sin path) and at the zone edge (the real path,
   the sign flip), TM only: in a homogeneous medium TE has the same lines.
2. CRYSTAL (the spec's benchmark). A square lattice of rods (eps 8.9,
   r = 0.2 a) in air, TM and TE, at the zone-edge points X and M and one
   generic k, against ``pc.bands_from_mesh`` for the same crystal.

SETUP. One cell, ``--cells`` per lattice constant along x and y, two along z
(periodic, kz = 0, so every z-varying mode is cut off far above the band).
eps is the solver's own sub-cell average of the rods. The source is a point
current at an off-symmetry node (Ez for TM, Ex for TE); four probes at other
off-symmetry nodes (Ez; Ex and Ey for TE) and their spectra summed as power,
so no line is lost to a probe on its node. A homogeneous line whose source x
probe overlap (|sum over its G of cos(2 pi (k + G) . d)| / its multiplicity,
d the probe's offset from the source in lattice units, the best of the four
probes) is under EXCITABLE is listed as not gated, as in V1 and V5. Every pc
band in the pulse band is gated: the source and probes sit off every
symmetry line of the cell.

THE GATE. V5's: peaks are matched to lines one-to-one, closest first. A run
fails if a matched peak is more than 1% from its line, if a gated line has
no peak, or if a peak is more than 1% from every line (a line nothing
explains). The setup is not the spec's (exit 2) if a homogeneous run's Yee
dispersion floor at the top of its band is over half the tolerance.

TIER 1. ``--tiny`` runs the homogeneous part alone, at 20 cells and a band
low enough for its dispersion floor, at k = (0.1, 0), (0.3, 0.2), X and M:
26 s on the VM under the usual caps (measured 2026-10-02). ``tests/test_fdtd_bench_v6.py`` runs it.

DEFAULTS (the full run, tier 2). 64 cells per lattice constant; homogeneous
n = 1.5 at kx = 0.1, 0.25, 0.4, 0.5 (X), k = (0.5, 0.5) (M) and (0.3, 0.2);
the crystal at X, M and (0.25, 0.1); pc at 64 divisions (within 0.1% of 96);
the band a / lambda = 0.15-0.8; 100 periods of the lowest band frequency
recorded after the source ends (60 already separate the closest pc pair,
1.5% apart). 15 runs; paired runs step two field sets (~2.4x the time).

Why 64 cells: the crystal's TE band 1 at X converges at about first order
(the scalar eps average across the rod's edge): -2.77% at 24 cells, -1.53% at
32, -0.95% at 48 (measured on the VM, 60 periods), so ~-0.7% at 64. TM is
within 0.4% at 32. The homogeneous part needs no more than 48 (its floor).

RUN IT ON THE JETSON (tier 2), as three commands so each stays inside the
30 min tier-2 timeout. Measured on the VM (nice 19, 2 threads) the parts take
about 10, 9 and 11 min; allow up to twice that on the Jetson's CPU:

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v6'
    git -C <repo> archive <sha> scripts tests/bench_box.py \\
        dockerfiles/maxwell_solver | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v6'
    # 1: homogeneous, exact |k + G| / n at six k (48 cells is enough here)
    ssh jetson 'cd ~/mse-bench/fdtd-v6 && (ulimit -v 5000000; nice -n 10 \\
      timeout 30m python3 scripts/fdtd_bench_v6.py --skip-crystal --cells 48)'
    # 2: crystal against the pc solver at X and M, TM and TE
    ssh jetson 'cd ~/mse-bench/fdtd-v6 && (ulimit -v 5000000; nice -n 10 \\
      timeout 30m python3 scripts/fdtd_bench_v6.py --skip-homogeneous \\
      --k 0.5 0 --k 0.5 0.5)'
    # 3: crystal at a general k (paired runs)
    ssh jetson 'cd ~/mse-bench/fdtd-v6 && (ulimit -v 5000000; nice -n 10 \\
      timeout 30m python3 scripts/fdtd_bench_v6.py --skip-homogeneous \\
      --k 0.25 0.1)'

V6 passes when all three exit 0. A bare ``fdtd_bench_v6.py`` runs all 15 in
one go (~30 min on the VM): too long for one tier-2 command.

Exit 0 PASS, 1 FAIL, 2 the setup is not the spec's, bad arguments, or the
bench did not run (an import, setup or solver-setup crash measured nothing,
so it is never reported as a FAIL). The whole solver directory is archived
because V1's bench_box imports the mse_* modules beside fdtd.
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


def _load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# A missing file on the bench host must read as "did not run" (exit 2), not
# as a FAIL: hold the import error and let main() report it.
_IMPORT_ERROR = None
try:
    import fdtd  # noqa: E402
    import fdtd_grid  # noqa: E402
    import pc  # noqa: E402

    _v1 = _load("fdtd_bench_v1")
    _v5 = _load("fdtd_bench_v5")
except Exception as _error:  # noqa: BLE001 -- any import failure is a no-run
    _IMPORT_ERROR = _error

#: The spec's numbers.
TOLERANCE = 0.01                    # relative, per band
#: The bench's own.
EXCITABLE = 0.05                    # source x probe overlap a gated line needs
DEGENERATE = 1e-3                   # lines closer than this are one line
PEAK_FLOOR = 0.02                   # of the tallest peak in band
ROD_EPS, ROD_RADIUS = 8.9, 0.2      # the crystal: rods in air, r / a

SOURCE_AT = (0.13, 0.29)            # lattice units, off every symmetry line
PROBES_AT = ((0.71, 0.37), (0.41, 0.83), (0.87, 0.61), (0.23, 0.57))

FULL = dict(cells=64, index=1.5, band=(0.15, 0.8), periods=100.0,
            homogeneous=((0.1, 0.0), (0.25, 0.0), (0.4, 0.0), (0.5, 0.0),
                         (0.5, 0.5), (0.3, 0.2)),
            crystal=((0.5, 0.0), (0.5, 0.5), (0.25, 0.1)), divisions=64)
TINY = dict(cells=20, index=1.5, band=(0.2, 0.7), periods=60.0,
            homogeneous=((0.1, 0.0), (0.3, 0.2), (0.5, 0.0), (0.5, 0.5)),
            crystal=(), divisions=0)


def homogeneous_lines(k, n, band, reach: int = 4):
    """The folded light lines in ``band`` (a / lambda): ``[(f, [G, ...])]``.

    ``k`` and every ``G`` are in units of 2 pi / a, so f a / c = |k + G| / n.
    Lines within DEGENERATE (relative) of each other merge into one, as V5's
    degenerate plate modes do.
    """
    kx, ky = k
    raw = sorted((math.hypot(kx + p, ky + q) / n, (p, q))
                 for p in range(-reach, reach + 1)
                 for q in range(-reach, reach + 1))
    lines = []
    for f, g in raw:
        if not band[0] <= f <= band[1]:
            continue
        if lines and abs(f - lines[-1][0]) <= DEGENERATE * lines[-1][0]:
            lines[-1][1].append(g)
            continue
        lines.append((f, [g]))
    return lines


def overlap(k, gs, source=SOURCE_AT, probes=PROBES_AT):
    """The best probe's source x probe overlap of one homogeneous line.

    The cos set is driven by a real point current, so a plane wave k + G
    reaches a probe at offset d with weight cos(2 pi (k + G) . d); a
    degenerate line sums its G at one frequency. Normalised by the line's
    multiplicity, so 1 is every G in phase.
    """
    best = 0.0
    for probe in probes:
        d = (probe[0] - source[0], probe[1] - source[1])
        total = sum(math.cos(2 * math.pi * ((k[0] + p) * d[0] + (k[1] + q) * d[1]))
                    for p, q in gs)
        best = max(best, abs(total) / len(gs))
    return best


def cell_eps(cells: int, a: float, index=None):
    """The one-cell grid: ``(nodes, eps)`` with ``cells`` per a along x and y
    and two cells along z. ``index`` fills it with one medium; ``None`` is the
    rod crystal, averaged by the solver's own ``average_eps``."""
    dx = a / cells
    nodes = (np.linspace(0.0, a, cells + 1), np.linspace(0.0, a, cells + 1),
             np.linspace(0.0, 2 * dx, 3))
    if index is not None:
        shape = tuple(len(v) for v in nodes)
        return nodes, np.full(shape, index * index, dtype=np.float32)

    def rods(points):
        x = np.mod(points[:, 0], a) - a / 2
        y = np.mod(points[:, 1], a) - a / 2
        return (x * x + y * y < (ROD_RADIUS * a) ** 2).astype(int)

    eps, _ = fdtd_grid.average_eps(nodes, rods, [1.0, ROD_EPS], samples=3)
    return nodes, eps.astype(np.float32)


def run_cell(eps, cells: int, a: float, k, polarisation: str, band, periods,
             xp=np):
    """One Bloch run; returns ``(peaks in a / lambda, steps, seconds)``."""
    dx = a / cells
    lo, hi = band
    centre = 0.5 * (lo + hi) * fdtd.C0 / a
    # Gaussian amplitude spectrum exp(-((f - fc)/bw)^2): >= 0.1 at both ends
    bandwidth = 0.5 * (hi - lo) * fdtd.C0 / a / math.sqrt(math.log(10.0))
    at = lambda xy: (round(xy[0] * cells), round(xy[1] * cells))  # noqa: E731
    if polarisation == "TM":
        component, probe_components = "z", ("z",)
    else:
        component, probe_components = "x", ("x", "y")
    source = fdtd.Source((*at(SOURCE_AT), 1), frequency=centre,
                         bandwidth=bandwidth, component=component)
    probes = [(*at(p), 1, c) for c in probe_components for p in PROBES_AT]
    dt = fdtd.courant_dt(dx, dx, dx)
    start = int(math.ceil(source.end_time() / dt))
    steps = start + int(math.ceil(periods * a / (lo * fdtd.C0) / dt))
    bloch = (2 * math.pi * k[0] / a, 2 * math.pi * k[1] / a, 0.0)
    clock = time.perf_counter()
    run = fdtd.simulate(eps, source, steps, probes, spacings=(dx, dx, dx),
                        boundary="periodic", bloch=bloch, xp=xp)
    seconds = time.perf_counter() - clock
    power = 0.0
    for trace in run["traces"]:
        frequencies, amplitude = fdtd.spectrum(
            np.asarray(trace, dtype=np.float64)[start:], run["dt"])
        power = power + np.asarray(amplitude) ** 2
    scale = a / fdtd.C0
    peaks = sorted(f * scale for f, _amp in _v1.find_peaks(
        frequencies, np.sqrt(power), lo / scale, hi / scale, 16,
        floor=PEAK_FLOOR))
    return peaks, steps, seconds


def crystal_lines(k, polarisation: str, band, divisions: int):
    """The pc solver's bands at ``k`` inside ``band``, as one-band lines."""
    _k, bands = pc.bands_from_mesh("square", divisions, ROD_RADIUS,
                                   math.sqrt(ROD_EPS), 1.0, [k], 10,
                                   polarisation=polarisation)
    lines = []
    for f in sorted(float(b) for b in bands[0]):
        if not band[0] <= f <= band[1]:
            continue
        if lines and abs(f - lines[-1][0]) <= DEGENERATE * lines[-1][0]:
            lines[-1][1].append("band")
            continue
        lines.append((f, ["band"]))
    return lines


def floor_problem(cells: int, n: float, top: float,
                  tolerance: float = TOLERANCE):
    """Why a homogeneous run's grid is too coarse for its band, or None."""
    floor = _v1.dispersion_error(1.0 / cells, 1.0 / (n * top))
    if floor > tolerance / 2:
        return (f"the dispersion floor {floor:.3%} at a / lambda = {top:g} is "
                f"over half the {tolerance:.0%} tolerance")
    return None


def report(label, lines, gated, peaks):
    """Print one run's rows; returns its failures, prefixed with ``label``."""
    as_lines = [(f, str(gs)) for f, gs in lines]
    rows, failures = _v5.judge(peaks, as_lines, gated)
    for j, peak, error in rows:
        if j is None:
            print(f"  peak {peak:.5f} matches no line")
        elif peak is None:
            print(f"  line {as_lines[j][0]:.5f} {as_lines[j][1]}: NOT FOUND")
        else:
            print(f"  line {as_lines[j][0]:.5f} {as_lines[j][1]}: peak "
                  f"{peak:.5f}, {error:+.3%}")
    for failure in failures:
        print(f"  FAIL {failure}")
    return [f"{label}: {f}" for f in failures]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--tiny", action="store_true",
                        help="tier 1: the homogeneous part at 20 cells")
    parser.add_argument("--cells", type=int, help="cells per lattice constant")
    parser.add_argument("--periods", type=float,
                        help="recorded periods of the band's lowest frequency")
    parser.add_argument("--divisions", type=int, help="pc mesh divisions")
    parser.add_argument("--k", type=float, nargs=2, action="append",
                        metavar=("KX", "KY"),
                        help="run only these k-points (units of 2 pi / a), "
                             "for both parts; repeat for more")
    parser.add_argument("--skip-homogeneous", action="store_true")
    parser.add_argument("--skip-crystal", action="store_true")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"))
    return parser


def main(argv=None) -> int:
    if _IMPORT_ERROR is not None:
        print(f"V6 DID NOT RUN: import failed: {_IMPORT_ERROR!r} (copy the "
              f"whole dockerfiles/maxwell_solver directory; see the docstring)")
        return 2
    args = build_parser().parse_args(argv)
    setup = dict(TINY if args.tiny else FULL)
    for key in ("cells", "periods", "divisions"):
        if getattr(args, key) is not None:
            setup[key] = getattr(args, key)
    if setup["cells"] < 8:
        print("--cells needs at least 8")
        return 2
    xp = np
    if args.xp == "cupy":
        import cupy as xp  # noqa: F811 -- only here, never at module level
    a = 1e-3                        # the lattice constant; bands are a / lambda
    cells, band, n = setup["cells"], setup["band"], setup["index"]
    if args.k:
        setup["homogeneous"] = setup["crystal"] = [tuple(k) for k in args.k]
    homogeneous = () if args.skip_homogeneous else setup["homogeneous"]
    crystal = () if args.skip_crystal else setup["crystal"]
    print(f"cell       {cells} x {cells} x 2, band a / lambda {band[0]:g}-"
          f"{band[1]:g}, {setup['periods']:g} periods recorded")

    setup_problems, failures, runs = [], [], 0
    if homogeneous:
        problem = floor_problem(cells, n, band[1])
        if problem:
            setup_problems.append(f"homogeneous: {problem}")
        _nodes, eps = cell_eps(cells, a, index=n)
        for k in homogeneous:
            lines = homogeneous_lines(k, n, band)
            overlaps = [overlap(k, gs) for _f, gs in lines]
            gated = _v5.gated_modes(overlaps, EXCITABLE)
            label = f"homogeneous n={n:g} k=({k[0]:g}, {k[1]:g}) TM"
            try:
                peaks, steps, seconds = run_cell(eps, cells, a, k, "TM", band,
                                                 setup["periods"], xp)
            except Exception:  # noqa: BLE001 -- the solver's failure IS the result
                import traceback
                traceback.print_exc()
                print(f"V6 FAIL: the solver raised during {label}")
                return 1
            runs += 1
            print(f"run        {label}: {steps} steps, {seconds:.1f} s; "
                  f"{len(lines)} line(s), {len(gated)} gated", flush=True)
            for j, (f, gs) in enumerate(lines):
                if j not in gated:
                    print(f"  line {f:.5f} {gs} overlap {overlaps[j]:.3f} "
                          "(not gated)")
            failures += report(label, lines, gated, peaks)

    if crystal:
        _nodes, eps = cell_eps(cells, a)
        for polarisation in ("TM", "TE"):
            for k in crystal:
                lines = crystal_lines(k, polarisation, band, setup["divisions"])
                gated = list(range(len(lines)))
                label = f"crystal k=({k[0]:g}, {k[1]:g}) {polarisation}"
                try:
                    peaks, steps, seconds = run_cell(
                        eps, cells, a, k, polarisation, band,
                        setup["periods"], xp)
                except Exception:  # noqa: BLE001
                    import traceback
                    traceback.print_exc()
                    print(f"V6 FAIL: the solver raised during {label}")
                    return 1
                runs += 1
                print(f"run        {label}: {steps} steps, {seconds:.1f} s; "
                      f"{len(lines)} pc band(s) at {setup['divisions']} "
                      "divisions", flush=True)
                failures += report(label, lines, gated, peaks)

    if runs == 0:
        print("nothing to run")
        return 2
    verdict, code = _v5.v5_verdict(failures, setup_problems)
    for problem in setup_problems:
        print(f"  setup: {problem}")
    print(f"V6 {verdict}: {runs} run(s), tolerance {TOLERANCE:.0%}, "
          f"{len(failures)} failure(s)")
    return code


def run(argv=None) -> int:
    """main(), with a crash in the bench's setup reported as "did not run"
    (exit 2): it measured nothing, and exit 1 would read as a FAIL of the
    solver. A solver exception mid-run never gets here; main() reports it as a
    FAIL (exit 1)."""
    try:
        return main(argv)
    except Exception:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("V6 DID NOT RUN: the bench crashed (exit 2, not a FAIL)")
        return 2


if __name__ == "__main__":
    sys.exit(run())
