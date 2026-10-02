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

THE GATE. Peaks are matched to lines one-to-one in frequency order. A run
fails if a matched peak is more than 1% from its line, if a gated line has
no peak, or if a peak is more than 1% from every line (a line nothing
explains). Crystal lines also pass a two-resolution convergence gate: each is
run at ``--coarse-cells`` and ``--cells`` and must be found at both and move
toward its pc value between them (unless both are within SETTLED, the
reference's own accuracy), so a line under 1% at one resolution is not enough.
Peaks are a / lambda. The setup is not the spec's (exit 2) if a homogeneous
run's Yee dispersion floor at the top of its band is over half the tolerance.

TIER 1. ``--tiny`` runs the homogeneous part alone, at 20 cells and a band
low enough for its dispersion floor, at k = (0.1, 0), (0.3, 0.2), X and M:
26 s on the VM under the usual caps (measured 2026-10-02). ``tests/test_fdtd_bench_v6.py`` runs it.

DEFAULTS (the full run, tier 2). Homogeneous n = 1.5 at kx = 0.1, 0.25, 0.4,
0.5 (X), k = (0.5, 0.5) (M) and (0.3, 0.2); the crystal at X, M and
(0.25, 0.1), at 48 and 64 cells with eps sampled EPS_SAMPLES per cell edge;
pc at 128 divisions; the band a / lambda = 0.15-0.8; 150 periods of the
lowest band frequency recorded after the source ends, so the spectrum's
resolution is not part of the error. Paired (general k) runs step two field
sets (~2.4x the time).

The crystal's TE band 1 at X is the slow line (measured on the VM, 150
periods, 9 samples): -1.166% at 48 cells, -0.959% at 64, -0.620% at 96,
first order through 64 and 96 to about +0.06%. At 64 cells it is under 1% by
only 0.04%; the convergence gate is what shows it is heading to the pc value.
TM lines are within 0.3% at 48.

Sampling the eps more finely is settled well before this (measured the same
way): at 48 cells, 9, 15, 18 and 24 samples per cell edge read band 1 as
-1.166%, -1.102%, -1.144%, -1.130% (a spread of 0.064%, against the gate's
0.25%). Fewer than that is not a coarser answer but a wrong one, and in the
flattering direction: 3 samples read -0.927% at 48 cells -- smaller than the
9-sample error, so aliasing at the rod edge hides the coarse line's true
error instead of converging it. 9 is the bench's choice for that reason.

RUN IT ON THE JETSON (tier 2), as six commands so each stays inside the
30 min tier-2 timeout. Estimated from VM runs (nice 19, 2 threads): about
7, 8, 10, 10, 12 and 12 min; allow up to twice that on the Jetson's CPU:

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v6'
    git -C <repo> archive <sha> scripts tests/bench_box.py \\
        dockerfiles/maxwell_solver | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v6'
    # 1-2: homogeneous, exact |k + G| / n at six k (48 cells is enough here)
    # 3-6: crystal against the pc solver, 48 and 64 cells each
    for args in "--skip-crystal --cells 48 --k 0.1 0 --k 0.25 0 --k 0.4 0" \\
                "--skip-crystal --cells 48 --k 0.5 0 --k 0.5 0.5 --k 0.3 0.2" \\
                "--skip-homogeneous --polarisation TM --k 0.5 0 --k 0.5 0.5" \\
                "--skip-homogeneous --polarisation TE --k 0.5 0 --k 0.5 0.5" \\
                "--skip-homogeneous --polarisation TM --k 0.25 0.1" \\
                "--skip-homogeneous --polarisation TE --k 0.25 0.1"; do
      ssh jetson "cd ~/mse-bench/fdtd-v6 && (ulimit -v 5000000; nice -n 10 \\
        timeout 30m python3 scripts/fdtd_bench_v6.py $args)" || echo "FAILED: $args"
    done

V6 passes when all six exit 0. A bare ``fdtd_bench_v6.py`` runs everything
in one go: too long for one tier-2 command.

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
PAD = 8                             # zero padding of each recorded trace
SETTLED = 0.0025                    # both resolutions this close: at the reference's accuracy
COARSE_MATCH = 0.05                 # a coarse peak further than this is no match
PEAK_FLOOR = 0.002                  # of the tallest peak in band (TE2 at X: 2.1% at 64 cells)
SIDELOBE_BINS = 2.37                # Hann's first sidelobe, in record bins
SIDELOBE_DB = 31.5                  # ... and how far down it is
SIDELOBE_MARGIN = 2.0               # a peak this far under the envelope is a sidelobe
SIDELOBE_REACH = 20.0               # record bins; past this the envelope is under the floor
ROD_EPS, ROD_RADIUS = 8.9, 0.2      # the crystal: rods in air, r / a
EPS_SAMPLES = 9                     # per cell edge; at 48 cells TE1 at X reads
                                    # -0.93% at 3 samples, -1.17% at 9 and
                                    # -1.10/-1.14/-1.13% at 15/18/24 (see above)

SOURCE_AT = (0.13, 0.29)            # lattice units, off every symmetry line
PROBES_AT = ((0.71, 0.37), (0.41, 0.83), (0.87, 0.61), (0.23, 0.57))

FULL = dict(cells=64, index=1.5, band=(0.15, 0.8), periods=150.0,
            homogeneous=((0.1, 0.0), (0.25, 0.0), (0.4, 0.0), (0.5, 0.0),
                         (0.5, 0.5), (0.3, 0.2)),
            crystal=((0.5, 0.0), (0.5, 0.5), (0.25, 0.1)), divisions=128,
            coarse_cells=48)
TINY = dict(cells=20, index=1.5, band=(0.2, 0.7), periods=60.0,
            homogeneous=((0.1, 0.0), (0.3, 0.2), (0.5, 0.0), (0.5, 0.5)),
            crystal=(), divisions=0, coarse_cells=0)


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

    eps, _ = fdtd_grid.average_eps(nodes, rods, [1.0, ROD_EPS],
                                 samples=EPS_SAMPLES)
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
        frequencies, amplitude = padded_spectrum(
            np.asarray(trace, dtype=np.float64)[start:], run["dt"])
        power = power + amplitude ** 2
    scale = a / fdtd.C0
    found = _v1.find_peaks(frequencies, np.sqrt(power), lo / scale,
                           hi / scale, 16, floor=PEAK_FLOOR)
    record_bin = 1.0 / ((steps - start) * run["dt"])
    peaks = sorted(f * scale for f, _amp in drop_sidelobes(found, record_bin))
    return peaks, steps, seconds


def sidelobe_envelope(distance_bins: float) -> float:
    """Hann's sidelobe envelope relative to its main peak, ``distance_bins``
    record bins (1 / record length) out: 31.5 dB down at the first sidelobe
    (2.37 bins), then falling 18 dB per octave (1 / d^3). Inside the first
    sidelobe it is the main lobe itself, so 1."""
    if distance_bins < SIDELOBE_BINS:
        return 1.0
    return 10.0 ** (-SIDELOBE_DB / 20.0) * (SIDELOBE_BINS / distance_bins) ** 3


def drop_sidelobes(peaks, record_bin: float, margin: float = SIDELOBE_MARGIN,
                   reach: float = SIDELOBE_REACH):
    """``peaks`` ((frequency, amplitude) pairs) without the Hann window's own
    sidelobes: a peak within ``reach`` record bins of a stronger one and no
    taller than ``margin`` times that one's sidelobe envelope at its distance.
    Zero padding resolves the sidelobes (2.37 bins out and 31.5 dB down, 3.47
    and 41 dB, ...) into peaks, and the pairing once took one for a band. A
    peak above the envelope, or further out, is a line and is kept."""
    kept = []
    for f, amp in peaks:
        lobe = False
        for g, big in peaks:
            d = abs(f - g) / record_bin
            if big > amp and 0 < d <= reach \
                    and amp <= margin * big * sidelobe_envelope(d):
                lobe = True
                break
        if not lobe:
            kept.append((f, amp))
    return kept


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


def padded_spectrum(trace, dt: float, pad: int = PAD):
    """Hann-windowed amplitude spectrum, zero-padded ``pad`` times, so the
    peak's position is not quantised to the record's bin 1 / (n dt): at 60
    periods that bin is ~1% of a line, as large as the tolerance."""
    values = np.asarray(trace, dtype=np.float64) * np.hanning(len(trace))
    n = pad * len(values)
    return np.fft.rfftfreq(n, d=dt), np.abs(np.fft.rfft(values, n=n))


def floor_problem(cells: int, n: float, top: float,
                  tolerance: float = TOLERANCE):
    """Why a homogeneous run's grid is too coarse for its band, or None."""
    floor = _v1.dispersion_error(1.0 / cells, 1.0 / (n * top))
    if floor > tolerance / 2:
        return (f"the dispersion floor {floor:.3%} at a / lambda = {top:g} is "
                f"over half the {tolerance:.0%} tolerance")
    return None


def pair_peaks(peaks, lines, cap: float = COARSE_MATCH):
    """``{line index: peak}`` for the lines a peak matches, in frequency
    order: the monotone one-to-one pairing with the least summed
    |error|, a line left unpaired costing ``cap`` and no pair further than
    ``cap``. Not V5's closest-first matcher: a grid shifts neighbouring lines
    the same way, and closest-first swaps two lines closer than twice that
    shift (0.8% apart, 0.9% shifted), so the gate would track the wrong line.
    ``lines`` must be sorted; extra peaks (sidelobes) cost nothing."""
    order = sorted(range(len(peaks)), key=lambda i: peaks[i])
    p = [peaks[i] for i in order]
    n, m = len(p), len(lines)
    # best[i][j]: least cost pairing the first i peaks with the first j lines
    best = [[0.0] * (m + 1) for _ in range(n + 1)]
    step = [[None] * (m + 1) for _ in range(n + 1)]
    for j in range(1, m + 1):
        best[0][j], step[0][j] = j * cap, "line"
    for i in range(1, n + 1):
        best[i][0], step[i][0] = 0.0, "peak"
        for j in range(1, m + 1):
            f = lines[j - 1][0]
            options = [(best[i - 1][j], "peak"), (best[i][j - 1] + cap, "line")]
            error = abs(p[i - 1] - f) / f
            if error <= cap:
                options.append((best[i - 1][j - 1] + error, "pair"))
            best[i][j], step[i][j] = min(options, key=lambda o: o[0])
    paired, i, j = {}, n, m
    while i > 0 or j > 0:
        move = step[i][j]
        if move == "pair":
            paired[j - 1] = p[i - 1]
            i, j = i - 1, j - 1
        elif move == "peak":
            i -= 1
        else:
            j -= 1
    return paired


def line_errors(peaks, lines, cap: float = COARSE_MATCH):
    """``{line index: relative error}`` over ``pair_peaks``."""
    return {j: (peak - lines[j][0]) / lines[j][0]
            for j, peak in pair_peaks(peaks, lines, cap).items()}


def convergence(lines, coarse_peaks, fine_peaks, coarse: int, fine: int,
                settled: float = SETTLED):
    """The two-resolution gate on crystal lines: ``(rows, failures)``.

    Each line must be found at both resolutions and approach the pc value
    between them, |error at ``fine``| < |error at ``coarse``|, unless both
    are within ``settled`` (the pc reference's and the spectrum's own
    accuracy, where a smaller step is noise). A line under 1% at the fine
    resolution that does not approach its value has shown no convergence.
    Each row is (line index, coarse error or None, fine error or None,
    first-order extrapolation or None)."""
    at_coarse = line_errors(coarse_peaks, lines)
    at_fine = line_errors(fine_peaks, lines)
    rows, failures = [], []
    for j, (f, _gs) in enumerate(lines):
        e_c, e_f = at_coarse.get(j), at_fine.get(j)
        limit = None
        if e_c is None or e_f is None:
            missing = coarse if e_c is None else fine
            failures.append(f"line {f:.5f}: no peak at {missing} cells, so "
                            "its convergence is not shown")
        else:
            # e(h) = e0 + c h through both points, h = 1 / cells
            limit = (fine * e_f - coarse * e_c) / (fine - coarse)
            if not (abs(e_c) <= settled and abs(e_f) <= settled) \
                    and abs(e_f) >= abs(e_c):
                failures.append(
                    f"line {f:.5f} does not approach the pc value: "
                    f"{e_c:+.3%} at {coarse} cells, {e_f:+.3%} at {fine}")
        rows.append((j, e_c, e_f, limit))
    return rows, failures


def judge(peaks, lines, gated, tolerance: float = TOLERANCE):
    """V5's gate with ``line_errors``' frequency-ordered pairing: ``(rows,
    failures)``, each row (line index or None, peak or None, error or None).
    A pair more than ``tolerance`` apart, a gated line with no peak, and a peak
    ``tolerance`` from every line each fail. Frequencies are a / lambda."""
    paired = pair_peaks(peaks, lines)
    claimed = sorted(paired.values())
    rows, failures = [], []
    for peak in sorted(peaks):
        if claimed and claimed[0] == peak:
            claimed.pop(0)
            continue
        rows.append((None, peak, None))
        nearest = min((abs(peak - f) / f for f, _ in lines), default=math.inf)
        if nearest > tolerance:
            failures.append(f"a peak at a / lambda = {peak:.5f} is "
                            f"{nearest:.2%} from every line")
    for j, (f, gs) in enumerate(lines):
        if j in paired:
            error = (paired[j] - f) / f
            rows.append((j, paired[j], error))
            if abs(error) > tolerance:
                failures.append(f"the peak paired with line {f:.5f} {gs} is "
                                f"{error:+.2%} from it")
        elif j in gated:
            rows.append((j, None, None))
            failures.append(f"no peak for line {gs} at a / lambda = {f:.5f}")
    return rows, failures


def report(label, lines, gated, peaks):
    """Print one run's rows; returns its failures, prefixed with ``label``."""
    as_lines = [(f, str(gs)) for f, gs in lines]
    rows, failures = judge(peaks, as_lines, gated)
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
    parser.add_argument("--coarse-cells", type=int,
                        help="the crystal's second, coarser resolution for "
                             "the convergence gate")
    parser.add_argument("--k", type=float, nargs=2, action="append",
                        metavar=("KX", "KY"),
                        help="run only these k-points (units of 2 pi / a), "
                             "for both parts; repeat for more")
    parser.add_argument("--polarisation", choices=("TM", "TE"),
                        help="the crystal in one polarisation only")
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
    for key in ("cells", "periods", "divisions", "coarse_cells"):
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

    coarse = setup["coarse_cells"]
    if crystal and not 8 <= coarse < cells:
        print(f"--coarse-cells must be at least 8 and under --cells ({cells})")
        return 2
    if crystal:
        eps_at = {c: cell_eps(c, a)[1] for c in (coarse, cells)}
        for polarisation in ((args.polarisation,) if args.polarisation
                             else ("TM", "TE")):
            for k in crystal:
                lines = crystal_lines(k, polarisation, band, setup["divisions"])
                gated = list(range(len(lines)))
                label = f"crystal k=({k[0]:g}, {k[1]:g}) {polarisation}"
                peaks_at = {}
                for c in (coarse, cells):
                    try:
                        peaks, steps, seconds = run_cell(
                            eps_at[c], c, a, k, polarisation, band,
                            setup["periods"], xp)
                    except Exception:  # noqa: BLE001
                        import traceback
                        traceback.print_exc()
                        print(f"V6 FAIL: the solver raised during {label} "
                              f"at {c} cells")
                        return 1
                    runs += 1
                    peaks_at[c] = peaks
                    print(f"run        {label} at {c} cells: {steps} steps, "
                          f"{seconds:.1f} s; {len(lines)} pc band(s) at "
                          f"{setup['divisions']} divisions", flush=True)
                failures += report(label, lines, gated, peaks_at[cells])
                rows, problems = convergence(lines, peaks_at[coarse],
                                             peaks_at[cells], coarse, cells)
                for j, e_c, e_f, limit in rows:
                    shown = lambda e: "none" if e is None else f"{e:+.3%}"  # noqa: E731
                    print(f"  converge {lines[j][0]:.5f}: {shown(e_c)} at "
                          f"{coarse}, {shown(e_f)} at {cells} (first-order "
                          f"limit {shown(limit)})")
                for problem in problems:
                    print(f"  FAIL {problem}")
                failures += [f"{label}: {problem}" for problem in problems]

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
