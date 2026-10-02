#!/usr/bin/env python3
"""Benchmark V4: a pulse reflects from a gold slab at normal incidence.

The spec's gate: the reflectance |scattered/incident|^2 at a scattered-field
probe is within 0.02 (absolute) of the analytic Fresnel reflectance of the
SAME Lorentz-Drude model, over the band where the incident spectrum is at
least 1e-2 of its peak. The slab is at least 10 skin depths thick with at
least 4 cells per skin depth, across that whole band.

SETUP. The pulse travels along +z through an ``open`` axis; x and y are
periodic, so the slab is infinite sideways, as the Fresnel result assumes (a
finite plate inside an open TF/SF box diffracts round its edges and misses by
10-20%). The TF/SF box spans x and y and holds the slab 3 cells from its z
faces. Two runs, both with the metal term so they share one dt (the metal can
shrink dt below the Courant step):

- REFERENCE: no metal anywhere (the term's region is empty). E_x inside the
  box, just past the low face, is the incident pulse alone.
- SLAB: E_x just before the low face, in the scattered-field region, is the
  reflected pulse alone.

Both traces are recorded until they have died away (the tail gate) and are
transformed WITHOUT a window: the two pulses sit at different times in the
record, so a window would weight them differently and bias the ratio. In a
lossless vacuum grid at normal incidence the incident amplitude is the same at
both probe planes, so the ratio of the two spectra is the reflection.

The analytic reflectance is the exact single-slab (Airy) result for the
model's eps at the slab's nominal thickness; at 10 skin depths the back face
changes it by less than 1e-8.

The grid is uniform: at normal incidence the 1-D incident line and the 3-D
grid then share one dispersion relation, so the reference is exact.

DEFAULTS (gold, 1.5 eV centre, bandwidth 0.3 f, band about 0.54-2.46 eV): about
6 nm cells, 4 x 4 transverse, ~110 cells along z with a 10-cell CPML, so
~2e3 cells and ~1e4 steps a run: ~2e7 cell-steps each. It is still a solver
run: on the Jetson. The record (source plus 40 transits) sets the bin width:
52 bins of 37 meV across the band at CPML 10, 60 of 32 meV at CPML 20, over
the same band; the PASS needs the judged bins to span it end to end.

RUN IT ON THE JETSON (tier 2):

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v4'
    git -C <repo> archive <sha> scripts/fdtd_bench_v4.py \\
        dockerfiles/maxwell_solver/fdtd.py | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v4'
    ssh jetson 'cd ~/mse-bench/fdtd-v4 && \\
      (ulimit -v 5000000; nice -n 10 timeout 30m python3 scripts/fdtd_bench_v4.py)'

Exit 0 PASS, 1 FAIL, 2 the setup is not the spec's (a refused metal band,
fewer than 4 cells per skin depth, a slab under 10 skin depths, bad arguments).
"""

from __future__ import annotations

import argparse
import math
import os
import resource
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

import fdtd  # noqa: E402

#: The spec's numbers.
REFLECTANCE_TOLERANCE = 0.02         # absolute
BAND_FLOOR = 1e-2                    # of the incident spectrum's peak
MIN_CELLS_PER_SKIN_DEPTH = 4.0
MIN_SKIN_DEPTHS = 10.0
#: The bench's own: a trace whose last TAIL_FRACTION still holds more than
#: TAIL_TOLERANCE of its peak was cut off, and its spectrum has a truncation
#: skirt the ratio would read as reflectance.
TAIL_FRACTION = 0.05
TAIL_TOLERANCE = 1e-3


def refractive_index(eps):
    """sqrt(eps) on the branch with Im n >= 0 (exp(-i w t): a decaying wave)."""
    n = np.sqrt(np.asarray(eps, dtype=complex))
    return np.where(n.imag < 0, -n, n)


def slab_reflectance(eps, thickness: float, frequency):
    """|r|^2 of vacuum | slab (eps, thickness) | vacuum at normal incidence."""
    n = refractive_index(eps)
    k0 = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64) / fdtd.C0
    r12 = (1.0 - n) / (1.0 + n)
    phase = np.exp(2j * k0 * n * thickness)
    r = (r12 - r12 * phase) / (1.0 - r12 * r12 * phase)
    return np.abs(r) ** 2


def skin_depth(eps, frequency):
    """1/e depth of the FIELD amplitude: c / (w Im n)."""
    n = refractive_index(eps)
    w = 2.0 * np.pi * np.asarray(frequency, dtype=np.float64)
    return fdtd.C0 / (w * n.imag)


def band(amplitude, floor: float = BAND_FLOOR) -> np.ndarray:
    """Bins whose incident amplitude is at least ``floor`` of its peak."""
    amplitude = np.asarray(amplitude, dtype=np.float64)
    top = float(amplitude.max()) if amplitude.size else 0.0
    if not (math.isfinite(top) and top > 0):
        return np.zeros(amplitude.shape, dtype=bool)
    return amplitude >= floor * top


def tail(trace, fraction: float = TAIL_FRACTION) -> float:
    """Largest |value| in the last ``fraction`` of a trace, over its peak."""
    trace = np.abs(np.asarray(trace, dtype=np.float64))
    top = float(trace.max()) if trace.size else 0.0
    if not (math.isfinite(top) and top > 0):
        return math.inf
    count = max(1, int(math.ceil(fraction * trace.size)))
    return float(trace[-count:].max()) / top


def reflectance(incident, reflected, dt: float):
    """(frequencies, |R_hat/I_hat|^2, |I_hat|) of two traces, unwindowed."""
    frequencies, inc = fdtd.spectrum(incident, dt, window=None)
    _, ref = fdtd.spectrum(reflected, dt, window=None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (np.asarray(ref) / np.asarray(inc)) ** 2
    return frequencies, ratio, np.asarray(inc)


def band_coverage(frequencies, in_band, f_low: float, f_high: float):
    """Failures if the judged bins do not span [f_low, f_high]: each end must
    sit within one bin of the band the metal and skin depths were checked
    over, on neither side, with no hole between. A shorter record has fewer,
    wider bins over the same band; this stops a narrower sweep passing."""
    frequencies = np.asarray(frequencies, dtype=np.float64)
    picks = np.flatnonzero(np.asarray(in_band, dtype=bool))
    if picks.size == 0:
        return []                            # v4_verdict names the empty band
    step = float(frequencies[1] - frequencies[0])
    lo, hi = float(frequencies[picks[0]]), float(frequencies[picks[-1]])
    ev = 2 * math.pi / fdtd.EV
    span = (f"the measured band {lo*ev:.4g}-{hi*ev:.4g} eV "
            f"({picks.size} bins)")
    nominal = f"the band {f_low*ev:.4g}-{f_high*ev:.4g} eV"
    failures = []
    if lo < f_low - step or hi > f_high + step:
        failures.append(f"{span} leaves {nominal} the metal and skin depths "
                        f"were checked over")
    if lo > f_low + step or hi < f_high - step:
        failures.append(f"{span} does not reach both ends of {nominal}")
    if picks[-1] - picks[0] + 1 != picks.size:
        failures.append(f"{span} has a hole")
    return failures


def v4_verdict(frequencies, measured, analytic, in_band, tails,
               tolerance: float = REFLECTANCE_TOLERANCE):
    """(passed, failures, worst |error|, frequency of the worst)."""
    failures = []
    frequencies = np.asarray(frequencies, dtype=np.float64)
    measured = np.asarray(measured, dtype=np.float64)
    analytic = np.asarray(analytic, dtype=np.float64)
    in_band = np.asarray(in_band, dtype=bool)
    worst, where = math.nan, math.nan
    for name, value in tails.items():
        if not (math.isfinite(value) and value <= TAIL_TOLERANCE):
            failures.append(f"the {name} trace still holds {value:.3g} of its "
                            f"peak in its last {TAIL_FRACTION:g} (limit "
                            f"{TAIL_TOLERANCE:g}); record longer")
    if not in_band.any():
        failures.append("the incident spectrum is empty: no band to judge")
        return False, failures, worst, where
    error = np.abs(measured[in_band] - analytic[in_band])
    if not np.all(np.isfinite(error)):
        failures.append("the reflectance is not finite inside the band")
        return False, failures, worst, where
    index = int(np.argmax(error))
    worst, where = float(error[index]), float(frequencies[in_band][index])
    if worst > tolerance:
        failures.append(f"the reflectance misses Fresnel by {worst:.4f} at "
                        f"{where * 2 * math.pi / fdtd.EV:.4g} eV (limit "
                        f"{tolerance:g} absolute)")
    return not failures, failures, worst, where


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--metal", default="Au", choices=fdtd.METALS)
    parser.add_argument("--centre-ev", type=float, default=1.5,
                        help="pulse centre photon energy, eV")
    parser.add_argument("--bandwidth-ratio", type=float, default=0.3,
                        help="pulse bandwidth as a fraction of the frequency")
    parser.add_argument("--cells-per-skin-depth", type=float, default=4.0,
                        help="at the thinnest skin depth in the band (spec >= 4)")
    parser.add_argument("--skin-depths", type=float, default=10.0,
                        help="slab thickness, in the thickest skin depth in "
                             "the band (spec >= 10)")
    parser.add_argument("--vacuum-cells", type=int, default=fdtd.BOX_CLEARANCE,
                        help="cells between the box faces and the slab")
    parser.add_argument("--gap", type=int, default=fdtd.BOX_CLEARANCE,
                        help="cells between the box and the CPML (spec >= 3)")
    parser.add_argument("--pml-cells", type=int, default=fdtd.DEFAULT_PML_CELLS)
    parser.add_argument("--transverse", type=int, default=4,
                        help="periodic nodes along x and along y")
    parser.add_argument("--record-transits", type=float, default=40.0,
                        help="record this many z transits after the source "
                             "ends (the metal's slow Drude tail needs them)")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    xp = np
    if args.xp == "cupy":
        import cupy as xp  # noqa: F811 -- only here, never at module level
    if args.transverse < 3:
        print("--transverse needs at least 3 nodes (a probe off the ghost slot)")
        return 2

    frequency = args.centre_ev * fdtd.EV / (2.0 * math.pi)
    bandwidth = args.bandwidth_ratio * frequency
    # The band the gate will judge, from the profile's own spectrum: the
    # envelope exp(-(t/sigma)^2), sigma = 1/(pi bw), has amplitude spectrum
    # exp(-((f - f0)/bw)^2), above BAND_FLOOR within this half-width. After the
    # runs the measured band is checked against it.
    half = bandwidth * math.sqrt(math.log(1.0 / BAND_FLOOR))
    f_low, f_high = frequency - half, frequency + half
    try:
        if f_low <= 0:
            raise fdtd.FDTDError(f"the band reaches {f_low:.3g} Hz; narrow it")
        fdtd.check_metal_band(args.metal, f_low, f_high)
    except fdtd.FDTDError as error:
        print(f"refused: {error}")
        return 2

    probe_band = np.linspace(f_low, f_high, 400)
    eps_band = fdtd.permittivity(args.metal, probe_band)
    depths = skin_depth(eps_band, probe_band)
    dz = float(depths.min()) / args.cells_per_skin_depth
    slab_cells = int(math.ceil(args.skin_depths * float(depths.max()) / dz))
    cells_per_depth = float(depths.min()) / dz
    depths_thick = slab_cells * dz / float(depths.max())
    spec_setup = (cells_per_depth >= MIN_CELLS_PER_SKIN_DEPTH - 1e-9
                  and depths_thick >= MIN_SKIN_DEPTHS - 1e-9)

    t = args.transverse
    low = args.pml_cells + args.gap
    k0 = low + args.vacuum_cells             # first metal node
    # Ex samples sit on the z nodes, and each stands for one cell of
    # thickness: slab_cells metal nodes are a slab_cells*dz slab (a dielectric
    # slab measured this way matches its Airy reflectance to 0.002 only with
    # that thickness; counting one node more or less misses by ~0.03).
    k1 = k0 + slab_cells - 1                 # last metal node
    high = k1 + 1 + args.vacuum_cells
    n = high + args.gap + args.pml_cells + 1
    shape = (t, t, n)
    spacings = (np.full(t - 1, dz), np.full(t - 1, dz), np.full(n - 1, dz))
    boundary = ("periodic", "periodic", "open")
    box = ((0, 0, low), (t - 1, t - 1, high))
    region = np.zeros(shape, dtype=bool)
    region[:, :, k0:k1 + 1] = True

    source = fdtd.Source(kind="plane", direction="+z", component="x", box=box,
                         frequency=frequency, bandwidth=bandwidth)
    dt_courant = fdtd.courant_dt(dz, dz, dz)
    dt_guess = fdtd.LorentzDrude(args.metal, region).stable_dt(
        dt_courant, (dz, dz, dz))
    transit = (n - 1) * dz / fdtd.C0
    n_steps = int(math.ceil((source.end_time() + args.record_transits * transit)
                            / dt_guess))
    inc_probe = (1, 1, low + 1, "x")         # inside the box: incident only
    ref_probe = (1, 1, low - 2, "x")         # scattered field: reflected only

    print(f"metal      {args.metal} (Rakić 1998 Lorentz-Drude), slab nodes "
          f"{k0}..{k1} = {slab_cells} cells = {slab_cells*dz*1e9:.4g} nm")
    print(f"band       {f_low*2*math.pi/fdtd.EV:.4g}-"
          f"{f_high*2*math.pi/fdtd.EV:.4g} eV (incident >= {BAND_FLOOR:g} of "
          f"peak); skin depth {depths.min()*1e9:.4g}-{depths.max()*1e9:.4g} nm")
    print(f"grid       {t} x {t} periodic x {n} open = {t*t*n} cells, dz = "
          f"{dz*1e9:.4g} nm: {cells_per_depth:.3g} cells per skin depth, slab "
          f"{depths_thick:.3g} skin depths; CPML {args.pml_cells}")
    print(f"steps      {n_steps} a run, dt ~ {dt_guess*1e18:.4g} as "
          f"({dt_guess/dt_courant:.3g} of Courant)")

    runs = {}
    total = 0.0
    for name, mask, probe in (("reference", np.zeros(shape, dtype=bool),
                               inc_probe),
                              ("slab", region, ref_probe)):
        start = time.perf_counter()
        runs[name] = fdtd.simulate(
            np.ones(shape, np.float32), source, n_steps, [probe],
            spacings=spacings, boundary=boundary, pml_cells=args.pml_cells,
            xp=xp, ade_terms=[fdtd.LorentzDrude(args.metal, mask)])
        total += time.perf_counter() - start
        print(f"run        {name}: {time.perf_counter() - start:.2f} s",
              flush=True)

    dt = runs["slab"]["dt"]
    if runs["reference"]["dt"] != dt:
        print(f"V4 FAIL: the two runs used different dt "
              f"({runs['reference']['dt']} vs {dt}); the spectra do not align")
        return 1
    incident = np.asarray(runs["reference"]["traces"][0], dtype=np.float64)
    reflected = np.asarray(runs["slab"]["traces"][0], dtype=np.float64)
    frequencies, measured, amplitude = reflectance(incident, reflected, dt)
    in_band = band(amplitude)
    coverage = band_coverage(frequencies, in_band, f_low, f_high)
    if coverage:
        print("V4 FAIL")
        for failure in coverage:
            print(f"  {failure}")
        return 1
    with np.errstate(divide="ignore", invalid="ignore"):
        eps_all = fdtd.permittivity(args.metal, frequencies)
    with np.errstate(divide="ignore", invalid="ignore"):
        analytic = slab_reflectance(eps_all, slab_cells * dz, frequencies)
    tails = {"incident": tail(incident), "reflected": tail(reflected)}
    passed, failures, worst, where = v4_verdict(frequencies, measured, analytic,
                                                in_band, tails)

    print(f"\n{'eV':>7} {'FDTD R':>9} {'Fresnel R':>10} {'error':>9}")
    picks = np.flatnonzero(in_band)
    for index in picks[np.linspace(0, picks.size - 1,
                                   min(12, picks.size)).astype(int)]:
        print(f"{frequencies[index]*2*math.pi/fdtd.EV:7.4g} "
              f"{measured[index]:9.5f} {analytic[index]:10.5f} "
              f"{measured[index] - analytic[index]:+9.5f}")
    work = t * t * n * n_steps * 2
    print(f"\ntails      incident {tails['incident']:.3g}, reflected "
          f"{tails['reflected']:.3g} of peak")
    print(f"cost       {work:.4g} cell-steps in {total:.2f} s, peak RSS "
          f"{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024:.1f} MB")

    if not passed:
        print("\nV4 FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1
    if not spec_setup:
        print(f"\nNOT A SPEC PASS: within {worst:.4f} of Fresnel, but the "
              f"setup has {cells_per_depth:.3g} cells per skin depth (spec "
              f">= {MIN_CELLS_PER_SKIN_DEPTH:g}) and a slab of "
              f"{depths_thick:.3g} skin depths (spec >= {MIN_SKIN_DEPTHS:g})")
        return 2
    ev = 2 * math.pi / fdtd.EV
    print(f"\nV4 PASS: over {int(in_band.sum())} bins spanning "
          f"{frequencies[in_band].min()*ev:.4g}-{frequencies[in_band].max()*ev:.4g}"
          f" eV the reflectance is within "
          f"{worst:.4f} of Fresnel for the same model (limit "
          f"{REFLECTANCE_TOLERANCE:g}), worst at "
          f"{where*2*math.pi/fdtd.EV:.4g} eV")
    return 0


if __name__ == "__main__":
    sys.exit(main())
