#!/usr/bin/env python3
"""Benchmark V2: a CW point source in open vacuum, amplitude against 1/r.

A z-directed point current at the centre of an open (CPML) box radiates as a
short dipole. Broadside to it -- along +x, at the source's own z level -- the
field is pure Ez and its amplitude falls as

    |E| ~ (1/r) * sqrt(1 - 1/(kr)^2 + 1/(kr)^4)

so beyond a wavelength it is 1/r to within 1.3%, and from the spec's 2
wavelengths (the default --r-min) to within 0.32% (the near-field factor is
printed in the table). The gate is the spec's: at r >= 2 wavelengths, r * |Ez(r)|
stays within 5% of its value at the nearest probe. The reference is that
nearest probe rather than a median, so a steady drift across the range is read
in full instead of being split either side of a middle value.

THE AMPLITUDE IS A LEAST-SQUARES FIT, not a peak. Over the recorded window each
probe trace is fitted with a cos wt + b sin wt + c and the amplitude is
hypot(a, b). The recording starts only after the turn-on ramp, the travel time
to the farthest probe and a settling allowance have passed, and the run checks
that it did: the fit over the first and second halves of the window must agree
within 1%, or the run fails as NOT STEADY rather than reporting a transient as
a falloff.

A SILENT PROBE IS A FAILURE, not a pass: a NaN or zero amplitude fails the run,
because a flat r * |E| of zero would otherwise be perfectly 1/r.

DEFAULTS (20 cells per wavelength, probes from 2 to 3 wavelengths): a 149^3
grid, ~3.3e6 cells and ~600 steps, so about 2e9 cell-steps. At the Orin's
measured 1.2e8 cell-steps/s that is under a minute. The cost line prints the
real figure. NOT for the VM: a tiny smoke (--cells-per-wavelength 6 --r-max 1.5
--pml-cells 4 --margin 1, 29^3 cells) only shows the script runs.

RUN IT ON THE JETSON (tier 2, one at a time):

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v2'
    tar -C <repo> -cf - scripts/fdtd_bench_v2.py dockerfiles/maxwell_solver/fdtd.py \\
        | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v2'
    ssh jetson 'cd ~/mse-bench/fdtd-v2 && \\
      (ulimit -v 5000000; nice -n 10 timeout 30m python3 scripts/fdtd_bench_v2.py)'

Add ``--xp cupy`` for the GPU path; the same update code runs.
"""

from __future__ import annotations

import argparse
import math
import os
import resource
import sys
import time
from pathlib import Path

# Cap threads before numpy loads, as the V1 bench does.
_THREADS = os.environ.get("MSE_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import fdtd  # noqa: E402

#: The fits over the two halves of the window must agree this closely, or the
#: field at that probe is still changing and the window is not steady state.
STEADY = 0.01


def amplitude(trace, t, omega):
    """hypot(a, b) of the least-squares fit a cos wt + b sin wt + c."""
    basis = np.stack([np.cos(omega * t), np.sin(omega * t), np.ones_like(t)], 1)
    coeffs, *_ = np.linalg.lstsq(basis, trace, rcond=None)
    return float(math.hypot(coeffs[0], coeffs[1]))


def near_field_factor(kr):
    """|short-dipole broadside field| x r, relative to its far-field value."""
    return math.sqrt(1.0 - 1.0 / kr ** 2 + 1.0 / kr ** 4)


def falloff(radii, amplitudes):
    """``r * A(r)`` relative to the first (nearest) probe, one per probe."""
    reference = radii[0] * amplitudes[0]
    return [r * a / reference for r, a in zip(radii, amplitudes)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frequency", type=float, default=1e9,
                        help="Hz; the result is scale-free, this only sets units")
    parser.add_argument("--cells-per-wavelength", type=float, default=20.0)
    parser.add_argument("--r-min", type=float, default=2.0,
                        help="nearest gated probe, in wavelengths; the spec's "
                             "V2 is r >= 2")
    parser.add_argument("--r-max", type=float, default=3.0,
                        help="farthest probe, in wavelengths")
    parser.add_argument("--margin", type=int, default=4,
                        help="cells between the farthest probe and the CPML")
    parser.add_argument("--pml-cells", type=int, default=fdtd.DEFAULT_PML_CELLS)
    parser.add_argument("--settle-periods", type=float, default=3.0,
                        help="extra periods after the ramp and the travel time")
    parser.add_argument("--record-periods", type=float, default=6.0)
    parser.add_argument("--tolerance", type=float, default=5.0,
                        help="percent; the proposal's V2 gate")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    xp = np
    if args.xp == "cupy":
        import cupy as xp  # noqa: F811 -- only here, never at module level

    wavelength = fdtd.C0 / args.frequency
    dx = wavelength / args.cells_per_wavelength
    near = int(math.ceil(args.r_min * args.cells_per_wavelength))
    far = int(math.floor(args.r_max * args.cells_per_wavelength))
    if far <= near:
        print(f"no probes: r-min {args.r_min} and r-max {args.r_max} wavelengths "
              f"give cells {near}..{far}")
        return 2
    half = far + args.margin + args.pml_cells
    n = 2 * half + 1
    centre = (half, half, half)
    # Ez sits at (i, j, k + 1/2) for the source and every probe alike, so a
    # probe d cells along +x is exactly d * dx from the source, broadside.
    offsets = list(range(near, far + 1))
    probes = [(half + d, half, half, "z") for d in offsets]
    radii = [d * dx for d in offsets]

    dt = fdtd.courant_dt(dx)
    period = 1.0 / args.frequency
    ramp = 5.0                                   # cw_profile's default turn-on
    start_periods = ramp + args.r_max + args.settle_periods
    record_from = int(math.ceil(start_periods * period / dt))
    window = int(math.ceil(args.record_periods * period / dt))
    window -= window % 2                         # two equal halves
    n_steps = record_from + window
    cells = n ** 3

    print(f"grid       {n}^3 = {cells} cells, dx = {dx*1e3:.4g} mm "
          f"({args.cells_per_wavelength:g} per wavelength), CPML {args.pml_cells}")
    print(f"source     CW Ez point current at {centre}, {args.frequency/1e9:g} GHz")
    print(f"probes     {len(probes)} along +x, broadside, {args.r_min:g} to "
          f"{offsets[-1] / args.cells_per_wavelength:.3g} wavelengths")
    print(f"steps      {n_steps} ({record_from} to steady state + {window} recorded, "
          f"{args.record_periods:g} periods), dt = {dt*1e12:.4g} ps")

    source = fdtd.Source(centre, frequency=args.frequency, profile="cw",
                         component="z")
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start = time.perf_counter()
    run = fdtd.simulate(np.ones((n, n, n), np.float32), source, n_steps, probes,
                        dx=dx, boundary="open", pml_cells=args.pml_cells, xp=xp)
    elapsed = time.perf_counter() - start
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"run        {elapsed:.2f} s, peak RSS {peak/1024:.1f} MB "
          f"(+{(peak-before)/1024:.3g} MB)")
    print(f"cost       {cells} cells x {n_steps} steps = {cells*n_steps:.4g} "
          f"cell-steps in {elapsed:.2f} s -> {cells*n_steps/elapsed:.4g} cell-steps/s")

    traces = np.asarray(run["traces"], dtype=np.float64)[:, record_from:]
    t = (record_from + np.arange(window)) * dt
    omega = 2.0 * math.pi * args.frequency
    k = 2.0 * math.pi / wavelength
    split = window // 2

    failures = []
    amplitudes = []
    for index, trace in enumerate(traces):
        whole = amplitude(trace, t, omega)
        first = amplitude(trace[:split], t[:split], omega)
        second = amplitude(trace[split:], t[split:], omega)
        amplitudes.append(whole)
        if not math.isfinite(whole) or whole <= 0.0:
            failures.append(f"probe at {radii[index]/wavelength:.3g} wavelengths "
                            f"saw nothing (amplitude {whole})")
        elif abs(first - second) > STEADY * whole:
            failures.append(f"NOT STEADY at {radii[index]/wavelength:.3g} "
                            f"wavelengths: the two half-window fits differ by "
                            f"{abs(first-second)/whole:.2%}")

    if failures:
        print("\nV2 FAIL")
        for failure in failures:
            print(f"  {failure}")
        return 1

    ratios = falloff(radii, amplitudes)
    reference_factor = near_field_factor(k * radii[0])
    print(f"\n{'r (lambda)':>10} {'r*|Ez| / ref':>13} {'off 1/r':>9} "
          f"{'dipole':>8} {'off dipole':>11}")
    worst = 0.0
    for radius, ratio in zip(radii, ratios):
        dipole = near_field_factor(k * radius) / reference_factor
        worst = max(worst, abs(ratio - 1.0))
        print(f"{radius/wavelength:10.3f} {ratio:13.5f} {100*(ratio-1):+8.2f}% "
              f"{dipole:8.5f} {100*(ratio/dipole-1):+10.2f}%")

    verdict = "PASS" if 100.0 * worst <= args.tolerance else "FAIL"
    print(f"\nV2 {verdict}: r*|Ez| stays within {100*worst:.2f}% of its value at "
          f"{radii[0]/wavelength:.3g} wavelengths over {len(radii)} probes "
          f"(tolerance {args.tolerance:g}%; the short dipole's own near-field "
          f"term accounts for up to "
          f"{100*abs(near_field_factor(k*radii[-1])/reference_factor-1):.2f}%)")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
