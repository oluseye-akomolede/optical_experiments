#!/usr/bin/env python3
"""Benchmark V3: a plane-wave pulse crosses empty open space and leaves.

The spec's two gates, for every direction and transverse polarisation (12
runs by default, ``--pairs`` picks fewer):

1. ENERGY. Three grid transit times after the source ends, the field energy
   over the physical region (CPML excluded) is below 1e-4 of its peak. The
   energy is the solver's own conserved quadratic, restricted to that region.
2. LEAKAGE. Throughout the run, every sample of the scattered-field region
   (outside the TF/SF box, CPML excluded) stays below 1e-3 of the incident
   peak. That region is read cell by cell from the solver's running maximum
   of |E_c|, not from a few probes, and each component's own Yee position
   decides whether a sample is inside the box: E_c sits at half indices along
   c, so the sample at node index ``high`` along c is at ``high + 1/2``,
   outside.

The incident peak is the largest |E_p| inside the box. A run in which it is
zero, or any number is NaN, FAILS: an absent wave leaks nothing and leaves no
energy, and would otherwise pass both gates.

The grid is graded by default (+/- ``--grade`` along x, uniform along y, the
reverse grading along z), because the spec has the 1-D incident line use the
3-D axis's own graded spacings and a uniform grid would not exercise that.

DEFAULTS (20 cells per wavelength, a 1.5-wavelength box, 3 cells to a 10-cell
CPML): a 57^3 grid, ~1.9e5 cells, ~600 steps a run, ~1.1e8 cell-steps a run,
~1.3e9 for all twelve. The energy diagnostic reduces six arrays a step, so this
is slower per cell-step than V2; the cost line prints the real figure. NOT for
the VM: a tiny smoke (--cells-per-wavelength 6 --box-wavelengths 1
--pml-cells 4 --pairs +x:z, 21^3 cells) only shows the script runs.

RUN IT ON THE JETSON (tier 2, one at a time):

    ssh jetson 'mkdir -p ~/mse-bench/fdtd-v3'
    git -C <repo> archive <sha> scripts/fdtd_bench_v3.py \\
        dockerfiles/maxwell_solver/fdtd.py | ssh jetson 'tar -xf - -C ~/mse-bench/fdtd-v3'
    ssh jetson 'cd ~/mse-bench/fdtd-v3 && \\
      (ulimit -v 5000000; nice -n 10 timeout 30m python3 scripts/fdtd_bench_v3.py)'

Add ``--xp cupy`` (with a python that has cupy) for the GPU path.
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

#: The spec's gates.
ENERGY_TOLERANCE = 1e-4
LEAK_TOLERANCE = 1e-3

ALL_PAIRS = tuple(f"{d}:{c}" for d in fdtd.DIRECTIONS for c in fdtd.COMPONENTS
                  if c != d[1])


NOT_PERIODIC = (False, False, False)


def _inside(box, component: str, periodic) -> tuple:
    c = fdtd.COMPONENTS.index(component)
    low, high = box
    return tuple(slice(None) if periodic[a] else
                 slice(low[a], high[a] if a == c else high[a] + 1)
                 for a in range(3))


def scattered_mask(shape, box, pml_cells: int, component: str,
                   periodic=NOT_PERIODIC) -> np.ndarray:
    """The samples of E_component in the scattered-field region.

    Physical region: node indices ``pml_cells .. n-1-pml_cells`` on every
    axis with a CPML, all of a periodic axis. Inside the box: along the
    component's own axis E sits at half indices, inside for ``low .. high-1``;
    along the other two at nodes, inside for ``low .. high``. A periodic axis
    has no box faces, so all of it is inside.
    """
    mask = np.zeros(shape, dtype=bool)
    mask[tuple(slice(None) if periodic[a] else
               slice(pml_cells, n - pml_cells)
               for a, n in enumerate(shape))] = True
    mask[_inside(box, component, periodic)] = False
    return mask


def incident_mask(shape, box, component: str,
                  periodic=NOT_PERIODIC) -> np.ndarray:
    """The samples of E_component inside the box (the complement's rule)."""
    mask = np.zeros(shape, dtype=bool)
    mask[_inside(box, component, periodic)] = True
    return mask


def measure(peak: dict, energy, box, pml_cells: int, polarisation: str,
            periodic=NOT_PERIODIC):
    """(incident peak, worst leak / incident, final energy / peak energy)."""
    shape = peak["x"].shape
    incident = float(peak[polarisation][incident_mask(
        shape, box, polarisation, periodic)].max())
    scattered = max(float(peak[c][scattered_mask(
        shape, box, pml_cells, c, periodic)].max())
                    for c in fdtd.COMPONENTS)
    energy = np.asarray(energy, dtype=np.float64)
    top = float(energy.max()) if energy.size else 0.0
    leak = scattered / incident if incident > 0 else math.inf
    residual = float(energy[-1]) / top if top > 0 else math.inf
    return incident, leak, residual


def v3_verdict(rows, energy_tolerance=ENERGY_TOLERANCE,
               leak_tolerance=LEAK_TOLERANCE):
    """``rows`` of (pair, incident, leak, residual) -> (passed, failures)."""
    failures = []
    for pair, incident, leak, residual in rows:
        if not (math.isfinite(incident) and incident > 0):
            failures.append(f"{pair}: no incident wave in the box "
                            f"(peak {incident})")
            continue
        if not (math.isfinite(leak) and leak < leak_tolerance):
            failures.append(f"{pair}: the scattered-field region reached "
                            f"{leak:.3g} of the incident peak "
                            f"(limit {leak_tolerance:g})")
        if not (math.isfinite(residual) and residual < energy_tolerance):
            failures.append(f"{pair}: {residual:.3g} of the peak energy is "
                            f"still in the grid (limit {energy_tolerance:g})")
    if not rows:
        failures.append("no runs")
    return not failures, failures


def graded(n: int, dx: float, grade: float, axis: int) -> np.ndarray:
    if grade == 0.0 or axis == 1:
        return np.full(n - 1, dx)
    ramp = np.linspace(1.0 - grade, 1.0 + grade, n - 1)
    return dx * (ramp if axis == 0 else ramp[::-1])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--frequency", type=float, default=1e9,
                        help="pulse centre frequency, Hz")
    parser.add_argument("--bandwidth-ratio", type=float, default=0.5,
                        help="pulse bandwidth as a fraction of the frequency")
    parser.add_argument("--cells-per-wavelength", type=float, default=20.0)
    parser.add_argument("--box-wavelengths", type=float, default=1.5,
                        help="TF/SF box edge")
    parser.add_argument("--gap", type=int, default=fdtd.BOX_CLEARANCE,
                        help="cells between the box and the CPML (spec >= 3)")
    parser.add_argument("--pml-cells", type=int, default=fdtd.DEFAULT_PML_CELLS)
    parser.add_argument("--grade", type=float, default=0.1,
                        help="+/- spacing grading along x and z; 0 is uniform")
    parser.add_argument("--transits", type=float, default=3.0,
                        help="grid transit times after the source ends")
    parser.add_argument("--pairs", default=",".join(ALL_PAIRS),
                        help="direction:component list, e.g. +x:z,-y:x")
    parser.add_argument("--periodic-transverse", type=int, default=0,
                        metavar="N",
                        help="make the two axes across each run's direction "
                             "periodic, N nodes each, with the box spanning "
                             "them (the V4 setup); 0 keeps the open cube")
    parser.add_argument("--xp", default="numpy", choices=("numpy", "cupy"))
    return parser


def layout(pair: str, n: int, low_index: int, edge: int, transverse: int):
    """(shape, box, boundary, periodic) for one run."""
    axis = fdtd.COMPONENTS.index(pair.split(":")[0][1])
    if not transverse:
        return ((n,) * 3, ((low_index,) * 3, (low_index + edge,) * 3), "open",
                NOT_PERIODIC)
    periodic = tuple(a != axis for a in range(3))
    shape = tuple(transverse if periodic[a] else n for a in range(3))
    low = tuple(0 if periodic[a] else low_index for a in range(3))
    high = tuple(transverse - 1 if periodic[a] else low_index + edge
                 for a in range(3))
    boundary = tuple("periodic" if periodic[a] else "open" for a in range(3))
    return shape, (low, high), boundary, periodic


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    pairs = [p.strip() for p in args.pairs.split(",") if p.strip()]
    for pair in pairs:
        if pair not in ALL_PAIRS:
            print(f"unknown pair {pair!r}; choose from {', '.join(ALL_PAIRS)}")
            return 2

    xp = np
    if args.xp == "cupy":
        import cupy as xp  # noqa: F811 -- only here, never at module level
    if args.periodic_transverse and args.periodic_transverse < 2:
        print("--periodic-transverse needs at least 2 nodes per axis")
        return 2

    wavelength = fdtd.C0 / args.frequency
    dx = wavelength / args.cells_per_wavelength
    edge = int(math.ceil(args.box_wavelengths * args.cells_per_wavelength))
    low_index = args.pml_cells + args.gap
    n = 2 * low_index + edge + 1
    box = ((low_index,) * 3, (low_index + edge,) * 3)
    spacings = tuple(graded(n, dx, args.grade, a) for a in range(3))
    dt = fdtd.courant_dt(*(s.min() for s in spacings))
    transit = max(float(s.sum()) for s in spacings) / fdtd.C0
    probe_source = fdtd.Source(kind="plane", direction="+x", component="z",
                               box=box, frequency=args.frequency,
                               bandwidth=args.bandwidth_ratio * args.frequency)
    n_steps = int(math.ceil((probe_source.end_time() + args.transits * transit)
                            / dt))
    transverse = args.periodic_transverse
    cells = n * transverse ** 2 if transverse else n ** 3

    if transverse:
        print(f"grid       {n} along travel x {transverse}^2 periodic across "
              f"= {cells} cells, dx = {dx*1e3:.4g} mm "
              f"({args.cells_per_wavelength:g} per wavelength, graded "
              f"+/-{args.grade:g} on x and z), CPML {args.pml_cells}")
        print(f"box        nodes {box[0][0]}..{box[1][0]} along travel "
              f"({edge} cells), {args.gap} cells to the CPML; spans the "
              f"periodic axes")
    else:
        print(f"grid       {n}^3 = {cells} cells, dx = {dx*1e3:.4g} mm "
              f"({args.cells_per_wavelength:g} per wavelength, graded "
              f"+/-{args.grade:g} on x and z), CPML {args.pml_cells}")
        print(f"box        nodes {box[0][0]}..{box[1][0]} on every axis "
              f"({edge} cells), {args.gap} cells to the CPML")
    print(f"source     plane-wave pulse, {args.frequency/1e9:g} GHz, bandwidth "
          f"{args.bandwidth_ratio:g} f, ends at {probe_source.end_time()*1e9:.4g} ns")
    print(f"steps      {n_steps} per run (source + {args.transits:g} transits of "
          f"{transit*1e9:.4g} ns), dt = {dt*1e12:.4g} ps, {len(pairs)} runs")

    rows = []
    total_elapsed = 0.0
    print(f"\n{'pair':>6} {'incident':>10} {'leak':>10} {'residual':>10} "
          f"{'run s':>7}")
    for pair in pairs:
        direction, component = pair.split(":")
        shape, run_box, boundary, periodic = layout(pair, n, low_index, edge,
                                                    transverse)
        run_spacings = tuple(graded(shape[a], dx, args.grade, a)
                             for a in range(3))
        physical = tuple(slice(None) if periodic[a] else
                         slice(args.pml_cells, shape[a] - args.pml_cells)
                         for a in range(3))
        source = fdtd.Source(kind="plane", direction=direction,
                             component=component, box=run_box,
                             frequency=args.frequency,
                             bandwidth=args.bandwidth_ratio * args.frequency)
        start = time.perf_counter()
        run = fdtd.simulate(np.ones(shape, np.float32), source, n_steps,
                            spacings=run_spacings, boundary=boundary,
                            pml_cells=args.pml_cells, xp=xp,
                            record_energy=True, energy_region=physical,
                            record_peak=True)
        elapsed = time.perf_counter() - start
        total_elapsed += elapsed
        incident, leak, residual = measure(run["peak"], run["energy"], run_box,
                                           args.pml_cells, component, periodic)
        rows.append((pair, incident, leak, residual))
        print(f"{pair:>6} {incident:10.4g} {leak:10.3e} {residual:10.3e} "
              f"{elapsed:7.2f}", flush=True)

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    work = cells * n_steps * len(pairs)
    print(f"\ncost       {work:.4g} cell-steps in {total_elapsed:.2f} s -> "
          f"{work/total_elapsed:.4g} cell-steps/s (energy diagnostic on), "
          f"peak RSS {peak_rss/1024:.1f} MB")

    passed, failures = v3_verdict(rows)
    worst_leak = max((r[2] for r in rows), default=math.nan)
    worst_residual = max((r[3] for r in rows), default=math.nan)
    if passed:
        print(f"\nV3 PASS: over {len(rows)} runs the scattered-field region "
              f"peaked at {worst_leak:.3g} of incident (limit "
              f"{LEAK_TOLERANCE:g}) and {worst_residual:.3g} of the peak energy "
              f"remained (limit {ENERGY_TOLERANCE:g})")
        return 0
    print("\nV3 FAIL")
    for failure in failures:
        print(f"  {failure}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
