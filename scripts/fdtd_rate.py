#!/usr/bin/env python3
"""FDTD rate and memory: the measurements the 3.2 estimate rests on.

For each setup (``metal`` walls, ``open`` walls, ``open`` plus one gold block
about 1/8 of the grid) at each size, runs ``simulate`` for ``--steps`` steps
and prints:

    cell-steps/s (wall time of the stepping run, after a warm-up run that
    compiles the kernels), peak memory, and fdtd.memory_estimate's prediction
    with the measured/predicted ratio.

Peak memory is the cupy pool's total after the run (the pool keeps every
block it ever needed, so its total at the end is the run's peak footprint),
or tracemalloc's peak with numpy. Each setup frees the pool first.

Tier 2 (the Jetson; the lead runs it). Default sizes are about 1 M and 4 M
cells, under 0.6 GB each by the formula; any size whose prediction exceeds
--max-gb (2.0) is skipped, not run. Ship and run:

    tar czf /tmp/fdtd-rate.tgz scripts/fdtd_rate.py dockerfiles/maxwell_solver/fdtd.py
    scp /tmp/fdtd-rate.tgz jetson:/tmp/
    ssh jetson "mkdir -p ~/mse-bench/fdtd-rate && cd ~/mse-bench/fdtd-rate && \
        tar xzf /tmp/fdtd-rate.tgz && (ulimit -v 5000000; nice -n 10 timeout 30m \
        python3 scripts/fdtd_rate.py)"
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import tracemalloc

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "dockerfiles", "maxwell_solver"))
import fdtd  # noqa: E402

SETUPS = ("metal", "open", "gold")
DX = 5e-9


def setup(kind: str, n: int):
    """The grid, boundary and metals for one setup on an n x n x n grid."""
    shape = (n, n, n)
    metals = []
    if kind == "gold":
        region = np.zeros(shape, dtype=bool)
        lo, hi = n // 4, n // 4 + n // 2     # half each axis: 1/8 of the grid
        region[lo:hi, lo:hi, lo:hi] = True
        metals = [("Au", region)]
    return shape, ("metal" if kind == "metal" else "open"), metals


def run(kind: str, n: int, steps: int, xp):
    shape, boundary, metals = setup(kind, n)
    eps = np.ones(shape, dtype=np.float32)
    # outside the 10-cell CPML and below the gold block (z from n/4)
    source = fdtd.Source((n // 2, n // 2, max(n // 4 - 2, 11)), frequency=4e14,
                         bandwidth=2e14)
    probes = [(n // 2, n // 2, n // 2, "z")]

    def once(count):
        terms = [fdtd.LorentzDrude(m, r) for m, r in metals]
        return fdtd.simulate(eps, source, count, probes, dx=DX,
                             boundary=boundary, ade_terms=terms, xp=xp)

    pool = None
    if xp is not np:
        pool = xp.get_default_memory_pool()
    once(2)                                   # warm-up: kernels compiled
    if pool is not None:
        pool.free_all_blocks()
    else:
        tracemalloc.start()
        tracemalloc.reset_peak()
    start = time.perf_counter()
    result = once(steps)                      # returns host traces: synced
    elapsed = time.perf_counter() - start
    if pool is not None:
        peak = pool.total_bytes()
        pool.free_all_blocks()
    else:
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    if not np.all(np.isfinite(result["traces"])):
        raise SystemExit(f"{kind} n={n}: non-finite traces")
    return n ** 3 * steps / elapsed, peak


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sizes", default="100,160",
                        help="grid edges in nodes (default 100,160: 1.0 M, 4.1 M)")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--setups", default=",".join(SETUPS))
    parser.add_argument("--max-gb", type=float, default=2.0,
                        help="skip a size the formula puts above this")
    parser.add_argument("--numpy", action="store_true",
                        help="use numpy even if cupy imports")
    args = parser.parse_args(argv)

    xp = np
    if not args.numpy:
        try:
            import cupy
            xp = cupy
        except ImportError:
            print("cupy not importable: measuring numpy")
    print(f"backend {xp.__name__}, steps {args.steps}, dx {DX:g} m")
    print(f"{'setup':6} {'n':>4} {'cells':>9} {'Mcell-steps/s':>14} "
          f"{'peak MB':>9} {'formula MB':>11} {'meas/formula':>13}")
    for kind in args.setups.split(","):
        if kind not in SETUPS:
            raise SystemExit(f"unknown setup {kind!r}; choose from {SETUPS}")
        for n in (int(s) for s in args.sizes.split(",")):
            shape, boundary, metals = setup(kind, n)
            predicted = fdtd.memory_estimate(shape, boundary=boundary,
                                             metals=metals)
            if predicted > args.max_gb * 1e9:
                print(f"{kind:6} {n:4} skipped: formula says "
                      f"{predicted / 1e9:.2f} GB > {args.max_gb} GB")
                continue
            rate, peak = run(kind, n, args.steps, xp)
            print(f"{kind:6} {n:4} {n ** 3:9d} {rate / 1e6:14.2f} "
                  f"{peak / 1e6:9.1f} {predicted / 1e6:11.1f} "
                  f"{peak / predicted:13.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
