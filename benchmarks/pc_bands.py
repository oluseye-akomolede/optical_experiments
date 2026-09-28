"""Control D -- the published MPB benchmark for the 2-D PhC band solver.

Source (fetched and read when the plan was written; see
docs/photonic-crystal-plan.md sec.4 "Control D"): MPB's own Scheme tutorial,
"Bands of a Triangular Lattice", <https://mpb.readthedocs.io/en/latest/Scheme_Tutorial/>.

    geometry:  dielectric cylinders in air,  radius 0.2 a,  epsilon 12
    lattice:   triangular (the tutorial changes only the lattice from its
               square-lattice subsection; every other parameter is unchanged)
    published TM gap:  band 1 (0.275065617068082) -> band 2 (0.446289918847647),
                       a 47.4729292989213 % gap, at MPB's resolution 32.

    Frequencies are in omega a / (2 pi c) -- the module's own convention.

Acceptance, stated in the plan BEFORE running: the computed gap edges must agree
with the published figures to within 1 % of the midgap frequency. This is a
numeric-vs-numeric comparison (MPB at resolution 32 has its own discretisation
error), so 1 % of midgap is the bar, not the 1e-10 the analytic oracle earns.
If the numbers fall outside 1 %, the difference is to be EXPLAINED, not tuned.

Why this is dev-only and never on the VM: the sweep is 16 k-points x 6 bands x
2 polarisations at the production mesh, which is a mode solve. It runs as a
bounded cluster Job (the ci namespace), like the 3-D benchmarks. scipy is
already in the solver image; nothing here ships in an image or a venv.

Usage:
    python benchmarks/pc_bands.py [--divisions 40] [--bands 6] [--points 16]
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "dockerfiles", "maxwell_solver"))
import pc  # noqa: E402

# The published MPB figures, verbatim from the tutorial quoted in the plan.
MPB_B1 = 0.275065617068082
MPB_B2 = 0.446289918847647
MPB_GAP_PCT = 47.4729292989213
R_OVER_A = 0.2
EPS_ROD = 12.0


def _gap_edges(bands):
    """The 1-2 gap edges over a k-path: max of band 1, min of band 2."""
    return bands[:, 0].max(), bands[:, 1].min()


def run(divisions, n_bands, points_per_segment):
    a1, a2 = pc.lattice_vectors("triangular")
    labels, kpts = pc.high_symmetry_path("triangular", points_per_segment)
    print(f"triangular lattice, r/a = {R_OVER_A}, eps = {EPS_ROD}, "
          f"divisions = {divisions}")
    print(f"k-path: {'-'.join(l.strip('$') for l in labels)}  "
          f"({len(kpts)} k-points)")

    results = {}
    for polarisation in ("TM", "TE"):
        t0 = time.time()
        _, bands = pc.bands_from_mesh(
            "triangular", divisions, R_OVER_A, EPS_ROD ** 0.5, 1.0,
            kpts, n_bands, polarisation=polarisation,
        )
        dt = time.time() - t0
        b1, b2 = _gap_edges(bands)
        results[polarisation] = (b1, b2, dt)
        print(f"  {polarisation}: band-1 max {b1:.6f}  band-2 min {b2:.6f}  "
              f"({dt:.1f} s)")

    return results


def verdict(results, divisions):
    b1, b2, _ = results["TM"]
    mid = 0.5 * (MPB_B1 + MPB_B2)
    d1 = abs(b1 - MPB_B1) / mid
    d2 = abs(b2 - MPB_B2) / mid
    gap = (b2 - b1) / (0.5 * (b1 + b2)) * 100.0

    print()
    print(f"MPB (resolution 32):  band 1 {MPB_B1:.6f} -> band 2 {MPB_B2:.6f}, "
          f"gap {MPB_GAP_PCT:.4f} %")
    print(f"this solve (N={divisions}): band 1 {b1:.6f} -> band 2 {b2:.6f}, "
          f"gap {gap:.4f} %")
    print(f"  band-1 deviation: {d1 * 100:.4f} % of midgap "
          f"(bar 1.0000 %)")
    print(f"  band-2 deviation: {d2 * 100:.4f} % of midgap "
          f"(bar 1.0000 %)")

    within = d1 <= 0.01 and d2 <= 0.01
    print(f"Control D: {'PASS' if within else 'FAIL'} "
          f"(1 % of midgap = {0.01 * mid:.6f} in absolute frequency)")

    # Control C rides along: it is the same geometry and it is qualitative.
    # Dielectric rods in air give a TM 1-2 gap and NO TE 1-2 gap.
    tb1, tb2, _ = results["TE"]
    tm_gap = b2 > b1
    te_gap = tb2 > tb1
    print(f"Control C: TM 1-2 gap {'present' if tm_gap else 'ABSENT'}, "
          f"TE 1-2 gap {'present' if te_gap else 'absent'} "
          f"-> {'PASS' if (tm_gap and not te_gap) else 'FAIL'} "
          f"(expected: TM present, TE absent)")

    return within and tm_gap and not te_gap


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--divisions", type=int, default=40,
                    help="divisions per lattice vector (production mesh = 40)")
    ap.add_argument("--bands", type=int, default=6)
    ap.add_argument("--points", type=int, default=16,
                    help="points per k-path segment")
    args = ap.parse_args()

    t0 = time.time()
    results = run(args.divisions, args.bands, args.points)
    ok = verdict(results, args.divisions)
    print(f"total wall clock: {time.time() - t0:.1f} s")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
