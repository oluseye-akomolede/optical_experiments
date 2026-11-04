"""Measure the peak resident memory of both electrostatics solve paths.

Task 1.7. ``mse_electrostatics`` refuses a solve it estimates cannot fit, and
that estimate is only as good as the per-unknown rates behind it. Those rates
were guesses (``design.md`` §5's memory law). This bench measures them.

What is measured, and how
-------------------------
Peak RSS of THIS process around one ``solve()`` call, read from
``/proc/self/status``'s ``VmHWM`` (the kernel's own high-water mark, which does
not depend on how often we sample) and from ``resource.getrusage``. The two are
reported side by side because they disagree in a useful way: ``VmHWM`` is the
high-water mark including memory the allocator has since returned to the OS,
while ``ru_maxrss`` is the same figure as the kernel accounts it. Reporting
both is what makes an implausible number visible instead of silent.

The baseline matters more than the peak. Importing numpy and scipy already
costs tens of MiB, and a rate fitted to (peak RSS) would charge the interpreter
to the solve. So each size runs in a FRESH subprocess: the bench re-invokes
itself with ``--one-case``, that child prints a baseline taken after the mesh
is built and before ``solve()`` is called, and the growth (``VmHWM`` at the end
minus baseline) is what gets fitted. Peak-of-process minus baseline is a
conservative figure -- it counts everything the child allocated, not just the
solve -- which is the right direction for a budget check.

Sizes
-----
Each path is measured at three or more sizes, because a fit needs three points
to be falsifiable: two always produce a line, and a line through two wrong
points is indistinguishable from a law. The meshes come from
``mse_bench_fixtures.two_layer_slab`` with its resolution parameter raised,
which keeps a known-good problem (a closed-form capacitance) underneath the
measurement -- a memory number from a broken solve is not a measurement of the
solver.

Usage
-----
    python solve_memory_bench.py                  # measure, print, fit
    python solve_memory_bench.py --path direct    # one path only
    python solve_memory_bench.py --json           # machine-readable

Intended for the Jetson (tier 2), one run at a time:
``ulimit -v 5000000; nice -n 10 timeout 30m python solve_memory_bench.py``
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import numpy as np

import mse_electrostatics as E

#: The mesh sizes to measure, as the slab fixture's ``n`` per axis. The direct
#: path gets the smaller ones (its fill is what grows fast), AMG the larger, and
#: the two overlap in the middle so the crossover can be read off rather than
#: assumed.
DIRECT_SIZES = (12, 18, 24, 30)
AMG_SIZES = (12, 18, 24, 30)

#: Reduced unknowns per slab ``n``. The slab is ``n x n x 4n`` box tets; only
#: the middle free nodes are solved. Measured, not assumed -- see ``--one-case``.
SIZES_SCALED_FROM = "the fixture's own node count, printed per case"


def _peak_kb() -> int:
    """This process's peak resident set, in kB, from the kernel's high-water mark."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    raise RuntimeError("no VmHWM in /proc/self/status; cannot measure peak memory")


def _rss_kb() -> int:
    """This process's CURRENT resident set, in kB. The baseline for a growth figure."""
    with open("/proc/self/status") as fh:
        for line in fh:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    raise RuntimeError("no VmRSS in /proc/self/status")


def _one_case(size: int, path: str) -> dict:
    """Build one mesh, solve it, and report memory. The child's whole job.

    Baseline is taken AFTER the mesh exists and BEFORE ``solve()`` is called, so
    the growth figure is the solve's own memory and not the mesh's plus the
    interpreter's.
    """
    import mse_bench_fixtures as F
    from mse_bench_fixtures import _box_tets

    problem, expected = F.two_layer_slab()
    # Scale the fixture by re-meshing at a chosen resolution. The fixture's
    # geometry is fixed; only the refinement changes, so every size measures the
    # SAME physical problem and the answers stay comparable.
    n = int(size)
    side = problem["side"]
    t_plate = problem["z_plate"]
    d_a = problem["d_a"]
    d_b = 40e-6  # the fixture's default; its problem dict does not carry it
    # Four bands in z, in the fixture's own order: plate, a, b, plate. Each band
    # contributes nodes at BOTH of its faces, so the interfaces sit on node
    # planes at every resolution -- the fixture's exactness depends on that, and
    # a band that skipped its far face would leave the next part with no tet.
    xs = np.linspace(0.0, side, n + 1)
    ys = np.linspace(0.0, side, n + 1)
    n_mid = max(2, n // 2)
    zs = np.concatenate([
        np.linspace(0.0, t_plate, 3),
        np.linspace(t_plate, t_plate + d_a, n_mid + 1)[1:],
        np.linspace(t_plate + d_a, t_plate + d_a + d_b, n_mid + 1)[1:],
        np.linspace(t_plate + d_a + d_b, 2 * t_plate + d_a + d_b, 3)[1:],
    ])
    points, tets = _box_tets(xs, ys, zs)
    # Parts and per-tet regions rebuilt for the new mesh, at the SAME z
    # boundaries the fixture used (and in the same order), or the measurement
    # would be of a different problem from the one with a closed form under it.
    cz = points[tets][:, :, 2].mean(axis=1)
    z_a_mid = t_plate + d_a
    region_of_tet = np.where(
        cz < t_plate, 0,
        np.where(cz < z_a_mid, 1,
                 np.where(cz < z_a_mid + d_b, 2, 3)),
    ).astype(np.int64)
    parts = ["bottom", "a", "b", "top"]

    baseline_kb = _rss_kb()
    out = E.solve(points, tets, parts, region_of_tet, problem["eps_static"],
                  problem["electrodes"], method=path)
    peak_kb = _peak_kb()
    return {
        "size": n,
        "path": out["method"],
        "n_free": int(out["n_free"]),
        "n_points": int(points.shape[0]),
        "baseline_kb": baseline_kb,
        "peak_kb": peak_kb,
        # The solve's own growth. Never negative in a sane run; a negative value
        # means the baseline was taken above the peak, which is a measurement
        # bug, and the fit refuses it rather than taking abs().
        "grow_kb": peak_kb - baseline_kb,
        "estimate_gib": out["memory_estimate"]["bytes"] / 2**30,
        "residual": out.get("residual"),
    }


def _run_child(size: int, path: str) -> dict:
    """Run one case in its own process, so the baseline is not a previous case's."""
    cmd = [sys.executable, __file__, "--one-case", str(size), "--path", path]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"size {size} {path} failed in a child process:\n{proc.stderr[-2000:]}"
        )
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"size {size} {path}: child printed no result line")


def _fit(cases: list[dict]) -> dict:
    """Fit ``bytes = C * n**p`` and ``bytes = r * n``, and report which holds.

    Through-the-origin linear was the first model tried and it is WRONG for the
    direct path: measured on the VM at 1.9k-28k free unknowns the per-unknown
    rate rises monotonically with size (6.3 -> 13 kB), and the through-origin
    fit misses its own points by up to 116%. That is the shape of fill growth --
    SuperLU's factor is not linear in ``n`` -- so a power law is fitted as well.

    Neither model is adopted by this bench. It reports both, with per-point
    errors, and a caller that finds neither fits must say so rather than quote
    the better-looking number. The worst case is reported as a MAXIMUM relative
    error across points, not the error at the largest point: a law that is right
    where it was fitted and wrong where it will be used is the failure mode a
    single-number fit hides.
    """
    good = [c for c in cases if c["grow_kb"] > 0]
    if len(good) < 3:
        return {"refused": f"{len(good)} positive-growth case(s); a fit needs three"}
    n = np.array([c["n_free"] for c in good], dtype=float)
    b = np.array([c["grow_kb"] * 1024.0 for c in good], dtype=float)

    out = {"n_cases": len(good), "cases_used": [c["size"] for c in good]}

    p, log_c = np.polyfit(np.log(n), np.log(b), 1)
    power = np.exp(log_c) * n**p
    e_power = np.abs(power - b) / b
    out["power_law"] = {
        "exponent": float(p),
        "coefficient": float(np.exp(log_c)),
        "max_rel_error": float(e_power.max()),
        "rel_errors": [float(x) for x in e_power],
    }

    rate = float((n * b).sum() / (n * n).sum())
    linear = rate * n
    e_lin = np.abs(linear - b) / b
    out["linear_through_origin"] = {
        "bytes_per_unknown": rate,
        "max_rel_error": float(e_lin.max()),
        "rel_errors": [float(x) for x in e_lin],
    }

    # A model is only usable if it is right ACROSS the range, not just at the
    # top. 10% is generous for a memory law and still catches the 116% miss.
    out["usable"] = [
        name for name, key in (("power_law", "power_law"),
                               ("linear_through_origin", "linear_through_origin"))
        if out[key]["max_rel_error"] <= 0.10
    ]
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--one-case", type=int, metavar="N",
                    help="internal: run a single case and print its JSON")
    ap.add_argument("--path", default="both", choices=("direct", "amg", "both"))
    ap.add_argument("--json", action="store_true", help="print JSON only")
    args = ap.parse_args(argv)

    if args.one_case is not None:
        path = "amg" if args.path == "amg" else "direct"
        print(json.dumps(_one_case(args.one_case, path)))
        return 0

    paths = ["direct", "amg"] if args.path == "both" else [args.path]
    results = {}
    for path in paths:
        sizes = DIRECT_SIZES if path == "direct" else AMG_SIZES
        cases = [_run_child(n, path) for n in sizes]
        results[path] = {"cases": cases, "fit": _fit(cases)}

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    for path, res in results.items():
        print(f"\n{path} path")
        print(f"  {'size':>5} {'n_free':>9} {'baseline':>9} {'peak':>9} "
              f"{'grow':>9} {'B/unknown':>10}")
        for c in res["cases"]:
            grow_mib = c["grow_kb"] / 1024.0
            rate = (c["grow_kb"] * 1024.0 / c["n_free"]) if c["n_free"] else 0.0
            print(f"  {c['size']:>5} {c['n_free']:>9,} {c['baseline_kb']/1024:>7.1f}M "
                  f"{c['peak_kb']/1024:>7.1f}M {grow_mib:>7.1f}M {rate:>10,.0f}")
        fit = res["fit"]
        if "refused" in fit:
            print(f"  fit: REFUSED -- {fit['refused']}")
        else:
            pw = fit["power_law"]
            ln = fit["linear_through_origin"]
            print(f"  fit power law: {pw['coefficient']:.0f} * n^{pw['exponent']:.2f}, "
                  f"worst point {pw['max_rel_error']*100:.1f}% off")
            print(f"  fit linear   : {ln['bytes_per_unknown']/1024:.2f} kB/unknown, "
                  f"worst point {ln['max_rel_error']*100:.1f}% off")
            if fit["usable"]:
                print(f"  usable model: {', '.join(fit['usable'])}")
            else:
                print("  usable model: NONE -- neither law fits across the range; "
                      "do not quote either as the per-node cost")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
