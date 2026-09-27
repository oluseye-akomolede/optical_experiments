"""Tier-2 bench: solve a PEC box or sphere and print the error table vs analytic.

This is meant to run on the Jetson (aarch64, ~5GB free, one solve at a time) or
as a bounded Kubernetes Job -- never natively on the VM. It uses the *production*
meshing path and the *production* eigensolver, so the numbers it prints are the
numbers a real upload would get.

    python tests/bench_box.py --a 10 --b 7 --d 5 --units mm --n-index 1.0 \
        --n-modes 10 --epw 5 --max-tets 80000
    python tests/bench_box.py --shape sphere --radius 5 --units mm \
        --n-modes 20 --max-tets 80000

It prints:

* the tet / edge / free-edge counts and the achieved mesh edge length,
* the computed vs analytic in-medium k, with the per-mode relative error,
* the solve time and the process peak RSS.

Two shapes, each with a closed-form spectrum to compare against:

``--shape box`` (default). A PEC box a x b x d has
k^2 = (m pi/a)^2 + (p pi/b)^2 + (q pi/d)^2 with **at most one index zero**:

* all three indices non-zero  -> doubly degenerate (a TE and a TM mode)
* exactly one index zero      -> single

For 10x7x5 mm, n=1, the first ten in-medium k (1/mm) are

    0.5478 0.7025 0.7721 0.7721 0.8336 0.8336 0.8886 0.9510 0.9955 0.9955

where 0.7721 is the accidental pair (2,1,0)/(0,1,1) and 0.8336 is the TE/TM
pair (1,1,1).

``--shape sphere``. A PEC sphere of radius R has TM_l modes at the roots of
d/dx [x j_l(x)] = 0 and TE_l modes at the roots of j_l(x) = 0, with x = kR and
each (2l+1)-fold degenerate. The roots are computed here with scipy rather than
hardcoded, so the benchmark does not depend on a table staying right. The lowest
five are TM1 2.7437 (x3), TM2 3.8702 (x5), TE1 4.4934 (x3), TM3 4.9734 (x7) and
TE2 5.7635 (x5) -- 23 modes, enough to check 20.

A sphere is the better test of *faceting*: its surface is approximated by flat
triangles, so the measure is a real, slightly larger cavity. Expect about 1%
error, comfortably under the 3% threshold but much larger than the box's.
"""

from __future__ import annotations

import argparse
import os
import resource
import sys
import time
from pathlib import Path

# Cap threads before numpy loads, matching the container settings, so the bench
# measures the same thing the Job does.
_THREADS = os.environ.get("MSE_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dockerfiles" / "maxwell_solver"))

import mse_eigen as EIG  # noqa: E402
import mse_fem as F  # noqa: E402
import mse_maxwell_solver as SOLVER  # noqa: E402
import mse_meshing as MESH  # noqa: E402
import mse_storage as STORE  # noqa: E402


def analytic_box_k(a: float, b: float, d: float, n_modes: int, n_max: int = 8):
    """Lowest ``n_modes`` analytic in-medium k for a PEC box, degeneracies included."""
    values: list[float] = []
    for m in range(n_max):
        for p in range(n_max):
            for q in range(n_max):
                nonzero = (m > 0) + (p > 0) + (q > 0)
                if nonzero < 2:
                    continue
                k2 = (m * np.pi / a) ** 2 + (p * np.pi / b) ** 2 + (q * np.pi / d) ** 2
                values.extend([k2] * (2 if nonzero == 3 else 1))
    values.sort()
    if len(values) < n_modes:
        raise SystemExit(
            f"analytic search range too small for {n_modes} modes; raise n_max"
        )
    return np.sqrt(np.array(values[:n_modes]))


def analytic_sphere_k(radius: float, n_modes: int, l_max: int = 12):
    """Lowest ``n_modes`` analytic in-medium k for a PEC sphere, in 1/m.

    TM_l modes solve ``d/dx [x j_l(x)] = 0`` and TE_l modes ``j_l(x) = 0``, with
    x = kR and each level (2l+1)-fold degenerate. Roots come from scipy so the
    benchmark cannot drift from a stale hardcoded table.
    """
    from scipy.optimize import brentq
    from scipy.special import spherical_jn

    def tm(l, x):
        return spherical_jn(l, x) + x * spherical_jn(l, x, derivative=True)

    def te(l, x):
        return spherical_jn(l, x)

    def roots(func, l, limit=8):
        xs = np.linspace(1e-9, 40.0, 40000)
        vals = func(l, xs)
        found = []
        for a, b, fa, fb in zip(xs[:-1], xs[1:], vals[:-1], vals[1:]):
            if not (np.isfinite(fa) and np.isfinite(fb)):
                continue
            if fa * fb < 0:
                found.append(brentq(lambda x: func(l, x), a, b))
            if len(found) >= limit:
                break
        return found

    levels: list[tuple[float, int]] = []
    for l in range(1, l_max + 1):
        degeneracy = 2 * l + 1
        for x in roots(tm, l):
            levels.append((x, degeneracy))
        for x in roots(te, l):
            levels.append((x, degeneracy))
    levels.sort()
    if not levels:
        raise SystemExit("could not find any analytic sphere roots")

    values: list[float] = []
    for x, degeneracy in levels:
        values.extend([x / radius] * degeneracy)
        if len(values) >= n_modes:
            break
    if len(values) < n_modes:
        raise SystemExit(
            f"analytic search found only {len(values)} sphere modes; raise l_max"
        )
    return np.array(values[:n_modes])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shape", default="box", choices=("box", "sphere"),
                        help="which analytic case to build (default: box)")
    parser.add_argument("--a", type=float, default=10.0,
                        help="box size in --units (box only)")
    parser.add_argument("--b", type=float, default=7.0, help="box only")
    parser.add_argument("--d", type=float, default=5.0, help="box only")
    parser.add_argument("--radius", type=float, default=5.0,
                        help="sphere radius in --units (sphere only)")
    parser.add_argument("--units", default="mm", choices=sorted(MESH.UNIT_SCALE))
    parser.add_argument("--n-index", type=float, default=1.0)
    parser.add_argument("--n-modes", type=int, default=10)
    parser.add_argument("--epw", type=float, default=None,
                        help="edges per wavelength (default: MSE_EDGES_PER_WAVELENGTH)")
    parser.add_argument("--max-tets", type=int,
                        default=int(os.environ.get("MSE_MAX_TETS",
                                                  SOLVER.DEFAULT_MAX_TETS)))
    parser.add_argument("--tolerance", type=float, default=3.0,
                        help="pass threshold in percent (default: 3.0)")
    parser.add_argument("--allow-coarse", action="store_true",
                        help="solve even if the mesh is too coarse to meet the "
                             "resolution rule (charts the accuracy trade-off)")
    parser.add_argument("--no-enforce-resolution", action="store_true",
                        help="alias for --allow-coarse, kept so an existing "
                             "invocation does not silently lose enforcement")
    parser.add_argument("--surface-oversample", type=int, default=1,
                        help="refine the surface well past the target element "
                             "edge, to exercise the surface decimation path on "
                             "a deliberately over-refined surface -- what a CAD "
                             "export or 3-D scan looks like. Sphere: extra "
                             "icosphere subdivisions. Box: finer facets, which "
                             "also demonstrates the planar over-count "
                             "limitation. 1 = facets at the target (default)")
    parser.add_argument("--write", action="store_true",
                        help="ALSO run the production writing stage "
                             "(sampling, .bin, .vtu, .glb, modes.json) into a "
                             "temporary directory, and print its time and peak "
                             "RSS separately. Off by default because it roughly "
                             "doubles the peak memory; on because skipping it "
                             "is how the (chunk, n_tets, 3) sampler reached the "
                             "cluster unseen")
    parser.add_argument("--write-dir", default=None,
                        help="where --write puts the job directory (default: a "
                             "fresh temp dir, printed then left in place)")
    return parser


def run_writing_stage(obj_bytes, job_id, units, n_index, n_modes, max_tets,
                      grid, write_dir, verbose=True) -> dict:
    """Drive the PRODUCTION ``solve()`` so the writing stage really runs.

    Deliberately not a re-implementation of the writing code: a bench that
    copies the pipeline stops measuring the pipeline the moment the two drift
    apart, and this is the stage that OOM-killed the Job. It writes through
    ``LocalStore`` -- the same store class the container uses for
    ``--local-out`` -- so the bytes written are the production bytes.

    Returns the summary dict ``solve()`` produces, whose ``timings`` already
    separates ``writing`` from the other stages, plus what was written.
    """
    import tempfile

    import mse_maxwell_solver as SOLVER

    root = Path(write_dir) if write_dir else Path(tempfile.mkdtemp(
        prefix="mse-bench-write-"))
    stores = STORE.LocalStore(root)
    stores.write(STORE.job_key(job_id, "input.obj"), obj_bytes)

    t0 = time.time()
    summary = SOLVER.solve(
        job_id, units, n_index, n_modes, stores, stores,
        max_tets=max_tets, grid=grid, verbose=verbose,
    )
    summary["wall_s"] = time.time() - t0
    summary["root"] = str(root)

    written = {}
    for path in sorted((root / "jobs" / job_id).glob("**/*")):
        if path.is_file():
            written[path.name] = path.stat().st_size
    summary["written"] = written
    return summary


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.epw is not None:
        os.environ["MSE_EDGES_PER_WAVELENGTH"] = str(args.epw)

    import trimesh

    a, b, d = args.a, args.b, args.d
    print(f"{args.shape} {a} x {b} x {d} {args.units}, n={args.n_index}, "
          f"{args.n_modes} modes, max_tets={args.max_tets}, "
          f"epw={os.environ.get('MSE_EDGES_PER_WAVELENGTH', '7')}")

    if args.shape == "sphere":
        # An icosphere whose facet edge sits AT the target element edge, so the
        # bench exercises the ordinary path a real upload takes. Making the
        # facets deliberately finer used to be what this did, and that was the
        # bug Task 3 fixed: a fine surface floors the interior element size, so
        # the "matched" surface was really testing the refusal path. The
        # subdivision count is derived from the target edge rather than guessed.
        radius = args.radius
        probe, _scale = MESH.load_surface(
            trimesh.creation.icosphere(subdivisions=1,
                                       radius=radius).export(file_type="obj").encode(),
            units=args.units,
        )
        target = MESH.target_edge_length(probe, n_modes=args.n_modes,
                                        units=args.units)
        # An icosphere of subdivision s has 20*4^s faces and an edge that halves
        # each step, so the facet edge is about 2R/2^s. Pick the s whose facets
        # are closest to the target from below (never finer than needed).
        subdivisions = 1
        facet = 2.0 * radius / 2 ** subdivisions
        while facet > target and subdivisions < 7:
            subdivisions += 1
            facet = 2.0 * radius / 2 ** subdivisions

        if args.surface_oversample > 1:
            # Deliberately too fine, to exercise the decimation path -- this is
            # what a CAD export or a 3-D scan looks like.
            oversample = int(args.surface_oversample)
            subdivisions = min(7, subdivisions + oversample)
            facet = 2.0 * radius / 2 ** subdivisions

        mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
        print(f"sphere radius {radius} {args.units}, icosphere subdivisions "
              f"{subdivisions} ({len(mesh.faces)} facets, facet edge ~{facet:.4g} "
              f"{args.units} vs target {target:.4g}"
              f"{', deliberately OVER-REFINED' if args.surface_oversample > 1 else ''})")
        obj_bytes = mesh.export(file_type="obj").encode()
        a = b = d = None
    else:
        box = trimesh.creation.box(extents=[a, b, d])
        if args.surface_oversample > 1:
            # A box whose facets are deliberately finer than the target edge:
            # the planar shape that exposes the decimation limitation. Coplanar
            # facets decimate to long thin diagonal slivers, which TetGen must
            # respect, so this case over-counts where the sphere does not.
            step = 2.0 * max(a, b, d) / 8.0 / (2 ** (args.surface_oversample - 1))
            box = box.subdivide_to_size(max_edge=step)
            print(f"box surface deliberately OVER-REFINED to {len(box.faces):,} "
                  f"facets (max edge {step:.4g} {args.units})")
        obj_bytes = box.export(file_type="obj").encode()

    # -- production meshing path -----------------------------------------
    t0 = time.time()
    surface, _scale = MESH.load_surface(obj_bytes, units=args.units)
    points_m, tets = MESH.tetrahedralize(
        surface, args.units, n_modes=args.n_modes, n_index=args.n_index,
        max_tets=args.max_tets, verbose=True,
        enforce_resolution=not (args.allow_coarse
                                 or args.no_enforce_resolution),
    )
    t_mesh = time.time() - t0

    # -- production assembly + solve --------------------------------------
    t0 = time.time()
    edges, tet_edges, _signs = F.build_edges(tets)
    n_edges = int(edges.shape[0])
    A, B = F.assemble_global(points_m, tets, tet_edges, n_edges)
    constrained = F.boundary_edges(tets, edges)
    A_free, B_free, free_edges = F.pec_eliminate(A, B, constrained, n_edges)
    t_assemble = time.time() - t0

    t0 = time.time()
    eigenvalues, _eigenvectors, _sigma = EIG.solve_modes(
        A_free, B_free, args.n_modes, points_m,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free_edges, constrained=constrained, verbose=True,
    )
    t_solve = time.time() - t0

    k_per_m, _k0, freq_hz, wavelength_m = EIG.to_physical(eigenvalues, args.n_index)

    # the eigensolver returns modes in ascending k already, but sort together so
    # the table cannot silently mismatch a frequency with the wrong wavenumber
    order = np.argsort(k_per_m)
    k_per_m = k_per_m[order]
    freq_hz = freq_hz[order]
    wavelength_m = wavelength_m[order]

    # convert the in-medium k back to the user's length unit to compare with the
    # analytic expression written in those units
    scale = MESH.unit_scale(args.units)
    k_user = k_per_m * scale                          # 1 / user unit
    if args.shape == "sphere":
        # analytic_sphere_k takes the radius in metres and returns k in 1/m;
        # multiply by the scale to express it per user unit, or the comparison
        # is off by 1000x for mm. (k_per_m * scale is how k_user is built, so
        # the analytic side has to use the same conversion.)
        expected = analytic_sphere_k(args.radius * scale, args.n_modes) * scale
    else:
        expected = analytic_box_k(a, b, d, args.n_modes)
    rel_pct = (k_user / expected - 1.0) * 100.0

    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0

    print()
    print("=" * 72)
    print(f"mesh : {len(tets):,} tets, {n_edges:,} edges, "
          f"{A_free.shape[0]:,} free, {constrained.size:,} PEC  ({t_mesh:.1f}s)")
    print(f"sizes: {points_m.shape[0]:,} points, "
          f"achieved median edge "
          f"{MESH.achieved_edge_length(points_m, tets) / scale:.4g} {args.units}")
    print(f"times: mesh {t_mesh:.1f}s, assemble {t_assemble:.1f}s, "
          f"solve {t_solve:.1f}s, total {t_mesh + t_assemble + t_solve:.1f}s")
    print(f"peak RSS: {rss_mb:.0f} MB")
    print("=" * 72)
    print(f"{'#':>3}  {'k computed':>12}  {'k analytic':>12}  {'error %':>9}  "
          f"{'freq GHz':>10}  {'lam0 mm':>9}")
    for i in range(len(k_user)):
        print(f"{i + 1:>3}  {k_user[i]:>12.6f}  {expected[i]:>12.6f}  "
              f"{rel_pct[i]:>+9.3f}  {freq_hz[i] / 1e9:>10.4f}  "
              f"{wavelength_m[i] * 1e3:>9.4f}")
    print("=" * 72)

    worst = float(np.max(np.abs(rel_pct)))
    spurious = int(np.sum(k_per_m <= EIG.SPURIOUS_FRACTION * k_per_m.max()))
    print(f"worst |error| = {worst:.3f}%  (threshold {args.tolerance}%)")
    print(f"non-physical modes: {spurious}")

    writing_ok = True
    if args.write:
        # The stage that reached the cluster unmeasured. It runs as a SECOND
        # solve, on purpose: the point is to watch the writing stage's own
        # time and peak, and sharing this process's already-high-water RSS with
        # the analytic solve above would hide both. Dividing RSS at the end by
        # the pre-writing value is what makes the number attributable.
        print()
        print("=" * 72)
        print("writing stage (production solve() through LocalStore)")
        rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        try:
            summary = run_writing_stage(
                obj_bytes, "bench", args.units, args.n_index, args.n_modes,
                args.max_tets, 64, args.write_dir,
            )
        except Exception as exc:  # noqa: BLE001 - report, do not hide
            print(f"WRITING STAGE FAILED: {type(exc).__name__}: {exc}")
            writing_ok = False
        else:
            rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
            timings = summary.get("timings", {})
            for stage in ("meshing", "assembling", "solving", "writing"):
                if stage in timings:
                    print(f"  {stage:>10}: {timings[stage]:6.1f} s")
            print(f"  {'total':>10}: {timings.get('total', 0.0):6.1f} s "
                  f"(wall {summary['wall_s']:.1f} s)")
            print(f"  peak RSS after writing: {rss_after:.0f} MB "
                  f"(was {rss_before:.0f} MB before the writing stage)")
            print(f"  job dir: {summary['root']}")
            total_bytes = sum(summary["written"].values())
            print(f"  wrote {len(summary['written'])} files, "
                  f"{total_bytes / 1e6:.1f} MB "
                  f"({summary['n_tets']:,} tets, {summary['n_modes']} modes)")
            for name, size in sorted(summary["written"].items()):
                print(f"      {name:>20}  {size / 1e6:9.3f} MB")
            # A solve that wrote no mode files did not exercise the stage.
            binary = [n for n in summary["written"] if n.endswith(".bin")]
            if len(binary) != summary["n_modes"]:
                print(f"  EXPECTED {summary['n_modes']} .bin files, "
                      f"got {len(binary)}")
                writing_ok = False
        print("=" * 72)

    ok = worst <= args.tolerance and spurious == 0 and writing_ok
    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
