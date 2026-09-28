"""The multi-material benchmark runner -- tier 2, the LEAD runs this on the Orin.

WHAT IT MEASURES. ``sphere_dielectric.py`` supplies the analytic spectrum of a
dielectric sphere of radius ``a`` and index ``n`` concentric inside a PEC sphere
of radius ``b``. This script builds that geometry as a two-part ``.obj``,
drives the REAL production path over it -- the same ``solve()`` the Jetson worker
calls, with a materials table exactly as the web app would send one -- and
reports the error against the analytic modes.

WHY THE REAL PATH AND NOT A SOLVER CALL. The feature under test is not "the
solver can do a dielectric core"; it is "an upload with a dense core reaches the
solver with the right per-element eps and comes back with the right spectrum".
A harness that built the PartSet and eps array itself would test its own
construction and be green while the upload path was broken. So this script goes
through ``load_parts`` -> ``merged_surface`` -> ``tetrahedralize`` ->
``solve()``: the same four calls, in the same order, with the same arguments the
route uses.

THE CASES. The point of the feature is a DENSE core in a COARSE cavity -- that
is what the uniform-edge decision (i') and the routing change exist for. So the
headline case is ``n_core = 2 or 3`` inside an ``n_wall = 1`` cavity, where a
solution that meshed every region at the wall's edge would leave the core
under-resolved and the mode error would show it. The empty-sphere case is the
CONTROL: it must reproduce the bare PEC sphere, which pins that the harness
itself is measuring the right geometry.

ORIN-SIZED, AND NOT RUNNABLE ON THE VM. Solving modes is tier 2 even under
``ulimit`` and ``nice``; the VM crashed on 2026-09-26 from exactly that, and its
tier-1 budget is a 4 GB cap for TINY unit tests. Do not run this script on the
VM. The per-case cost is in the comment on ``CASES`` and the command is in
``_command_hint``.

The script assumes the tree is at ``~/mse-bench/src-<sha>`` and the GPU venv at
``~/mse-bench/gpu-venv``. It takes ``--cases`` to select a subset, ``--n-modes``
and ``--grid`` to match the production job settings, ``--backend``, and
``--out`` to write the JSON report. It takes no lock itself -- wrap it in
``flock`` on ``~/mse-bench/.gpu.lock``, the SAME lock the worker takes, so a
bench run and a real job can never share the GPU:

    flock ~/mse-bench/.gpu.lock systemd-run --user --scope -p MemoryMax=5G \
        env MSE_SOLVER_BACKEND=gpu ~/mse-bench/gpu-venv/bin/python \
        ~/mse-bench/src-<sha>/benchmarks/run_multimaterial_bench.py --out /tmp/bench.json

``systemd-run --scope -p MemoryMax=5G`` is how the bench runs are bounded on the
Orin (from ``worker/README.md``): ``ulimit -v`` is an address-space limit and
stops CUDA initialising there, so the cgroup limit is used instead. Each case
runs in the SAME process, so a case that raises does not take the others down --
it is caught, recorded with its traceback, and the rest continue.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

# The solver tree is a sibling of this file's directory, exactly as the worker
# resolves it (worker/mse_worker.py: _SOLVER_DIR). Same rule here so a
# relocated tree still works and there is no second spelling of the path.
_SOLVER_DIR = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
if str(_SOLVER_DIR) not in sys.path:
    sys.path.insert(0, str(_SOLVER_DIR))
_BENCH_DIR = Path(__file__).resolve().parent
if str(_BENCH_DIR) not in sys.path:
    sys.path.insert(0, str(_BENCH_DIR))

import sphere_dielectric as REF  # noqa: E402
from mse_common import unit_scale  # noqa: E402

#: The geometry. a/b = 1/3 with n = 2 separates the lowest modes (measured gaps
#: 1.25, 1.37, 1.41), which the analytic module chose so a misidentified root is
#: visible. b = 10 mm keeps the cavity comfortably larger than the target edge
#: at 20 modes, so the case is resolved rather than marginal.
B_MM = 10.0
A_OVER_B = REF.A_OVER_B

#: The cases, each a (label, n_wall, n_core, materials) tuple.
#:
#: ``n_wall = 1.0`` for every case is not a limitation of the geometry -- it is
#: the POINT. The PEC shell's interior is vacuum in the analytic reference, and
#: the multi-part upload expresses the whole physics through the CORE's index.
#: What varies is the core's density, which is what the routing and the uniform
#: edge exist to handle.
#:
#:   empty      the control. No core index (vacuum everywhere): must reproduce
#:              the bare PEC sphere, which validates the harness end to end.
#:   n2 / n3    the headline. A dense core in a vacuum cavity. n3 is the harder
#:              case -- a shorter wavelength inside means the core must be
#:              meshed finer, which is exactly what h_min = h*n_wall/n_core does.
#:
#: THE COST, measured from the estimate alone (no mesh, no solve) at b = 10 mm,
#: a/b = 1/3, n_wall = 1, at the sizes this script defaults to:
#:
#:     --n-modes 10    empty ~26k, n2 76,408, n3 257,879
#:     --n-modes 20    empty ~52k, n2 152,817, n3 515,758
#:
#: So the default is 10 modes: every case fits well under the 600k cap, and n3
#: -- the one worth running -- is the largest. At 20 modes n3 is 515k, close
#: enough to the cap that its own retry loop may hit it; raise it deliberately,
#: not by accident. (The estimate is scale-invariant: V/h^3 is constant because
#: both scale with the object, so the cavity's SIZE does not change these
#: numbers -- the shape and the mode count do. Sizing down the sphere would not
#: have made n3 cheaper; only the mode count would.)
CASES = [
    ("empty", 1.0, None),
    ("n2", 1.0, 2.0),
    ("n3", 1.0, 3.0),
]


#: Measured mean icosphere facet edge as a fraction of ``2R/2**s``. See
#: ``_facet_subdivisions`` for the table: 0.582 at s = 1, rising to ~0.605 by
#: s = 5 (the ratio drifts as the tessellation approaches the sphere). This is
#: the mean, and the search compares it against the target, so a chosen
#: subdivision can land ~4% finer than the target in the worst case -- well
#: inside the factor-of-2 error the ``2R/2**s`` form had.
_ICOSPHERE_EDGE_RATIO = 0.598


def _facet_edge(radius: float, subdivisions: int) -> float:
    """The measured mean facet edge of an icosphere, in the radius's units."""
    return _ICOSPHERE_EDGE_RATIO * 2.0 * radius / 2 ** subdivisions


def _facet_subdivisions(radius: float, target_edge: float) -> int:
    """The icosphere subdivision whose FACET edge best matches ``target_edge``.

    This is not a detail -- it is the trap ``tests/bench_box.py`` documents at
    its own sphere builder: a surface finer than the target floors the interior
    element size, so an over-fine sphere is not a better-resolved test, it is a
    test of the decimation and refusal path. Measured here the hard way: a 24x48
    lat/long sphere (2208 facets on a 10 mm ball, edge ~1.3 mm against a 0.19 mm
    target) drove TetGen to `malloc(): invalid size (unsorted)` -- a heap abort
    inside the C++ library, which reads as a mesher bug and is really an input
    this far past what maxvolume asks for.

    An icosphere of subdivision ``s`` has ``20*4^s`` faces, and its edge is
    shorter than the LAT/LONG quad chord ``2R/2^s`` by ~1/sqrt(3), because each
    icosahedron face is a triangle rather than a chord across a quad. Measured
    (R = 10 mm; mean over the three edges of each face):

        s   faces    mean edge    2R/2^s    ratio
        1      80        5.823    10.000    0.582
        2     320        2.993     5.000    0.599
        3    1280        1.507     2.500    0.603
        4    5120        0.755     1.250    0.604
        5   20480        0.378     0.625    0.605

    So ``2R/2^s`` UNDERSTATES the real edge by ~1.72x. The measured ratio is in
    ``_ICOSPHERE_EDGE_RATIO`` and drives the search, so the thing being tested
    ("is the facet at the target") is the real edge, not a proxy for it.

    The contract is ``tests/bench_box.py:259-264``'s: step subdivisions UP while
    the facet is COARSER than the target, and stop at the first one at or below
    it -- "closest to the target from below (never finer than needed)". The
    search direction is upward and the stopping test is ``edge <= target``, NOT
    ``>=``: an edge far coarser than the target is not acceptable, it is
    unrefined. (I inverted this once, which returned a bare 20-face icosahedron
    for every target down to 0.02 mm.)

    One correction to bench_box's version: it compares ``2R/2^s``, the quad
    chord, against the target, so it stops one subdivision too late and
    over-refines by ~2x. Measured here, R = 10, target = 1.45: the correct
    answer is s = 3 (real edge 1.507, 1.04x the target); the ``2R/2^s`` form
    picks s = 4 (real edge 0.755, 0.52x -- twice as fine as asked, the
    over-refinement this function exists to prevent).

    Raises when the target is finer than s = 7 can reach: 20*4^7 is already 327k
    facets, so a finer target is a bad input rather than a reason to keep
    dividing, and returning 7 anyway would hand the mesher a surface 5x coarser
    than it asked for with no signal that the request was not met.
    """
    for s in range(1, 8):
        if _facet_edge(radius, s) <= target_edge:
            return s
    raise ValueError(
        f"target facet edge {target_edge:g} is finer than an icosphere of "
        f"subdivision 7 can make for radius {radius:g} "
        f"(edge {_facet_edge(radius, 7):g}); refine the target or the geometry"
    )


def _parts_obj(specs, target_edge: float) -> bytes:
    """A ``.obj`` from ``(name, radius)`` pairs, in mm.

    Written out through trimesh's OBJ exporter (via an ``o`` line per part, so
    the part NAMES are visible in the file). The names are what ``mse_parts``'
    line-by-line walk reads, so the names in the file are the names the
    materials table must key on -- that correspondence is the thing the upload
    route proves.

    ``specs`` with ONE entry is a one-part upload: the empty (vacuum) control,
    which must go down the one-part path exactly as a user's single-material
    file would. Passing a two-part file with ``materials=None`` would instead
    take the one-part path ON a two-part file, which is not a case the product
    has -- the route sends a materials table whenever the file has parts.

    Every sphere is faceted at ``target_edge`` (see ``_facet_subdivisions``),
    so both parts exercise the ordinary path a real upload takes.
    """
    import trimesh

    lines: list[str] = []
    n_verts_written = 0
    for name, r in specs:
        s = _facet_subdivisions(r, target_edge)
        mesh = trimesh.creation.icosphere(subdivisions=s, radius=r)
        verts = np.asarray(mesh.vertices, float)
        faces = np.asarray(mesh.faces, np.int64)
        lines.append(f"o {name}")
        for v in verts:
            lines.append(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}")
        # ``f`` indices are 1-based into the WHOLE file's vertex list, so they
        # are offset by the number of vertices written for earlier parts -- NOT
        # by a line count, which was the first version's bug (it indexed past
        # the end and the parser said so).
        for f in faces:
            lines.append(
                f"f {f[0]+1+n_verts_written} "
                f"{f[1]+1+n_verts_written} {f[2]+1+n_verts_written}"
            )
        n_verts_written += len(verts)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _analytic(n_core: float | None, a_over_b: float = A_OVER_B):
    """The analytic reference for the case, as MULTIPLETS.

    Returns ``(n, spectrum)`` where ``spectrum`` is ``(kb, l, family,
    multiplicity)`` over l = 1..L_MAX for both TE and TM.

    It used to return only the l = 1 TE and TM arrays, and that is what made
    the first Orin run report "8 spurious" modes: the solver has no l cutoff, so
    its l = 2 quintets were compared against a reference that did not contain
    them. The spectrum is the whole union now, and matching is per multiplet.

    ``a_over_b`` defaults to the IDEAL geometry and is overridden with the
    MESHED ratio by the caller -- see ``_effective_radii`` for why scoring at
    the ideal charges the solver for the faceting.
    """
    n = 1.0 if n_core is None else float(n_core)
    b = 1.0  # the reference module works in units of b
    return n, REF.spectrum(a_over_b, b, n)


def _effective_radii(parts, n_core, units: str = "mm"):
    """``(a_eff, b_eff, a_over_b_eff)`` from the FACETED surfaces, in mm.

    A faceted sphere of the requested radius encloses LESS volume than the ideal
    one, so the mesh actually built is not the geometry the analytic reference
    describes. Scoring at the IDEAL ``a/b`` charges the solver for the faceting,
    which is a property of the mesh generator rather than of the physics under
    test. Measured by the lead: at the IDEAL a/b = 1/3 the TE errors are
    1.0-1.4%, and at the MESHED geometry every dielectric multiplet is within
    0.39%.

    The radii come from ``PartSet.filled_volume_m3`` -- the SAME helper
    ``mse_common.region_plan`` uses, so the geometry this scores against and the
    geometry the mesher plans are one quantity rather than two that could drift.
    The faceted surfaces are exactly what the tets fill, which is why the
    volumes have to come from here and not from the solver's mesh: the runner is
    not allowed to touch solver code, and it does not need to.

    ``units`` is passed EXPLICITLY because ``filled_volume_m3`` has no default,
    deliberately -- a defaulted unit argument is how a whole multi-part estimate
    came out wrong by the cube of the unit ratio. The caller must say which unit
    it means, and this caller means what the job was submitted in.

    The cavity radius is the OUTER wall's ENCLOSED volume, not its filled
    volume. The wall fills only the vacuum shell, but the PEC boundary the modes
    live inside is the whole sphere its surface describes, core included --
    ``PartSet.surfaces["wall"].volume`` is that, and it is the quantity the
    benchmark's ``b`` means. Using the filled volume here would subtract the
    core and report a cavity smaller than the wall, which is backwards.

    Returns the IDEAL values whenever the parts or the named regions are
    unusable, so a caller that cannot get the volumes falls back to the previous
    behaviour rather than to a quietly wrong number.
    """
    ideal = (A_OVER_B * B_MM, B_MM, A_OVER_B)
    if parts is None:
        return ideal
    try:
        by_name = {name: i for i, name in enumerate(parts.names)}
        # ENCLOSED, in the user's units: scale**3 is the same conversion
        # ``filled_volume_m3`` applies, done here on the raw volume.
        wall = parts.parts[by_name["wall"]]
        v_cav = abs(float(wall.volume)) * unit_scale(units) ** 3
        v_core = (parts.filled_volume_m3(by_name["core"], units)
                  if n_core is not None else None)
    except (KeyError, TypeError, ValueError, IndexError):
        return ideal
    if not (v_cav > 0) or (v_core is not None and not (v_core > 0)):
        return ideal

    def cbrt_radius_mm(v):
        return (3.0 * v / (4.0 * np.pi)) ** (1.0 / 3.0) / 1e-3

    b_eff = cbrt_radius_mm(v_cav)
    if v_core is None:
        return (0.0, b_eff, 0.0)
    a_eff = cbrt_radius_mm(v_core)
    return (a_eff, b_eff, a_eff / b_eff)


def _run_case(label: str, n_core: float | None, *, n_modes: int, grid: int,
              work: Path, verbose: bool, backend: str = "gpu") -> dict:
    """Build, mesh, solve and score ONE case over the real production path."""
    import mse_maxwell_solver as SOLVER

    a = A_OVER_B * B_MM
    # The facet edge each part must be built at. Facets are sized from the
    # WHOLE-object target edge, the same edge the mesher's own surface check
    # uses, so the surface is never finer than the interior it bounds (the
    # bench_box trap). A dense core does NOT get finer facets even though it
    # is meshed finer: the interior refinement is TetGen's job, and putting the
    # whole n_core factor into the SURFACE would floor the interior and turn a
    # resolution test into a decimation test.
    import trimesh
    import mse_meshing as MESH
    probe = trimesh.creation.icosphere(subdivisions=2, radius=B_MM)
    target_units = MESH.target_edge_length(probe, n_modes=n_modes,
                                           units="mm", n_index=1.0)
    if n_core is None:
        # The CONTROL: a plain vacuum cavity, one part. It exercises the
        # one-part path (which the feature must leave byte-identical) and
        # validates the harness -- it must reproduce the bare PEC sphere.
        obj = _parts_obj([("wall", B_MM)], target_units)
    else:
        obj = _parts_obj([("wall", B_MM), ("core", a)], target_units)

    work.mkdir(parents=True, exist_ok=True)
    store_dir = work / "store"
    store_dir.mkdir(exist_ok=True)
    job_id = f"bench-{label}"

    store = SOLVER.STORE.LocalStore(str(store_dir))
    store.write(SOLVER.STORE.job_key(job_id, "input.obj"), obj)

    materials = None if n_core is None else {"wall": 1.0, "core": float(n_core)}

    t0 = time.time()
    summary = SOLVER.solve(
        job_id, "mm", 1.0, n_modes, store, store,
        max_tets=600_000, grid=grid, verbose=verbose,
        backend=backend, materials=materials,
    )
    elapsed = time.time() - t0

    modes_json = json.loads(store.read(SOLVER.STORE.job_key(job_id, "modes.json")))
    freqs = np.array([m["freq_hz"] for m in modes_json["modes"]], float)

    # k = 2 pi f / c, and b is metres, so kb = 2 pi f b / c.
    C = 299_792_458.0
    b_m = B_MM * 1e-3
    kb = 2.0 * np.pi * freqs * b_m / C

    # Score at the geometry the mesh ACTUALLY built, and keep the ideal one
    # alongside it so the faceting cost is visible rather than hidden.
    #
    # The parts come from parsing the SAME bytes the solver was handed, with the
    # SAME parser -- a map built by zipping the order this script THINKS it
    # wrote against the region ids would be a second source of truth for the one
    # thing that decides which volume is the core's. The volumes come from
    # ``filled_volume_m3``, the helper the mesher's own region plan uses, so this
    # scores against the geometry the mesher planned rather than a re-derivation
    # of it. Nothing here touches solver code.
    parts = SOLVER.PARTS.load_parts(obj, units="mm") if materials else None
    a_eff, b_eff, ab_eff = _effective_radii(parts, n_core, units="mm")
    _n, ref = _analytic(n_core, ab_eff)
    _n_ideal, ref_ideal = _analytic(n_core, A_OVER_B)
    return {
        "label": label,
        "n_core": n_core,
        "n_tets": summary["n_tets"],
        "elapsed_s": elapsed,
        "kb_computed": kb.tolist(),
        "reference": [{"kb": k, "l": l, "family": f, "multiplicity": m}
                      for k, l, f, m in ref],
        "reference_ideal": [{"kb": k, "l": l, "family": f, "multiplicity": m}
                            for k, l, f, m in ref_ideal],
        "a_over_b": ab_eff,
        "a_over_b_ideal": A_OVER_B,
        "a_eff_mm": a_eff,
        "b_eff_mm": b_eff,
        "achieved_median_edge_m": modes_json.get("achieved_median_edge_m"),
        "solver_backend": modes_json.get("solver_backend"),
        "solver_worst_residual": modes_json.get("solver_worst_residual"),
        "n_modes_requested": n_modes,
    }


def _score(case: dict, key: str = "reference") -> dict:
    """Score the computed spectrum against the analytic MULTIPLETS.

    Delegates to the reference module's ``score_multiplets``, which clusters
    the computed modes into degenerate multiplets and matches them one-to-one
    against the reference multiplets, checking the mode COUNT as well as the
    centre.

    This replaced a scorer that matched individual modes against individual
    modes of an l = 1-only reference. That scorer reported 8 "spurious" modes on
    the first Orin run; every one of them was real l >= 2 physics the reference
    did not contain. Matching multiplets is not a refinement of the old scorer,
    it is the fix for what it got wrong: the reference is a union over l with
    degeneracy 2l+1, and a scorer that cannot see the degeneracy cannot see the
    spectrum.

    ``n_requested`` is passed so the scorer can tell a partial top multiplet
    (the mode-count cutoff, expected) from a genuinely missing mode (a defect).
    """
    computed = np.asarray(case.get("kb_computed", []), float)
    if computed.size == 0:
        return {"n_matched": 0, "note": "no modes came back"}
    reference = [(r["kb"], r["l"], r["family"], r["multiplicity"])
                 for r in case[key]]
    return REF.score_multiplets(computed, reference,
                                n_requested=case.get("n_modes_requested"))



def _command_hint() -> str:
    return (
        "flock ~/mse-bench/.gpu.lock systemd-run --user --scope -p MemoryMax=5G \\\n"
        "    env MSE_SOLVER_BACKEND=gpu ~/mse-bench/gpu-venv/bin/python \\\n"
        "    ~/mse-bench/src-<sha>/benchmarks/run_multimaterial_bench.py \\\n"
        "    --out /tmp/mse-bench-<sha>.json"
    )


def main(argv: list[str]) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", default="empty,n2,n3",
                   help="comma-separated subset of: " +
                        ",".join(c[0] for c in CASES))
    p.add_argument("--n-modes", type=int, default=10)
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--out", default=None, help="write the JSON report here")
    p.add_argument("--work", default=None, help="work directory (default: a temp dir)")
    p.add_argument("--backend", default="gpu", choices=("cpu", "gpu"),
                   help="solver backend; the tier-1 smoke run uses cpu")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    wanted = {c.strip() for c in args.cases.split(",") if c.strip()}
    selected = [c for c in CASES if c[0] in wanted]
    if not selected:
        print(f"no cases selected; available: {[c[0] for c in CASES]}",
              file=sys.stderr)
        return 2

    import tempfile
    work_root = Path(args.work) if args.work else Path(
        tempfile.mkdtemp(prefix="mse-bench-"))
    print(f"[bench] work root: {work_root}")
    print(f"[bench] command: {_command_hint()}")
    print()

    report = {"cases": []}
    failures = 0
    for label, n_wall, n_core in selected:
        print(f"=== case {label}: n_wall={n_wall} n_core={n_core} ===")
        try:
            case = _run_case(label, n_core, n_modes=args.n_modes, grid=args.grid,
                             work=work_root / label, verbose=not args.quiet,
                             backend=args.backend)
        except Exception as exc:  # noqa: BLE001 - a case failing is data
            failures += 1
            print(f"[bench] case {label} RAISED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            report["cases"].append({"label": label, "n_core": n_core,
                                    "error": f"{type(exc).__name__}: {exc}"})
            continue
        score = _score(case)
        case["score"] = score
        report["cases"].append(case)
        print(f"[bench] {label}: {case['n_tets']:,} tets, "
              f"{case['elapsed_s']:.1f}s, backend {case['solver_backend']}")
        # The geometry the score is taken at, and the ideal it would be taken at
        # if the mesh were not faceted. Both, because the gap between them is
        # the mesh generator's cost and not the solver's error -- printing only
        # one of them hides which of the two a number is measuring.
        print(f"[bench] {label}: scored at a/b={case['a_over_b']:.5f} "
              f"(a_eff {case['a_eff_mm']:.4f} mm, b_eff {case['b_eff_mm']:.4f} mm); "
              f"ideal a/b={case['a_over_b_ideal']:.5f}")
        if score.get("n_matched"):
            print(f"[bench] {label}: {score['n_matched']}/"
                  f"{score['n_clusters']} computed multiplets matched a "
                  f"reference multiplet, worst rel err "
                  f"{score['worst_rel_error']:.4f}")
            for m in score["matched"]:
                flag = "" if m["count_matches"] else (
                    "  [partial: mode-count cutoff]"
                    if m["truncated_at_request"] else "  [*** FEWER MODES ***]")
                print(f"    kb {m['centre']:.4f} x{m['count']} -> "
                      f"{m['reference_kb']:.4f} l={m['l']} {m['family']} "
                      f"x{m['reference_multiplicity']} "
                      f"err {m['rel_error']*100:.3f}%{flag}")
        if score.get("spurious"):
            # Its own line: the matched/coverage numbers are blind to it.
            print(f"[bench] {label}: WARNING {len(score['spurious'])} computed "
                  f"multiplet(s) match no reference (spurious)")
            for s in score["spurious"]:
                print(f"    kb {s['centre']:.4f} x{s['count']} "
                      f"nearest ref {s['nearest_rel']}")
        if score.get("count_mismatches"):
            print(f"[bench] {label}: *** {len(score['count_mismatches'])} "
                  f"multiplet(s) short of modes below the cutoff ***")
        print()

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"[bench] wrote {args.out}")

    print("=" * 60)
    print(f"[bench] {len(report['cases']) - failures}/{len(selected)} cases ran; "
          f"{failures} raised")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
