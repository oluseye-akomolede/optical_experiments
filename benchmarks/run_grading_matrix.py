"""A validation matrix for per-region mesh grading, as a runnable bench mode.

WHY THIS EXISTS. The owner's question (2026-10-13): *"Is this tightening just
for one job? The meshing needs to work generally for any mesh supplied."* The
other bench's fixtures are one concentric two-region family of spheres; grading
can hold on that family and still be wrong for the next upload. This file is
the answer in a form that can be RUN, not prose: a case list laid out over the
axes an arbitrary upload actually varies, one case per invocation, each sized to
fit the tier-2 budget.

THE AXES, and which case covers each (``_coverage`` exposes this as data so a
test can fail when a table edit drops one):

* index contrast 1.5, 2, 3, 4             -- c1.5-thick .. c4-thick
* shell thickness, thin (the thickness     -- c2-thin, c3-thin, c4-thin
  clamp BITES) and thick (it does not)       vs the *-thick cases
* number of regions 2 and 3                -- *-thick vs three-region
* shape: non-spherical, non-concentric     -- box-sphere, offset-sphere
* reference: analytic and uniform-refined  -- concentric vs the shape cases

THE CRITERION, per case, is the owner's: the graded mesh's error is no worse
than the uniform mesh's, both solved on the product's own pipeline and scored
against the SAME reference. UNMEASURED is its own outcome (the uniform solve
raised, or the reference could not be built) and is never a pass -- the Orin's
n3 uniform reference OOMed once at 199,062 elements, which is exactly the
"could not ask" shape this rig treats as its signature defect.

THE REFERENCE, per case, is one of two things and the case says which:

* ``analytic`` for concentric two-index spheres -- the closed-form spectrum from
  ``sphere_dielectric``, the same reference the other bench scores against.
* ``uniform-refined`` where no closed form exists (a box, an off-centre
  inclusion, three distinct indices). The reference is the SAME geometry solved
  on a uniform mesh at ``h_min/K`` for K in ``REFINE_FACTORS``; the coarser
  level is scored against the finest, so the reference's OWN convergence is
  printed beside every error it is used to judge. A reference whose convergence
  is not shown cannot support a no-worse-than claim.

WHAT THIS FILE DOES NOT DO. It does not change the grading rule; it measures it.
And nothing here is a constant fitted to a case: the per-region edges come from
``mse_common.region_plan`` and the predicted counts from
``web.meshing.estimate_tets``, both production functions.

USAGE
    python3 benchmarks/run_grading_matrix.py --list
    python3 benchmarks/run_grading_matrix.py --case c3-thick
    python3 benchmarks/run_grading_matrix.py --case box-sphere --bands

EXIT CODES are the other bench's: 0 every criterion measured and passed, 2 bad
arguments, 3 a criterion FAILED, 4 a criterion could not be measured.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

_BENCH_DIR = Path(__file__).resolve().parent
_SOLVER_DIR = _BENCH_DIR.parent / "dockerfiles" / "maxwell_solver"
_WEB_DIR = _BENCH_DIR.parent / "web"
for _d in (_BENCH_DIR, _SOLVER_DIR, _WEB_DIR):
    if str(_d) not in sys.path:
        sys.path.insert(0, str(_d))

import run_multimaterial_bench as BENCH  # noqa: E402
import sphere_dielectric as REF  # noqa: E402
from mse_common import region_plan as common_region_plan  # noqa: E402

#: Same exit vocabulary as ``run_multimaterial_bench``: a criterion that could
#: not be measured is NOT the same as one that passed.
MATRIX_OK = 0
MATRIX_USAGE = 2
MATRIX_FAIL = 3
MATRIX_INCOMPLETE = 4

#: The tier-2 budget: one solve at a time on the Orin, under the 5 GB limit and
#: 30 minutes. The element mark is empirical -- the n3 uniform reference at
#: 199,062 elements OOMed once and solved once -- and is used to WARN at
#: ``--list`` time, not to refuse: a case whose uniform reference cannot be
#: solved is a case the criterion reports as UNMEASURED, which is the outcome
#: the row asks to be made visible.
TIER2_MAX_TETS = 200_000

#: The uniform reference's refinement factors. The finest is the reference the
#: errors are scored against; the coarser levels measure the reference's own
#: convergence, and K = 1.5 costs a third of what K = 2 costs.
REFINE_FACTORS = (1.5, 2.0)


# ---------------------------------------------------------------- the fixtures
#
# Each builder returns ``(named_meshes, layers, scale_mm)``:
#   named_meshes -- (name, trimesh) OUTERMOST FIRST, the order ``PartSet``
#                   requires (the first part contains every other)
#   layers       -- outermost first; each {"n", "kind", "centre", "r"/"half"},
#                   which assigns eps per tet by GEOMETRY -- the nesting rule a
#                   dielectric follows -- rather than by a region id the mesher
#                   happened to keep
#   scale_mm     -- the length ``kb`` is normalised by for this case
#
# ``target_edge`` is the whole-object edge, so no part's SURFACE is finer than
# the interior it bounds (the bench's surface-floors-the-mesh trap).


def _mesh_lines(parts) -> bytes:
    """OBJ bytes in mm from ``(name, trimesh)`` pairs.

    The writer rule of ``run_multimaterial_bench._parts_obj`` -- one ``o`` line
    per part, face indices offset by the vertices written before them --
    generalised to any mesh, because the shape axis needs a box and a
    translated sphere. Sphere parts here still go through the bench's own
    faceting rule (``_facet_subdivisions``), so a sphere part is identical
    whichever builder wrote it.
    """
    lines: list[str] = []
    n_written = 0
    for name, mesh in parts:
        verts = np.asarray(mesh.vertices, float)
        faces = np.asarray(mesh.faces, np.int64)
        lines.append(f"o {name}")
        lines.extend(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}" for v in verts)
        lines.extend(f"f {f[0] + 1 + n_written} {f[1] + 1 + n_written} "
                     f"{f[2] + 1 + n_written}" for f in faces)
        n_written += len(verts)
    return ("\n".join(lines) + "\n").encode("utf-8")


def _sphere(name, radius, target_edge, centre=(0.0, 0.0, 0.0)):
    import trimesh
    s = BENCH._facet_subdivisions(radius, target_edge)
    mesh = trimesh.creation.icosphere(subdivisions=s, radius=radius)
    if any(centre):
        mesh = mesh.copy()
        mesh.apply_translation(centre)
    return (name, mesh)


def _box(name, half_extents):
    import trimesh
    return (name, trimesh.creation.box(extents=[2.0 * h for h in half_extents]))


def _concentric(radii, materials, target_edge):
    order = sorted(materials, key=lambda nm: radii[nm], reverse=True)
    meshes = [_sphere(nm, radii[nm], target_edge) for nm in order]
    layers = [{"n": float(materials[nm]), "kind": "sphere",
               "centre": (0.0, 0.0, 0.0), "r": float(radii[nm])} for nm in order]
    return meshes, layers, float(radii[order[0]])


def _offset(b_mm, a_mm, offset_mm, materials, target_edge):
    meshes = [_sphere("wall", b_mm, target_edge),
              _sphere("core", a_mm, target_edge, centre=(0.0, 0.0, offset_mm))]
    layers = [{"n": float(materials["wall"]), "kind": "sphere",
               "centre": (0.0, 0.0, 0.0), "r": float(b_mm)},
              {"n": float(materials["core"]), "kind": "sphere",
               "centre": (0.0, 0.0, float(offset_mm)), "r": float(a_mm)}]
    return meshes, layers, float(b_mm)


def _box_sphere(half_extents, a_mm, materials, target_edge):
    meshes = [_box("wall", half_extents), _sphere("core", a_mm, target_edge)]
    layers = [{"n": float(materials["wall"]), "kind": "box",
               "centre": (0.0, 0.0, 0.0),
               "half": tuple(float(h) for h in half_extents)},
              {"n": float(materials["core"]), "kind": "sphere",
               "centre": (0.0, 0.0, 0.0), "r": float(a_mm)}]
    return meshes, layers, float(min(half_extents))


# --------------------------------------------------------------- the case list

#: The other bench's outer radius, so ``c3-thick`` is that bench's n3 fixture.
B_MM = 10.0
A_THICK = B_MM / 3.0     # wall 6.67 mm thick: the clamp cannot bite
A_THIN = 0.9 * B_MM      # wall 1.0 mm thick: the clamp bites from n = 2 up


def _case(label, build, materials, reference, axes, n_modes=10, note=""):
    return {"label": label, "build": build, "materials": materials,
            "reference": reference, "axes": axes, "n_modes": n_modes,
            "note": note}


MATRIX = [
    _case("c1.5-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 1.5}, te),
          {"wall": 1.0, "core": 1.5}, "analytic",
          {"contrast": 1.5, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          note="the lowest contrast the feature is meant to help at"),
    _case("c2-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 2.0}, te),
          {"wall": 1.0, "core": 2.0}, "analytic",
          {"contrast": 2.0, "shell": "thick", "regions": 2,
           "shape": "concentric"}),
    _case("c3-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "analytic",
          {"contrast": 3.0, "shell": "thick", "regions": 2,
           "shape": "concentric"},
          note="the other bench's n3 fixture, reproduced here"),
    _case("c4-thick",
          lambda te: _concentric({"wall": B_MM, "core": A_THICK},
                                 {"wall": 1.0, "core": 4.0}, te),
          {"wall": 1.0, "core": 4.0}, "analytic",
          {"contrast": 4.0, "shell": "thick", "regions": 2,
           "shape": "concentric"}),
    _case("c2-thin",
          lambda te: _concentric({"wall": B_MM, "core": A_THIN},
                                 {"wall": 1.0, "core": 2.0}, te),
          {"wall": 1.0, "core": 2.0}, "analytic",
          {"contrast": 2.0, "shell": "thin", "regions": 2,
           "shape": "concentric"},
          note="wall 1.0 mm thick vs the 1.19 mm the rule asks: clamp bites"),
    _case("c3-thin",
          lambda te: _concentric({"wall": B_MM, "core": A_THIN},
                                 {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "analytic",
          {"contrast": 3.0, "shell": "thin", "regions": 2,
           "shape": "concentric"}),
    _case("c4-thin",
          lambda te: _concentric({"wall": B_MM, "core": A_THIN},
                                 {"wall": 1.0, "core": 4.0}, te),
          {"wall": 1.0, "core": 4.0}, "analytic",
          {"contrast": 4.0, "shell": "thin", "regions": 2,
           "shape": "concentric"}),
    _case("three-region",
          lambda te: _concentric({"wall": B_MM, "mid": 0.6 * B_MM,
                                  "core": 0.25 * B_MM},
                                 {"wall": 1.0, "mid": 2.0, "core": 4.0}, te),
          {"wall": 1.0, "mid": 2.0, "core": 4.0}, "uniform-refined",
          {"contrast": 4.0, "shell": "thick", "regions": 3,
           "shape": "concentric"},
          n_modes=5, note="three distinct indices: no closed-form spectrum"),
    _case("box-sphere",
          lambda te: _box_sphere((5.0, 3.5, 2.5), 1.6,
                                 {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "uniform-refined",
          {"contrast": 3.0, "shell": "thick", "regions": 2, "shape": "box"},
          n_modes=5, note="non-spherical cavity: a 10x7x5 box"),
    _case("offset-sphere",
          lambda te: _offset(B_MM, 2.5, 4.0, {"wall": 1.0, "core": 3.0}, te),
          {"wall": 1.0, "core": 3.0}, "uniform-refined",
          {"contrast": 3.0, "shell": "thick", "regions": 2, "shape": "offset"},
          n_modes=5, note="non-concentric: the core is 4 mm off centre"),
]


def _coverage(cases=None) -> dict:
    """Which value each axis has a case for. Read by a test."""
    cov: dict[str, set] = {k: set() for k in
                           ("contrast", "shell", "regions", "shape", "reference")}
    for c in (cases if cases is not None else MATRIX):
        for k in ("contrast", "shell", "regions", "shape"):
            cov[k].add(c["axes"][k])
        cov["reference"].add(c["reference"])
    return cov


# ------------------------------------------------------------------- geometry


def _contains(layer, p) -> bool:
    cx, cy, cz = layer["centre"]
    if layer["kind"] == "sphere":
        return ((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2
                <= layer["r"] ** 2)
    hx, hy, hz = layer["half"]
    return (abs(p[0] - cx) <= hx and abs(p[1] - cy) <= hy
            and abs(p[2] - cz) <= hz)


def _eps_by_layers(points_m, tets, layers) -> np.ndarray:
    """``eps_r`` per tet: the innermost layer containing its centroid."""
    cent = np.asarray(points_m, float)[np.asarray(tets, np.int64)].mean(axis=1)
    cent = cent * 1e3                                    # mm, like ``layers``
    eps = np.ones(cent.shape[0], float)
    for layer in layers:                    # outermost first: inner ones win
        eps[np.array([_contains(layer, p) for p in cent])] = layer["n"] ** 2
    return eps


def _outer_surface(parts):
    return parts.parts[parts.wall_index if parts.wall_index is not None else 0]


# ------------------------------------------------------------------ the runner


def _build(case, n_modes: int):
    """``(obj, parts, plan, h_min_mm, layers, scale_mm)`` for one case.

    The per-region edges come from ``mse_common.region_plan`` -- the production
    function the mesher and the web estimate BOTH call -- so the matrix cannot
    disagree with the product about what edge a region was asked for.
    """
    import trimesh
    import mse_meshing as MESH
    import mse_maxwell_solver as SOLVER

    probe = trimesh.creation.icosphere(subdivisions=2, radius=B_MM)
    target = float(MESH.target_edge_length(probe, n_modes=n_modes, units="mm",
                                           n_index=1.0))
    meshes, layers, scale_mm = case["build"](target)
    obj = _mesh_lines(meshes)
    parts = SOLVER.PARTS.load_parts(obj, units="mm")
    plan = common_region_plan(parts, case["materials"], target * 1e-3, "mm")
    h_min = min(i["edge_m"] for i in plan.values()) * 1e3
    return obj, parts, plan, h_min, layers, scale_mm


def _kb(freqs, scale_mm):
    """``kb = 2 pi f L / c`` with L the case's own scale, in metres."""
    return 2.0 * np.pi * np.asarray(freqs, float) * (scale_mm * 1e-3) / 299_792_458.0


def _graded(case, obj, work, *, grid, backend, verbose):
    """The GRADED run, through the product's own entry point."""
    import mse_maxwell_solver as SOLVER

    store_dir = work / "store"
    store_dir.mkdir(parents=True, exist_ok=True)
    store = SOLVER.STORE.LocalStore(str(store_dir))
    job_id = f"matrix-{case['label']}"
    store.write(SOLVER.STORE.job_key(job_id, "input.obj"), obj)
    t0 = time.time()
    summary = SOLVER.solve(job_id, "mm", 1.0, case["n_modes"], store, store,
                           max_tets=600_000, grid=grid, verbose=verbose,
                           backend=backend, materials=case["materials"])
    modes = json.loads(store.read(SOLVER.STORE.job_key(job_id, "modes.json")))
    freqs = np.array([m["freq_hz"] for m in modes["modes"]], float)
    return summary, modes, freqs, time.time() - t0


def _uniform(materials, obj, case, h_min_mm, *, k: float, layers, backend,
             np_verbose=False):
    """One uniform mesh+solve at ``h_min/k``, the same bytes.

    Mirrors ``run_multimaterial_bench._run_case``'s reference half: the mesh is
    ``tetrahedralize`` with ``min_feature_m`` forcing the fine edge onto a
    one-part mesh (``materials=None``), and the spectrum comes from
    ``_solve_mesh`` -- the product's own five solver calls -- with eps supplied
    per tet by geometry, because a uniform mesh carries no regions.
    """
    import mse_meshing as MESH
    import mse_maxwell_solver as SOLVER

    parts = SOLVER.PARTS.load_parts(obj, units="mm")
    merged = SOLVER.PARTS.merged_surface(parts)
    points, tets, _region = MESH.tetrahedralize(
        merged, "mm", n_modes=case["n_modes"], n_index=1.0, max_tets=600_000,
        min_feature_m=(h_min_mm / k) * 1e-3, verbose=np_verbose,
        materials=None, parts=None)
    eps = _eps_by_layers(points, tets, layers)
    kb = BENCH._solve_mesh(points, tets, eps, case["n_modes"], backend)
    return points, tets, kb


def _clusters_pairs(kb):
    """A measured spectrum in ``score_multiplets``' reference shape.

    ``l`` and ``family`` are report labels: this reference carries no
    degeneracy labels, so the matcher is used for its clustering and one-to-one
    matching, not for its physics.
    """
    return [(float(c), 0, "uniform-ref", int(n))
            for c, n in REF.cluster_modes(np.asarray(kb, float), 0.02)]


def criterion(graded_worst, uniform_worst, unmeasured_reason):
    """The owner's bar as one pure function of the two numbers.

    Returns ``(verdict, detail)``, verdict in PASS / FAIL / UNMEASURED. An
    unmeasured bar never returns PASS: a mesh whose reference could not be
    solved is not a mesh that beat it.
    """
    if unmeasured_reason:
        return "UNMEASURED", unmeasured_reason
    if graded_worst is None or uniform_worst is None:
        return "UNMEASURED", "one of the two worst errors is missing"
    ratio = (graded_worst / uniform_worst) if uniform_worst else float("inf")
    verdict = "PASS" if graded_worst <= uniform_worst else "FAIL"
    return verdict, (f"graded {graded_worst * 100:.3f}% vs uniform "
                     f"{uniform_worst * 100:.3f}% ({ratio:.2f}x the uniform mesh)")


def run_case(case, *, grid, backend, work: Path, verbose, bands=False) -> dict:
    """Build, mesh and solve ONE matrix case, and judge it."""
    import mse_maxwell_solver as SOLVER

    n_modes = case["n_modes"]
    out: dict = {"label": case["label"], "n_modes": n_modes,
                 "reference_kind": case["reference"], "axes": case["axes"],
                 "note": case.get("note", "")}
    try:
        obj, parts, plan, h_min, layers, scale_mm = _build(case, n_modes)
    except Exception as exc:  # noqa: BLE001 - a fixture the producer rejects
        out["build_error"] = f"{type(exc).__name__}: {exc}"
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": f"the fixture could not be built: {out['build_error']}",
                "case": out}
    out.update(h_min_mm=h_min, scale_mm=scale_mm,
               plan={str(k): {kk: round(vv, 9) if isinstance(vv, float) else vv
                              for kk, vv in v.items()}
                     for k, v in plan.items()})

    try:
        summary, modes, freqs, elapsed = _graded(case, obj, work, grid=grid,
                                                 backend=backend,
                                                 verbose=verbose)
        kb_graded = _kb(freqs, scale_mm)
        out["n_tets_graded"] = int(summary["n_tets"])
        out["elapsed_s_graded"] = round(elapsed, 1)
        out["graded_regions"] = ((modes.get("mesh") or {}).get("regions") or {})
    except Exception as exc:  # noqa: BLE001 - a failed graded solve is data
        out["graded_error"] = f"{type(exc).__name__}: {exc}"
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": f"the GRADED solve raised: {out['graded_error']}",
                "case": out}

    # The uniform reference's spectra, measured, one level per factor.
    levels: dict[float, np.ndarray] = {}
    uniform_error = None
    out["uniform"] = {}
    for k in ({1.0} | set(REFINE_FACTORS)):
        try:
            _p, tets_u, kb_u = _uniform(case["materials"], obj, case, h_min,
                                        k=k, layers=layers, backend=backend)
            levels[k] = kb_u
            out["uniform"][str(k)] = {"n_tets": int(tets_u.shape[0])}
        except Exception as exc:  # noqa: BLE001 - a failed reference is data
            uniform_error = f"at h_min/{k}: {type(exc).__name__}: {exc}"
            out["uniform"][str(k)] = {"error": f"{type(exc).__name__}: {exc}"}
    if 1.0 not in levels:
        return {"label": case["label"], "verdict": "UNMEASURED",
                "detail": f"the uniform solve raised: {uniform_error}",
                "case": out}

    # THE REFERENCE.
    meta: dict = {"reference_levels": {k: v for k, v in out["uniform"].items()}}
    if case["reference"] == "analytic":
        a_eff, _b_eff, ab_eff = BENCH._effective_radii(
            parts, case["materials"].get("core"), units="mm")
        _n, spectrum = BENCH._analytic(case["materials"].get("core"), ab_eff)
        reference = [(float(k), int(l), f, int(m)) for k, l, f, m in spectrum]
        out["a_over_b_eff"] = ab_eff
    else:
        finest = REFINE_FACTORS[-1]
        if finest not in levels:
            return {"label": case["label"], "verdict": "UNMEASURED",
                    "detail": f"the reference could not be built: {uniform_error}",
                    "case": out, "meta": meta}
        reference = _clusters_pairs(levels[finest])
        coarser = REFINE_FACTORS[0]
        if coarser in levels:
            conv = REF.score_multiplets(levels[coarser], reference,
                                        n_requested=n_modes)
            meta["reference_convergence"] = conv.get("worst_rel_error")

    graded = REF.score_multiplets(kb_graded, reference, n_requested=n_modes)
    uniform = REF.score_multiplets(levels[1.0], reference, n_requested=n_modes)
    out["graded_score"] = {k: graded[k] for k in
                           ("n_matched", "worst_rel_error") if k in graded}
    out["uniform_score"] = {k: uniform[k] for k in
                            ("n_matched", "worst_rel_error") if k in uniform}

    if bands and case["axes"]["shape"] == "concentric":
        try:
            import mse_meshing as MESH
            merged = SOLVER.PARTS.merged_surface(parts)
            gp, gt, gr = MESH.tetrahedralize(
                merged, "mm", n_modes=n_modes, n_index=1.0, max_tets=600_000,
                min_feature_m=None, verbose=False, materials=case["materials"],
                parts=parts)
            fine_region = int(((modes.get("mesh") or {}).get("fine_region")))
            a_eff, _b, _ab = BENCH._effective_radii(
                parts, case["materials"].get("core"), units="mm")
            out["bands"] = BENCH._band_edges_mm(gp, gt, float(a_eff),
                                                region_of_tet=gr,
                                                fine_region=fine_region)
        except Exception as exc:  # noqa: BLE001
            out["bands_error"] = f"{type(exc).__name__}: {exc}"

    verdict, detail = criterion(graded.get("worst_rel_error"),
                                uniform.get("worst_rel_error"), None)
    return {"label": case["label"], "verdict": verdict, "detail": detail,
            "graded_worst": graded.get("worst_rel_error"),
            "uniform_worst": uniform.get("worst_rel_error"),
            "meta": meta, "case": out}


# ---------------------------------------------------------------------- --list


def predicted_counts(case) -> dict:
    """Predicted element count for every mesh a case would build.

    The numbers are the product's own estimator (``web.meshing.estimate_tets``,
    the function that ROUTES an upload), not a re-derivation, so a case that
    would route to the GPU here is the case that would route there in
    production. The refined levels are that estimate scaled by ``k**3`` -- the
    exact factor when the only thing changing is the edge.
    """
    import meshing as WEBMESH

    _obj, parts, plan, h_min, _layers, _scale = _build(case, case["n_modes"])
    outer = _outer_surface(parts)
    counts = {
        "graded": int(WEBMESH.estimate_tets(outer, case["n_modes"], "mm", c=14.0,
                                            parts=parts,
                                            materials=case["materials"])),
        "uniform": int(WEBMESH.estimate_tets(outer, case["n_modes"], "mm", c=14.0,
                                             min_feature_m=h_min * 1e-3)),
    }
    if case["reference"] == "uniform-refined":
        for k in REFINE_FACTORS:
            counts[f"ref{k}"] = int(round(counts["uniform"] * k ** 3))
    return {"plan": plan, "h_min_mm": h_min, "counts": counts,
            "over_ceiling": sorted(k for k, n in counts.items()
                                   if n > TIER2_MAX_TETS)}


def list_cases(cases, as_json=False) -> int:
    rows, incomplete = [], 0
    for case in cases:
        try:
            pred = predicted_counts(case)
        except Exception as exc:  # noqa: BLE001 - a failed estimate is data
            pred = {"error": f"{type(exc).__name__}: {exc}"}
            incomplete += 1
        rows.append({"label": case["label"], "reference": case["reference"],
                     "n_modes": case["n_modes"], "axes": case["axes"],
                     "note": case.get("note", ""), **pred})
    if as_json:
        print(json.dumps(rows, indent=2))
        return MATRIX_INCOMPLETE if incomplete else MATRIX_OK
    print(f"{'case':>13} {'ref':>6} {'reg':>4} {'h_min':>7} {'graded':>9} "
          f"{'uniform':>9}  axes")
    for r in rows:
        if "error" in r:
            print(f"{r['label']:>13} {r['reference']:>6}   --- COUNT FAILED: "
                  f"{r['error']}")
            continue
        c = r["counts"]
        extra = ("" if r["reference"] == "analytic"
                 else f"  refs {[c[f'ref{k}'] for k in REFINE_FACTORS]}")
        print(f"{r['label']:>13} {r['reference']:>6} {r['axes']['regions']:>4} "
              f"{r['h_min_mm']:>7.4f} {c['graded']:>9,} {c['uniform']:>9,}  "
              f"{r['axes']['shape']}, {r['axes']['shell']}, "
              f"n={r['axes']['contrast']}{extra}")
        if r["over_ceiling"]:
            print(f"{'':>13}   ! over the tier-2 mark {TIER2_MAX_TETS:,}: "
                  f"{r['over_ceiling']} -- expect UNMEASURED, not a pass")
    print("\naxes covered: "
          + "; ".join(f"{k}={sorted(v, key=str)}" for k, v in
                      _coverage(cases).items()))
    return MATRIX_INCOMPLETE if incomplete else MATRIX_OK


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true",
                   help="print the case list with predicted element counts")
    p.add_argument("--case", default=None, help="run ONE case by label")
    p.add_argument("--grid", type=int, default=40)
    p.add_argument("--backend", default="gpu", choices=("cpu", "gpu"))
    p.add_argument("--json", default=None, help="write the case result here")
    p.add_argument("--work", default=None)
    p.add_argument("--bands", action="store_true",
                   help="also report the per-band achieved edges (concentric "
                        "cases; costs one extra mesh build)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    if args.list:
        return list_cases(MATRIX)
    labels = [c["label"] for c in MATRIX]
    if not args.case:
        print("nothing to do: pass --list or --case <label>", file=sys.stderr)
        return MATRIX_USAGE
    if args.case not in labels:
        print(f"unknown case {args.case!r}; known: {labels}", file=sys.stderr)
        return MATRIX_USAGE
    case = next(c for c in MATRIX if c["label"] == args.case)
    work = (Path(args.work) if args.work
            else Path(tempfile.mkdtemp(prefix=f"mse-matrix-{args.case}-")))
    print(f"[matrix] work root: {work}")
    print(f"[matrix] case {case['label']}: {case['note'] or case['axes']}")
    try:
        result = run_case(case, grid=args.grid, backend=args.backend, work=work,
                          verbose=not args.quiet, bands=args.bands)
    except Exception as exc:  # noqa: BLE001
        traceback.print_exc()
        print(f"[matrix] case RAISED: {type(exc).__name__}: {exc}")
        return MATRIX_INCOMPLETE
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=2))
    print(f"[matrix] {result['verdict']}: {result['detail']}")
    out = result.get("case") or {}
    if out.get("n_tets_graded"):
        us = out.get("uniform", {}).get("1.0", {}) or {}
        print(f"[matrix] graded {out['n_tets_graded']:,} tets; uniform "
              f"{us.get('n_tets') or 'unmeasured'} tets")
        for rid, reg in (out.get("graded_regions") or {}).items():
            edge = reg.get("achieved_edge_units")
            print(f"[matrix]   graded region {rid}: built "
                  f"{edge if edge is None else round(edge, 4)} units")
    if result.get("meta", {}).get("reference_convergence") is not None:
        print("[matrix] reference's own convergence (coarse vs finest): "
              f"{result['meta']['reference_convergence'] * 100:.3f}%")
    if out.get("bands"):
        print("[matrix] bands: " + "  ".join(
            f"{k} {v['edge_mm']:.4f} mm" for k, v in out["bands"].items()))
    return {"PASS": MATRIX_OK, "FAIL": MATRIX_FAIL,
            "UNMEASURED": MATRIX_INCOMPLETE}[result["verdict"]]


if __name__ == "__main__":
    raise SystemExit(main())
