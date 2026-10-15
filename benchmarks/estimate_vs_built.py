"""The upload-time tet estimate next to what TetGen actually builds.

Claim under test: ``web/app.py`` routes every upload (cluster / Jetson / refuse)
on an element count computed BEFORE any mesh exists, and that estimate runs high
against the mesh the app really builds.  This script puts the product's own
estimate next to the product's own mesh, and prints the quantity that explains
the ratio.

The estimate is the VOLUME LAW times a measured margin:

    est = margin * 6*sqrt(2) * sum_r filled_V_r / h_r^3

with ``margin = web.config.est_tets_margin()`` (default ``EST_TETS_MARGIN``),
``h_r`` the edge the region plan gives each region (the edge the mesher hands
``target_maxvolume``) and ``V_r`` the region's FILLED volume -- never its
enclosed volume, which double-counts everything nested in it. The law is exact:
a tetrahedralisation whose elements all sat at edge ``h`` holds

    6*sqrt(2) = 8.485  tets per h^3

because a regular tetrahedron of edge ``h`` has volume ``h^3 / (6*sqrt(2))``.
The mesher instead hands TetGen ``maxvolume = TETGEN_VOLUME_SLACK * h^3 /
(6*sqrt(2))`` (slack 4.0), which permits elements up to ~ 4^(1/3) = 1.59 h, and
TetGen's ``minratio``/``mindihedral`` produce "somewhat larger" elements than
that bound (``target_maxvolume``'s own words).  The ``built/volaw`` column is
exactly that shortfall: ``1.000`` would mean the mesh landed on the volume law,
and ``est/built`` is ``margin / built/volaw``.

Every row also carries ``old/built``: the estimate the FORMULA THIS CHANGE
REPLACES would have given, ``C * V / h^3`` with ``C = 14.0``, recomputed here
because the constant is no longer in the product.  It is the "before" side of
the comparison the grading re-run needs, and it is recomputed from the same
recorded edge and volume as the "after" column so the two differ by the formula
and by nothing else.

Three modes:

  tier 1, tiny fixtures, real mesher (one-part and nested paths):
      python benchmarks/estimate_vs_built.py --build

  tier 1, the SHAPE FAMILIES the margin has to hold on, real mesher:
      python benchmarks/estimate_vs_built.py --families

  no compute at all, reading recorded run JSONs:
      python benchmarks/estimate_vs_built.py --results DIR [DIR ...]

The recorded-JSON mode never meshes anything: it reads the per-region plan a
recorded run wrote and the count that run built. That plan is the GRADING
BENCH's own (keys ``edge_m``, ``unclamped_m``, ``thickness_m``), whose
per-region clamp scheme is not main's ``mse_common.region_edges`` -- main's own
rule narrows every region to ``min(h, h*n_wall/n_max)`` and would give a finer
h_min, hence a larger estimate, on the same upload. So these rows measure the
lead's recorded runs against the plan they recorded, not main's live estimate;
the live estimate is measured by ``--build``/``--families``, which call the
product's own ``web.meshing.estimate_tets``.

Dev-only.  Uses the app's own estimator and the real mesher; no network.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOLVER = ROOT / "dockerfiles" / "maxwell_solver"
for _p in (str(ROOT), str(SOLVER)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import mse_common as COMMON  # noqa: E402
import mse_meshing as MESHSOLVER  # noqa: E402
import mse_parts as PARTS  # noqa: E402
from web import config as CFG  # noqa: E402
from web import meshing as WEB  # noqa: E402

TETS_PER_H3 = COMMON.TETS_PER_UNIT_VOLUME  # the law, read from the mesher's own
N_MODES = 10      # tier 1: a 6 mm box at n=10 is ~6k tets, seconds
MULTI_N_MODES = 4  # a n=2 core multiplies the count by 8; keep the fixture tiny
#: The constant this change REMOVES (``web.config.EST_TETS_C``), kept here only
#: to compute the "before" column. It is not read from the product: it is not in
#: the product any more, which is the point of the change.
OLD_C = 14.0

#: The cap the bench hands the mesher, read from the PRODUCT rather than typed
#: here: ``web.app`` passes ``CFG.max_tets()`` to the solver and the solver
#: passes it to ``tetrahedralize``, so this is the number a real upload gets.
#: A smaller literal is not a smaller fixture -- the mesher's over-cap retry
#: COARSENS ``maxvolume`` and rebuilds, so a bench cap below a family's
#: unconstrained count silently measures a coarser mesh than the product would
#: make (the 2026-10-15 finding: at 60,000 two families came back ~50x smaller
#: than the same fixtures at the product's cap). ``--max-tets`` overrides it, and
#: step 3 (the Orin) raises every family's n_modes with ``--family-scale-n``
#: (the count goes like n_modes^3, so 2 is ~8x).
MAX_TETS = CFG.max_tets()
FAMILY_N_SCALE = 1.0
FACES = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
         (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
ROWS = []


def _box_obj(parts) -> bytes:
    """OBJ bytes for nested axis-aligned boxes; ``parts`` = [(name, side_mm)].

    ``f`` indices are 1-based over the file's ``v`` lines, NOT over file lines:
    counting lines (the ``o``/``f`` ones included) indexes a second box into the
    first one's vertices and trimesh raises IndexError.
    """
    lines, vertices = [], 0
    for name, side in parts:
        corners = [(sx * side / 2, sy * side / 2, sz * side / 2)
                   for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        lines.append(f"o {name}")
        base = vertices  # 0-based count of vertices already written
        for (vx, vy, vz) in corners:
            lines.append(f"v {vx!r} {vy!r} {vz!r}")
        vertices += len(corners)
        for (a, b, d) in FACES:
            lines.append(f"f {base + a + 1} {base + b + 1} {base + d + 1}")
    return ("\n".join(lines) + "\n").encode()


def _row(label, h_m, volume_m3, est, built, achieved_m=None, algebra=None,
         est_old=None, n_modes=None, family=None):
    volaw = TETS_PER_H3 * volume_m3 / h_m ** 3
    text = (f"{label:<26} h {h_m * 1e3:7.4f} mm  V {volume_m3 * 1e9:9.2f} mm3  "
            f"est {est:8d}  built {built:8d}  est/built {est / built:5.2f}  "
            f"built/volaw {built / volaw:5.3f}")
    if est_old is not None:
        text += f"  old/built {est_old / built:5.2f}"
    if achieved_m:
        text += f"  achieved/h {achieved_m / h_m:5.3f}"
    if algebra is not None and algebra != est:
        text += f"  [plan algebra gives {algebra}]"
    print(text)
    ROWS.append({"label": label, "h_m": h_m, "volume_m3": volume_m3, "est": est,
                 "built": built, "ratio": est / built, "volaw": built / volaw,
                 "old_ratio": (est_old / built) if est_old else None,
                 "n_modes": n_modes, "family": family,
                 "achieved_h": (achieved_m / h_m) if achieved_m else None})


def build_fixtures(size_mm: float, core_ratio: float, core_index: float, n_modes: int):
    """One tiny mesh per upload path, through the product's own calls.

    Both branches mirror the solver's own entry point
    (``mse_maxwell_solver.solve``): a multi-part upload goes through
    ``PARTS.load_parts`` + ``PARTS.merged_surface``, a one-part upload through
    ``MESHSOLVER.load_surface``.  n_modes is small so the mesh stays tier-1 tiny.
    """
    # (1) single part: the legacy ``min_feature_m=`` path.
    surface, _ = MESHSOLVER.load_surface(_box_obj([("1 body", size_mm)]), units="mm")
    feature_m = size_mm * 1e-3 * 0.5
    h_m = COMMON.effective_edge_m(surface, n_modes=n_modes, units="mm",
                                  min_feature_m=feature_m)
    est = WEB.estimate_tets(surface, n_modes, "mm", margin=CFG.est_tets_margin(),
                            min_feature_m=feature_m)
    _points, tets, _regions = MESHSOLVER.tetrahedralize(
        surface, "mm", n_modes=n_modes, max_tets=MAX_TETS, verbose=False,
        min_feature_m=feature_m)
    _row(f"one-part {size_mm:g}mm", h_m, COMMON.enclosure_volume_m3(surface, "mm"),
         est, int(tets.shape[0]), MESHSOLVER.LAST_ACHIEVED_EDGE_M,
         algebra=int(CFG.est_tets_margin() * TETS_PER_H3
                     * (COMMON.enclosure_volume_m3(surface, "mm") / h_m ** 3)),
         est_old=int(OLD_C * COMMON.enclosure_volume_m3(surface, "mm") / h_m ** 3),
         n_modes=n_modes, family="--build")

    # (2) nested two-part upload: the live ``materials=`` path.
    obj = _box_obj([("1 wall", size_mm), ("2 core", size_mm * core_ratio)])
    parts = PARTS.load_parts(obj, units="mm")
    materials = {parts.names[0]: 1.0, parts.names[1]: core_index}
    surface = PARTS.merged_surface(parts)
    # The wall's edge is what goes INTO the plan; the plan narrows it to h_min
    # (``region_plan`` floors ``region_edges``' h_r at the wall's edge, i.e.
    # min(h, h*n_wall/n_max)). The estimator and the mesher
    # both mesh at h_min, so h_min is the h this row must report -- reporting
    # the wall edge here would print an estimate 1.65x low and hide the effect
    # the row exists to show.
    wall_edge_m = COMMON.effective_edge_m(surface, n_modes=n_modes, units="mm",
                                          parts=parts, materials=materials)
    est = WEB.estimate_tets(surface, n_modes, "mm", margin=CFG.est_tets_margin(),
                            parts=parts, materials=materials)
    plan = COMMON.region_plan(parts, materials, wall_edge_m, "mm")
    h_m = min(float(info["edge_units"]) for info in plan.values())
    volume_m3 = sum(float(info["filled_volume_m3"]) for info in plan.values())
    _points, tets, _regions = MESHSOLVER.tetrahedralize(
        surface, "mm", n_modes=n_modes, parts=parts, materials=materials,
        max_tets=MAX_TETS, verbose=False)
    _row(f"nested {size_mm:g}mm core {core_ratio:g} n{core_index:g}", h_m, volume_m3,
         est, int(tets.shape[0]), MESHSOLVER.LAST_ACHIEVED_EDGE_M,
         algebra=int(CFG.est_tets_margin() * TETS_PER_H3 * (volume_m3 / h_m ** 3)),
         est_old=int(OLD_C * volume_m3 / h_m ** 3),
         n_modes=n_modes, family="--build")


def _box(extents):
    import trimesh
    return trimesh.creation.box(extents=extents)


def _sphere(radius, subdivisions=3, bump=0.0):
    import numpy as np
    import trimesh

    mesh = trimesh.creation.icosphere(subdivisions=subdivisions, radius=radius)
    if bump:
        # A perturbed sphere: curvature the volume law cannot see, which is the
        # "a fine or curved surface floors the interior" case the old fitted
        # constant claimed to absorb. The radial factor is one number per
        # VERTEX, so it multiplies along axis 1 (3 coordinates) and not the
        # vertex axis -- as an (n,) array it broadcasts against 3 and raises.
        v = mesh.vertices
        factor = 1.0 + bump * np.sin(3.0 * v[:, 0]) * np.cos(2.0 * v[:, 1])
        mesh.vertices = v * factor[:, None]
    return mesh


def _obj_bytes(name, mesh_or_scene):
    """OBJ bytes with named objects -- the upload's own multi-part convention.

    ``trimesh.Scene`` writes one ``o <name>`` per geometry, which is what the
    app's part-index parsing reads, so a fixture built here exercises the same
    loader as an upload rather than a hand-built surface.
    """
    import trimesh

    scene = (mesh_or_scene if hasattr(mesh_or_scene, "geometry")
             else trimesh.Scene({name: mesh_or_scene}))
    return scene.export(file_type="obj").encode()


#: The shape families the margin has to hold on -- the lead's list (single part,
#: 2-3 regions, thin and thick shells, off-centre, non-spherical), each built
#: through the product's own loaders and the real mesher. ``parts`` maps the
#: per-region index n to (part name, builder); the wall is always n=1.
FAMILIES = [
    ("one-part 6mm cube", {1.0: ("1 body", lambda: _box((6, 6, 6)))}, N_MODES),
    ("one-part plate 8x4x2", {1.0: ("1 body", lambda: _box((8, 4, 2)))}, N_MODES),
    ("one-part sphere r3", {1.0: ("1 body", lambda: _sphere(3.0))}, N_MODES),
    ("one-part bumpy sphere r3",
     {1.0: ("1 body", lambda: _sphere(3.0, bump=0.15))}, N_MODES),
    # 2 regions, thick wall: a 1.5 mm core in a 6 mm box
    ("2-region core 1.5mm n=2",
     {1.0: ("wall", lambda: _box((6, 6, 6))),
      2.0: ("core", lambda: _box((1.5, 1.5, 1.5)))}, MULTI_N_MODES),
    # ... and a DENSE core: the family the mesher's guard lets through 1.86x
    # coarser than the edge the plan asks for (recorded as a found issue, not
    # fixed here), which is where an over-estimate can get large.
    ("2-region dense core n=3.5",
     {1.0: ("wall", lambda: _box((6, 6, 6))),
      3.5: ("core", lambda: _box((1.5, 1.5, 1.5)))}, MULTI_N_MODES),
    # 3 regions: shell + core
    ("3-region shell+core n=2/3",
     {1.0: ("wall", lambda: _box((6, 6, 6))),
      2.0: ("shell", lambda: _box((4, 4, 4))),
      3.0: ("core", lambda: _box((2, 2, 2)))}, MULTI_N_MODES),
    # off-centre core: not a concentric partition of the cavity
    ("2-region off-centre n=2",
     {1.0: ("wall", lambda: _box((6, 6, 6))),
      2.0: ("core", lambda: _box((2, 2, 2)).apply_translation((1.2, 0.8, -1.0)))},
     MULTI_N_MODES),
    # thin wall: 0.4 mm of shell around a 5.2 mm core
    ("2-region thin wall 0.4mm",
     {1.0: ("wall", lambda: _box((6, 6, 6))),
      2.0: ("core", lambda: _box((5.2, 5.2, 5.2)))}, MULTI_N_MODES),
]


def family_rows():
    """Every family through the product's own loader, region plan and mesher."""
    print("SHAPE FAMILIES (real TetGen, tier-1 sizes)")
    for label, parts_spec, n_modes in FAMILIES:
        n = max(1, int(round(n_modes * FAMILY_N_SCALE)))
        try:
            _family_row(label, parts_spec, n)
        except Exception as exc:  # noqa: BLE001 - a family that cannot build is a row
            print(f"{label:<28} FAILED: {type(exc).__name__}: {exc}")
            ROWS.append({"label": label, "family": label, "error": str(exc)})


def _family_row(label, parts_spec, n_modes):
    import numpy as np

    names = {idx: name for idx, (name, _) in parts_spec.items()}
    materials = {name: idx for idx, (name, _) in parts_spec.items()}
    surfaces = {names[idx]: build() for idx, (name, build) in parts_spec.items()}

    if len(surfaces) == 1:
        body = _obj_bytes(next(iter(surfaces)), next(iter(surfaces.values())))
        surface, _ = MESHSOLVER.load_surface(body, units="mm")
        est = WEB.estimate_tets(surface, n_modes, "mm", margin=CFG.est_tets_margin())
        volume_m3 = COMMON.enclosure_volume_m3(surface, "mm")
        h_m = COMMON.effective_edge_m(surface, n_modes=n_modes, units="mm")
        _points, tets, _regions = MESHSOLVER.tetrahedralize(
            surface, "mm", n_modes=n_modes, max_tets=MAX_TETS, verbose=False)
    else:
        import trimesh

        body = _obj_bytes("upload", trimesh.Scene(surfaces))
        parts = PARTS.load_parts(body, units="mm")
        surface = PARTS.merged_surface(parts)
        est = WEB.estimate_tets(surface, n_modes, "mm",
                                margin=CFG.est_tets_margin(),
                                parts=parts, materials=materials)
        edge_m = COMMON.effective_edge_m(surface, n_modes=n_modes, units="mm",
                                         parts=parts, materials=materials)
        plan = COMMON.region_plan(parts, materials, edge_m, "mm")
        h_m = min(float(info["edge_units"]) for info in plan.values())
        volume_m3 = sum(float(info["filled_volume_m3"]) for info in plan.values())
        _points, tets, _regions = MESHSOLVER.tetrahedralize(
            surface, "mm", n_modes=n_modes, parts=parts, materials=materials,
            max_tets=MAX_TETS, verbose=False)

    built = int(np.asarray(tets).shape[0])
    _row(label, h_m, volume_m3, est, built,
         achieved_m=float(MESHSOLVER.LAST_ACHIEVED_EDGE_M),
         est_old=int(OLD_C * volume_m3 / (h_m ** 3)), n_modes=n_modes, family=label)


def _summarize(rows, subject):
    """Per-batch extremes, and the two numbers the config constants derive from.

    Printed for EVERY batch on purpose: the margin is the highest
    ``built/volaw`` a family showed (below it the estimate under-reads and routes
    a big job to the CPU), and the refusal factor is the highest ``est/built``
    (above it the site refuses uploads whose real mesh fits). A table without
    those two numbers is the shape that let a two-fixture fit be called measured.
    """
    rows = [r for r in rows if "ratio" in r]
    if not rows:
        print(f"[est-vs-built] {subject}: no comparable rows")
        return
    ratios = sorted(r["ratio"] for r in rows)
    volaws = sorted(r["volaw"] for r in rows)
    print(f"[est-vs-built] {subject}: {len(rows)} rows, est/built "
          f"{ratios[0]:.2f}..{ratios[-1]:.2f} (median "
          f"{statistics.median(ratios):.2f}), built/volaw "
          f"{volaws[0]:.3f}..{volaws[-1]:.3f}")
    print(f"[est-vs-built] {subject}: the margin must be >= {volaws[-1]:.3f} "
          f"(the highest built/volaw -- below it the estimate under-reads); at "
          f"that margin the worst over-estimate is "
          f"{volaws[-1] / volaws[0]:.2f}x on {len(rows)} rows")
    olds = [r["old_ratio"] for r in rows if r.get("old_ratio")]
    if olds:
        print(f"[est-vs-built] {subject}: the formula this change replaces would "
              f"read est/built {min(olds):.2f}..{max(olds):.2f} on the same meshes")


def _cases(blob, path):
    """The case dicts in a recorded run file, in whichever shape it uses.

    A matrix/bench run writes ONE case per file, under ``"case"``; an aggregate
    writes a ``"cases"`` list. Both are read here so a change to either writer
    shows up as a row rather than as a silently empty table.
    """
    if isinstance(blob, dict) and isinstance(blob.get("cases"), list):
        return blob["cases"]
    if isinstance(blob, dict) and isinstance(blob.get("case"), dict):
        return [blob["case"]]
    if isinstance(blob, dict) and "plan" in blob:
        return [blob]
    print(f"[est-vs-built] no case in this file, skipped: {path}")
    return []


def results_table(dirs):
    """Read recorded run JSONs and print the same rows from what they recorded.

    Rows are NOT deduplicated across directories: the same label in two run
    directories is two different runs, and the whole point of the row is the
    comparison. Each row names the directory it came from, and the summary is
    printed per directory for the same reason.
    """
    for d in dirs:
        before = len(ROWS)
        seen, duplicates = set(), 0
        for path in sorted(Path(d).glob("*.json")) + sorted(Path(d).glob("*/*.json")):
            try:
                blob = json.loads(path.read_text())
            except Exception:  # noqa: BLE001 - a stray file is not a finding
                print(f"[est-vs-built] unreadable as JSON, skipped: {path}")
                continue
            for case in _cases(blob, path):
                label = case.get("label") or blob.get("label")
                plan = case.get("plan") or {}
                built = case.get("n_tets_uniform_region_aware")
                if not built or not plan:
                    print(f"[est-vs-built] {label or path}: no plan or no built "
                          "count, skipped")
                    continue
                edges = [float(e["edge_m"]) for e in plan.values() if e.get("edge_m")]
                if not edges:
                    print(f"[est-vs-built] {label}: no edge_m in the plan, skipped")
                    continue
                h_m = min(edges)
                volume_m3 = sum(float(e["filled_volume_m3"]) for e in plan.values())
                # The uniform rule the mesher applies on main: one edge (h_min)
                # for every region. The recorded plan's own per-region edges are
                # the grading bench's clamp, which is a different scheme -- the
                # uniform number is computed here so these rows are the SAME
                # quantity as the live-estimate rows, not the bench's.
                est = int(CFG.est_tets_margin() * TETS_PER_H3
                          * (volume_m3 / h_m ** 3))
                # A run directory holds each case once at its top level AND
                # once per matrix revision subdirectory, so the same
                # measurement arrives many times. Collapse identical rows (the
                # same case, edge, volume and count) and say how many were
                # dropped: a repeated row would otherwise look like agreement.
                identity = (label, round(h_m, 12), round(volume_m3, 18), int(built))
                if identity in seen:
                    duplicates += 1
                    continue
                seen.add(identity)
                recorded = case.get("h_min_mm_product")
                if recorded and abs(float(recorded) - h_m * 1e3) > 1e-6 * float(recorded):
                    # The plan's edge and the run's own reported h_min disagree:
                    # say so rather than quietly measuring a different h.
                    print(f"[est-vs-built] {label}: plan h {h_m * 1e3:.6g} mm vs the "
                          f"run's recorded h_min {recorded} mm")
                _row(f"{label} [{Path(d).name}]", h_m, volume_m3, est, int(built),
                     est_old=int(OLD_C * volume_m3 / h_m ** 3), family="recorded")
        _summarize(ROWS[before:], f"dir {Path(d).name}"
                   + (f", {duplicates} duplicate rows dropped" if duplicates else ""))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true", help="tier-1 tiny fixtures, real mesher")
    ap.add_argument("--families", action="store_true",
                    help="tier-1 shape families (the margin's own sweep), real mesher")
    ap.add_argument("--results", nargs="*", default=[],
                    help="directories of recorded run JSONs (no compute)")
    ap.add_argument("--size-mm", type=float, default=6.0, help="fixture side (tier 1: keep tiny)")
    ap.add_argument("--core-ratio", type=float, default=0.25)
    ap.add_argument("--core-index", type=float, default=3.5, help="n of the core region")
    ap.add_argument("--n-modes", type=int, default=N_MODES,
                    help="kept small: the count goes like n_modes, tier 1 is tiny")
    ap.add_argument("--max-tets", type=int, default=None,
                    help="the mesher cap the bench runs under; default is the "
                         "product's own CFG.max_tets() (do not lower it to keep a "
                         "run small: the mesher coarsens to fit, so the numbers "
                         "stop being the product's)")
    ap.add_argument("--family-scale-n", type=float, default=1.0,
                    help="multiply every family's n_modes (step 3 on the Orin)")
    args = ap.parse_args()
    global MAX_TETS, FAMILY_N_SCALE
    MAX_TETS = CFG.max_tets() if args.max_tets is None else args.max_tets
    FAMILY_N_SCALE = args.family_scale_n
    print(f"[est-vs-built] margin = CFG.est_tets_margin() = "
          f"{CFG.est_tets_margin():g}   tets per h^3 at edge h = {TETS_PER_H3:.3f}   "
          f"maxvolume = {COMMON.TETGEN_VOLUME_SLACK} * h^3/(6*sqrt(2))   "
          f"(the formula this change replaces used C = {OLD_C:g} * V/h^3)")
    # The cap belongs in the transcript, beside every number: a table whose
    # fixture cap is not written down cannot be told from the product's.
    # Compared by VALUE, not by "was a flag passed": passing the product's own
    # cap is not an override, and labelling it as one makes an honest run look
    # like a fixture run (the lead, 2026-10-15: ``--max-tets 600000`` printed
    # "(OVERRIDDEN: not the product's 600,000)").
    override = "" if MAX_TETS == CFG.max_tets() else (
        f" (OVERRIDDEN: not the product's {CFG.max_tets():,})")
    print(f"[est-vs-built] mesher cap = {MAX_TETS:,}{override}"
          f"   n_modes scale = {FAMILY_N_SCALE:g}")
    if not (args.results or args.build or args.families):
        ap.print_help()
        return 0
    if args.results:
        results_table(args.results)
    if args.build:
        before = len(ROWS)
        build_fixtures(args.size_mm, args.core_ratio, args.core_index, args.n_modes)
        _summarize(ROWS[before:], f"fixtures {args.size_mm:g} mm, n_modes "
                                  f"{args.n_modes}")
    if args.families:
        before = len(ROWS)
        family_rows()
        _summarize(ROWS[before:], "shape families")
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
