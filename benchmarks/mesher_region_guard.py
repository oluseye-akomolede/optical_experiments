"""Can a region ship under-resolved while the too-coarse guard passes?

The guard and the resolution rule do not read the same number.

  * ``maxvolume`` is built from ``h_min`` -- the FINEST region edge
    (``mse_meshing.py`` ~:506, ``region_plan['h_min_units']``).
  * ``allowed_edge`` USED TO BE ``MAX_EDGE_OVERSHOOT (1.6) * edge_m``
    (~:568), and ``edge_m`` is ``effective_edge_m(...)``, which -- by its own
    docstring and its own recorded history ("an earlier version took
    min(base/n_r) here ... over-predicted the count by more than 10x") -- is the
    WALL's edge, NOT ``h_min``. That wall-based reading is what this script was
    written to measure the structural slack of. The merged guard now reads each
    region's OWN edge ``h_r = h_wall * n_wall / n_r`` from the plan
    (``region_plan[...]['own_edge_units']``). BOTH readings are printed; the
    wall-based one is labelled PRE-FIX.

Under the OLD rule the guard's tolerance in units of a region's OWN required
edge ``h_r`` was

    allowed_edge / h_r = MAX_EDGE_OVERSHOOT * n_r / n_wall

-- 1.6x for the wall itself (the reference region), but 1.6 * 3.5 = 5.6x for a
core at n=3.5 and 1.6 * 3 = 4.8x for the n=3 core of the 3-region family. The
question this script answers with numbers is whether that structural slack is
ever EXERCISED: is any region actually delivered coarser than its own ``h_r``,
let alone 1.6x coarser, on a mesh the guard accepts? Under the merged per-region
guard the second reading is the product's verdict -- ``REGION GUARD`` -- and the
first (wall-based, pre-fix) is kept beside it as the evidence for the change.

Measured per region, on the product's own code path and cap, with the guard's
own instrument (``achieved_edge_length``, the median tet edge) applied to the
tets that region owns:

  * ``h_wall``      the wall's edge; the guard's own reference (mm)
  * ``h_r``         the region's required edge = h_wall * n_wall / n_r (mm)
  * ``ach``         median tet edge of THAT region's tets (mm)
  * ``ach/h_r``     > 1 means the region is coarser than its own rule needs
  * ``ach/h_min``   > 1 means it is coarser than the edge the cavity was meshed
                    at (the plan's per-region target is uniform ``h_min``)
  * ``guard``       the guard's own verdict: global achieved vs 1.6 * h_wall

Run it on the VM (tier 1, bounded; all fixtures are 6 mm boxes):

    (ulimit -v 4194304; OMP_NUM_THREADS=1 nice -n 19 .venv-mse/bin/python \
        benchmarks/mesher_region_guard.py)

Findings on 2026-10-15 are recorded in
``openspec/changes/mesher-region-guard/proposal.md``.
"""

from __future__ import annotations

import contextlib
import io
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
for _p in (_ROOT, _ROOT / "dockerfiles" / "maxwell_solver", _ROOT / "web"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import mse_common as COMMON  # noqa: E402
import mse_meshing as MESHSOLVER  # noqa: E402
import mse_parts as PARTS  # noqa: E402
from benchmarks.estimate_vs_built import FAMILIES, _obj_bytes  # noqa: E402
from web import config as CFG  # noqa: E402

#: The product's cap, not a tier-1 literal: a lower cap makes the mesher
#: COARSEN ``maxvolume`` and rebuild, which answers a different question.
MAX_TETS = CFG.max_tets()

SCALES = (1.0, 0.5, 2.0)


def prepare(label: str, parts_spec, n_modes: int):
    """Build one family and derive its reference edges -- everything but the cap.

    The per-region required edges are derived HERE, once, so this probe and the
    cap-retry probe cannot drift apart about what the resolution rule asks for.
    """
    import trimesh
    from types import SimpleNamespace

    names = {n: name for n, (name, _) in parts_spec.items()}
    materials = {name: n for n, (name, _) in parts_spec.items()}
    surfaces = {names[n]: build() for n, (name, build) in parts_spec.items()}
    multi = len(surfaces) > 1

    if multi:
        body = _obj_bytes("upload", trimesh.Scene(surfaces))
        parts = PARTS.load_parts(body, units="mm")
        surface = PARTS.merged_surface(parts)
    else:
        body = _obj_bytes(next(iter(surfaces)), next(iter(surfaces.values())))
        surface, _ = MESHSOLVER.load_surface(body, units="mm")
        parts = None

    unit_m = COMMON.unit_scale("mm")
    # ``effective_edge_m`` with parts/materials is the value the guard reads.
    h_wall_m = COMMON.effective_edge_m(
        surface, n_modes=n_modes, units="mm",
        parts=parts, materials=materials,
    )
    h_wall_units = h_wall_m / unit_m
    if multi:
        plan = COMMON.region_plan(parts, materials, h_wall_units, "mm")
        by_region = parts.region_materials(materials)
        n_wall = float(by_region[int(parts.region_of_part[parts.wall_index])])
        h_min_units = min(float(info["edge_units"]) for info in plan.values())
        part_of_region = {int(rid): info["part"] for rid, info in plan.items()}
    else:
        # One region, and its id is not known until the run returns the
        # per-tet region array (``tetrahedralize`` always returns one, all
        # zeros here, so no caller branches on the region count).
        by_region = None
        n_wall = 1.0
        h_min_units = h_wall_units
        part_of_region = {}

    return SimpleNamespace(
        label=label, n_modes=n_modes, surface=surface, parts=parts,
        materials=materials, multi=multi, unit_m=unit_m,
        h_wall_m=h_wall_m, h_wall_units=h_wall_units,
        allowed_m=COMMON.MAX_EDGE_OVERSHOOT * h_wall_m,
        n_wall=n_wall, by_region=by_region, part_of_region=part_of_region,
        h_min_units=h_min_units,
    )


def _run(label: str, parts_spec, n_modes: int) -> None:
    c = prepare(label, parts_spec, n_modes)
    surface, parts, materials = c.surface, c.parts, c.materials
    unit_m, h_wall_m = c.unit_m, c.h_wall_m
    h_wall_units, allowed_m = c.h_wall_units, c.allowed_m
    n_wall, by_region = c.n_wall, c.by_region
    part_of_region, h_min_units = c.part_of_region, c.h_min_units
    print(f"\n{label}   (n_modes={n_modes})", flush=True)
    print(f"  h_wall {h_wall_units:.4f} mm (the guard's reference, "
          f"{COMMON.MAX_EDGE_OVERSHOOT:g}x -> {allowed_m / unit_m:.4f} mm); "
          f"h_min {h_min_units:.4f} mm (the request)")

    buf = io.StringIO()
    outcome, err = "accepted", ""
    try:
        with contextlib.redirect_stdout(buf):
            points, tets, regions = MESHSOLVER.tetrahedralize(
                surface, "mm", n_modes=n_modes, parts=parts,
                materials=materials, max_tets=MAX_TETS, verbose=True,
            )
    except Exception as exc:  # noqa: BLE001 - a refusal is a row, not a crash
        outcome, err = type(exc).__name__, str(exc)
        points = tets = regions = None
    log = buf.getvalue()
    # The mesher's own words: "too coarse; refining with maxvolume V" (refine
    # arm) and "N tets is over the M cap; retrying with maxvolume V" (cap arm).
    # Counting a string the mesher never prints would make this check unable to
    # fail -- it once read "raising maxvolume" and reported 0 on a run whose
    # count had visibly fallen.
    refines = log.count("too coarse; refining")
    coarsens = log.count("retrying with maxvolume")

    if points is None:
        print(f"  OUTCOME {outcome}: {err.splitlines()[0][:100]}")
        print(f"  retries: refine {refines}, coarsen {coarsens}")
        return

    # ``tetrahedralize`` returns ``points`` in metres (it applies
    # ``unit_scale_m`` on the way out); the guard, inside the loop, measures the
    # same metres. So this is the guard's own number, not a re-scaled copy.
    achieved_global = MESHSOLVER.achieved_edge_length(points, tets)
    verdict = "PASS" if achieved_global <= allowed_m else "FAIL"
    print(f"  n={len(tets):,}  OUTCOME {outcome}  retries: refine {refines}, "
          f"coarsen {coarsens}")
    print(f"  PRE-FIX (wall-based) GUARD {verdict}: global achieved "
          f"{achieved_global / unit_m:.4f} mm = {achieved_global / allowed_m:.3f}x "
          f"the {allowed_m / unit_m:.4f} mm allowed")
    if err:
        print(f"  (a warning was printed: {err.splitlines()[0][:80]})")

    # The one-part path registers no regions, so TetGen tags no tets and
    # ``regions`` is None there; the whole mesh is that single region.
    reg = (np.asarray(regions) if regions is not None
           else np.ones(len(tets), dtype=int))
    if by_region is None:
        by_region = {int(r): 1.0 for r in np.unique(reg)}
        part_of_region = {r: "1 body" for r in by_region}
    worst_ratio, worst_rid = 0.0, None
    for rid in sorted(by_region):
        n_r = float(by_region[rid])
        mask = reg == rid
        if not mask.any():
            print(f"    region {rid:g} ({part_of_region.get(rid, '?')}, "
                  f"n={n_r:g}): NO TETS")
            continue
        h_r_units = h_wall_units * n_wall / n_r
        ach = MESHSOLVER.achieved_edge_length(points, tets[mask])
        ratio_own = ach / unit_m / h_r_units
        if ratio_own > worst_ratio:
            worst_ratio, worst_rid = ratio_own, rid
        print(f"    region {rid:g} ({part_of_region.get(rid, '?'):<6} "
              f"n={n_r:<4g} tets={int(mask.sum()):>7})  "
              f"h_r {h_r_units:.4f} -> ach {ach / unit_m:.4f} mm  "
              f"ach/h_r {ratio_own:6.3f}  "
              f"ach/h_min {ach / unit_m / h_min_units:6.3f}  "
              f"(pre-fix wall-based slack "
              f"{COMMON.MAX_EDGE_OVERSHOOT * n_r / n_wall:.2f}x its own)")
    # THE PRODUCT'S VERDICT: every region against 1.6x its OWN required edge,
    # which is what ``tetrahedralize`` refuses on since the per-region guard.
    v = "PASS" if worst_ratio <= COMMON.MAX_EDGE_OVERSHOOT else "FAIL"
    print(f"  REGION GUARD {v}: worst region {worst_rid:g} at "
          f"{worst_ratio:.3f}x its OWN required edge, against the "
          f"{COMMON.MAX_EDGE_OVERSHOOT:g}x the guard allows")


def main() -> int:
    scales = SCALES
    if len(sys.argv) > 1:
        scales = tuple(float(a) for a in sys.argv[1:])
    print("PER-REGION ACHIEVED EDGE vs EACH REGION'S OWN REQUIRED EDGE")
    print(f"product code path, cap = {MAX_TETS:,}, scales {scales}")
    for scale in scales:
        print(f"\n{'=' * 72}\nSCALE n x{scale:g}\n{'=' * 72}")
        for label, parts_spec, n_modes in FAMILIES:
            n = max(1, int(round(n_modes * scale)))
            try:
                _run(label, parts_spec, n)
            except Exception as exc:  # noqa: BLE001
                print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
