"""Tier-1 probe: what asks for the n3 band medians, and what they are made of.

This is a DIAGNOSTIC, not a criterion: it prints and asserts nothing, and it
never runs a solver. It exists so the step-1 finding on row
`qitem-20261013130141-600283f7` can be reproduced rather than believed:

    (ulimit -v 4194304; nice -n 19 .venv-mse/bin/python \
        benchmarks/probe_n3_bands.py)

PART A -- the plan dict's own numbers for the n3 fixture (a 10 mm sphere with a
3.333 mm core at n_core = 3, the same recipe `run_multimaterial_bench` uses).
No mesher: `mse_common.region_plan` is called directly, so what it prints is
exactly the edge each region was ASKED for, beside the clamp's own inputs.

PART B -- a tiny analogue mesh at the same relative recipe (b = 6 mm, edge
0.9 mm, n_core = 3, 24,887 tets) whose achieved per-band medians can be
measured on the VM in seconds. The analogue is what turns "the Orin's shell
number is 0.504 x h_wall" from a coincidence into a mechanism: the same ratio
has to fall out at a different scale for the number to be a property of the
sizing background rather than of the Orin's fixture.

THE ORIN NUMBERS ARE A RECORD, NOT AN INPUT. `ORIN_11357F4` below is the lead's
measurement on the Orin; nothing here computes with it. It is quoted so the
analogue's ratios can be read beside the ratios they are meant to explain.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for _d in (ROOT / "benchmarks", ROOT / "dockerfiles" / "maxwell_solver",
           ROOT / "web"):
    sys.path.insert(0, str(_d))

import numpy as np  # noqa: E402
import trimesh  # noqa: E402
import run_multimaterial_bench as B  # noqa: E402
import mse_common as C  # noqa: E402
import mse_meshing as M  # noqa: E402

SOLVER = __import__("mse_maxwell_solver")

#: Achieved median edge per band (mm), measured on the Orin at rev 11357f4 by
#: the lead, on the n3 fixture below. Quoted for comparison only.
ORIN_11357F4 = {"core": 0.6640, "shell": 0.9014, "wall": 1.4530}
#: The same bands on that run's UNIFORM reference mesh, whose single edge is
#: h_min. Kept here because the comparison "graded wall vs uniform wall" is the
#: whole point of the bands.
ORIN_11357F4_UNIFORM = {"core": 0.7085, "shell": 0.7253, "wall": 0.5287}

B_MM, N_MODES = 10.0, 10
A_MM = B_MM / 3.0
MATERIALS = {"wall": 1.0, "core": 3.0}


def fixture(b_mm, a_mm):
    probe = trimesh.creation.icosphere(subdivisions=2, radius=b_mm)
    target = M.target_edge_length(probe, n_modes=N_MODES, units="mm",
                                  n_index=1.0)
    obj = B._parts_obj([("wall", b_mm), ("core", a_mm)], target)
    return obj, SOLVER.PARTS.load_parts(obj, units="mm"), float(target)


def part_a():
    print("=" * 72)
    print("A. the n3 fixture: what `region_edges` asks, and what the clamp took")
    print("=" * 72)
    _obj, parts, target = fixture(B_MM, A_MM)
    edge_m = target * 1e-3
    plan = C.region_plan(parts, MATERIALS, edge_m, "mm")
    print(f"target_edge_length (whole object, n_index=1) = {target:.4f} mm")
    print(f"{'region':>6} {'part':>6} {'n':>4} {'asked':>9} {'unclamped':>10} "
          f"{'thickness':>10} {'clamped':>8}")
    for rid, i in sorted(plan.items()):
        e, u, t = (i["edge_m"] * 1e3, i["unclamped_m"] * 1e3,
                   i["thickness_m"] * 1e3)
        print(f"{rid:>6} {i['part']:>6} {i['n']:>4.1f} {e:>9.4f} {u:>10.4f} "
              f"{t:>10.4f} {str(e < u * (1 - 1e-9)):>8}")
    h_min = min(i["edge_m"] for i in plan.values()) * 1e3
    h_wall = max(i["edge_m"] for i in plan.values()) * 1e3
    h_eff = h_wall * (h_min / h_wall) ** C.TRANSITION_EDGE_EXPONENT
    print(f"\nh_min {h_min:.4f} mm   h_wall {h_wall:.4f} mm   ratio "
          f"{h_wall / h_min:.3f}")
    print(f"h_eff (the wall priced at the edge the TRANSITION delivers, "
          f"exponent {C.TRANSITION_EDGE_EXPONENT}) = {h_eff:.4f} mm "
          f"= {h_eff / h_wall:.3f} x h_wall")
    outer = parts.parts[parts.wall_index if parts.wall_index is not None else 0]
    print(f"product element estimate (graded, transition-priced) = "
          f"{C.estimate_tets_by_region(parts, MATERIALS, edge_m, outer, 'mm', 14.0)}")
    try:
        import meshing as WEBMESH
        for label, kw in (("graded", dict(parts=parts, materials=MATERIALS)),
                          ("uniform@h_min", dict(min_feature_m=h_min * 1e-3))):
            print(f"web.meshing.estimate_tets {label:>14} = "
                  f"{WEBMESH.estimate_tets(outer, N_MODES, 'mm', c=14.0, **kw)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] web.meshing.estimate_tets unavailable: "
              f"{type(exc).__name__}: {exc}")

    print("\nOrin 11357f4, achieved median edge per band, beside what was asked:")
    print(f"  {'band':>5} {'asked':>8} {'built':>8} {'built/asked':>11}")
    for band in ("core", "shell", "wall"):
        asked = h_min if band == "core" else h_wall
        built = ORIN_11357F4[band]
        print(f"  {band:>5} {asked:>8.4f} {built:>8.4f} {built / asked:>11.3f}")
    print("  neither built number is h_min and neither is h_wall; note also that")
    print("  `wall` in this helper is `~core` -- the WHOLE outer region, so the")
    print("  number is a mixture of the interface transition and the wall bulk.")
    print("  (uniform mesh, same bands: "
          + " ".join(f"{b} {v:.4f}" for b, v in ORIN_11357F4_UNIFORM.items())
          + "; its one edge is h_min)")


def tet_sizes_mm(points_m, tets):
    """Per-tet mean edge length in mm (all six edges of the tet)."""
    p = np.asarray(points_m, float) * 1e3
    t = np.asarray(tets, np.int64)
    e = np.zeros(t.shape[0], float)
    for i, j in ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)):
        e += np.linalg.norm(p[t[:, i]] - p[t[:, j]], axis=1)
    return e / 6.0


def part_b(b_mm=6.0, edge_mm=0.9, n_core=3.0, shell_mm=0.6):
    print()
    print("=" * 72)
    print(f"B. analogue mesh (b={b_mm} mm, edge={edge_mm} mm, n_core={n_core}):")
    print("   what each band's median is actually made of")
    print("=" * 72)
    obj = B._parts_obj([("wall", b_mm), ("core", b_mm / 3.0)], edge_mm)
    parts = SOLVER.PARTS.load_parts(obj, units="mm")
    materials = {"wall": 1.0, "core": float(n_core)}
    merged = SOLVER.PARTS.merged_surface(parts)
    orig = M.target_edge_length          # this rig's tier-1 fixture control:
    M.target_edge_length = lambda *a, **k: edge_mm   # pin the whole-object edge
    try:
        pts, tets, region = M.tetrahedralize(
            merged, "mm", n_modes=N_MODES, n_index=1.0, max_tets=600_000,
            min_feature_m=None, verbose=False, materials=materials, parts=parts)
    finally:
        M.target_edge_length = orig
    a_eff, _b_eff, _ab = B._effective_radii(parts, n_core, units="mm")
    plan = C.region_plan(parts, materials, edge_mm * 1e-3, "mm")
    fine = int(min(plan, key=lambda r: plan[r]["edge_m"]))
    bands = B._band_edges_mm(pts, tets, float(a_eff), region_of_tet=region,
                             fine_region=fine, shell_mm=shell_mm)
    print(f"tets {tets.shape[0]:,};  a_eff {a_eff:.4f} mm;  "
          f"h_min {edge_mm / n_core:.3f} mm;  h_wall {edge_mm:.3f} mm")
    print("band medians from the BENCH'S OWN helper (the same function the Orin")
    print("run's numbers came from, `wall` included):")
    for name in ("core", "shell", "wall"):
        e = bands[name]["edge_mm"]
        print(f"  {name:>5}: {bands[name]['n_tets']:6d} tets  {e:.4f} mm  "
              f"= {e / edge_mm:.3f} x h_wall")

    radius = B._tet_radius_mm(pts, tets)
    size = tet_sizes_mm(pts, tets)
    # The helper's own masks, re-derived so a sub-population can be isolated.
    masks = {
        "shell": (radius > a_eff) & (radius <= a_eff + shell_mm),
        "wall bulk": radius > a_eff + shell_mm,
    }
    for label, sel in masks.items():
        vals = size[sel]
        hist, edges = np.histogram(vals, bins=[0, 0.15, 0.25, 0.35, 0.5, 0.7,
                                               0.9, 1.2, 10.0])
        pct = 100.0 * hist / len(vals)
        print(f"\n  {label:>10}: {int(sel.sum()):5d} tets  median "
              f"{np.median(vals):.3f} mm = {np.median(vals) / edge_mm:.3f} x h_wall")
        print("      " + "  ".join(
            f"{edges[i]:.2f}-{edges[i + 1]:.2f}:{pct[i]:.0f}%"
            for i in range(len(hist)) if pct[i] >= 0.5))
        near_fine = np.mean(np.abs(vals - edge_mm / n_core)
                            <= 0.15 * edge_mm / n_core)
        near_wall = np.mean(np.abs(vals - edge_mm) <= 0.15 * edge_mm)
        print(f"      within 15% of h_min: {near_fine * 100:.0f}%   of h_wall: "
              f"{near_wall * 100:.0f}%   neither (the ramp): "
              f"{(1 - near_fine - near_wall) * 100:.0f}%")
    print("\n  READ: `shell` is 0-15% h_min and 0% h_wall -- it is almost all")
    print("  ramp, so its median is no region's target, and `wall` (~core) is a")
    print("  mixture of that ramp with a bulk that IS built at h_wall.")


if __name__ == "__main__":
    part_a()
    part_b()
