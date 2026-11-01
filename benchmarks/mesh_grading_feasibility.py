"""Feasibility probe: can a NESTED PLC be graded in the tetgen 0.8.4 we ship?

Row qitem-20261012130556-d64fb79e (STEP 1, hard gate). The starting fact
(09-28, row 0d338445) is that ``TetGen.add_region(max_vol=...)`` is INERT on a
nested PLC across nine orders of magnitude under every flag. This probe tests
the other candidate: TetGen's mesh-sizing function (``-m``) -- a background
mesh whose nodes carry a per-node target EDGE length.

RESULT: it works.

Method
------
Build a tiny nested fixture (two concentric cubes) in the mesher's canonical
O(1) frame, tetrahedralize a COARSE background mesh of the same PLC, tag every
background node with a target edge (fine inside the core, coarse in the wall),
and hand that grid to ``tetrahedralize(bgmesh=...)``. The wrapper (pyvista
``tetgen`` 0.8.4) appends ``m`` to the switches and feeds the sizing field.

The two controls that can fail
------------------------------
1. Reduction: the graded mesh must have clearly fewer tets than a uniform mesh
   at the core's resolution.
2. Resolution: the graded CORE median edge must match what the uniform-at-h_core
   mesh actually achieved -- you may not buy the reduction by coarsening the
   region that has to stay fine.

The resolution target is the MEASURED uniform core edge, not ``h_core``.
``maxvolume=mv(h)`` over-refines: TetGen's elements come out at ~0.4-0.5 h, so
comparing against ``h_core`` would call a correct graded mesh "too coarse".

What does NOT work
------------------
* ``add_region(max_vol=...)`` -- inert on a nested PLC (carried over from 09-28;
  re-confirmed by the fixture here meshing to the same size at any max_vol).
* ``fixedvolume=True`` / a global ``maxvolume`` alongside ``bgmesh`` -- they
  OVERRIDE the sizing field and coarsen the whole mesh (measured).
* A coarse target LARGER than the region can hold (e.g. a wall target wider than
  the wall shell's thickness) -- TetGen lands on the size floor and the "coarse"
  drive degrades the WHOLE mesh, making it coarser than uniform even in the
  core. The coarse target must sit inside the range the region can actually
  hold (roughly, below the shell thickness).

Defaults below are the measured PASS: a/b=0.25, h_core=0.10, ratio=3.0,
h_bg=0.22 -> GRADED 2813 tets vs UNIFORM 39454 (93% fewer) with the core edge
0.0432 vs 0.0433 (0.2% off). The whole run, four meshes, is ~1.6 s.

    (ulimit -v 4194304; nice -n 19 .venv-mse/bin/python -I \
        benchmarks/mesh_grading_feasibility.py)

Optional argv (sweep mode, one compact line per run):
    wall_side a_over_b h_core ratio h_bg
"""

import sys

import numpy as np
import pyvista as pv
import tetgen
import trimesh

WALL_SIDE = 2.0
A_OVER_B = 0.25          # core / wall side; the wall shell is (0.5 - a/b/2) wide
RATIO = 3.0              # h_wall / h_core
H_CORE = 0.10            # canonical (longest side = 1)
H_BG = 0.22              # background-mesh element size, canonical

# Optional argv: wall_side a_over_b h_core ratio h_bg  -> one compact line.
ARGV = [float(a) for a in sys.argv[1:6]] if len(sys.argv) >= 6 else None
if ARGV:
    WALL_SIDE, A_OVER_B, H_CORE, RATIO, H_BG = ARGV


def nested_surface(wall_side, core_side):
    """Two concentric cube shells, outward-oriented, in one PLC."""
    w = trimesh.creation.box(extents=[wall_side] * 3)
    c = trimesh.creation.box(extents=[core_side] * 3)
    vw, fw = np.asarray(w.vertices, float), np.asarray(w.faces, np.int64)
    vc, fc = np.asarray(c.vertices, float), np.asarray(c.faces, np.int64)
    return np.vstack([vw, vc]), np.vstack([fw, fc + len(vw)])


def canonical(V):
    """Centre + scale to longest side 1 -- the frame the real mesher uses."""
    lo, hi = V.min(0), V.max(0)
    c = (lo + hi) / 2.0
    L = float(np.max(hi - lo))
    return (V - c) / L, c, 1.0 / L


def mv(h):
    """Volume of a regular tetrahedron of edge h -- the maxvolume the product asks."""
    return (h ** 3) / (6.0 * np.sqrt(2.0))


def tetra(V, F, maxvolume, fixed=True, **kw):
    t = tetgen.TetGen(V, F)
    t.tetrahedralize(order=1, mindihedral=20, minratio=2.0,
                     maxvolume=maxvolume, fixedvolume=fixed, **kw)
    return np.asarray(t.node, float)[:, :3], np.asarray(t.elem, np.int64)


def edges_of(pts, tets):
    xyz = pts[tets]
    loc = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    return np.linalg.norm(xyz[:, loc[:, 0], :] - xyz[:, loc[:, 1], :], axis=2)


def stats(pts, tets, core_half):
    """Median edge inside vs outside the core cube, and the two counts."""
    e = edges_of(pts, tets)
    cen = np.abs(pts[tets].mean(axis=1))
    inside = (cen <= core_half + 1e-9).all(axis=1)
    med = lambda m: float(np.median(e[m])) if m.any() else float("nan")
    return med(inside), med(~inside), int(inside.sum()), int((~inside).sum())


def vprint(*a, **k):
    if not ARGV:
        print(*a, **k)


def main():
    wall_side, core_side = WALL_SIDE, A_OVER_B * WALL_SIDE
    V, F = nested_surface(wall_side, core_side)
    V, off, sc = canonical(V)
    core_half = (core_side / 2.0) * sc
    h_core, h_wall = H_CORE, H_CORE * RATIO
    vprint(f"fixture: wall_side={wall_side} core_side={core_side} "
           f"core_half_c={core_half:.3f} faces={len(F)} "
           f"wall_thickness_c={0.5 - core_half:.3f}")
    vprint(f"h_core={h_core} h_wall={h_wall} (ratio {RATIO})\n")

    ref = {}
    for label, h in (("uniform@h_core", h_core), ("uniform@h_wall", h_wall)):
        pts, tets = tetra(V, F, mv(h), regionattrib=True)
        ci, co, ni, no = stats(pts, tets, core_half)
        ref[label] = (len(tets), ci, co)
        vprint(f"{label:16s}: n_tets={len(tets):6d} core_edge={ci:.4f} "
               f"wall_edge={co:.4f} (core={ni} wall={no})")

    # Preserve the resolution uniform@h_core ACTUALLY achieved (see module doc).
    n_uni_tgt, ci_uni_tgt, _ = ref["uniform@h_core"]

    # --- the graded attempt: a background mesh with a core/wall size field ---
    bg_pts, bg_tets = tetra(V, F, mv(H_BG), regionattrib=True)
    inside = (np.abs(bg_pts) <= core_half + 1e-9).all(axis=1)
    size = np.where(inside, ci_uni_tgt, h_wall)
    vprint(f"\nbgmesh: {len(bg_pts)} nodes ({int(inside.sum())} in core), "
           f"{len(bg_tets)} tets")
    grid = pv.UnstructuredGrid({pv.CellType.TETRA: bg_tets}, bg_pts)
    grid.point_data["target_size"] = size

    # No fixedvolume/maxvolume here: they OVERRIDE the sizing field.
    gpts, gtets = tetra(V, F, 0.0, fixed=False, regionattrib=True, bgmesh=grid)
    ci, co, ni, no = stats(gpts, gtets, core_half)
    vprint(f"{'graded (bgmesh)':16s}: n_tets={len(gtets):6d} core_edge={ci:.4f} "
           f"wall_edge={co:.4f} (core={ni} wall={no})")

    n_uni, ci_uni, co_uni = ref["uniform@h_core"]
    n_wall, ci_wall, co_wall = ref["uniform@h_wall"]
    red = 100.0 * (1 - len(gtets) / n_uni)
    ok_reduce = len(gtets) < 0.75 * n_uni
    ok_core = abs(ci - ci_uni) <= 0.25 * ci_uni
    vprint(f"\nCONTROL: tets {len(gtets)} vs uniform {n_uni} "
           f"-> {red:.0f}% fewer  [{'PASS' if ok_reduce else 'FAIL'}]")
    vprint(f"CONTROL: core edge {ci:.4f} vs target {ci_uni:.4f} "
           f"-> {100*abs(ci-ci_uni)/ci_uni:.0f}% off  "
           f"[{'PASS' if ok_core else 'FAIL'}]")
    vprint(f"(wall edge {co:.4f}; uniform@h_wall wall edge {co_wall:.4f}; "
           f"uniform@h_wall n_tets {n_wall})")

    if ARGV:
        print(f"a/b={A_OVER_B:<4} h_core={h_core:<5} h_wall={h_wall:<5} "
              f"wall_thk={0.5-core_half:.3f} | bgmesh={len(bg_tets):5d}t "
              f"| GRADED={len(gtets):6d}t core_e={ci:.4f} wall_e={co:.4f} "
              f"| UNIFORM={n_uni:6d}t core_e={ci_uni:.4f} "
              f"| cut={red:4.0f}% core_off={100*abs(ci-ci_uni)/max(ci_uni,1e-9):3.0f}% "
              f"[{'PASS' if (ok_reduce and ok_core) else 'FAIL'}]")


if __name__ == "__main__":
    main()
