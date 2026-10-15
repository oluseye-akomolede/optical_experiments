"""Does the mesher get the mesh it asked TetGen for? Measured, not argued.

The upload estimate charges every region at ``h_min`` -- the finest region's
requested edge. That is not a simplification of what the mesher does; it IS what
the mesher does: ``mse_meshing.py`` computes ONE global ``maxvolume`` from
``h_min`` (``region_plan['h_min_units']`` -> ``target_maxvolume``) and hands it
to TetGen for the whole domain. There is no per-region edge anywhere in the
product: regions are registered by seed only, with no per-region volume (see the
measured-INERT note in ``mse_meshing.py``), so ``h_r == h_min`` for every region
is the mesher's request, exactly.

What the mesher does NOT control is what comes back. ``maxvolume`` is a
constraint TetGen is asked to satisfy, not one it guarantees, and it can miss it
in EITHER direction: on some geometries it returns tetrahedra larger than the
bound it was given, on others the whole mesh comes back several times FINER than
the bound it was given (measured: a 3-region nested fixture built 419,176
elements where the bound licenses 46,825, i.e. 9x the bound's own count). This
script measures the gap per family, at the tetrahedron level:

  * ``over`` -- how many tets exceed the requested maxvolume (a tet at the bound
    is legal; this counts only the violators);
  * ``worst`` -- the largest tet, as a multiple of the bound;
  * ``avg`` / ``med`` -- the mean and median tet, as multiples of the bound
    (> 1 means the mesh as a whole missed the request coarse; < 1 fine);
  * ``fine`` -- the share of tets at half the requested edge or finer;
  * ``eq-edge`` -- the volume-equivalent delivered edge; and
  * a per-region breakdown, because "the mesh came out coarse" is only
    actionable when it names WHICH region.

The control column re-meshes the same PLC with no regions registered (the
one-part path, which asks for its own coarser edge) -- it shows the bound IS
honoured when the request is coarse, so a violation is a property of the request
and the geometry, not of region registration.

Run it on the VM (tier 1, all families are 6 mm boxes):

    (ulimit -v 4194304; nice -n 19 .venv-mse/bin/python benchmarks/mesher_edge_cause.py)

The findings this produced on 2026-10-15 are recorded in
``openspec/changes/fix-upload-tet-estimate/proposal.md``.
"""

from __future__ import annotations

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

#: The cap the PRODUCT meshes under, not a smaller tier-1 literal. This is not a
#: detail: when a fixture's unconstrained mesh crosses a lower cap, the mesher's
#: over-cap retry COARSENS ``maxvolume`` and rebuilds, so the measurement stops
#: being about the request the plan made. Two families did exactly that at
#: 60,000 (2026-10-15), which is how a bench constant became a product claim.
MAX_TETS = CFG.max_tets()


def _tet_volumes_mm3(points, tets) -> np.ndarray:
    """Signed-volume magnitudes of every tet, in mm^3.

    ``tetrahedralize`` returns points in metres; the requested maxvolume is in
    the upload's own units (mm here), so the points are converted before the
    comparison rather than the bound.
    """
    p = np.asarray(points, dtype=float) * 1e3
    t = np.asarray(tets, dtype=np.int64)
    a, b, c, d = (p[t[:, i]] for i in range(4))
    return np.abs(np.einsum("ij,ij->i", np.cross(b - a, c - a), d - a)) / 6.0


def _request(surface, parts, materials, n_modes):
    """The edge and maxvolume the mesher will compute for this upload.

    Both are read from the same functions the mesher reads, so the comparison is
    against the request that was really made, not a re-derivation of it.
    """
    edge_m = COMMON.effective_edge_m(surface, n_modes=n_modes, units="mm",
                                     parts=parts, materials=materials)
    if parts is None:
        h_m = edge_m
    else:
        plan = COMMON.region_plan(parts, materials, edge_m, "mm")
        h_m = min(float(info["edge_units"]) for info in plan.values())
    h_units = h_m / COMMON.unit_scale("mm")
    return h_units, float(COMMON.target_maxvolume(surface, h_units))


def _stats(tag, points, tets, bound_mm3, regions=None):
    v = _tet_volumes_mm3(points, tets)
    over = int((v > bound_mm3 * (1 + 1e-6)).sum())
    # A full octave finer than the request: the edge is half the requested one.
    # Both directions matter -- a mesh can miss the bound coarse OR fine, and
    # only the fine direction makes the upload estimate read too LOW.
    fine = int((v < bound_mm3 / 8.0).sum())
    # the edge of a regular tet holding the mean volume: V = h^3 / (6*sqrt(2)),
    # so h = (V * 6*sqrt(2))^(1/3) -- the same law the estimate counts with.
    eq_edge = (v.mean() * COMMON.TETS_PER_UNIT_VOLUME) ** (1 / 3)
    print(f"  {tag:<28} n={len(v):>7}  "
          f"over {over:>6} ({100 * over / len(v):5.1f}%)  "
          f"fine {100 * fine / len(v):5.1f}%  "
          f"med {np.median(v) / bound_mm3:6.3f}x  avg {v.mean() / bound_mm3:6.3f}x  "
          f"eq-edge {eq_edge:.4f} mm")
    if regions is not None:
        reg = np.asarray(regions)
        for rid in sorted(set(reg.tolist())):
            mask = reg == rid
            sub = v[mask]
            sub_over = int((sub > bound_mm3 * (1 + 1e-6)).sum())
            print(f"      region {rid}: n={mask.sum():>6}  "
                  f"over {100 * sub_over / max(mask.sum(), 1):5.1f}%  "
                  f"worst {sub.max() / bound_mm3:5.2f}x  "
                  f"avg {sub.mean() / bound_mm3:6.3f}x")
    return over, len(v)


def _family(label, parts_spec, n_modes):
    import trimesh

    names = {idx: name for idx, (name, _) in parts_spec.items()}
    materials = {name: idx for idx, (name, _) in parts_spec.items()}
    surfaces = {names[idx]: build() for idx, (name, build) in parts_spec.items()}
    print(f"\n{label}  (n_modes={n_modes})")

    if len(surfaces) == 1:
        body = _obj_bytes(next(iter(surfaces)), next(iter(surfaces.values())))
        surface, _ = MESHSOLVER.load_surface(body, units="mm")
        h_units, bound = _request(surface, None, None, n_modes)
        points, tets, regions = MESHSOLVER.tetrahedralize(
            surface, "mm", n_modes=n_modes, max_tets=MAX_TETS, verbose=False)
        print(f"  requested h_min {h_units:.4f} mm; no regions (one part)")
        _stats("product path", points, tets, bound, regions)
        return

    body = _obj_bytes("upload", trimesh.Scene(surfaces))
    parts = PARTS.load_parts(body, units="mm")
    surface = PARTS.merged_surface(parts)
    h_units, bound = _request(surface, parts, materials, n_modes)
    print(f"  requested h_min {h_units:.4f} mm; "
          f"{len(surfaces)} regions, all at that edge (one global maxvolume)")
    points, tets, regions = MESHSOLVER.tetrahedralize(
        surface, "mm", n_modes=n_modes, parts=parts, materials=materials,
        max_tets=MAX_TETS, verbose=False)
    _stats("product path (regions)", points, tets, bound, regions)

    # Control: the same PLC, no regions, its own (coarser) request. If the bound
    # is honoured here, the violation above is not region registration.
    h2, bound2 = _request(surface, None, None, n_modes)
    points, tets, _ = MESHSOLVER.tetrahedralize(
        surface, "mm", n_modes=n_modes, max_tets=MAX_TETS, verbose=False)
    print(f"  control: no regions, h_min {h2:.4f} mm")
    _stats("control (no regions)", points, tets, bound2)


def main() -> int:
    print("DELIVERED MESH vs THE MAXVOLUME THE MESHER REQUESTED")
    print(f"tier-1 fixtures, real TetGen, product code path, "
          f"cap = CFG.max_tets() = {MAX_TETS:,}")
    for label, parts_spec, n_modes in FAMILIES:
        try:
            _family(label, parts_spec, n_modes)
        except Exception as exc:  # noqa: BLE001 - a family that cannot build is a row
            print(f"  FAILED: {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
