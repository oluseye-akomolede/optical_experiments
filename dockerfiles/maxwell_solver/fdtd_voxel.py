"""Voxelise an upload's parts onto an FDTD grid (task 3.1).

Every sample point gets the index of the innermost part that contains it
(``part index + 1``), or ``OUTSIDE`` (0) when no part does. The materials table
then turns indices into the grid ``simulate`` takes: a relative permittivity per
node, a PEC mask, and one node mask per Rakić metal. Outside the parts the grid
is a metal wall (``boundary="metal"``: the upload is a cavity) or vacuum
(``boundary="open"``: the object sits in the gap the grid builder left).

Inside/outside is decided by casting a ray along +z through each sample's
(x, y) and counting the part's surface crossings below the sample: an odd count
is inside. The point-in-mesh winding test the cavity tool uses answers the same
question, but at (points x faces) cost; one ray answers every sample on its
(x, y) line, which is what makes a few-million-node grid affordable.

A ray through a triangle edge or vertex would count that crossing twice, or not
at all. Axis-aligned uploads (boxes, films) put mesh edges exactly on grid
lines, so this is the common case, not a corner case. Each query is therefore
shifted by ``_JITTER`` x the scene size along three incommensurate directions,
so no ray can pass through an edge or vertex of a mesh written to grid-friendly
coordinates. The shift is ~1e-7 of the scene, far below any cell.
"""

from __future__ import annotations

import numpy as np

import fdtd
import fdtd_grid
import mse_geometry as GEO

OUTSIDE = 0
_JITTER = np.array([np.sqrt(2.0) - 1.0, np.sqrt(3.0) - 1.5, np.pi - 3.0]) * 1e-6
_PAIR_BUDGET = 4_000_000     # (rays x candidate faces) per vectorised block


class VoxelError(fdtd.FDTDError):
    """A materials table or boundary that the voxeliser cannot build."""


def nesting_depths(parts) -> list:
    """How many other parts enclose each part (0 for an outermost part)."""
    depths = []
    for i, mine in enumerate(parts):
        depths.append(sum(
            1 for j, other in enumerate(parts)
            if j != i and GEO.part_contains(other, mine)
            and not GEO.part_contains(mine, other)))
    return depths


def _ray_crossings(tris, rays):
    """Per ray, the sorted z of every crossing with ``tris`` (list of arrays).

    ``tris`` is (n, 3, 3); ``rays`` is (m, 2) of (x, y). Faces parallel to z
    project to zero area and are skipped: a jittered ray cannot hit them.
    """
    out = [np.empty(0) for _ in range(len(rays))]
    if len(tris) == 0 or len(rays) == 0:
        return out
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    area = ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1])
            - (c[:, 0] - a[:, 0]) * (b[:, 1] - a[:, 1]))
    keep = area != 0.0
    a, b, c, area = a[keep], b[keep], c[keep], area[keep]
    lo = np.minimum(np.minimum(a, b), c)[:, :2]
    hi = np.maximum(np.maximum(a, b), c)[:, :2]
    order = np.argsort(rays[:, 0], kind="stable")
    block = max(1, _PAIR_BUDGET // max(len(a), 1))
    for start in range(0, len(order), block):
        idx = order[start:start + block]
        r = rays[idx]
        # faces whose xy box can meet any ray of this block (sorted by x)
        near = ((hi[:, 0] >= r[:, 0].min()) & (lo[:, 0] <= r[:, 0].max())
                & (hi[:, 1] >= r[:, 1].min()) & (lo[:, 1] <= r[:, 1].max()))
        if not near.any():
            continue
        fa, fb, fc, fs = a[near], b[near], c[near], area[near]
        px, py = r[:, 0:1], r[:, 1:2]
        # barycentric weights of the ray's (x, y) in each projected face
        wa = ((fb[:, 0] - px) * (fc[:, 1] - py)
              - (fc[:, 0] - px) * (fb[:, 1] - py)) / fs
        wb = ((fc[:, 0] - px) * (fa[:, 1] - py)
              - (fa[:, 0] - px) * (fc[:, 1] - py)) / fs
        wc = 1.0 - wa - wb
        hit = (wa > 0) & (wb > 0) & (wc > 0)
        z = wa * fa[:, 2] + wb * fb[:, 2] + wc * fc[:, 2]
        for k in np.flatnonzero(hit.any(axis=1)):
            out[idx[k]] = np.sort(z[k, hit[k]])
    return out


class PartIndex:
    """``index_of(points)`` for a list of closed parts, in part coordinates."""

    def __init__(self, parts, depths=None):
        self.tris = [GEO.of_faces(p.vertices, p.faces) for p in parts]
        self.depths = list(nesting_depths(parts) if depths is None else depths)
        lo = np.min([np.min(t.reshape(-1, 3), axis=0) for t in self.tris], axis=0)
        hi = np.max([np.max(t.reshape(-1, 3), axis=0) for t in self.tris], axis=0)
        self.shift = _JITTER * max(float(np.max(hi - lo)), 1e-300)

    def __call__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3) + self.shift
        rays, line = np.unique(pts[:, :2], axis=0, return_inverse=True)
        line = line.reshape(-1)
        by_line = np.argsort(line, kind="stable")
        starts = np.searchsorted(line[by_line], np.arange(len(rays) + 1))
        index = np.full(len(pts), OUTSIDE, dtype=np.int64)
        best = np.full(len(pts), -1, dtype=np.int64)
        for p, tris in enumerate(self.tris):
            crossings = _ray_crossings(tris, rays)
            inside = np.zeros(len(pts), dtype=bool)
            for r in np.flatnonzero([len(z) > 0 for z in crossings]):
                members = by_line[starts[r]:starts[r + 1]]
                below = np.searchsorted(crossings[r], pts[members, 2])
                inside[members] = below % 2 == 1
            deeper = inside & (self.depths[p] > best)
            index[deeper] = p + 1
            best[deeper] = self.depths[p]
        return index


def _material_table(names, materials):
    """Per index: ('eps', value) | ('pec',) | ('metal', name)."""
    missing = [n for n in names if n not in materials]
    unknown = [k for k in materials if k not in names]
    if missing or unknown:
        bits = []
        if missing:
            bits.append("no material given for part(s) "
                        + ", ".join(repr(m) for m in missing))
        if unknown:
            bits.append("the table names part(s) "
                        + ", ".join(repr(u) for u in unknown)
                        + " that are not in this file")
        raise VoxelError("; ".join(bits) + ". The file's parts are "
                         + ", ".join(repr(n) for n in names) + ".")
    table = []
    for name in names:
        value = materials[name]
        if isinstance(value, str):
            if value == "pec":
                table.append(("pec",))
            elif value in fdtd.METALS:
                table.append(("metal", value))
            else:
                raise VoxelError(
                    f"part {name!r}: unknown material {value!r}. Use a "
                    f"refractive index, 'pec', or one of {', '.join(fdtd.METALS)}.")
        elif isinstance(value, (int, float, np.integer, np.floating)) \
                and not isinstance(value, (bool, np.bool_)):
            n = float(value)
            if not np.isfinite(n) or n < 1.0:
                raise VoxelError(f"part {name!r}: refractive index {value!r} "
                                 "must be a finite number of at least 1.")
            table.append(("eps", n * n))
        else:
            raise VoxelError(
                f"part {name!r}: material {value!r} is not a refractive index "
                f"or a material name. Use a number of at least 1, 'pec', or one "
                f"of {', '.join(fdtd.METALS)}.")
    return table


def voxelise(names, parts, materials, nodes, *, boundary: str = "metal",
             length_scale: float = 1.0, samples: int = 3) -> dict:
    """The grid ``simulate`` takes, from named parts and a materials table.

    ``materials`` maps every part name to a refractive index, ``"pec"`` or a
    Rakić metal name. ``nodes`` are the grid's node coordinates in metres and
    ``length_scale`` is metres per mesh unit. Returns ``eps`` (relative
    permittivity per node), ``pec`` (bool mask, or None when there is no
    conductor) and ``metals`` ({name: node mask} for each metal present).
    """
    if boundary not in fdtd.BOUNDARIES:
        raise VoxelError(f"boundary {boundary!r} must be one of "
                         f"{', '.join(fdtd.BOUNDARIES)}.")
    names = list(names)
    table = _material_table(names, materials)
    finder = PartIndex(parts)
    # index 0 (outside) is a conductor for a cavity and vacuum for open space
    outside = ("pec",) if boundary == "metal" else ("eps", 1.0)
    kinds = [outside] + table
    eps_of = [k[1] if k[0] == "eps" else 1.0 for k in kinds]
    pec_ids = [i for i, k in enumerate(kinds) if k[0] == "pec"]
    metal_ids = [i for i, k in enumerate(kinds) if k[0] == "metal"]
    scale = 1.0 / float(length_scale)
    eps, masks = fdtd_grid.average_eps(
        nodes, lambda pts: finder(np.asarray(pts) * scale), eps_of,
        samples=samples, metals=pec_ids + metal_ids)
    pec = None
    for i in pec_ids:
        pec = masks[i] if pec is None else (pec | masks[i])
    metals = {}
    for i in metal_ids:
        name = kinds[i][1]
        metals[name] = metals[name] | masks[i] if name in metals else masks[i]
    metals = {k: v for k, v in metals.items() if v.any()}
    if pec is not None and not pec.any():
        pec = None
    return {"eps": eps, "pec": pec, "metals": metals}
