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
OVERLAP_LATTICE = 16         # per axis, in each pair's bounding-box intersection


class VoxelError(fdtd.FDTDError):
    """A materials table or boundary that the voxeliser cannot build."""


def _encloses(parts) -> list:
    """``inside[i][j]``: part ``i`` lies within part ``j`` (and not both ways).

    ``GEO.part_contains`` is exact but slow (seconds per pair of ~1000-face
    parts), so a pair is asked only when ``i``'s bounding box lies within
    ``j``'s: every vertex inside or on ``j`` lies within ``j``'s box, give or
    take ``point_in_mesh``'s on-surface tolerance (1e-9 of ``j``'s extent, at
    least 1e-9), which the 1e-8 slack covers. Disjoint, side-by-side and
    overlapping parts are never asked.
    """
    lo = [np.asarray(p.bounds[0], float) for p in parts]
    hi = [np.asarray(p.bounds[1], float) for p in parts]

    def could(i, j):
        slack = max(float(np.max(hi[j] - lo[j])), 1.0) * 1e-8
        return bool(np.all(lo[i] >= lo[j] - slack) and np.all(hi[i] <= hi[j] + slack))

    contains = [[i != j and could(i, j) and GEO.part_contains(other, mine)
                 for j, other in enumerate(parts)] for i, mine in enumerate(parts)]
    return [[contains[i][j] and not contains[j][i] for j in range(len(parts))]
            for i in range(len(parts))]


def nesting_depths(parts) -> list:
    """How many other parts enclose each part (0 for an outermost part)."""
    return [sum(row) for row in _encloses(parts)]


def priority_of(material) -> float:
    """Who owns a point on a face two parts share: a conductor (``pec`` or a
    Rakić metal) over any dielectric, and between dielectrics the higher
    refractive index."""
    return float(material) if not isinstance(material, str) else np.inf


class OverlapError(VoxelError):
    """Two parts, neither inside the other, share volume."""


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
    """``index_of(points)`` for a list of closed parts, in part coordinates.

    A point inside nested parts belongs to the innermost. A point on a face
    two parts share (a film flush against its plates) belongs to the part
    with the higher ``priority`` (``priority_of`` each part's material: the
    conductor, else the higher index); without priorities, or between equals,
    the jitter decides as before. Each point is tested shifted both ways
    along the jitter: the two answers differ exactly where it sits on a face.
    A point strictly inside two parts neither of which encloses the other is
    overlapping volume, refused as ``OverlapError`` naming both parts.
    """

    def __init__(self, parts, depths=None, *, priority=None, names=None):
        self.tris = [GEO.of_faces(p.vertices, p.faces) for p in parts]
        nested = _encloses(parts) if len(parts) > 1 else [[False]]
        self.nested = nested
        self.depths = list([sum(r) for r in nested] if depths is None else depths)
        self.apart = [(p, q) for p in range(len(parts)) for q in range(p)
                      if not nested[p][q] and not nested[q][p]]
        self.priority = None if priority is None else [float(x) for x in priority]
        self.names = (list(names) if names is not None
                      else [f"part {i + 1}" for i in range(len(parts))])
        lo = np.min([np.min(t.reshape(-1, 3), axis=0) for t in self.tris], axis=0)
        hi = np.max([np.max(t.reshape(-1, 3), axis=0) for t in self.tris], axis=0)
        self.shift = _JITTER * max(float(np.max(hi - lo)), 1e-300)

    def _inside(self, pts, which=None):
        """(parts, points) bool: each point inside each part (only the parts
        in ``which``, when given; the other rows stay False)."""
        rays, line = np.unique(pts[:, :2], axis=0, return_inverse=True)
        line = line.reshape(-1)
        by_line = np.argsort(line, kind="stable")
        starts = np.searchsorted(line[by_line], np.arange(len(rays) + 1))
        inside = np.zeros((len(self.tris), len(pts)), dtype=bool)
        for p in range(len(self.tris)) if which is None else which:
            crossings = _ray_crossings(self.tris[p], rays)
            for r in np.flatnonzero([len(z) > 0 for z in crossings]):
                members = by_line[starts[r]:starts[r + 1]]
                below = np.searchsorted(crossings[r], pts[members, 2])
                inside[p, members] = below % 2 == 1
        return inside

    def _deepest(self, inside):
        index = np.full(inside.shape[1], OUTSIDE, dtype=np.int64)
        best = np.full(inside.shape[1], -1, dtype=np.int64)
        for p in range(len(self.tris)):
            deeper = inside[p] & (self.depths[p] > best)
            index[deeper] = p + 1
            best[deeper] = self.depths[p]
        return index

    def __call__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        up = self._inside(pts + self.shift)
        index = self._deepest(up)
        if len(self.tris) < 2:
            return index
        down = self._inside(pts - self.shift)
        strict = up & down
        for p, q in self.apart:
            both = strict[p] & strict[q]
            if both.any():
                raise self._overlap(p, q, pts[int(np.flatnonzero(both)[0])])
        if self.priority is not None:
            other = self._deepest(down)
            face = (other != index) & (index > OUTSIDE) & (other > OUTSIDE)
            rank = np.array([-np.inf] + self.priority)
            take = face & (rank[other] > rank[index])
            index[take] = other[take]
        return index

    def _overlap(self, p, q, at):
        return OverlapError(
            f"parts {self.names[q]!r} and {self.names[p]!r} overlap "
            f"(both contain the point {[float(f'{x:.6g}') for x in at]} "
            f"in model units): parts may touch, but not share volume.")

    def refuse_overlap(self, lattice: int = OVERLAP_LATTICE) -> None:
        """Fail fast on parts that share volume, before any grid exists.

        Only where two non-nested parts' bounding boxes intersect can they
        overlap, so only that box is probed, on a ``lattice``^3 lattice of
        cell centres. Flush neighbours meet in a box of zero thickness, which
        is skipped at no cost. A lattice can step over a sliver thinner than
        its spacing: this check is sampled, and the voxeliser's refusal (every
        grid node tested) stays authoritative.
        """
        lo = [t.reshape(-1, 3).min(axis=0) for t in self.tris]
        hi = [t.reshape(-1, 3).max(axis=0) for t in self.tris]
        thin = 2.0 * float(np.max(np.abs(self.shift)))
        frac = (np.arange(lattice) + 0.5) / lattice
        for p, q in self.apart:
            a, b = np.maximum(lo[p], lo[q]), np.minimum(hi[p], hi[q])
            if np.any(b - a <= thin):
                continue
            axes = [a[i] + frac * (b[i] - a[i]) for i in range(3)]
            pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            strict = (self._inside(pts + self.shift, (p, q))
                      & self._inside(pts - self.shift, (p, q)))
            both = strict[p] & strict[q]
            if both.any():
                raise self._overlap(p, q, pts[int(np.flatnonzero(both)[0])])


def refuse_overlap(names, parts, lattice: int = OVERLAP_LATTICE) -> None:
    """``PartIndex.refuse_overlap`` for an upload's named parts."""
    if len(parts) > 1:
        PartIndex(parts, names=names).refuse_overlap(lattice)


def _segment_distance2(p, a, b):
    """Squared distance from points ``p`` (m, 1, 3) to segments a-b (1, k, 3)."""
    ab = b - a
    t = np.einsum("mki,mki->mk", p - a, np.broadcast_to(ab, (p.shape[0],) + ab.shape[1:]))
    t = np.clip(t / np.maximum(np.einsum("mki,mki->mk", ab, ab), 1e-300), 0.0, 1.0)
    d = p - (a + t[..., None] * ab)
    return np.einsum("mki,mki->mk", d, d)


def surface_distance(tris, points, within=None):
    """Each point's distance to the nearest of ``tris`` ((n, 3, 3)).

    The plane distance where the point projects inside a triangle, else the
    nearest of its three edges. Blocked so (points x triangles) stays under
    ``_PAIR_BUDGET``. Only triangles that can be a block's nearest are tested:
    those whose box is no farther from the block's box than some triangle can
    be from every point of it. With ``within``, triangles farther than that
    are skipped too, and a point with none nearer reads inf.
    """
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    best = np.full(len(points), np.inf)
    if len(tris) == 0 or len(points) == 0:
        return best
    a, b, c = tris[:, 0], tris[:, 1], tris[:, 2]
    n = np.cross(b - a, c - a)
    norm = np.linalg.norm(n, axis=1)
    flat = norm > 0
    unit = np.where(flat[:, None], n / np.where(flat, norm, 1.0)[:, None], 0.0)
    tlo = np.minimum(np.minimum(a, b), c)
    thi = np.maximum(np.maximum(a, b), c)
    order = np.argsort(points[:, 0], kind="stable")
    block = max(1, _PAIR_BUDGET // max(len(a), 1))
    for start in range(0, len(order), block):
        idx = order[start:start + block]
        p = points[idx]
        plo, phi = p.min(axis=0), p.max(axis=0)
        # a triangle no nearer than the block's farthest-point bound to the
        # nearest triangle box cannot be any point's nearest
        gap = np.maximum(np.maximum(tlo - phi, plo - thi), 0.0)
        box_d = np.linalg.norm(gap, axis=1)
        bound = (box_d + np.linalg.norm(thi - tlo, axis=1)).min() \
            + np.linalg.norm(phi - plo)
        if within is not None:
            bound = min(bound, float(within))
        k = np.flatnonzero(box_d <= bound)
        if not len(k):
            continue
        pa, pb, pc, pn, pf = a[k][None], b[k][None], c[k][None], unit[k][None], flat[k][None]
        q = p[:, None, :]
        plane = np.einsum("mki,mki->mk", q - pa, np.broadcast_to(pn, (len(p),) + pn.shape[1:]))
        foot = q - plane[..., None] * pn
        # the foot lies inside when it is on the inner side of all three edges
        inside = pf.copy()
        for u, v in ((pa, pb), (pb, pc), (pc, pa)):
            side = np.einsum("mki,mki->mk", np.cross(v - u, foot - u),
                             np.broadcast_to(pn, (len(p),) + pn.shape[1:]))
            inside = inside & (side >= 0)
        d2 = np.minimum(np.minimum(_segment_distance2(q, pa, pb),
                                   _segment_distance2(q, pb, pc)),
                        _segment_distance2(q, pc, pa))
        d2 = np.where(inside, plane * plane, d2)
        best[idx] = np.sqrt(d2.min(axis=1))
    return best


class CoatedIndex:
    """``PartIndex`` plus coating shells: index ``len(parts) + 1 + j`` is the
    ``j``-th coating in ``coatings``.

    ``coatings`` is ``[(part number, thickness in part units, priority)]``.
    A coating is a shell of its thickness OUTSIDE its part's surface: a sample
    is painted when it lies within the thickness of the part's surface and
    belongs to the space the coating grows into, which is open space
    (``paint_outside``, the open boundary) or a part that encloses the coated
    one. It is never painted into the coated part, the parts inside it, or a
    neighbour that does not enclose it (a neighbour keeps its volume), nor into
    a metal boundary's wall. Where two shells meet, the higher ``priority``
    (``priority_of``) wins, as on a shared face.
    """

    def __init__(self, finder, coatings, *, paint_outside: bool):
        self.finder = finder
        self.coatings = list(coatings)
        self.parts = len(finder.tris)
        nested = finder.nested
        self.into = []
        for p, _t, _rank in self.coatings:
            host = [OUTSIDE] if paint_outside else []
            host += [q + 1 for q in range(self.parts) if nested[p][q]]
            self.into.append(np.array(host, dtype=np.int64))
        self.box = []
        for p, t, _rank in self.coatings:
            flat = finder.tris[p].reshape(-1, 3)
            self.box.append((flat.min(axis=0) - t, flat.max(axis=0) + t))
        self.tris = finder.tris

    def __call__(self, points):
        pts = np.asarray(points, dtype=float).reshape(-1, 3)
        base = self.finder(pts)
        index = base.copy()
        # ascending priority, so where shells meet the higher one paints last
        for j in sorted(range(len(self.coatings)), key=lambda j: self.coatings[j][2]):
            p, t, _prio = self.coatings[j]
            lo, hi = self.box[j]
            near = (np.isin(base, self.into[j])
                    & np.all((pts >= lo) & (pts <= hi), axis=1))
            cand = np.flatnonzero(near)
            if not len(cand):
                continue
            # a sample on the shell's outer face is in the shell: the coating
            # owns its face as a part owns its own
            reach = t * (1 + 1e-9)
            hit = cand[surface_distance(self.tris[p], pts[cand], reach) <= reach]
            index[hit] = self.parts + 1 + j
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
    return [_kind(f"part {name!r}", materials[name]) for name in names]


def _kind(what, value):
    """('eps', n^2) | ('pec',) | ('metal', name) for one material."""
    if isinstance(value, str):
        if value == "pec":
            return ("pec",)
        if value in fdtd.METALS:
            return ("metal", value)
        raise VoxelError(
            f"{what}: unknown material {value!r}. Use a "
            f"refractive index, 'pec', or one of {', '.join(fdtd.METALS)}.")
    if isinstance(value, (int, float, np.integer, np.floating)) \
            and not isinstance(value, (bool, np.bool_)):
        n = float(value)
        if not np.isfinite(n) or n < 1.0:
            raise VoxelError(f"{what}: refractive index {value!r} "
                             "must be a finite number of at least 1.")
        return ("eps", n * n)
    raise VoxelError(
        f"{what}: material {value!r} is not a refractive index "
        f"or a material name. Use a number of at least 1, 'pec', or one "
        f"of {', '.join(fdtd.METALS)}.")


def coating_table(names, coatings):
    """``[(part number, material, thickness_m)]`` from ``{part name:
    (material, thickness_m)}``, in part order, refused by name when a coating
    names no part or carries a bad material or thickness."""
    coatings = dict(coatings or {})
    unknown = [k for k in coatings if k not in names]
    if unknown:
        raise VoxelError("a coating names part(s) "
                         + ", ".join(repr(u) for u in unknown)
                         + " that are not in this file. The file's parts are "
                         + ", ".join(repr(n) for n in names) + ".")
    out = []
    for p, name in enumerate(names):
        if name not in coatings:
            continue
        pair = coatings[name]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise VoxelError(f"the coating of part {name!r} must be a material "
                             f"and a thickness, got {pair!r}.")
        material, thickness = pair
        _kind(f"the coating of part {name!r}", material)
        if isinstance(thickness, (bool, np.bool_)) or not isinstance(
                thickness, (int, float, np.integer, np.floating)) \
                or not (np.isfinite(thickness) and thickness > 0):
            raise VoxelError(f"the coating of part {name!r} needs a positive "
                             f"thickness in metres, got {thickness!r}.")
        out.append((p, material, float(thickness)))
    return out


def coated_index(names, parts, materials, coatings, *, boundary: str,
                 length_scale: float):
    """The voxeliser's point lookup: ``PartIndex`` (shared-face rule
    included), wrapped in ``CoatedIndex`` when any part is coated. Returns
    ``(index_of, coats)`` with ``coats`` from ``coating_table``; index
    ``len(parts) + 1 + j`` is ``coats[j]``. Part coordinates."""
    finder = PartIndex(parts, priority=[priority_of(materials[n]) for n in names],
                       names=names)
    coats = coating_table(names, coatings)
    if not coats:
        return finder, coats
    scale = 1.0 / float(length_scale)
    return CoatedIndex(finder, [(p, t * scale, priority_of(m)) for p, m, t in coats],
                       paint_outside=boundary == "open"), coats


def voxelise(names, parts, materials, nodes, *, boundary: str = "metal",
             length_scale: float = 1.0, samples: int = 3,
             coatings=None) -> dict:
    """The grid ``simulate`` takes, from named parts and a materials table.

    ``materials`` maps every part name to a refractive index, ``"pec"`` or a
    Rakić metal name. ``nodes`` are the grid's node coordinates in metres and
    ``length_scale`` is metres per mesh unit. Returns ``eps`` (relative
    permittivity per node), ``pec`` (bool mask, or None when there is no
    conductor) and ``metals`` ({name: node mask} for each metal present).

    ``coatings`` maps a part name to ``(material, thickness in metres)``: a
    shell of that material over the part's surface (``CoatedIndex`` says
    where it is painted), so a core coated in gold lays out as a gold shell
    around the core's material.
    """
    if boundary not in fdtd.BOUNDARIES:
        raise VoxelError(f"boundary {boundary!r} must be one of "
                         f"{', '.join(fdtd.BOUNDARIES)}.")
    names = list(names)
    table = _material_table(names, materials)
    finder, coats = coated_index(names, parts, materials, coatings,
                                 boundary=boundary, length_scale=length_scale)
    # index 0 (outside) is a conductor for a cavity and vacuum for open space
    outside = ("pec",) if boundary == "metal" else ("eps", 1.0)
    kinds = [outside] + table + [_kind(f"the coating of part {names[p]!r}", m)
                                 for p, m, _t in coats]
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
