"""Dependency-free geometry for multi-material cavities.

WHY THIS FILE EXISTS. Photonics feature 1 puts several dielectric parts in one
PEC cavity, and the mesher has to answer three geometric questions about them:

* **Which part is a point in?** -- to find a region seed, and to check that each
  tetrahedron landed in the region it was asked for.
* **Do two parts overlap?** -- an overlap means the owner's parts disagree about
  what occupies that space, and no single ``eps_r`` can describe it.
* **Do two parts touch?** -- two solids sharing a boundary surface have no
  measure-zero gap for a region attribute to fill, and TetGen is free to give
  the shared tets either material.

THE OBVIOUS IMPLEMENTATIONS ARE NOT AVAILABLE. ``trimesh.contains`` needs
``rtree`` or ``embree``; ``trimesh.intersection`` needs ``manifold3d``. Measured
in the real solver image: all three are ABSENT, and every one of those calls
RAISES there. A check written against them passes in the venv -- which has them
-- and fails in the deployed image, which is the crashloop shape this
repository has already been burned by.

So the containment test is written here, on numpy alone: the exact winding
number by solid angle. It is the standard robust formulation, and it is exact
rather than approximate (no ray direction to choose, no facet to miss), which
matters because this decides which material a region gets.

IMPORTS GO ONE WAY, as in :mod:`mse_common`: this module imports only numpy, and
the solver modules import from it. It is imported by the web image too -- the
cheap structural checks at upload time run there -- so nothing heavy may come
here.
"""

from __future__ import annotations

import numpy as np

from mse_common import MeshingError

#: The temporary arrays in the winding and on-surface tests all have shape
#: ``(n_points_in_chunk, n_faces, 3)``, so their size is the PRODUCT of the two
#: counts -- not the point count. That product is what has to be bounded, and a
#: fixed number of POINTS does not bound it: 4000 points against a 5120-face
#: surface wants 469 MiB in a single allocation and dies under a 4 GiB address
#: space, while the same 4000 points against a 12-face box is nothing.
#:
#: So the budget is expressed in PAIRS. At 8 bytes per float64 element and a
#: handful of concurrent temporaries per pair, ~2e6 pairs keeps one chunk in the
#: tens of megabytes, which is safe in a 1-CPU web pod and in the solver.
WINDING_PAIR_BUDGET = 2_000_000

#: The smallest chunk worth taking: one point at a time. A surface with more
#: faces than the whole budget still gets tested, one point per call, rather
#: than being refused or silently truncated.
WINDING_MIN_CHUNK = 1


def _chunk_for(n_faces: int, budget: int = WINDING_PAIR_BUDGET) -> int:
    """Points per chunk that keep ``(points, faces, 3)`` temporaries bounded.

    ``n_faces`` of 0 cannot happen (a surface with no faces is refused before
    this), but returning the minimum rather than dividing by zero keeps the
    failure at the caller's own check instead of here.
    """
    if n_faces <= 0:
        return WINDING_MIN_CHUNK
    return max(WINDING_MIN_CHUNK, int(budget) // int(n_faces))


def _winding_numbers(faces_vertices: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact winding number of ``points`` about the closed surface.

    ``faces_vertices`` is ``(n_faces, 3, 3)`` -- the three corner positions of
    every triangle -- and ``points`` is ``(n_points, 3)``. Returns
    ``(n_points,)`` float64 winding numbers.

    The formula is the solid angle subtended by each triangle: for a triangle
    with corner vectors ``a``, ``b``, ``c`` seen from the point,

        Omega = 2 * atan2(a . (b x c),
                          |a||b||c| + (a.b)|c| + (b.c)|a| + (c.a)|b|)

    and the winding number is the total divided by ``4*pi``. For a closed
    surface it is an integer: 1 inside, 0 outside. ``atan2`` is used rather than
    ``arccos`` because it is defined at 0 and pi, where the arccos form loses
    precision exactly at the grazing rays.

    A point exactly ON the surface is the degenerate case; its winding number is
    ill-defined and the result is not to be trusted. Callers that care pass
    points they know are clear of the surface, or treat the surface case with
    :func:`touching_parts`, which is written for it.
    """
    pts = np.asarray(points, dtype=np.float64)
    tri = np.asarray(faces_vertices, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must be (n, 3), got {pts.shape}")
    if tri.ndim != 3 or tri.shape[1:] != (3, 3):
        raise ValueError(f"faces_vertices must be (n, 3, 3), got {tri.shape}")

    # (n_points, n_faces, 3): the corner vectors from each point. This is the
    # one large temporary; the caller bounds n_points.
    rel = tri[None, :, :] - pts[:, None, None, :]
    a = rel[:, :, 0, :]
    b = rel[:, :, 1, :]
    c = rel[:, :, 2, :]

    la = np.linalg.norm(a, axis=2)
    lb = np.linalg.norm(b, axis=2)
    lc = np.linalg.norm(c, axis=2)

    num = np.einsum("...i,...i->...", a, np.cross(b, c))
    den = (
        la * lb * lc
        + np.einsum("...i,...i->...", a, b) * lc
        + np.einsum("...i,...i->...", b, c) * la
        + np.einsum("...i,...i->...", c, a) * lb
    )
    # Each triangle contributes at most 2*pi; sum, then normalise by 4*pi.
    omega = 2.0 * np.arctan2(num, den)
    return omega.sum(axis=1) / (4.0 * np.pi)


def of_faces(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``(n_faces, 3, 3)`` corner positions from a vertex array and face index."""
    v = np.asarray(vertices, dtype=np.float64)
    f = np.asarray(faces, dtype=np.int64)
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must be (n, 3), got {f.shape}")
    return v[f]


def point_in_mesh(
    surface,
    points: np.ndarray,
    *,
    chunk: int | None = None,
    on_surface_eps: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(inside, on_surface)`` for ``points`` against a closed ``surface``.

    ``inside`` is a bool array. ``on_surface`` marks points whose distance to
    the surface is at or below ``on_surface_eps`` -- those are the ones whose
    winding number is not trustworthy, so the caller is told which they are
    rather than being handed a coin flip.

    ``on_surface_eps`` defaults to a scale-relative value: ``1e-9`` times the
    surface's own longest bounding-box side. A fixed absolute epsilon cannot be
    right for both a 1.55 um cavity and a 100 mm one, and the failure it causes
    is silent -- a point on the interface counted as strictly inside, which is
    exactly the miscount that made a correctly-conforming mesh look broken.

    ``chunk`` is points per call. Left None it is derived from the surface's
    face count so the ``(chunk, n_faces, 3)`` temporaries stay bounded; pass it
    explicitly only to compare a chunked run against an unchunked one, and note
    that an explicit value is honoured even when it is too big to allocate --
    which is how the bounded/unbounded cost is asserted in the tests.
    """
    tris = of_faces(surface.vertices, surface.faces)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    step = int(chunk) if chunk is not None else _chunk_for(tris.shape[0])
    step = max(step, WINDING_MIN_CHUNK)

    if on_surface_eps is None:
        bounds = np.asarray(surface.bounds, dtype=np.float64)
        extent = float(np.max(bounds[1] - bounds[0]))
        on_surface_eps = max(extent, 1.0) * 1e-9

    inside = np.zeros(pts.shape[0], dtype=bool)
    for start in range(0, pts.shape[0], step):
        w = _winding_numbers(tris, pts[start:start + step])
        inside[start:start + step] = np.abs(w) > 0.5

    on_surface = _points_on_surface(
        tris, pts, float(on_surface_eps), chunk=step
    )
    return inside, on_surface


def _points_on_surface(
    tris: np.ndarray, pts: np.ndarray, eps: float, chunk: int | None = None
) -> np.ndarray:
    """Which points are within ``eps`` of the surface -- the TRIANGLES, not
    their planes.

    The same chunked shape as the winding test, and for the same reason: this is
    also O(points x faces).

    DISTANCE TO THE TRIANGLE, DELIBERATELY. An earlier version measured the
    distance to each triangle's INFINITE PLANE, which is far cheaper and quite
    wrong: the six planes of a box are not the box. Measured on two unit boxes
    0.5 apart, the plane test reported all 8 of one box's vertices as lying ON
    the other, because a vertex at ``(0.5, -0.5, -0.5)`` sits exactly on the
    plane of the other box's ``y = -0.5`` face -- a triangle on the far side of
    a gap. Every pair of axis-aligned parts has faces in the same planes, so the
    plane test would call almost every pair "touching" and refuse it.

    The correct test decomposes the distance to a triangle into the point's
    offset along the normal and its projection inside or outside the triangle's
    edges, taking the closest feature (face, edge, or vertex). It is more work
    per (point, triangle) pair but it is the only one of the two that answers
    the question asked.

    A degenerate triangle has no plane; it is skipped, since its area is zero
    and no point can touch it.
    """
    if pts.shape[0] == 0:
        return np.zeros(0, dtype=bool)

    v0 = tris[:, 0]
    e1 = tris[:, 1] - v0
    e2 = tris[:, 2] - v0
    normals = np.cross(e1, e2)
    lengths = np.linalg.norm(normals, axis=1)
    safe = lengths > 0
    if not safe.any():
        return np.zeros(pts.shape[0], dtype=bool)
    # ``normals`` is a fresh array from ``np.cross`` (not a view of a caller's
    # data), so normalising in place is safe.
    normals[safe] /= lengths[safe, None]

    out = np.zeros(pts.shape[0], dtype=bool)
    step = int(chunk) if chunk is not None else _chunk_for(tris.shape[0])
    step = max(step, WINDING_MIN_CHUNK)
    for start in range(0, pts.shape[0], step):
        stop = start + step
        block = pts[start:stop]
        # (n_points, n_faces, 3) -- see the note in the winding function about
        # the ``[:, None, :]`` indexing; a spurious extra axis silently changes
        # which axis a later reduction collapses.
        w = block[:, None, :] - v0[None, :, :]
        d1 = np.einsum("...i,...i->...", w, e1[None, :, :])
        d2 = np.einsum("...i,...i->...", w, e2[None, :, :])
        a11 = np.einsum("...i,...i->...", e1[None, :, :], e1[None, :, :])
        a12 = np.einsum("...i,...i->...", e1[None, :, :], e2[None, :, :])
        a22 = np.einsum("...i,...i->...", e2[None, :, :], e2[None, :, :])
        det = a11 * a22 - a12 * a12
        det = np.where(np.abs(det) < 1e-300, 1.0, det)   # degenerate triangles

        # Barycentric coordinates of the projection onto the triangle's plane.
        u = (a22 * d1 - a12 * d2) / det
        v = (a11 * d2 - a12 * d1) / det

        # Face region: the projection is inside the triangle.
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)

        # Otherwise the closest point is on an edge or at a vertex. Each case is
        # the distance to the corresponding segment, clamped to its endpoints.
        def seg(p0, p1, q):
            """Distance from every (point, face) pair to one triangle edge.

            ``q`` is ``(n_points, n_faces, 3)``; ``p0``/``p1`` are
            ``(n_faces, 3)`` and need a ``None`` axis to broadcast against it.
            Getting this backwards raises on the shapes, but the version that
            raises is the lucky one: a broadcast that happened to succeed along
            the wrong axis would return a plausible distance for the wrong pair.
            """
            ab = p1 - p0                       # (n_faces, 3)
            ap = q - p0[None, :, :]            # (n_points, n_faces, 3)
            ab_b = ab[None, :, :]              # (1, n_faces, 3)
            ab2 = np.einsum("...i,...i->...", ab_b, ab_b)   # (1, n_faces)
            ab2 = np.where(ab2 > 0, ab2, 1.0)               # zero-length edge
            t = np.einsum("...i,...i->...", ap, ab_b) / ab2
            t = np.clip(t, 0.0, 1.0)
            proj = p0[None, :, :] + t[..., None] * ab_b
            return np.linalg.norm(q - proj, axis=2)

        d_face = np.abs(np.einsum("...i,...i->...", w, normals[None, :, :]))
        # ``q`` is the point array broadcast to one row per (point, face) pair,
        # NOT the raw ``block``: ``seg`` distances are per pair, and handing it
        # the un-broadcast array makes the subtraction fail (or, worse, succeed
        # against the wrong axis).
        q = block[:, None, :]
        d_e01 = seg(v0, tris[:, 1], q)
        d_e02 = seg(v0, tris[:, 2], q)
        d_e12 = seg(tris[:, 1], tris[:, 2], q)

        dist = np.where(inside, d_face, np.minimum(np.minimum(d_e01, d_e02), d_e12))
        out[start:stop] = (dist <= eps).any(axis=1)
    return out


def part_bounds(part) -> tuple[np.ndarray, np.ndarray]:
    """``(lo, hi)`` bounding box of a part, as float64."""
    b = np.asarray(part.bounds, dtype=np.float64)
    return b[0], b[1]


def bounds_intersect(lo_a, hi_a, lo_b, hi_b, tol: float = 0.0) -> bool:
    """Do two axis-aligned boxes overlap (or come within ``tol``)?

    This is the CHEAP pre-filter that lets the upload request avoid the O(n*m)
    containment work entirely: two parts whose bounding boxes are disjoint can
    neither overlap nor touch, so no vertex of either can be on the other's
    surface. Only the pairs that survive this are ever tested point-by-point.

    ``tol`` is in the same length units as the boxes (metres in the app, since
    the surface is scaled before this is called). It gives a margin so a pair
    that is merely CLOSE is still tested rather than being filtered out on a
    rounding error -- the filter must be conservative, because the thing it is
    filtering for is a refusal.
    """
    lo_a = np.asarray(lo_a, dtype=np.float64)
    hi_a = np.asarray(hi_a, dtype=np.float64)
    lo_b = np.asarray(lo_b, dtype=np.float64)
    hi_b = np.asarray(hi_b, dtype=np.float64)
    return bool(
        np.all(lo_a <= hi_b + tol) and np.all(lo_b <= hi_a + tol)
    )


def parts_may_interact(parts, tol: float = 0.0) -> list[tuple[int, int]]:
    """Index pairs whose bounding boxes overlap -- the candidates to test.

    Returns ``(i, j)`` with ``i < j``. Everything not in this list is proven
    apart, so the expensive checks only ever see this list.
    """
    boxes = [part_bounds(p) for p in parts]
    pairs: list[tuple[int, int]] = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if bounds_intersect(*boxes[i], *boxes[j], tol=tol):
                pairs.append((i, j))
    return pairs


def overlap_vertices(part_a, part_b, *, tol: float | None = None):
    """Vertices of each part that lie strictly inside the other.

    Returns ``(a_inside_b, b_inside_a)`` -- two index arrays. A non-empty result
    proves an interpenetration; an EMPTY result proves nothing.

    THAT ASYMMETRY IS THE IMPORTANT PART. This is a vertex test, and vertices
    are not enough to decide overlap in general. Measured: two identical 2x2x2
    boxes offset by 1.0 in x overlap by half a box, and NOT ONE vertex of either
    is strictly inside the other -- every one lands exactly ON the other's
    surface. A vertex test therefore cannot see that overlap at all, and
    reporting its absence would be the could-not-ask shape: a check that ran and
    whose empty answer reads as "nothing there".

    So this is a cheap POSITIVE detector and a fast path, not the overlap
    decision. The decision belongs to TetGen's ``docheck``, which sees the whole
    triangulated surface rather than its vertices -- see
    :func:`mse_parts.check_parts`. This function exists because it is free when
    the boxes already overlap and it names the offending vertices, which makes a
    refusal message concrete.

    A vertex ON the other surface is excluded here and reported by
    :func:`touching_vertices` instead, so the two refusals name different
    problems.

    The tolerance is scale-relative for the same reason as
    :func:`point_in_mesh`'s.
    """
    a_in_b, a_on_b = point_in_mesh(part_b, np.asarray(part_a.vertices), on_surface_eps=tol)
    b_in_a, b_on_a = point_in_mesh(part_a, np.asarray(part_b.vertices), on_surface_eps=tol)
    return np.flatnonzero(a_in_b & ~a_on_b), np.flatnonzero(b_in_a & ~b_on_a)


def part_contains(outer, inner, *, tol: float | None = None) -> bool:
    """True iff every vertex of ``inner`` is inside (or on) ``outer``.

    This is what distinguishes CONTAINMENT from OVERLAP, which are otherwise the
    same measurement: a nested part's vertices are all inside its parent, so
    :func:`overlap_vertices` reports them, and a caller that read that as
    "these parts collide" would refuse exactly the feature this row is for.

    The test is one-sided on purpose. ``inner`` inside ``outer`` is containment
    (a material inside the cavity wall); ``outer`` stuck inside ``inner`` as well
    would be each part inside the other, which is a genuine contradiction and is
    caught by the caller comparing both directions.

    Vertices are the criterion, so the same caveat as :func:`overlap_vertices`
    applies: this can prove containment for the cases that matter (a part whose
    vertices are all within a parent) and cannot prove its absence in exotic
    ones. The mesher's ``docheck`` remains the exact instrument.
    """
    verts = np.asarray(inner.vertices, dtype=np.float64)
    inside, on = point_in_mesh(outer, verts, on_surface_eps=tol)
    return bool(np.all(inside | on))


def touching_vertices(part_a, part_b, *, tol: float | None = None):
    """Vertices of either part lying ON the other's surface.

    Returns ``(a_on_b, b_on_a)`` -- two index arrays. TetGen's ``docheck``
    catches overlaps but accepts exactly-touching parts (measured: two boxes
    sharing a face plane tetrahedralize to 12 tets with no error), so this is
    the one case the mesher cannot refuse on its own. Without it, the shared
    tets get whichever region the attribute pass happens to assign, and the
    owner sees a material they did not ask for with nothing failing.
    """
    _, a_on_b = point_in_mesh(part_b, np.asarray(part_a.vertices), on_surface_eps=tol)
    _, b_on_a = point_in_mesh(part_a, np.asarray(part_b.vertices), on_surface_eps=tol)
    return np.flatnonzero(a_on_b), np.flatnonzero(b_on_a)


def formation_error(message: str, part_name: str) -> MeshingError:
    """A refusal that NAMES the part it is about.

    Every geometric refusal in this feature goes through here. A message that
    says "two parts overlap" without saying which two leaves the owner with a
    file they cannot fix, which is the could-not-ask shape: the check ran, and
    the answer it produced is unusable.
    """
    return MeshingError(f"{message} (part {part_name!r})")


# --------------------------------------------------------------------------
# Roundness: may this device use the axisymmetric (r, z) form? (task 2.6)
# --------------------------------------------------------------------------
#
# The axisymmetric form solves a 2-D (r, z) mesh weighted by 2*pi*r, which is
# only the same device if every cross-section is a circle about one axis. This
# measures that from the uploaded surface, on numpy alone (the web image runs it
# at upload, and has no rtree / shapely / networkx).
#
# HOW. Find candidate axes, slice the surface with planes across each one, and
# for every closed loop in every slice take
#
#   circumradius = the farthest loop point from the axis,
#   inradius     = the nearest point of the loop's SEGMENTS to the axis,
#
# so a faceted circle of N sides measures 1/cos(pi/N), not 1. The device's
# figure is the worst loop of the worst slice; the axis is the candidate about
# which that worst figure is smallest.
#
# THE CANDIDATES. Any symmetric tensor of a body of revolution has the symmetry
# axis as an eigenvector, so the eigenvectors of two such tensors -- the
# area-weighted normal tensor and the surface second moment -- are tried. Two,
# because each has one aspect ratio at which its eigenvalues coincide and its
# eigenvectors stop meaning anything (a cylinder with L = 2R for the normal
# tensor), and the two coincide at different aspect ratios. The line passes
# through the area-weighted surface centroid, which lies on the axis of any
# body of revolution.
#
# PER LOOP, NOT PER DEVICE. A tapered pillar or a core inside a tube has round
# loops of different radii; comparing the biggest loop with the smallest would
# refuse them. Each loop is judged on its own.

#: The axisymmetric form's limit on circumradius/inradius - 1 (design.md, "Two
#: forms": "round to within 2 % (circumradius vs inradius)"). One definition:
#: the solver and the upload refusal both read it from here.
ROUNDNESS_TOLERANCE = 0.02

#: Planes per candidate axis. Odd, so a symmetric device is sliced at its middle.
ROUNDNESS_SLICES = 9


class NotRoundError(ValueError):
    """The device's cross-sections are not round enough for the (r, z) form."""


def _weld(vertices: np.ndarray, faces: np.ndarray):
    """Merge coincident vertices so slice segments can be chained by edge."""
    span = float(np.ptp(vertices, axis=0).max()) or 1.0
    key = np.round(vertices / (span * 1e-9)).astype(np.int64)
    _, first, inverse = np.unique(key, axis=0, return_index=True, return_inverse=True)
    return vertices[first], inverse.reshape(-1)[faces]


def _candidate_axes(vertices: np.ndarray, faces: np.ndarray):
    """The surface centroid and the candidate axis directions (unit vectors)."""
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    area = 0.5 * np.linalg.norm(cross, axis=1)
    total = area.sum()
    if total <= 0.0:
        raise NotRoundError("The surface has no area, so its roundness cannot be measured.")
    centroid = (area[:, None] * tri.mean(axis=1)).sum(axis=0) / total
    normals = cross / np.where(area > 0, 2.0 * area, 1.0)[:, None]
    normal_tensor = np.einsum("i,ij,ik->jk", area, normals, normals)
    # Exact second moment of each triangle about the centroid:
    # A/12 * (sum v v^T + (sum v)(sum v)^T).
    rel = tri - centroid
    s = rel.sum(axis=1)
    moment = (np.einsum("i,ikj,ikl->jl", area, rel, rel)
              + np.einsum("i,ij,ik->jk", area, s, s)) / 12.0
    axes = []
    for tensor in (normal_tensor, moment):
        for vec in np.linalg.eigh(tensor)[1].T:
            if all(abs(float(vec @ a)) < 0.999 for a in axes):
                axes.append(vec / np.linalg.norm(vec))
    return centroid, axes


def _slice_heights(t: np.ndarray, n: int) -> np.ndarray:
    """``n`` heights across ``t``'s range, none on a vertex (so no degenerate cut)."""
    lo, hi = float(t.min()), float(t.max())
    levels = np.unique(t)
    eps = 1e-7 * (hi - lo)
    heights = lo + (np.arange(n) + 0.5) / n * (hi - lo)
    for k, h in enumerate(heights):
        if np.abs(levels - h).min() < eps:
            above = levels[levels > h + eps]
            below = levels[levels < h - eps]
            nxt = above[0] if above.size else hi
            prv = below[-1] if below.size else lo
            heights[k] = 0.5 * (h + nxt) if nxt - h >= h - prv else 0.5 * (h + prv)
    return heights


def _loops_at(vertices, faces, t, height):
    """The closed loops of the cut at ``t == height``, as arrays of 3-D points."""
    s = t[faces] - height
    pos = s > 0
    crossing = pos.any(axis=1) & ~pos.all(axis=1)
    point_of = {}
    neighbours: dict = {}
    for f, sf in zip(faces[crossing], s[crossing]):
        ends = []
        for a, b in ((0, 1), (1, 2), (2, 0)):
            if (sf[a] > 0) != (sf[b] > 0):
                i, j = int(f[a]), int(f[b])
                key = (min(i, j), max(i, j))
                if key not in point_of:
                    w = sf[a] / (sf[a] - sf[b])
                    point_of[key] = vertices[i] + w * (vertices[j] - vertices[i])
                ends.append(key)
        p, q = ends
        neighbours.setdefault(p, []).append(q)
        neighbours.setdefault(q, []).append(p)
    if any(len(v) != 2 for v in neighbours.values()):
        raise NotRoundError(
            "The surface is not a closed manifold where it was cut, so its "
            "roundness cannot be measured.")
    loops, seen = [], set()
    for start in neighbours:
        if start in seen:
            continue
        loop, prev, cur = [], None, start
        while cur not in seen:
            seen.add(cur)
            loop.append(point_of[cur])
            a, b = neighbours[cur]
            prev, cur = cur, (b if a == prev else a)
        loops.append(np.array(loop))
    return loops


def _loop_radii(points: np.ndarray, point_on_axis, direction):
    """(circumradius, inradius) of one closed loop about the axis line."""
    rel = points - point_on_axis
    radial = rel - np.outer(rel @ direction, direction)
    a = radial
    b = np.roll(radial, -1, axis=0)
    ab = b - a
    denom = np.einsum("ij,ij->i", ab, ab)
    w = np.clip(-np.einsum("ij,ij->i", a, ab) / np.where(denom > 0, denom, 1.0), 0.0, 1.0)
    nearest = a + w[:, None] * ab
    return (float(np.linalg.norm(radial, axis=1).max()),
            float(np.linalg.norm(nearest, axis=1).min()))


def roundness_about(vertices, faces, point_on_axis, direction,
                    n_slices: int = ROUNDNESS_SLICES) -> dict:
    """The worst loop's circumradius, inradius and ratio about one given axis."""
    vertices, faces = _weld(np.asarray(vertices, float), np.asarray(faces, np.int64))
    direction = np.asarray(direction, float) / np.linalg.norm(direction)
    point_on_axis = np.asarray(point_on_axis, float)
    t = (vertices - point_on_axis) @ direction
    worst = None
    for height in _slice_heights(t, n_slices):
        for loop in _loops_at(vertices, faces, t, height):
            big, small = _loop_radii(loop, point_on_axis, direction)
            ratio = big / small if small > 0 else float("inf")
            if worst is None or ratio > worst["ratio"]:
                worst = {"circumradius": big, "inradius": small, "ratio": ratio}
    worst.update(axis_point=point_on_axis, axis_direction=direction)
    return worst


def measure_roundness(vertices, faces, n_slices: int = ROUNDNESS_SLICES) -> dict:
    """Roundness of a closed surface about its detected axis.

    Returns ``circumradius``, ``inradius`` (both in the surface's own units),
    ``ratio`` (circumradius / inradius of the worst loop of the worst slice),
    ``axis_point`` and ``axis_direction``.
    """
    vertices, faces = _weld(np.asarray(vertices, float), np.asarray(faces, np.int64))
    centroid, axes = _candidate_axes(vertices, faces)
    results = [roundness_about(vertices, faces, centroid, a, n_slices) for a in axes]
    return min(results, key=lambda r: r["ratio"])


def roundness_refusal(measured: dict, units: str,
                      tolerance: float = ROUNDNESS_TOLERANCE) -> str | None:
    """The plain-English refusal for ``measured``, or ``None`` when it is round."""
    ratio = measured["ratio"]
    if ratio <= 1.0 + tolerance:
        return None
    return (
        "This device is not round enough for the axisymmetric (r, z) form. "
        f"Measured about its axis, its cross-sections have a circumradius of "
        f"{measured['circumradius']:.4g} {units} and an inradius of "
        f"{measured['inradius']:.4g} {units}: the circumradius is "
        f"{100.0 * (ratio - 1.0):.2f}% larger than the inradius, and the "
        f"axisymmetric form allows at most {100.0 * tolerance:g}%. "
        "Use the 3-D form for this device instead."
    )


def require_round(vertices, faces, units: str,
                  tolerance: float = ROUNDNESS_TOLERANCE) -> dict:
    """``measure_roundness``, raising :class:`NotRoundError` past ``tolerance``."""
    measured = measure_roundness(vertices, faces)
    message = roundness_refusal(measured, units, tolerance)
    if message:
        raise NotRoundError(message)
    return measured
