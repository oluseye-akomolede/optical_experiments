"""Sampling eigenvectors onto a regular grid and writing the result files.

The web app in task 2 renders ``mode_XX.bin`` directly, so the layout is part of
the contract and must not drift:

* ``mode_XX.bin`` -- ``float32``, C order, shape ``[nz][ny][nx][3]`` with the
  three components ``(Ex, Ey, Ez)`` last. ``NaN`` where the sample point falls
  outside the object, so the viewer can mask it out.
* ``mode_XX.vtu`` -- the same samples as a ParaView unstructured grid of points.

Each mode is normalised so that ``max|E| == 1``.
"""

from __future__ import annotations

import numpy as np

#: default number of samples along the longest bounding-box axis
DEFAULT_GRID = 64

#: Samples per locating chunk. This is what sets the locator's peak allocation
#: -- a property of the grid and the local mesh density, never of ``n_tets``.
_LOCATE_CHUNK = 20_000

#: Radius of the candidate ball, as a multiple of the mesh's largest tet vertex
#: radius ``r_max``, taken around the sample point ITSELF. The tets whose
#: centroids fall in that ball are the candidates.
#:
#: The factor is 1.0 because the bound is exact, not because 1.0 tested well. If
#: ``p`` is inside a tet ``S`` then ``p`` is a convex combination of S's four
#: vertices, so ``||p - c_S|| <= max_i ||a_i - c_S|| = R_S``, where ``R_S`` is
#: S's largest vertex radius. Since ``r_max`` is that maximum over the mesh,
#: ``||p - c_S|| <= R_S <= r_max`` and a ball of radius ``r_max`` around ``p``
#: always contains the containing tet's centroid. Measured on a TetGen mesh,
#: ``||p - c_S|| / R_S`` peaked at 0.9393 over 4562 interior samples -- under 1,
#: as the bound says.
#:
#: Why this rule and not the alternatives, all measured on a 7070-tet TetGen
#: mesh with a 64^3 grid (262144 points):
#:
#:   rule                                        candidates/point     peak
#:   ball of r_max around the point (this one)                193   0.40 GB
#:   ball of r_max around the sample's nearest vertex         345   0.72 GB
#:   naive (n_samples x n_tets x 3)                          7070  44.48 GB
#:
#: A nearest-k search that expands k is cheapest in the common case (the
#: containing centroid's rank is 0 at the median) but has no cheap termination
#: proof. Every candidate bound tried for "no containing tet is left"
#: (`3 r_max` mesh-wide, the nearest centroid's own vertex radius, and the
#: distance to the nearest mesh vertex) was either uselessly loose or simply
#: wrong -- the vertex-based one dropped real hits at three mesh sizes. A single
#: ball needs no loop and no proof-by-termination at all.
#:
#: The cost does grow with the mesh, because ``r_max`` grows relative to the
#: mean cell as a mesh refines; this is not a constant. It is a 111x reduction
#: from the naive array, which is what the Job's 14 Gi limit needs.
#:
#: ``Delaunay.find_simplex`` was measured and rejected. It is O(1) memory and
#: agreed with the TetGen tets on all 24000 samples of a convex box, but on a
#: non-convex L-shape it called ALL 4299 samples in the notch "inside" when the
#: solid has no material there -- it would render a field inside every
#: concavity the owner uploads, silently. The Delaunay hull bridges concavities;
#: the tets do not.
_LOCATE_RADIUS_FACTOR = 1.0


class SamplingError(Exception):
    """Sampling could not be done (no points, empty mesh, ...)."""


def grid_shape(points: np.ndarray, n_longest: int = DEFAULT_GRID):
    """Cell counts along (nx, ny, nz) scaled so the longest axis gets ``n_longest``."""
    extent = np.asarray(points).max(axis=0) - np.asarray(points).min(axis=0)
    longest = float(np.max(extent))
    if longest <= 0:
        raise SamplingError("mesh has zero size")

    ratios = extent / longest
    shape = np.maximum(np.round(ratios * n_longest).astype(int), 2)
    return tuple(int(s) for s in shape)


def sample_grid(points: np.ndarray, n_longest: int = DEFAULT_GRID, padding: float = 0.02):
    """Build the sampling grid covering the bounding box (with a little padding).

    Padding matters for the viewer: samples just outside the surface are needed
    so the slice planes are not clipped exactly at the object's extent.
    """
    points = np.asarray(points, dtype=float)
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    extent = hi - lo
    longest = float(np.max(extent))
    if longest <= 0:
        raise SamplingError("mesh has zero size")

    lo = lo - padding * extent
    hi = hi + padding * extent

    nx, ny, nz = grid_shape(points, n_longest)
    xs = np.linspace(lo[0], hi[0], nx)
    ys = np.linspace(lo[1], hi[1], ny)
    zs = np.linspace(lo[2], hi[2], nz)

    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")   # (nz, ny, nx) each
    samples = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
    return samples, (nx, ny, nz), lo, (hi - lo) / np.array([nx - 1, ny - 1, nz - 1])


#: barycentric coordinates are clamped at this tolerance: a sample exactly on a
#: shared face must be counted as inside, and floating-point inverse of a
#: skewed tet is not exact.
_BARY_TOL = 1e-9


def _tet_geometry(points: np.ndarray, tets: np.ndarray):
    """Edge matrix, inverse and vertex-0 for every tet.

    ``J[:, :, i]`` is the i-th edge vector (node i+1 minus node 0), so
    ``J^-1 @ (x - v0)`` gives the barycentric coords of nodes 1..3. Returns
    ``(v0, inv_J, det_J)``; a singular tet gets a zero inverse and is skipped.
    """
    n_tets = tets.shape[0]
    xyz = points[tets]                                    # (n_tets, 4, 3)
    v0 = xyz[:, 0, :]
    J = np.stack(
        [xyz[:, 1, :] - v0, xyz[:, 2, :] - v0, xyz[:, 3, :] - v0], axis=2
    )
    det_J = np.linalg.det(J)
    ok = np.abs(det_J) > 1e-30
    inv_J = np.zeros_like(J)
    inv_J[ok] = np.linalg.inv(J[ok])                      # (n_tets, 3, 3)
    return v0, inv_J, n_tets, ok


def barycentric(points: np.ndarray, tets: np.ndarray, which: np.ndarray,
                samples: np.ndarray, geometry=None):
    """Barycentric coords of each sample in its paired tet.

    ``which[i]`` is the tet for ``samples[i]``; returns ``(len(samples), 4)``.
    Uses the SAME convention as everywhere else in this module: x = v0 + J @ b
    with J's columns the edge vectors, so b = J^-1 (x - v0).
    """
    v0, inv_J, _, _ = geometry if geometry is not None else _tet_geometry(
        points, tets
    )
    d = samples - v0[which]                               # (m, 3)
    b = np.einsum("mij,mj->mi", inv_J[which], d)          # (m, 3)
    return np.concatenate([1.0 - b.sum(axis=1, keepdims=True), b], axis=1)


def brute_force_mask(points: np.ndarray, tets: np.ndarray,
                     samples: np.ndarray, chunk: int = 20_000) -> np.ndarray:
    """Which samples are inside *any* tet, by testing every tet.

    O(n_samples x n_tets) time and memory -- this is the implementation that
    OOM-killed the Job, kept ONLY as the reference the fast path is tested
    against. Nothing in the production path calls it.

    Chunked so a test can use it on a small mesh without allocating the
    (n_samples, n_tets) intermediate all at once; the memory bound test asserts
    it is still the naive shape.
    """
    v0, inv_J, n_tets, _ = _tet_geometry(points, tets)
    inside = np.zeros(samples.shape[0], dtype=bool)
    for start in range(0, samples.shape[0], chunk):
        stop = min(start + chunk, samples.shape[0])
        s = samples[start:stop]                           # (m, 3)
        d = s[:, None, :] - v0[None, :, :]                # (m, n_tets, 3)
        b1 = np.einsum("mtj,tj->mt", d, inv_J[:, 0, :])
        b2 = np.einsum("mtj,tj->mt", d, inv_J[:, 1, :])
        b3 = np.einsum("mtj,tj->mt", d, inv_J[:, 2, :])
        b0 = 1.0 - b1 - b2 - b3
        hit = ((b0 >= -_BARY_TOL) & (b1 >= -_BARY_TOL)
               & (b2 >= -_BARY_TOL) & (b3 >= -_BARY_TOL))
        inside[start:stop] = hit.any(axis=1)
    return inside


def _candidate_pairs(chunk: np.ndarray, centroid_tree, radius: float):
    """``(owner, cand)`` -- the deduplicated (point, candidate tet) pairs.

    The single place the candidate set is built, so ``candidate_counts`` measures
    the same gathering ``locate_samples`` performs rather than a copy of it.
    ``owner`` indexes ``chunk``, ``cand`` indexes tets, and the pairs come out
    ascending by (owner, cand) -- which is what makes the first-hit tie-break in
    ``locate_samples`` agree with brute force on shared faces.
    """
    balls = centroid_tree.query_ball_point(chunk, radius)
    lens = np.fromiter((len(b) for b in balls), dtype=np.int64,
                       count=chunk.shape[0])
    if not lens.sum():
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    owner = np.repeat(np.arange(chunk.shape[0]), lens)
    cand = np.concatenate([np.asarray(b, dtype=np.int64) for b in balls if len(b)])
    return owner, cand


def candidate_counts(points: np.ndarray, tets: np.ndarray,
                     samples: np.ndarray) -> np.ndarray:
    """How many candidate tets ``locate_samples`` examines for each sample.

    Exists so the scaling property can be asserted on the real gathering code
    rather than on a reimplementation of it in the test -- a test that rebuilds
    the ball itself would keep passing after the implementation drifted away
    from it, which is how an earlier attempt at this test missed a real defect.
    """
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    samples = np.asarray(samples, dtype=float)
    if tets.size == 0 or samples.size == 0:
        return np.zeros(samples.shape[0], dtype=np.int64)

    from scipy.spatial import cKDTree

    centroids = points[tets].mean(axis=1)
    r_max = float(
        np.linalg.norm(points[tets] - centroids[:, None, :], axis=2).max()
    )
    tree = cKDTree(centroids)

    counts = np.zeros(samples.shape[0], dtype=np.int64)
    for start in range(0, samples.shape[0], _LOCATE_CHUNK):
        stop = min(start + _LOCATE_CHUNK, samples.shape[0])
        owner, _cand = _candidate_pairs(
            samples[start:stop], tree, _LOCATE_RADIUS_FACTOR * r_max
        )
        if owner.size:
            rows, per_row = np.unique(owner, return_counts=True)
            counts[start + rows] = per_row
    return counts


def locate_samples(points: np.ndarray, tets: np.ndarray, samples: np.ndarray):
    """Find the containing tet and barycentric coords for each sample, ONCE.

    This is the deploy fix. The old sampler rebuilt a ``(chunk, n_tets, 3)``
    difference array for every chunk and then again for every mode: at 10k tets
    and 262k samples that is ~15 GB, and it is why a Job that meshed and solved
    in 7 s then sat at its 14 Gi limit with 0 CPU until it was deleted.

    Each sample examines the tets whose centroids lie within ``r_max`` of it,
    where ``r_max`` is the largest tet vertex radius in the mesh. That is
    exhaustive, not heuristic: if ``p`` is inside a tet ``S`` then
    ``||p - c_S|| <= R_S <= r_max``, so ``c_S`` is always in the ball and ``S``
    is always a candidate. Nothing is ever dropped, which is what the agreement
    test against brute force pins. See ``_LOCATE_RADIUS_FACTOR`` for the
    measurements behind the rule and for why the cheaper-looking alternatives
    (vertex balls, expanding-k search, ``Delaunay.find_simplex``) were rejected.

    The result does not depend on the eigenvector, so callers locate once per
    job and reuse it for every mode.

    Returns ``(idx, tet_of, bary)`` where ``idx`` indexes ``samples``, ``tet_of``
    is the containing tet, and ``bary`` its 4 barycentric coords. Samples
    outside every tet are simply absent, so the inside mask is ``tet_id >= 0``
    over the full sample array.
    """
    samples = np.asarray(samples, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    if tets.size == 0 or samples.size == 0:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty((0, 4), dtype=float))

    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=float)
    v0_all, inv_J, _, ok = _tet_geometry(points, tets)
    centroids = points[tets].mean(axis=1)
    r_max = float(
        np.linalg.norm(points[tets] - centroids[:, None, :], axis=2).max()
    )
    radius = _LOCATE_RADIUS_FACTOR * r_max
    tree = cKDTree(centroids)

    idx_parts, tet_parts, bary_parts = [], [], []
    for start in range(0, samples.shape[0], _LOCATE_CHUNK):
        stop = min(start + _LOCATE_CHUNK, samples.shape[0])
        chunk = samples[start:stop]                       # (m, 3)
        owner, cand = _candidate_pairs(chunk, tree, radius)
        if cand.size == 0:
            continue

        diff = chunk[owner] - v0_all[cand]                # (C, 3)
        b = np.einsum("cij,cj->ci", inv_J[cand], diff)    # (C, 3)
        bary = np.concatenate([1.0 - b.sum(axis=1, keepdims=True), b], axis=1)
        tolerated = (bary >= -_BARY_TOL).all(axis=1) & ok[cand]

        hit_rows = np.nonzero(tolerated)[0]
        if hit_rows.size == 0:
            continue
        # first candidate that contains the point wins, exactly as
        # `argmax(inside, axis=1)` gives it in brute force
        pts_here = owner[hit_rows]
        first = np.concatenate([[True], pts_here[1:] != pts_here[:-1]])
        pick = hit_rows[first]
        idx_parts.append(pts_here[first] + start)
        tet_parts.append(cand[pick])
        bary_parts.append(bary[pick])

    if not idx_parts:
        return (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64),
                np.empty((0, 4), dtype=float))
    idx = np.concatenate(idx_parts)
    # chunks are ascending and each chunk's points come out ascending, but sort
    # anyway so the contract is "ascending sample index" without the caller
    # relying on chunking details
    order = np.argsort(idx, kind="stable")
    return (idx[order], np.concatenate(tet_parts)[order],
            np.concatenate(bary_parts)[order])


def sample_field(
    points: np.ndarray,
    tets: np.ndarray,
    edges: np.ndarray,
    tet_edges: np.ndarray,
    signs: np.ndarray,
    eigenvector: np.ndarray,
    n_longest: int = DEFAULT_GRID,
    free_edges: np.ndarray | None = None,
    n_edges: int | None = None,
    located=None,
    grid=None,
):
    """Evaluate one edge-element eigenvector on the regular grid.

    ``eigenvector`` holds one coefficient per edge in ``tet_edges``, which is
    the **full** edge numbering *after* PEC elimination has removed nothing --
    i.e. it is the reduced vector plus ``free_edges`` telling us where each
    reduced unknown lives in the full numbering. When ``free_edges`` is given,
    ``eigenvector`` is the reduced (free-only) vector and the PEC-constrained
    edges are taken as zero, which is exactly the boundary condition
    ``n x E = 0``.

    ``located``/``grid`` are optional precomputed results. Both are independent
    of the eigenvector, so a caller writing 20 modes locates once and reuses it
    -- locating is the expensive part and repeating it per mode is what made the
    Job exceed its memory limit. Omitting them is correct but recomputes.

    Returns ``(field, shape, origin, spacing)`` where ``field`` is
    ``(nz, ny, nx, 3)`` float32 with ``NaN`` outside the object.
    """
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    if grid is None:
        samples, shape, origin, spacing = sample_grid(points, n_longest)
    else:
        samples, shape, origin, spacing = grid

    n_total = int(n_edges) if n_edges is not None else int(tet_edges.max()) + 1
    if free_edges is None:
        full_coeffs = np.asarray(eigenvector, dtype=float)
        if full_coeffs.size != n_total:
            raise SamplingError(
                f"eigenvector has {full_coeffs.size} entries but the mesh has "
                f"{n_total} edges"
            )
    else:
        full_coeffs = np.zeros(n_total, dtype=float)
        full_coeffs[np.asarray(free_edges, dtype=np.int64)] = np.asarray(
            eigenvector, dtype=float
        )

    n_tets = tets.shape[0]
    # one geometry helper for the whole module, so the edge-vector convention
    # (J's COLUMNS are the edges, so b = J^-1 (x - v0)) is stated exactly once
    _, inv_J, n_tets, _ = _tet_geometry(points, tets)

    grad_ref = np.array(
        [-1.0, -1.0, -1.0,
          1.0,  0.0,  0.0,
          0.0,  1.0,  0.0,
          0.0,  0.0,  1.0], dtype=float
    ).reshape(4, 3)
    # grad(lambda) = J^-T @ grad_ref -- the TRANSPOSE of J, exactly as in
    # mse_fem. Getting this wrong is invisible on a right-angled reference tet
    # and would silently rotate the sampled field on a general one.
    grads = np.einsum("nji,aj->nai", inv_J, grad_ref)     # (n_tets, 4, 3)

    # local edge coefficients for each tet, with the global orientation sign
    # applied so they are expressed in the (min node -> max node) convention
    local_coeffs = np.zeros((n_tets, 6), dtype=float)
    idx = tet_edges.ravel()
    local_coeffs[:] = (full_coeffs[idx] * signs.ravel()).reshape(n_tets, 6)

    n_samples = samples.shape[0]
    field = np.zeros((n_samples, 3), dtype=float)
    hit_any = np.zeros(n_samples, dtype=bool)

    if located is None:
        located = locate_samples(points, tets, samples)
    idx, tet_of, lam = located

    if idx.size:
        hit_any[idx] = True
        # N_a = lambda_i grad lambda_j - lambda_j grad lambda_i, evaluated only
        # at the points that were actually located: O(n_inside), not
        # O(n_samples x n_tets).
        a_idx = np.array([0, 0, 0, 1, 1, 2])
        b_idx = np.array([1, 2, 3, 2, 3, 3])
        lj_ = grads[tet_of]                               # (h, 4, 3)
        li = lam[:, a_idx][:, :, None] * lj_[:, b_idx, :]
        ljv = lam[:, b_idx][:, :, None] * lj_[:, a_idx, :]
        basis = li - ljv                                  # (h, 6, 3)
        field[idx] = np.einsum("hc,hcd->hd", local_coeffs[tet_of], basis)

    field[~hit_any] = np.nan

    # normalise to max|E| = 1 over the sampled volume
    finite = np.isfinite(field).all(axis=-1)
    if finite.any():
        mag = np.linalg.norm(field[finite], axis=-1)
        peak = float(np.max(mag))
        if peak > 0:
            field = field / peak

    field = field.reshape(shape[2], shape[1], shape[0], 3)   # (nz, ny, nx, 3)
    return field.astype(np.float32), shape, origin, spacing


def write_vtu(path, field: np.ndarray, shape, origin, spacing):
    """Write the sampled field as a ParaView ``.vtu`` point cloud.

    ``path`` may be a filename or any binary file object, so callers can either
    write straight to disk or hand the bytes to object storage.

    Built with pyvista/VTK rather than by assembling the XML here. The format
    looks like text -- it is XML with base64 arrays -- but the appended-data
    framing (per-block lengths, per-block compression, and a base64 unit per
    block rather than one continuous stream) is where a hand-written writer goes
    wrong silently: a file with a mis-framed header is well-formed XML, opens
    without complaint, and then has VTK allocate against a length it read out of
    the payload. The library that defines the format writes it.

    VTK's own writer emits the arrays as base64 float32 with zlib compression
    rather than as ascii. That is the point of this change: the ascii spelling
    rounded every sample to six significant digits (``%.6g``), so the field in
    ParaView was not quite the field that was computed. The binary spelling is
    the exact float32 the solver held, at a fraction of the size.
    """
    import pyvista as pv
    import vtk

    nz, ny, nx = shape
    zs = origin[2] + np.arange(nz) * spacing[2]
    ys = origin[1] + np.arange(ny) * spacing[1]
    xs = origin[0] + np.arange(nx) * spacing[0]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")

    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    count = coords.shape[0]

    # One vertex cell per sample, so ParaView shows the field where it was
    # sampled rather than interpolating it onto cells it invented. The prefix
    # column is the VTK cell format: how many point ids follow each cell.
    cells = np.hstack([np.ones((count, 1), dtype=np.int64),
                       np.arange(count, dtype=np.int64).reshape(-1, 1)])
    grid = pv.UnstructuredGrid(cells.ravel(),
                               np.full(count, vtk.VTK_VERTEX, dtype=np.uint8),
                               coords)
    grid["E"] = field.reshape(-1, 3)
    grid["E_magnitude"] = np.linalg.norm(field, axis=-1).ravel()

    if hasattr(path, "write"):
        # pyvista writes by filename, so an in-memory caller gets a temporary
        # file and the bytes are forwarded to it. The solver hands the result
        # straight to object storage, which is why this branch exists at all.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".vtu") as tmp:
            grid.save(tmp.name)
            path.write(tmp.read())
    else:
        grid.save(str(path))
