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


def _barycentric_mask(points: np.ndarray, tets: np.ndarray, samples: np.ndarray,
                      chunk: int = 200_000):
    """Boolean mask: which sample points lie inside *any* tet.

    A point is inside a tet when all four barycentric coordinates are >= 0.
    Computed in chunks so a 64^3 grid (262k points) against a 150k-tet mesh
    never materialises a huge intermediate.
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
    # inverse maps (x - v0) to the barycentric coords of nodes 1..3
    A1 = inv_J[:, 0, :]
    A2 = inv_J[:, 1, :]
    A3 = inv_J[:, 2, :]

    inside = np.zeros(samples.shape[0], dtype=bool)

    for start in range(0, samples.shape[0], chunk):
        stop = min(start + chunk, samples.shape[0])
        s = samples[start:stop]                           # (m, 3)
        d = s[:, None, :] - v0[None, :, :]                # (m, n_tets, 3)
        b1 = np.einsum("mtj,tj->mt", d, A1)
        b2 = np.einsum("mtj,tj->mt", d, A2)
        b3 = np.einsum("mtj,tj->mt", d, A3)
        b0 = 1.0 - b1 - b2 - b3
        hit = (b0 >= -1e-9) & (b1 >= -1e-9) & (b2 >= -1e-9) & (b3 >= -1e-9)
        inside[start:stop] = hit.any(axis=1)

    return inside


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
):
    """Evaluate one edge-element eigenvector on the regular grid.

    ``eigenvector`` holds one coefficient per edge in ``tet_edges``, which is
    the **full** edge numbering *after* PEC elimination has removed nothing --
    i.e. it is the reduced vector plus ``free_edges`` telling us where each
    reduced unknown lives in the full numbering. When ``free_edges`` is given,
    ``eigenvector`` is the reduced (free-only) vector and the PEC-constrained
    edges are taken as zero, which is exactly the boundary condition
    ``n x E = 0``.

    Returns ``(field, shape, origin, spacing)`` where ``field`` is
    ``(nz, ny, nx, 3)`` float32 with ``NaN`` outside the object.
    """
    points = np.asarray(points, dtype=float)
    tets = np.asarray(tets, dtype=np.int64)
    samples, shape, origin, spacing = sample_grid(points, n_longest)

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
    xyz = points[tets]
    v0 = xyz[:, 0, :]
    J = np.stack(
        [xyz[:, 1, :] - v0, xyz[:, 2, :] - v0, xyz[:, 3, :] - v0], axis=2
    )
    det_J = np.linalg.det(J)
    ok = np.abs(det_J) > 1e-30
    inv_J = np.zeros_like(J)
    inv_J[ok] = np.linalg.inv(J[ok])

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
    chunk_pts = 60_000

    a_idx = np.array([0, 0, 0, 1, 1, 2])
    b_idx = np.array([1, 2, 3, 2, 3, 3])

    for start in range(0, n_samples, chunk_pts):
        stop = min(start + chunk_pts, n_samples)
        s = samples[start:stop]
        d = s[:, None, :] - v0[None, :, :]                # (m, n_tets, 3)
        # x = v0 + J @ b, so b = J^-1 (x - v0) = (x - v0)^T J^-T, i.e. the ROWS
        # of J^-1. Using its columns instead would silently scramble the
        # barycentric test on any non-right-angled tet.
        b1 = np.einsum("mtj,tj->mt", d, inv_J[:, :, 0])
        b2 = np.einsum("mtj,tj->mt", d, inv_J[:, :, 1])
        b3 = np.einsum("mtj,tj->mt", d, inv_J[:, :, 2])
        bary = np.stack([1.0 - b1 - b2 - b3, b1, b2, b3], axis=2)   # (m, n_tets, 4)

        inside = (
            (bary[:, :, 0] >= -1e-9) & (bary[:, :, 1] >= -1e-9)
            & (bary[:, :, 2] >= -1e-9) & (bary[:, :, 3] >= -1e-9)
        )
        # first containing tet wins
        tet_of = np.argmax(inside, axis=1)
        hit = inside[np.arange(inside.shape[0]), tet_of]
        hit_any[start:stop] = hit

        if hit.any():
            rows = np.nonzero(hit)[0]
            tt = tet_of[rows]
            lam = bary[rows, tt]                          # (h, 4)
            # N_a = lambda_i grad lambda_j - lambda_j grad lambda_i
            lj_ = grads[tt]                               # (h, 4, 3)
            li = lam[:, a_idx][:, :, None] * lj_[:, b_idx, :]
            ljv = lam[:, b_idx][:, :, None] * lj_[:, a_idx, :]
            basis = li - ljv                              # (h, 6, 3)
            field[start + rows] = np.einsum(
                "hc,hcd->hd", local_coeffs[tt], basis
            )

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
    """
    from xml.etree import ElementTree as ET

    nz, ny, nx = shape
    zs = origin[2] + np.arange(nz) * spacing[2]
    ys = origin[1] + np.arange(ny) * spacing[1]
    xs = origin[0] + np.arange(nx) * spacing[0]
    zz, yy, xx = np.meshgrid(zs, ys, xs, indexing="ij")

    coords = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
    ex = field[:, :, :, 0].ravel()
    ey = field[:, :, :, 1].ravel()
    ez = field[:, :, :, 2].ravel()
    mag = np.linalg.norm(field, axis=-1).ravel()

    root = ET.Element("VTKFile", type="UnstructuredGrid", version="0.1",
                      byte_order="LittleEndian")
    grid = ET.SubElement(root, "UnstructuredGrid")
    piece = ET.SubElement(
        grid, "Piece",
        NumberOfPoints=str(coords.shape[0]), NumberOfCells=str(coords.shape[0]),
    )
    pts = ET.SubElement(piece, "Points")
    da = ET.SubElement(pts, "DataArray", type="Float32", NumberOfComponents="3",
                       format="ascii")
    da.text = "\n".join(" ".join(f"{v:.6g}" for v in row) for row in coords)

    cells = ET.SubElement(piece, "Cells")
    conn = ET.SubElement(cells, "DataArray", type="Int32", Name="connectivity",
                         format="ascii")
    conn.text = " ".join(str(i) for i in range(coords.shape[0]))
    offs = ET.SubElement(cells, "DataArray", type="Int32", Name="offsets",
                         format="ascii")
    offs.text = " ".join(str(i + 1) for i in range(coords.shape[0]))
    types = ET.SubElement(cells, "DataArray", type="UInt8", Name="types",
                          format="ascii")
    types.text = " ".join("1" for _ in range(coords.shape[0]))   # VTK_VERTEX

    pd = ET.SubElement(piece, "PointData")
    vec = ET.SubElement(pd, "DataArray", type="Float32", Name="E",
                        NumberOfComponents="3", format="ascii")
    vec.text = "\n".join(
        f"{a:.6g} {b:.6g} {c:.6g}"
        for a, b, c in zip(ex, ey, ez, strict=False)
    )
    mg = ET.SubElement(pd, "DataArray", type="Float32", Name="E_magnitude",
                       NumberOfComponents="1", format="ascii")
    mg.text = "\n".join(f"{v:.6g}" for v in mag)

    ET.ElementTree(root).write(path, encoding="UTF-8", xml_declaration=True)
