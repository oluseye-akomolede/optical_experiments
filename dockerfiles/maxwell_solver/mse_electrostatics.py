r"""Scalar P1 (linear tetrahedron) assembly for the electrostatics problem

    -div(eps_static * eps0 * grad phi) = rho

The weak form over a test function ``v`` that vanishes on the electrodes is

    \int eps_static eps0 grad(phi) . grad(v) dV  =  \int rho v dV
                                                   + (electrode terms)

so each tet contributes the 4x4 matrix

    K_ij = eps_static * eps0 * vol * (grad lambda_i . grad lambda_j)

with ``grad lambda_i`` the constant barycentric gradients. That is the whole of
the multi-material physics: the material enters only as a per-tet scalar, so a
mesh whose tets carry a part index assembles with one broadcast -- the same
shape :func:`mse_fem.element_matrices` takes as its ``eps_by_tet`` array
argument (an array there, a table-to-array helper here).

This module is the SCALAR problem and does not touch :mod:`mse_fem`'s edge
elements; it shares only :func:`mse_fem.tet_geometry`, so the barycentric
gradients (and the ``inv(J).T`` trap they carry) exist in one place.

Units: ``points`` in metres, ``eps_static`` dimensionless (relative), so the
matrix is in ``eps0 * m``. The permittivity of free space is applied here so a
caller never has to remember it.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

import mse_fem

#: vacuum permittivity, F/m (CODATA), applied to the relative input
EPS0 = 8.8541878128e-12


def check_tet_regions(region_of_tet: np.ndarray, parts) -> np.ndarray:
    """Validate the mesh's own per-tet region ids, and return them as an array.

    The material of a linear tet is a property of the TET, not of its nodes:
    ``tetrahedralize`` returns ``region_of_tet`` from TetGen's region
    attributes, so every tet belongs wholly to one part even where the boundary
    runs between two tets sharing a face.

    Deriving the part from node membership instead is wrong and looks right:
    a boundary face's three nodes belong to both tets on either side, so every
    connected two-part mesh would be rejected as "straddling". This takes the
    mesh's answer.

    Refuses a region id with no matching part (a tet whose material nobody
    stated -- it would otherwise be assembled at some default) and a part that
    no tet uses (a part the caller believes is in the device and is not).
    """
    region_of_tet = np.asarray(region_of_tet, dtype=np.int64).ravel()

    used = sorted(int(r) for r in np.unique(region_of_tet).tolist())
    out_of_range = [r for r in used if r < 0 or r >= len(parts)]
    if out_of_range:
        raise ValueError(
            f"the mesh has {len(out_of_range)} region id(s) {out_of_range} with "
            f"no part; this mesh declares {len(parts)} part(s) ({', '.join(parts)})"
        )
    unused = [p for i, p in enumerate(parts) if i not in used]
    if unused:
        raise ValueError(
            f"part(s) {', '.join(unused)} have no tets in this mesh; refusing so "
            "a part the caller believes is present is not silently absent"
        )
    return region_of_tet


def eps_by_tet(
    parts: list[str], eps_static: dict[str, float], tet_part: np.ndarray
) -> np.ndarray:
    """Per-tet relative permittivity from the caller's per-part table.

    Refuses, at the call and before any matrix is built: a part with no
    ``eps_static`` (a defaulted permittivity is a different device), an
    ``eps_static`` naming a part that does not exist (a typo would otherwise be
    dropped silently), and a non-positive or non-finite value (zero makes the
    matrix singular and a negative makes it indefinite -- neither is a device
    anyone can build).
    """
    missing = [p for p in parts if p not in eps_static]
    if missing:
        raise ValueError(
            f"no eps_static for part(s): {', '.join(missing)}; every part needs "
            "its own static permittivity"
        )
    unknown = [p for p in eps_static if p not in parts]
    if unknown:
        raise ValueError(
            f"eps_static names part(s) that do not exist: {', '.join(unknown)}; "
            f"this mesh has {', '.join(parts)}"
        )
    values = np.empty(len(parts), dtype=np.float64)
    for i, p in enumerate(parts):
        v = float(eps_static[p])
        if not np.isfinite(v):
            raise ValueError(f"eps_static for part {p!r} is not finite: {eps_static[p]!r}")
        if v <= 0.0:
            raise ValueError(
                f"eps_static for part {p!r} must be positive, got {v!r}; a "
                "non-positive permittivity has no electrostatic solution"
            )
        values[i] = v
    return values[np.asarray(tet_part, dtype=np.int64)]


def element_stiffness(points: np.ndarray, tets: np.ndarray, eps_tet: np.ndarray):
    """Closed-form 4x4 P1 stiffness per tet, in ``eps0 * m``.

    ``K_ij = eps_static * eps0 * vol * (grad lambda_i . grad lambda_j)``, with
    ``vol`` the absolute tet volume and ``eps_tet`` the ``(n_tets,)`` relative
    permittivity from :func:`eps_by_tet`.

    Returns ``(K_elem, volumes)`` with shapes ``(n_tets, 4, 4)`` and
    ``(n_tets,)``.
    """
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)
    det_J, grads = mse_fem.tet_geometry(points, tets)
    volumes = np.abs(det_J) / 6.0

    eps_tet = np.asarray(eps_tet, dtype=np.float64).reshape(-1)
    if eps_tet.shape[0] != tets.shape[0]:
        raise ValueError(
            f"eps_tet has {eps_tet.shape[0]} entries for {tets.shape[0]} tets"
        )

    # (n_tets, 4, 4): dot product of every pair of barycentric gradients
    # node indices a,b (axis 1) paired; k is the coordinate axis (2) and is
    # contracted. Writing "nai,naj->nij" reuses i for the node AND leaves j
    # unmatched-free, which einsum resolves to the 3-wide coordinate axis -- a
    # silent (n,3,3) for a 4-node element.
    dots = np.einsum("nak,nbk->nab", grads, grads)
    scale = (eps_tet * EPS0 * volumes).reshape(-1, 1, 1)
    return scale * dots, volumes


def assemble_stiffness(
    points: np.ndarray,
    tets: np.ndarray,
    parts: list[str],
    eps_static: dict[str, float],
    region_of_tet: np.ndarray,
    n_points: int | None = None,
):
    """Assemble the global sparse P1 stiffness matrix.

    ``region_of_tet`` is the per-tet part index the MESHER returned (TetGen
    region attributes), not anything derived from node membership -- see
    :func:`check_tet_regions`.

    Returns ``(K, tet_part)`` with ``K`` a ``(n_points, n_points)`` CSR matrix.

    ``K`` is symmetric and every row sums to zero: a constant potential has no
    flux, which is the discrete statement of the divergence theorem and an
    exact property of this assembly (each ``K_elem`` has that property, and the
    scatter preserves it).
    """
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)

    tet_part = check_tet_regions(region_of_tet, parts)
    eps_tet = eps_by_tet(parts, eps_static, tet_part)
    K_elem, _ = element_stiffness(points, tets, eps_tet)

    if n_points is None:
        n_points = int(points.shape[0])

    # K_elem.ravel() visits (i, j) with j fastest, so the row index repeats each
    # node 4 times within a tet (repeat along axis 1) and the column index tiles
    # the tet's four nodes (tile along axis 1). Getting these the other way round
    # scatters K to the wrong entries while still producing a symmetric-looking,
    # row-sum-zero matrix on a single-tet mesh -- only the multi-tet checks see it.
    rows = np.repeat(tets, 4, axis=1).ravel()        # (n_tets*16,)
    cols = np.tile(tets, (1, 4)).ravel()
    K = sparse.coo_matrix(
        (K_elem.ravel(), (rows, cols)), shape=(n_points, n_points)
    ).tocsr()
    K.sum_duplicates()
    K.eliminate_zeros()
    return K, tet_part
