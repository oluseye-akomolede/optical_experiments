r"""Lowest-order Nédélec (Whitney) edge-element assembly for the vector Maxwell
eigenproblem, plus edge bookkeeping and the PEC boundary elimination.

The weak form we discretise is

    find E, k^2  with   \int curl E . curl v  =  k^2 \int eps_r E . v
    PEC: n x E = 0 on the boundary

with ``eps_r = n^2`` uniform. On linear tetrahedra the lowest-order edge element
attaches one degree of freedom to each *edge* of the mesh, with the basis

    N_a(x) = lambda_i grad lambda_j - lambda_j grad lambda_i      (a = (i,j))

For a linear tet the gradients of the barycentric functions are constant, so
both element matrices have closed forms and every tet can be handled by the same
vectorised arithmetic. ``for a in (i,j)`` runs over the six edges in the order
``(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)``.

Global edge orientation is min-node -> max-node, as the task specifies. The
oriented-to-reference sign is carried in the global edge table, so the assembled
matrices are symmetric and independent of the order TetGen happened to emit
nodes in.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse

#: the six tet edges, as (local_a, local_b) with local_a < local_b
TET_EDGE_PAIRS = np.array(
    [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)], dtype=np.int64
)

_SIGNS = ({1: 1.0, -1: -1.0},)


def build_edges(tets: np.ndarray):
    """Build the global edge table for a tet mesh.

    Returns ``(edges, tet_edges, signs)`` where

    * ``edges``     is ``(n_edges, 2)`` int64, each row ``(min_node, max_node)``;
    * ``tet_edges`` is ``(n_tets, 6)`` int64, the global edge index of each local edge;
    * ``signs``     is ``(n_tets, 6)`` float64, ``+1`` when the tet's local edge
      orientation agrees with the global orientation and ``-1`` when it is flipped.
    """
    tets = np.asarray(tets, dtype=np.int64)
    n_tets = tets.shape[0]

    local_a = tets[:, TET_EDGE_PAIRS[:, 0]]          # (n_tets, 6)
    local_b = tets[:, TET_EDGE_PAIRS[:, 1]]          # (n_tets, 6)

    lo = np.minimum(local_a, local_b)
    hi = np.maximum(local_a, local_b)

    # signs: local orientation a->b agrees with lo->hi unless a > b
    signs = np.where(local_a < local_b, 1.0, -1.0)

    flat = np.stack([lo.ravel(), hi.ravel()], axis=1)
    edges, inverse = np.unique(flat, axis=0, return_inverse=True)
    tet_edges = inverse.reshape(n_tets, TET_EDGE_PAIRS.shape[0])

    return edges, tet_edges, signs


def element_matrices(points: np.ndarray, tets: np.ndarray, tet_edges: np.ndarray):
    """Vectorised closed-form curl-curl (A) and mass (B) element matrices.

    ``points`` is ``(n_points, 3)`` in metres, ``tets`` is ``(n_tets, 4)``.
    Everything is assembled in one pass over all tets, in metres, so the
    eigenvalues are ``k^2`` in ``1/m^2``.

    Returns ``(A_elem, B_elem, volumes)`` with shapes ``(n_tets, 6, 6)``,
    ``(n_tets, 6, 6)`` and ``(n_tets,)``.
    """
    points = np.asarray(points, dtype=np.float64)
    tets = np.asarray(tets, dtype=np.int64)

    xyz = points[tets]                               # (n_tets, 4, 3)
    v0 = xyz[:, 0, :]
    J = np.stack(
        [xyz[:, 1, :] - v0, xyz[:, 2, :] - v0, xyz[:, 3, :] - v0], axis=2
    )                                                # (n_tets, 3, 3), columns are edges

    det_J = np.linalg.det(J)
    volumes = np.abs(det_J) / 6.0

    # barycentric gradients: grad lambda_i for i = 0..3, shape (n_tets, 3, 4)
    #
    # lambda_i is affine in x, so grad(lambda) = J^-T @ grad_ref -- the TRANSPOSE
    # of J, not J. Getting this wrong is invisible on the right-angled reference
    # tet (where J = I) and only shows up on a general tet.
    inv_J = np.linalg.inv(J)                         # (n_tets, 3, 3)
    grad_ref = np.array(
        [-1.0, -1.0, -1.0,
          1.0,  0.0,  0.0,
          0.0,  1.0,  0.0,
          0.0,  0.0,  1.0], dtype=np.float64
    ).reshape(4, 3)
    grads = np.einsum("nji,aj->nai", inv_J, grad_ref)   # (n_tets, 4, 3)

    # curl of the reference basis functions is constant:
    #   curl N_a = 2 * grad lambda_i x grad lambda_j
    a_idx = TET_EDGE_PAIRS[:, 0]
    b_idx = TET_EDGE_PAIRS[:, 1]
    curls = 2.0 * np.cross(
        grads[:, a_idx, :], grads[:, b_idx, :]
    )                                                # (n_tets, 6, 3)

    # curl-curl:  vol * (2 g_i x g_j) . (2 g_k x g_l),  vol = |detJ|/6
    A_elem = np.einsum("nac,nbc->nab", curls, curls) * (np.abs(det_J) / 6.0)[:, None, None]

    # Whitney mass. It is NOT a constant times |detJ|: for edges a = (i, j) and
    # b = (k, l), with Mn_pq = vol * (1 + delta_pq) / 20,
    #     B_ab = Mn_ik (g_j . g_l) - Mn_il (g_j . g_k)
    #          - Mn_jk (g_i . g_l) + Mn_jl (g_i . g_k)
    gi = grads[:, a_idx, :]                          # (n_tets, 6, 3) = g_i
    gj = grads[:, b_idx, :]                          # (n_tets, 6, 3) = g_j
    gk = grads[:, a_idx, :]                          # (n_tets, 6, 3) = g_k
    gl = grads[:, b_idx, :]                          # (n_tets, 6, 3) = g_l

    vol = (np.abs(det_J) / 6.0)[:, None, None]       # (n_tets, 1, 1)
    delta = np.eye(4)[None, :, :]                    # (1, 4, 4) node deltas
    Mn = vol * (1.0 + delta) / 20.0                  # (n_tets, 4, 4), node indexed

    # pull the four node-pair blocks of Mn into edge-pair form
    Mn_ik = Mn[:, a_idx][:, :, a_idx]                # (n_tets, 6, 6)
    Mn_il = Mn[:, a_idx][:, :, b_idx]
    Mn_jk = Mn[:, b_idx][:, :, a_idx]
    Mn_jl = Mn[:, b_idx][:, :, b_idx]

    B_elem = (
        Mn_ik * np.einsum("nac,nbc->nab", gj, gl)
        - Mn_il * np.einsum("nac,nbc->nab", gj, gk)
        - Mn_jk * np.einsum("nac,nbc->nab", gi, gl)
        + Mn_jl * np.einsum("nac,nbc->nab", gi, gk)
    )

    # orientation signs, so A and B are symmetric in the *global* edge ordering
    _, _, signs = build_edges(tets)
    s_out = signs[:, :, None]
    s_in = signs[:, None, :]
    A_elem = A_elem * s_out * s_in
    B_elem = B_elem * s_out * s_in

    return A_elem, B_elem, volumes


def assemble_global(points, tets, tet_edges, n_edges, dtype=np.float64):
    """Assemble the global sparse A and B from the element matrices.

    Accumulates the COO triplets with ``coo_matrix``, which sums duplicate
    ``(row, col)`` entries on conversion. That is O(nnz) in memory; a
    ``np.bincount`` over ``rows * n_edges + cols`` would instead allocate a
    dense ``n_edges^2`` array (1.96 TiB at 500k edges), so it is not used here.
    """
    A_elem, B_elem, _ = element_matrices(points, tets, tet_edges)

    ii = np.repeat(tet_edges, 6, axis=1)              # (n_tets, 36)
    jj = np.tile(tet_edges, (1, 6))                   # (n_tets, 36)
    rows = ii.ravel()
    cols = jj.ravel()

    shape = (n_edges, n_edges)
    A = sparse.coo_matrix((A_elem.ravel(), (rows, cols)), shape=shape).tocsr()
    B = sparse.coo_matrix((B_elem.ravel(), (rows, cols)), shape=shape).tocsr()
    A = A.astype(dtype)
    B = B.astype(dtype)
    A.eliminate_zeros()
    B.eliminate_zeros()
    A.sum_duplicates()
    B.sum_duplicates()
    return A, B


def boundary_edges(tets: np.ndarray, edges: np.ndarray):
    """Indices into ``edges`` of every edge that lies on the boundary surface.

    Boundary faces are the four triangular faces of each tet that are not shared
    with another tet. All three edges of a boundary face are PEC-constrained.
    """
    tets = np.asarray(tets, dtype=np.int64)
    face_nodes = np.array(
        [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)], dtype=np.int64
    )
    faces = tets[:, face_nodes].reshape(-1, 3)        # (4*n_tets, 3)
    faces = np.sort(faces, axis=1)

    unique_faces, counts = np.unique(faces, axis=0, return_counts=True)
    open_faces = unique_faces[counts == 1]

    if open_faces.size == 0:
        return np.empty(0, dtype=np.int64)

    # all three edges of every open face
    e01 = np.sort(open_faces[:, [0, 1]], axis=1)
    e02 = np.sort(open_faces[:, [0, 2]], axis=1)
    e12 = np.sort(open_faces[:, [1, 2]], axis=1)
    wanted = np.unique(np.concatenate([e01, e02, e12], axis=0), axis=0)

    lookup = {tuple(e): i for i, e in enumerate(map(tuple, edges))}
    idx = [lookup[tuple(e)] for e in wanted if tuple(e) in lookup]
    return np.array(sorted(idx), dtype=np.int64)


def pec_eliminate(A, B, constrained: np.ndarray, n_edges: int):
    """Enforce ``n x E = 0`` by striking constrained rows/columns.

    We delete rather than pin: a pinned edge would leave a spurious mode at the
    pin value. The remaining system ``A_ff x = k^2 B_ff x`` has the same
    physical eigenvalues and a much smaller gradient null space.
    """
    if constrained.size == 0:
        return A, B, np.arange(n_edges, dtype=np.int64)

    keep = np.setdiff1d(np.arange(n_edges, dtype=np.int64), constrained)
    return A[keep][:, keep], B[keep][:, keep], keep
