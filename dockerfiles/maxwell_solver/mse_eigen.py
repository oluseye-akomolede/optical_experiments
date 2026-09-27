"""Shift-invert eigen solve for the discretised vector Maxwell problem.

Naive shift-invert does not work here. The curl-curl operator has a huge null
space of gradient fields -- one vector per interior node, so thousands on a real
mesh -- and the mass matrix is singular on it. Asking for the smallest
eigenvalues puts the Lanczos iteration straight into that null space: it loses
orthogonality and returns junk sitting at the shift.

**Projected shift-invert.** We remove the null space exactly instead of trusting
the iteration to avoid it:

1. Build the discrete gradient ``G`` on the *free* (non-PEC) edges against the
   *interior* nodes: for edge ``e = (i -> j)``, ``G[e, j] = +1`` and
   ``G[e, i] = -1``. For a simply-connected cavity ``null(A_free) = range(G_f)``
   exactly, so this is the whole null space.
2. Use a *negative* shift ``sigma = -s`` with ``s`` of the order of the first
   mode, ``s ~ (pi/D)^2``. Then ``K = A + s B`` is symmetric positive definite
   and can be factorised once.
3. Project onto the divergence-free subspace with the B-orthogonal projector
   ``P y = y - G_f (G_f^T B G_f)^-1 G_f^T B y``. ``G_f^T B G_f`` is a sparse SPD
   nodal matrix, factored once as well.
4. Solve ``eigsh(A, k, M=B, sigma=-s, which='LM', OPinv=P o K^-1, v0=P(random))``.
   SciPy maps back ``lambda = sigma + 1/theta``, so the lowest physical modes are
   the *largest* ``theta`` and ``which='LM'`` is the right selector. With the
   null space projected out there are no spurious vectors to filter, so the
   ``lambda < 1e-6 * max(lambda)`` test becomes an assertion rather than a filter.

Numbers are float64 throughout.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as spla

#: shift magnitude as a fraction of (pi/D)^2; the first physical mode is near there
SHIFT_FRACTION = 1.0

#: eigenvalues below this fraction of the largest returned one are spurious
SPURIOUS_FRACTION = 1e-6


class EigenError(Exception):
    """The eigenproblem could not be solved as posed (e.g. too few elements)."""


def bbox_diagonal(points: np.ndarray) -> float:
    extent = np.asarray(points, dtype=np.float64).max(axis=0) - np.asarray(
        points, dtype=np.float64
    ).min(axis=0)
    diagonal = float(np.linalg.norm(extent))
    if diagonal <= 0:
        raise EigenError("mesh has zero extent")
    return diagonal


def default_shift(points: np.ndarray) -> float:
    """Magnitude ``s`` of the negative shift: ``SHIFT_FRACTION * (pi/D)^2``.

    The first physical in-medium ``k^2`` of a cavity with bounding-box diagonal D
    sits at or above ``(pi/D)^2``, so ``K = A + s B`` is positive definite for
    ``s`` at this scale.
    """
    d = bbox_diagonal(points)
    return SHIFT_FRACTION * (np.pi / d) ** 2


def build_gradient(points, tets, edges, tet_edges, free_edges, constrained):
    """Discrete gradient ``G_f`` from interior-node values to free-edge values.

    Returns ``(G_f, interior_nodes)``. ``G_f`` is ``(n_free, n_interior)`` with
    ``+1`` at the edge's max node and ``-1`` at its min node, restricted to the
    interior nodes' columns. ``null(A_free) = range(G_f)`` for a simply-connected
    cavity.
    """
    free_edges = np.asarray(free_edges, dtype=np.int64)
    constrained = np.asarray(constrained, dtype=np.int64)

    boundary_nodes = np.unique(edges[constrained].ravel()) if constrained.size else np.empty(0, np.int64)
    all_nodes = np.unique(tets)
    interior = np.setdiff1d(all_nodes, boundary_nodes)

    if interior.size == 0:
        raise EigenError("no interior nodes; the mesh is all boundary")

    col_of = {int(n): i for i, n in enumerate(interior)}
    row_of = np.full(edges.shape[0], -1, dtype=np.int64)
    row_of[free_edges] = np.arange(free_edges.size)

    rows: list[int] = []
    cols: list[int] = []
    vals: list[float] = []
    for g in free_edges:
        i, j = int(edges[g, 0]), int(edges[g, 1])
        r = int(row_of[g])
        if j in col_of:
            rows.append(r)
            cols.append(col_of[j])
            vals.append(+1.0)
        if i in col_of:
            rows.append(r)
            cols.append(col_of[i])
            vals.append(-1.0)

    G = sparse.coo_matrix(
        (np.asarray(vals, float), (np.asarray(rows, np.int64), np.asarray(cols, np.int64))),
        shape=(free_edges.size, interior.size),
    ).tocsr()
    return G, interior


def solve_modes(
    A: sparse.spmatrix,
    B: sparse.spmatrix,
    n_modes: int,
    points: np.ndarray,
    edges: np.ndarray,
    tets: np.ndarray,
    tet_edges: np.ndarray,
    free_edges: np.ndarray,
    constrained: np.ndarray,
    verbose: bool = True,
    spd_ordering: bool = True,
):
    """Solve for the lowest ``n_modes`` physical eigenvalues of ``A x = k^2 B x``.

    ``A``/``B`` must already be reduced to the free edges. ``edges``, ``tets``,
    ``tet_edges`` and ``constrained`` refer to the *unreduced* numbering so the
    gradient projector can be built.

    Returns ``(eigenvalues, eigenvectors, sigma)`` with eigenvalues ``k^2`` in
    ``1/m^2``, ascending.
    """
    n_free = A.shape[0]
    if n_free < 2 * n_modes + 2:
        raise EigenError(
            f"the mesh is too coarse: only {n_free} free edge unknowns for "
            f"{n_modes} requested modes. Use a finer mesh."
        )

    s = default_shift(points)
    sigma = -s

    # Both matrices below are symmetric positive definite, and both are factored
    # as such. That is what these three arguments say, and it is worth stating
    # what the default cost instead, because the default is *correct* -- it just
    # does work the problem does not need:
    #
    #   permc_spec='MMD_AT_PLUS_A'  SuperLU's default is COLAMD, an ordering for
    #       a general unsymmetric matrix. K and S are symmetric, so the fill it
    #       produces is larger than it needs to be, and the fill is what the
    #       factorisation, the solves and the memory all scale with.
    #   diag_pivot_thresh=0.0       The default 1.0 does threshold partial
    #       pivoting, which permutes rows to avoid small pivots. For an SPD
    #       matrix that risk does not exist -- every pivot is positive -- so the
    #       pivoting only destroys the ordering chosen above. Setting 0.0 keeps
    #       the symmetric ordering intact, which is also what makes the
    #       factorisation of a symmetric matrix symmetric.
    #   options={'SymmetricMode': True}  tells SuperLU the same thing again, so
    #       it stores L*D*L^T rather than a general L*U.
    #
    # Measured by the lead on the Jetson box (10x7x5 mm, 76,809 tets, 85,771
    # unknowns): solve 255 s -> 98.5 s, L+U 178M -> 80M, peak 2.1 GB -> 1.25 GB,
    # with the same worst-case error (0.058% vs analytic) and the same 122
    # OPinv applications -- so the speedup is the factorisation, not a
    # looser convergence.
    #
    # `permc_spec` is spelled out rather than passed positionally: splu takes it
    # by keyword, and a positional argument would silently bind to the wrong
    # parameter if SciPy ever reorders the signature.
    #
    # `spd_ordering=False` restores SuperLU's defaults. It exists for the tests:
    # the two orderings must find the same eigenvalues, and the only way to
    # assert that is to be able to run both. It is not a runtime switch and
    # nothing in the solver passes it.
    spd_lu = {
        "permc_spec": "MMD_AT_PLUS_A",
        "diag_pivot_thresh": 0.0,
        "options": {"SymmetricMode": True},
    } if spd_ordering else {}

    # negative shift => K = A + s B is SPD
    K = (A + s * B).tocsc()
    try:
        K_lu = spla.splu(K, **spd_lu)
    except Exception as exc:  # noqa: BLE001
        raise EigenError(f"could not factor A + sB: {exc}") from exc

    # B-orthogonal projector onto the divergence-free subspace
    G, _ = build_gradient(points, tets, edges, tet_edges, free_edges, constrained)
    GtB = (G.T @ B).tocsr()
    S = (GtB @ G).tocsc()                       # sparse SPD nodal (interior) matrix
    # S is SPD too, and for the same reason: B is SPD and G has full column rank
    # (its columns are the gradients of the nodal basis functions, which are
    # linearly independent, one per interior node). So it takes the same
    # arguments. Checked rather than assumed -- the test suite factors both on a
    # small box and asserts the same eigenvalues as the default ordering.
    try:
        S_lu = spla.splu(S, **spd_lu)
    except Exception as exc:  # noqa: BLE001
        raise EigenError(f"could not factor G^T B G: {exc}") from exc

    def project(x):
        return x - G @ S_lu.solve(GtB @ x)

    def opinv(x):
        return project(K_lu.solve(x))

    op = spla.LinearOperator((n_free, n_free), matvec=opinv, dtype=np.float64)
    v0 = project(np.random.default_rng(0).standard_normal(n_free))
    norm_v0 = np.linalg.norm(v0)
    if norm_v0 > 0:
        v0 = v0 / norm_v0

    k = min(n_modes + 8, n_free - 1)
    eigenvalues = eigenvectors = None

    for attempt in range(4):
        try:
            eigenvalues, eigenvectors = spla.eigsh(
                A, k=k, M=B, sigma=sigma, which="LM", OPinv=op, v0=v0,
            )
        except Exception as exc:  # noqa: BLE001
            if attempt == 3:
                raise EigenError(f"eigen solve failed: {exc}") from exc
            k = min(k * 2, n_free - 1)
            continue

        order = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        cutoff = SPURIOUS_FRACTION * float(np.max(np.abs(eigenvalues)))
        eigenvalues = eigenvalues[eigenvalues > cutoff]
        eigenvectors = eigenvectors[:, : eigenvalues.size]

        if eigenvalues.size >= n_modes:
            break
        if verbose:
            print(f"[mse] only {eigenvalues.size} usable modes from k={k}, retrying",
                  flush=True)
        k = min(k * 2, n_free - 1)

    if eigenvalues.size < n_modes:
        raise EigenError(
            f"found only {eigenvalues.size} physical modes, needed {n_modes}; "
            "the mesh may be too coarse"
        )

    eigenvalues = eigenvalues[:n_modes]
    eigenvectors = eigenvectors[:, :n_modes]

    if verbose:
        print(f"[mse] solved {n_modes} modes, sigma={sigma:.4e} 1/m^2, "
              f"k2 in [{eigenvalues[0]:.4e}, {eigenvalues[-1]:.4e}]", flush=True)

    return eigenvalues, eigenvectors, sigma


def to_physical(
    eigenvalues: np.ndarray,
    n_index: float,
    speed_of_light: float = 299_792_458.0,
    n_points=None,
):
    """Convert ``k^2`` (in-medium, 1/m^2) to the reported physical quantities."""
    k = np.sqrt(np.clip(np.asarray(eigenvalues, dtype=float), 0.0, None))
    n_index = float(n_index)
    if n_index <= 0:
        raise EigenError(f"refractive index must be positive, got {n_index}")

    k0 = k / n_index
    freq_hz = speed_of_light * k0 / (2.0 * np.pi)
    vacuum_wavelength_m = np.where(
        k0 > 0, 2.0 * np.pi / np.maximum(k0, 1e-300), np.inf
    )
    return k, k0, freq_hz, vacuum_wavelength_m
