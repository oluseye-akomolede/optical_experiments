"""2-D photonic-crystal band diagrams: scalar Helmholtz on a Bloch unit cell.

WHY THIS FILE EXISTS. Feature 2 of the photonics roadmap: an infinite 2-D
lattice of rods or holes, square and triangular, TE and TM, a band diagram
along the irreducible-zone path. It is a *scalar* problem, which is what makes
it different from everything else in this directory -- there are no vector
elements and no gradient projector here.

THE TWO POLARISATIONS, and why the mapping is written out. The PhC names its
polarisation for the OUT-OF-PLANE field; the 1-D layer problem names it for the
field's relation to the INTERFACE NORMAL. Those are different reference
directions, and mixing them up is the single easiest error in this feature --
it was made twice in the plan. The one fact that settles it:

    the lattice is in x-y, so z is out of plane; a stripe varies along x, so the
    interface normal is x, and z-hat . x-hat = 0.

So ``E_z`` (PhC **TM**) is TANGENTIAL to every interface -> the layer problem's
**s**; ``H_z`` (PhC **TE**) puts E in-plane, normal to the interfaces -> **p**.

    TM (E_z):  -lap(psi) = (w/c)^2 eps psi      plain stiffness, eps on the MASS
    TE (H_z):  -div((1/eps) grad psi) = (w/c)^2 psi   1/eps on the STIFFNESS

Units. Everything internal is in the cell's own length unit; frequencies are
returned as the dimensionless ``w a / (2 pi c)``, which is what makes them
comparable to a published band diagram regardless of the cell size.

The 1-D layered transfer-matrix oracle in :func:`layered_traces` is validated
against an independent ODE shooting solve in ``tests/test_pc.py`` before any
mesh is compared to it -- the plan's Control B. That oracle is the reference
every dispersion claim is measured against, so it is checked on its own first.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_matrix, csc_matrix, identity
from scipy.sparse import linalg as spla

try:  # the solver image ships mse_common; a bare checkout may not have it on path
    from mse_common import MeshingError
except ImportError:  # pragma: no cover - exercised by the image smoke test
    class MeshingError(Exception):
        """A failure the user can act on (bad file, open surface, mesh too big)."""


class PCError(MeshingError):
    """A photonic-crystal request the user can act on.

    Subclasses ``MeshingError`` so the web side's existing refusal handling --
    which catches that type and shows the message -- names the limit without a
    second except clause.
    """


# ---------------------------------------------------------------------------
# Lattices
# ---------------------------------------------------------------------------

#: The two lattices, as (a1, a2) in units of the lattice constant ``a``.
#:
#: The triangular basis is the 60-degree one. MPB writes it as
#: ``basis1 = (sqrt(3)/2, 1/2)``, ``basis2 = (sqrt(3)/2, -1/2)``; the pair below
#: is the same lattice rotated 30 degrees, which has the SAME Gram matrix and so
#: the same band frequencies (checked in the tests). Rotation cannot move a
#: band, but the two spellings must be recognised as one lattice or a reader
#: comparing against the tutorial will think they differ.
LATTICES = {
    "square": (np.array([1.0, 0.0]), np.array([0.0, 1.0])),
    "triangular": (np.array([np.sqrt(3) / 2, 0.5]),
                   np.array([np.sqrt(3) / 2, -0.5])),
}


def lattice_vectors(lattice: str) -> tuple[np.ndarray, np.ndarray]:
    """``(a1, a2)`` in units of ``a`` for a named lattice."""
    try:
        a1, a2 = LATTICES[lattice]
    except KeyError:
        raise PCError(
            f"unknown lattice {lattice!r}; expected one of {sorted(LATTICES)}"
        ) from None
    return a1.copy(), a2.copy()


def reciprocal_vectors(a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(b1, b2)`` with ``b_i . a_j = 2 pi delta_ij``."""
    A = np.column_stack([a1, a2])
    B = 2 * np.pi * np.linalg.inv(A).T
    return B[:, 0], B[:, 1]


def high_symmetry_path(lattice: str, points_per_segment: int = 10):
    """The irreducible-zone path and its labels, in units of ``2 pi / a``.

    Square: Gamma-X-M-Gamma. Triangular: Gamma-M-K-Gamma. Returned as
    ``(labels, k_points)`` where ``k_points`` is ``(n, 2)`` in Cartesian
    ``2 pi / a`` units, endpoints included exactly once.

    The triangular corners are **derived from the reciprocal basis**, not taken
    from a formula. This matters: the textbook ``K = (b1 + b2)/3`` is correct
    only for one orientation of the basis. For the basis used here it gives
    ``|K| = 2.42`` where the physical corner is ``4 pi / (3a) = 4.19``. The
    corners are instead the first-Brillouin-zone vertices nearest Gamma, which
    for this lattice are ``(2 b1 + b2)/3`` (the K corner, ``|K| = 4 pi/3a``) and
    ``b1 / 2`` (the M point, ``|M| = 2 pi / (sqrt3 a)``) -- each asserted against
    its closed form in the tests rather than trusted.
    """
    if points_per_segment < 2:
        raise PCError(
            f"points_per_segment must be at least 2, got {points_per_segment}"
        )
    a1, a2 = lattice_vectors(lattice)
    b1, b2 = reciprocal_vectors(a1, a2)
    two_pi_over_a = 2 * np.pi  # |a_i| = 1 for both lattices above

    if lattice == "square":
        verts = [(0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.0)]
        labels = ["$\\Gamma$", "X", "M", "$\\Gamma$"]
    elif lattice == "triangular":
        M = (b1 / 2.0) / two_pi_over_a
        K = ((2 * b1 + b2) / 3.0) / two_pi_over_a
        verts = [(0.0, 0.0), tuple(M), tuple(K), (0.0, 0.0)]
        labels = ["$\\Gamma$", "M", "K", "$\\Gamma$"]
    else:
        raise PCError(
            f"unknown lattice {lattice!r}; expected one of {sorted(LATTICES)}"
        ) from None

    chunks = []
    for i in range(len(verts) - 1):
        t = np.linspace(0.0, 1.0, points_per_segment,
                        endpoint=(i == len(verts) - 2))
        p0 = np.array(verts[i])
        p1 = np.array(verts[i + 1])
        chunks.append(p0[None, :] + t[:, None] * (p1 - p0)[None, :])
    return labels, np.vstack(chunks)


# ---------------------------------------------------------------------------
# Unit-cell meshing with periodic node pairing
# ---------------------------------------------------------------------------

def unit_cell_mesh(lattice: str, divisions: int, r_over_a: float = 0.0,
                   n_rod: float = 1.0, n_bg: float = 1.0, n_theta: int = 24):
    """Triangulate one unit cell and pair its periodic boundary nodes.

    Returns ``(points, triangles, eps_by_tri, pairs)``:

    * ``points`` ``(n, 2)`` in units of ``a``;
    * ``triangles`` ``(m, 3)`` int, Delaunay on a STRUCTURED point set, so the
      triangulation is deterministic for a given ``divisions``;
    * ``eps_by_tri`` ``(m,)`` the relative permittivity of each triangle,
      ``n_rod**2`` inside the rod/hole and ``n_bg**2`` outside. A rod is placed
      at the cell centre and kept strictly inside the cell, so no triangle
      straddles the eps discontinuity only when the circle is resolved by the
      point set -- see the refusal below;
    * ``pairs`` a list of ``(slave, master, phase_vec)`` where a master is the
      node the slave equals under a lattice translation, and ``phase_vec`` is the
      translation (as a multiple of ``a1``/``a2`` integer coefficients) whose
      Bloch phase the slave carries.

    Pairing is done by HASHING the translated coordinate, never by index
    arithmetic: index arithmetic is correct in the interior of an edge and wrong
    at the corners, which is exactly where the error would hide.
    """
    if divisions < 4:
        raise PCError(f"divisions must be at least 4, got {divisions}")
    if not 0.0 <= r_over_a < 0.5:
        raise PCError(
            f"r/a must be in [0, 0.5) so the rod stays inside the cell; got "
            f"{r_over_a}"
        )
    a1, a2 = lattice_vectors(lattice)

    # A structured grid in the FRACTIONAL (u, v) coordinates of the lattice, so
    # the same code works for the rhombic triangular cell.
    u = np.linspace(0.0, 1.0, divisions + 1)
    U, V = np.meshgrid(u, u)
    frac = np.column_stack([U.ravel(), V.ravel()])
    pts = frac @ np.vstack([a1, a2])

    from scipy.spatial import Delaunay
    tri = Delaunay(pts)
    triangles = tri.simplices

    centre = 0.5 * (a1 + a2)
    # A rod centred at the cell centre: shift the circle's centre into the cell.
    rod_radius = r_over_a  # times |a_i| = 1
    tri_centres = pts[triangles].mean(axis=1)
    # Fold each triangle centre into the fundamental cell (translate by lattice
    # vectors) so the circle test is on the physical cell content, not on the
    # arbitrary choice of which copy Delaunay happened to use.
    folded = _fold_into_cell(tri_centres, a1, a2)
    inside = np.linalg.norm(folded - centre, axis=1) < rod_radius
    eps_by_tri = np.where(inside, float(n_rod) ** 2, float(n_bg) ** 2)

    pairs = _periodic_pairs(pts, a1, a2)
    return pts, triangles, eps_by_tri, pairs


def _fold_into_cell(points: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> np.ndarray:
    """Map points back into the fundamental cell by lattice translations."""
    A = np.column_stack([a1, a2])
    frac = points @ np.linalg.inv(A).T
    frac = frac - np.floor(frac)  # [0, 1) in both fractional coordinates
    return frac @ np.vstack([a1, a2])


def _periodic_pairs(points: np.ndarray, a1: np.ndarray, a2: np.ndarray):
    """Pair boundary nodes under the two lattice translations.

    A node is a slave if it lies on the ``u=1`` or ``v=1`` face of the cell and
    its partner (translated by ``-a1`` or ``-a2``) is a node. Corners are
    special: the corner maps to itself under the product of the shifts, so it is
    handled once, with the combined phase.
    """
    tol = 1e-9
    A = np.column_stack([a1, a2])
    frac = points @ np.linalg.inv(A).T
    frac = np.round(frac, 9)  # kill floating noise so the hash is exact

    table: dict[tuple[float, float], int] = {}
    for i, (u, v) in enumerate(frac):
        table[(float(u), float(v))] = i

    pairs = []
    for i, (u, v) in enumerate(frac):
        on_u = abs(u - 1.0) < tol
        on_v = abs(v - 1.0) < tol
        if not (on_u or on_v):
            continue
        if on_u and on_v:
            # The corner: subtract both shifts and apply both phases.
            key = (float(np.round(u - 1.0, 9)), float(np.round(v - 1.0, 9)))
            j = table.get(key)
            if j is not None:
                pairs.append((i, j, np.array([1, 1])))
            continue
        if on_u:
            key = (float(np.round(u - 1.0, 9)), float(v))
            j = table.get(key)
            if j is not None:
                pairs.append((i, j, np.array([1, 0])))
        else:
            key = (float(u), float(np.round(v - 1.0, 9)))
            j = table.get(key)
            if j is not None:
                pairs.append((i, j, np.array([0, 1])))
    return pairs


# ---------------------------------------------------------------------------
# The two scalar operators, and the Bloch reduction
# ---------------------------------------------------------------------------

def _p1_matrices(points, triangles, eps_by_tri):
    """P1 stiffness and eps-weighted mass on a triangle mesh.

    Same construction as the 3-D element matrices in :mod:`mse_fem`, in 2-D: the
    stiffness is ``area * G G^T`` with ``G`` the (constant) gradients of the
    three barycentric functions, and the mass is ``area/12 * (1 + delta)``, here
    scaled per element by ``eps_r`` -- the ONE place eps enters, so the physics
    and any estimate cannot disagree about where it goes.
    """
    n = len(points)
    rows, cols, kv = [], [], []
    mrows, mcols, mv = [], [], []
    for t, (i, j, k) in enumerate(triangles):
        p0, p1, p2 = points[i], points[j], points[k]
        det = ((p1[0] - p0[0]) * (p2[1] - p0[1])
               - (p2[0] - p0[0]) * (p1[1] - p0[1]))
        area = abs(det) / 2.0
        if area < 1e-14:
            continue
        s = np.sign(det)
        b = np.array([points[j, 1] - points[k, 1], points[k, 1] - points[i, 1],
                      points[i, 1] - points[j, 1]]) / (2 * area * s)
        c = np.array([points[k, 0] - points[j, 0], points[i, 0] - points[k, 0],
                      points[j, 0] - points[i, 0]]) / (2 * area * s)
        G = np.column_stack([b, c])
        Ke = area * (G @ G.T)
        Me = area / 12.0 * np.array([[2.0, 1.0, 1.0],
                                     [1.0, 2.0, 1.0],
                                     [1.0, 1.0, 2.0]])
        idx = (i, j, k)
        for r in range(3):
            for cc in range(3):
                rows.append(idx[r]); cols.append(idx[cc]); kv.append(Ke[r, cc])
                mrows.append(idx[r]); mcols.append(idx[cc])
                mv.append(Me[r, cc] * eps_by_tri[t])
    K = coo_matrix((kv, (rows, cols)), shape=(n, n)).tocsr()
    M = coo_matrix((mv, (mrows, mcols)), shape=(n, n)).tocsr()
    return K, M


def bloch_reduce(K, M, pairs, k, a1, a2):
    """Fold the periodic slave nodes onto their masters, carrying the phase.

    For a slave ``s`` paired with master ``m`` under translation ``T``, Bloch
    periodicity gives ``u_s = e^{i k . (T a)} u_m``. Substituting that into the
    quadratic forms and summing the slave's row/column into the master's leaves
    a Hermitian problem on the master (free) nodes of HALF the boundary size.

    Returns ``(K_red, M_red, free)`` with ``free`` the surviving node indices.
    """
    n = K.shape[0]
    slave_of = {}
    phase_of = {}
    for slave, master, tvec in pairs:
        T = tvec[0] * a1 + tvec[1] * a2
        phase = np.exp(1j * float(np.dot(k, T)))
        slave_of[slave] = master
        phase_of[slave] = phase

    free = np.array([i for i in range(n) if i not in slave_of], dtype=np.int64)
    index = {int(node): pos for pos, node in enumerate(free)}

    Kc = K.tocoo()
    Mc = M.tocoo()
    # Build the complex reduced matrices by scattering each entry through the
    # substitution. An entry (r, c) lands at (map(r), map(c)) with the product of
    # the two phases; a slave's entries are summed onto its master, which is what
    # makes the result Hermitian.
    def scatter(mat):
        rows, cols, vals = [], [], []
        for r, c, v in zip(mat.row, mat.col, mat.data):
            pr = phase_of.get(int(r), 1.0)
            pc_ = phase_of.get(int(c), 1.0)
            mr = index.get(int(r), index.get(slave_of.get(int(r), -1)))
            mc = index.get(int(c), index.get(slave_of.get(int(c), -1)))
            if mr is None or mc is None:
                continue
            rows.append(mr); cols.append(mc)
            # u_r = phase_r u_{m_r}, and the form is val * conj(u_r) * u_c only
            # for the stiffness (symmetric, real); the phase on the row is the
            # conjugation side. Stiffness and mass are both real-symmetric here,
            # so the reduced matrix is (phase_r * val * phase_c) with the row
            # phase conjugated.
            vals.append(np.conj(pr) * v * pc_)
        return coo_matrix((vals, (rows, cols)), shape=(len(free), len(free))).tocsc()

    return scatter(Kc), scatter(Mc), free


def solve_bands(K, M, pairs, k_points, a1, a2, n_bands, sigma=None):
    """Lowest ``n_bands`` frequencies at each k-point.

    Sparse shift-invert at a NEGATIVE shift, mirroring :mod:`mse_eigen`: at Gamma
    the lowest eigenvalue is exactly 0, so ``sigma = 0`` is singular and a small
    negative shift is required. Frequencies are returned as ``w a / (2 pi c)``.
    """
    if sigma is None:
        sigma = -0.5
    out = []
    for k in np.atleast_2d(k_points):
        Kr, Mr, free = bloch_reduce(K, M, pairs, k, a1, a2)
        n = Kr.shape[0]
        kk = min(n_bands + 8, n - 1)
        v0 = np.random.default_rng(0).normal(size=n).astype(np.complex128)
        v0 = v0 / np.linalg.norm(v0)
        lam = None
        for attempt in range(4):
            try:
                lam, _ = spla.eigsh(Kr, k=kk, M=Mr, sigma=sigma, which="LM", v0=v0)
                break
            except Exception as exc:  # noqa: BLE001
                if attempt == 3:
                    raise PCError(f"band solve failed at k={k}: {exc}") from exc
                kk = min(kk * 2, n - 1)
        lam = np.sort(np.real(lam))[:n_bands]
        out.append(np.sqrt(np.clip(lam, 0.0, None)) / (2 * np.pi))
    return np.array(out)


# ---------------------------------------------------------------------------
# The 1-D layered oracle (Control B)
# ---------------------------------------------------------------------------

def layered_traces(lam, ky, eps2, fill, eps1=1.0, a=1.0):
    """Monodromy TRACES of the 1-D layered cell, per polarisation.

    The cell is a stripe of ``eps2`` occupying the fraction ``fill`` of the cell
    along x, at in-plane wavevector ``(kx, ky)`` -- ``ky`` is the component
    PARALLEL to the layers and is conserved. Returns ``(trace_s, trace_p)``.

    Mapping, from ``z-hat . x-hat = 0``: the PhC's **TM (E_z)** is tangential to
    the interfaces, i.e. the layer problem's **s**; **TE (H_z)** gives **p**.
    The two closed forms are:

        s: (psi, psi')            continuous, admittance eta_s = kx_j
        p: (psi, (1/eps) psi')    continuous, admittance eta_p = kx_j / eps_j

    ``eta_p`` is ``kx/eps``, NOT ``kx/eps^2``. They agree only at ``eps = 1``,
    which is why every check of this oracle runs at ``eps2 != 1``.
    """
    k1 = np.sqrt(lam * eps1 - ky ** 2 + 0j)
    k2 = np.sqrt(lam * eps2 - ky ** 2 + 0j)
    d1 = fill * a
    d2 = (1.0 - fill) * a
    return (_tr(_m_s(k2, d2) @ _m_s(k1, d1)),
            _tr(_m_p(k2, eps2, d2) @ _m_p(k1, eps1, d1)))


def _tr(matrix):
    return np.trace(matrix)


def _m_s(k, d):
    if abs(k) < 1e-14:
        return np.array([[1.0, d], [0.0, 1.0]], dtype=np.complex128)
    c, s = np.cos(k * d), np.sin(k * d)
    return np.array([[c, s / k], [-k * s, c]], dtype=np.complex128)


def _m_p(k, eps, d):
    """Monodromy of the continuous state ``(psi, (1/eps) psi')``."""
    if abs(k) < 1e-14:
        return np.array([[1.0, eps * d], [0.0, 1.0]], dtype=np.complex128)
    c, s = np.cos(k * d), np.sin(k * d)
    return np.array([[c, eps * s / k], [-k * s / eps, c]], dtype=np.complex128)


def layered_bands(eps2, fill, ky, n_bands, eps1=1.0, a=1.0, lam_max=200.0,
                  n_scan=20000):
    """Band-edge eigenvalues of the 1-D layered cell, per polarisation.

    Returns ``(bands_s, bands_p)``: the lowest ``n_bands`` values of ``lam`` at
    which the monodromy trace equals ``2 cos(kx a)`` for some real ``kx``, found
    by scanning the trace for crossings of ``|trace| = 2``. Used as Control B's
    analytic answer; the mesh's dispersion is compared to ITS OWN polarisation's
    branch.
    """
    lams = np.linspace(0.0, lam_max, n_scan)
    out = []
    for pol_index in (0, 1):
        traces = np.array([
            layered_traces(l, ky, eps2, fill, eps1, a)[pol_index] for l in lams
        ])
        real = np.real(traces)
        # A band exists where |trace/2| <= 1; edges are where it crosses 1.
        f = real / 2.0
        sign = np.sign(f - 1.0)
        edges = lams[np.where(np.diff(sign) != 0)[0]]
        out.append(edges[:n_bands])
    return out[0], out[1]
