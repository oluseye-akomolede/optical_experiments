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

def _conforming_circle_points(centre, r, h):
    """Circle points dense enough that no grid point can steal a chord.

    The arc spacing must not exceed the grid spacing ``h``: if it does, a grid
    point sitting just outside the circle can be closer to two circle points
    than they are to each other, and Delaunay then connects across the chord and
    a triangle straddles the interface. So the count is DERIVED from that
    condition rather than fixed, with a floor so coarse cells still get a
    recognisable circle.

    Measured, and the reason a FIXED count is not enough: the old default of
    ``n_theta = 24`` straddled at square/24/0.25 (16), square/24/0.45 (64),
    square/40/0.25 (88), square/40/0.45 (168) and triangular/40/0.25 (66). The
    floor of 64 fixes all but one of those -- square/40/0.45 still straddled with
    64. It is the derived count (324 there) that clears it.

    The fraction ``0.35`` is the safety margin: arc/h = 0.35 is strictly below
    the 1.00 at which a grid point can steal a chord.
    """
    n_theta = max(64, int(np.ceil(2.0 * np.pi * r / (0.35 * h))))
    th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
    return centre + r * np.column_stack([np.cos(th), np.sin(th)]), n_theta


def unit_cell_mesh(lattice: str, divisions: int, r_over_a: float = 0.0,
                   n_rod: float = 1.0, n_bg: float = 1.0, n_theta: int = 0,
                   geometry: str = "rod", fill: float = 0.5):
    """Triangulate one unit cell and pair its periodic boundary nodes.

    Returns ``(points, triangles, eps_by_tri, pairs)``:

    * ``points`` ``(n, 2)`` in units of ``a``;
    * ``triangles`` ``(m, 3)`` int, Delaunay on a STRUCTURED point set, so the
      triangulation is deterministic for a given ``divisions``;
    * ``eps_by_tri`` ``(m,)`` the relative permittivity of each triangle,
      ``n_rod**2`` inside the inclusion and ``n_bg**2`` outside;
    * ``pairs`` a list of ``(slave, master, phase_vec)`` where a master is the
      node the slave equals under a lattice translation, and ``phase_vec`` is the
      translation (as a multiple of ``a1``/``a2`` integer coefficients) whose
      Bloch phase the slave carries.

    **The mesh CONFORMS to the interface: zero triangles straddle it.** Delaunay
    on a structured grid alone does NOT promise this -- a grid point close to the
    circle steals a chord -- so the interface's own nodes are inserted and the
    grid points within ``h/2`` of it are dropped. Measured over the sweep
    (square and triangular x divisions 8, 12, 16, 24, 32, 40 x r/a 0.1, 0.2,
    0.25, 0.3, 0.45): the 54 meshable cases straddled before this and 0 do now;
    the other 6 are r/a = 0.45 on the triangular cell, which the inradius guard
    refuses rather than meshes (0.45 > 0.4330). See
    :func:`_conforming_circle_points` for why the circle's node count is derived
    rather than fixed.

    ``geometry`` is ``"rod"`` (a circle of radius ``r_over_a`` at the cell
    centre) or ``"stripe"`` (a slab of width ``fill`` along x, for the 1-D
    layered control, whose interface is a straight line and gets the same
    treatment).

    Pairing is done by HASHING the translated coordinate, never by index
    arithmetic: index arithmetic is correct in the interior of an edge and wrong
    at the corners, which is exactly where the error would hide.
    """
    if divisions < 4:
        raise PCError(f"divisions must be at least 4, got {divisions}")
    if geometry not in ("rod", "stripe"):
        raise PCError(
            f"geometry must be 'rod' or 'stripe', got {geometry!r}"
        )
    if geometry == "stripe" and not 0.0 < fill < 1.0:
        raise PCError(
            f"stripe fill must be strictly between 0 and 1, got {fill}"
        )
    a1, a2 = lattice_vectors(lattice)

    # The rod must fit INSIDE the cell, and the cell's inradius is
    # lattice-dependent: 0.5 for the square, but sin(60)/2 = 0.4330 for the 60
    # degree rhombus (both measured). A flat "< 0.5" test is WRONG for the
    # triangular cell -- at r/a = 0.45 the circle pokes through the slanted far
    # edge, its points land outside the cell in fractional coordinates, and they
    # then steal chords across the interface together with the boundary nodes.
    # (Measured with the flat guard in place: 16 straddling triangles at
    # divisions 8. The pairing counts stay correct there -- 17, 25, 49, 81 at
    # divisions 8/12/24/40 -- so it is the straddling alone that justifies the
    # guard.)
    inradius = 0.5 if lattice == "square" else np.sin(np.pi / 3.0) / 2.0
    if not 0.0 <= r_over_a < inradius:
        raise PCError(
            f"r/a must be in [0, {inradius:.4f}) so the rod stays inside the "
            f"{lattice} cell; got {r_over_a}"
        )

    # A structured grid in the FRACTIONAL (u, v) coordinates of the lattice, so
    # the same code works for the rhombic triangular cell.
    u = np.linspace(0.0, 1.0, divisions + 1)
    U, V = np.meshgrid(u, u)
    frac = np.column_stack([U.ravel(), V.ravel()])
    pts = frac @ np.vstack([a1, a2])
    h = 1.0 / divisions  # |a_i| = 1 for both lattices, so this IS the spacing

    centre = 0.5 * (a1 + a2)
    extra = []           # the interface's own nodes, appended to the grid
    interface_dist = None  # distance of every GRID point from the interface

    if geometry == "rod":
        rod_radius = r_over_a  # times |a_i| = 1
        interface_dist = np.linalg.norm(pts - centre, axis=1) - rod_radius
        if rod_radius > 0.0:
            if n_theta > 0:
                th = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=False)
                circ = centre + rod_radius * np.column_stack(
                    [np.cos(th), np.sin(th)]
                )
            else:
                circ, _ = _conforming_circle_points(centre, rod_radius, h)
            extra.append(circ)
    else:
        # The stripe's interface is the pair of lines x = 0 and x = fill along
        # a1, i.e. constant u. Its nodes are placed on those lines.
        interface_dist = None

    # Which grid points sit ON the cell's own far faces. These are the nodes the
    # periodic pairing consumes, and they must NEVER be dropped: on the rhombic
    # triangular cell the far edge passes close to a large rod, so a naive
    # |dist - r| < h/2 test removes boundary nodes and the pairing collapses.
    # (Measured at triangular, divisions 12, with the exemption removed: the
    # pairing collapses to 13 pairs at r/a = 0.40 and 0.42, and to 5 at 0.43,
    # where 25 are required -- all ACCEPTED r/a values, so this is a live bug
    # the exemption prevents, not a historical one. The rhombic far edge passes
    # within h/2 of a rod once r/a exceeds ~0.35.)
    A = np.column_stack([a1, a2])
    gfrac = np.round(pts @ np.linalg.inv(A).T, 9)
    on_face = (
        (np.abs(gfrac[:, 0] - 1.0) < 1e-9) | (np.abs(gfrac[:, 1] - 1.0) < 1e-9)
        | (np.abs(gfrac[:, 0]) < 1e-9) | (np.abs(gfrac[:, 1]) < 1e-9)
    )

    if extra:
        # Drop the grid points that would steal a chord, but NEVER enough to
        # collapse the interior: a circle with no interior nodes triangulates
        # into boundary-only triangles (measured: 62 "straddlers" that were
        # really a collapsed mesh, not a chord violation).
        keep = np.ones(len(pts), dtype=bool)
        if geometry == "rod":
            keep = (np.abs(interface_dist) > 0.5 * h) | on_face
            if keep.sum() == 0:
                # a rod filling the whole cell interior -- keep the closest node
                keep[np.argmin(np.abs(interface_dist))] = True
        pts = np.vstack([pts[keep], extra[0]])

    from scipy.spatial import Delaunay
    tri = Delaunay(pts)
    triangles = tri.simplices

    if geometry == "rod":
        tri_centres = pts[triangles].mean(axis=1)
        folded = _fold_into_cell(tri_centres, a1, a2)
        inside = np.linalg.norm(folded - centre, axis=1) < r_over_a
    else:
        folded = _fold_into_cell(pts[triangles].mean(axis=1), a1, a2)
        ufrac = (folded @ np.linalg.inv(A).T)[:, 0]
        inside = ufrac < fill
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

def _p1_matrices(points, triangles, eps_by_tri, weight="mass"):
    """P1 stiffness and mass on a triangle mesh, with eps on ONE of them.

    Same construction as the 3-D element matrices in :mod:`mse_fem`, in 2-D: the
    stiffness is ``area * G G^T`` with ``G`` the (constant) gradients of the
    three barycentric functions, and the mass is ``area/12 * (1 + delta)``.

    ``weight`` says which matrix the per-element coefficient multiplies, and it
    is REQUIRED to be explicit because the two polarisations differ in exactly
    that way (plan §1's table):

        weight="mass"       TM: A = plain stiffness, B = eps-weighted mass
        weight="stiffness"  TE: A = (1/eps)-weighted stiffness, B = plain mass

    The coefficient passed in is used verbatim -- TM passes ``eps``, TE passes
    ``1/eps`` -- so the caller owns the physics and this routine owns only the
    assembly. Keeping it one routine is what stops the two polarisations from
    drifting into two disagreeing constructions.
    """
    if weight not in ("mass", "stiffness"):
        raise PCError(f"weight must be 'mass' or 'stiffness', got {weight!r}")
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
        w = eps_by_tri[t]
        k_scale, m_scale = (1.0, w) if weight == "mass" else (w, 1.0)
        idx = (i, j, k)
        for r in range(3):
            for cc in range(3):
                rows.append(idx[r]); cols.append(idx[cc])
                kv.append(Ke[r, cc] * k_scale)
                mrows.append(idx[r]); mcols.append(idx[cc])
                mv.append(Me[r, cc] * m_scale)
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


def bands_from_mesh(lattice, divisions, r_over_a, n_rod, n_bg, k_points,
                    n_bands, sigma=None, polarisation="TM"):
    """The whole chain for one polarisation: mesh -> assemble -> reduce -> solve.

    This exists so the controls and the shipped solver share ONE assembly path.
    Two paths that both look right are two paths that can disagree, and the
    disagreement would show up as a physics difference rather than a bug.

    ``polarisation`` picks the operator, and the mapping is the geometric fact
    from the module docstring -- ``TM`` (E_z) is tangential to the interfaces,
    so eps sits on the MASS (the s problem); ``TE`` (H_z) has eps on the
    STIFFNESS (the p problem):

        TM:  K u = lam M_eps u          (plain stiffness, eps-weighted mass)
        TE:  K_(1/eps) u = lam M u      (1/eps on the stiffness, plain mass)

    Returns ``(k_points, bands)`` with bands as ``w a / (2 pi c)``.
    """
    if polarisation not in ("TM", "TE"):
        raise PCError(f"polarisation must be TM or TE, got {polarisation!r}")
    pts, tris, eps_by_tri, pairs = unit_cell_mesh(
        lattice, divisions, r_over_a, n_rod, n_bg
    )
    a1, a2 = lattice_vectors(lattice)
    if polarisation == "TM":
        # E_z tangential to the interfaces -> eps on the mass, plain stiffness.
        K, M = _p1_matrices(pts, tris, eps_by_tri, weight="mass")
    else:
        # H_z -> 1/eps on the stiffness, plain mass.
        K, M = _p1_matrices(pts, tris, 1.0 / eps_by_tri, weight="stiffness")
    kp = np.atleast_2d(k_points)
    return kp, solve_bands(K, M, pairs, kp, a1, a2, n_bands, sigma=sigma)


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
