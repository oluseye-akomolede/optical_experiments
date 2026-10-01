"""The numpy-only geometry of a photonic-crystal unit cell.

WHY THIS FILE EXISTS. ``pc.py`` is the SOLVER: at module level it imports
``scipy.sparse`` (and ``scipy.spatial`` inside its triangulation). The web image
deliberately ships no scipy -- ``dockerfiles/rendercheck/image_smoke.sh`` asserts
that image contains no graph engine -- so any web module that imported ``pc``
died at import with ``ModuleNotFoundError: No module named 'scipy'``.

The web form is not a solver. It needs three things from this geometry and none
of the solve: the high-symmetry path (how many k-points a run costs), the lattice
vectors, and the number of nodes the worker's mesh will have -- because that
count is the cap the owner is refused on, so it must be the SOLVER's own count
rather than a second estimate that could disagree with it.

So the numpy-only half lives here, and BOTH images import it: the solver
(``pc.py``, which adds the triangulation and the eigenproblem on top) and the web
(``pc_form.py``, which stops at the point set). Splitting it is what lets the web
image keep its no-scipy property structurally: there is no import path from the
web image's pc geometry to scipy, rather than a convention that pc.py will not
grow a module-level scipy import -- which is exactly the convention that broke.
"""

from __future__ import annotations

import numpy as np

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


def snapped_fill(fill: float, divisions: int) -> float:
    """The grid fill a stripe at ``fill`` is actually built at.

    The stripe's interface is conforming only on a node column ``u = k/divisions``
    (see :func:`unit_cell_mesh`), so the requested fill is snapped to the nearest
    one, clamped to ``[1, divisions-1]`` so a stripe always has non-zero width on
    both sides. This is public because the mesh and the 1-D oracle must agree on
    it: asking the oracle for the un-snapped fill compares two different
    geometries.
    """
    if divisions < 1:
        raise PCError(f"divisions must be positive, got {divisions}")
    k = int(round(fill * divisions))
    k = min(max(k, 1), divisions - 1)
    return k / divisions


def unit_cell_points(lattice: str, divisions: int, r_over_a: float = 0.0,
                     n_theta: int = 0, geometry: str = "rod",
                     fill: float = 0.5):
    """The unit cell's node cloud, BEFORE triangulation.

    Returns ``(points, a1, a2, fill)``: the point set :func:`pc.unit_cell_mesh`
    triangulates, the lattice vectors, and the EFFECTIVE fill (the stripe's
    requested fill snapped onto a node column -- see :func:`snapped_fill`).

    This is the whole of the mesh except the Delaunay, and it is what the web
    form counts nodes from. Triangulation is the caller's: the solver needs the
    triangles, the web form needs one integer, and only the solver should pay for
    ``scipy``.

    The guards are here, not in the caller, so the form and the solver refuse the
    SAME requests with the SAME message: the cap the owner sees is the mesh the
    worker builds, or neither.
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
    if geometry == "stripe":
        # SNAP the fill to the grid. The stripe's interface is a straight line
        # parallel to a2, and on a STRUCTURED grid the only way it is conforming
        # is if it lands exactly on a node column u = k/divisions. Inserting
        # partial columns instead does not work: the line's end nodes fall on
        # the cell's v = 0 / v = 1 faces, where they join the periodic pairing
        # and a triangle still crosses (measured: 2-3 straddlers at fill = 0.31,
        # 0.45, 0.62 with insertion). Snapped, the column exists in the grid and
        # the count is 0 at every fill tested.
        #
        # This is a REAL geometry change, so the effective fill is exposed as
        # :func:`snapped_fill` and every consumer of the mesh -- the solver and
        # the layered oracle in Control B -- must call it before comparing the
        # two. Otherwise the mesh is built at k/divisions while the oracle is
        # asked for the requested fill: two different problems, which is the one
        # failure mode this control exists to catch.
        fill = snapped_fill(fill, divisions)
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
    extra = []             # the rod's own nodes, appended to the grid
    interface_dist = None  # distance of every GRID point from the interface
                           # (rod only -- the stripe needs neither)

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
        # The stripe needs NO inserted nodes: its interface is snapped to a node
        # column u = k/divisions (see ``snapped_fill``), so the grid already
        # conforms and no triangle crosses it. ``extra`` stays empty, so the drop
        # below is a no-op and the point set is exactly the structured grid.
        pass

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
        # Only the rod reaches here: the stripe's interface is snapped onto a
        # node column, so its ``extra`` is empty and the grid is already
        # conforming. The rod's grid points within h/2 of the circle are dropped
        # (they would steal a chord) but the cell's own boundary nodes are kept.
        keep = (np.abs(interface_dist) > 0.5 * h) | on_face
        if keep.sum() == 0:
            # an inclusion filling the whole cell interior -- keep the closest
            # node rather than collapse to a boundary-only triangulation
            keep[np.argmin(np.abs(interface_dist))] = True
        pts = np.vstack([pts[keep], extra[0]])

    return pts, a1, a2, fill
