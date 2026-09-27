"""The little that BOTH images need, with dependencies light enough for both.

WHY THIS FILE EXISTS. It is not a tidy-up; it is a crashloop that reached
production. The web image and the solver image are built from two Dockerfiles
that COPY different directories. ``web/config.py`` imported
``mse_maxwell_solver`` for two integer defaults, and ``web/meshing.py`` imported
``mse_meshing`` for the sizing functions -- both living in
``dockerfiles/maxwell_solver/``, which the web Dockerfile does not copy. The
image built, pushed, and died at import:

    /opt/app/config.py:112
    ModuleNotFoundError: No module named 'mse_maxwell_solver'

Every test passed, because the tests put ``dockerfiles/maxwell_solver`` on
``sys.path``. Nothing in the suite started the web IMAGE, so nothing asked the
question the container asks first. The site was down for about six minutes.

THE RULE THIS FILE ENCODES. Anything both images import lives HERE, and this
module may import only what both images already have: numpy and (inside the
functions that need it) trimesh. Neither Dockerfile gains a dependency to share
this, which is what keeps the web image small enough to stay free of the native
toolchain.

WHAT MUST NOT COME HERE. Anything that pulls in the heavy solver stack -- the
tetrahedralizer, pyvista/vtk, cuDSS, the scipy factorisations, the fields and
storage modules. If a helper needs one of those, it belongs in
:mod:`mse_meshing` and the web app must reach it some other way. The one test
that would catch a violation is ``tests/test_web_imports.py``, which checks that
every import reachable from ``web/`` resolves to either a stdlib module, a
pinned requirement, or a file the web Dockerfile actually copies.

IMPORTS GO ONE WAY. ``mse_meshing`` and ``mse_maxwell_solver`` import from this
module; this module imports from neither. That is what makes it safe for the
web image to import directly, and it is why the solver modules re-export these
names rather than defining their own -- there is exactly one definition of each,
and the re-export is a binding, not a copy. A second definition would be two
sources of truth for the element cap, which is the defect class the cap's own
docstring already describes.
"""

from __future__ import annotations

import os

import numpy as np

#: unit name -> metres per unit
#:
#: ``um`` and ``nm`` are here because optical structures are routinely specified
#: in them: a 1550 nm telecom cavity is 1.55 um across, which in mm is a number
#: nobody wants to type (0.00155). Adding a unit is not cosmetic -- the solver
#: works in metres throughout, so a wrong scale is a wrong mode frequency by
#: exactly that factor.
#:
#: ``in`` is the odd one out: it is not a metric prefix, and a 1-inch cavity is
#: 0.0254 m rather than 1e-3 of something. It is here because CAD exports use it.
UNIT_SCALE = {
    "m": 1.0,
    "cm": 1e-2,
    "mm": 1e-3,
    "um": 1e-6,
    "nm": 1e-9,
    "in": 0.0254,
}

#: VTK cell id for a linear tetrahedron
VTK_TETRA = 10

#: TetGen's ``maxvolume`` is a volume bound, but the elements it emits with
#: ``minratio``/``mindihedral`` run larger than the bound. Calibrated so that
#: ``target_maxvolume`` actually lands near the requested edge length.
TETGEN_VOLUME_SLACK = 4.0

#: How much larger than the requested edge length the achieved median edge may
#: be before we refuse to solve. A mesh this coarse no longer resolves the
#: requested modes, and the frequencies would be wrong without anything failing.
MAX_EDGE_OVERSHOOT = 1.6

#: default element cap. Chosen so a worst-case solve stays well under the 14Gi
#: container limit; the lead calibrates the real figure on the Jetson.
#:
#: THE ONE SOURCE OF TRUTH for this number. ``web/config.py`` imports it rather
#: than keeping its own copy, and both the cluster Job env and the worker's
#: claim payload carry the value this function resolves. It used to be 80_000
#: here while the app capped uploads at 600_000, with nothing passing the app's
#: number to the solver: every job the app accepted above 80k tets -- exactly
#: the large jobs the GPU exists for -- died in the solver on a limit the app
#: had already told the owner was fine. Two constants that must agree and are
#: never compared is the whole defect class; there is now one constant.
#:
#: It lives in THIS module rather than in the solver's entrypoint for the same
#: reason it exists at all: the app must be able to read it without importing
#: the solver, and the solver must not keep a second copy for the app to drift
#: from.
DEFAULT_MAX_TETS = 600_000

#: elements above which the CLUSTER cannot solve a job, but the GPU still can.
#: The upload-time estimate decides the route, but an estimate can be wrong in
#: either direction; this is the exact count re-deciding, at the first point the
#: exact count exists. Deliberately lower than DEFAULT_MAX_TETS: it is a
#: routing boundary, not a capability limit.
DEFAULT_CPU_MAX_TETS = 150_000

#: what a ``TooBigForProcessor`` writes into its termination message. The app
#: reads this string back out of the pod's last state to tell "this job is on
#: the wrong machine" from "this job is broken" -- see the class docstring in
#: ``mse_maxwell_solver``.
#:
#: Defined here because the APP has to match it and the app cannot import the
#: solver. A token the two sides spell out separately is a token that can drift
#: to a string nobody writes and nobody reads.
REROUTE_TOKEN = "MSE_REROUTE"

#: How much headroom over Weyl's law to allow. Weyl's count is asymptotic and
#: mildly under-predicts for a small cavity (for the 10x7x5 box at N=20 it gives
#: k = 1.19/mm against a true 1.262/mm, about 6%), so the geometric estimate is
#: scaled by this before it becomes a wavelength.
WEYL_SAFETY = 1.2


class MeshingError(Exception):
    """A failure the user can act on (bad file, open surface, mesh too big)."""


def unit_scale(units: str) -> float:
    """Metres per unit for one of the supported unit names."""
    try:
        return UNIT_SCALE[units]
    except KeyError:
        raise MeshingError(
            f"unsupported units {units!r}; expected one of {sorted(UNIT_SCALE)}"
        ) from None


def apply_scale(surface, scale: float):
    """Return ``surface`` with every vertex multiplied by ``scale``.

    This is the whole of the free scale factor: the object is scaled before
    anything else looks at it, so the volume, the surface edge, the target edge
    length and the decimation all see the same size -- which is the only way
    ``est_tets`` computed in the app can agree with the mesh the solver builds.

    Scaling leaves the shape exactly similar, so the ``scale``-free quantities
    (the *ratio* of surface edge to target edge, the face count decimation
    chooses) are unchanged; only the quantities with a length dimension move,
    and they move together. That is why this can be a single multiply at the
    edge and not a parameter threaded through the sizing functions.
    """
    if surface is None:
        return None
    factor = float(scale)
    if not np.isfinite(factor) or factor <= 0:
        raise MeshingError(f"scale must be a positive number, not {scale!r}")
    if factor == 1.0:
        # Deliberately the same object: an unscaled upload must behave exactly
        # as it did before this parameter existed, copy included.
        return surface
    scaled = surface.copy()
    scaled.vertices = np.asarray(scaled.vertices, dtype=float) * factor
    return scaled


def enclosure_volume_m3(surface, units: str) -> float:
    """Volume actually enclosed by the surface, in cubic metres.

    NOT the bounding box: Weyl's law counts modes per unit *enclosed* volume, so
    a hollow or non-convex object needs its own volume or the mesh comes out far
    too fine. Falls back to the box volume when trimesh cannot produce a signed
    volume (a degenerate or nearly-closed surface), because a coarse estimate
    only costs accuracy, while raising here would refuse a usable upload.
    """
    scale = unit_scale(units)
    try:
        volume = float(abs(surface.volume)) * scale ** 3
    except Exception:  # noqa: BLE001 - trimesh raises several things here
        volume = 0.0
    if not np.isfinite(volume) or volume <= 0:
        bounds = np.asarray(surface.bounds, dtype=float)
        extent = bounds[1] - bounds[0]
        volume = float(np.prod(extent)) * scale ** 3
    if not np.isfinite(volume) or volume <= 0:
        raise MeshingError("this object has no enclosed volume to mesh")
    return volume


def target_edge_length(
    surface,
    n_modes: int,
    units: str,
    n_index: float = 1.0,
) -> float:
    """Element edge length (in the user's units) that resolves the top mode.

    The mode *count* in three dimensions follows Weyl's law,
    ``N(k) ~ V k^3 / (3 pi^2)``, so the k of the N-th mode is

        k_N ~ (3 pi^2 N / V)^(1/3)

    with V the enclosed volume. k grows like the CUBE ROOT of the mode number,
    not linearly -- an earlier version of this function used
    ``k_high = n_modes * pi / D`` (D = bbox diagonal), which over-refines badly:
    for the 10x7x5 box at 20 modes it asked for k = 4.76/mm when the true 20th
    mode is at 1.262/mm, about 4x too fine in edge length and ~60x too many
    elements.

    n_index deliberately does not enter: the geometric k is independent of the
    filling medium (the in-medium wavelength is what changes, and that is
    handled downstream). The parameter is kept for call compatibility.
    """
    scale = unit_scale(units)
    volume_m3 = enclosure_volume_m3(surface, units)
    count = max(int(n_modes), 1)

    k_high = WEYL_SAFETY * (3.0 * np.pi ** 2 * count / volume_m3) ** (1.0 / 3.0)
    wavelength_m = 2.0 * np.pi / max(k_high, 1e-30)

    edges_per_wavelength = float(os.getenv("MSE_EDGES_PER_WAVELENGTH", "7"))
    edge_m = wavelength_m / edges_per_wavelength
    return edge_m / scale                            # back to the user's units


def effective_edge_m(
    scaled_surface,
    n_modes: int,
    units: str,
    min_feature_m: float | None = None,
    n_index: float = 1.0,
    physics_m: float | None = None,
) -> float:
    """The edge length the mesh is built to, in METRES on the scaled surface.

    This is the ONE place the physics edge and the owner's declared smallest
    feature are compared. Both callers -- ``tetrahedralize`` and the web app's
    ``estimate_tets`` -- call it and neither re-derives the comparison, because
    the whole class of bug this replaces was two call sites combining the same
    two numbers in two different length spaces.

    The arguments are already in the one space that matters:

    * ``scaled_surface`` has been through :func:`apply_scale`, so every length
      measured on it is the object the solver was actually asked to solve.
    * ``min_feature_m`` is METRES, converted once at the edge of the system
      (the upload form, from its own unit selector) and carried in metres from
      there. It is NOT converted here -- that is deliberate. The parameter used
      to be called ``min_feature_mm`` and arrived as a bare number in the
      object's units, which meant every reader had to know *which* unit it was
      and every one of them could get it wrong. A value that is already metres
      cannot be converted wrongly.

    ``target_edge_length`` returns the user's units, so it is converted here --
    ``physics_m`` is the only length in this function that needs a factor, and
    it needs exactly one.

    ``physics_m`` may be passed in by a caller that already computed it. The
    solver's tetrahedralizer does, because it needs the same number for its own
    verbose line AND needs the call to go through ``mse_meshing``'s module
    global so the resolution the guard tests can be scripted. Passing it keeps
    one evaluation instead of two that could disagree at the last bit -- and the
    ``min`` below is exactly where a last-bit disagreement would show up as two
    different edges for one object.
    """
    if physics_m is None:
        physics_m = target_edge_length(
            scaled_surface, n_modes=n_modes, units=units, n_index=n_index
        ) * unit_scale(units)

    if min_feature_m is None:
        return physics_m
    feature_m = float(min_feature_m)
    if not np.isfinite(feature_m) or feature_m <= 0:
        # A nonsense feature is ignored rather than divided by. Same rule as
        # the app's validation, applied here too because this function is
        # reachable from the solver's own CLI.
        return physics_m
    # Whichever is SMALLER wins: a feature left unresolved and a top mode left
    # unresolved both give frequencies that are confidently wrong.
    return min(physics_m, feature_m)


def target_maxvolume(surface, edge_length: float) -> float:
    """TetGen ``maxvolume`` that gives roughly the requested edge length.

    TetGen's ``maxvolume`` is a bound on element volume, not edge length. For an
    equilateral-ish tet of edge ``h`` the volume is ``h^3 / (6 sqrt(2))``; TetGen
    with ``minratio``/``mindihedral`` produces elements somewhat larger than that
    bound, so we scale by ``TETGEN_VOLUME_SLACK``. Calibrated against real
    tetrahedralizations: the exact factor only needs to be within ~2x, because
    the achieved edge length is what the accuracy depends on and the hard
    ``MSE_MAX_TETS`` cap is enforced on the count TetGen actually returns.
    """
    h = float(edge_length)
    return TETGEN_VOLUME_SLACK * (h ** 3) / (6.0 * np.sqrt(2.0))


def estimate_tet_count(surface, edge_length: float) -> tuple[int, float]:
    """Rough ``(n_tets, volume)`` estimate for a mesh of the given edge length.

    The volume is the bounding-box volume, which overestimates a non-convex
    object but is far better behaved than an ``A^(3/2)`` guess (that formula is
    off by ~2.6x for a plain box). It only needs to be the right order of
    magnitude: the real count is read back from TetGen and capped there.
    """
    bounds = np.asarray(surface.bounds, dtype=float)
    volume = max(float(np.prod(bounds[1] - bounds[0])), 1e-18)
    maxvolume = target_maxvolume(surface, edge_length)
    return int(volume / max(maxvolume, 1e-30)), volume
