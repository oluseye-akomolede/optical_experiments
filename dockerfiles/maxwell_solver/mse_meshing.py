"""Surface loading/cleanup and tetrahedralization for the MSE Maxwell solver.

Input is a user-uploaded .obj. The solver needs a *closed* surface: the PEC
boundary condition is enforced on the boundary triangles, and an open surface
would leave the cavity leaking. We therefore clean and merge the surface, and
refuse (with a human-readable message) anything that is not watertight.

The mesh is scaled to metres before it ever reaches the physics, so every
downstream quantity (k, frequency, wavelength) is in SI.

Two things about element size are worth knowing before changing this file:

1. Resolution is driven by ``maxvolume`` from Weyl's law -- and ``maxvolume``
   only takes effect together with ``fixedvolume=True``.
2. ``maxvolume`` sets the *interior* element size only as far as the surface
   allows. TetGen cannot make interior elements much larger than the surface
   facets, so a surface finer than the target edge floors the whole mesh no
   matter what ``maxvolume`` says. Real uploads (CAD exports, 3-D scans) are
   routinely several times finer than the target, which is why
   :func:`decimate_to_edge_length` coarsens the surface before tetrahedralizing.
   Without it every such upload was refused as "too large", even a sphere.
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

#: How many times to re-tetrahedralize while trying to fit the element cap.
MAX_MESH_ATTEMPTS = 4

#: A surface whose median triangle edge is below this fraction of the target
#: element edge floors the whole mesh: TetGen cannot make interior elements much
#: larger than the surface facets, so a fine surface forces a fine interior no
#: matter what ``maxvolume`` says. Owner uploads from CAD or a 3-D scan are
#: routinely far finer than the target (0.26x on an off-the-shelf icosphere),
#: which is why decimating has to happen before Tetrahedralize, not after.
SURFACE_DECIMATE_TRIGGER = 0.7

#: Raise ``maxvolume`` and the element count should fall by roughly the same
#: factor. If a pass changes the count by less than this fraction, the surface
#: is what is setting the element size and no further pass can help -- stop
#: instead of spending three full tetrahedralizations discovering that.
CAP_RETRY_RESPONSE = 0.20


class MeshingError(Exception):
    """A failure the user can act on (bad file, open surface, mesh too big)."""


def _coarse_message(n_modes: int, achieved: float, target: float,
                    units: str) -> str:
    """Plain English for 'the mesh cannot resolve what you asked for'."""
    return (
        f"This object is too large or detailed to resolve {n_modes} modes within "
        "the solver's size limit. Try fewer modes, a smaller object, or a lower "
        "refractive index."
    )


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


def effective_edge_m(
    scaled_surface,
    n_modes: int,
    units: str,
    min_feature_m: float | None = None,
    n_index: float = 1.0,
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
    """
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


def load_surface(obj_bytes_or_path, units: str = "mm"):
    """Load and clean an .obj surface, returning ``(trimesh.Trimesh, metres_per_unit)``.

    The returned mesh is in the *user's* units; callers scale to metres. Merging
    coincident vertices is what makes a nominally-closed surface actually
    watertight, which is the check the solver depends on.
    """
    import trimesh

    if isinstance(obj_bytes_or_path, (bytes, bytearray)):
        import io

        try:
            mesh = trimesh.load(io.BytesIO(bytes(obj_bytes_or_path)), file_type="obj")
        except Exception as exc:  # noqa: BLE001 - surface the parser's own words
            raise MeshingError(f"could not parse this .obj file: {exc}") from exc
    else:
        try:
            mesh = trimesh.load(str(obj_bytes_or_path), file_type="obj")
        except Exception as exc:  # noqa: BLE001
            raise MeshingError(f"could not parse this .obj file: {exc}") from exc

    # trimesh may hand back a Scene for multi-object objs; take the biggest part.
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.faces)]
        if not geometries:
            raise MeshingError("this .obj contains no faces")
        mesh = max(geometries, key=lambda g: len(g.faces))

    if mesh is None or len(mesh.faces) == 0:
        raise MeshingError("this .obj contains no faces")

    mesh = mesh.copy()
    # weld duplicate vertices, then drop degenerate and duplicate faces
    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()

    if len(mesh.faces) == 0:
        raise MeshingError("this .obj has no usable triangles after cleanup")

    if not mesh.is_watertight:
        raise MeshingError(
            "this object is not a closed surface (it has holes or open edges), so "
            "it cannot be treated as a cavity. Please upload a watertight .obj."
        )

    return mesh, unit_scale(units)


def _surface_area(vertices: np.ndarray, faces: np.ndarray) -> float:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    return float(np.sum(np.linalg.norm(cross, axis=1)) / 2.0)


def median_surface_edge(surface) -> float:
    """Median triangle edge length of the surface, in the surface's own units.

    The median, not the mean: a decimated surface is deliberately bimodal (a
    box at 284 triangles still has long thin ones along the face diagonals), and
    the mean is dragged by the few very long edges while the median tracks the
    typical facet the interior has to attach to.
    """
    edges = getattr(surface, "edges_unique_length", None)
    if edges is None or len(edges) == 0:
        return 0.0
    return float(np.median(edges))


def target_face_count(surface, edge_length: float) -> int:
    """Triangles an equilateral triangulation of this surface needs at ``h``.

    ``faces ~ area / (sqrt(3)/4 * h^2)`` -- the task's formula, which is the
    area of an equilateral triangle of edge ``h``. Note this is the *ideal
    isotropic* count: a quadric decimation lands somewhat coarser than the
    formula for a given face number on a shape with long thin facets, which is
    fine because the caller re-measures the achieved edge and steps.
    """
    faces = getattr(surface, "faces", None)
    if faces is None or len(faces) == 0:
        return 0
    area = _surface_area(np.asarray(surface.vertices, dtype=float),
                         np.asarray(surface.faces, dtype=np.int64))
    denom = (np.sqrt(3.0) / 4.0) * float(edge_length) ** 2
    if denom <= 0:
        return len(faces)
    return int(area / denom)


def decimate_surface(surface, want_faces: int):
    """Quadric-decimate a surface to ``want_faces``.

    Returns a new ``trimesh.Trimesh``, or ``None`` when the decimation produced
    nothing usable -- the caller must then keep the surface it already had,
    because a missing mesh is not a better outcome than a fine one.

    pyvista's ``PolyData.decimate`` takes a **reduction fraction**, not a face
    count (its docstring says "fraction of the original mesh to remove"). An
    integer above 1 is passed straight through to ``SetTargetReduction`` and
    returns an empty mesh with no error, so the conversion is done here and the
    result is checked.
    """
    import pyvista as pv
    import trimesh

    faces = np.asarray(surface.faces, dtype=np.int64)
    n_faces = int(faces.shape[0])
    want = int(np.clip(int(want_faces), 4, n_faces))
    if want >= n_faces:
        return surface

    points = np.asarray(surface.vertices, dtype=float)
    # VTK's cell array is [3, i, j, k, 3, i, j, k, ...]; int64 works, but the
    # padded form is what PolyData expects (a flat int array is misread as a
    # connectivity list and raises CellSizeError).
    cells = np.column_stack(
        [np.full(n_faces, 3, dtype=np.int64), faces]
    ).ravel()

    try:
        pd = pv.PolyData(points, cells)
        reduced = pd.decimate(1.0 - want / n_faces)
        out_faces = np.asarray(reduced.faces, dtype=np.int64)
        out_points = np.asarray(reduced.points, dtype=float)
    except Exception:  # noqa: BLE001 - a decimation failure is not fatal
        return None

    if out_faces.size == 0 or out_points.shape[0] == 0:
        return None

    decimated = trimesh.Trimesh(
        out_points, out_faces.reshape(-1, 4)[:, 1:], process=False
    )
    if len(decimated.faces) == 0:
        return None

    # The watertightness the whole PEC boundary condition rests on. A gentler
    # attempt may still succeed, so the caller retries rather than giving up.
    decimated.merge_vertices()
    decimated.update_faces(decimated.nondegenerate_faces())
    decimated.remove_unreferenced_vertices()
    if not decimated.is_watertight:
        return None
    return decimated


def decimate_to_edge_length(surface, edge_length: float, verbose: bool = True):
    """Coarsen an over-fine surface toward ``edge_length``.

    Returns ``(surface, note)``. ``note`` is a short human-readable line for the
    log, or ``None`` when nothing was needed. The original surface is returned
    unchanged whenever decimation cannot do better, because a fine mesh is
    slower, not wrong -- only an *unmanageable* one is refused, and that is
    decided later by the element cap.

    The loop steps toward the target instead of jumping to it. A single quadric
    pass all the way to the target edge is what tears curved surfaces open, and
    the achieved edge is re-measured each time because the relation between
    face count and achieved edge is only approximate.
    """
    current = surface
    current_edge = median_surface_edge(current)
    if current_edge <= 0 or edge_length <= 0:
        return surface, None

    ratio = current_edge / edge_length
    if ratio >= SURFACE_DECIMATE_TRIGGER:
        # Facets are already at or above the target; TetGen refines the interior
        # itself (facet Steiner points), so nothing to do.
        return surface, None

    original_edge = current_edge
    original_faces = len(current.faces)

    # ONE pass, to the isotropically-correct face count for the target edge.
    # Stepping toward the target over several passes (gentler, then bolder) was
    # tried and is worse: re-decimating an already-decimated surface moves the
    # achieved edge unpredictably, and on a moderately fine sphere the first
    # pass already lands within 2% of the target, so extra passes only add risk.
    #
    # The face count is a request, not a promise -- quadric decimation is driven
    # by geometric error, so a flat-surfaced shape can collapse far past it (a
    # 114k-face box asked for 58k came back with 1,816). That is acceptable: the
    # requested count is the isotropically right one, and the *outcome* is
    # checked downstream, where the achieved edge is measured and an
    # under-resolved interior is refused rather than reported as fact.
    want = target_face_count(current, edge_length)
    if want <= 0 or want >= len(current.faces):
        return surface, None

    candidate = decimate_surface(current, want)
    if candidate is None:
        # The quadric pass opened the surface. Try a gentler reduction before
        # giving up: a surface we can coarsen a little is better than none.
        candidate = decimate_surface(current, (len(current.faces) + want) // 2)
    if candidate is None:
        note = (
            f"could not coarsen this surface to the target edge without opening "
            f"it; meshing the original {original_faces:,} triangles instead"
        )
        return surface, note

    current, current_edge = candidate, median_surface_edge(candidate)
    note = None

    if current is surface:
        return surface, note

    ratio_after = current_edge / edge_length
    if ratio_after < SURFACE_DECIMATE_TRIGGER:
        # Coarsened as far as it wants to go and still under the target. This is
        # the honest outcome: the mesh may still be refused by the element cap,
        # and the cap message is what the user needs to see.
        note = (
            f"surface could only be coarsened to {current_edge:.4g} "
            f"({ratio_after:.2f}x the target edge {edge_length:.4g})"
        )
    elif note is None:
        note = (
            f"coarsened the surface from {original_faces:,} to "
            f"{len(current.faces):,} triangles (median facet edge "
            f"{original_edge:.4g} -> {current_edge:.4g}, target "
            f"{edge_length:.4g})"
        )
    return current, note


#: How much headroom over Weyl's law to allow. Weyl's count is asymptotic and
#: mildly under-predicts for a small cavity (for the 10x7x5 box at N=20 it gives
#: k = 1.19/mm against a true 1.262/mm, about 6%), so the geometric estimate is
#: scaled by this before it becomes a wavelength.
WEYL_SAFETY = 1.2


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


# The median edge the LAST tetrahedralize() actually achieved, in metres on
# the scaled object -- the measurement, as opposed to the target it was
# asked for. The solver writes it into modes.json so the job page can show
# the owner what they got next to what they asked for. Module state, the
# same shape as mse_eigen.LAST_WORST_RESIDUAL: it describes the run that
# just happened, and tetrahedralize returns (points, tets) so there is
# nowhere else for it to travel.
LAST_ACHIEVED_EDGE_M: float | None = None


def tetrahedralize(
    surface,
    units: str,
    n_modes: int = 20,
    n_index: float = 1.0,
    max_tets: int = 150_000,
    verbose: bool = True,
    enforce_resolution: bool = True,
    min_feature_m: float | None = None,
    scale: float = 1.0,
):
    """Tetrahedralize a cleaned surface and return ``(points_m, tets)`` in metres.

    Resolution is driven by TetGen's ``maxvolume`` -- **which only takes effect
    together with ``fixedvolume=True``**. Without ``fixedvolume`` the Python
    wrapper's ``maxvolume`` is ignored entirely (a 10x7x5 box yields 6 tets at
    any ``maxvolume``), which is why an earlier version of this module tried to
    pre-refine the surface instead. With ``fixedvolume=True`` TetGen inserts
    interior *and* facet Steiner points itself, so the interior is refined too
    and no surface subdivision is needed.

    Surface edges lying exactly on the requested spacing are inserted as
    interior points rather than retriangulating the facets, which keeps the
    boundary vertices exactly as the user's .obj supplied them -- TetGen does
    not move vertices, so the geometry is preserved.

    ``min_feature_m`` is the owner's declared smallest feature in METRES --
    converted once at the upload form from its own nm/um/mm selector, never
    from the object's units. It lowers the target edge and never raises it.
    ``scale`` multiplies every vertex before anything is measured, so the
    volume, the surface edge and the target edge all describe the object the
    solver was actually asked to solve.

    Every length below is METRES measured on the SCALED surface. An earlier
    version mixed spaces right here -- ``edge_length`` was user units of the
    scaled object while ``min_feature_mm`` was a raw number of the unscaled
    one, and a local ``scale = unit_scale(units)`` rebound this function's
    ``scale`` parameter, so ``allowed_edge`` came out in metres of the
    UNSCALED object against an ``achieved`` in metres of the scaled one. At
    scale 1 the two factors cancel, which is why the guard tests passed and hid
    it. The rename of that local to ``unit_scale_m`` is the fix's own guard:
    the shadowing cannot come back if the names cannot collide.

    ``points_m`` is ``(n_points, 3)`` float64 in metres, ``tets`` is
    ``(n_tets, 4)`` indices into it. Raises :class:`MeshingError` when the
    result would blow the element cap.
    """
    import tetgen

    surface = apply_scale(surface, scale)

    # Metres per user unit, from here down. Named ``unit_scale_m`` and not
    # ``scale``: the parameter above is the owner's free scale factor, and an
    # earlier version rebound the name here, which is how the two length spaces
    # got mixed two lines later.
    unit_scale_m = unit_scale(units)

    # The one comparison of physics edge against declared feature, in metres.
    # ``edge_m`` is what the mesh is built to; ``edge_units`` is the same length
    # in the scaled surface's own units, which is the space TetGen's maxvolume
    # and the decimation speak. Derived from ``edge_m`` rather than recomputed,
    # so the two cannot drift.
    edge_m = effective_edge_m(
        surface, n_modes=n_modes, units=units,
        min_feature_m=min_feature_m, n_index=n_index,
    )
    edge_units = edge_m / unit_scale_m
    physics_m = target_edge_length(
        surface, n_modes=n_modes, units=units, n_index=n_index
    ) * unit_scale_m
    if verbose and min_feature_m is not None and edge_m < physics_m:
        print(
            f"[mse] resolving the {edge_m:g} m feature: target edge "
            f"{physics_m:.4g} m -> {edge_m:g} m",
            flush=True,
        )

    # An over-fine surface floors the interior element size no matter what
    # maxvolume asks for, which is how an ordinary CAD or scan upload gets
    # refused as "too large". Coarsen it toward the target edge first.
    surface, decimation_note = decimate_to_edge_length(
        surface, edge_units, verbose=verbose
    )
    if verbose and decimation_note:
        print(f"[mse] {decimation_note}", flush=True)

    maxvolume = target_maxvolume(surface, edge_units)

    vertices = np.asarray(surface.vertices, dtype=float)
    faces = np.asarray(surface.faces, dtype=np.int32)

    # Both sides in metres of the SCALED object: ``achieved`` below is measured
    # on ``points * unit_scale_m``, so the allowance must be too. The old
    # ``edge_length * scale`` multiplied a user-units edge by a metres-per-unit
    # factor, giving metres of an object this function never meshed.
    allowed_edge = MAX_EDGE_OVERSHOOT * edge_m

    n_tets, tets, points = 0, None, None
    achieved = 0.0
    previous_tets = None
    for attempt in range(MAX_MESH_ATTEMPTS):
        tet = tetgen.TetGen(vertices, faces)
        # minratio/mindihedral keep element quality up; fixedvolume makes
        # maxvolume binding. All three are needed for a usable interior.
        tet.tetrahedralize(
            order=1, mindihedral=20, minratio=2.0,
            maxvolume=maxvolume, fixedvolume=True,
        )
        grid = tet.grid
        tets = np.asarray(grid.cells_dict[VTK_TETRA], dtype=np.int64)
        points = np.asarray(grid.points, dtype=float)
        n_tets = int(tets.shape[0])

        if n_tets == 0:
            raise MeshingError("tetrahedralization produced no elements")

        achieved = achieved_edge_length(points * unit_scale_m, tets)
        over_cap = n_tets > max_tets
        too_coarse = achieved > allowed_edge

        if not over_cap and not too_coarse:
            break

        if too_coarse and not enforce_resolution:
            # The bench's --allow-coarse path: chart the accuracy trade-off
            # against the cap deliberately, and say so, because a mesh this
            # coarse gives frequencies we would otherwise refuse to report.
            if verbose:
                print(
                    f"[mse] WARNING: mesh is {achieved / allowed_edge:.1f}x "
                    f"coarser than the resolution rule allows "
                    f"(achieved median edge {achieved / unit_scale_m:.4g} "
                    f"{units}, limit {allowed_edge / unit_scale_m:.4g} "
                    f"{units}); solving anyway because enforce_resolution was "
                    "disabled",
                    flush=True,
                )
            break

        if too_coarse and over_cap:
            # The mesh cannot resolve the requested modes AND is too big to
            # build. Coarsening is the only lever left and it makes accuracy
            # worse, so there is no path to a correct answer. Refuse, rather
            # than report frequencies we know are wrong.
            raise MeshingError(
                _coarse_message(n_modes, achieved, edge_m, units)
            )

        if too_coarse:
            # Within the cap but too coarse: refining is the fix and there is
            # room for it. MAX_EDGE_OVERSHOOT is a FAILURE, not a warning --
            # wrong frequencies shown with confidence are worse than no answer.
            if attempt == MAX_MESH_ATTEMPTS - 1:
                raise MeshingError(
                    _coarse_message(n_modes, achieved, edge_m, units)
                )
            # tets ~ 1/maxvolume, so scale by the third power of the edge ratio
            ratio = achieved / allowed_edge
            maxvolume /= max(ratio, 1.0) ** 3
            if verbose:
                print(
                    f"[mse] mesh is {ratio:.1f}x too coarse; refining with "
                    f"maxvolume {maxvolume:.4g}",
                    flush=True,
                )
            continue

        # Raising maxvolume did not reduce the count: the SURFACE, not
        # maxvolume, is setting the element size. Another pass costs a full
        # tetrahedralization and cannot succeed, so stop and say so.
        if previous_tets is not None:
            change = (previous_tets - n_tets) / max(previous_tets, 1)
            if change < CAP_RETRY_RESPONSE:
                raise MeshingError(
                    f"this object needs about {n_tets:,} elements, above the "
                    f"{max_tets:,} limit, and the surface is too finely "
                    "triangulated for the solver to make its interior elements "
                    "any larger. Try fewer modes, a smaller object, or a lower "
                    "refractive index."
                )
        previous_tets = n_tets

        if attempt == MAX_MESH_ATTEMPTS - 1:
            raise MeshingError(
                f"this object would need about {n_tets:,} elements, above the "
                f"{max_tets:,} limit. Try a smaller or simpler object."
            )

        # tets scales as 1/maxvolume, so one jump lands near the cap. Aim just
        # under it (1.15) because the estimate is only proportional: the lead
        # measured 573k -> 139k -> 46.7k with the old 1.05 crawl, 24.7s of
        # meshing for a single fit.
        maxvolume *= (n_tets / max_tets) * 1.15
        if verbose:
            print(
                f"[mse] {n_tets:,} tets is over the {max_tets:,} cap; "
                f"retrying with maxvolume {maxvolume:.4g}",
                flush=True,
            )

    points_m = points * unit_scale_m

    if verbose:
        print(
            f"[mse] mesh: {n_tets:,} tets, {points_m.shape[0]:,} points "
            f"(surface {len(vertices):,}), target edge {edge_units:.4g} {units}, "
            f"achieved median edge {achieved / unit_scale_m:.4g} {units}",
            flush=True,
        )

    global LAST_ACHIEVED_EDGE_M
    LAST_ACHIEVED_EDGE_M = float(achieved)
    return points_m, tets


def achieved_edge_length(points_m: np.ndarray, tets: np.ndarray) -> float:
    """Median edge length of the tetrahedralization, in metres."""
    xyz = points_m[tets]
    local = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    lengths = np.linalg.norm(
        xyz[:, local[:, 0], :] - xyz[:, local[:, 1], :], axis=2
    )
    return float(np.median(lengths))
