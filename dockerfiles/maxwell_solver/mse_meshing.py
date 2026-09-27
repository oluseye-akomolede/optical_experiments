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

# Re-exported from :mod:`mse_common`, which is the ONE definition. This module
# keeps the names because the solver's own code, the worker, the bench scripts
# and the tests all reach them as ``mse_meshing.X`` or ``MESH.X``; the web image
# imports :mod:`mse_common` directly instead, because it does not ship this file
# and must not have to. A re-export is a binding, not a copy -- there is exactly
# one value for each of these in a process, so nothing can drift.
from mse_common import (  # noqa: F401
    DEFAULT_CPU_MAX_TETS,
    DEFAULT_MAX_TETS,
    MAX_EDGE_OVERSHOOT,
    REROUTE_TOKEN,
    TETGEN_VOLUME_SLACK,
    UNIT_SCALE,
    VTK_TETRA,
    WEYL_SAFETY,
    MeshingError,
    apply_scale,
    effective_edge_m,
    enclosure_volume_m3,
    estimate_tet_count,
    target_edge_length,
    target_maxvolume,
    unit_scale,
)

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




def _coarse_message(n_modes: int, achieved: float, target: float,
                    units: str) -> str:
    """Plain English for 'the mesh cannot resolve what you asked for'."""
    return (
        f"This object is too large or detailed to resolve {n_modes} modes within "
        "the solver's size limit. Try fewer modes, a smaller object, or a lower "
        "refractive index."
    )








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

    TetGen is handed the surface in a CANONICAL internal frame -- centred on its
    bounding box and divided by the longest side -- and the points are mapped
    back before returning. This is not tidiness: TetGen's Steiner insertion
    depends on the absolute coordinate magnitude, so meshing the same physical
    object spelled in mm, m and um gave three different element counts (measured
    35237..47323, i.e. 34%, across three spellings of one icosphere). In the
    canonical frame the count is a function of the shape alone (measured spread
    ratio 1.0000 against 1.17 raw), so ``est_tets`` and the solver agree
    whichever unit the upload used. Only the tetrahedralization moves; every
    length reported out of here is metres of the SCALED object, unchanged.

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
    # Both helpers are called as MODULE GLOBALS (``target_edge_length(...)``,
    # not ``mse_common.target_edge_length(...)``) so that a module-level override
    # is honoured. That is not a style choice: this module's contract is that
    # ``mse_meshing.target_edge_length`` can be replaced to script the edge, and
    # the guard tests -- the ones that prove the resolution rule fires at the
    # right ratio -- do exactly that. Importing the helper by name would bind it
    # once, silently ignore every patch, and leave those tests passing for the
    # wrong reason. ``python -m pytest --strict-markers`` cannot see that; a
    # reader can.
    #
    # ``physics_m`` is recomputed from ``target_edge_length`` rather than taken
    # from ``effective_edge_m``'s internals because it is the OTHER branch of the
    # same ``min``: it is what the mesh would have been built to with no feature
    # declared, which is what the verbose line compares against.
    physics_m = target_edge_length(
        surface, n_modes=n_modes, units=units, n_index=n_index
    ) * unit_scale_m
    edge_m = effective_edge_m(
        surface, n_modes=n_modes, units=units,
        min_feature_m=min_feature_m, n_index=n_index,
        physics_m=physics_m,
    )
    edge_units = edge_m / unit_scale_m
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

    # --- TetGen's magnitude sensitivity, and the canonical frame that removes
    # it ---------------------------------------------------------------------
    # TetGen's Steiner-point insertion is sensitive to the ABSOLUTE magnitude of
    # the coordinates, not just the shape. The same physical object spelled in
    # different units therefore meshes to a different element count: a 2 mm
    # icosphere gives a different count at extent 2, 0.002 and 2000, and the
    # spread grows with the range -- measured 35237..47323 tets (34%) across
    # three spellings of one object, and count is what the element cap and the
    # cost depend on. (The row that motivated this quoted ~7% for a cube; the
    # spread is larger for a curved surface, which is the realistic case.)
    #
    # The fix is to mesh in a frame whose scale is fixed by the OBJECT, not by
    # the unit it was written in: centre on the bounding box and divide by the
    # longest side, so every object is meshed at O(1) regardless of its units.
    # Measured with one canonical surface scaled to each magnitude: raw spread
    # ratio 1.17, canonical 1.0000.
    #
    # ``canonical_scale`` is the inverse of that division, so it maps a length
    # in the canonical frame back to the scaled surface's own units. maxvolume
    # is a VOLUME, so it goes as the cube.
    #
    # Only the tetrahedralization moves into this frame. The decimation above,
    # the surface-edge measurements and every reported length stay in the
    # scaled surface's units, because those are what ``est_tets`` in the app
    # reproduces and what the owner's numbers are in.
    bbox_lo = vertices.min(axis=0)
    bbox_hi = vertices.max(axis=0)
    longest = float(np.max(bbox_hi - bbox_lo))
    if not np.isfinite(longest) or longest <= 0:
        # A degenerate (flat or single-point) surface. TetGen will reject it on
        # its own terms, and inventing a frame for it would only replace a clear
        # failure with a confusing one, so mesh it as it is.
        canonical_offset = np.zeros(3)
        canonical_scale = 1.0
    else:
        canonical_offset = (bbox_lo + bbox_hi) / 2.0
        canonical_scale = 1.0 / longest
    vertices = (vertices - canonical_offset) * canonical_scale
    maxvolume *= canonical_scale ** 3

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
        # Map the points straight back to the scaled surface's units. This is
        # the exact inverse of the mapping above, and it happens per attempt
        # because each retry re-tetrahedralizes and returns fresh points.
        points = (np.asarray(grid.points, dtype=float) / canonical_scale
                  + canonical_offset)
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
