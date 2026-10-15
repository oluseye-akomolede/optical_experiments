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
    TETS_PER_UNIT_VOLUME,
    UNIT_SCALE,
    VTK_TETRA,
    WEYL_SAFETY,
    MeshingError,
    apply_scale,
    effective_edge_m,
    region_plan as region_plan_by_region,
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
                    units: str, region: str | None = None,
                    index: float | None = None) -> str:
    """Plain English for 'the mesh cannot resolve what you asked for'.

    One-part uploads (``region is None``) keep the original wording: a single
    cavity has no region to blame, and there is no finer diagnostic available to
    the user than "too large or detailed".

    A multi-region upload NAMES the failing region instead, because that failure
    is LOCAL: the object can be small enough to mesh comfortably and still hold a
    region -- a dense core -- whose own resolution needs elements far finer than
    the wall around it. ``achieved`` and ``target`` are then BOTH in the user's
    ``units`` (``target`` is the region's OWN required edge, not the wall's).
    "a smaller object" is deliberately absent from this arm: it cannot help,
    because scaling the object scales the region's requirement with it, so the
    ratio the guard reads is size-invariant. What can give is the mode count or
    the region's refractive index. The allowance is INTERPOLATED from
    ``MAX_EDGE_OVERSHOOT`` rather than retyped, so the sentence cannot drift from
    the number the guard actually used.
    """
    if region is None:
        return (
            f"This object is too large or detailed to resolve {n_modes} modes within "
            "the solver's size limit. Try fewer modes, a smaller object, or a lower "
            "refractive index."
        )
    at_index = f" at its {index:g}x refractive index" if index else ""
    return (
        f"The '{region}' region needs elements about {target:g} {units} across to "
        f"resolve {n_modes} modes{at_index}, but they came out about {achieved:g} "
        f"{units} -- past the {MAX_EDGE_OVERSHOOT:g}x the resolution rule allows. "
        "Try fewer modes, or a lower refractive index for that region."
    )


def _resolution_verdict(region_plan, points, tets, attrib, achieved,
                        allowed_edge, unit_scale_m):
    """The resolution rule, applied to EACH REGION's own required edge.

    Returns ``(too_coarse, worst)``. ``too_coarse`` is True when the mesh cannot
    resolve what was asked; ``worst`` describes the region that came out furthest
    past ITS OWN allowance, as

        (ratio, region_name, index, target_units, achieved_units)

    ``ratio`` is achieved / allowance for that region, ``target_units`` is the
    region's OWN required edge -- NOT the wall's -- and ``achieved_units`` the
    measured median edge of its tets, both in the upload's own units, ready for
    the refusal message. ``region_name`` is ``None`` for a one-part upload, where
    there is no region to name and the message must not invent one.

    WHY PER REGION. The guard used to compare the WHOLE mesh's median edge
    against ``MAX_EDGE_OVERSHOOT * edge_m`` -- the WALL's allowance -- for every
    upload. But ``maxvolume`` comes from ``h_min``, the finest region's edge, and
    a cap retry can raise ``maxvolume`` until a dense region is built several
    times coarser than its own rule while the wall's median still sits inside the
    wall's allowance. A high index is exactly where that hurts: 1.6x the wall is
    5.6x the rule for an n=3.5 core. Each region is now judged against
    ``MAX_EDGE_OVERSHOOT * own_edge_units`` (the per-region rule that
    ``mse_common.region_edges`` states), and the verdict is the WORST region in
    the mesh. The wall itself is unchanged -- its own edge IS ``edge_m``, so its
    allowance is the one the old global check applied, and an upload whose densest
    part is the wall reduces to exactly the old test.

    A region that received no tets is skipped rather than scored 0: TetGen
    returns one attribute per tet, so an empty region simply has none, and the
    region that IS present is still judged.
    """
    if region_plan is None or attrib is None:
        ratio = achieved / allowed_edge if allowed_edge > 0 else 0.0
        own_edge_units = allowed_edge / MAX_EDGE_OVERSHOOT / unit_scale_m
        return ratio > 1.0, (ratio, None, None, own_edge_units,
                             achieved / unit_scale_m)

    worst = None
    for region_id, info in region_plan["shared"].items():
        mask = attrib == int(region_id)
        if not mask.any():
            continue
        achieved_m = achieved_edge_length(points * unit_scale_m, tets[mask])
        own_units = float(info["own_edge_units"])
        allowed_units = MAX_EDGE_OVERSHOOT * own_units
        ratio = (achieved_m / unit_scale_m / allowed_units
                 if allowed_units > 0 else 0.0)
        if worst is None or ratio > worst[0]:
            worst = (ratio, info["part"], float(info["n"]), own_units,
                     achieved_m / unit_scale_m)

    if worst is None:
        # No tet carried a region id the plan knows. TetGen's attributes have
        # agreed with the plan on every upload measured, so this is a defensive
        # arm rather than a live path -- and it must not read as "passed". With
        # no per-region measurement available, judge the WHOLE mesh against the
        # TIGHTEST region's own allowance: the strictest reading of the same
        # rule, and the one that fails closed.
        region_id, info = min(region_plan["shared"].items(),
                              key=lambda kv: float(kv[1]["own_edge_units"]))
        own_units = float(info["own_edge_units"])
        allowed_units = MAX_EDGE_OVERSHOOT * own_units
        ratio = (achieved / unit_scale_m / allowed_units
                 if allowed_units > 0 else 0.0)
        worst = (ratio, info["part"], float(info["n"]), own_units,
                 achieved / unit_scale_m)

    return worst[0] > 1.0, worst


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
    materials: dict | None = None,
    parts=None,
):
    """Tetrahedralize and return ``(points_m, tets, region_of_tet)`` in metres.

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

    The resolution guard is PER REGION for a multi-part upload: each region's
    measured median edge is compared against ``MAX_EDGE_OVERSHOOT`` times THAT
    region's own required edge (``own_edge_units``, from
    :func:`mse_common.region_edges`), and a mesh is refused when any region is
    past its own allowance. A regionLESS (one-part) upload keeps the single
    global comparison it always had -- its whole-mesh median against
    ``MAX_EDGE_OVERSHOOT * edge_m`` -- which is also exactly the wall's own test
    in the multi-part case, so an upload whose densest part is the wall is
    unchanged. See ``_resolution_verdict`` for why the wall's allowance cannot
    stand in for a dense core's.

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

    # The merged PLC must carry every part's faces -- checked HERE, on the
    # surface as the caller handed it, and NOT after the decimation below.
    #
    # This check used to sit after ``decimate_to_edge_length``, where it was
    # UNSOUND: decimation's whole job is to return a surface with fewer faces
    # than it was given (on a 24x48 sphere asked for a coarse edge it collapsed
    # 4416 merged faces to 1068), so the check could only pass when decimation
    # had been a no-op. It therefore fired on perfectly valid multi-part
    # uploads whose facets were fine enough to coarsen -- the exact
    # false-positive shape this rig calls could-not-ask's twin, a guard that
    # refuses a correct upload. Moving it before the decimation tests the thing
    # it was always about (did the caller concatenate the parts?) and leaves
    # the face count free to fall afterwards, which is expected.
    if parts is not None and getattr(parts, "is_multi", False):
        n_expected = sum(len(p.faces) for p in parts.parts)
        if len(surface.faces) != n_expected:
            raise MeshingError(
                f"internal error: the merged surface has {len(surface.faces)} "
                f"face(s) for {n_expected} across {len(parts)} part(s)"
            )

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
    # ``parts``/``materials`` go in because ``effective_edge_m`` must know this
    # is a multi-part upload, but the value it returns is the WALL's edge -- it
    # does NOT shrink to a denser region (see its docstring: an earlier version
    # took ``min(base / n_r)`` here and over-predicted the count by more than
    # 10x). For a multi-part upload the one-part edge is only a starting point:
    # ``region_plan`` below pulls ``edge_units`` down to ``h_min``, the finest
    # region's own requirement, before ``maxvolume`` is taken from it. So
    # ``edge_m`` here is the GLOBAL one-part edge the mesher and the app's
    # estimate share, and the rule that turns it into the region edge lives in
    # ``mse_common`` where both callers take it from.
    edge_m = effective_edge_m(
        surface, n_modes=n_modes, units=units,
        min_feature_m=min_feature_m, n_index=n_index,
        physics_m=physics_m,
        parts=parts, materials=materials,
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

    # --- the FINEST REGION EDGE, resolved before maxvolume ------------------
    #
    # For a multi-part upload EVERY region is meshed at ``h_min`` -- the finest
    # region edge -- and ``maxvolume`` is its volume. That is the whole of the
    # uniform rule, and it is what makes one global number do the job a
    # per-region volume list cannot: see ``mse_common.region_edges`` for the
    # measurement (a nine-order sweep of a region's ``max_vol`` moved that
    # region's element count by exactly zero on a nested PLC).
    #
    # ``edge_units`` is the wall's edge for the one-part path and for the wall
    # region's own target; a denser region pulls it DOWN here, so the cavity is
    # resolved for the densest part and the wall is over-resolved -- the
    # deliberate price of correctness. For n_max <= n_wall this is
    # ``edge_units`` exactly, so an upload whose densest part IS the wall meshes
    # at the wall's edge, and the one-part path (which never reaches here) is
    # untouched.
    region_plan = None
    multi = parts is not None and getattr(parts, "is_multi", False)
    if multi:
        region_plan = _region_plan(parts, materials, edge_units, units, verbose)
        edge_units = region_plan["h_min_units"]

    maxvolume = target_maxvolume(surface, edge_units)

    # The merged-PLC face count was already checked against the parts in the
    # ``multi`` branch above; this note stays next to ``maxvolume`` because that
    # is the quantity it is about: the global ``maxvolume`` is now the SAME
    # quantity for both paths -- one edge, one volume -- and the region branch
    # differs only by ``regionattrib`` (which tags each tet with its part) and by
    # ``docheck`` (TetGen's own overlap instrument). The one-part branch keeps
    # its exact call, so its meshes stay byte-identical; for a multi-part upload
    # whose densest region is the wall, ``edge_units`` here is the value the
    # one-part path would have used.

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
    # ``attrib`` is the per-tet region attribute, and it only exists for a
    # multi-part upload -- but it is initialised here rather than only inside the
    # ``region_plan is not None`` branch below, because the verdict helper takes
    # it as an argument on EVERY pass and ``None`` is what tells that helper the
    # mesh has one region.
    attrib = None
    achieved = 0.0
    previous_tets = None
    for attempt in range(MAX_MESH_ATTEMPTS):
        tet = tetgen.TetGen(vertices, faces)

        # The one-part path is UNCHANGED, and that is a requirement rather than
        # a convenience: regionattrib=True passes -A to TetGen, which changes
        # its switches even for a single region and so changes the mesh. The
        # acceptance criterion for this feature is that a one-part upload meshes
        # byte-identically to before it existed, and the only way to honour that
        # is for the single-part case to run exactly the call it always ran.
        # Regions are a BRANCH, not a replacement.
        if region_plan is None:
            # minratio/mindihedral keep element quality up; fixedvolume makes
            # maxvolume binding. All three are needed for a usable interior.
            tet.tetrahedralize(
                order=1, mindihedral=20, minratio=2.0,
                maxvolume=maxvolume, fixedvolume=True,
            )
        else:
            # docheck=True is the OVERLAP instrument. TetGen checks its own PLC
            # for self-intersections and refuses a surface that has them, which
            # is exactly an overlapping pair of parts -- and it does not
            # false-positive on nested parts (both directions measured). It is
            # the mesher's own view of the geometry rather than a second
            # geometric opinion computed here, which is why it is preferred.
            #
            # It does NOT catch touching parts (measured: two boxes sharing a
            # face plane give 12 tets, no error), so touching is refused earlier
            # by mse_parts.load_parts. The two cases need different instruments
            # and neither substitutes for the other.
            #
            # The region SEEDS are placed from the parts, recomputed here so
            # the canonical frame is applied to the same points the vertices
            # were. No per-region VOLUME is passed: see below.
            for region_id, seed_canonical in _region_requests(
                region_plan, canonical_offset, canonical_scale
            ):
                tet.add_region(region_id, tuple(seed_canonical))
            # THE SWITCHES ARE THE ONE-PART SWITCHES, plus ``-A``. No
            # ``varvolume``, no per-region ``max_vol``.
            #
            # This is a MEASURED decision, not a stylistic one. Per-region
            # volumes supplied through ``add_region``'s ``max_vol`` were
            # measured INERT on a nested PLC: sweeping a region's volume over
            # NINE orders of magnitude left that region's own element count and
            # the total byte-identical (hand-built two-shell cube, no
            # application code; identical under ``varvolume=True``,
            # ``fixedvolume=True``, and with neither flag set). An earlier
            # version of this comment claimed ``varvolume`` was the mechanism
            # and ``fixedvolume`` was silently overriding it -- that claim did
            # NOT reproduce, and the instrument errors that produced it (a
            # ``steinerleft`` cap that bounded every run identically, and
            # reading only the global tet total, which the wall region
            # dominates) are recorded on the row.
            #
            # So the resolution is set the way it provably works: ONE global
            # ``maxvolume``, at ``h_min`` -- the finest region edge -- computed
            # above. Every region is meshed at it. TetGen still needs the
            # region SEEDS to know which part each tet is in, which is what
            # ``add_region`` is now used for here and all it is used for.
            #
            # ``docheck=True`` is the OVERLAP instrument. TetGen checks its own
            # PLC for self-intersections and refuses a surface that has them,
            # which is exactly an overlapping pair of parts -- and it does not
            # false-positive on nested parts (both directions measured). It is
            # the mesher's own view of the geometry rather than a second
            # geometric opinion computed here, which is why it is preferred.
            #
            # It does NOT catch touching parts (measured: two boxes sharing a
            # face plane give 12 tets, no error), so touching is refused earlier
            # by mse_parts.load_parts. The two cases need different instruments
            # and neither substitutes for the other.
            tet.tetrahedralize(
                order=1, mindihedral=20, minratio=2.0,
                maxvolume=maxvolume, fixedvolume=True,
                regionattrib=True, docheck=True,
            )
        # Read the mesh from tetgen's OWN arrays, not from ``tet.grid``.
        #
        # ``tet.grid`` builds a pyvista.UnstructuredGrid, and pyvista pulls in
        # VTK -- the single heaviest dependency in this image (~1 GB of the
        # 1.74 GB, plus 212 MB of apt shared libraries that exist only for it).
        # The mesh is available without any of that.
        #
        # The two facts this swap depends on are MEASURED, not assumed, because
        # either being wrong is silent -- an off-by-one index produces a mesh
        # that is nearly right and tetrahedralizes without error:
        #
        #   * ``.elem`` is 0-BASED and byte-identical to ``cells_dict[VTK_TETRA]``
        #     (verified on both a single tet and a 32-tet sphere; the index range
        #     is 0..n_nodes-1, so a 1-based read would run past the end of
        #     ``.node``). This is the opposite of what tetgen's FILE format docs
        #     suggest -- the 1-based rule applies to the .ele file, not this
        #     array -- which is exactly why it is pinned by a test rather than
        #     taken from the docs.
        #   * ``.node`` is exactly 3 columns (x, y, z), with no trailing region
        #     attribute being read as a coordinate.
        #
        # tests/test_meshing_guard.py holds both, so a future tetgen that
        # changes either fails there rather than in a deployed solve.
        tets = np.asarray(tet.elem, dtype=np.int64)
        # The region attributes are re-read on EVERY pass for the same reason
        # ``tets`` and ``node`` are: each retry re-tetrahedralizes and returns a
        # fresh attribute array. Reading it once outside the loop is the defect
        # this feature is *about* -- it would quietly give every tet one
        # material while every count still agreed, and the eigenvalues would be
        # for a uniformly-filled cavity the owner did not upload.
        #
        # ``tet.attributes`` (not ``attrib``) is the tet-indexed array; it is
        # (n_tets, 1) float64. It is confirmed to carry the region ids by
        # tests/test_meshing_guard.py, not taken from the docs -- the tetgen
        # docs' own example shows a differently-shaped return.
        if region_plan is not None:
            attrib = np.asarray(tet.attributes, dtype=np.int64).reshape(-1)
            if attrib.shape[0] != tets.shape[0]:
                raise MeshingError(
                    f"internal error: TetGen returned {attrib.shape[0]} region "
                    f"attributes for {tets.shape[0]} tetrahedra"
                )
        # Map the points straight back to the scaled surface's units. This is
        # the exact inverse of the mapping above, and it happens per attempt
        # because each retry re-tetrahedralizes and returns fresh points.
        points = (np.asarray(tet.node, dtype=float)[:, :3] / canonical_scale
                  + canonical_offset)
        n_tets = int(tets.shape[0])

        if n_tets == 0:
            raise MeshingError("tetrahedralization produced no elements")

        achieved = achieved_edge_length(points * unit_scale_m, tets)
        over_cap = n_tets > max_tets
        # The coarse/fine test is PER REGION when the upload has regions: each
        # region's measured median edge against ``MAX_EDGE_OVERSHOOT`` times that
        # region's OWN required edge. See ``_resolution_verdict``. ``achieved``
        # stays the whole-mesh median -- it is what the log line and the app's
        # resolution caption report, and it is NOT what this test uses.
        too_coarse, worst = _resolution_verdict(
            region_plan, points, tets, attrib, achieved, allowed_edge,
            unit_scale_m,
        )
        (worst_ratio, worst_region, worst_index, worst_target_units,
         worst_achieved_units) = worst

        if not over_cap and not too_coarse:
            break

        if too_coarse and not enforce_resolution:
            # The bench's --allow-coarse path: chart the accuracy trade-off
            # against the cap deliberately, and say so, because a mesh this
            # coarse gives frequencies we would otherwise refuse to report.
            if verbose:
                where = (f" the '{worst_region}' region" if worst_region else "")
                print(
                    f"[mse] WARNING: mesh is {worst_ratio:.1f}x "
                    f"coarser than the resolution rule allows{where} "
                    f"(achieved median edge {worst_achieved_units:.4g} "
                    f"{units}, limit "
                    f"{worst_target_units * MAX_EDGE_OVERSHOOT:.4g} "
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
                _coarse_message(n_modes, worst_achieved_units,
                                worst_target_units, units,
                                region=worst_region, index=worst_index)
            )

        if too_coarse:
            # Within the cap but too coarse: refining is the fix and there is
            # room for it. MAX_EDGE_OVERSHOOT is a FAILURE, not a warning --
            # wrong frequencies shown with confidence are worse than no answer.
            if attempt == MAX_MESH_ATTEMPTS - 1:
                raise MeshingError(
                    _coarse_message(n_modes, worst_achieved_units,
                                    worst_target_units, units,
                                    region=worst_region, index=worst_index)
                )
            # tets ~ 1/maxvolume, so scale by the third power of the edge ratio.
            # The WORST region's ratio, not the whole-mesh one: refining to the
            # mesh median can leave the region that is actually failing short.
            ratio = worst_ratio
            maxvolume /= max(ratio, 1.0) ** 3
            if verbose:
                where = (f" ('{worst_region}' region)" if worst_region else "")
                print(
                    f"[mse] mesh is {ratio:.1f}x too coarse{where}; refining "
                    f"with maxvolume {maxvolume:.4g}",
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
    # A third return value even when there is one region, so no caller branches
    # on how many regions there turned out to be: an all-zeros array is the
    # same shape as a two-region one, and a caller written for the one-region
    # case keeps working unchanged. The one-part path still runs the SAME
    # TetGen call it always did -- region_of_tet is all zeros because the
    # mesher was never told about regions, not because a region was added and
    # then ignored.
    #
    # When regions ARE in play this is TetGen's own ``attrib``, from the pass
    # that actually succeeded -- not a plan computed before meshing. The tets
    # are indexed the same way the attributes are, and neither is touched by
    # the canonical frame (that maps POINTS only), so no correction is needed:
    # verified rather than assumed, because an off-by-one or a frame-mapped
    # index would tag every tet with a neighbouring region's material.
    if region_plan is None:
        region_of_tet = np.zeros(n_tets, dtype=np.int64)
    else:
        region_of_tet = attrib.astype(np.int64, copy=True)
    return points_m, tets, region_of_tet


def _region_plan(parts, materials, edge_units: float, units: str,
                 verbose: bool) -> dict:
    """The resolution plan for a multi-part upload, under the uniform rule.

    Returns a dict the mesher reads:

        region_of_part   region id per part, from the PartSet
        seeds_units      per-region seed point, in the SCALED surface's units
        h_min_units      the ONE edge the whole cavity is meshed at

    ``materials`` maps part name -> n and must name every part. A region with no
    material is refused rather than defaulted, because the default would be
    vacuum and the owner would get frequencies for a cavity that is partly empty
    with nothing saying so.

    ``h_min_units`` is ``min_r(edge_units * n_wall / n_r)``, floored at
    ``edge_units`` -- the finest region edge. See
    :func:`mse_common.region_edges` for why the mesher uses ONE edge rather than
    a per-region volume list: the per-region mechanism is not realisable in this
    TetGen build (measured inert over nine orders on a nested PLC).

    There is deliberately NO per-region volume in the plan. Producing one and
    passing it to ``add_region`` is the shape that LOOKED like it worked and
    did not -- a plan whose numbers are correct and whose effect is nil. If a
    future TetGen makes per-region grading possible, this is the function to
    change, and the row records the measurement that would have to be redone.
    """
    # The per-region fill and edge come from ``mse_common.region_plan`` -- the
    # SAME call the web app's estimate makes. Deriving them here instead would
    # be two expressions of one quantity in two processes, which is the shape
    # the row names: the app routes on its estimate and the mesher builds the
    # mesh, and if they disagree the route is decided for a job nobody runs.
    # ``units`` goes in: without it ``region_plan`` converted every part's
    # volume as if it were written in mm, so the mesher's own verbose figures
    # were wrong by the cube of the unit ratio for any non-mm upload. The edge
    # beside them was always true, so the two disagreed silently.
    shared = region_plan_by_region(parts, materials, edge_units, units)
    region_of_part = parts.region_of_part

    # Every region is meshed at the same edge. Resolved as a MIN over the plan
    # rather than recomputed from the materials, so the value here is the value
    # the estimate used even if the two ever disagree about how to get it.
    h_min_units = min(float(info["edge_units"]) for info in shared.values())

    seeds_units = {}
    for part_index, region_id in enumerate(region_of_part):
        seeds_units[region_id] = np.asarray(parts.seeds[part_index], dtype=float)

    if verbose:
        print(
            f"[mse] uniform region edge h_min = {h_min_units:.4g} {units} "
            f"for all {len(region_of_part)} region(s)",
            flush=True,
        )
        for part_index, region_id in enumerate(region_of_part):
            info = shared[int(region_id)]
            print(
                f"[mse] region {region_id}: part {info['part']!r}, "
                f"n={materials[info['part']]:g}, fills "
                f"{info['filled_volume_m3']:.4g} m3",
                flush=True,
            )
    return {
        "region_of_part": region_of_part,
        "seeds_units": seeds_units,
        "h_min_units": h_min_units,
        "shared": shared,
    }


def _region_requests(region_plan, canonical_offset, canonical_scale):
    """``(region_id, seed)`` pairs in the canonical frame.

    ONLY the seeds. There is no per-region volume, because the mesher computes
    none: every region is meshed at the one global ``h_min`` (see
    ``_region_plan`` and ``mse_common.region_edges``). TetGen still needs the
    seeds -- they are how it decides which part each tet lies in, which is what
    the region ATTRIBUTE is for -- but they carry no size.

    The seeds are mapped with the same affine the vertices get, and ``add_region``
    must receive the point in the frame TetGen is actually meshing, or a seed
    written for the scaled surface can land in the wrong region once the
    canonical frame moves the geometry.

    WHAT THIS FUNCTION USED TO DO, kept because the shape is a trap: it computed
    a per-region ``max_vol`` as a RATIO of the plan's own base, so every ratio
    was ``<= 1`` and each bound was at most the global's -- the global capped
    the whole mesh at the coarsest region's resolution. Rewriting it to
    absolute per-region volumes did NOT fix that, because the region volume
    itself turned out to be inert on nested input (nine-order measurement on the
    row). The uniform rule removes the quantity entirely rather than computing
    it correctly for a mechanism that cannot use it.
    """
    out = []
    for region_id, seed_units in sorted(region_plan["seeds_units"].items()):
        seed_canonical = (
            np.asarray(seed_units, dtype=float) - canonical_offset
        ) * canonical_scale
        out.append((int(region_id), seed_canonical))
    return out


def achieved_edge_length(points_m: np.ndarray, tets: np.ndarray) -> float:
    """Median edge length of the tetrahedralization, in metres."""
    xyz = points_m[tets]
    local = np.array([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    lengths = np.linalg.norm(
        xyz[:, local[:, 0], :] - xyz[:, local[:, 1], :], axis=2
    )
    return float(np.median(lengths))
