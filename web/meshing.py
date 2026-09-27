"""Upload validation: the same checks the solver will apply, done early.

Rejecting a bad .obj here costs the owner a second; letting it through costs ten
minutes of a solver Job that is certain to fail. The rules mirror
``dockerfiles/maxwell_solver/mse_meshing.py`` deliberately -- the solver is the
authority, this is the courteous early answer.
"""

from __future__ import annotations

import io

#: extensions we accept
ALLOWED_SUFFIXES = (".obj",)


class UploadError(Exception):
    """A problem with the upload that we can explain to the owner in one line."""


def validate_upload(filename: str, data: bytes, max_bytes: int) -> None:
    """Raise :class:`UploadError` unless this looks like a usable .obj.

    Checks, in the order the owner would care about them: the name, the size,
    that trimesh can parse it at all, that it has faces, and that it is a closed
    surface -- the solver enforces PEC on the boundary, so an open surface would
    silently leak.
    """
    name = (filename or "").strip()
    if not name:
        raise UploadError("Please choose a file to upload.")

    lower = name.lower()
    if not lower.endswith(ALLOWED_SUFFIXES):
        raise UploadError("Please upload an .obj file (other formats are not supported).")

    if not data:
        raise UploadError("That file is empty.")

    if len(data) > max_bytes:
        raise UploadError(
            f"That file is larger than the {max_bytes // (1024 * 1024)} MB limit. "
            "Please upload a smaller model."
        )

    import trimesh

    try:
        mesh = trimesh.load(io.BytesIO(data), file_type="obj")
    except Exception as exc:  # noqa: BLE001 - surface the parser's own words
        raise UploadError(f"That file could not be read as an .obj model: {exc}") from exc

    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.faces)]
        if not geometries:
            raise UploadError("That .obj file contains no triangles.")
        mesh = max(geometries, key=lambda g: len(g.faces))

    if mesh is None or len(mesh.faces) == 0:
        raise UploadError("That .obj file contains no triangles.")

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    if not mesh.is_watertight:
        raise UploadError(
            "This object is not a closed surface (it has holes or open edges). "
            "The solver treats your object as a cavity, so it must be sealed. "
            "Please close the holes and upload it again."
        )

    return mesh


def load_surface(data: bytes):
    """The validated surface for ``data``, or ``None`` if it cannot be read.

    ``validate_upload`` already emptied this bottle; re-reading is cheap
    (trimesh on a 50 MB .obj is tens of milliseconds) and keeps the estimate and
    the validation looking at the same object rather than threading a mesh
    through two call sites.
    """
    import trimesh

    try:
        mesh = trimesh.load(io.BytesIO(data), file_type="obj")
    except Exception:  # noqa: BLE001 - validate_upload already reported this
        return None
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if len(g.faces)]
        if not geometries:
            return None
        mesh = max(geometries, key=lambda g: len(g.faces))
    if mesh is None or len(mesh.faces) == 0:
        return None
    return mesh


# --------------------------------------------------------------------------
# The upload-time tet estimate -- how the route is chosen before anything runs
# --------------------------------------------------------------------------
#
# Why an estimate at all: the routing decision (cluster CPU vs Jetson GPU) has
# to be made at upload, before any work is queued, and the only way to know the
# real element count is to run TetGen, which is the expensive thing being
# scheduled. So the count is predicted from the geometry alone.
#
# The model: ``est_tets = C * V / h^3``, where V is the ENCLOSED volume (the
# solver's own sizing input) and h the target edge length the solver would use
# at this mode count. C was fitted on 13 measured meshes spanning smooth boxes,
# a sphere and a bumpy sphere at 7-25 edges per wavelength.
#
# C = 14 is the CONSERVATIVE fit: it keeps the estimate at or above the measured
# count on nearly every point. The asymmetry is deliberate -- an under-estimate
# sends a big job to the CPU, where it dies at the memory limit after minutes of
# work, while an over-estimate costs a small job about a second on the GPU.
#
# A term for the surface was tried and dropped. The measured meshes include a
# non-monotonic pair (the bumpy sphere needs FEWER elements at 25 edges per
# wavelength than at 22) because the surface floors the interior: a fine or
# curved surface forces refinement the volume law cannot see, and C absorbs that
# in the conservative direction instead. See the rig memory on this.

#: coefficient of the fit; overridden by MSE_EST_TETS_C
EST_TETS_C = 14.0


def target_edge_length_mm(surface, n_modes: int, units: str) -> float:
    """The solver's target edge length, in the user's units.

    Imported from the solver rather than reimplemented: the estimate is only
    useful if it predicts what the solver will actually do, and a copy of the
    formula here would drift from the original at the first calibration.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mse_meshing as MESH  # noqa: PLC0415 - deliberately late

    return float(MESH.target_edge_length(surface, n_modes, units=units))


def _solver_module():
    """The solver's meshing module, imported late so tier-1 tests stay cheap."""
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mse_meshing as MESH  # noqa: PLC0415 - deliberately late

    return MESH


#: How far above the requested edge a mesh may land before the solver refuses it.
#: Re-exported so the job page can state the allowance in the same number the
#: mesher enforces, rather than a copy that could drift from it.
MAX_EDGE_OVERSHOOT = _solver_module().MAX_EDGE_OVERSHOOT


def unit_scale_m(units: str) -> float:
    """Metres per unit, from the solver's own table.

    Re-exported so the upload form converts a feature to metres with the SAME
    table the mesher uses. A second copy of ``{"mm": 1e-3, ...}`` here would be a
    place for the two to disagree, and a disagreement is a length-space bug.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mse_meshing as MESH  # noqa: PLC0415 - deliberately late

    return float(MESH.unit_scale(units))


def effective_edge_mm(surface, n_modes: int, units: str,
                      min_feature_m: float | None = None) -> float:
    """The edge length the solver will mesh to, in the user's units.

    A thin conversion over ``mse_meshing.effective_edge_m``, which owns the
    physics-edge/feature comparison. Kept for callers that want user units (the
    page shows the owner the unit they chose), so the comparison itself lives in
    exactly ONE place and cannot drift between the estimate and the mesh.

    ``surface`` is expected to be SCALED already -- the metre result is measured
    on the scaled object, so converting an unscaled one back to user units would
    describe a different object.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mse_meshing as MESH  # noqa: PLC0415 - deliberately late

    return float(MESH.effective_edge_m(
        surface, n_modes=n_modes, units=units, min_feature_m=min_feature_m,
    )) / MESH.unit_scale(units)


def estimate_tets(surface, n_modes: int, units: str, c: float | None = None,
                  min_feature_m: float | None = None,
                  scale: float = 1.0) -> int:
    """Predicted element count for meshing ``surface`` at ``n_modes``.

    Returns 0 when the surface is unusable, which the caller reads as "do not
    route on this" rather than "this is tiny": a failed estimate must never be
    the reason a job is sent to the slower machine. See the class of defect this
    rig calls *could-not-ask must not look like nothing-there*.

    ``min_feature_m`` is METRES (converted once at the upload form) and
    ``scale`` is the owner's free scale factor. They matter here for the same
    reason they matter to the solver: the element count goes like ``V / h^3``,
    so a scaled object and a finer target edge both move the count hard -- a
    feature-driven refinement of 2x is 8x the elements, which is exactly the
    difference between the cluster and the GPU queue.

    The edge comes from ``mse_meshing.effective_edge_m``, in METRES, on the
    SCALED surface -- one length space, one comparison, shared with the solver.
    An earlier version took it from ``effective_edge_mm``, which converted the
    metre edge back to *user units of an already-scaled surface*; multiplying
    that by ``unit_scale`` applied the scale a second time, so the estimate and
    the mesh disagreed by ``scale^3`` -- and the estimate is what routes the job.
    """
    if surface is None:
        return 0
    import numpy as np

    try:
        import sys
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import mse_meshing as MESH  # noqa: PLC0415 - deliberately late

        surface = MESH.apply_scale(surface, scale)
        volume_m3 = MESH.enclosure_volume_m3(surface, units)
        edge_m = MESH.effective_edge_m(
            surface, n_modes=n_modes, units=units, min_feature_m=min_feature_m,
        )
    except Exception:  # noqa: BLE001 - an estimate must not break an upload
        return 0

    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        return 0
    if not np.isfinite(edge_m) or edge_m <= 0:
        return 0
    coefficient = EST_TETS_C if c is None else float(c)
    return int(coefficient * volume_m3 / (edge_m ** 3))
