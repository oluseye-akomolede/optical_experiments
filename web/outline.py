"""The uploaded object as an outline, in the frame the field movie draws in.

WHY NOT THE CAVITY'S ``surface.glb``. That file is the cavity solver's own
export, in the USER'S units, and the viewer places it by recovering a ratio from
the grid's extent (``surfaceScaleFromGrid``) and dropping it at the grid box's
centre. That trick is only right for a mesh centred on its own bounding box, and
the td worker writes no surface at all.

WHAT THE MOVIE'S FRAME ACTUALLY IS, read from the worker rather than assumed:
the worker solves with ``length_scale = unit_scale(units) * scale`` and stores
every frame position as ``origin_m[a] + nodes[a][cell]`` -- absolute metres,
``origin_m`` being the parts' combined bounding-box minimum -- so a sample and
the upload's own coordinates times ``length_scale`` are the same frame, and the
outline's whole transform is one multiply with no offset.

The frame is NOT the object's own box, and a reader must not take it for one:
``fdtd_layout`` samples an open scene over the object PLUS
``OPEN_PAD_CELLS * dx`` on each side (16 cells at the values both boundaries
share today -- BOX_CLEARANCE + 3 + DEFAULT_PML_CELLS) and none on metal, and the
manifest records the samples, not the object. So the positions a frame samples
bound the object and, on open, the padding around it. This outline is the
object, and only the object. No padding belongs in the transform either way: the
pad moves which points the field was sampled at, never where the object is --
but it does mean a check must not use the sampled positions as the object's box
(a review row of 2026-10-03 was exactly that mistake).

WHY A ROUTE AND NOT A FILE THE WORKER WRITES. Every job's raw upload is kept at
``jobs/<id>/input.obj``, so deriving the outline when it is asked for gives jobs
already solved -- the owner's film job included -- their outline with no re-run
and no change to the worker.
"""

from __future__ import annotations

from pathlib import Path


class OutlineError(Exception):
    """The outline cannot be built from this job's upload, with the reason."""


def _solver_modules():
    """``(mse_parts, mse_common)``: the parser and unit table the worker uses.

    Late and local, the same shape as ``web/meshing.py``: both files are copied
    into the web image by ``dockerfiles/web/Dockerfile`` (and
    ``tests/test_web_imports.py`` reads that COPY list, so a name that is not
    there fails at commit rather than at container start). Importing them here
    keeps the tier-1 suite cheap -- nothing heavy is touched until a td outline
    is actually asked for.
    """
    import sys

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mse_common  # noqa: PLC0415 - deliberately late
    import mse_parts  # noqa: PLC0415 - deliberately late

    return mse_parts, mse_common


def outline_glb(data: bytes, units, scale) -> bytes:
    """A glb of the uploaded object in METRES, in the td field's own frame.

    ``data`` is the job's stored ``input.obj``; ``units`` and ``scale`` are the
    job's own. The factor is the exact one the worker solved with -- the shared
    ``unit_scale`` table, the worker's own ``float(scale or 1.0)``, and the
    shared ``apply_scale`` (which also refuses a zero, negative or non-finite
    factor) -- so the outline cannot disagree with the field by a unit or a
    scale. ``or 1.0`` rather than a ``None`` test on purpose: it is what the
    worker line does, so a stored ``""`` or ``0`` has to mean 1.0 here too or
    the two would draw the same job at different sizes. Every part is placed at
    its absolute metre coordinates: the caller draws it unscaled and unmoved.
    """
    mse_parts, mse_common = _solver_modules()

    try:
        length_scale = (mse_common.unit_scale(str(units))
                        * float(scale or 1.0))
    except mse_common.MeshingError as exc:
        raise OutlineError(str(exc)) from exc
    except (TypeError, ValueError) as exc:
        raise OutlineError(f"this job's scale {scale!r} is not a number") from exc

    try:
        names, parts = mse_parts.split_obj_parts(data)
    except mse_parts.MeshingError as exc:
        raise OutlineError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - a bad stored upload is a 4xx too
        raise OutlineError(f"this job's upload could not be read: {exc}") from exc

    try:
        parts = [mse_common.apply_scale(part, length_scale) for part in parts]
    except mse_common.MeshingError as exc:
        raise OutlineError(str(exc)) from exc

    geometry = _scene_geometries(names, parts)
    if not geometry:
        raise OutlineError("this job's upload has no faces to draw")

    import trimesh

    try:
        return trimesh.Scene(geometry).export(file_type="glb")
    except Exception as exc:  # noqa: BLE001 - surface the exporter's own words
        raise OutlineError(f"the outline could not be written as a glb: {exc}") from exc


def _scene_geometries(names, parts) -> dict:
    """The scaled parts keyed by name, for ``trimesh.Scene``.

    Keys are made unique, because two parts under one key would land on each
    other in the scene's dict and only the last would be drawn. Through the route
    that cannot happen -- ``mse_parts.parse_obj_groups`` refuses a repeated group
    name outright, and a single unnamed part arrives as ``UNNAMED_PART`` -- so
    this guards the function's own contract, and it carries its own test rather
    than a mutation a caller cannot reach.
    """
    out = {}
    for index, (name, part) in enumerate(zip(names, parts)):
        if len(part.faces) == 0:
            continue
        key = str(name)
        if key in out:
            key = f"{key} #{index + 1}"
        out[key] = part
    return out
