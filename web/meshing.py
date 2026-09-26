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
