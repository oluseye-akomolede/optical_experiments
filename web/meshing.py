"""Upload validation: the same checks the solver will apply, done early.

Rejecting a bad .obj here costs the owner a second; letting it through costs ten
minutes of a solver Job that is certain to fail. The rules come from the shared
module ``dockerfiles/maxwell_solver/mse_common.py`` deliberately -- the solver is
the authority, this is the courteous early answer, and sharing the definition is
what keeps the two from disagreeing.
"""

from __future__ import annotations

import io

#: extensions we accept
ALLOWED_SUFFIXES = (".obj",)


class UploadError(Exception):
    """A problem with the upload that we can explain to the owner in one line."""


def validate_upload(filename: str, data: bytes, max_bytes: int, *,
                    closed: bool = True) -> None:
    """Raise :class:`UploadError` unless this looks like a usable .obj.

    Checks, in the order the owner would care about them: the name, the size,
    that trimesh can parse it at all, that it has faces, and that it is a closed
    surface -- the solver enforces PEC on the boundary, so an open surface would
    silently leak.

    ``closed=False`` stops after the faces: the size and readability rails
    only. A td upload passes it, because the closed check below merges every
    part into one mesh and, when that is not watertight, falls back to the
    CAVITY parser, whose touch rule refuses a film flush between two plates
    (shared faces leave the merged mesh open). td_form's own split judges a
    td upload's geometry, naming any open part.
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
    if not closed:
        return

    mesh.merge_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())

    if not mesh.is_watertight:
        # A one-part file has no part NAME to give, so the message below
        # describes the file. A multi-part one is different: the owner has to
        # know WHICH part is open, and the check above cannot say -- it merges
        # every geometry into one mesh (trimesh hands back the biggest) before
        # looking. So ``parse_parts`` is tried first: it splits, then checks
        # each part by name, and its message is used verbatim when it has one.
        #
        # This is not tidiness. An earlier version refused here for a two-part
        # file with an open core, naming no part -- so the owner could not tell
        # which of their two groups to fix. That is the could-not-ask shape: a
        # refusal the owner cannot act on.
        #
        # Note what this does NOT do: it does not ask "how many parts are
        # there?" and then act on the answer. Splitting is the only way to know,
        # so the call IS the check, and asking the same question twice bought
        # nothing but a place to lose the answer.
        try:
            parse_parts(data, "mm")
        except UploadError as exc:
            raise exc
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
# The model has two parts, and they are now separated on purpose:
#
#   est = MARGIN * sum_r filled_V_r / h_r^3   (in tet-per-volume terms:)
#       = MARGIN * 6*sqrt(2) * sum_r filled_V_r / h_r^3
#
# The first, ``6*sqrt(2)``, is GEOMETRY and is exact: it is the number of
# regular tets of edge h that fit in a unit volume (a regular tet of edge h has
# volume h^3/(6 sqrt(2))). It is ``mse_common.TETS_PER_UNIT_VOLUME`` -- the same
# constant ``target_maxvolume`` inverts to ask TetGen for an edge. Nothing is
# fitted here and nothing can drift: the estimate and the mesh share one
# expression for the shape of the lattice.
#
# The second, MARGIN, is MEASURED, and it exists because TetGen does not put the
# elements on a regular lattice: with ``TETGEN_VOLUME_SLACK`` = 4 it may legally
# land well BELOW the law, and a curved or fine surface can force it above. It
# is one-sided by design -- an under-estimate sends a big job to the CPU, where
# it dies at the memory limit after minutes of work, while an over-estimate
# costs a small job about a second on the GPU -- so the margin is meant to cover
# the worst case measured across the shape families, not the median one. The
# sweep found cases above it anyway (see below and ``config.EST_TETS_MARGIN``),
# which is a fact about the mesher and not about this formula.
#
# The previous form of this comment described a fitted coefficient C = 14 over
# ``V/h^3``. 14/(6 sqrt(2)) = 1.650 was that fit's largest single component: the
# lattice constant, which is now written down exactly instead of absorbed into a
# number nobody could read. The rest of that fit was the tilt for where TetGen
# lands, which is what MARGIN is. Splitting them is what makes the estimate
# auditable: the geometry part can only change if the mesh definition changes,
# and the margin's size is a measurement that lives in the sweep's table.
#
# The measured band, and the one thing that is out of it, are in
# ``openspec/changes/fix-upload-tet-estimate/``; ``benchmarks/estimate_vs_built.py``
# reproduces the table by building REAL meshes through this product's own calls.

#: margin over the volume law; overridden by MSE_EST_TETS_MARGIN.
#:
#: 1.35 is the step-1 measurement of the worst under-estimate the family sweep
#: produced (a 2-region object with a 2x finer core at 1.289), rounded up to the
#: next 0.05 -- ~5% of headroom. It is NOT a bound: measured at the product's own
#: cap, a 3.5x finer core measures 1.414 and leaves this reading 0.95x of the
#: real mesh, the same families at half their mode counts measure 1.868 (0.72x),
#: and a three-region object of nested thin sheets measures 8.952 (0.15x). See
#: ``config.EST_TETS_MARGIN`` for those and for why none of them is answered by
#: raising the constant here. Held equal to ``config.EST_TETS_MARGIN`` by a test,
#: because this one is the fallback for callers that pass no margin and that one
#: is what the upload route passes.
EST_TETS_MARGIN = 1.35


def _common_module():
    """``mse_common``: the sizing helpers both images share.

    THIS USED TO IMPORT ``mse_meshing``. That module lives in
    ``dockerfiles/maxwell_solver/``, which the web Dockerfile does not copy, so
    the web image built, pushed and crashlooped at container start:

        ModuleNotFoundError: No module named 'mse_maxwell_solver'

    (and the same for every ``mse_meshing`` import here). The tests never caught
    it because they put the solver directory on ``sys.path``; no test started
    the IMAGE. ``mse_common`` is copied into the web image and imports only
    numpy, so every function this file needs comes from there now -- the same
    single definition the solver uses, reached through a module the app
    actually has.

    Imported late so tier-1 tests stay cheap: numpy is already loaded by then,
    and the heavy solver stack is never touched at all.
    """
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import mse_common  # noqa: PLC0415 - deliberately late

    return mse_common


def target_edge_length_mm(surface, n_modes: int, units: str) -> float:
    """The solver's target edge length, in the user's units.

    Imported from the shared module rather than reimplemented: the estimate is
    only useful if it predicts what the solver will actually do, and a copy of
    the formula here would drift from the original at the first calibration.
    """
    COMMON = _common_module()
    return float(COMMON.target_edge_length(surface, n_modes, units=units))


#: How far above the requested edge a mesh may land before the solver refuses it.
#: Re-exported so the job page can state the allowance in the same number the
#: mesher enforces, rather than a copy that could drift from it.
MAX_EDGE_OVERSHOOT = _common_module().MAX_EDGE_OVERSHOOT


def unit_scale_m(units: str) -> float:
    """Metres per unit, from the shared table.

    Re-exported so the upload form converts a feature to metres with the SAME
    table the mesher uses. A second copy of ``{"mm": 1e-3, ...}`` here would be a
    place for the two to disagree, and a disagreement is a length-space bug.
    """
    return float(_common_module().unit_scale(units))


def effective_edge_mm(surface, n_modes: int, units: str,
                      min_feature_m: float | None = None) -> float:
    """The edge length the solver will mesh to, in the user's units.

    A thin conversion over ``mse_common.effective_edge_m``, which owns the
    physics-edge/feature comparison. Kept for callers that want user units (the
    page shows the owner the unit they chose), so the comparison itself lives in
    exactly ONE place and cannot drift between the estimate and the mesh.

    ``surface`` is expected to be SCALED already -- the metre result is measured
    on the scaled object, so converting an unscaled one back to user units would
    describe a different object.
    """
    COMMON = _common_module()
    return float(COMMON.effective_edge_m(
        surface, n_modes=n_modes, units=units, min_feature_m=min_feature_m,
    )) / COMMON.unit_scale(units)


def estimate_tets(surface, n_modes: int, units: str,
                  margin: float | None = None,
                  min_feature_m: float | None = None,
                  scale: float = 1.0,
                  parts=None, materials: dict | None = None) -> int:
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

    ``parts``/``materials`` are a MULTI-PART upload (a validated
    ``PartSet`` and its name -> n table), and are ``None`` for the one-part
    upload that this function has always handled. With them the answer is the
    volume law summed over the plan's regions, each at the edge that region is
    meshed to (see ``mse_common.estimate_tets_by_region``). On today's mesher
    those edges are all the same ``h_min`` -- one cavity, one edge -- so this is
    the same number the uniform calculation gives; it is written against the
    plan so it follows the plan if that ever stops being true. The count is what
    ROUTES the job, so the estimate and the mesh must be the same expression:
    using the wall's edge alone would under-predict a job with a dense core and
    send it to the GPU to die, which is what this branch exists to prevent.

    NOTE ON THE CALLER: ``web/app.py``'s upload route parses the
    ``part_index.<name>`` fields and passes ``parts``/``materials`` in, so this
    branch is on the real upload path -- a two-part mm upload with a dense core
    is routed by it (``test_the_dense_core_upload_routes_on_the_real_mesh_not_the_estimate``
    drives the route and compares the app's decision against the mesh TetGen
    really builds).

    The edge comes from ``mse_common.effective_edge_m``, in METRES, on the
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
        COMMON = _common_module()

        surface = COMMON.apply_scale(surface, scale)
        margin_value = EST_TETS_MARGIN if margin is None else float(margin)
        if materials and parts is not None and getattr(parts, "is_multi", False):
            # MULTI-PART: the volume law per region, at the edge each region is
            # meshed to. The edges come from the shared region plan, so the
            # estimate cannot describe a different mesh than the mesher builds,
            # and ``filled_volume_m3`` per region is what is summed -- the wall's
            # ENCLOSED volume double-counts everything nested in it, and the
            # count is what ROUTES the job.
            edge_units = COMMON.effective_edge_m(
                surface, n_modes=n_modes, units=units,
                min_feature_m=min_feature_m, parts=parts, materials=materials,
            )
            return COMMON.estimate_tets_by_region(
                parts, materials, edge_units, surface, units, margin_value
            )

        volume_m3 = COMMON.enclosure_volume_m3(surface, units)
        edge_m = COMMON.effective_edge_m(
            surface, n_modes=n_modes, units=units, min_feature_m=min_feature_m,
        )
    except Exception:  # noqa: BLE001 - an estimate must not break an upload
        return 0

    if not np.isfinite(volume_m3) or volume_m3 <= 0:
        return 0
    if not np.isfinite(edge_m) or edge_m <= 0:
        return 0
    law = COMMON.TETS_PER_UNIT_VOLUME * volume_m3 / (edge_m ** 3)
    return int(margin_value * law)


# --------------------------------------------------------------------------
# Part parsing for the upload form -- ONE parser, used by both POSTs
# --------------------------------------------------------------------------
#
# The form asks the owner to assign a refractive index per part name. The names
# have to come from somewhere, and they must be the SAME names the solver will
# derive from the same bytes -- otherwise the owner assigns "core" an index and
# it lands on a region called something else, with every count still agreeing.
#
# So this is the solver's own parser, exposed. A client-side scan of the .obj
# in JavaScript would be a second source of truth for exactly the thing that
# decides the physics, and the two would drift the first time either changed.

def parse_parts(data: bytes, units: str) -> list[str]:
    """The part NAMES ``data`` will solve as, in file order.

    Raises ``UploadError`` with the solver's own message when the file cannot be
    split -- an open part, two parts that touch, duplicate group names, or a
    name/part count mismatch. This is the same call the solver makes, so every
    refusal the owner sees here is a refusal they would otherwise have met as a
    failed job minutes later.

    ``units`` is not used by the split (names do not depend on it) and is taken
    for signature symmetry with the other parse helpers.
    """
    import mse_parts as PARTS

    return list(load_parts(data, units).names)


def load_parts(data: bytes, units: str):
    """The validated ``PartSet`` for ``data`` -- the solver's own parser.

    ``parse_parts`` takes the names from this and the upload route takes the
    PartSet, so BOTH POSTs and the final solve are looking at one split of one
    file. A second call site that re-split the bytes would be a second source of
    truth for which region gets which material, and the failure it produces --
    an index applied to the wrong region with every count agreeing -- is exactly
    what naming parts by NAME rather than by index was meant to prevent.
    """
    import mse_parts as PARTS

    try:
        return PARTS.load_parts(data, units=units)
    except UploadError:
        raise
    except Exception as exc:  # noqa: BLE001 - every failure is the same here
        raise UploadError(str(exc)) from exc
