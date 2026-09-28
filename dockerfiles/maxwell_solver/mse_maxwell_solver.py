"""MSE Maxwell cavity-mode solver -- the job entrypoint.

Reads an .obj from object storage, treats it as a closed perfectly-conducting
cavity filled with a uniform medium of refractive index ``n``, computes the
lowest ``n_modes`` full-vector electromagnetic modes with lowest-order Nedelec
edge elements, and writes the surface, the mode table and a sampled E field per
mode back to storage. Progress is recorded in the ``mse`` schema.

Run as a plain ``batch/v1`` Job (there is no Spark; see the task amendments).
Locally, ``--local-in``/``--local-out`` run the same pipeline against a
directory with no MinIO and no database, which is how the tests and the Jetson
bench exercise it.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
import traceback

# Cap BLAS/OpenMP threads before numpy is imported. The container is allotted
# 4 CPUs; letting each BLAS call spawn a thread per core oversubscribes the node
# and, worse, makes peak RSS depend on the host's core count.
_THREADS = os.environ.get("MSE_THREADS", "4")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import numpy as np  # noqa: E402  (must follow the thread caps)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import mse_common  # noqa: E402
import mse_db  # noqa: E402
import mse_eigen as EIG  # noqa: E402
import mse_fem as F  # noqa: E402
import mse_fields as FLD  # noqa: E402
import mse_meshing as MESH  # noqa: E402
import mse_parts as PARTS  # noqa: E402
import mse_storage as STORE  # noqa: E402

# Re-exported from :mod:`mse_common`. These two constants and the reroute token
# are needed by the WEB app as well, and the web image does not ship this file.
# They live in mse_common so that the app can import them without importing the
# solver; the names are re-exported here because the solver's own code, the
# worker, the bench scripts and the tests all reach them as
# ``mse_maxwell_solver.X``. A re-export is a binding, not a copy.
from mse_common import (  # noqa: E402,F401
    DEFAULT_CPU_MAX_TETS,
    DEFAULT_MAX_TETS,
    REROUTE_TOKEN,
)

#: the cap a *solve* runs with. Settable, because tests assert on the guard at a
#: tiny count rather than by meshing 600k tetrahedra.
_MAX_TETS_OVERRIDE: int | None = None


def max_tets() -> int:
    """Element count above which a solve is refused as too large.

    The single accessor for the absolute cap. Every caller that used to read the
    constant or the CLI default now asks here, so a test can set a small cap and
    a deploy can set a real one, and neither can disagree with the other.
    """
    if _MAX_TETS_OVERRIDE is not None:
        return _MAX_TETS_OVERRIDE
    try:
        return max(1, int(os.environ.get("MSE_MAX_TETS", DEFAULT_MAX_TETS)))
    except ValueError:
        return DEFAULT_MAX_TETS


def set_max_tets(limit: int | None) -> None:
    """Force the cap for this process (tests, and the bench's ``--max-tets``).

    ``None`` restores the environment/default behaviour. Module state rather
    than a parameter threaded through every caller because the cap is a
    property of the machine, not of a job -- unlike ``min_feature_m``, which
    is a property of the upload and travels as an argument.

    **A latch, not a job-scoped setting.** The override outlives the call that
    set it, so whoever sets it owns restoring it. That is the right shape for
    the two real callers -- a test asserts at a tiny cap for the whole test, and
    ``main()`` sets the cap for the whole of one process -- but it is a trap for
    anyone calling ``main(argv)`` twice in one process, and it did in fact leak:
    a test running ``main(["--max-tets", "4000"])`` left every later test in the
    session solving at 4,000 elements. Prefer passing ``max_tets`` to
    :func:`solve` when the cap belongs to one call.
    """
    global _MAX_TETS_OVERRIDE
    _MAX_TETS_OVERRIDE = None if limit is None else max(1, int(limit))

class TooBigForProcessor(Exception):
    """This job is fine but this machine is not -- re-route it, do not fail it.

    Distinct from :class:`mse_meshing.MeshingError` on purpose. A job over the
    *cluster's* limit has a home on the GPU, so the app re-queues it there; a
    job over the absolute limit has no home and is failed with a message the
    owner can act on. Collapsing the two would either fail a runnable job or
    re-queue an impossible one forever.

    **How the app hears about it.** A non-zero exit code is invisible to the
    app: every failure in the solver is exit 1, and Kubernetes reports a Job as
    ``BackoffLimitExceeded``/``Error`` either way. The only channel that carries
    a *reason* out of a dead container is its termination message, so this
    exception writes :data:`REROUTE_TOKEN` into
    ``/dev/termination-log`` (see :func:`main`) and the app greps for it. The
    token is a phrase, not the message text, because the text is for a human and
    would otherwise be load-bearing.
    """


def cpu_max_tets() -> int:
    """Element count above which a CPU solve is refused as mis-routed."""
    try:
        return max(1, int(os.environ.get("MSE_CPU_MAX_TETS", DEFAULT_CPU_MAX_TETS)))
    except ValueError:
        return DEFAULT_CPU_MAX_TETS

#: samples along the longest bounding-box axis for mode_XX.bin
DEFAULT_GRID = FLD.DEFAULT_GRID


def peak_rss_mb() -> float:
    """Peak resident set size of this process, in MiB."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mse_maxwell_solver",
        description="Compute the lowest electromagnetic cavity modes of an .obj.",
    )
    parser.add_argument("--job-id", required=True, help="job identifier (uuid)")
    parser.add_argument("--units", default="mm", choices=sorted(MESH.UNIT_SCALE),
                        help="units of the uploaded mesh (default: mm)")
    parser.add_argument("--n-index", type=float, default=1.0,
                        help="refractive index of the filling medium (default: 1.0)")
    parser.add_argument("--n-modes", type=int, default=20,
                        help="how many modes to compute (default: 20)")
    parser.add_argument("--local-in", metavar="DIR",
                        help="read jobs/<id>/... from this directory instead of MinIO")
    parser.add_argument("--local-out", metavar="DIR",
                        help="write jobs/<id>/... to this directory instead of MinIO")
    parser.add_argument("--max-tets", type=int, default=None,
                        help=f"cap on tetrahedra (default: {DEFAULT_MAX_TETS}, or "
                             f"$MSE_MAX_TETS if set)")
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID,
                        help=f"samples along the longest axis (default: {DEFAULT_GRID})")
    parser.add_argument("--epw", type=float, default=None,
                        help="edges per wavelength for mesh resolution (bench use)")
    parser.add_argument("--min-feature", type=float, default=None,
                        metavar="METRES",
                        help="smallest feature to resolve, in METRES -- not "
                             "--units (default: the physics edge alone)")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="multiply every vertex by this before any "
                             "measurement (default: 1.0)")
    parser.add_argument("--materials-from", default=None, metavar="KEY",
                        help="object-store key holding the part-name -> n JSON "
                             "table. The TABLE is a file, not an argument: "
                             "argv is world-readable on the shared VM and this "
                             "grows with the part count. Absent means the "
                             "one-part vacuum path.")
    parser.add_argument("--quiet", action="store_true", help="less progress logging")
    return parser


def load_materials(store_in, job_id, key: str | None) -> dict | None:
    """Read the materials table named by ``--materials-from``.

    ``key`` is None for a one-part upload, and that returns None -- the vacuum
    path, byte-identical to before this existed.

    When ``key`` is given the object MUST be there and MUST parse. Every way it
    can fail raises, and none of them falls back to None:

        missing       the app decided this job has materials and launched a Job
                      for it; a network blip that read as "no table" would
                      solve the upload as vacuum and report success. That is the
                      whole defect class this feature is written against -- the
                      owner gets frequencies for a different object, and nothing
                      anywhere says so.
        not JSON      same, and just as silent.
        not an object a JSON list or scalar would sail through json.loads and
                      then fail much later inside the mesher, where the message
                      would be about geometry rather than about this file.

    The key is NOT defaulted from a constant here. The caller passes it, so
    there is exactly one spelling of the object's name in the tree rather than
    a copy on each side of the wire that can drift.
    """
    if key is None:
        return None
    try:
        raw = store_in.read(key)
    except Exception as exc:  # noqa: BLE001 - any read failure is this failure
        raise MESH.MeshingError(
            f"this job has a materials table but it could not be read from "
            f"{key!r}: {exc}"
        ) from exc
    try:
        table = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise MESH.MeshingError(
            f"the materials table at {key!r} is not valid JSON: {exc}"
        ) from exc
    if not isinstance(table, dict):
        raise MESH.MeshingError(
            f"the materials table at {key!r} must be a JSON object mapping "
            f"part name to refractive index, got {type(table).__name__}"
        )
    return table


def make_store(local_root: str | None, job_id: str) -> STORE.ObjectStore:
    if local_root:
        return STORE.LocalStore(local_root)
    return STORE.S3Store()


def solve(job_id: str, units: str, n_index: float, n_modes: int,
          store_in: STORE.ObjectStore, store_out: STORE.ObjectStore,
          max_tets: int, grid: int, verbose: bool = True,
          db: "mse_db.JobStore | None" = None,
          backend: str | None = None,
          min_feature_m: float | None = None,
          scale: float = 1.0,
          materials: dict | None = None):
    """Run the pipeline and write every result. Returns a summary dict.

    ``backend`` selects the solver backend (``cpu``/``gpu``); ``None`` reads
    ``MSE_SOLVER_BACKEND``. It is resolved once, here, and then passed explicitly
    to :func:`mse_eigen.solve_modes`, so the mesh-count guard below and the
    factorisation cannot disagree about which machine is running -- reading the
    env twice would let a test set it after the guard had already decided.

    ``min_feature_m`` (METRES, converted once at the upload form from its own
    nm/um/mm selector -- never from ``units``) and ``scale`` reach the mesh only,
    and they are the same two numbers the app used for its ``est_tets``; if they
    were applied in one place and not the other, routing would be wrong for
    exactly the detailed jobs the GPU exists for. That "same two numbers" is
    only meaningful with one length space: when the feature arrived in the
    object's units and the app and the mesher disagreed about which, the two
    estimates differed by a factor of the scale.

    ``materials`` maps a ``.obj`` part NAME to that part's ``n``, or ``None``
    for a one-part vacuum upload. The name is the key, not an index into the
    part list: geometry is re-derived from the upload on this side, so an index
    would be a second key that can drift from the one the owner assigned a
    material to, and a drifted index applies a material to the wrong region
    while every count still agrees.

    Every ``n`` must be finite and ``>= 1``. ``n < 1`` is refused rather than
    clamped: a phase velocity above c is not a lossy simulation, it is a
    different physical model, and silently accepting it would produce
    frequencies that look plausible and are not. ``None`` and ``{}`` are NOT
    interchangeable here -- ``None`` is vacuum, ``{}`` is a table the owner
    submitted naming no parts, which can only be a mismatch between the upload
    and the file.

    Raises on failure after recording the stage that failed; the caller turns
    that into a ``failed`` status and a non-zero exit.
    """
    if n_index <= 0:
        raise ValueError("The refractive index must be greater than zero.")
    if n_modes < 1:
        raise ValueError("At least one mode must be requested.")
    if materials is not None:
        if not isinstance(materials, dict) or not materials:
            raise ValueError(
                "the materials table must be a non-empty object mapping part "
                "name to refractive index"
            )
        for part, n_part in materials.items():
            if not isinstance(part, str) or not part:
                raise ValueError(
                    f"a materials table key must be a non-empty part name, "
                    f"got {part!r}"
                )
            if not isinstance(n_part, (int, float)) or isinstance(n_part, bool):
                raise ValueError(
                    f"material {part!r} has a non-numeric refractive index "
                    f"{n_part!r}"
                )
            if not float(n_part) > 0:
                raise ValueError(
                    f"material {part!r} has refractive index {n_part!r}; it "
                    f"must be finite and greater than zero"
                )
            if not np.isfinite(float(n_part)):
                raise ValueError(
                    f"material {part!r} has a non-finite refractive index "
                    f"{n_part!r}"
                )

    backend = backend or EIG.BACKEND.selected_backend()

    timings: dict[str, float] = {}
    t_start = time.time()

    def stage(name: str, message: str | None = None):
        if verbose:
            print(f"[mse] stage: {name}", flush=True)
        if db is not None:
            db.set_stage(job_id, "running", stage=name, message=message)

    # -- meshing ----------------------------------------------------------
    stage("meshing")
    t0 = time.time()
    obj_bytes = store_in.read(STORE.job_key(job_id, "input.obj"))
    # ONE branch, taken on whether the upload has a materials table:
    #
    #   materials is None  the one-part vacuum path, byte-identical to before
    #                      this feature existed. load_surface takes the surface
    #                      and tetrahedralize runs the TetGen call it always did.
    #   materials present  a multi-part upload: load_parts splits and VALIDATES
    #                      (closed, non-overlapping, non-touching, one wall,
    #                      a provable seed per region) and tetrahedralize meshes
    #                      the whole PLC with a region attribute per part.
    #
    # The branch is on the presence of the table and not on the part count,
    # because a one-part file WITH a materials table must still be validated
    # (the table must name that part) even though it needs no regions.
    parts = None
    if materials is not None:
        parts = PARTS.load_parts(obj_bytes, units=units)
        if len(parts) == 1 and parts.names[0] not in materials:
            raise MESH.MeshingError(
                f"no refractive index given for part {parts.names[0]!r}. "
                "Every part needs one."
            )
        surface = PARTS.merged_surface(parts)
        points_m, tets, region_of_tet = MESH.tetrahedralize(
            surface, units, n_modes=n_modes, n_index=n_index,
            max_tets=max_tets, verbose=verbose,
            min_feature_m=min_feature_m, scale=scale,
            materials=materials, parts=parts,
        )
    else:
        surface, _scale = MESH.load_surface(obj_bytes, units=units)
        points_m, tets, region_of_tet = MESH.tetrahedralize(
            surface, units, n_modes=n_modes, n_index=n_index,
            max_tets=max_tets, verbose=verbose,
            min_feature_m=min_feature_m, scale=scale,
        )
    timings["meshing"] = time.time() - t0

    # eps_r per tet = n_region^2, from the regions the MESH actually reported.
    # This is the ONE place eps_r is defined: the physics (the mass matrix) and
    # everything that talks about it read this array, rather than each
    # recomputing n^2 from the materials table and risking a disagreement.
    eps_by_tet = None
    if parts is not None and len(parts) > 1:
        materials_by_region = parts.region_materials(materials)
        known = set(materials_by_region)
        seen = set(int(r) for r in np.unique(region_of_tet).tolist())
        unknown = seen - known
        if unknown:
            # A tet tagged with a region the plan never asked for means the
            # attribute pass and the mesh disagree -- exactly the failure that
            # would silently give some tets vacuum and others a material. Name
            # it rather than default the missing region to 1.0.
            raise MESH.MeshingError(
                f"internal error: {len(unknown)} region id(s) "
                f"{sorted(unknown)} appear in the mesh but were never "
                "requested. This is a bug, not a problem with your upload."
            )
        # Index by region id; region_of_tet is a tet-indexed array so this is
        # a straight gather, with no frame correction (the canonical frame maps
        # points, not tet indices).
        lookup = np.ones(max(known) + 1, dtype=np.float64)
        for region_id, n_region in materials_by_region.items():
            lookup[int(region_id)] = float(n_region) ** 2
        eps_by_tet = lookup[region_of_tet]

    # -- is this machine the right one? ------------------------------------
    #
    # The upload-time estimate chose which queue this job went on, and an
    # estimate is exactly what it says: the mesh's own count is the first point
    # where the truth is known, and it is known *here*, before assembly, which
    # is where the expensive part starts. So this is the cheap moment to admit
    # the route was wrong.
    #
    # Two different mistakes, and they must not be conflated:
    #
    #   too big for THIS processor (cpu, over MSE_CPU_MAX_TETS)
    #       The job is fine; the machine is wrong. Raise TooBigForProcessor so
    #       the app can re-route it to the Jetson queue. Failing here would
    #       throw away work that is about to run perfectly well elsewhere.
    #
    #   too big for ANY processor (over MSE_MAX_TETS)
    #       No backend can do it. Raise MeshingError, which the owner sees as a
    #       plain "too detailed" message -- there is no point re-routing.
    #
    # Without this the estimate is load-bearing on its own, and a job that
    # under-estimated its way onto the CPU would be killed by the memory limit
    # with a message that says nothing about why.
    n_tets_actual = int(tets.shape[0])
    if str(backend).lower() == "cpu" and n_tets_actual > cpu_max_tets():
        raise TooBigForProcessor(
            f"this model needs {n_tets_actual:,} elements, over the "
            f"{cpu_max_tets():,} the cluster can solve; re-route to the GPU"
        )
    if n_tets_actual > max_tets:
        raise MESH.MeshingError(
            f"This model is too detailed: it needs {n_tets_actual:,} elements "
            f"and the limit is {max_tets:,}. Please simplify the model."
        )

    # -- assembling -------------------------------------------------------
    stage("assembling")
    t0 = time.time()
    edges, tet_edges, signs = F.build_edges(tets)
    n_edges = int(edges.shape[0])
    # ``B`` is the eps-weighted mass: with materials it enters solve_modes
    # already scaled per element, which is what makes the divergence-free
    # projector enforce div(eps E) = 0 without a second weight parameter. A
    # caller that forgot ``eps_by_tet`` here would assemble the VACUUM mass and
    # silently project the wrong subspace, so this argument is gated by a test
    # that drives this function end to end.
    A, B = F.assemble_global(points_m, tets, tet_edges, n_edges,
                             eps_by_tet=eps_by_tet)
    constrained = F.boundary_edges(tets, edges)
    A_free, B_free, free_edges = F.pec_eliminate(A, B, constrained, n_edges)
    timings["assembling"] = time.time() - t0
    if verbose:
        print(f"[mse] edges {n_edges:,}, free {A_free.shape[0]:,}, "
              f"PEC {constrained.size:,}", flush=True)

    # -- solving ----------------------------------------------------------
    stage("solving")
    t0 = time.time()
    # Not gated on `verbose`: which machine factorised the matrices is not
    # progress chatter, it is the first thing to check when a run is slower or
    # less accurate than expected. Reported after the call, from what the solver
    # actually used, so an env var that was ignored cannot look like one that
    # took effect.
    eigenvalues, eigenvectors, sigma = EIG.solve_modes(
        A_free, B_free, n_modes, points_m,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free_edges, constrained=constrained, verbose=verbose,
        backend=backend,
    )
    residual_note = (
        f", worst refined residual {EIG.LAST_WORST_RESIDUAL:.2e}"
        if EIG.LAST_WORST_RESIDUAL is not None else ""
    )
    print(f"[mse] solver backend: {EIG.LAST_BACKEND_DESCRIPTION}{residual_note}",
          flush=True)
    timings["solving"] = time.time() - t0

    # The projection removes the gradient null space exactly, so a zero or
    # near-zero eigenvalue here means something is wrong: assert rather than
    # silently filter, and let the failure surface.
    largest = float(np.max(np.abs(eigenvalues))) if eigenvalues.size else 0.0
    spurious = eigenvalues <= EIG.SPURIOUS_FRACTION * max(largest, 1e-300)
    if bool(np.any(spurious)):
        raise EIG.EigenError(
            f"the solver returned {int(spurious.sum())} non-physical modes"
        )

    k, k0, freq_hz, wavelength_m = EIG.to_physical(eigenvalues, n_index)

    # -- writing ----------------------------------------------------------
    stage("writing")
    t0 = time.time()
    modes_json = write_results(
        job_id, store_out, surface, points_m, tets, edges, tet_edges, signs,
        eigenvectors, k, k0, freq_hz, wavelength_m, units, n_index, grid,
        free_edges=free_edges, n_edges=n_edges, verbose=verbose,
        min_feature_m=min_feature_m, scale=scale,
    )
    timings["writing"] = time.time() - t0
    timings["total"] = time.time() - t_start

    if db is not None:
        db.delete_modes(job_id)
        db.insert_modes(job_id, [
            (i + 1, float(k0[i]), float(freq_hz[i]), float(wavelength_m[i]))
            for i in range(len(freq_hz))
        ])

    rss = peak_rss_mb()
    if verbose:
        print(f"[mse] peak RSS {rss:.0f} MB; timings "
              f"{ {k2: round(v2, 1) for k2, v2 in timings.items()} }", flush=True)

    if db is not None:
        db.set_stage(job_id, "done", stage="done", peak_rss_mb=rss)

    return {
        "job_id": job_id,
        "n_tets": int(tets.shape[0]),
        "n_edges": n_edges,
        "n_modes": int(len(freq_hz)),
        "peak_rss_mb": rss,
        "timings": timings,
        "modes": modes_json["modes"],
    }


def write_results(job_id, store_out, surface, points_m, tets, edges, tet_edges,
                  signs, eigenvectors, k, k0, freq_hz, wavelength_m,
                  units, n_index, grid, free_edges=None, n_edges=None,
                  verbose=True, min_feature_m=None, scale=1.0):
    """Write surface.glb, modes.json, mode_XX.bin and mode_XX.vtu.

    ``eigenvectors`` are the PEC-reduced vectors; ``free_edges``/``n_edges`` let
    the sampler place them back into the full edge numbering.
    """
    import io

    # surface in the user's units (what the viewer displays)
    glb = surface.export(file_type="glb")
    store_out.write(STORE.job_key(job_id, "surface.glb"), glb)

    nx, ny, nz = FLD.grid_shape(points_m, grid)
    origin = None
    spacing = None

    # Locate the grid points ONCE for the whole job, then reuse it for every
    # mode. Which tet contains a point, and where inside it, does not depend on
    # the eigenvector, so repeating it per mode was pure waste -- and it was
    # waste of the most expensive kind: the brute-force version allocated a
    # (chunk, n_tets, 3) array per mode and OOM-killed the Job at its 14 Gi
    # limit after a 7 s solve.
    samples, gshape, gorigin, gspacing = FLD.sample_grid(points_m, grid)
    located = FLD.locate_samples(points_m, tets, samples)
    if verbose:
        print(f"[mse] located {located[0].size:,} of {samples.shape[0]:,} grid "
              f"points inside the mesh", flush=True)

    modes = []
    for index in range(len(freq_hz)):
        field, shape, origin, spacing = FLD.sample_field(
            points_m, tets, edges, tet_edges, signs,
            np.asarray(eigenvectors[:, index], dtype=float),
            n_longest=grid, free_edges=free_edges, n_edges=n_edges,
            located=located, grid=(samples, gshape, gorigin, gspacing),
        )
        store_out.write(
            STORE.job_key(job_id, f"mode_{index + 1:02d}.bin"),
            np.ascontiguousarray(field, dtype=np.float32).tobytes(),
        )
        vtu = io.BytesIO()
        FLD.write_vtu(vtu, field, shape, origin, spacing)
        store_out.write(
            STORE.job_key(job_id, f"mode_{index + 1:02d}.vtu"), vtu.getvalue()
        )

        modes.append({
            "index": index + 1,
            # k0 is the FREE-SPACE wavenumber, which is what the contract calls
            # for; `k` (in-medium) is what fell out of the eigensolver.
            "k0_per_m": float(k0[index]),
            "k_per_m": float(k[index]),
            "freq_hz": float(freq_hz[index]),
            "vacuum_wavelength_m": float(wavelength_m[index]),
        })
        if verbose:
            print(f"[mse] mode {index + 1:02d}: {freq_hz[index] / 1e9:.4f} GHz  "
                  f"lambda0 {wavelength_m[index] * 1e3:.4f} mm", flush=True)

    modes_json = {
        "units": units,
        "n_index": float(n_index),
        "n_tets": int(tets.shape[0]),
        "n_edges": int(edges.shape[0]),
        # Which machine factorised K and S, and for the GPU path how much
        # residual the refinement left. Recorded because two runs of the same
        # upload on different backends should differ only in time, and a reader
        # comparing them needs to know which is which without the log.
        "solver_backend": EIG.LAST_BACKEND_DESCRIPTION,
        "solver_worst_residual": EIG.LAST_WORST_RESIDUAL,
        # What the mesh was actually asked for. The units alone do not pin the
        # physical size once a scale factor exists, and the viewer, the bench
        # and the next reader all need to know which object produced these
        # frequencies without re-reading the Job.
        "scale": float(scale),
        "min_feature": None if min_feature_m is None else float(min_feature_m),
        # The median element edge the mesher ACTUALLY achieved, in metres of the
        # scaled object -- the measurement, beside the target the line above
        # states. Read from the mesher's module state rather than threaded
        # through as a parameter: tetrahedralize returns (points, tets), and
        # this describes the run that just happened, the same way
        # LAST_BACKEND_DESCRIPTION does. The job page shows it next to the
        # requested resolution so the owner can see the allowance being used
        # rather than having to trust that it was.
        "achieved_median_edge_m": MESH.LAST_ACHIEVED_EDGE_M,
        "grid": {
            "nx": int(nx), "ny": int(ny), "nz": int(nz),
            "origin": [float(v) for v in origin],
            "spacing": [float(v) for v in spacing],
        },
        "modes": modes,
    }
    store_out.write(
        STORE.job_key(job_id, "modes.json"),
        json.dumps(modes_json, indent=2).encode("utf-8"),
    )
    return modes_json


def termination_message(exc: BaseException, limit: int = 4096) -> str | None:
    """The text to leave in ``/dev/termination-log``, or ``None`` for no signal.

    Only the re-routable outcome is worth a channel that outlives the container:
    every other failure is already named for the owner by ``mse.jobs.message``,
    which the solver writes to the database itself. This one is different -- the
    app has to *act* on it (move the job to another queue), and the app is not
    watching the database for it.

    Truncated to ``limit`` because Kubernetes caps the field at 4 KiB and
    silently truncates over it; truncating here means the token is never the
    part that gets cut.
    """
    if not isinstance(exc, TooBigForProcessor):
        return None
    return f"{REROUTE_TOKEN}: {exc}"[:limit]


def _write_termination_log(text: str, path: str = "/dev/termination-log") -> bool:
    """Best-effort write of the pod's termination message. Never raises.

    Absent when running locally (no such path) or under ``--local-in``, which is
    not a failure: the local paths do not consume the signal.
    """
    try:
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
        return True
    except OSError:
        return False


#: The words that mean "the GPU ran and said no to THIS job". Spelled the ways
#: cuDSS (CUDSS_STATUS_ALLOC_FAILED), nvmath's prose and the driver say it.
#: Shared by both ``_names_*`` helpers below so the two cannot disagree about
#: which side of the line an allocation failure falls on.
ALLOC_FAILURE_WORDS = (
    "alloc_failed", "alloc failed", "allocation failed", "allocation_failed",
    "out of memory", "outofmemory", "insufficient memory",
)
#: The words that mean "the GPU would not come up" -- the opposite verdict.
GPU_INIT_WORDS = (
    "initializ", "initialis",
    "no device", "no cuda", "no gpu",
    "device not found", "driver",
    "context creation", "context init",
    "nvrtc", "cuda_driver",
)


def _names_an_allocation_failure(exc: BaseException) -> bool:
    """Does this ``EigenError`` come from a cuDSS allocation failure?

    The type cannot say: ``mse_eigen`` wraps every cuDSS status as
    ``EigenError(...) from exc``, so ALLOC_FAILED and a genuinely unsolvable
    problem arrive looking identical. The status NAME is carried in the message
    and in the wrapped exception's, which is the only evidence there is.

    Walks the chain because the word may sit either side of the re-raise: nvmath
    puts the status in its own exception's message, and ``mse_eigen``'s wrapper
    may or may not repeat it.
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    depth = 0
    while node is not None and depth < 10 and id(node) not in seen:
        seen.add(id(node))
        depth += 1
        text = f"{type(node).__name__}: {node}".lower()
        if any(word in text for word in ALLOC_FAILURE_WORDS):
            return True
        node = node.__cause__ or node.__context__
    return False


def _names_a_gpu_init_failure(exc: BaseException) -> bool:
    """Could this ``EigenError`` mean the GPU never came up?

    The mirror of ``_names_an_allocation_failure``, and it exists for the same
    reason: ``mse_eigen`` wraps every cuDSS failure the same way, so "the GPU is
    absent or its driver is broken" and "this object has no solution" are the
    same TYPE. The distinction matters to the OWNER, not just to the worker's
    retry logic: an object-blaming sentence ("try a simpler object") is FALSE for
    a GPU fault, and it is shown on the job page while the job is being retried.

    Checked AFTER the allocation words and in the same chain walk, so a message
    carrying both reads as the allocation failure (the GPU ran and answered).
    """
    seen: set[int] = set()
    node: BaseException | None = exc
    depth = 0
    while node is not None and depth < 10 and id(node) not in seen:
        seen.add(id(node))
        depth += 1
        name = type(node).__name__
        text = f"{name}: {node}".lower()
        if any(word in text for word in ALLOC_FAILURE_WORDS):
            return False
        if name in ("CUDARuntimeError", "CUDADriverError"):
            return True
        if any(word in text for word in GPU_INIT_WORDS):
            return True
        node = node.__cause__ or node.__context__
    return False


def owner_message(exc: BaseException) -> str:
    """Plain-English text for the job page.

    Only the failures the user can act on (bad upload, too big, too coarse) are
    shown verbatim; anything else gets a generic apology, because a stack trace
    or a numpy message is not something the owner can use. The technical detail
    still goes to the job log.
    """
    if isinstance(exc, (MESH.MeshingError, FLD.SamplingError)):
        return str(exc)
    if isinstance(exc, TooBigForProcessor):
        # Reached only when nothing re-routed it. The owner cannot act on a
        # routing failure, so they get the same plain "too detailed" answer they
        # would have got from the cap -- naming the scheduler's problem to them
        # would be noise, and the log carries the real reason.
        return (
            "This model is too large to solve with the processors available "
            "right now. Please try again later, or upload a simpler model."
        )
    if isinstance(exc, ValueError):
        return str(exc)
    if isinstance(exc, EIG.EigenError):
        if _names_an_allocation_failure(exc):
            # ALLOC_FAILED is not a verdict about the object's difficulty, and
            # the generic EigenError text below would be a LIE about it: the
            # object is not too complex to solve, the GPU had no room for it
            # right now. That same status also comes from other GPU memory
            # users and from a previous job's context in the warm worker, so a
            # coarser setting is not always the answer -- it is offered first
            # because the owner can act on it, and "try again later" is what is
            # actually true.
            return (
                "This object could not fit in the GPU's memory. If it is very "
                "detailed, try a coarser detail setting; otherwise please try "
                "again later."
            )
        if _names_a_gpu_init_failure(exc):
            # The GPU was absent or its driver was broken. Blaming the object
            # here -- which is what the sentence below does -- would be false,
            # and this text is shown on the job page while the job is requeued,
            # so the owner would be told to simplify an object that is fine.
            # The wording must not promise a retry that may not happen either:
            # the app gives up after MAX_RELEASES, and this may be the last one.
            return (
                "The graphics processor was not available for this run. This "
                "is a problem on our side, not with your object -- the job will "
                "be retried, and please try again later if it does not finish."
            )
        return (
            "This object could not be solved for the requested number of modes. "
            "Try a simpler object, fewer modes, or a larger size."
        )
    if isinstance(exc, STORE.StorageError):
        return str(exc)
    return (
        "Something went wrong while computing the modes. The problem has been "
        "recorded; please try again, or contact support if it keeps happening."
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    # keep the resolution rule in one place
    if args.epw is not None:
        os.environ["MSE_EDGES_PER_WAVELENGTH"] = str(args.epw)

    # Resolve the cap ONCE, here, after parsing. The flag defaults to None so
    # this call is the only place ``max_tets()`` is consulted -- a parser default
    # would have frozen the value at build_parser() time and a test (or a later
    # set_max_tets) would silently not apply.
    effective_max_tets = max_tets() if args.max_tets is None else int(args.max_tets)
    # Resolved for this run, and handed back in the ``finally`` below.
    # ``set_max_tets`` writes a module global, so a run that did not restore it
    # would leave every LATER caller in the process -- including callers that
    # pass no cap at all -- solving at this run's limit. Threading
    # ``effective_max_tets`` into ``solve()`` keeps this run's own behaviour
    # independent of the global, so the restore costs nothing.
    previous_max_tets = max_tets()
    set_max_tets(effective_max_tets)

    verbose = not args.quiet
    job_id = str(args.job_id)

    db = None
    store_in = store_out = None
    try:
        store_in = make_store(args.local_in, job_id)
        store_out = make_store(args.local_out or args.local_in, job_id)
        if not (args.local_in or args.local_out):
            db = mse_db.JobStore()
            db.connect()
            # The schema is owned by the web app's initContainer, which applies
            # it once before any Job can be queued. A Job must NOT run DDL: it
            # would repeat identical work on every upload, and every Job failing
            # at startup because the script was not where this file expected it
            # is exactly the deploy failure that removed this call.
        materials = load_materials(store_in, job_id, args.materials_from)
        summary = solve(
            job_id, args.units, args.n_index, args.n_modes,
            store_in, store_out, effective_max_tets, args.grid,
            verbose=verbose, db=db,
            min_feature_m=args.min_feature, scale=args.scale,
            materials=materials,
        )
    except BaseException as exc:  # noqa: BLE001 - record, then re-raise as exit code
        print(f"[mse] FAILED: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        traceback.print_exc()
        rss = peak_rss_mb()
        print(f"[mse] peak RSS {rss:.0f} MB", flush=True)
        if db is not None:
            db.fail(job_id, owner_message(exc), peak_rss_mb=rss)
        # Leave the re-route signal where the app can read it after this process
        # is gone. Done LAST and only on this exception: the termination message
        # is read from the dead container, so anything written here supersedes
        # nothing, and a crash before this line still leaves the database row
        # (written above) naming the failure for the owner.
        signal = termination_message(exc)
        if signal is not None:
            if _write_termination_log(signal):
                print(f"[mse] wrote {REROUTE_TOKEN} to the termination log",
                      flush=True)
            else:
                print(f"[mse] {REROUTE_TOKEN}: could not write the termination "
                      f"log; the app will see a plain failure", flush=True)
        return 1
    finally:
        if db is not None:
            db.close()
        # See above. In the ``finally`` so a FAILED run restores it too: the
        # failure path is exactly when a later run is most likely.
        set_max_tets(previous_max_tets)

    if verbose:
        print(f"[mse] done: {summary['n_modes']} modes, "
              f"{summary['n_tets']:,} tets, {summary['n_edges']:,} edges", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
