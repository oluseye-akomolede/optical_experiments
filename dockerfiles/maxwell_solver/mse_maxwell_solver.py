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

import mse_db  # noqa: E402
import mse_eigen as EIG  # noqa: E402
import mse_fem as F  # noqa: E402
import mse_fields as FLD  # noqa: E402
import mse_meshing as MESH  # noqa: E402
import mse_storage as STORE  # noqa: E402

#: default element cap. Chosen so a worst-case solve stays well under the 14Gi
#: container limit; the lead calibrates the real figure on the Jetson.
DEFAULT_MAX_TETS = 80_000

#: elements above which the CLUSTER cannot solve a job, but the GPU still can.
#: The upload-time estimate decides the route, but an estimate can be wrong in
#: either direction; this is the exact count re-deciding, at the first point the
#: exact count exists. Deliberately lower than DEFAULT_MAX_TETS: it is a
#: routing boundary, not a capability limit.
DEFAULT_CPU_MAX_TETS = 150_000

#: what a ``TooBigForProcessor`` writes into its termination message. The app
#: reads this string back out of the pod's last state to tell "this job is on
#: the wrong machine" from "this job is broken" -- see the class docstring.
REROUTE_TOKEN = "MSE_REROUTE"


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
    parser.add_argument("--max-tets", type=int,
                        default=int(os.environ.get("MSE_MAX_TETS", DEFAULT_MAX_TETS)),
                        help=f"cap on tetrahedra (default: {DEFAULT_MAX_TETS})")
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID,
                        help=f"samples along the longest axis (default: {DEFAULT_GRID})")
    parser.add_argument("--epw", type=float, default=None,
                        help="edges per wavelength for mesh resolution (bench use)")
    parser.add_argument("--quiet", action="store_true", help="less progress logging")
    return parser


def make_store(local_root: str | None, job_id: str) -> STORE.ObjectStore:
    if local_root:
        return STORE.LocalStore(local_root)
    return STORE.S3Store()


def solve(job_id: str, units: str, n_index: float, n_modes: int,
          store_in: STORE.ObjectStore, store_out: STORE.ObjectStore,
          max_tets: int, grid: int, verbose: bool = True,
          db: "mse_db.JobStore | None" = None,
          backend: str | None = None):
    """Run the pipeline and write every result. Returns a summary dict.

    ``backend`` selects the solver backend (``cpu``/``gpu``); ``None`` reads
    ``MSE_SOLVER_BACKEND``. It is resolved once, here, and then passed explicitly
    to :func:`mse_eigen.solve_modes`, so the mesh-count guard below and the
    factorisation cannot disagree about which machine is running -- reading the
    env twice would let a test set it after the guard had already decided.

    Raises on failure after recording the stage that failed; the caller turns
    that into a ``failed`` status and a non-zero exit.
    """
    if n_index <= 0:
        raise ValueError("The refractive index must be greater than zero.")
    if n_modes < 1:
        raise ValueError("At least one mode must be requested.")

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
    surface, _scale = MESH.load_surface(obj_bytes, units=units)
    points_m, tets = MESH.tetrahedralize(
        surface, units, n_modes=n_modes, n_index=n_index,
        max_tets=max_tets, verbose=verbose,
    )
    timings["meshing"] = time.time() - t0

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
    A, B = F.assemble_global(points_m, tets, tet_edges, n_edges)
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
                  verbose=True):
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
        summary = solve(
            job_id, args.units, args.n_index, args.n_modes,
            store_in, store_out, args.max_tets, args.grid,
            verbose=verbose, db=db,
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

    if verbose:
        print(f"[mse] done: {summary['n_modes']} modes, "
              f"{summary['n_tets']:,} tets, {summary['n_edges']:,} edges", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
