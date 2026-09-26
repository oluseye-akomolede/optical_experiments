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
          db: "mse_db.JobStore | None" = None):
    """Run the pipeline and write every result. Returns a summary dict.

    Raises on failure after recording the stage that failed; the caller turns
    that into a ``failed`` status and a non-zero exit.
    """
    if n_index <= 0:
        raise ValueError("The refractive index must be greater than zero.")
    if n_modes < 1:
        raise ValueError("At least one mode must be requested.")

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
    eigenvalues, eigenvectors, sigma = EIG.solve_modes(
        A_free, B_free, n_modes, points_m,
        edges=edges, tets=tets, tet_edges=tet_edges,
        free_edges=free_edges, constrained=constrained, verbose=verbose,
    )
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

    modes = []
    for index in range(len(freq_hz)):
        field, shape, origin, spacing = FLD.sample_field(
            points_m, tets, edges, tet_edges, signs,
            np.asarray(eigenvectors[:, index], dtype=float),
            n_longest=grid, free_edges=free_edges, n_edges=n_edges,
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


def owner_message(exc: BaseException) -> str:
    """Plain-English text for the job page.

    Only the failures the user can act on (bad upload, too big, too coarse) are
    shown verbatim; anything else gets a generic apology, because a stack trace
    or a numpy message is not something the owner can use. The technical detail
    still goes to the job log.
    """
    if isinstance(exc, (MESH.MeshingError, FLD.SamplingError)):
        return str(exc)
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
            db.apply_schema()
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
