"""Run a claimed time-domain (FDTD) job: the request in, the results out.

This is the ``kind = 'td'`` counterpart to ``pc_solve``. It owns the translation
from the row's stored request (``tds``) to the solver's call, so a test can
drive it without a lease, a store or a GPU.

**The request is re-derived, never trusted.** As in ``pc_solve``, the grid and
the TF/SF box are rebuilt here from the request and the upload, never read back
from a stored estimate: a count is not a grid.

``solve_td`` (4.2) runs it: the upload is split into parts, laid out on a
graded grid (``fdtd_layout``), sized (``fdtd.estimate_run``) and checked
against the free memory before a single device array exists, then stepped
with a 45-minute stop. Result files are 4.3.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

_SOLVER_DIR = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"

#: Run-time limit: the 30-minute estimate limit plus a margin (spec).
RUN_LIMIT_S = 45 * 60


class TDSolveError(Exception):
    """The claimed job does not carry a request this module can solve.

    A distinct type for the reason ``PCSolveError`` is one: a malformed row is
    deterministic (fail the job, do not requeue it), unlike a transient solver
    error (release and retry).
    """


#: The request keys, in one place. The phase-5 form's ``request_json`` will
#: write exactly these and a test will pin the two lists against each other, as
#: ``test_pc_solve`` does for ``pcs``. Every key is always present; a key that
#: does not apply to the chosen source holds JSON null (``direction`` for a
#: point source, ``position``/``polarisation`` axis rules for a plane wave,
#: ``samples``/``sample_duration_s`` unless the profile is ``drawn``,
#: ``bandwidth`` for ``cw``). Present-but-null and absent are different on
#: purpose: absent means the writer and this reader disagree about the
#: contract, which is the defect this list exists to catch.
#:
#: The TF/SF box is not a key: it is derived from the geometry's bounding box
#: and the clearance rule, for the reason ``k_points`` is not a pc key.
REQUEST_KEYS = ("boundary", "source_kind", "profile", "frequency_hz",
                "bandwidth", "position", "direction", "polarisation",
                "samples", "sample_duration_s", "duration_s", "probes", "dx_m")


def request_from(job: dict) -> dict:
    """The request dict out of a claimed job, or raise naming what is wrong.

    ``tds`` is NULL for every other kind and ``{}`` would be a td job with every
    field missing; both are refused. The check is by KEY, not truthiness, since
    ``null`` and ``0`` are legitimate values for some keys. Keys outside the
    contract are dropped rather than passed on, so results written back into
    the same column can never be read as input.
    """
    tds = job.get("tds")
    if not isinstance(tds, dict):
        raise TDSolveError(
            f"a td job must carry a 'tds' request object, got {type(tds).__name__}"
        )
    missing = [k for k in REQUEST_KEYS if k not in tds]
    if missing:
        raise TDSolveError(f"the 'tds' request is missing {', '.join(missing)}")
    return {k: tds[k] for k in REQUEST_KEYS}


class JetsonBusy(Exception):
    """The job needs more device memory than is free right now.

    NOT a ``TDSolveError``: it is no verdict on the job. The worker maps it to
    the uncounted "Jetson busy" deferral, never to ``/failed`` or ``/release``.
    """


def _solver():
    if str(_SOLVER_DIR) not in sys.path:
        sys.path.insert(0, str(_SOLVER_DIR))
    import fdtd, fdtd_layout, mse_common, mse_parts  # noqa: E401,PLC0415 - late
    return fdtd, fdtd_layout, mse_common, mse_parts


def _backend():
    """cupy when a device is present, else numpy (a CPU-only test box)."""
    try:
        import cupy  # noqa: PLC0415
        if cupy.cuda.runtime.getDeviceCount() > 0:
            return cupy
    except Exception:  # noqa: BLE001 - no cupy or no device: run on the host
        pass
    import numpy  # noqa: PLC0415
    return numpy


def available_bytes(xp) -> int:
    """Memory a new run can take. The Orin's GPU shares system RAM, so this
    is the kernel's MemAvailable, plus the blocks cupy's pool already holds
    for this process (free to reuse, but not counted by the kernel)."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                free = int(line.split()[1]) * 1024
                break
        else:
            raise OSError("/proc/meminfo has no MemAvailable line")
    pool = getattr(xp, "get_default_memory_pool", None)
    if pool is not None:
        free += int(pool().free_bytes())
    return free


def is_out_of_memory(exc: BaseException) -> bool:
    """A host or device allocation failure. cupy's OutOfMemoryError is a
    MemoryError subclass; a raw CUDA allocation failure arrives as a
    CUDARuntimeError naming cudaErrorMemoryAllocation."""
    if isinstance(exc, MemoryError):
        return True
    return "cudaErrorMemoryAllocation" in str(exc)


def _gb(n) -> str:
    return f"{n / 1e9:.1f} GB"


def solve_td(job: dict, store_in, store_out=None, stage=None, *, xp=None,
             free_bytes=None, clock=time.monotonic) -> dict:
    """Run a claimed td job. Raises ``TDSolveError`` (a readable verdict on
    the job), ``JetsonBusy`` (try again later), or passes anything else up
    for the worker's classifier. ``xp``, ``free_bytes`` (a callable) and
    ``clock`` are seams for the tests."""
    fdtd, fdtd_layout, mse_common, mse_parts = _solver()
    request = request_from(job)
    job_id = str(job["job_id"])
    xp = _backend() if xp is None else xp
    free_bytes = (lambda: available_bytes(xp)) if free_bytes is None else free_bytes

    try:
        length_scale = (mse_common.unit_scale(str(job.get("units")))
                        * float(job.get("scale") or 1.0))
        names, parts = mse_parts.split_obj_parts(
            store_in.read(f"jobs/{job_id}/input.obj"))
        lay = fdtd_layout.layout(names, parts, request,
                                 length_scale=length_scale,
                                 materials=job.get("materials"),
                                 n_index=job.get("n_index"))
        est = fdtd.estimate_run(lay["nodes"], duration=float(request["duration_s"]),
                                boundary=request["boundary"], metals=lay["metals"],
                                pec=lay["pec"])
        fdtd.check_limits(est)
    except (fdtd.FDTDError, ValueError) as exc:
        # FDTDError covers LayoutError, VoxelError and EstimateError; the
        # meshing errors are ValueErrors. Each already reads as a sentence.
        raise TDSolveError(str(exc)) from exc

    free = int(free_bytes())
    if est["bytes"] > free:
        raise JetsonBusy(f"The Jetson is busy: this run needs {_gb(est['bytes'])} "
                         f"of memory and {_gb(free)} is free. It will start "
                         f"when memory frees up.")

    if request["source_kind"] == "plane":
        place = {"kind": "plane", "direction": request["direction"],
                 "box": lay["box"]}
    else:
        place = {"cell": lay["cell"]}
    try:
        source = fdtd.Source(
            frequency=float(request["frequency_hz"]), profile=request["profile"],
            bandwidth=request["bandwidth"], component=request["polarisation"],
            samples=request["samples"], duration=request["sample_duration_s"],
            **place)
    except fdtd.FDTDError as exc:
        raise TDSolveError(str(exc)) from exc
    terms = [fdtd.LorentzDrude(name, region)
             for name, region in dict(lay["metals"]).items()]

    started = clock()

    def check(step):
        if clock() - started > RUN_LIMIT_S:
            raise TDSolveError(
                f"The run was stopped after {RUN_LIMIT_S // 60} minutes, the "
                f"limit for one run, at step {step:,} of {est['steps']:,}. "
                f"Increase dx or shorten the duration.")

    if stage is not None:
        stage.stage = "solving"
    try:
        result = fdtd.simulate(lay["eps"], source, est["steps"], lay["probes"],
                               spacings=lay["spacings"],
                               boundary=request["boundary"], pec=lay["pec"],
                               xp=xp, ade_terms=terms, check=check)
    except TDSolveError:
        raise
    except Exception as exc:  # noqa: BLE001 - only OOM is translated
        if is_out_of_memory(exc):
            raise TDSolveError(
                f"The Jetson ran out of memory during the run (it needed about "
                f"{_gb(est['bytes'])}; another job probably started after this "
                f"one was checked). Please submit it again.") from exc
        raise
    return {"job_id": job_id, "kind": "td", "modes": [], "result": result,
            "estimate": est, "caveats": list(lay["caveats"])}
