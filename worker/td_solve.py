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
with a 45-minute stop. ``write_results`` (4.3) writes what it produced:

``trace.bin``      every probe's trace, float32 little-endian, C order,
                   shaped ``(n_probes, steps)``; row ``p`` is probe ``p``.
``spectrum.json``  the manifest: the time step, the probes, each probe's
                   spectrum over the source band, the trace's and the movie's
                   shapes, and the caveats. Its presence is what makes a td
                   job's results exist (the app's done route checks it).
``frame_NN.bin``   movie frame ``NN`` (1-based, at most ``fdtd.MAX_FRAMES``):
                   ``|E|`` float32 little-endian, C order, shaped
                   ``frames.shape`` = (sx, sy, sz), in the grid's x, y, z order
                   (NOT the cavity field files' (nz, ny, nx)).

The page serves all three through ``/jobs/<id>/files/<name>`` (404 until the
job is done). Storage names are flat because the worker API refuses any
name with a path separator.
"""
from __future__ import annotations

import json
import multiprocessing
import pickle
import signal
import sys
import time

import numpy as np
from pathlib import Path

_SOLVER_DIR = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"

#: Run-time limit: the 30-minute estimate limit plus a margin (spec).
RUN_LIMIT_S = 45 * 60
#: How long past ``RUN_LIMIT_S`` the worker waits for a child that has not
#: stopped itself before killing it (it is stuck inside one step).
STOP_GRACE_S = 5 * 60
#: The longest the worker lets a child spend before it reports ``solving``
#: (reading the object, laying out and voxelising the grid), from spawn.
SETUP_LIMIT_S = RUN_LIMIT_S + STOP_GRACE_S

#: The result files' names (see the module docstring). ``FRAME_NAME`` takes the
#: 1-based frame number; two digits hold ``fdtd.MAX_FRAMES`` (60).
TRACE_NAME = "trace.bin"
SPECTRUM_NAME = "spectrum.json"
FRAME_NAME = "frame_{:02d}.bin"
SPECTRUM_VERSION = 1


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
                "samples", "sample_duration_s", "duration_s", "probes", "dx_m",
                "coatings")


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


def available_bytes(xp, meminfo="/proc/meminfo") -> int:
    """Memory a new run can take. The Orin's GPU shares system RAM, so this
    is the kernel's MemAvailable, plus the blocks cupy's pool already holds
    for this process (free to reuse, but not counted by the kernel).
    ``meminfo`` is a seam for the tests."""
    with open(meminfo) as fh:
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


def oom_message(need=None) -> str:
    """The owner's sentence for a run that ran out of memory: raised in the
    run (``need`` known) or a kernel kill seen from outside (``need`` not)."""
    needed = "" if need is None else f" (it needed about {_gb(need)})"
    return (f"The Jetson ran out of memory during the run{needed}; another job "
            f"probably started after this one was checked. Please submit it "
            f"again.")


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
                                pec=lay["pec"], record_frames=True)
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

    plan = fdtd.frame_plan(lay["shape"], est["steps"])
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
                               xp=xp, ade_terms=terms, check=check,
                               frames=plan)
    except TDSolveError:
        raise
    except Exception as exc:  # noqa: BLE001 - only OOM is translated
        if is_out_of_memory(exc):
            raise TDSolveError(oom_message(est["bytes"])) from exc
        raise
    caveats = list(lay["caveats"])
    names = []
    if store_out is not None:
        if stage is not None:
            stage.stage = "writing results"
        names = write_results(store_out, job_id, result, lay, request,
                              caveats, plan)
    return {"job_id": job_id, "kind": "td", "modes": [], "result": result,
            "estimate": est, "caveats": caveats, "files": names}



# ---------------------------------------------------------------------------
# The run in its own process
# ---------------------------------------------------------------------------
# On the Orin the GPU shares system RAM, so a run that outgrows memory is
# often not a MemoryError at all: the kernel's OOM killer (the cgroup's
# MemoryMax, or the whole box) sends SIGKILL, and no except clause anywhere
# sees it. In the worker's own process that killed the worker, the lease
# expired, and the app requeued the job as a counted release -- retried up to
# the cap instead of failed once with the reason. In a child, the killer picks
# the child (it holds the memory), and the parent, still holding the lease,
# reads exit status -9 and fails the job with the OOM sentence.
#
# -9 is read as out-of-memory because nothing else sends it to this child
# alone: a systemd stop's final SIGKILL takes the whole cgroup, parent too, so
# no parent is left to misread it. The unit's ``OOMPolicy=continue`` is what
# keeps systemd from stopping the service when the kernel kills the child.

class _PipeStage:
    """The child's ``stage``: each assignment goes up the pipe, so the
    parent's heartbeat reports what the child is doing."""

    def __init__(self, conn):
        self._conn = conn

    @property
    def stage(self):
        return None

    @stage.setter
    def stage(self, name):
        self._conn.send(("stage", name))


def _chain(exc: BaseException) -> list:
    """``exc`` and its causes, each as (pickle or None, type name, message).

    Pickling an exception drops ``__cause__``/``__context__``, and the
    worker's ``classify`` walks that chain (a GPU-init failure wrapped in
    another error is transient), so every link travels separately."""
    out, seen, node = [], set(), exc
    while node is not None and len(out) < 10 and id(node) not in seen:
        seen.add(id(node))
        try:
            blob = pickle.dumps(node)
            pickle.loads(blob)
        except Exception:  # noqa: BLE001 - unpicklable: name and text travel
            blob = None
        out.append((blob, type(node).__name__, str(node)))
        node = node.__cause__ or node.__context__
    return out


def _rebuild(links: list) -> BaseException:
    """The parent's copy of the child's exception, causes re-linked. A link
    that cannot be unpickled here becomes a RuntimeError subclass of the same
    NAME, which is what ``classify``'s name checks read."""
    nodes = []
    for blob, name, text in links:
        exc = None
        if blob is not None:
            try:
                exc = pickle.loads(blob)
            except Exception:  # noqa: BLE001 - fall back to the name
                exc = None
        if exc is None:
            exc = type(name, (RuntimeError,), {"__module__": __name__})(text)
        nodes.append(exc)
    for exc, cause in zip(nodes, nodes[1:]):
        exc.__cause__ = cause
    return nodes[0]


def _child(conn, job, store_in, store_out, run=None):
    """The child's whole life: run the solve, send one outcome, exit.

    SIGTERM and SIGINT are ignored, as they are in effect for an in-process
    solve (the worker's handler only sets a flag it reads after the solve): a
    systemd stop sends them to the whole cgroup, and a child that died of
    them would turn an operator's restart into a failed job."""
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    run = solve_td if run is None else run
    try:
        out = run(job, store_in, store_out, _PipeStage(conn))
        # The traces and frames are in the files already; only the summary
        # crosses back.
        conn.send(("ok", {k: v for k, v in out.items() if k != "result"}))
    except BaseException as exc:  # noqa: BLE001 - every outcome is reported
        conn.send(("error", _chain(exc)))
    finally:
        conn.close()


def solve_td_isolated(job: dict, store_in, store_out=None, stage=None, *,
                      run=None, context=None, poll_s=1.0,
                      deadline_s=RUN_LIMIT_S + STOP_GRACE_S,
                      setup_s=SETUP_LIMIT_S,
                      reap_s=60.0, clock=time.monotonic) -> dict:
    """``solve_td`` in a spawned child; its outcome, raised or returned here.

    Spawned, never forked: a forked copy of a process that has touched CUDA
    cannot use the GPU. Two clocks, matching the child's own:

    - setup (reading the mesh, laying out the grid) is capped at ``setup_s``
      from spawn, and a child that never reports ``solving`` within it gets
      the plain setup verdict;
    - the run clock restarts when the child reports ``solving``, which is
      when the child's own ``RUN_LIMIT_S`` starts. The child stops itself at
      that limit, but only between steps; one still running ``deadline_s``
      after ``solving`` is stuck inside a step, so it is killed and the job
      gets the time-limit verdict.

    The pipe is always drained before a clock is read, and an outcome found
    in it wins over a deadline, even one that has just killed the child: a
    child that finished must never be failed as too slow. A child that has
    not exited ``reap_s`` after its outcome (or after a kill) is killed, and
    every child is reaped before this returns. ``run`` (a module-level
    function, run in the child instead of ``solve_td``), ``context``,
    ``deadline_s``, ``setup_s``, ``reap_s`` and ``clock`` are seams for the
    tests."""
    ctx = context or multiprocessing.get_context("spawn")
    receive, send = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_child, args=(send, job, store_in, store_out, run),
                       daemon=True)
    proc.start()
    send.close()
    outcome, overdue, solving, started = None, None, False, clock()

    def take(message):
        nonlocal outcome, solving, started
        if message[0] != "stage":
            outcome = message
            return
        if stage is not None:
            stage.stage = message[1]
        if message[1] == "solving" and not solving:
            solving, started = True, clock()

    try:
        while outcome is None:
            if receive.poll(poll_s):
                try:
                    take(receive.recv())
                except EOFError:  # the child is gone without an outcome
                    break
                if outcome is not None or receive.poll(0):
                    continue
            if clock() - started > (deadline_s if solving else setup_s):
                overdue = "run" if solving else "setup"
                proc.kill()
                # An outcome sent just before the kill still wins.
                try:
                    while outcome is None and receive.poll(0):
                        take(receive.recv())
                except EOFError:
                    pass
                break
            if not proc.is_alive() and not receive.poll(0):
                break
    finally:
        receive.close()
        proc.join(reap_s)
        if proc.is_alive():
            proc.kill()
            proc.join()
    if outcome is None:
        if overdue == "setup":
            raise TDSolveError(
                f"The run's setup (reading the object and building the grid) "
                f"took more than {setup_s / 60:.0f} minutes, so it was stopped "
                f"before the solve began. Increase dx, or simplify the object.")
        if overdue == "run":
            raise TDSolveError(
                f"The run was stopped after {RUN_LIMIT_S // 60} minutes, the "
                f"limit for one run. Increase dx or shorten the duration.")
        if proc.exitcode == -signal.SIGKILL:
            raise TDSolveError(oom_message())
        raise TDSolveError(
            f"The time-domain solver stopped unexpectedly (exit status "
            f"{proc.exitcode}) and left no result. Please report this job.")
    if outcome[0] == "error":
        raise _rebuild(outcome[1])
    return outcome[1]


#: How the manifest's spectra are scaled; the viewer labels its axis with it.
SPECTRUM_SCALE = "largest peak = 1"


def normalise_spectra(amps: list, components: list) -> tuple:
    """Scale every probe's in-band spectrum by the largest peak across all
    probes, so the curves compare with one another and do not grow with the
    step count (a raw |rfft| does). Returns ``(spectra, notes)``.

    A probe with no signal in the band (the source cannot excite its
    component there) stays flat zero with a plain note; when no probe has
    any, nothing is divided, so nothing becomes NaN.
    """
    amps = [np.asarray(a, dtype=float) for a in amps]
    peak = max((float(a.max()) for a in amps if a.size), default=0.0)
    notes = [f"probe {p + 1} ({components[p]} component) recorded no signal in "
             f"the band, so its spectrum is flat zero: the source does not "
             f"excite that field component there."
             for p, a in enumerate(amps) if not (a.size and a.max() > 0)]
    if peak > 0:
        amps = [a / peak for a in amps]
    return amps, notes


def spectrum_band(request: dict, band_top_hz: float) -> tuple:
    """The band ``spectrum.json`` keeps (Hz): the pulse's ``f ± bandwidth``,
    otherwise 0 to twice the source frequency, so a cw or drawn source's line
    sits mid-band with room either side."""
    f = float(request["frequency_hz"])
    if request["profile"] == "pulse":
        return max(0.0, f - float(request["bandwidth"])), float(band_top_hz)
    return 0.0, 2.0 * float(band_top_hz)


def _finite(name: str, values) -> None:
    if not np.all(np.isfinite(values)):
        raise TDSolveError(
            f"The run produced values that are not finite numbers in its "
            f"{name}, so it went unstable and there is nothing correct to "
            f"show. Please report this job.")


def frame_positions(lay: dict, stride, shape) -> list:
    """Where each movie sample sits, per axis, in metres (the probes' frame).

    The grid is graded, so the samples are NOT evenly spaced: a viewer needs
    these to place them. A frame keeps every ``stride``-th node from the
    first, so its samples are those nodes; a count that disagrees with the
    frame's shape is refused rather than written as a misplaced movie.
    """
    origin = [float(v) for v in lay["origin_m"]]
    out = []
    for a in range(3):
        kept = np.asarray(lay["nodes"][a], dtype=float)[::int(stride[a])]
        if kept.size != int(shape[a]):
            raise TDSolveError(
                f"the field movie has {int(shape[a])} samples along "
                f"{'xyz'[a]} but stride {int(stride[a])} keeps {kept.size} "
                f"of the grid's {len(lay['nodes'][a])} nodes")
        out.append([origin[a] + float(v) for v in kept])
    return out


def write_results(store_out, job_id: str, result: dict, lay: dict,
                  request: dict, caveats: list, plan: dict | None) -> list:
    """Write ``trace.bin``, the movie frames and, LAST, ``spectrum.json``.

    The manifest goes last because its presence is what says the results
    exist: a run that dies between files leaves no manifest, never a manifest
    pointing at frames that were not written. A non-finite value anywhere is
    refused before anything is written, since JSON has no NaN and a frame of
    NaN would draw as nothing at all. Returns the names written, in order.
    """
    fdtd = _solver()[0]
    key = lambda name: f"jobs/{job_id}/{name}"  # noqa: E731 - the worker's _key
    traces = np.asarray(result["traces"], dtype="<f4")
    frames = result.get("frames")
    dt = float(result["dt"])
    _finite("probe traces", traces)
    if frames is not None:
        frames = np.asarray(frames, dtype="<f4")
        _finite("field movie", frames)

    plan = plan or {}
    positions = (None if frames is None else
                 frame_positions(lay, plan["stride"], frames.shape[1:]))
    f_lo, f_hi = spectrum_band(request, lay["band_top_hz"])
    origin = [float(v) for v in lay["origin_m"]]
    probes, amps = [], []
    for p, probe in enumerate(lay["probes"]):
        cell, component = tuple(int(i) for i in probe[:3]), probe[3]
        freqs, amp = fdtd.spectrum(traces[p], dt)
        keep = (freqs >= f_lo) & (freqs <= f_hi)
        amp = np.asarray(amp)[keep]
        _finite(f"probe {p + 1} spectrum", amp)
        amps.append(amp)
        probes.append({
            "cell": list(cell), "component": component,
            "position_m": [origin[a] + float(lay["nodes"][a][cell[a]])
                           for a in range(3)],
            "spectrum": {"frequency_hz": [float(f) for f in freqs[keep]]}})
    amps, notes = normalise_spectra(amps, [p["component"] for p in probes])
    for probe, amp in zip(probes, amps):
        probe["spectrum"]["amplitude"] = [float(v) for v in amp]
    names = []
    store_out.write(key(TRACE_NAME), np.ascontiguousarray(traces).tobytes())
    names.append(TRACE_NAME)
    n_frames = 0 if frames is None else int(frames.shape[0])
    for n in range(n_frames):
        name = FRAME_NAME.format(n + 1)
        store_out.write(key(name), np.ascontiguousarray(frames[n]).tobytes())
        names.append(name)

    manifest = {
        "version": SPECTRUM_VERSION, "dt_s": dt, "steps": int(result["steps"]),
        "band_hz": [f_lo, f_hi], "probes": probes,
        "trace": {"name": TRACE_NAME, "dtype": "float32", "order": "C",
                  "shape": list(traces.shape)},
        "frames": None if frames is None else {
            "count": n_frames, "dtype": "float32", "order": "C",
            "axes": "xyz", "shape": list(frames.shape[1:]),
            "steps": [int(k) for k in plan.get("steps", [])],
            "stride": [int(k) for k in plan.get("stride", [])],
            "positions_m": positions,
            "quantity": "|E|"},
        "spectrum_scale": SPECTRUM_SCALE,
        "caveats": list(caveats) + notes}
    store_out.write(key(SPECTRUM_NAME),
                    json.dumps(manifest, allow_nan=False).encode())
    names.append(SPECTRUM_NAME)
    return names
