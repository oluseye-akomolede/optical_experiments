#!/usr/bin/env python3
"""Run ONE real Jetson job end to end, with no cluster, no MinIO and no Postgres.

WHY THIS EXISTS. On deploy 3 every Jetson job failed AFTER a successful solve --
20 modes written, mode 1 at the right frequency -- and the worker's journal held
only the generic owner sentence, because the real exception was swallowed. The
app's log held nothing either. Neither end could name the failure, so the
deploy rolled back with no diagnosis. This script is the missing instrument: it
drives the REAL app and the REAL worker against each other, with the REAL solve,
so the traceback the worker now logs has somewhere to appear.

WHAT IT IS NOT. It is not a replacement for the unit suite or for
``tests/pg_race_check.py``. Those pin the SQL against real Postgres; this pins
the WIRE and the worker's job lifecycle, which is where the deploy-3 failure
lived. It is deliberately NOT part of ``tests/`` -- it imports the solver, which
needs a GPU on the Orin, and a suite that cannot run everywhere gets skipped
everywhere.

HOW TO RUN (on the Orin, in the worker venv, from the repo root):

    MSE_SOLVER_BACKEND=gpu python3 scripts/jetson_worker_e2e.py

Exit status is 0 only when the job reaches DONE. On any failure it prints the
worker's own traceback and exits non-zero, so it is usable as a smoke gate.

THE THREE REAL PIECES, and nothing is doubled that the deploy had real:
  * the WEB APP is built by ``web/app.py:create_app`` with in-memory doubles for
    storage and the job store -- the same doubles ``tests/test_web_routes.py``
    uses, because those are what a unit test can trust and what keeps this free
    of Postgres and MinIO;
  * the HTTP is a real ``werkzeug`` server on 127.0.0.1, so the worker's real
    ``urllib`` client, its real headers, its real multipart-free PUT bodies and
    its real error handling all run;
  * the WORKER is the real ``Worker.run_job`` with the real ``real_solve``, so
    the mesher, the solver and the result writer are the production ones.
"""

from __future__ import annotations

import argparse
import inspect
import math
import os
import socket
import sys
import tempfile
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _free_port() -> int:
    """A port nothing is listening on, so two runs in a row cannot collide."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _load_harness(port: int, work_root: Path, token: str):
    """Build the real app, the real worker and the real solver.

    Imported inside the function so ``--help`` works with no GPU and no venv
    setup, and so the import error (if any) is reported by ``main``'s handler
    rather than at module import, where it would look like a broken script.
    """
    sys.path.insert(0, str(ROOT / "web"))
    sys.path.insert(0, str(ROOT / "tests"))
    sys.path.insert(0, str(ROOT / "worker"))

    import app as APP                       # noqa: PLC0415
    import mse_worker as WORKER             # noqa: PLC0415
    from test_web_routes import (           # noqa: PLC0415
        FakeDb, FakeLauncher, MemoryStore,
    )

    store = MemoryStore(work_root)
    jobs_db = FakeDb()
    app, _store, _db, _launcher = APP.create_app({
        "TESTING": False,
        "MSE_LOCAL_STORE": str(work_root),
        "MSE_DB_MOCK": jobs_db,
        "MSE_JOB_LAUNCHER": FakeLauncher(),
    })
    return APP, WORKER, app, store, jobs_db


def _enqueue_one_job(jobs_db, store, work_root: Path, n_modes: int) -> str:
    """Put one small mm-scale box job on the GPU queue, and its input in store.

    The box is the SAME 10x7x5 the unit tests use, so a failure here is
    comparable to a passing unit test rather than to a different object.
    """
    import trimesh  # noqa: PLC0415

    job_id = "e2e00000-0000-0000-0000-000000000001"
    jobs_db.create_job(job_id, "box.obj", "mm", 1.0, n_modes,
                       target="jetson", est_tets=20_000)

    import mse_storage as STORE  # noqa: PLC0415
    obj = trimesh.creation.box(extents=[10.0, 7.0, 5.0]).export(file_type="obj")
    # ``export`` returns ``str`` for obj and the store wants bytes, which is
    # what the app's own upload route stores -- the request body.
    if isinstance(obj, str):
        obj = obj.encode("utf-8")
    # Through the SAME store object the app was built with, so the key shape is
    # the app's own. Writing the file by hand would work today and would stop
    # testing the app's path the day the key scheme changes.
    store.write(STORE.job_key(job_id, "input.obj"), obj)
    return job_id


def _check_the_harness_models_the_real_store(work_root: Path) -> None:
    """Prove the reader and the writer here agree on WHERE a result file is.

    WHY THIS EXISTS. The first version of this harness wrote its stub results
    through a bucket-shaped store (``<root>/jobs/<id>/x``) while the worker
    handed ``run_job`` the per-job store the production wiring builds
    (``<root>/<id>/x``). Each half was defensible and together they agreed on a
    layout NEITHER of them uses on the Orin: the harness reported PASS while
    production failed at upload time on every job. That is the same defect this
    harness was written to diagnose, one level up.

    It is an assertion rather than a comment because a fake that is wrong in the
    same direction as the code cannot fail, and the only way to know is to make
    the two halves state their answer out loud. A mismatch here aborts the run
    (exit 2, COULD-NOT-RUN) instead of producing a green result nobody can trust.
    """
    import mse_storage as STORE  # noqa: PLC0415

    probe = "_e2e_layout_probe"
    #: The probe filename must be one the reader ACCEPTS. An arbitrary name
    #: takes the "the solver wrote X which the API does not accept" branch of
    #: ``_result_names``, so the probe would report a layout failure that is
    #: really an allow-list refusal -- an instrument bug, and this harness has
    #: already had one of those.
    probe_name = "modes.json"
    problem = None
    for layout, store in (
        ("bucket root", STORE.LocalStore(work_root)),
        ("job work dir", STORE.LocalStore(work_root / "job")),
    ):
        key = f"jobs/{probe}/{probe_name}"
        try:
            where = Path(store._path(key))      # what the WRITER would touch
            store.write(key, b"{}")
        except Exception as exc:                # noqa: BLE001
            problem = f"{layout}: the store refused to write {key!r}: {exc!r}"
            break
        found = _e2e_sibling_names(store, probe)
        where.unlink(missing_ok=True)
        print(f"[e2e] store layout '{layout}': key {key!r} -> {where}"
              f" (reader sees {found})")
        if probe_name not in found:
            problem = (f"{layout}: wrote {key!r} to {where} and the reader did "
                       f"NOT find it -- the two halves disagree on where a "
                       f"result file lives")
            break
    probe_dir = work_root / "job" / "jobs" / probe
    if probe_dir.exists():
        import shutil  # noqa: PLC0415
        shutil.rmtree(probe_dir, ignore_errors=True)
    if problem:
        raise RuntimeError(
            "the harness cannot model the real result location, so a PASS "
            "here would prove nothing. " + problem
        )


def _e2e_sibling_names(store, job_id: str) -> list[str]:
    """The names the WORKER's reader finds, using the worker's own function.

    Called as the deploy does, through ``_result_names``, so a harness that
    hand-rolls its own listing cannot agree with itself and disagree with
    production -- which is exactly what happened.
    """
    import mse_worker as WORKER  # noqa: PLC0415

    try:
        return WORKER._result_names(store, job_id)
    except Exception:                            # noqa: BLE001
        return []


def _check_the_backend_is_the_one_that_was_asked_for(worker_mod, wanted: str) -> None:
    """The environment must agree with what this harness says it chose.

    The first run of this harness printed ``backend 'cpu'`` and then failed the
    job with ``MSE_SOLVER_BACKEND=gpu needs the CUDA stack``: a backend already
    exported in the shell had been set with ``setdefault``, so the harness's own
    decision was discarded, its report was still true, and nothing in the output
    distinguished that from a genuine failure.
    """
    actual = os.environ.get("MSE_SOLVER_BACKEND")
    if actual != wanted:
        raise RuntimeError(
            f"asked for MSE_SOLVER_BACKEND={wanted!r} and the environment holds "
            f"{actual!r}; the run would not have exercised what was requested"
        )


#: The keyword ``real_solve`` passes to the solver. Read out of the source, not
#: retyped, so this guard cannot agree with a stale copy of its own expectation
#: (a test restating a value checks its own arithmetic).
_REAL_SOLVE_BACKEND_PATTERN = r"backend\s*=\s*[\"'](\w+)[\"']"


def _real_solve_backends(worker_mod) -> list[str]:
    """Every ``backend=`` literal in ``real_solve``'s ``SOLVER.solve`` call."""
    import re  # noqa: PLC0415
    src = inspect.getsource(worker_mod.real_solve)
    # Only the call that crosses into the solver, not the whole function body.
    call = src[src.index("SOLVER.solve("):] if "SOLVER.solve(" in src else src
    return re.findall(_REAL_SOLVE_BACKEND_PATTERN, call)


def _check_the_worker_honours_the_backend(worker_mod, backend: str) -> None:
    """The worker must not be able to pick a backend the environment forbade.

    THIS IS WHY THE HARNESS'S FIRST TWO RUNS WERE MISREAD. They printed
    ``backend 'cpu'``, went to the GPU path anyway, and failed with
    ``MSE_SOLVER_BACKEND=gpu needs the CUDA stack``: ``real_solve`` passes a
    HARDCODED ``backend="gpu"`` to ``SOLVER.solve``, which ``mse_maxwell_solver``
    forwards into ``mse_eigen``, where ``factor(..., backend=backend)`` uses it
    as ``backend or selected_backend()`` -- so the environment variable never
    reaches the solve at all. The harness had therefore been running an
    experiment nobody chose, which is the same defect it exists to diagnose, one
    level up.

    It is an assertion about the WORKER's own source, so it fires identically on
    the Orin: if the Orin's solve picks a backend other than the one this run
    asked for, that is a finding, and it must not be reported as a solver fault
    or as a passing run. Deliberately NOT fixed here -- the deploy ordering is
    the lead's call, and a soft change to the backend is exactly the kind of
    "fix" that must not precede a traceback.
    """
    literal = _real_solve_backends(worker_mod)
    if backend not in literal:
        raise RuntimeError(
            f"this run asked for backend {backend!r}, but the worker's "
            f"real_solve passes backend={literal} to the solver, so the "
            f"environment cannot select it. That is a defect in the worker's "
            f"backend selection, not in this harness; the run would NOT have "
            f"tested the path it reported. Investigate before reporting a "
            f"result from it."
        )


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--port", type=int, default=0,
                    help="port to serve on (default: a free one)")
    ap.add_argument("--n-modes", type=int, default=20,
                    help="modes to solve (the deploy's job used 20)")
    ap.add_argument("--keep", action="store_true",
                    help="keep the work directory instead of deleting it")
    ap.add_argument("--stub-solve", action="store_true",
                    help="replace ONLY the GPU arithmetic with a stub that "
                         "writes the same result files through the worker's "
                         "own store layout; the worker, the wire and the "
                         "uploads stay real. For running this off the Orin")
    ap.add_argument("--backend", default=None,
                    help="MSE_SOLVER_BACKEND (default: gpu where the CUDA "
                         "stack is importable, else cpu). A missing CUDA stack "
                         "would otherwise fail the job for a reason that is "
                         "true but useless, because the point here is the WORKER")
    args = ap.parse_args(argv)

    port = args.port or _free_port()
    work_root = Path(tempfile.mkdtemp(prefix="mse-e2e-"))
    # A fixed, obviously-fake token. This is a LOCAL harness: the server and the
    # client are the same process tree and nothing leaves 127.0.0.1. No real
    # token is read, generated or logged here.
    token = "e2e-not-a-real-token"
    url = f"http://127.0.0.1:{port}"
    os.environ["MSE_WORKER_TOKEN"] = token
    # The GPU backend is what deploy 3 ran and what the Orin should exercise by
    # default, but on a box without the CUDA stack it fails the job for a true-
    # but-useless reason and buries the thing this harness is about. So the
    # backend follows what the environment can actually import, and --backend
    # overrides it. A job that fails for a missing GPU stack is still printed
    # with its traceback, which is the harness doing its job.
    #
    # The chosen value is ASSERTED below rather than trusted: a stray
    # MSE_SOLVER_BACKEND in the environment silently overrode the choice the
    # first time this ran, the harness told the truth and the reader could not
    # tell it from a real failure.
    backend = args.backend
    if backend is None:
        try:
            import cupy  # noqa: F401,PLC0415
            backend = "gpu"
        except Exception:  # noqa: BLE001
            backend = "cpu"
    os.environ["MSE_SOLVER_BACKEND"] = backend
    print(f"[e2e] solver backend {backend!r}")

    print(f"[e2e] work root {work_root}")
    print(f"[e2e] serving {url}")

    backend = args.backend
    explicit = backend is not None
    if backend is None:
        try:
            import cupy  # noqa: F401,PLC0415
            backend = "gpu"
        except Exception:  # noqa: BLE001
            backend = "cpu"
    os.environ["MSE_SOLVER_BACKEND"] = backend
    print(f"[e2e] solver backend {backend!r}"
          f"{'' if explicit else ' (chosen from what this box can import)'}")

    APP, WORKER, app, store, jobs_db = _load_harness(port, work_root, token)
    print("[e2e] verifying the reader and the writer agree on the result layout")
    _check_the_harness_models_the_real_store(work_root)
    _check_the_backend_is_the_one_that_was_asked_for(WORKER, backend)
    if not args.stub_solve:
        # Only meaningful for the real solve: --stub-solve replaces it wholesale,
        # so the worker's own backend choice is not exercised at all.
        _check_the_worker_honours_the_backend(WORKER, backend)
    job_id = _enqueue_one_job(jobs_db, store, work_root, args.n_modes)
    print(f"[e2e] enqueued {job_id}: {args.n_modes} modes, units mm, 10x7x5 box")

    from werkzeug.serving import make_server  # noqa: PLC0415

    server = make_server("127.0.0.1", port, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # Wait for the socket to actually accept, rather than sleeping a guessed
    # interval: a bind race here reads as "the app refused the claim", which is
    # the wrong diagnosis and exactly the kind this rig keeps making.
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                break
        except OSError:
            time.sleep(0.05)
    else:
        print("[e2e] FAILED: the server never accepted a connection", flush=True)
        return 2

    config = WORKER.Config(url, token, work_root=work_root,
                           lock_path=work_root / ".gpu.lock")
    api = WORKER.WorkerAPI(url, token)

    # The worker's own traceback sink is wired to stdout, which is what makes
    # the deploy-3 failure visible here: ``_fail`` now calls it on every job
    # failure, so the REAL exception type and traceback are printed instead of
    # the generic owner sentence alone.
    lines: list[str] = []

    def sink(message: str, flush: bool = False) -> None:  # noqa: ARG001
        lines.append(message)
        print(message, flush=True)

    worker = WORKER.Worker(
        config, api, WORKER.BenchLock(config.lock_path),
        solve=WORKER.real_solve,
        store_factory=WORKER.store_factory_for(WORKER._SOLVER_DIR),
        log=sink,
    )

    if args.stub_solve:
        # A LOCAL substitute for the GPU solve, so this harness is runnable
        # where there is no GPU (the builder's box) as well as on the Orin. It
        # drives the SAME worker, the SAME wire and the SAME upload step; it
        # only replaces the arithmetic. The layout it writes through is the
        # worker's own per-job store, which is what the assertion above pins.
        import trimesh  # noqa: PLC0415

        def stub(job, store_in, store_out, stage=None):  # noqa: ARG001
            import mse_storage as STORE  # noqa: PLC0415
            jid = job["job_id"]
            # Real STL bytes, not a placeholder: the app validates the artifact
            # it is sent, and a stub that uploads b"glb" would prove only that
            # the wire carries bytes.
            mesh = trimesh.creation.box(extents=[10.0, 7.0, 5.0])
            store_out.write(STORE.job_key(jid, "surface.glb"), mesh.export(
                file_type="glb"))
            modes = {}
            for n in range(1, int(job.get("n_modes") or 1) + 1):
                name = f"mode_{n:02d}"
                store_out.write(STORE.job_key(jid, name + ".bin"),
                                (n.to_bytes(2, "little") * 4))
                store_out.write(STORE.job_key(jid, name + ".vtu"),
                                b"<?xml version='1.0'?><VTKFile/>")
                modes[f"{n:02d}"] = {"freq_ghz": n}
            import json  # noqa: PLC0415
            store_out.write(STORE.job_key(jid, "modes.json"),
                            json.dumps(modes).encode())
            return {"n_modes": len(modes)}

        worker._solve = stub
        print("[e2e] --stub-solve: the GPU arithmetic is replaced; the worker, "
              "the wire and the uploads are the real ones")

    job = api.claim()
    if job is None:
        print("[e2e] FAILED: the worker could not claim the job it just "
              "enqueued -- the wire, not the solver, is at fault", flush=True)
        server.shutdown()
        return 2
    print(f"[e2e] claimed {job['job_id']}: max_tets={job.get('max_tets')} "
          f"cpu_max_tets={job.get('cpu_max_tets')}")

    outcome = "unknown"
    traceback_text = None
    try:
        outcome = worker.run_job(job)
    except BaseException:                       # noqa: BLE001 - that is the point
        traceback_text = traceback.format_exc()
    finally:
        server.shutdown()

    row = jobs_db.jobs.get(job_id, {})
    print("\n[e2e] ------------------ outcome ------------------")
    print(f"[e2e] worker returned: {outcome}")
    print(f"[e2e] row status: {row.get('status')!r} stage: {row.get('stage')!r}")
    print(f"[e2e] row message: {row.get('message')!r}")

    files = sorted(p.name for p in (work_root / "jobs" / job_id).iterdir()
                   ) if (work_root / "jobs" / job_id).is_dir() else []
    print(f"[e2e] result files on disk ({len(files)}): {files[:6]}"
          f"{' ...' if len(files) > 6 else ''}")

    if traceback_text:
        print("[e2e] the worker raised out of run_job:\n" + traceback_text)

    # The worker's own journal lines, filtered to the ones that carry a failure.
    # Printed whether or not the job succeeded: on a PASS this shows the
    # observability lines were exercised, and its ABSENCE on a FAIL is itself
    # the finding.
    failure_lines = [ln for ln in lines
                     if "failed" in ln or "Traceback" in ln
                     or "Error" in ln or "refused a call" in ln]
    if failure_lines:
        print("[e2e] the worker's own failure lines:")
        for ln in failure_lines:
            print("   " + ln)

    ok = outcome == "done" and row.get("status") == "done"

    # The mode rows, which is the whole point of the deploy-3 fix and which the
    # status check above cannot see: a DONE with an empty mode list passed this
    # harness at e824780. The rows are read through the APP's own job list, so a
    # store written without the writer's own convention shows up here.
    modes_ok = False
    try:
        stored = jobs_db.list_modes(job_id)
    except Exception as exc:                     # noqa: BLE001 - report, do not hide
        stored = None
        print(f"[e2e] could not read the mode rows: {type(exc).__name__}: {exc}")
    if stored is not None:
        n_modes = int(row.get("n_modes") or 0)
        indices = [m.get("index") for m in stored]
        finite = all(
            isinstance(m.get(f), (int, float)) and math.isfinite(m.get(f))
            for m in stored
            for f in ("k0_per_m", "freq_hz", "vacuum_wavelength_m")
        ) if stored else False
        print(f"[e2e] mode rows in mse.modes: {len(stored)} (job says "
              f"n_modes={n_modes}); indices={indices[:8]}"
              f"{' ...' if len(indices) > 8 else ''}")
        modes_ok = (
            stored
            and len(stored) == n_modes
            and indices == list(range(1, n_modes + 1))
            and finite
        )
        if stored and not modes_ok:
            print("[e2e] the rows are present but the wrong shape or count -- a "
                  "job page would show a mode list that disagrees with the job")

    ok = ok and modes_ok
    if outcome == "done" and row.get("status") == "done" and not modes_ok:
        print("\n[e2e] the job reached DONE with no usable mode rows. That is "
              "the deploy-3 shape: 'finished' with nothing to show for it, "
              "which reads as a job that HAS no modes rather than one that "
              "never reported them.")

    if not args.keep:
        import shutil  # noqa: PLC0415
        shutil.rmtree(work_root, ignore_errors=True)
    else:
        print(f"[e2e] kept {work_root}")

    print(f"\n[e2e] RESULT: {'PASS' if ok else 'FAIL'}")
    if not ok and not failure_lines:
        print("[e2e] NOTE: the job failed and the worker logged NO failure "
              "line -- the observability change did not fire, which is itself "
              "a defect worth reporting before the traceback is chased.")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                            # noqa: BLE001 - this is the point
        print("COULD-NOT-RUN: the harness itself failed:\n"
              + traceback.format_exc())
        sys.exit(2)
