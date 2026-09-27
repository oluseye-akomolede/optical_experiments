"""The MSE web app: upload an .obj, watch the solver, view the modes.

Owner-facing decisions this implements: .obj upload, a units menu (mm default),
a refractive index field (default 1.0), the 20 lowest optical modes, and a
viewer that shows the object with coloured field slices through it. Tailnet
only, no login.

Concurrency is one. An upload is recorded as ``queued`` and a background tick
launches the oldest queued job whenever no solver Job is occupying the single
namespace slot; a finished Job is deleted as soon as its result is in the
database, because the quota counts finished Jobs too.

Result files are proxied from MinIO -- the bucket stays private and nothing is
presigned.
"""

from __future__ import annotations

import os
import threading
import time
import uuid
from datetime import datetime, timezone

from flask import (Flask, abort, flash, jsonify, redirect, render_template,
                   request, send_file, url_for)

import config as CFG
import db as DB
import fields as FLD
import jobs as JOBS
import storage as STORE
from meshing import UploadError, validate_upload

#: how many slices of the field volume to send for the 3-D point cloud
CLOUD_SAMPLES = 24

#: where the solver puts one mode's field, and the ONE place that decides it.
#: Both the index (which reads a size to decide `available`) and the fetch route
#: (which reads the bytes) go through this, so they cannot drift apart and leave
#: the list promising a mode the fetcher cannot serve.
FIELD_NAME = "mode_{:02d}.bin"


def create_app(config: dict | None = None) -> Flask:
    """Build the app. Tests pass overrides here instead of touching env."""
    app = Flask(__name__)
    app.secret_key = os.environ.get("FLASK_SECRET_KEY") or os.urandom(32)
    app.config.update(
        MAX_CONTENT_LENGTH=CFG.MAX_UPLOAD_BYTES,
        DRY_RUN=CFG.DRY_RUN,
    )
    if config:
        app.config.update(config)

    store = _store(app)
    jobs_db = _job_store(app)
    launcher = _launcher(app)

    app.extensions["mse"] = {"store": store, "db": jobs_db, "jobs": launcher}

    # -- pages -------------------------------------------------------------
    @app.route("/healthz")
    def healthz():
        return jsonify({"status": "ok"})

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            return _handle_upload(app, store, jobs_db)
        return render_template("index.html", jobs=jobs_db.list_jobs(25),
                              units=CFG.UNIT_CHOICES, default_units=CFG.DEFAULT_UNITS,
                              max_mb=CFG.MAX_UPLOAD_BYTES // (1024 * 1024))

    @app.route("/jobs/<job_id>")
    def job_page(job_id):
        job = _require_job(jobs_db, job_id)
        return render_template("job.html", job=job,
                              queue_position=jobs_db.queue_position(job_id))

    # -- JSON the page polls ----------------------------------------------
    @app.route("/jobs/<job_id>/status")
    def job_status(job_id):
        job = _require_job(jobs_db, job_id)
        payload = {
            "job_id": job["job_id"],
            "status": job["status"],
            "stage": job["stage"],
            "message": job["message"],
            "created_at": job["created_at"],
            "finished_at": job["finished_at"],
            "peak_rss_mb": job["peak_rss_mb"],
            "queue_position": jobs_db.queue_position(job_id),
        }
        if job["status"] == DB.DONE:
            payload["modes"] = jobs_db.list_modes(job_id)
            payload["manifest_url"] = url_for("job_file", job_id=job_id,
                                              name="modes.json")
            payload["surface_url"] = url_for("job_file", job_id=job_id,
                                             name="surface.glb")
        return jsonify(payload)

    # -- proxied result files ---------------------------------------------
    @app.route("/jobs/<job_id>/files/<path:name>")
    def job_file(job_id, name):
        """Serve one result file. Only the files the solver writes are allowed."""
        if not _safe_result_name(name):
            abort(404)
        job = _require_job(jobs_db, job_id)
        if job["status"] != DB.DONE and not name.endswith("modes.json"):
            abort(404)
        try:
            data = store.read(STORE.job_key(job_id, name))
        except FileNotFoundError:
            abort(404)
        except STORE.StorageError:
            abort(502)
        return send_file(_bytes_io(data), mimetype=_mimetype(name),
                         download_name=name, conditional=True)

    # -- derived payloads the viewer needs --------------------------------
    #
    # The payload is split so a phone downloads one mode's field, not twenty.
    # ``slices.json`` is now the *index*: one entry per mode with its frequency
    # and its index, no field data. The heavy arrays live in
    # ``slices/<n>.json`` and are fetched when that mode is first selected. A
    # job whose modes are all unreadable is still a payload, not an error --
    # the index is what the mode table renders from.
    @app.route("/jobs/<job_id>/slices.json")
    def job_slices(job_id):
        """The mode index: what each mode is, without any field data."""
        manifest = _done_manifest(store, jobs_db, job_id)
        return jsonify(_mode_index(manifest, store, job_id))

    @app.route("/jobs/<job_id>/slices/<int:number>.json")
    def job_slices_for_mode(job_id, number):
        """One mode's slices and point cloud, decoded from its own field file."""
        manifest = _done_manifest(store, jobs_db, job_id)
        mode = next((m for m in manifest.get("modes", [])
                     if int(m["index"]) == number), None)
        if mode is None:
            abort(404)
        try:
            field = FLD.decode_mode(
                store.read(STORE.job_key(job_id, FIELD_NAME.format(number))),
                manifest["grid"])
        except (FileNotFoundError, FLD.FieldError):
            # The mode is in the table but its field file is gone or unusable.
            # That is the same shape of failure as an unknown mode number, so it
            # answers the same way rather than pretending a field exists.
            abort(404)
        return jsonify({
            "grid": manifest["grid"],
            "index": number,
            "slices": [FLD.slice_payload(field, manifest["grid"], axis, pos)
                       for axis, pos in _slice_positions(manifest["grid"])],
            "cloud": FLD.downsample(field, manifest["grid"], CLOUD_SAMPLES),
        })

    return app, store, jobs_db, launcher


def run_tick(store, jobs_db, launcher, log=print) -> dict:
    """One pass of the scheduler: reclaim dead Jobs, then launch at most one.

    Safe to call concurrently and safe to call when nothing is queued. Returns a
    small summary so the tests (and the log) can see what it decided.
    """
    result = {"deleted": [], "failed": 0, "launched": None, "waited": False}

    # A Job whose result is already in the database is holding the namespace's
    # single quota slot for nothing -- the quota counts finished Jobs too. Delete
    # it BEFORE asking what is active, or the slot never frees and the queue
    # deadlocks after the first upload. The ttlSecondsAfterFinished on the Job is
    # only a backstop for the case where the app is not running to do this.
    if launcher is not None:
        for name in _finished_job_names(launcher, jobs_db):
            try:
                launcher.delete(name)
                result["deleted"].append(name)
                log(f"[mse-web] deleted finished job {name}")
            except Exception as exc:  # noqa: BLE001 - never let a tick crash
                log(f"[mse-web] could not delete finished job {name}: {exc}")

    namespace_active = _active_jobs(launcher)
    if launcher is not None and getattr(launcher, "dry_run", False):
        # No cluster to ask, so the database is the only view of what is taken.
        # Without this a second tick would launch the same job twice.
        namespace_active = namespace_active or (
            ["<running>"] * jobs_db.count_running())

    # Reconcile rows the app thinks are running against what the cluster
    # actually has. A row can only be `running` because the app set it when it
    # launched the Job, so a row with no Job behind it did not finish -- it was
    # deleted, evicted, or lost with its node. Failing it is the honest outcome:
    # re-queueing (what this used to do) re-runs a solve that already died, so
    # an object that OOMs would OOM forever and the owner would watch a spinner
    # that never ends. Kubernetes' own reasons are mapped to plain English.
    result["failed"] = _fail_dead_jobs(launcher, jobs_db, log)

    # `namespace_active` is already the concurrency gate: the namespace admits
    # exactly one Job, including finished ones, so any Job on the list means the
    # slot is taken.
    if namespace_active:
        result["waited"] = True
        return result

    job = jobs_db.next_queued()
    if not job:
        return result

    try:
        name = launcher.create(job["job_id"], job["units"], job["n_index"],
                               job["n_modes"])
    except JOBS.QuotaExceeded:
        # The quota counts finished Jobs, so a 403 here is "wait", not "fail".
        result["waited"] = True
        log("[mse-web] namespace quota is full; leaving the job queued")
        return result
    except JOBS.JobLaunchError as exc:
        jobs_db.mark_failed(job["job_id"], str(exc))
        log(f"[mse-web] could not start job {job['job_id']}: {exc}")
        return result

    jobs_db.mark_running(job["job_id"], name)
    result["launched"] = name
    log(f"[mse-web] started {name} for job {job['job_id']}")
    return result


def _fail_dead_jobs(launcher, jobs_db, log=print) -> int:
    """Mark rows whose Job has died as failed. Returns how many.

    Three states matter, and they are decided by crossing the database against
    the cluster:

    * **Job present, no failure** -- still running. Never touched: the solve is
      live and deleting or failing it would throw the work away.
    * **Job present and FAILED** -- the solve died (deadline, OOM, error). Its
      reason names the cause for the owner.
    * **Job absent** -- it was deleted, evicted, or lost with its node, so the
      row would otherwise stay ``running`` forever. Nothing will ever finish it.

    A launcher that cannot report (no cluster, dry run, a failed list call)
    makes this return 0 rather than guess: failing a live solve on a transient
    API error would be much worse than a spinner that lasts one more tick.
    """
    if launcher is None or not hasattr(launcher, "job_states"):
        # A launcher that cannot report Job conditions cannot tell a dead solve
        # from a live one; leave the rows for a launcher that can.
        return 0

    # Each call is made on its own, and each can fail alone. "Could not ask" and
    # "there is nothing there" are different answers, so a raise from either one
    # ends the reconcile for this tick -- failing a live solve on an answer we
    # never got is much worse than a spinner that lasts one more tick.
    #
    # Both calls must succeed, not just one. They are separate requests to the
    # cluster and either can fail first; a launcher that raised from
    # ``active_job_names`` while ``job_states`` answered an empty dict is a shape
    # that a placeholder cannot rescue, because an empty dict reads as "no Job
    # here has failed" and every running row then falls through to the ''else''.
    # That shape used to be unreachable by luck -- both calls hitting the same
    # 403 -- and this makes it unreachable by construction.
    try:
        names = launcher.active_job_names()
    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop
        log(f"[mse-web] could not list solver jobs: {exc}")
        return 0
    try:
        states = launcher.job_states()
    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop
        log(f"[mse-web] could not read job states: {exc}")
        return 0
    present = set(names)

    # There is deliberately no "present is empty, so fail nothing" guard here.
    # It looks safer and is not: a Job that was deleted by hand (or lost with its
    # node) leaves an empty list too, and that is precisely the case BUG B exists
    # to fail -- the guard would have restored the permanent spinner the fix
    # removed. Both list calls raise on an unreadable cluster, so an empty list
    # here can only mean a successful look that found nothing.

    # In dry-run there is no cluster, so `present` is empty and every running
    # row would look dead. The in-memory launcher's own state is the only view
    # available, and it reports nothing as failed -- so do nothing.
    if getattr(launcher, "dry_run", False):
        return 0

    failed = 0
    for job in jobs_db.list_jobs(50):
        if job["status"] != DB.RUNNING:
            continue
        name = job.get("k8s_job_name")
        if name and name in states:
            message = JOBS.failure_message(states[name])
        elif name and name in present:
            continue                       # alive and well; not ours to touch
        else:
            message = JOBS.failure_message(None)
        jobs_db.mark_failed(job["job_id"], message)
        failed += 1
        log(f"[mse-web] job {job['job_id']} failed: its Job "
            f"{name or '<unnamed>'} is {('reported FAILED' if name in states else 'gone')}")
    return failed


def _finished_job_names(launcher, jobs_db) -> list[str]:
    """Solver Jobs that are still on the cluster but whose row is terminal.

    Only names the database already recorded are returned, so a Job this app
    did not launch is never touched. ``mark_running`` stores the name, so a
    queued row with no name is skipped. ``active_job_names`` is the authority on
    what is really there: if the Job is already gone (TTL reaped it, or a human
    deleted it) it will not be in the list and there is nothing to do.
    """
    try:
        present = set(_active_jobs(launcher))
    except Exception:  # noqa: BLE001 - listing failed; do not delete blind
        return []
    if not present:
        return []

    names = []
    for job in jobs_db.list_jobs(50):
        if job["status"] not in (DB.DONE, DB.FAILED):
            continue
        name = job.get("k8s_job_name")
        if name and name in present:
            names.append(name)
    return names


def _active_jobs(launcher) -> list[str]:
    """Jobs occupying a slot, or ``["<unknown>"]`` when the cluster cannot be asked.

    Those two must not collapse into one answer. An empty list means "nothing is
    running"; "we could not ask" is not that, and neither is it a licence to
    launch. The launcher raises ``CannotAskCluster`` on a 403 instead of
    returning empty; catching it here and handing back a placeholder keeps every
    caller's truthiness test correct -- the concurrency gate stays shut, and
    nothing downstream mistakes the placeholder for a Job it may delete or report
    on. A caller that needs to know *why* it cannot ask should call the launcher
    directly, as ``_fail_dead_jobs`` does, and let the exception through.
    """
    if launcher is None:
        return []
    try:
        return launcher.active_job_names()
    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop
        print(f"[mse-web] could not list solver jobs: {exc}", flush=True)
        return ["<unknown>"]


def _handle_upload(app, store, jobs_db):
    """Validate, store and queue one upload."""
    uploaded = request.files.get("meshfile")
    if uploaded is None or not uploaded.filename:
        flash("Please choose a file to upload.")
        return redirect(url_for("index"))

    units = request.form.get("units", CFG.DEFAULT_UNITS)
    if units not in CFG.UNIT_CHOICES:
        flash("Please choose one of the supported units.")
        return redirect(url_for("index"))

    try:
        n_index = float(request.form.get("n_index", "1.0"))
    except ValueError:
        flash("The refractive index must be a number.")
        return redirect(url_for("index"))
    if not (n_index >= 1.0) or n_index != n_index or n_index == float("inf"):
        flash("The refractive index must be 1.0 or greater.")
        return redirect(url_for("index"))

    data = uploaded.read()
    try:
        validate_upload(uploaded.filename, data, CFG.MAX_UPLOAD_BYTES)
    except UploadError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    job_id = uuid.uuid4()
    try:
        store.write(STORE.job_key(job_id, "input.obj"), data)
    except STORE.StorageError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    jobs_db.create_job(job_id, uploaded.filename, units, n_index,
                       CFG.DEFAULT_N_MODES)

    # Try to start it right away; if the slot is busy it simply stays queued and
    # the tick will pick it up.
    try:
        run_tick(store, jobs_db, app.extensions["mse"]["jobs"])
    except Exception as exc:  # noqa: BLE001 - the tick is best-effort
        app.logger.warning("scheduler tick failed after upload: %s", exc)

    return redirect(url_for("job_page", job_id=job_id))


def _require_job(jobs_db, job_id):
    try:
        uuid.UUID(str(job_id))
    except ValueError:
        abort(404)
    job = jobs_db.get_job(job_id)
    if job is None:
        abort(404)
    return job


def _safe_result_name(name: str) -> bool:
    """Only the filenames the solver actually writes may be proxied."""
    import re

    return bool(re.fullmatch(r"modes\.json|surface\.glb|mode_\d{2}\.bin|"
                             r"mode_\d{2}\.vtu", name))


def _mimetype(name: str) -> str:
    if name.endswith(".json"):
        return "application/json"
    if name.endswith(".glb"):
        return "model/gltf-binary"
    if name.endswith(".vtu"):
        return "application/xml"
    return "application/octet-stream"


def _done_manifest(store, jobs_db, job_id) -> dict:
    """The solver's mode table for a finished Job, or the right abort.

    Both payload routes need exactly this preamble, and it fails the same way
    for both: not found, not finished, or unreadable.
    """
    job = _require_job(jobs_db, job_id)
    if job["status"] != DB.DONE:
        abort(404)
    try:
        return FLD.load_modes_json(store.read(STORE.job_key(job_id, "modes.json")))
    except FLD.FieldError:
        abort(502)
    except FileNotFoundError:
        abort(404)


def _mode_index(manifest: dict, store, job_id: str) -> dict:
    """What the viewer needs to draw the mode *list*, and nothing heavier.

    ``available`` is honest rather than decorative: a mode whose field file is
    missing or the wrong size cannot be fetched, so the viewer greys it out
    instead of requesting it and showing a failure. Finding that out costs a
    HEAD (or a stat) against the size the grid demands -- not a read of a
    multi-megabyte blob, which would be most of the cost the split exists to
    avoid, and not a decode of one, which would be all of it.

    The filename is the one the fetch route uses, from the one place that names
    it (``FIELD_NAME``). An index that promised a mode the fetcher could not
    deliver would be the same defect in a different costume: the list says the
    mode is there, and the truth arrives only after a doomed request.
    """
    modes = []
    for mode in manifest.get("modes", []):
        index = int(mode["index"])
        try:
            size = store.size(STORE.job_key(job_id, FIELD_NAME.format(index)))
        except STORE.StorageError:
            size = None
        modes.append({
            "index": index,
            "freq_hz": mode.get("freq_hz"),
            "vacuum_wavelength_m": mode.get("vacuum_wavelength_m"),
            "available": FLD.field_is_readable(manifest["grid"], size),
        })
    return {"grid": manifest["grid"], "units": manifest.get("units"),
            "n_index": manifest.get("n_index"), "modes": modes}


def _slice_positions(grid: dict):
    """One representative slice per family, taken at the object's centre."""
    out = []
    for entry in FLD.slice_axes(grid):
        count = entry["count"]
        out.append((entry["name"],
                    entry["origin_m"] + entry["spacing_m"] * (count // 2)))
    return out


def _bytes_io(data: bytes):
    import io

    return io.BytesIO(data)


def _store(app):
    root = app.config.get("MSE_LOCAL_STORE")
    if root:
        return STORE.LocalStore(root)
    return STORE.S3Store()


def _job_store(app):
    dsn = app.config.get("MSE_DSN")
    if dsn:
        return DB.JobStore(dsn)
    dsn = DB.dsn_from_env()
    if app.config.get("MSE_DB_MOCK") is not None:
        return app.config["MSE_DB_MOCK"]
    return DB.JobStore(dsn)


def _launcher(app):
    if app.config.get("MSE_JOB_LAUNCHER") is not None:
        return app.config["MSE_JOB_LAUNCHER"]
    return JOBS.K8sJobLauncher(
        namespace=CFG.NAMESPACE, image=CFG.solver_image(),
        pull_secret=CFG.pull_secret(), cpu=CFG.JOB_CPU, memory=CFG.JOB_MEMORY,
        active_deadline_seconds=CFG.JOB_ACTIVE_DEADLINE_SECONDS,
        ttl_seconds_after_finished=CFG.JOB_TTL_SECONDS_AFTER_FINISHED,
        dry_run=app.config.get("DRY_RUN", CFG.DRY_RUN),
    )


def start_ticker(app, store, jobs_db, launcher):
    """Background thread: run :func:`run_tick` forever.

    Nothing of ours can push into the cluster, so the tick is how the next
    queued upload gets started once the previous Job is gone.
    """
    stop = threading.Event()

    def _loop():
        while not stop.is_set():
            try:
                with app.app_context():
                    run_tick(store, jobs_db, launcher)
            except Exception as exc:  # noqa: BLE001 - keep the thread alive
                print(f"[mse-web] tick failed: {exc}", flush=True)
            stop.wait(CFG.TICK_SECONDS)

    thread = threading.Thread(target=_loop, name="mse-tick", daemon=True)
    thread.start()
    return stop, thread


#: Set once the ticker is running. Belt and braces with the single gunicorn
#: worker: the scheduler must run exactly once per process, and a second thread
#: would let two uploads launch at the same time and fight over the one Job slot.
_TICKER_STARTED = False


# `create_app` returns several things so the tests can reach the collaborators;
# the WSGI entrypoint only wants the app.
def build_app() -> Flask:
    """WSGI entrypoint. Starts the ticker exactly once per process.

    Deployed as ``gunicorn -w 1 --threads 8``: one worker, so there is one
    process and this runs once. The guard below is what keeps that promise if
    the worker count is ever raised by mistake -- a second ticker would race the
    first for the single namespace Job slot.
    """
    global _TICKER_STARTED
    app, store, jobs_db, launcher = create_app()
    if os.environ.get("MSE_NO_TICK", "") in ("", "0", "false", "False"):
        if not _TICKER_STARTED:
            _TICKER_STARTED = True
            start_ticker(app, store, jobs_db, launcher)
    return app


if __name__ == "__main__":
    build_app().run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
