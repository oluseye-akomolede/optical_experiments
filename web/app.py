"""The MSE web app: upload an .obj, watch the solver, view the modes.

Owner-facing decisions this implements: .obj upload, a units menu (mm default),
a refractive index field (default 1.0), the 20 lowest optical modes, and a
viewer that shows the object with a rotatable 3-D plot of the field. Tailnet
only, no login.

Concurrency is one. An upload is recorded as ``queued`` and a background tick
launches the oldest queued job whenever no solver Job is occupying the single
namespace slot; a finished Job is deleted as soon as its result is in the
database, because the quota counts finished Jobs too.

Result files are proxied from MinIO -- the bucket stays private and nothing is
presigned.
"""

from __future__ import annotations

import gzip
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import psycopg2
from flask import (Flask, abort, flash, jsonify, make_response, redirect,
                   render_template, request, send_file, url_for)
from werkzeug.exceptions import HTTPException
from flask.json.provider import DefaultJSONProvider

import config as CFG
import db as DB
import fields as FLD
import jobs as JOBS
import pc_form as PC
import storage as STORE
import worker_api as WORKER
from meshing import (MAX_EDGE_OVERSHOOT, UploadError, effective_edge_mm,
                     estimate_tets, load_parts, load_surface, parse_parts,
                     unit_scale_m, validate_upload)

#: The form field prefix for a per-part refractive index: ``part_index.<name>``.
#: The NAME is in the field key rather than a positionally-indexed list, because
#: a name cannot drift from the part it belongs to the way an index can -- the
#: form and the solver both re-derive the part list from the same bytes, and the
#: only thing that has to survive the round trip is the name.
_MATERIAL_FIELD_PREFIX = "part_index."

#: where the solver puts one mode's field, and the ONE place that decides it.
#: Both the index (which reads a size to decide `available`) and the fetch route
#: (which reads the bytes) go through this, so they cannot drift apart and leave
#: the list promising a mode the fetcher cannot serve.
FIELD_NAME = "mode_{:02d}.bin"

#: Non-empty when the scheduler has hit a failure that is NOT transient -- a
#: statement that will raise on every tick, as opposed to a database that is
#: briefly unreachable. Module state, read by ``/healthz``, because the whole
#: point is that a failing tick must leave a trace somewhere other than a log
#: line nobody reads: the tick's own return value is ``[]`` either way, and
#: "nothing expired" and "the reclaim is broken" must not look the same.
#:
#: It is a LIST, not an exception slot: the first failure is the one worth
#: reading, and appending keeps it while later ones accumulate behind it.
SCHEDULER_DEGRADED: list[str] = []

#: How many reclaims have failed in a row, so ``/healthz`` can distinguish a
#: one-off from a statement that fails every tick -- the difference between a
#: blip someone should look at and a broken deploy someone must fix.
_reclaim_failures = 0


class StrictJSON(DefaultJSONProvider):
    """A JSON provider that refuses to write ``NaN``/``Infinity``.

    ``json.dumps`` emits the bare tokens ``NaN`` and ``Infinity`` by default.
    They are not JSON, and the browser's ``JSON.parse`` rejects the whole
    document over one of them -- so the failure surfaces as "the results could
    not be loaded" in a viewer, on a payload the server thought it had sent
    successfully, with nothing in the logs. That is how the owner's 3-D view
    broke on 2026-09-27.

    ``allow_nan=False`` turns that into a ``ValueError`` raised inside the
    request: a 500 in tests and in production, naming the route, instead of a
    silent browser failure. The viewer path converts non-finite samples to
    ``null`` (see ``fields._round``), so this never fires for correct data --
    it is the tripwire for a future field that forgets, which is precisely the
    bug that shipped.
    """

    def dumps(self, obj, **kwargs):
        kwargs.setdefault("allow_nan", False)
        return super().dumps(obj, **kwargs)


def create_app(config: dict | None = None) -> tuple[Flask, object, object, object]:
    """Build the app *and its collaborators*, and hand all four back.

    The return is a 4-tuple, not a bare ``Flask``: the store, the jobs database
    and the launcher are built from the same config, and a caller that wants to
    test a route against a store it controls needs them. ``build_app`` below is
    the WSGI entrypoint and drops them, which is where a caller who only wants
    the app should look.
    """
    app = Flask(__name__)
    app.json = StrictJSON(app)
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
        """Liveness, plus whether the scheduler is still able to do its job.

        ``status: ok`` means the process is up. ``degraded`` is the separate
        question, and it is reported rather than folded into the status code:
        a scheduler that cannot run its reclaim is still serving pages and
        still has the cluster fallback, so it is not DOWN -- but it is not
        healthy either, and a monitor that only reads the status code would
        never learn that the GPU queue stopped bounding its crash loops.
        """
        body = {"status": "ok", "degraded": bool(SCHEDULER_DEGRADED)}
        if SCHEDULER_DEGRADED:
            body["degraded_reason"] = SCHEDULER_DEGRADED[0]
        return jsonify(body)

    @app.route("/", methods=["GET", "POST"])
    def index():
        if request.method == "POST":
            # The mode selector decides which handler sees this POST. It is
            # dispatched on a form field, not on whether a file arrived: a pc
            # form that carried a file by accident must be REFUSED, not quietly
            # handled as a cavity upload, and the two handlers share no
            # validation. The default is the cavity path, so a form from before
            # this selector existed (no `mode` field at all) uploads exactly as
            # it did.
            if request.form.get("mode") == PC.MODE:
                return _handle_pc_upload(app, store, jobs_db)
            return _handle_upload(app, store, jobs_db)
        return render_template("index.html", jobs=jobs_db.list_jobs(25),
                              units=CFG.UNIT_CHOICES, default_units=CFG.DEFAULT_UNITS,
                              feature_unit_choices=CFG.FEATURE_UNIT_CHOICES,
                              default_feature_unit=CFG.DEFAULT_FEATURE_UNIT,
                              mode_choices=CFG.MODE_CHOICES,
                              default_n_modes=CFG.DEFAULT_N_MODES,
                              lattices=PC.LATTICE_CHOICES,
                              polarisations=PC.POLARISATIONS,
                              max_bands=PC.MAX_BANDS,
                              max_divisions=PC.MAX_DIVISIONS,
                              max_mb=CFG.MAX_UPLOAD_BYTES // (1024 * 1024))

    @app.route("/parts", methods=["POST"])
    def parts():
        """The part names in an uploaded .obj -- the FIRST of the form's two POSTs.

        This is a PARSE-ONLY endpoint: it keeps NO server-side state between the
        two POSTs. The form sends the file here, gets back the names, and the
        owner assigns a refractive index to each; the final POST re-sends the
        file plus the table. Nothing is held in a session, a temp dir or a cache,
        because a name held server-side is a name that can belong to a different
        upload than the one that finally arrives.

        The parse is the SOLVER'S parser, not a second one. The owner is choosing
        which region gets which index, so a client-side or duplicate server-side
        scan is a second source of truth for exactly the thing that decides the
        physics -- and it would drift the first time either side changed.

        Returns ``{"parts": [...]}`` on success and ``{"error": "..."}`` with a
        4xx on a refusal, because every failure here is one the owner can act on
        (a part that is not closed, two parts that touch) and the message names
        the part. A refusal that only said "invalid file" would be the
        could-not-ask shape: the owner cannot fix what the page will not name.
        """
        uploaded = request.files.get("meshfile")
        if uploaded is None or not uploaded.filename:
            return jsonify({"error": "Please choose a file to upload."}), 400
        units = request.form.get("units", CFG.DEFAULT_UNITS)
        if units not in CFG.UNIT_CHOICES:
            return jsonify({"error": "Please choose one of the supported "
                                     "units."}), 400
        data = uploaded.read()
        try:
            # ``validate_upload`` first for the SIZE and readability rails (it
            # is the one place MAX_UPLOAD_BYTES is enforced), then the solver's
            # own parser for everything geometric.
            #
            # The order matters and is not cosmetic. ``validate_upload`` merges
            # a multi-part file into one mesh before checking it is closed, so
            # its refusal cannot name which part is open -- and naming the part
            # is the entire reason this endpoint exists. The parser splits
            # first and names what it rejects, so it speaks last.
            validate_upload(uploaded.filename, data, CFG.MAX_UPLOAD_BYTES)
            names = parse_parts(data, units)
        except UploadError as exc:
            return jsonify({"error": str(exc)}), 400
        return jsonify({"parts": names})

    @app.route("/jobs/<job_id>")
    def job_page(job_id):
        job = _require_job(jobs_db, job_id)
        # The achieved edge lives in the manifest, so it exists only for a
        # finished job. A missing manifest is an ABSENCE, not a failure: the
        # page then shows the request alone, which is the truth for a job that
        # has not meshed yet.
        achieved_m = None
        if job.get("status") == DB.DONE:
            manifest = _done_manifest(store, jobs_db, job_id)
            if manifest:
                achieved_m = manifest.get("achieved_median_edge_m")
        # A pc job's own header fields, read from the row's stored request --
        # the same reason the materials table is rendered from the STORED JSON:
        # a page that re-derived them from the form could show a lattice the
        # solve never used while every number on it looked right. Absent for a
        # cavity job, so the template's `and pc_request` guard is also the
        # kind guard rather than a second one that could drift from it.
        pc_request = job.get("pcs") if job.get("kind") == DB.PC else None
        return render_template("job.html", job=job,
                              effective_size=_effective_size(job),
                              resolution=_resolution_caption(job, achieved_m),
                              max_edge_overshoot=MAX_EDGE_OVERSHOOT,
                              pc_request=pc_request,
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
            # The route, so the page picks a view without a second request.
            # Adding a key is backward-compatible: an old reader ignores it.
            "kind": job["kind"],
        }
        if job["status"] == DB.DONE:
            if job["kind"] == DB.PC:
                # A pc job's result is the band diagram the worker wrote back
                # into pcs. The 3-D keys (modes/manifest/surface) are NOT sent:
                # sending an empty mode list would read as "no modes" rather
                # than "not a mode job" -- the deploy-3 shape, at the page.
                pcs = job.get("pcs") or {}
                payload["bands"] = pcs.get("bands")
            else:
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
        # A .vtu is the one result file the user takes *away* -- the viewer
        # renders the field itself, so the only reason to fetch this is to open
        # it in ParaView, and `as_attachment` is what makes a browser save it
        # instead of rendering it. The name is made readable rather than left as
        # the solver's `mode_03.vtu`: on a downloads shelf, twenty files called
        # `mode_NN.vtu` from twenty jobs are indistinguishable, so the object's
        # name goes in front.
        if name.endswith(".vtu"):
            return send_file(_bytes_io(data), mimetype=_mimetype(name),
                             as_attachment=True,
                             download_name=_vtu_download_name(job, name),
                             conditional=True)
        return send_file(_bytes_io(data), mimetype=_mimetype(name),
                         download_name=name, conditional=True)

    # -- derived payloads the viewer needs --------------------------------
    #
    # The payload is split so a phone downloads one mode's field, not twenty.
    # ``index.json`` is the *index*: one entry per mode with its frequency and
    # its index, no field data. The heavy arrays live in
    # ``field/<n>.bin`` and are fetched when that mode is first selected. A job
    # whose modes are all unreadable is still a payload, not an error -- the
    # index is what the mode table renders from.
    @app.route("/jobs/<job_id>/index.json")
    def job_index(job_id):
        """The mode index: what each mode is, without any field data."""
        manifest = _done_manifest(store, jobs_db, job_id)
        try:
            index = _mode_index(manifest, store, job_id)
        except STORE.StorageError:
            # We could not check the field files, so we cannot say which modes
            # are available -- and a 200 here would say all of them are gone.
            abort(502)
        return jsonify(index)

    @app.route("/jobs/<job_id>/field/<int:number>.bin")
    def job_field(job_id, number):
        """One mode's field as float16 bytes, for the volume plot.

        Raw little-endian float16, ``(nz, ny, nx, 3)`` C order, NaN outside the
        object -- the browser uploads it straight into a HalfFloat 3D texture,
        so any reshaping here would only add a chance to get the axis order
        wrong. gzip because the payload is large and mostly smooth; the ratio is
        modest (the object fills most of its bounding box, so few samples are
        NaN) but it is still a third off.

        Availability: 404 for a mode the table does not have or whose file is
        gone, 502 when storage cannot be reached -- a store that cannot answer
        must never look like a mode that is not there.
        """
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
            abort(404)
        except STORE.StorageError:
            # As on the index route: a store that cannot answer must not look
            # like a mode that is not there.
            abort(502)
        payload = FLD.encode_field16(field, manifest["grid"])
        response = make_response(gzip.compress(payload, 6))
        response.headers["Content-Type"] = "application/octet-stream"
        response.headers["Content-Encoding"] = "gzip"
        response.headers["X-Field-Dtype"] = "float16-le"
        return response

    # The GPU workers are a separate queue with a separate entry point, so they
    # do not go through the Flask app object the pages are built on.
    WORKER.register(app, jobs_db, store)
    WORKER.register_blueprint(app)

    _log_worker_failures(app)

    return app, store, jobs_db, launcher


#: One slot for the reason of the request IN FLIGHT. A dict rather than a
#: module global holding a string, so the mutation is a rebind a test can see.
#: Keyed on nothing: the wrapper and ``after_request`` both run inside ONE
#: request, and a stale entry would mislabel the NEXT request's reason, which is
#: why ``after_request`` POPS rather than reads.
_WORKER_ABORT_REASON: dict[str, str] = {}


def _log_worker_failures(app) -> None:
    """Log the reason for every non-2xx the worker API returns.

    On deploy 3 every Jetson job failed right after a successful solve and the
    app said nothing at all: a 4xx/5xx on ``/worker/*`` was returned to the
    worker and forgotten, so the worker's journal held only the generic owner
    sentence and the app's log held nothing. The one place that knew the reason
    -- the abort's description, or the traceback of a raised exception -- kept
    it to itself, and the failure could not be diagnosed from either side.

    This is the app's half of the fix; the worker's half is ``log_exception`` in
    ``worker/mse_worker.py``. Together they mean a failing worker call leaves a
    record on both ends, which is what makes the two ends reconcilable.

    WARNING, not ERROR: a 409 is a NORMAL outcome here -- it is how a worker
    learns its lease is gone -- and logging it at ERROR would train whoever
    reads the log to ignore the level that matters.

    **WHY THE HANDLERS ARE WRAPPED RATHER THAN A SIGNAL USED.** The obvious hook
    is Flask's own ``got_request_exception`` signal, and it does not work here:
    reading Flask 3.1's ``app.py``, that signal is sent from ``handle_exception``
    ALONE -- the unhandled path that produces a 500 -- while an ``abort()``
    becomes a response in ``handle_user_exception`` and returns before any signal
    fires. Every failure this blueprint produces is an ``abort``, so a
    signal-based hook logged NOTHING for them; measured, not assumed, because
    that hook was written first and reported "no reason given" for a 503 whose
    description was right there. ``request.routing_exception`` is no help either
    -- it is set only for URL-ROUTING failures.

    Wrapping both handlers is what sees every reason, because every failure on
    this blueprint leaves Flask through one of the two. ``after_request`` stays
    as well, for the responses that never raise at all: a route returning
    ``(..., 4xx)`` directly has no exception, and a handler-only hook would miss
    it.
    """
    original_user = app.handle_user_exception
    original_500 = app.handle_exception

    def _note(exception) -> None:
        """Stash the reason for the request in flight, if it is a worker one."""
        try:
            path = request.path
        except RuntimeError:      # no request context: nothing to attribute
            return
        if not path.startswith("/worker/"):
            return
        if isinstance(exception, HTTPException):
            _WORKER_ABORT_REASON["reason"] = (
                getattr(exception, "description", "") or str(exception))
        else:
            # A real exception: the traceback IS the reason, and ERROR is the
            # right level, because an unhandled raise is a bug in the app rather
            # than a worker that lost its lease.
            app.logger.exception(
                "worker API %s %s raised %s", request.method, path,
                type(exception).__name__)

    def user_handler(exception):  # noqa: ANN001
        _note(exception)
        return original_user(exception)

    def server_handler(exception):  # noqa: ANN001
        _note(exception)
        return original_500(exception)

    app.handle_user_exception = user_handler
    app.handle_exception = server_handler

    @app.after_request
    def _record_worker_failure(response):  # noqa: ANN001 - Flask's signature
        if not request.path.startswith("/worker/"):
            return response
        if response.status_code < 400:
            return response
        # Consumed once, so a later request cannot inherit this one's reason.
        reason = _WORKER_ABORT_REASON.pop("reason", "")
        if not reason and response.status_code >= 500:
            reason = "no reason given (the traceback above, if any, is the rest)"
        app.logger.warning(
            "worker API %s %s -> %s (%s)", request.method, request.path,
            response.status_code, reason or "no reason given")
        return response


def run_tick(store, jobs_db, launcher, log=print) -> dict:
    """One pass of the scheduler: reclaim dead Jobs, then launch at most one.

    Safe to call concurrently and safe to call when nothing is queued. Returns a
    small summary so the tests (and the log) can see what it decided.
    """
    result = {"deleted": [], "failed": 0, "launched": None, "waited": False,
              "rerouted": 0, "reclaimed": []}

    # -- the GPU queue ------------------------------------------------------
    #
    # Two things can go wrong with a job that was routed to the Jetson, and both
    # end the same way: it goes back on the cluster queue. Doing this FIRST, and
    # before the Job slot is considered, means the cluster picks the job up on
    # the same tick that it gives up on the GPU.
    #
    # 1. Nobody claimed it. The Jetson may be off, busy with another rig, or its
    #    worker may be wedged. `gpu_queued_at` is the clock.
    # 2. A worker claimed it and died. Its lease expired with no renewal.
    #
    # Both are best-effort: a database error here must not stop the rest of the
    # tick, because the cluster queue still works and it is the fallback that
    # failed, not the solver.
    result["reclaimed"] = _reclaim_expired_leases(jobs_db, log)
    result["rerouted"] = _reroute_abandoned_gpu_jobs(jobs_db, launcher, log)

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
                               job["n_modes"],
                               min_feature_m=job.get("min_feature_m"),
                               scale=job.get("scale", 1.0) or 1.0,
                               kind=job.get("kind"))
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


def _reclaim_expired_leases(jobs_db, log=print) -> list[str]:
    """Expire leases whose worker stopped renewing. Returns the job ids.

    An expired lease is the only evidence a dead GPU worker leaves: it renews on
    every progress call, so a lease that ran out was not renewed, and a worker
    that cannot renew is a worker that is not running the job any more. The row
    goes back in the claimable pool; the wait-window fallback below is what
    ultimately moves it to the cluster if no worker comes back for it.
    """
    # Declared once, at the top: Python requires the ``global`` statement to
    # appear before any use of the name in the function, and using it inside the
    # ``except`` below while declaring it in the success path is a SyntaxError.
    global _reclaim_failures

    try:
        ids = jobs_db.reclaim_expired_leases()
    except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
        # Transport trouble: the database is down, restarting, or the
        # connection died. The next tick will try again and the cluster queue
        # still works, so this is a log line and a shrug. NOT degraded: raising
        # an alarm for a blip trains whoever reads it to ignore the alarm.
        log(f"[mse-web] could not reach the database to reclaim expired GPU "
            f"leases: {exc}")
        return []
    except Exception as exc:  # noqa: BLE001
        # Everything else is a BROKEN STATEMENT, not a broken connection: a
        # type-adaptation error, a bad column, a constraint. It will fail
        # identically on every tick, so it is not transient and the next tick
        # will not fix it. This is the path that bounds a crash-looping worker,
        # so a silent failure here makes the bound unreachable exactly when it
        # is needed -- and returning [] would look identical to "nothing
        # expired". Log it at ERROR and record it where /healthz can report it,
        # so a swallowed exception is never the only trace.
        _reclaim_failures += 1
        note = (f"{type(exc).__name__}: {exc}"
                f" (last seen {datetime.now(timezone.utc).isoformat()}"
                f", {_reclaim_failures} in a row)")
        SCHEDULER_DEGRADED.clear()
        SCHEDULER_DEGRADED.append(note)
        log(f"[mse-web] ERROR: reclaiming expired GPU leases failed and will "
            f"keep failing: {note}")
        return []
    # A tick that got through clears the flag. It tracks the CURRENT state, not
    # "something once went wrong": a statement that has been fixed (a deploy, a
    # migration) must stop reporting degraded without needing a restart, and a
    # permanent latch would make the signal useless the first time someone read
    # it and did not act immediately.
    SCHEDULER_DEGRADED.clear()
    _reclaim_failures = 0
    for job_id in ids:
        log(f"[mse-web] job {job_id}: its GPU worker's lease expired; "
            f"the job is claimable again")
    return ids


def _reroute_abandoned_gpu_jobs(jobs_db, launcher, log=print) -> int:
    """Move GPU jobs nobody claimed in time back to the cluster. Returns how many.

    The dispatch's fallback: a job that waited longer than ``MSE_GPU_WAIT_S``
    for a worker was routed on an estimate, and the estimate is no longer worth
    holding the job hostage. It goes back on the cluster queue where it is
    bounded by the same single Job slot as everything else.

    **The cluster's limit is respected, not assumed.** A job whose estimate is
    over ``MSE_CPU_MAX_TETS`` cannot be solved on the cluster, so rerouting it
    would only make it fail there after a queue wait. That case is left on the
    GPU queue and logged plainly: a job waiting for a worker that may still come
    is better than a job guaranteed to fail. This mirrors the solver's own split
    between "wrong machine" and "too big for any machine".
    """
    if launcher is None:
        return 0
    try:
        waiting = jobs_db.jetson_waiting(CFG.gpu_wait_seconds())
    except Exception as exc:  # noqa: BLE001 - never let a tick crash the loop
        log(f"[mse-web] could not list waiting GPU jobs: {exc}")
        return 0

    if not waiting:
        return 0

    # No GPU worker is configured, so no worker is coming and waiting out the
    # window is pointless. Only meaningful when a token is absent, which is also
    # when the worker API answers 503.
    if not WORKER.configured_token():
        log("[mse-web] no GPU worker token is configured, so the GPU queue "
            "cannot drain; rerouting eligible jobs now")

    limit = CFG.max_tets()
    moved = 0
    for entry in waiting:
        job_id = entry["job_id"]
        est = entry.get("est_tets")
        if est is not None and int(est) > limit:
            log(f"[mse-web] job {job_id}: estimated {int(est):,} elements, over "
                f"the {limit:,} the cluster can solve; it stays queued for the GPU")
            continue
        if jobs_db.reroute_to_cluster(
                job_id,
                "This object was queued for the graphics processor, which has "
                "not picked it up, so it has been moved to the cluster queue."):
            moved += 1
            log(f"[mse-web] job {job_id}: no GPU worker claimed it in "
                f"{CFG.gpu_wait_seconds():.0f}s; moved to the cluster queue")
    return moved


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

    ONLY CLUSTER JOBS ARE CONSIDERED, and that exclusion is the whole point of
    the ``target`` check below. A JETSON row never has a ``k8s_job_name`` and
    never will: the GPU worker is a systemd unit the owner runs, not a Job this
    app launches, so it is absent from ``active_job_names`` by construction. To a
    reaper that asks only "is this row's Job gone?", every Jetson job answered
    YES from the moment it was claimed, and it was failed at the first tick after
    the claim -- the second dispatch deploy died exactly this way, with
    ``job <id> failed: its Job <unnamed> is gone`` for every GPU job.

    A Jetson row's liveness is its LEASE, not a Job: the worker renews on every
    heartbeat, and a worker that dies leaves the lease to expire, which
    ``reclaim_expired_leases`` already handles (requeue, and fail past the
    bound). Two mechanisms for one question is how they dissagree; each machine
    gets the one that can actually see it.
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
        # A Jetson row is not this function's to judge (see the docstring): it
        # has no Job, so "no Job present" is its NORMAL state rather than
        # evidence of death. Skip it entirely -- the lease owns it.
        #
        # ``target`` is read from the row, never inferred from the missing name:
        # a CLUSTER row whose launch failed before ``mark_running`` stored a name
        # also has no name, and THAT one really is dead and must still be failed.
        if job.get("target") == DB.JETSON:
            continue  # JETSON-REAPER-SKIP-ANCHOR
        name = job.get("k8s_job_name")
        if name and name in states:
            reason = states[name]
        elif name and name in present:
            continue                       # alive and well; not ours to touch
        else:
            reason = None

        # The solver knows the exact element count and the app does not, so a
        # job that says "this machine is too small for me" is re-queued rather
        # than failed. The move is guarded on the row still being a *cluster*
        # job: the solver only ever raises this on the cpu backend, and a jetson
        # row that somehow carries the reason must not be sent to the cluster
        # it was routed away from a moment ago.
        if reason == JOBS._REROUTE_REASON and job.get("target") != DB.JETSON:
            if jobs_db.reroute_to_cluster(
                    job["job_id"],
                    "This object needs more elements than the cluster solver "
                    "can handle, so it has been queued for the graphics "
                    "processor instead."):
                log(f"[mse-web] job {job['job_id']} re-routed to the GPU queue: "
                    f"its Job {name or '<unnamed>'} reported the mesh is too "
                    f"large for the cluster")
                continue
            # Nothing moved: the row was not a queued cluster job any more, so
            # someone else already dealt with it. Fall through to the plain
            # failure rather than claim a reroute that did not happen.

        jobs_db.mark_failed(job["job_id"], JOBS.failure_message(reason))
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

    try:
        n_modes = int(request.form.get("n_modes", CFG.DEFAULT_N_MODES))
    except ValueError:
        flash("Please choose one of the offered mode counts.")
        return redirect(url_for("index"))
    if n_modes not in CFG.MODE_CHOICES:
        flash("Please choose one of the offered mode counts.")
        return redirect(url_for("index"))

    # The optional minimum-feature-size control. Blank means "auto": the solver's
    # own physics-determined edge length, unchanged.
    #
    # The number and its unit arrive separately, and the unit is the owner's
    # choose from FEATURE_UNIT_CHOICES -- NOT the object's units. It is converted
    # to metres HERE, once, and everything downstream carries metres. That is the
    # fix for the length-space bug: the value used to be a bare number named
    # `min_feature_mm` that every reader had to interpret, and the label, the
    # flashes and the name disagreed about which unit it was.
    detail = (request.form.get("detail") or "").strip()
    feature_unit = (request.form.get("feature_unit")
                    or CFG.DEFAULT_FEATURE_UNIT).strip()
    min_feature_value = None
    min_feature_m = None
    if detail and detail != "auto":
        if feature_unit not in CFG.FEATURE_UNIT_CHOICES:
            flash("Please choose one of the offered units for the smallest "
                  "feature.")
            return redirect(url_for("index"))
        try:
            min_feature_value = float(detail)
        except ValueError:
            flash("The smallest feature must be a number, or left on "
                  "automatic.")
            return redirect(url_for("index"))
        if not (min_feature_value > 0) or min_feature_value != min_feature_value:
            flash(f"The smallest feature must be greater than zero "
                  f"{feature_unit}.")
            return redirect(url_for("index"))
        min_feature_m = min_feature_value * unit_scale_m(feature_unit)

    # The free scale factor: the object is multiplied by this before anything
    # measures it. Blank or 1 means "as uploaded", which is also what every job
    # created before this control existed gets.
    scale_text = (request.form.get("scale") or "").strip()
    scale = 1.0
    if scale_text:
        try:
            scale = float(scale_text)
        except ValueError:
            flash("The scale must be a number, or left blank.")
            return redirect(url_for("index"))
        if not (CFG.MIN_SCALE <= scale <= CFG.MAX_SCALE) or scale != scale:
            flash(f"The scale must be between {CFG.MIN_SCALE:g} and "
                  f"{CFG.MAX_SCALE:g}.")
            return redirect(url_for("index"))

    data = uploaded.read()
    try:
        validate_upload(uploaded.filename, data, CFG.MAX_UPLOAD_BYTES)
    except UploadError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    # Decide the route from the geometry, before anything is queued. The
    # estimate is deliberately conservative, and an estimate that FAILED (0,
    # meaning "could not ask") routes to the cluster rather than the GPU: a
    # failed estimate must never send a job to the slower machine, and it is the
    # failure mode this rig keeps finding where a non-answer is read as one.
    # --- the materials table, if the form sent one ------------------------
    #
    # The form posts one ``part_index.<name>`` field per part it showed the
    # owner. Absent entirely means the one-part path, byte-identical to before
    # this existed -- NOT an empty table, which would be a form that named no
    # parts and can only be a mismatch.
    #
    # The part list is RE-DERIVED here from the bytes this POST actually
    # carried, and compared with the names the form used. The owner assigned an
    # index to a name in an earlier POST; if the file changed in between, the
    # name they chose belongs to a different upload, and applying it silently
    # would put a material on the wrong region while every count agreed. This is
    # the same check the /parts endpoint exists to make possible.
    materials = None
    parts = None
    material_fields = {k: v for k, v in request.form.items()
                       if k.startswith(_MATERIAL_FIELD_PREFIX)}
    if material_fields:
        try:
            parts = load_parts(data, units)
        except UploadError as exc:
            flash(str(exc))
            return redirect(url_for("index"))

        materials = {}
        for field, raw in material_fields.items():
            part_name = field[len(_MATERIAL_FIELD_PREFIX):]
            try:
                n_part = float(raw)
            except (TypeError, ValueError):
                flash(f"The refractive index for part {part_name!r} must be a "
                      f"number.")
                return redirect(url_for("index"))
            if not (CFG.MIN_MATERIAL_N <= n_part <= CFG.MAX_MATERIAL_N):
                flash(f"The refractive index for part {part_name!r} must be "
                      f"between {CFG.MIN_MATERIAL_N:g} and "
                      f"{CFG.MAX_MATERIAL_N:g}.")
                return redirect(url_for("index"))
            materials[part_name] = n_part

        # Every part needs an index, and no index may name a part that is not
        # in this file. Both are mismatches between the form and the bytes, and
        # the first would otherwise leave a region at whatever the default
        # happened to be -- frequencies for a cavity that is partly empty with
        # nothing saying so.
        missing = [n for n in parts.names if n not in materials]
        if missing:
            flash("No refractive index was given for part(s) "
                  + ", ".join(repr(m) for m in missing)
                  + ". Every part needs one.")
            return redirect(url_for("index"))
        unknown = [k for k in materials if k not in parts.names]
        if unknown:
            flash("The form named part(s) "
                  + ", ".join(repr(u) for u in unknown)
                  + " that are not in this upload. Its parts are "
                  + ", ".join(repr(n) for n in parts.names) + ".")
            return redirect(url_for("index"))

    surface = load_surface(data)
    est_tets = estimate_tets(surface, n_modes, units, c=CFG.est_tets_c(),
                             min_feature_m=min_feature_m, scale=scale,
                             parts=parts, materials=materials)

    if est_tets > CFG.upload_refuse_tets():
        flash(
            f"This model is too detailed to solve: it needs roughly "
            f"{est_tets:,} elements and the most we can solve is "
            f"{CFG.max_tets():,}. Please simplify the model (fewer tiny "
            f"features, or a coarser surface) and upload it again."
        )
        return redirect(url_for("index"))

    target = DB.JETSON if est_tets >= CFG.gpu_min_tets() else DB.CLUSTER

    job_id = uuid.uuid4()
    try:
        store.write(STORE.job_key(job_id, "input.obj"), data)
    except STORE.StorageError as exc:
        flash(str(exc))
        return redirect(url_for("index"))

    jobs_db.create_job(job_id, uploaded.filename, units, n_index,
                       n_modes, target=target, est_tets=est_tets,
                       min_feature_m=min_feature_m,
                       min_feature_value=min_feature_value,
                       min_feature_unit=(feature_unit if min_feature_value
                                         is not None else None),
                       scale=scale, materials=materials)

    # Try to start it right away; if the slot is busy it simply stays queued and
    # the tick will pick it up.
    try:
        run_tick(store, jobs_db, app.extensions["mse"]["jobs"])
    except Exception as exc:  # noqa: BLE001 - the tick is best-effort
        app.logger.warning("scheduler tick failed after upload: %s", exc)

    return redirect(url_for("job_page", job_id=job_id))


def _handle_pc_upload(app, store, jobs_db):
    """Validate, cap-check and queue one photonic-crystal request.

    The cavity handler's sibling and its deliberate opposite in one respect:
    there is NO FILE. A band diagram is a property of the unit cell and the
    index contrast, so the request is a handful of numbers and there is nothing
    to upload. A file that arrives anyway is REFUSED rather than ignored: a
    request whose parameters came from a form and whose scope silently came from
    a leftover file input is a job that solves something the owner did not ask
    for, and the two handlers share no validation that would catch it.

    Everything numeric is validated by :mod:`pc_form`, which refuses by name and
    counts the cell through the solver's own mesher. This function's job is the
    part that needs the database: write the row and start it. It is a job like
    any other -- `kind='pc'` and the same claim/solve/store journey the cavity
    jobs take -- with ONE difference that is a routing fact, not a preference:
    it is created ``target=JETSON``.

    The pc solver exists nowhere but the Jetson worker. There is no cluster pc
    path in this change, so a pc row created ``target=cluster`` would be handed
    by ``run_tick`` to ``launcher.create``, which builds a CAVITY Job, and the
    band diagram would come back as a cavity solve of nothing. The route, the
    queue and the launcher all agree on this: the row is created for the Jetson,
    ``next_queued`` skips ``kind='pc'``, and ``launcher.create`` refuses one
    outright rather than trusting the filter upstream.
    """
    uploaded = request.files.get("meshfile")
    if uploaded is not None and uploaded.filename:
        # Not merely ignored. See the docstring: a pc form that carries a file
        # is a mismatch between what the owner saw and what the request means.
        flash("A band-diagram request has no file to upload. Pick the lattice "
              "and the parameters, and leave the file box empty.")
        return redirect(url_for("index"))

    try:
        planned = PC.plan_request(request.form)
    except PC.PCRequestError as exc:
        # Refused by name, with the limit named: this is the same shape as the
        # solver's own refusals and it is the point of the caps, not a failure
        # of them.
        flash(str(exc))
        return redirect(url_for("index"))

    request_json = PC.request_json(planned)
    job_id = uuid.uuid4()
    # The parameters the owner typed are the job's NAME. A pc job has no file,
    # so a list of rows all called the job id would be a page of UUIDs; the
    # lattice and the size are what tell one apart from another.
    label = (f"{planned['lattice']} lattice, r/a={planned['r_over_a']:g}, "
             f"{planned['n_bands']} bands, {planned['polarisation']}")

    jobs_db.create_job(job_id, label, CFG.DEFAULT_UNITS, 1.0,
                       planned["divisions"], target=DB.JETSON,
                       est_tets=planned["nodes"], kind=DB.PC,
                       pcs=request_json)

    try:
        run_tick(store, jobs_db, app.extensions["mse"]["jobs"])
    except Exception as exc:  # noqa: BLE001 - the tick is best-effort
        app.logger.warning("scheduler tick failed after pc upload: %s", exc)

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


def _effective_size(job: dict) -> str | None:
    """How big this object is inside the solver, in human units.

    Shown because neither `units` nor `scale` says it alone: a model 2 units
    across at scale 1000 is 2 km, and the frequencies that come back are the
    frequencies of the 2 km object. Reporting the effective size is what lets a
    surprising result be traced to a scale factor rather than argued about.

    Reads ONLY the job row -- no surface, no storage -- because it is called on
    every job-page render, including long after the result exists and the
    upload's own geometry is no longer the thing being displayed.

    Returns ``None`` when there is nothing to scale: an unscaled job's size is
    whatever the .obj said, and inventing a number for it would be worse than
    saying nothing.
    """
    try:
        scale = job.get("scale")
        n_modes = int(job.get("n_modes") or CFG.DEFAULT_N_MODES)
        units = job.get("units") or CFG.DEFAULT_UNITS
        factor = float(scale) if scale not in (None, "") else 1.0
        if factor == 1.0:
            return None
        # mse_common, not mse_meshing: the web image ships the shared module and
        # not the solver package (see config.py's import comment), and all this
        # caption needs is the unit table -- the same one object either way.
        root = Path(__file__).resolve().parent.parent / "dockerfiles" / "maxwell_solver"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        import mse_common as MESH  # noqa: PLC0415 - deliberately late, as elsewhere

        # The equivalent size in mm, which is the unit the owner reads the
        # upload form in: scaling by 1000 in mm is the same object as 1000x the
        # numbers, and mm is what makes the result comparable to the form.
        per_mm = MESH.unit_scale("mm") / MESH.unit_scale(units)
        equivalent_mm = factor * per_mm
        if equivalent_mm >= 1000.0:
            return f"{equivalent_mm / 1000.0:,.4g} m of model per unit of file"
        return f"{equivalent_mm:,.4g} mm of model per unit of file"
    except Exception:  # noqa: BLE001 - a caption must never break the page
        return None


def _resolution_caption(job: dict, achieved_m: float | None = None) -> dict | None:
    """The mesh resolution the owner asked for, and what they actually got.

    Both numbers, each LABELLED with its unit, because the feature's unit is its
    own and is not the object's -- a bare "0.2" beside a "0.27" is a pair the
    reader has to guess at, and a number the page cannot explain reads as a
    defect.

    "requested" is the length the mesh was ASKED to resolve: the smaller of the
    physics edge and the owner's declared feature. "achieved" is the median
    element edge the solver measured, reported through ``modes.json`` by
    ``write_results``; it is the same physical length, re-expressed in the unit
    the owner typed, so the two can be read against each other directly.

    ``achieved`` is present only once the job has finished -- before that there
    is no manifest and no measurement, and the page says so by showing the
    request alone rather than a placeholder that looks like data.

    Returns ``None`` when there is no feature to report (auto), so the page can
    say nothing rather than invent a number.
    """
    value = job.get("min_feature_value")
    unit = job.get("min_feature_unit")
    if value is None or not unit:
        return None
    try:
        caption = {
            "requested": f"{float(value):g} {unit}",
            "requested_m": job.get("min_feature_m"),
        }
        if achieved_m:
            # metres -> the owner's unit, via the SAME table the upload form
            # converted with. Referencing a bare ``MESH`` here was a NameError
            # (that name is only ever a local inside ``_effective_size``), which
            # would have crashed the job page for every finished job that set a
            # feature -- i.e. exactly the pages this caption was added for.
            scale = unit_scale_m(unit)
            caption["achieved"] = f"{float(achieved_m) / scale:.4g} {unit}"
            caption["achieved_m"] = float(achieved_m)
            # Above the request by up to MAX_EDGE_OVERSHOOT is the allowance,
            # not an error. Say which side of the line this run is on, so a
            # reader is not left to judge a bare ratio.
            req_m = job.get("min_feature_m")
            caption["over_request"] = bool(
                req_m and float(achieved_m) > float(req_m)
            )
            caption["within_allowance"] = bool(
                req_m and float(achieved_m) <= MAX_EDGE_OVERSHOOT * float(req_m)
            )
        return caption
    except (TypeError, ValueError):
        return None


def _safe_result_name(name: str) -> bool:
    """Only the filenames the solver actually writes may be proxied.

    A full match against a fixed alternation, not a prefix or a substring test:
    this string reaches a store lookup, and the only reason a traversing name
    cannot escape is that nothing shaped like one matches here at all. The
    backslash and the dot are literals rather than metacharacters, so ``..`` and
    ``/`` have no way to satisfy the pattern.
    """
    import re

    return bool(re.fullmatch(r"modes\.json|surface\.glb|mode_\d{2}\.bin|"
                             r"mode_\d{2}\.vtu", name))


def _vtu_download_name(job: dict, name: str) -> str:
    """``my-lens/mode_03.vtu`` -> ``my-lens-mode03.vtu``.

    Only the stem is kept: a browser turns a path separator into a directory
    (or, on some platforms, refuses the name outright), and the download shelf
    is not the place to reproduce the solver's ``mode_NN`` spelling. Anything
    that is not a safe filename character is dropped rather than escaped, so the
    result cannot become a path however the upload was named.
    """
    import re

    stem = os.path.basename(job.get("original_filename") or "") or "mode"
    stem = re.sub(r"\.(obj|stl|glb)$", "", stem, flags=re.IGNORECASE)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-.") or "mode"
    index = re.fullmatch(r"mode_(\d+)\.vtu", name)
    number = index.group(1) if index else "00"
    return f"{stem[:60]}-mode{number}.vtu"


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
    except STORE.StorageError:
        # A store we cannot reach is not a job that is gone. Every derived
        # payload route reads the manifest first, so without this the whole
        # viewer answers 500 during a storage outage -- which reads as a broken
        # app rather than as "we could not check", the distinction this file
        # keeps having to make. (Found by a test for the field route; the bug
        # was in this shared helper, so it was already reachable from the
        # index and per-mode routes.)
        abort(502)


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

    **``available`` is only ever set false from a successful check.** The store
    already draws the line this needs: ``None`` means "looked, and it is not
    there", while ``StorageError`` means "could not look at all". Collapsing the
    second into the first is the bug this function has already been fixed for
    once, and it is the same one the 403 handling was fixed for: a failure to ask
    rendered as an answer of nothing. So ``StorageError`` is allowed to
    propagate, and the route turns it into a 502 rather than a 200 full of false
    flags. A viewer that greys out every mode because the bucket was unreachable
    has been told something false about the data.
    """
    modes = []
    for mode in manifest.get("modes", []):
        index = int(mode["index"])
        # No except: a store that cannot answer must abort the request, not
        # quietly report every mode as missing.
        size = store.size(STORE.job_key(job_id, FIELD_NAME.format(index)))
        modes.append({
            "index": index,
            "freq_hz": mode.get("freq_hz"),
            "vacuum_wavelength_m": mode.get("vacuum_wavelength_m"),
            "available": FLD.field_is_readable(manifest["grid"], size),
        })
    return {"grid": manifest["grid"], "units": manifest.get("units"),
            "n_index": manifest.get("n_index"), "modes": modes}


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
        # BOTH caps, from the app's own configuration: the app is what decides
        # which uploads are allowed through, so the app is what must tell the
        # solver where the limits are. These used to be left unset, which meant
        # the solver fell back to its own default and refused jobs the app had
        # already accepted.
        max_tets=CFG.max_tets(), cpu_max_tets=CFG.cpu_max_tets(),
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
