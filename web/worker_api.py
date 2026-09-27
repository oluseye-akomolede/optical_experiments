"""The worker API: what a GPU worker calls to claim and run a job.

The Jetson has no inbound port and no Kubernetes credentials. It cannot be
reached, it cannot be scheduled to, and it cannot write to the bucket or the
database itself -- everything it does goes through this HTTP surface, which is
the only thing standing between a token and the solver's output.

Design rules, in the order they are load-bearing:

**Closed by default.** ``MSE_WORKER_TOKEN` unset means the whole API is
disabled and answers 503. This is not a nicety: this service is on a tailnet
with no login, so an API that fell open when a Secret failed to mount would be
an unauthenticated route to every upload and every result file. The failure
mode of a missing secret must be "the workers stop", never "everyone is a
worker".

**Constant-time comparison.** The token is compared with
``hmac.compare_digest``. A plain ``==`` leaks the token one byte at a time to
anyone who can time the response, and this route is reachable by anything on
the tailnet.

**Only the lease holder writes.** Every write is guarded on the caller still
holding a live lease, so a worker that slept through its lease and came back
cannot overwrite its successor's output.

**Allow-list, not deny-list, on filenames.** A worker uploads a fixed, tiny set
of names; anything else is refused before it reaches storage. ``..`` and
absolute paths are refused by construction rather than by sanitising.

Nothing here logs the token, and nothing reads it into an error message.
"""

from __future__ import annotations

import hmac
import os
import re

from flask import Blueprint, abort, jsonify, request

import config as CFG
import db as DB
import storage as STORE

bp = Blueprint("worker", __name__)

#: environment variable holding the shared worker token
TOKEN_ENV = "MSE_WORKER_TOKEN"


class WorkerAPIDisabled(Exception):
    """No token is configured, so there are no workers and no API.

    A distinct type from "the caller sent a bad token": one is a
    misconfiguration to fix, the other is a request to refuse. They get
    different status codes (503 vs 401) because conflating them would tell a
    worker with a wrong token to wait for an operator who has nothing to fix.
    """


def configured_token() -> str:
    """The token workers must present, or ``""`` when the API is disabled.

    Read per request rather than cached at import: the value arrives from a
    Secret, and a process that cached an empty value at startup would stay
    disabled after the Secret appeared.
    """
    return os.environ.get(TOKEN_ENV, "") or ""


def token_ok(presented: str) -> bool:
    """Whether ``presented`` is the configured token, compared in constant time.

    ``compare_digest`` also refuses to raise on non-ASCII input (it returns
    False), which matters because the header is attacker-controlled.
    """
    expected = configured_token()
    if not expected:
        return False
    try:
        return hmac.compare_digest(str(presented), expected)
    except (TypeError, ValueError):
        return False


def _require_worker() -> str:
    """The caller's owner id (their token fingerprint), or abort.

    Returns a stable *identifier* for the caller -- a short hash of the token --
    so a lease is tied to a token without the token ever being stored, logged,
    or compared in a query. Two workers share one token by design (there is one
    Jetson), so this identifies "the worker" rather than a specific machine; the
    lease is what serialises them.
    """
    if not configured_token():
        abort(503, description="the worker API is not enabled on this server")
    header = request.headers.get("Authorization", "")
    prefix = "Bearer "
    if not header.startswith(prefix):
        abort(401, description="missing bearer token")
    presented = header[len(prefix):].strip()
    if not token_ok(presented):
        abort(401, description="bad bearer token")
    return owner_id(presented)


def owner_id(token: str) -> str:
    """A short, stable, non-reversible name for a token.

    The full token must never be written to a row, a log, or an error -- a
    fingerprint is enough to tie a lease to a caller and cannot be replayed.
    """
    import hashlib

    return "w-" + hashlib.sha256(token.encode("utf-8")).hexdigest()[:16]


#: filenames a worker may upload. A closed set, checked by full match -- so
#: ``../modes.json``, ``/etc/passwd`` and ``mode_01.bin/../x`` are all refused
#: without any path sanitising.
#:
#: Full match, NOT ``match`` with a ``$`` anchor: Python's ``$`` also matches
#: just before a trailing newline, so ``"mode_01.bin\n"`` satisfied the anchored
#: pattern and became a second, distinct storage key for the same mode. The
#: ``\\Z`` in the pattern is belt-and-braces on top of ``fullmatch`` because this
#: one string is the whole traversal defence and a future edit that swaps
#: ``fullmatch`` for ``match`` should still not reopen it.
_ALLOWED_EXACT = ("modes.json", "surface.glb")
_ALLOWED_PATTERN = re.compile(r"\Amode_\d{2}\.(bin|vtu)\Z")

#: per-file ceiling. The field grid is what makes a .bin big; 64 MiB is well
#: over the largest the app has ever produced and well under what would let a
#: worker fill the bucket.
MAX_FILE_BYTES = 64 * 1024 * 1024


def allowed_name(name: str) -> bool:
    """Whether a worker may write a file called ``name``.

    Rejects anything with a path separator outright, before the allow-list is
    consulted, so a traversal attempt is refused even if it somehow matched a
    pattern later.

    ``fullmatch`` rather than ``match``: the pattern is ``\\A...\\Z`` anchored
    too, and this is the one call where "almost the right name" must never be
    good enough -- a name that differs from an allowed one by a trailing newline
    is a different storage key, and the two would then diverge silently.
    """
    if not name or "/" in name or "\\" in name or name in (".", ".."):
        return False
    if name in _ALLOWED_EXACT:
        return True
    return bool(_ALLOWED_PATTERN.fullmatch(name))


def register(app, jobs_db, store, log=print):
    """Attach the worker routes to ``app``.

    The collaborators are stashed on the app rather than closed over, and the
    routes are declared once at import time. A blueprint cannot have routes
    added after it has been registered *anywhere* -- Flask raises -- so a
    factory that built its routes inside this function could only ever be
    called once per process. Tests build many apps, and a second one used to
    fail with an assertion about setup being finished.
    """
    app.extensions.setdefault("mse", {})
    app.extensions["mse"]["worker_deps"] = {"jobs_db": jobs_db, "store": store,
                                            "log": log}


def _deps():
    """The collaborators for the app handling the current request."""
    from flask import current_app

    return current_app.extensions["mse"]["worker_deps"]


def _register_routes():
    """Declare every worker route on the blueprint. Called once, at import."""
    lease_seconds = CFG.lease_seconds()

    @bp.post("/worker/claim")
    def claim():
        """Hand out one job, or 204 when there is none.

        Atomic: ``claim_jetson`` uses ``FOR UPDATE SKIP LOCKED``, so two workers
        polling at once get different jobs or no job, never the same one.
        """
        owner = _require_worker()
        job = _deps()["jobs_db"].claim_jetson(owner, lease_seconds)
        if job is None:
            return ("", 204)
        _deps()["log"](f"[mse-web] worker {owner} claimed {job['job_id']}")
        return jsonify({
            "job_id": job["job_id"],
            "units": job["units"],
            "n_index": float(job["n_index"]),
            "n_modes": int(job["n_modes"]),
            # Swapped from the job row by SQL, one for one: a NULL integer is
            # not a value, so it stays JSON null.
            "min_feature_m": job.get("min_feature_m"),
            "scale": float(job.get("scale") or 1.0),
            # The element caps travel with the job. They are properties of the
            # MACHINES, not of this upload, but a worker that knows only the
            # row cannot learn them: it has no cluster access and no reason to
            # hold a copy of a constant that must equal the app's. Sending them
            # per claim is what keeps the solver's cap and the app's ceiling the
            # same number -- the defect this closes was the app approving an
            # upload the solver then refused at 80k.
            "max_tets": CFG.max_tets(),
            "cpu_max_tets": CFG.cpu_max_tets(),
            # WHICH lease this claim took. Sent for the same reason the caps
            # are: the worker cannot see the row, so a value the app chose and
            # the worker needs has to travel with the claim.
            #
            # The worker does NOT derive its beat from this. Earlier it beat at
            # ``this / 3``, which for the 300 s default is 100 s -- longer than
            # many real solves, so the first beat arrived after the work was
            # done and a long job looked to the app like a dead worker. The beat
            # interval is now the worker's own ``MSE_BEAT_SECONDS``. This value
            # stays in the payload because the worker checks its beat against it
            # and warns when the beat cannot keep the lease alive, which is a
            # real misconfiguration the worker alone cannot see.
            "lease_seconds": lease_seconds,
        })

    @bp.get("/worker/jobs/<job_id>/input.obj")
    def input_obj(job_id):
        """The uploaded mesh, for the worker that holds the lease."""
        owner = _require_worker()
        job = _held(_deps()["jobs_db"], job_id, owner)
        try:
            data = _deps()["store"].read(STORE.job_key(job["job_id"], "input.obj"))
        except STORE.StorageError:
            abort(404, description="the uploaded mesh is not in storage")
        return data, 200, {"Content-Type": "application/octet-stream"}

    @bp.post("/worker/jobs/<job_id>/progress")
    def progress(job_id):
        """A heartbeat, and the lease renewal.

        The worker's whole liveness contract is this call: a lease that is not
        renewed expires, and an expired lease is how the app detects a worker
        that died without saying so.
        """
        owner = _require_worker()
        body = request.get_json(silent=True) or {}
        if not _deps()["jobs_db"].renew_lease(job_id, owner, lease_seconds):
            abort(409, description="this worker no longer holds the lease")
        stage = str(body.get("stage", "running"))[:64]
        try:
            _deps()["jobs_db"].set_jetson_stage(job_id, stage)
        except AttributeError:
            pass
        return jsonify({"ok": True, "lease_seconds": lease_seconds})

    @bp.put("/worker/jobs/<job_id>/files/<name>")
    def put_file(job_id, name):
        """Upload one result file, if this worker holds the lease.

        The name is checked against a closed allow-list. The size is checked
        against ``MAX_FILE_BYTES`` *before* the body is read where the server
        can tell (Content-Length), and again after, because a chunked upload
        carries no length up front.
        """
        owner = _require_worker()
        _held(_deps()["jobs_db"], job_id, owner)
        if not allowed_name(name):
            abort(400, description=f"{name!r} is not a file this API accepts")
        declared = request.content_length
        if declared is not None and declared > MAX_FILE_BYTES:
            abort(413, description="file is larger than the limit")
        data = request.get_data(cache=False)
        if len(data) > MAX_FILE_BYTES:
            abort(413, description="file is larger than the limit")
        if not _deps()["jobs_db"].renew_lease(job_id, owner, lease_seconds):
            abort(409, description="this worker no longer holds the lease")
        try:
            _deps()["store"].write(STORE.job_key(job_id, name), data)
        except STORE.StorageError as exc:
            abort(500, description=f"could not store {name}: {exc}")
        return jsonify({"ok": True, "bytes": len(data)})

    @bp.post("/worker/jobs/<job_id>/done")
    def done(job_id):
        """Mark the job finished. Only the current lease holder may."""
        owner = _require_worker()
        job = _held(_deps()["jobs_db"], job_id, owner)
        _deps()["jobs_db"].finish_jetson(job["job_id"], owner)
        _deps()["log"](f"[mse-web] worker {owner} finished {job_id}")
        return jsonify({"ok": True})

    @bp.post("/worker/jobs/<job_id>/failed")
    def failed(job_id):
        """The worker is giving up on this job for good. TERMINAL.

        The job is marked FAILED with the worker's message, the lease is
        cleared, and it is NOT requeued. This is the only honest thing to do
        with a deterministic failure -- a mesh the mesher refuses, a box that is
        too detailed, a solver exception -- because requeueing it means it comes
        straight back to a worker that fails it the same way. Forever, with the
        reason visible only in a log the owner never reads.

        Transient reasons do NOT belong here: they go to ``/release``, which
        hands the job back with no verdict. This route used to release-and-
        requeue, which is why a deterministic failure looped; the fix is to make
        the two cases two verbs rather than one blurry one.
        """
        owner = _require_worker()
        body = request.get_json(silent=True) or {}
        message = str(body.get("message", ""))[:2000] or "the GPU worker failed"
        if not _deps()["jobs_db"].fail_jetson(job_id, owner, message):
            abort(409, description="this worker no longer holds the lease")
        _deps()["log"](f"[mse-web] worker {owner} failed {job_id}: {message}")
        return jsonify({"ok": True})

    @bp.post("/worker/jobs/<job_id>/release")
    def release(job_id):
        """Hand the job back with NO verdict, because the reason was transient.

        Requeues, so another worker (or this one, later) can take it, and the
        release is COUNTED. The bound is enforced inside ``release_jetson`` as
        one atomic statement rather than here: a read-count-then-decide form
        lets two concurrent releases both read 2, both requeue, and the job that
        should have failed instead goes round again. See that method's docstring.

        The response says which of the three things happened, so a worker (and a
        test) can tell "requeued" from "this was the last try and it is now
        terminal".
        """
        owner = _require_worker()
        body = request.get_json(silent=True) or {}
        reason = str(body.get("message", ""))[:2000]
        outcome = _deps()["jobs_db"].release_jetson(job_id, owner, reason)
        if outcome is None:
            abort(409, description="this worker no longer holds the lease")
        if outcome == "failed":
            _deps()["log"](f"[mse-web] {job_id} failed after {DB.MAX_RELEASES} "
                           f"releases" + (f": {reason}" if reason else ""))
            return jsonify({"ok": True, "released": False, "failed": True})
        _deps()["log"](f"[mse-web] worker {owner} released {job_id}"
                       + (f": {reason}" if reason else ""))
        return jsonify({"ok": True, "released": True, "failed": False})




_register_routes()


def register_blueprint(app) -> None:
    """Register the worker routes on ``app``.

    Separate from :func:`register` because Flask forbids adding routes to a
    blueprint after it has been registered anywhere in the process. The routes
    are therefore declared once, at import, and each app this factory builds
    gets that same blueprint.
    """
    app.register_blueprint(bp)


def _held(jobs_db, job_id, owner) -> dict:
    """The job, if ``owner`` holds a live lease on it; else abort.

    A 409 rather than a 403/404 split: the caller is authenticated and the job
    may well exist, it just is not theirs any more. The worker's correct
    response is to stop, not to retry with different credentials.
    """
    if not jobs_db.lease_held_by(job_id, owner):
        abort(409, description="this worker does not hold the lease on that job")
    job = jobs_db.get_job(job_id)
    if job is None:
        abort(404, description="no such job")
    return job
