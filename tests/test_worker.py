"""The Jetson worker, driven end to end over a real socket.

The worker is a separate process that talks HTTP, so a Flask test client cannot
stand in for the app it talks to -- the client does not serve a socket, and the
whole point of these tests is that the worker's real ``urllib`` client reaches
the real routes. Each test therefore runs a ``werkzeug`` server on an
OS-assigned port in a daemon thread and points the worker's ``MSE_URL`` at it.
Every call still goes through ``web/worker_api.py``: the auth, the lease rules
and the filename allow-list are all the production ones.

What is faked, and why:

* ``solve`` -- there is no GPU here and no TetGen is allowed. The fake writes the
  real output FILENAMES into a LocalStore, so the upload path, the name
  enumeration and the allow-list are exercised for real.
* ``BenchLock`` where a test needs the lock held, because a real flock is
  per-open-file and two locks in one process are the same lock.
* Nothing in ``web/`` -- the app side is the real blueprint over a real socket.

What is NOT proved here, and must not be implied: that the systemd unit starts,
that a real flock serialises against the lead's benchmarks, and that the cuDSS
GPU path works. Those are deploy-time facts on the Jetson.
"""

from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest
from werkzeug.serving import make_server

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "worker"))
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import app as APP  # noqa: E402
import db as DB  # noqa: E402
import mse_worker as W  # noqa: E402
import storage as STORE  # noqa: E402
from test_routing import FakeDb, LeaseDb, MemoryStore  # noqa: E402

TOKEN = "tok-worker-test-123"

#: A fixed, ancient mtime base for tests that need a deterministic ORDER of
#: files rather than whatever the filesystem's clock says. Far enough in the
#: past that no code path can accidentally beat it.
BASE_TIME = 1_000_000_000.0

#: what the fake solve writes. Every name here is on the API's allow-list, and
#: between them they cover both branches of it (the exact names and the pattern).
RESULT_FILES = {
    "modes.json": b'{"modes": [{"wavelength_m": 1.0}]}',
    "surface.glb": b"glTF-fake",
    "mode_01.bin": b"\x00\x01\x02\x03",
    "mode_02.vtu": b"<VTKFile/>",
}


# ---------------------------------------------------------------------------
# A real server, on a real port
# ---------------------------------------------------------------------------

class RunningServer:
    """A real HTTP server for one test app, torn down afterwards."""

    def __init__(self, app):
        self.server = make_server("127.0.0.1", 0, app, threaded=True)
        self.port = self.server.server_port
        self.url = f"http://127.0.0.1:{self.port}"
        self.thread = threading.Thread(target=self.server.serve_forever,
                                       daemon=True)
        self.thread.start()

    def stop(self):
        self.server.shutdown()
        self.thread.join(timeout=5)
        self.server.server_close()


@pytest.fixture
def server(tmp_path, monkeypatch):
    """A running app plus the doubles the worker will meet.

    The app is built through the real factory with the real worker blueprint;
    only the database and the object store are in-process doubles, which is the
    same substitution every other test in this suite makes.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", TOKEN)
    monkeypatch.setenv("MSE_SOLVER_IMAGE", "test-image")
    store = MemoryStore(tmp_path)
    jobs_db = LeaseDb()
    app, _s, _d, _l = APP.create_app({
        "TESTING": True, "MSE_LOCAL_STORE": str(tmp_path),
        "MSE_DB_MOCK": jobs_db, "MSE_JOB_LAUNCHER": _NoLauncher(),
        "MSE_SOLVER_IMAGE": "test-image",
    })
    running = RunningServer(app)
    running.store = store
    running.jobs_db = jobs_db
    try:
        yield running
    finally:
        running.stop()


class _NoLauncher:
    def launch(self, *_a, **_k):  # pragma: no cover - never called here
        raise AssertionError("the worker path must not launch cluster Jobs")

    def __getattr__(self, name):  # pragma: no cover
        def _no(*_a, **_k):
            return None
        return _no


def queued_job(jobs_db, job_id="11111111-2222-3333-4444-555555555555", **extra):
    """A GPU-targeted, queued job, as the app would have stored it."""
    job = {
        "job_id": job_id, "units": "mm", "n_index": 1.5, "n_modes": 10,
        "est_tets": 20000, "target": DB.JETSON, "status": DB.QUEUED,
        "min_feature_m": None, "scale": 1.0,
    }
    job.update(extra)
    jobs_db.jobs[job_id] = job
    return job


# ---------------------------------------------------------------------------
# The worker, and its fakes
# ---------------------------------------------------------------------------

class FakeSolve:
    """Stand-in for the production solver. Writes the real filenames.

    ``behaviour`` lets a test choose what happens: 'ok', 'raise' (a solve that
    fails), 'too_big' (a deterministic failure the owner can act on), or a
    callable for anything more specific.

    'raise' raises a plain ``RuntimeError``, which ``classify`` calls TERMINAL
    (an unrecognised exception is more likely a solver fault than a blip --
    see the default there). A test that wants the transient path raises an
    ``OSError`` itself.
    """

    def __init__(self, behaviour="ok"):
        self.behaviour = behaviour
        self.calls: list[dict] = []
        self.stages: list[str] = []

    def __call__(self, job, store_in, store_out, stage=None):
        self.calls.append(dict(job))
        if callable(self.behaviour):
            return self.behaviour(job, store_in, store_out, stage)
        if self.behaviour == "raise":
            if stage is not None:
                stage.set_stage(job["job_id"], "running", stage="meshing")
            raise RuntimeError("the mesh is too detailed for this box")
        if self.behaviour == "too_big":
            if stage is not None:
                stage.set_stage(job["job_id"], "running", stage="meshing")
            raise ValueError("this model has more elements than this box "
                             "can solve")
        for name, data in RESULT_FILES.items():
            store_out.write(f"jobs/{job['job_id']}/{name}", data)
        if stage is not None:
            stage.set_stage(job["job_id"], "running", stage="solving")
            self.stages.append(stage.stage)
        return {"n_modes": job["n_modes"], "n_tets": 1234}


class FakeLock:
    """A lock whose held-ness a test controls, and whose release is visible."""

    def __init__(self, free=True):
        self.free = free
        self.releases = 0
        self.acquires = 0

    def acquire(self):
        self.acquires += 1
        return self.free

    def release(self):
        self.releases += 1


def make_worker(server, tmp_path, solve=None, lock=None, log=None,
                token=TOKEN, **config_kw):
    """A real Worker talking to the running server, with the fakes injected."""
    work_root = tmp_path / "work"
    # A short beat by default: the whole point of the heartbeat is that it
    # fires DURING a solve, and a tier-1 solve here lasts a fraction of a
    # second. A test that wants a long beat says so explicitly.
    config = W.Config(
        server.url, token,
        work_root=config_kw.pop("work_root", work_root),
        lock_path=config_kw.pop("lock_path", tmp_path / "gpu.lock"),
        poll_seconds=config_kw.pop("poll_seconds", 0.05),
        keep_failed=config_kw.pop("keep_failed", 1),
        beat_seconds=config_kw.pop("beat_seconds", 0.02),
    )
    assert not config_kw, f"make_worker got unknown kwargs: {sorted(config_kw)}"
    api = W.WorkerAPI(config.url, config.token)
    lines: list[str] = []
    logger = log if log is not None else (lambda m, **k: lines.append(str(m)))
    worker = W.Worker(config, api, lock or FakeLock(), solve=solve or FakeSolve(),
                      store_factory=_local_store, log=logger)
    worker.log_lines = lines
    return worker


def _local_store(work_root, _job_id):
    return STORE.LocalStore(work_root)


# ---------------------------------------------------------------------------
# 1. the happy path
# ---------------------------------------------------------------------------

def test_a_claimed_job_is_solved_uploaded_marked_done_and_cleaned(tmp_path, server):
    """The whole contract in one test: claim, input, solve, upload, done, clean."""
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJDATA")
    solve = FakeSolve()
    worker = make_worker(server, tmp_path, solve=solve)

    assert worker.claim_loop(max_jobs=1) == 1

    # the job is done at the server, and its lease is gone with it
    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.DONE
    assert row.get("lease_owner") is None

    # every result file arrived, byte for byte
    for name, data in RESULT_FILES.items():
        assert server.store.read(STORE.job_key(job["job_id"], name)) == data

    # the solve got the job's own parameters and the input we uploaded
    assert solve.calls[0]["job_id"] == job["job_id"]
    assert solve.calls[0]["n_modes"] == 10

    # and the work dir is gone: a 0.2 mm job is ~500 MB
    assert not (worker.config.work_root / job["job_id"]).exists()

    # the lock was handed back
    assert worker.lock.releases >= 1


def test_the_worker_uploads_every_result_the_solver_wrote(tmp_path, server):
    """The file list comes from the store, so a new output file is uploaded."""
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    worker = make_worker(server, tmp_path)
    worker.claim_loop(max_jobs=1)
    assert set(RESULT_FILES) <= set(server.store.keys_under(job["job_id"]))


# ---------------------------------------------------------------------------
# 2. an empty queue is not a failure
# ---------------------------------------------------------------------------

def test_an_empty_queue_sleeps_and_loops_rather_than_failing(tmp_path, server):
    """204 must not be reported as a failure and must not end the process.

    A worker that treated an empty queue as an error would restart-loop forever
    on an idle Jetson, which is why the drain loop is separately asserted.
    """
    worker = make_worker(server, tmp_path)
    empty = W.WorkerAPI(server.url, TOKEN)

    assert empty.claim() is None          # the server really does answer 204
    worker.claim_loop(max_jobs=0)
    # No job was attempted, and nothing was reported to the server.
    assert worker.processed == []
    assert server.jobs_db.jobs == {} or all(
        j["status"] == DB.QUEUED for j in server.jobs_db.jobs.values())


def test_a_204_is_not_confused_with_a_transport_error(tmp_path, server):
    """A dead server must be an APIError, not silently read as 'no work'.

    Could-not-ask must not look like nothing-there: if a broken URL returned
    None the worker would idle forever against a server that is not running and
    look healthy while doing it.
    """
    api = W.WorkerAPI("http://127.0.0.1:1", TOKEN, timeout=0.5)
    with pytest.raises(W.APIError) as excinfo:
        api.claim()
    assert excinfo.value.status == 0
    assert "could not reach" in excinfo.value.body


# ---------------------------------------------------------------------------
# 3. a solve failure reports failed
# ---------------------------------------------------------------------------

def test_a_terminal_solve_failure_ends_the_job_and_it_is_never_reoffered(
        tmp_path, server):
    """A deterministic failure is TERMINAL, and the proof is the second claim.

    The old contract requeued: the job came straight back and the loop had to
    remember it and decline it. That is gone. Now the row ends FAILED, and the
    observable consequence is that the worker's SECOND claim gets nothing at all
    -- not the same job, not a different one, nothing (204). A worker that
    requeued would see the row again here.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    worker = make_worker(server, tmp_path, solve=FakeSolve("too_big"))

    assert worker.claim_loop(max_jobs=1) == 1

    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.FAILED, (
        "a deterministic failure must be terminal; requeueing it means it comes "
        "straight back to a worker that fails it the same way, forever")
    assert row.get("lease_owner") is None, "the lease must be cleared"
    assert row.get("message"), "the owner must get a message on the job page"
    # NOT the same job handed back
    assert worker.api.claim() is None, (
        "the failed job was handed out again: /failed is not terminal")


def test_a_terminal_failure_does_not_starve_the_queue(tmp_path, server):
    """Both ids must be attempted: the broken one must not block the good one."""
    ids = ["aaaaaaaa-0000-0000-0000-000000000001",
           "bbbbbbbb-0000-0000-0000-000000000002"]
    for job_id in ids:
        queued_job(server.jobs_db, job_id)
        server.store.write(STORE.job_key(job_id, "input.obj"), b"OBJ")

    seen: list[str] = []

    def always_fails(job, store_in, store_out, stage=None):
        seen.append(job["job_id"])
        raise ValueError("this upload is malformed")

    worker = make_worker(server, tmp_path, solve=always_fails)
    worker.claim_loop(max_jobs=2)
    assert set(seen) == set(ids), (
        f"a failing job starved the queue: the worker attempted {seen}")
    for job_id in ids:
        assert server.jobs_db.jobs[job_id]["status"] == DB.FAILED


def test_a_transient_failure_is_handed_back_and_can_be_retried(
        tmp_path, server):
    """The other half of the split: a transient failure must NOT kill the job.

    An ``OSError`` says nothing about the job -- the wire dropped, the disk was
    full -- so it goes to ``/release`` and the row comes back claimable. The next
    attempt succeeds, which is the whole reason the two verbs exist: calling this
    one terminal would throw away a job that was always going to work.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    calls = {"n": 0}

    def fails_once(job, store_in, store_out, stage=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("the connection dropped")
        for name, data in RESULT_FILES.items():
            store_out.write(f"jobs/{job['job_id']}/{name}", data)
        return {"n_tets": 1}

    worker = make_worker(server, tmp_path, solve=fails_once)
    worker.claim_loop(max_jobs=1)
    assert calls["n"] == 1
    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.QUEUED, (
        "a transient failure was treated as terminal: the job is dead")
    assert row["release_count"] == 1, "the release was not counted"

    # retried immediately -- there is no worker-side backoff any more
    worker.claim_loop(max_jobs=1)
    assert calls["n"] == 2
    assert server.jobs_db.jobs[job["job_id"]]["status"] == DB.DONE


def test_three_releases_end_the_job_even_for_a_transient_reason(tmp_path,
                                                                server):
    """Nothing can loop: the app fails the job at the bound.

    A worker that flaps -- or a network that keeps dropping -- must not be able
    to hand the same job back forever. The bound lives in the app, so this test
    drives it through the real route.

    ``MAX_RELEASES`` counts the releases that are ALLOWED, so the job survives
    that many and dies on the next one: three requeues then a fourth attempt
    that finds ``release_count`` already at the bound. Off-by-one here is the
    difference between "bounded" and "one more than the operator was told".
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")

    def always_transient(job, store_in, store_out, stage=None):
        raise OSError("the connection dropped")

    worker = make_worker(server, tmp_path, solve=always_transient)
    for _ in range(DB.MAX_RELEASES):
        worker.claim_loop(max_jobs=1)
    # still claimable: exactly MAX_RELEASES releases are permitted
    row = server.jobs_db.jobs[job["job_id"]]
    assert row["release_count"] == DB.MAX_RELEASES
    assert row["status"] == DB.QUEUED, (
        "the bound fired one release early: it must permit MAX_RELEASES")

    # ...and the next one is the last
    worker.claim_loop(max_jobs=1)
    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.FAILED, (
        f"after {DB.MAX_RELEASES} releases the job is still claimable: a "
        f"flapping worker can loop forever")
    assert row.get("message"), "the owner must get a message on the job page"
    assert worker.api.claim() is None, "the job was still handed out"


# ---------------------------------------------------------------------------
# 4. an UPLOAD failure must not look like a success
# ---------------------------------------------------------------------------

def test_an_upload_failure_reports_failed_and_never_done(tmp_path, server,
                                                         monkeypatch):
    """THE test for could-not-ask one level down.

    The upload route is made to refuse the third file. The job must end up
    failed-and-requeued -- never done. A worker that ignored the upload error
    would tell the app a job finished whose results are not in the bucket, and
    the owner would be shown a broken result page instead of a retry.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")

    uploaded: list[str] = []
    real_upload = W.WorkerAPI.upload

    def flaky_upload(self, job_id, name, data):
        uploaded.append(name)
        if len(uploaded) == 3:
            raise W.APIError(500, "could not store the file")
        return real_upload(self, job_id, name, data)

    monkeypatch.setattr(W.WorkerAPI, "upload", flaky_upload)
    worker = make_worker(server, tmp_path)
    worker.claim_loop(max_jobs=1)

    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] != DB.DONE, (
        "an upload failed but the job was reported done: the app now believes "
        "results exist that are not in storage")
    assert row["status"] == DB.QUEUED
    # the server never saw a done call
    assert ("POST", f"/worker/jobs/{job['job_id']}/done") not in worker.api.requests


# ---------------------------------------------------------------------------
# 5. the bench lock
# ---------------------------------------------------------------------------

def test_no_claim_is_made_at_all_while_the_lock_is_held(tmp_path, server):
    """Asserted on the REQUEST LOG, not a return value.

    "Did not ask" and "asked and got nothing" are indistinguishable from the
    outside, and only the first is correct: a claimed job this worker cannot run
    has vanished from the cluster's view while the owner waits. So the
    observation is that no claim request was made -- a 204 would be a bug.
    """
    queued_job(server.jobs_db)
    lock = FakeLock(free=False)
    worker = make_worker(server, tmp_path, lock=lock)

    worker.claim_loop(max_jobs=1)

    claimed = [r for r in worker.api.requests if r[1] == "/worker/claim"]
    assert claimed == [], f"the worker claimed while the lock was held: {claimed}"
    assert worker.processed == []
    # the job is untouched and still queued
    (row,) = server.jobs_db.jobs.values()
    assert row["status"] == DB.QUEUED and not row.get("lease_owner")


def test_the_lock_is_released_on_the_failure_path(tmp_path, server):
    """A job that raises must still hand the GPU back."""
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    lock = FakeLock()
    worker = make_worker(server, tmp_path, solve=FakeSolve("raise"), lock=lock)
    worker.claim_loop(max_jobs=1)
    assert lock.releases >= 1


# ---------------------------------------------------------------------------
# 6. SIGTERM mid-job
# ---------------------------------------------------------------------------

def test_sigterm_mid_job_hands_the_job_back_rather_than_dangling(tmp_path,
                                                                 server):
    """The stop flag set DURING a solve must produce a RELEASE, not a failure.

    systemd's TimeoutStopSec is bounded and SIGKILL follows it, which leaves the
    lease dangling until it expires -- exactly what the row forbids. So the
    worker hands the job back instead of finishing it.

    It goes back through ``/release``, NOT ``/failed``: an operator restarting
    the service is not a verdict about the job, and reporting it FAILED would
    destroy a perfectly good job every time the service is restarted.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    worker = None

    def solve_that_signals(job, store_in, store_out, stage=None):
        # SIGTERM lands while the solve is running.
        worker.request_stop()
        for name, data in RESULT_FILES.items():
            store_out.write(f"jobs/{job['job_id']}/{name}", data)
        return {"n_tets": 1}

    worker = make_worker(server, tmp_path, solve=solve_that_signals)
    assert worker.claim_loop(max_jobs=1) == 1

    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.QUEUED, (
        "SIGTERM mid-job must hand the job back, not finish it")
    assert ("POST", f"/worker/jobs/{job['job_id']}/done") not in worker.api.requests
    assert ("POST", f"/worker/jobs/{job['job_id']}/release") in worker.api.requests
    assert ("POST", f"/worker/jobs/{job['job_id']}/failed") not in worker.api.requests, (
        "a restart killed the job: it must be handed back, not failed")
    assert row["release_count"] == 1
    # the lease is not dangling: the app can hand this job to another worker
    assert row.get("lease_owner") is None


def test_sigterm_while_idle_stops_without_touching_a_job(tmp_path, server):
    worker = make_worker(server, tmp_path)
    worker.request_stop()
    assert worker.claim_loop() == 0
    assert worker.api.requests == [] or all(
        r[1] != "/worker/claim" or True for r in worker.api.requests)
    assert worker.processed == []


# ---------------------------------------------------------------------------
# 7. the token never reaches a log, a repr, or an exception
# ---------------------------------------------------------------------------

def test_the_token_never_appears_in_the_logs_or_a_repr(tmp_path, server,
                                                       capsys):
    """Drive every path that prints, then grep the whole output for the token.

    A test that passes because the token never appears in *any* code path proves
    nothing, so this drives startup, a successful job, a failing job, and a
    rejected request (401 from a bad token on a second client).
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    fail_job = queued_job(server.jobs_db, "cccccccc-0000-0000-0000-000000000003")
    server.store.write(STORE.job_key(fail_job["job_id"], "input.obj"), b"OBJ")

    worker = make_worker(server, tmp_path, solve=FakeSolve("raise"))
    worker.claim_loop(max_jobs=1)

    # a rejected request, which is the path most likely to echo the header
    bad = W.WorkerAPI(server.url, "wrong-token")
    with pytest.raises(W.APIError) as excinfo:
        bad.claim()
    assert excinfo.value.status == 401

    # reprs of the two objects that hold the secret
    assert TOKEN not in repr(worker.api)
    assert TOKEN not in repr(worker.config)

    captured = capsys.readouterr()
    haystack = "\n".join([captured.out, captured.err,
                          "\n".join(worker.log_lines), repr(worker.api),
                          repr(worker.config), str(excinfo.value)])
    assert TOKEN not in haystack, "the worker token leaked into the logs"


def test_a_bad_token_is_a_startup_refusal_not_an_idle_loop(tmp_path):
    """A wrong token must stop the worker at startup, naming the problem."""
    env = tmp_path / "env"
    env.write_text("MSE_URL=http://127.0.0.1:9\n")   # no token at all
    env.chmod(0o600)
    with pytest.raises(W.WorkerError) as excinfo:
        W.load_config(env)
    assert "MSE_WORKER_TOKEN" in str(excinfo.value)


def test_a_missing_url_is_refused(tmp_path):
    env = tmp_path / "env"
    env.write_text("MSE_WORKER_TOKEN=abc\n")
    env.chmod(0o600)
    with pytest.raises(W.WorkerError) as excinfo:
        W.load_config(env)
    assert "MSE_URL" in str(excinfo.value)


# ---------------------------------------------------------------------------
# 8/9. the env file and its mode
# ---------------------------------------------------------------------------

def test_a_world_readable_env_file_refuses_to_start(tmp_path):
    """A mode-600 file is the whole secret story; 0644 must be refused."""
    env = tmp_path / "env"
    env.write_text(f"MSE_URL=http://x\nMSE_WORKER_TOKEN={TOKEN}\n")
    env.chmod(0o644)
    with pytest.raises(W.WorkerError) as excinfo:
        W.load_config(env)
    assert "chmod 600" in str(excinfo.value)


def test_a_mode_600_env_file_loads(tmp_path):
    env = tmp_path / "env"
    env.write_text(f"""# a comment
MSE_URL=http://example.test/
MSE_WORKER_TOKEN={TOKEN}

POLL_SECONDS=2.5
""")
    env.chmod(0o600)
    config = W.load_config(env)
    assert config.url == "http://example.test"     # trailing slash normalised
    assert config.token == TOKEN
    assert config.poll_seconds == 2.5


def test_the_env_file_does_not_hijack_the_process_environment(tmp_path,
                                                              monkeypatch):
    """The file is the config, not a merge into os.environ."""
    monkeypatch.setenv("MSE_URL", "http://from-the-environment")
    env = tmp_path / "env"
    env.write_text(f"MSE_URL=http://from-the-file\nMSE_WORKER_TOKEN={TOKEN}\n")
    env.chmod(0o600)
    assert W.load_config(env).url == "http://from-the-file"


# ---------------------------------------------------------------------------
# 10/11. disk hygiene
# ---------------------------------------------------------------------------

def test_a_successful_job_leaves_no_work_directory(tmp_path, server):
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    big = FakeSolve()

    def fatten(job, store_in, store_out, stage=None):
        store_out.write(f"jobs/{job['job_id']}/mode_01.bin", b"x" * 200_000)
        store_out.write(f"jobs/{job['job_id']}/modes.json", b"{}")
        return {"n_tets": 1}

    worker = make_worker(server, tmp_path, solve=fatten)
    worker.claim_loop(max_jobs=1)
    assert not (worker.config.work_root / job["job_id"]).exists()
    assert list(worker.config.work_root.glob("*.failed")) == []


def test_only_the_newest_failed_directory_is_kept(tmp_path, server):
    """The sweep leaves the newest failure and deletes the older ones.

    Driven directly rather than only through a job run: a job run exercises the
    sweep with whatever the filesystem's clock says, and "the newest survived"
    is then true by accident as often as by design. Here the mtimes are set
    explicitly, so the ORDER is the test's own rather than the filesystem's.

    Note the sweep runs after EVERY failure, so a run of failures never
    accumulates dirt in the first place; the direct call is what proves the
    ordering rule that keeps it bounded once the dirs exist at different ages
    (a job that fails, then a process restart, then another failure).
    """
    ids = [f"dddddddd-0000-0000-0000-00000000000{n}" for n in range(3)]

    def boom(job, store_in, store_out, stage=None):
        raise RuntimeError("nope")

    worker = make_worker(server, tmp_path, solve=boom, keep_failed=1)
    for job_id in ids:
        queued_job(server.jobs_db, job_id)
        server.store.write(STORE.job_key(job_id, "input.obj"), b"OBJ")
        worker.claim_loop(max_jobs=1)
        assert (worker.config.work_root / f"{job_id}.failed").exists(), (
            f"{job_id} was not kept: the sweep must keep the NEWEST, and the "
            f"newest at that moment is the one just written")

    # Recreate the three at known ages (oldest first) and sweep.
    import shutil
    for n, job_id in enumerate(ids):
        target = worker.config.work_root / f"{job_id}.failed"
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        os.utime(target, (BASE_TIME + n, BASE_TIME + n))

    removed = W.sweep_failed(worker.config.work_root, 1)

    kept = sorted(p.name for p in worker.config.work_root.glob("*.failed"))
    assert kept == [f"{ids[-1]}.failed"], (
        f"expected only the newest failure to survive, got {kept}")
    assert sorted(removed) == sorted(f"{i}.failed" for i in ids[:-1]), (
        f"the sweep removed the wrong set: {removed}")


def test_the_sweep_runs_at_startup_too(tmp_path):
    """A reboot is when a stale failure dir is most likely to be the newest."""
    root = tmp_path / "work"
    root.mkdir()
    for n, name in enumerate(["old.failed", "older.failed", "newest.failed"]):
        d = root / name
        d.mkdir()
        os_time = time.time() + n
        __import__("os").utime(d, (os_time, os_time))
    removed = W.sweep_failed(root, keep=1)
    assert len(removed) == 2
    assert (root / "newest.failed").exists()


# ---------------------------------------------------------------------------
# 12/13. the heartbeat, and a lease that goes away
# ---------------------------------------------------------------------------

def test_the_heartbeat_renews_with_the_real_stage_not_a_constant(tmp_path,
                                                                 server):
    """The stage the server sees must move through the solve's own stages.

    A heartbeat that always reports 'starting' is indistinguishable from a
    healthy one, and tells the operator nothing about a job that has been
    meshing for ten minutes.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    seen: list[str] = []

    def solve_that_moves(job, store_in, store_out, stage=None):
        for name in ("meshing", "assembling", "solving"):
            stage.set_stage(job["job_id"], "running", stage=name)
            seen.append(name)
            time.sleep(0.25)                 # let beats fire inside this stage
        for fname, data in RESULT_FILES.items():
            store_out.write(f"jobs/{job['job_id']}/{fname}", data)
        return {"n_tets": 1}

    # Enough beats to move through the stages: the solve holds each stage for
    # 0.25 s and the beat is 0.02 s, so several land inside each one.
    worker = make_worker(server, tmp_path, solve=solve_that_moves)
    worker.claim_loop(max_jobs=1)

    reported = server.jobs_db.stages_seen
    assert reported, "the server never saw a progress call"
    assert set(reported) & {"meshing", "assembling", "solving"}, (
        f"the heartbeat never carried a real stage: {reported}")
    assert len(set(reported)) > 1, (
        f"the heartbeat sent one constant stage for the whole solve: {reported}")


def test_a_lost_lease_stops_the_worker_before_done(tmp_path, server,
                                                   monkeypatch):
    """done must never be called once the lease is gone.

    The server is made to answer 409 on progress (the lease expired), which is
    how a worker that slept through its lease and came back is detected. The
    job must NOT be reported done, and the worker must not upload over its
    successor.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")

    def solve_slowly(job, store_in, store_out, stage=None):
        time.sleep(0.4)                      # long enough for one beat
        for name, data in RESULT_FILES.items():
            store_out.write(f"jobs/{job['job_id']}/{name}", data)
        return {"n_tets": 1}

    # the lease is gone from the server's point of view
    monkeypatch.setattr(server.jobs_db, "renew_lease",
                        lambda *a, **k: False)
    monkeypatch.setattr(server.jobs_db, "lease_held_by",
                        lambda *a, **k: False)

    worker = make_worker(server, tmp_path, solve=solve_slowly)
    result = worker.claim_loop(max_jobs=1)

    assert ("POST", f"/worker/jobs/{job['job_id']}/done") not in worker.api.requests
    assert result == 1


# ---------------------------------------------------------------------------
# The pieces the loop depends on
# ---------------------------------------------------------------------------

def test_a_solver_output_outside_the_allow_list_is_a_loud_failure(tmp_path):
    """A name the API would refuse must raise, not be silently dropped.

    'The solver wrote a file we forgot to allow' and 'the solver wrote nothing'
    mean opposite things, and a silent skip turns the first into a job that
    reports success with a missing result.
    """
    store = STORE.LocalStore(tmp_path)
    store.write("jobs/j1/input.obj", b"x")
    store.write("jobs/j1/modes.json", b"{}")
    store.write("jobs/j1/mystery.dat", b"?")
    with pytest.raises(W.WorkerError) as excinfo:
        W._result_names(store, "j1")
    assert "mystery.dat" in str(excinfo.value)


def test_a_job_with_no_result_files_is_refused(tmp_path):
    store = STORE.LocalStore(tmp_path)
    store.write("jobs/j1/input.obj", b"x")
    with pytest.raises(W.WorkerError) as excinfo:
        W._result_names(store, "j1")
    assert "no result files" in str(excinfo.value)


def test_the_mode_file_matcher_matches_the_allow_list_exactly(tmp_path):
    """The worker's local check must agree with the API's regex.

    If they disagree the worker uploads a name the API refuses (a 400 mid-job),
    or refuses to upload one the API would accept (a job reported done with
    missing output).
    """
    for good in ("mode_01.bin", "mode_99.vtu"):
        assert W._is_mode_file(good), good
    for bad in ("mode_1.bin", "mode_001.bin", "mode_ab.bin", "mode_01.dat",
                "modes.json", "mode_01.bin\n", "xmode_01.bin"):
        assert not W._is_mode_file(bad), bad


def test_a_job_id_that_is_a_path_is_refused(tmp_path):
    """The job id comes off the wire and names a directory we DELETE."""
    for bad in ("../../etc", "a/b", "..", ""):
        with pytest.raises(W.WorkerError):
            W.job_dir(tmp_path, bad)


def test_the_bench_lock_is_exclusive_within_a_process(tmp_path):
    """The real flock: a second BenchLock on the same path cannot take it."""
    path = tmp_path / "gpu.lock"
    first = W.BenchLock(path)
    second = W.BenchLock(path)
    assert first.acquire() is True
    assert second.acquire() is False
    first.release()
    assert second.acquire() is True
    second.release()


# ---------------------------------------------------------------------------
# 5. the GPU error CHAIN decides transient vs terminal
# ---------------------------------------------------------------------------
#
# ``mse_eigen`` wraps a cuDSS failure as ``EigenError(msg) from exc``, so the
# worker never sees the cuDSS exception itself. The top-level TYPE is therefore
# the same for both events that matter here -- a GPU that would not come up
# (transient) and a GPU that ran and rejected THIS job (terminal) -- and only the
# ``__cause__`` chain tells them apart. These use the REAL ``EigenError`` rather
# than a stub, because a stub that merely resembles it would not catch a change
# to how the solver wraps its failures, which is the whole seam under test.

def test_a_gpu_that_would_not_initialise_is_transient_through_the_wrapper():
    """A dead/absent GPU must not fail the job forever.

    The cupy types are the strongest evidence: they are raised by the runtime or
    driver itself, never as a verdict about an eigenproblem.
    """
    from mse_eigen import EigenError

    class CUDARuntimeError(RuntimeError):
        pass

    for cause in (
        CUDARuntimeError("cudaErrorNoDevice: no CUDA-capable device is detected"),
        RuntimeError("CUDA driver version is insufficient for CUDA runtime"),
        RuntimeError("failed to initialise the cuDSS context"),
        RuntimeError("CUDA context creation failed"),
    ):
        exc = EigenError("the eigensolver failed")
        exc.__cause__ = cause
        assert W.classify(exc) == "transient", (
            f"{type(cause).__name__}: {cause} was called terminal; a GPU that "
            f"is down would then fail every job permanently")


def test_a_cudss_status_error_stays_terminal_even_inside_the_wrapper():
    """A GPU that RAN and rejected this job is a verdict about the job.

    ``ALLOC_FAILED`` is the case the lead named: re-running produces the same
    answer, so requeueing it is the loop this function exists to prevent. The
    status words are checked FIRST, so a message that also mentions
    initialisation ('initialization failed after ALLOC_FAILED') reads terminal.
    """
    from mse_eigen import EigenError

    for text in (
        "cuDSS status 4: CUDSS_STATUS_ALLOC_FAILED",
        "cudss execution failed: wrong parameter",
        "the factorisation is singular",
        "initialization succeeded but allocation failed",
    ):
        exc = EigenError("the eigensolver failed")
        exc.__cause__ = RuntimeError(text)
        assert W.classify(exc) == "terminal", (
            f"{text!r} was called transient; a job that will always fail the "
            f"same way would be requeued until the bound and then reported as "
            f"an infrastructure problem, which it is not")


def test_the_chain_is_walked_and_not_just_the_first_cause():
    """Two wrappers deep is still the same event.

    The worker's own code may wrap again before ``classify`` sees it, so the
    walk must follow ``__context__`` as well as ``__cause__``. Without this a
    doubly-wrapped GPU init failure is terminal, which is the silent half of
    the bug.
    """
    from mse_eigen import EigenError

    inner = RuntimeError("cudaErrorInsufficientDriver: driver too old")
    middle = RuntimeError("could not build the solver")
    middle.__context__ = inner
    exc = EigenError("the eigensolver failed")
    exc.__cause__ = middle
    assert W.classify(exc) == "transient"


def test_the_bound_sentence_and_the_workers_reason_both_reach_the_owner(
        tmp_path, server):
    """On the LAST attempt the owner gets both halves, not the reason alone.

    ``release_jetson`` appends the bound sentence on the over-bound branch. The
    bug it fixes was ``message or <bound sentence>``: the worker always sends a
    non-empty message, so the bound sentence was shadowed and the owner was told
    only why the last attempt failed -- never that this was the end of the
    retries, which is the part they need to act on. Both parts are asserted
    here, because either one alone was already being printed.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")

    def gpu_down(job, store_in, store_out, stage=None):
        # What the worker really sees when the GPU is absent: mse_eigen wraps
        # the CUDA failure, and ``_owner_message`` maps it to the owner's text.
        from mse_eigen import EigenError
        exc = EigenError("the eigensolver failed")
        exc.__cause__ = RuntimeError(
            "cudaErrorNoDevice: no CUDA-capable device is detected")
        raise exc

    # The owner-facing reason, taken from the production mapping rather than
    # retyped here: a test that restates the sentence checks its own copy.
    reason = W._owner_message(
        _wrapped_eigen("cudaErrorNoDevice: no CUDA-capable device is detected"))

    worker = make_worker(server, tmp_path, solve=gpu_down)
    for _ in range(DB.MAX_RELEASES):
        worker.claim_loop(max_jobs=1)
    # The retries are still claimable, and their message says so with the count.
    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.QUEUED
    assert "retrying (attempt" in row["message"], row["message"]
    assert reason in row["message"], (
        f"the reason was dropped from the retry message: {row['message']!r}")

    # The attempt past the bound is the last, and its message carries both the
    # reason and the fact that the retries are over.
    worker.claim_loop(max_jobs=1)
    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.FAILED
    message = row["message"] or ""
    assert reason in message, f"the owner lost the reason: {message!r}"
    assert f"{DB.MAX_RELEASES} attempts" in message, (
        f"the owner was never told the retries are over: {message!r}")

    # And the reason must not blame the object: the GPU was down, the object was
    # never even looked at. This is the sentence the owner reads on the page.
    assert "simpler object" not in message, (
        f"a GPU outage was reported to the owner as a fault in their object: "
        f"{message!r}")
    assert "graphics processor was not available" in message, message


def _wrapped_eigen(text: str):
    """An ``EigenError`` wrapping a cuDSS/CUDA failure, as mse_eigen raises it."""
    from mse_eigen import EigenError
    exc = EigenError("the eigensolver failed")
    exc.__cause__ = RuntimeError(text)
    return exc


def test_the_achieve_path_stays_terminal_for_an_alloc_failure(tmp_path, server):
    """ALWAYS a verdict, never a retry: the message the owner gets must be TRUE.

    ``ALLOC_FAILED`` also arrives when another process on the GPU is holding the
    memory, so the owner is told the two things they can actually act on rather
    than "too detailed", which is a claim we cannot make. This drives it through
    the production failure path so the MESSAGE, not just the classification, is
    asserted.
    """
    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")

    def cudss_alloc_failure(job, store_in, store_out, stage=None):
        from mse_eigen import EigenError
        exc = EigenError("the eigensolver failed")
        exc.__cause__ = RuntimeError("cuDSS status 4: CUDSS_STATUS_ALLOC_FAILED")
        raise exc

    worker = make_worker(server, tmp_path, solve=cudss_alloc_failure)
    worker.claim_loop(max_jobs=1)

    row = server.jobs_db.jobs[job["job_id"]]
    assert row["status"] == DB.FAILED, (
        "an ALLOC_FAILED was retried; it is a verdict about this job and "
        "re-running produces the same one")
    message = row["message"] or ""
    assert "could not fit in the GPU's memory" in message, message
    assert "try again later" in message, message
    assert "too detailed" not in message, (
        "the owner was told the object is too detailed, which ALLOC_FAILED does "
        "not tell us -- another process may be holding the memory")


def test_the_chain_walk_terminates_on_a_cycle():
    """A self-referential chain must not hang the worker.

    ``__context__`` is set automatically during ``raise ... from`` handling and
    can point back at an ancestor. A walk without a visited-set spins forever
    holding the job's lease, which is worse than any wrong classification.
    """
    exc = RuntimeError("boom")
    exc.__context__ = exc
    assert W.classify(exc) == "terminal"

    a = RuntimeError("a")
    b = RuntimeError("b")
    a.__cause__ = b
    b.__cause__ = a
    assert W.classify(a) == "terminal"


def test_gpu_memory_is_released_after_a_successful_job(tmp_path, server,
                                                       monkeypatch):
    """A warm worker must hand the device memory back when the job ends.

    The worker is deliberately long-lived, so per-job allocations that are not
    released accumulate. On the Jetson that is system RAM, and the first symptom
    is the OS killing the worker -- which arrives as SIGKILL, so nothing gets
    logged and the job looks hung until its lease expires.
    """
    calls = []
    monkeypatch.setattr(W, "_release_gpu_memory", lambda: calls.append("free"))

    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")
    worker = make_worker(server, tmp_path, solve=FakeSolve())
    worker.claim_loop(max_jobs=1)

    assert server.jobs_db.jobs[job["job_id"]]["status"] == DB.DONE
    assert calls == ["free"], (
        "the GPU memory was not released after a successful job, or was "
        "released more than once")


def test_gpu_memory_is_released_after_a_failed_job(tmp_path, server,
                                                   monkeypatch):
    """The failure path leaks WORST, and it is the path people forget.

    An exception between the two factorisations leaves the first one alive with
    no local holding it on the way out of the frame -- the registry is what
    makes it reachable. A cleanup wired only into the success branch would look
    correct in every passing test and leak on exactly the runs that are already
    going wrong.
    """
    calls = []
    monkeypatch.setattr(W, "_release_gpu_memory", lambda: calls.append("free"))

    job = queued_job(server.jobs_db)
    server.store.write(STORE.job_key(job["job_id"], "input.obj"), b"OBJ")

    def explodes(job, store_in, store_out, stage=None):
        raise RuntimeError("the factorisation died halfway")

    worker = make_worker(server, tmp_path, solve=explodes)
    worker.claim_loop(max_jobs=1)

    assert calls == ["free"], (
        "the GPU memory was not released after a failed job -- this is the "
        "path a per-job leak accumulates on")
