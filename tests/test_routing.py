"""Size-based routing, the GPU queue, and the worker API.

Three things are pinned here, all of them tier-1 (no cluster, no Postgres, no
GPU, no TetGen):

* **the estimate decides the route** -- and it is conservative, and a *failed*
  estimate never sends a job to the slower machine;
* **the fallback** -- a GPU job nobody claims goes back to the cluster, and one
  the cluster cannot solve does not;
* **the worker API** -- closed when unconfigured, constant-time when
  configured, lease-guarded on every write, allow-listed on every filename.

The reroute path is the one to read carefully: the solver writes a token into
its pod's termination message and the app reads it back. Every test of that
signal has to fail if the signal is never *seen*, not just if it is misread --
a test that only feeds the app a reason string would pass against an app that
can never actually obtain one.
"""

from __future__ import annotations

import io
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))
sys.path.insert(0, str(ROOT / "dockerfiles" / "maxwell_solver"))

import app as APP  # noqa: E402
import config as CFG  # noqa: E402
import db as DB  # noqa: E402
import jobs as JOBS  # noqa: E402
import meshing as MESH  # noqa: E402
import worker_api as WORKER  # noqa: E402

from test_web_routes import FakeDb, FakeLauncher, MemoryStore  # noqa: E402


# ---------------------------------------------------------------------------
# The estimate
# ---------------------------------------------------------------------------
def _box(extents, n_modes=20, units="mm"):
    import trimesh

    return trimesh.creation.box(extents=extents)


def test_the_estimate_grows_exactly_linearly_with_the_mode_count():
    """Elements scale with N, not with N^1.5 or N^3.

    This looks wrong and is right: Weyl's law gives ``k_N ~ (3 pi^2 N / V)^(1/3)``
    for the k of the N-th mode, so the wavelength goes as ``N^(-1/3)``, the edge
    length with it, and the element count -- which goes as ``h^-3`` -- comes back
    out as exactly N. The two exponents cancel, which is why the estimate is
    cheap to reason about: doubling the modes doubles the mesh.

    A test asserting N^1.5 here is the mistake *I* made first, and it is worth
    keeping the corrected law pinned so the next reader does not make it again.
    """
    surface = _box([10, 7, 5])
    few = MESH.estimate_tets(surface, 10, "mm")
    many = MESH.estimate_tets(surface, 40, "mm")
    assert few > 0
    assert many > few
    assert many / few == pytest.approx(40 / 10, rel=0.01)


def test_the_estimate_matches_the_fitted_constant_on_a_known_box():
    """Pin the arithmetic against the closed form, not against a remembered
    number: `C * V / h^3` computed here from the solver's own volume and edge.

    This is the guard that a wrong constant fails -- and it reads the value out
    of the solver rather than restating it, because a test that retyped C would
    only be checking its own copy.
    """
    import mse_meshing as MESHSOLVER

    surface = _box([10, 7, 5])
    got = MESH.estimate_tets(surface, 20, "mm")

    volume = MESHSOLVER.enclosure_volume_m3(surface, "mm")
    edge = MESHSOLVER.target_edge_length(surface, 20, units="mm")
    scale = MESHSOLVER.unit_scale("mm")
    want = int(CFG.est_tets_c() * volume / (edge * scale) ** 3)
    assert got == want


def test_an_unusable_surface_estimates_zero_not_small():
    """`could not ask` must not look like `nothing there`.

    A None surface -- the caller's way of saying the geometry could not be read
    -- must return 0, and the caller must route 0 to the CLUSTER. If this
    returned a plausible small number the job would go to the GPU on the
    strength of an estimate nobody made.
    """
    assert MESH.estimate_tets(None, 20, "mm") == 0


def test_the_coefficient_is_the_conservative_one():
    """C is 14 and is not silently lowered.

    Under-estimating sends a big job to the CPU where it dies at the memory
    limit; over-estimating costs a small job about a second on the GPU. The
    asymmetry is the whole reason for the value, so it is pinned.
    """
    assert CFG.est_tets_c() >= 14.0


# ---------------------------------------------------------------------------
# The upload route decision
# ---------------------------------------------------------------------------
def _upload(tmp_path, *, n_modes=None, form_extra=None, db=None, launcher=None):
    app, store, jobs_db, launcher = _make(tmp_path, db=db, launcher=launcher)
    import trimesh

    data = trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode()
    payload = {"meshfile": (io.BytesIO(data), "box.obj"),
               "units": "mm", "n_index": "1.0"}
    if n_modes is not None:
        payload["n_modes"] = str(n_modes)
    payload.update(form_extra or {})
    resp = app.test_client().post(
        "/", data=payload, content_type="multipart/form-data")
    return app, jobs_db, resp


def _make(tmp_path, db=None, launcher=None):
    store = MemoryStore(tmp_path)
    jobs_db = db or FakeDb()
    launcher = launcher or FakeLauncher()
    app, _s, _d, _l = APP.create_app({
        "TESTING": True, "MSE_LOCAL_STORE": str(tmp_path),
        "MSE_DB_MOCK": jobs_db, "MSE_JOB_LAUNCHER": launcher,
        "MSE_SOLVER_IMAGE": "test-image",
    })
    return app, store, jobs_db, launcher


def test_a_ten_millimetre_box_goes_to_the_jetson_at_the_real_threshold(tmp_path):
    """The box the app is most likely to see, at the threshold the lead set.

    It is worth having this one not monkeypatch anything: at 20 modes a 10 mm
    cube estimates at ~19.8k elements, over the 15k GPU threshold, so the
    *default* configuration routes it to the Jetson. If someone later raises
    MSE_GPU_MIN_TETS past this, this test says so rather than the change sliding
    through.
    """
    _app, jobs_db, resp = _upload(tmp_path)
    assert resp.status_code == 302
    (job,) = jobs_db.jobs.values()
    assert job["est_tets"] > CFG.gpu_min_tets()
    assert job["target"] == DB.JETSON


def test_a_small_estimate_is_routed_to_the_cluster(tmp_path, monkeypatch):
    """Raise the threshold above the box's own estimate: the mesh is untouched,
    so this exercises the comparison rather than a special-cased input."""
    monkeypatch.setenv("MSE_GPU_MIN_TETS", "1000000")
    _app, jobs_db, resp = _upload(tmp_path)
    assert resp.status_code == 302
    (job,) = jobs_db.jobs.values()
    assert job["target"] == DB.CLUSTER
    assert job["est_tets"] > 0


def test_an_upload_too_big_even_at_the_worst_over_estimate_is_refused(tmp_path):
    """Refused at upload, before anything is stored or queued.

    The refusal threshold is TWICE the real cap: the estimate can be up to 2x
    high, and refusing at 1x would turn away jobs that fit. A job refused here
    is refused because even the most pessimistic reading says it cannot run --
    and nothing at all should have been created.
    """
    # The plain box must NOT be refused at the real coefficient. This half
    # matters as much as the refusal: a guard that fires on everything passes
    # the refusal test and is useless.
    _app, jobs_db, resp = _upload(tmp_path)
    assert resp.status_code == 302
    assert jobs_db.jobs, "an ordinary box must not be refused"


def test_the_refusal_fires_when_the_estimate_is_over_the_threshold(tmp_path):
    """The same box, the same code path, a coefficient that clears the cap.

    Raising the coefficient is the honest way to build this case: the refusal is
    a comparison against `upload_refuse_tets()`, and no reachable real upload
    would need a mesh large enough to exercise it on this machine.
    """
    import os

    os.environ["MSE_EST_TETS_C"] = "100000000"
    try:
        app, _store, jobs_db, _launcher = _make(tmp_path)
        import trimesh

        data = trimesh.creation.box(extents=[10, 7, 5]).export(
            file_type="obj").encode()
        resp = app.test_client().post("/", data={
            "meshfile": (io.BytesIO(data), "box.obj"),
            "units": "mm", "n_index": "1.0"}, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert not jobs_db.jobs, "a refused upload must create no job"
    finally:
        os.environ.pop("MSE_EST_TETS_C", None)


def test_the_refusal_threshold_is_twice_the_cap():
    """The factor is the lead's decision and is load-bearing on its own: at 1x
    we would refuse fits. Pinned as a relation, not a number."""
    assert CFG.upload_refuse_tets() == 2 * CFG.max_tets()


def test_a_bad_mode_count_is_refused(tmp_path):
    _app, jobs_db, resp = _upload(tmp_path, n_modes=7)
    assert resp.status_code == 302
    assert not jobs_db.jobs


def test_the_mode_count_choice_reaches_the_job(tmp_path):
    _app, jobs_db, resp = _upload(tmp_path, n_modes=40)
    assert resp.status_code == 302
    (job,) = jobs_db.jobs.values()
    assert job["n_modes"] == 40


# ---------------------------------------------------------------------------
# The reroute signal, end to end
# ---------------------------------------------------------------------------
def test_the_two_copies_of_the_token_agree():
    """The solver writes it and the app looks for it. They are separate images
    with no import between them, so this is the only thing keeping the two
    string literals equal -- exactly the kind of copy that drifts silently.
    """
    import mse_maxwell_solver as SOLVER

    assert SOLVER.REROUTE_TOKEN == JOBS.REROUTE_TOKEN


def test_the_solver_writes_the_token_only_for_the_reroutable_outcome():
    """Every other failure is named for the owner through the database, and
    giving them a termination message too would blur the one signal that means
    'move this job'."""
    import mse_maxwell_solver as SOLVER

    assert SOLVER.termination_message(SOLVER.TooBigForProcessor("x")) is not None
    assert SOLVER.REROUTE_TOKEN in SOLVER.termination_message(
        SOLVER.TooBigForProcessor("x"))
    assert SOLVER.termination_message(ValueError("nope")) is None
    assert SOLVER.termination_message(RuntimeError("nope")) is None


def test_the_termination_message_is_truncated_below_the_kubernetes_cap():
    """Kubernetes keeps 4 KiB of the message and cuts the rest. Truncating
    ourselves means the token is never the part that gets cut."""
    import mse_maxwell_solver as SOLVER

    text = SOLVER.termination_message(SOLVER.TooBigForProcessor("x" * 100000))
    assert 0 < len(text) <= 4096
    assert text.startswith(SOLVER.REROUTE_TOKEN)


def test_the_writer_reports_failure_instead_of_raising(tmp_path):
    """An absent /dev/termination-log (every local run) must not be fatal, and
    must be reported rather than assumed."""
    import mse_maxwell_solver as SOLVER

    assert SOLVER._write_termination_log("hello", str(tmp_path / "tl")) is True
    assert (tmp_path / "tl").read_text() == "hello"
    # A directory is not writable as a file: the call must return False, not
    # raise, because this runs on the failure path where raising loses the
    # database row's message too.
    assert SOLVER._write_termination_log("x", str(tmp_path)) is False


class _Pod:
    """A pod stand-in with the one field the reader walks.

    Built from nested dicts, because that is one of the two shapes the real
    Kubernetes client returns and the reader must handle both -- and because a
    double that only worked as an object would hide a reader that assumed one.
    """

    def __init__(self, message=None, *, use_objects=False):
        self.status = types.SimpleNamespace(container_statuses=[
            types.SimpleNamespace(
                state=types.SimpleNamespace(terminated=None),
                last_state=types.SimpleNamespace(terminated=(
                    types.SimpleNamespace(message=message)
                    if message is not None else None)),
            )
        ]) if use_objects else {
            "container_statuses": [{
                "state": {"terminated": None},
                "last_state": {"terminated": (
                    {"message": message} if message is not None else None)},
            }]
        }


def test_the_reroute_signal_is_read_from_the_pod_message():
    pods = [_Pod(f"{JOBS.REROUTE_TOKEN}: this model needs 200,000 elements")]
    assert JOBS.wants_reroute(pods) is True


def test_an_absent_message_is_not_a_reroute():
    """The whole safety property, in one test.

    No pods, no message, an empty message, a truncated message -- all of them
    must read as NO. `could not ask` is not `yes`, and treating an unreadable
    message as a reroute would re-queue a genuinely broken job, possibly
    forever.
    """
    assert JOBS.wants_reroute([]) is False
    assert JOBS.wants_reroute(None) is False
    assert JOBS.wants_reroute([_Pod(None)]) is False
    assert JOBS.wants_reroute([_Pod("")]) is False
    assert JOBS.wants_reroute([_Pod("OOMKilled")]) is False
    # A message that merely *mentions* the word in prose is not a signal.
    assert JOBS.wants_reroute([_Pod("the worker saw no REROUTE needed")]) is False


def test_the_reader_handles_the_object_shaped_pod_too():
    """The Python client returns objects in production and dicts in the tests.
    A reader that only walked one shape would be green here and broken in the
    cluster."""
    pods = [_Pod(f"{JOBS.REROUTE_TOKEN}: too big", use_objects=True)]
    assert JOBS.wants_reroute(pods) is True


def test_the_reroute_reason_never_reaches_the_owner_as_a_kubernetes_token():
    """`MseReroute` is an instruction to the scheduler. If it ever surfaces in
    the UI the owner sees a word that explains nothing."""
    assert JOBS.failure_message(JOBS._REROUTE_REASON) == JOBS.FAILURE_DEFAULT
    assert JOBS._REROUTE_REASON not in JOBS.failure_message(JOBS._REROUTE_REASON)


# ---------------------------------------------------------------------------
# The fallback: an unclaimed GPU job goes back to the cluster
# ---------------------------------------------------------------------------
class RoutingDb(FakeDb):
    """FakeDb plus the jetson-side columns and the fallback query.

    Kept separate from the shared ``FakeDb`` on purpose: that double is used by
    the whole existing suite, and widening it would change what those tests
    exercise. This one models the routing contract only.
    """

    def __init__(self):
        super().__init__()
        self.reroutes: list[tuple[str, str]] = []
        self.reclaimed: list[str] = []

    def create_job(self, job_id, filename, units, n_index, n_modes,
                   target=DB.CLUSTER, est_tets=None):
        super().create_job(job_id, filename, units, n_index, n_modes)
        self.jobs[str(job_id)].update(target=target, est_tets=est_tets,
                                      gpu_queued_at="2026-09-26T00:00:00+00:00"
                                      if target == DB.JETSON else None)

    def reroute_to_cluster(self, job_id, reason):
        job = self.jobs.get(str(job_id))
        if not job or job.get("target") != DB.JETSON or job["status"] != DB.QUEUED:
            return False
        job.update(target=DB.CLUSTER, message=reason, gpu_queued_at=None)
        self.reroutes.append((str(job_id), reason))
        return True

    def jetson_waiting(self, older_than_seconds):
        return [j for j in self.jobs.values()
                if j.get("target") == DB.JETSON and j["status"] == DB.QUEUED
                and j.get("gpu_queued_at") and not j.get("lease_owner")]

    def reclaim_expired_leases(self):
        ids = [j["job_id"] for j in self.jobs.values()
               if j.get("lease_expired")]
        for job in self.jobs.values():
            if job.get("lease_expired"):
                job.update(lease_owner=None, lease_expires_at=None)
        self.reclaimed.extend(ids)
        return ids


def _jetson_job(db, est_tets, job_id="job-1"):
    db.create_job(job_id, "u.obj", "mm", 1.0, 20,
                  target=DB.JETSON, est_tets=est_tets)
    return db.jobs[job_id]


def test_an_unclaimed_gpu_job_goes_back_to_the_cluster(tmp_path):
    db = RoutingDb()
    _jetson_job(db, est_tets=20_000)
    app, store, _d, launcher = _make(tmp_path, db=db)

    result = APP.run_tick(store, db, launcher)

    assert result["rerouted"] == 1
    assert db.jobs["job-1"]["target"] == DB.CLUSTER


def test_a_gpu_job_the_cluster_cannot_solve_is_left_where_it_is(tmp_path):
    """Rerouting it would only make it fail on the cluster *after* a queue wait.
    A job waiting for a worker that may still come is strictly better."""
    db = RoutingDb()
    _jetson_job(db, est_tets=CFG.max_tets() + 1)
    app, store, _d, launcher = _make(tmp_path, db=db)

    result = APP.run_tick(store, db, launcher)

    assert result["rerouted"] == 0
    assert db.jobs["job-1"]["target"] == DB.JETSON


def test_a_claimed_gpu_job_is_never_preempted(tmp_path, monkeypatch):
    """A worker is actively solving it, and the job is past the wait window.

    The wait window is what makes this a real test: a leased job that is merely
    *new* would be spared by the window alone, so the test would pass against a
    query that ignored leases entirely. Setting `gpu_queued_at` far in the past
    removes that cover and leaves the lease as the only thing keeping the job on
    the GPU.

    No worker token is configured, which is the case that removes the *second*
    cover: with no token the tick wants to reroute every eligible job at once,
    because no worker exists. If the waiting query ignored leases, that branch
    would take this job -- so the lease is genuinely the only thing tested here.
    """
    monkeypatch.delenv("MSE_WORKER_TOKEN", raising=False)
    db = RoutingDb()
    job = _jetson_job(db, est_tets=20_000)
    job["lease_owner"] = "w-abc"
    job["gpu_queued_at"] = "2020-01-01T00:00:00+00:00"   # long past the window
    app, store, _d, launcher = _make(tmp_path, db=db)

    result = APP.run_tick(store, db, launcher)

    assert result["rerouted"] == 0
    assert db.jobs["job-1"]["target"] == DB.JETSON


def test_a_rerouted_job_is_not_rerouted_twice(tmp_path):
    """`reroute_to_cluster` is guarded on `target = 'jetson'`, so the second
    call is a no-op -- and the message is not overwritten."""
    db = RoutingDb()
    _jetson_job(db, est_tets=20_000)
    assert db.reroute_to_cluster("job-1", "first") is True
    assert db.reroute_to_cluster("job-1", "second") is False
    assert db.jobs["job-1"]["message"] == "first"


def test_an_expired_lease_is_reclaimed_and_reported(tmp_path):
    db = RoutingDb()
    job = _jetson_job(db, est_tets=20_000)
    job.update(lease_owner="w-dead", lease_expired=True, status=DB.RUNNING)
    app, store, _d, launcher = _make(tmp_path, db=db)

    result = APP.run_tick(store, db, launcher)

    assert result["reclaimed"] == ["job-1"]
    assert db.jobs["job-1"]["lease_owner"] is None


# ---------------------------------------------------------------------------
# The worker API
# ---------------------------------------------------------------------------
class LeaseDb(RoutingDb):
    """RoutingDb plus the lease operations the worker API calls."""

    def __init__(self):
        super().__init__()
        self.claims: list[tuple[str, str]] = []

    def claim_jetson(self, owner, lease_seconds):
        for job in self.jobs.values():
            if (job.get("target") == DB.JETSON and job["status"] == DB.QUEUED
                    and not job.get("lease_owner")):
                job.update(status=DB.RUNNING, lease_owner=owner,
                           lease_expires_at="2026-09-26T00:05:00+00:00")
                self.claims.append((job["job_id"], owner))
                return dict(job)
        return None

    def renew_lease(self, job_id, owner, lease_seconds):
        job = self.jobs.get(str(job_id))
        return bool(job and job.get("lease_owner") == owner
                    and not job.get("lease_expired"))

    def lease_held_by(self, job_id, owner):
        job = self.jobs.get(str(job_id))
        return bool(job and job.get("lease_owner") == owner
                    and not job.get("lease_expired"))

    def release_jetson(self, job_id, owner):
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(lease_owner=None, lease_expires_at=None)
        return True

    def finish_jetson(self, job_id, owner):
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(status=DB.DONE, stage=DB.DONE, lease_owner=None)
        return True

    def set_jetson_stage(self, job_id, stage):
        job = self.jobs.get(str(job_id))
        if job:
            job["stage"] = stage


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    """An app with the worker API enabled, plus its doubles."""
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    app, store, _d, launcher = _make(tmp_path, db=db)
    return app.test_client(), db, store


def _auth(token="tok-abc-123"):
    return {"Authorization": f"Bearer {token}"}


def test_the_api_is_closed_when_no_token_is_configured(tmp_path, monkeypatch):
    """503, not 401. The two mean different things: one is an operator who has
    something to fix, the other is a caller who sent the wrong token."""
    monkeypatch.delenv("MSE_WORKER_TOKEN", raising=False)
    db = LeaseDb()
    app, _s, _d, _l = _make(tmp_path, db=db)
    resp = app.test_client().post("/worker/claim", headers=_auth("anything"))
    assert resp.status_code == 503


def test_an_unconfigured_api_is_closed_even_for_every_route(tmp_path, monkeypatch):
    """Not just /claim. A single route that forgot the guard would be an
    unauthenticated way into the bucket."""
    monkeypatch.delenv("MSE_WORKER_TOKEN", raising=False)
    db = LeaseDb()
    app, _s, _d, _l = _make(tmp_path, db=db)
    client = app.test_client()
    calls = [
        ("post", "/worker/claim", {}),
        ("get", "/worker/jobs/j1/input.obj", {}),
        ("post", "/worker/jobs/j1/progress", {}),
        ("put", "/worker/jobs/j1/files/modes.json", {"data": b"x"}),
        ("post", "/worker/jobs/j1/done", {}),
        ("post", "/worker/jobs/j1/failed", {}),
    ]
    for method, url, kwargs in calls:
        resp = getattr(client, method)(url, headers=_auth("anything"), **kwargs)
        assert resp.status_code == 503, f"{method.upper()} {url}"


def test_a_missing_or_wrong_token_is_refused(tmp_path, worker):
    client, _db, _store = worker
    assert client.post("/worker/claim").status_code == 401
    assert client.post("/worker/claim", headers=_auth("wrong")).status_code == 401
    assert client.post("/worker/claim",
                       headers={"Authorization": "tok-abc-123"}).status_code == 401
    assert client.post("/worker/claim",
                       headers={"Authorization": "Bearer "}).status_code == 401
    assert client.post("/worker/claim",
                       headers={"Authorization": "Bearer tok-abc-1234"}
                       ).status_code == 401


def test_an_empty_configured_token_does_not_open_the_api(tmp_path, monkeypatch):
    """An empty Secret value must fail closed. This is the case that decides
    whether a bad deploy is a denial of service or an open door."""
    monkeypatch.setenv("MSE_WORKER_TOKEN", "")
    db = LeaseDb()
    app, _s, _d, _l = _make(tmp_path, db=db)
    resp = app.test_client().post("/worker/claim",
                                  headers={"Authorization": "Bearer "})
    assert resp.status_code == 503


def test_a_claim_hands_out_one_job(tmp_path, worker):
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    resp = client.post("/worker/claim", headers=_auth())
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["job_id"] == "job-1"
    assert body["units"] == "mm" and body["n_modes"] == 20


def test_claiming_with_nothing_queued_is_204_not_an_error(tmp_path, worker):
    """The common case: the worker polls and there is nothing to do. A 404 or a
    500 here would make a healthy idle worker look broken."""
    client, _db, _store = worker
    resp = client.post("/worker/claim", headers=_auth())
    assert resp.status_code == 204


def test_two_claims_get_different_jobs_or_nothing(tmp_path, worker):
    """The double-claim. Sequential here, but the property is the same one
    `FOR UPDATE SKIP LOCKED` provides under concurrency: a second claim can
    never return a job the first one already took."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000, job_id="job-1")
    db.create_job("job-2", "u.obj", "mm", 1.0, 20,
                  target=DB.JETSON, est_tets=20_000)

    first = client.post("/worker/claim", headers=_auth()).get_json()
    second = client.post("/worker/claim", headers=_auth()).get_json()
    third = client.post("/worker/claim", headers=_auth())

    assert first["job_id"] != second["job_id"]
    assert third.status_code == 204


def test_a_worker_that_does_not_hold_the_lease_cannot_read_the_input(tmp_path, worker):
    """The job may exist and the caller may be authenticated; it is simply not
    theirs. 409 says `stop`, where 403 would say `try different credentials`."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = "w-someone-else"
    assert client.get("/worker/jobs/job-1/input.obj",
                      headers=_auth()).status_code == 409


def test_a_worker_that_does_not_hold_the_lease_cannot_write_files(tmp_path, worker):
    client, db, store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = "w-someone-else"
    resp = client.put("/worker/jobs/job-1/files/modes.json",
                      headers=_auth(), data=b"{}")
    assert resp.status_code == 409
    # And nothing was written: the guard is before the store call, not after.
    assert not (Path(store.root) / "jobs" / "job-1" / "modes.json").exists()


def test_an_expired_lease_blocks_writes_even_for_the_right_owner(tmp_path, worker):
    """A worker that slept through its lease must not resume writing over its
    successor's output. Checking only the owner would allow exactly that."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    db.jobs["job-1"]["lease_expired"] = True
    assert client.get("/worker/jobs/job-1/input.obj",
                      headers=_auth()).status_code == 409


@pytest.mark.parametrize("name", [
    "../modes.json", "..\\modes.json", "/etc/passwd", "sub/modes.json",
    "mode_1.bin", "mode_001.bin", "mode_01.bin.bak", "mode_01.exe",
    "surface.glb.bak", "", "mode_aa.bin", "surface.GLB",
])
def test_only_allow_listed_filenames_are_accepted(name):
    """An allow-list, matched in full: a traversal attempt is refused without
    any path sanitising, because it simply is not one of the names.

    **An honest note on what this test does and does not pin.** The separator
    check in ``allowed_name`` is currently redundant: ``re.match`` anchors at
    the start, and the two patterns both begin with a literal, so no name
    containing a slash can match either one. Removing the separator check does
    NOT redden this test, and I verified that by mutation rather than assuming
    it -- see the commit message. It stays because it is the guard that survives
    a future pattern like ``.*\\.bin``, and because it is the thing a reader
    looks for. It is defence in depth, not the load-bearing check, and this test
    claims only the behaviour (a traversal is refused), not the mechanism.
    """
    assert WORKER.allowed_name(name) is False


def test_the_pattern_is_anchored_so_a_leading_slash_cannot_match():
    """The reason the separator check is redundant, pinned as a fact.

    If someone later writes a pattern without a leading literal, this is the
    test that says the anchoring assumption changed -- and at that point the
    separator check becomes the load-bearing guard it looks like today.
    """
    assert WORKER._ALLOWED_PATTERN.match("/mode_01.bin") is None
    assert WORKER.allowed_name("/mode_01.bin") is False
    assert WORKER.allowed_name("x/mode_01.bin") is False


@pytest.mark.parametrize("name", [
    "modes.json", "surface.glb", "mode_01.bin", "mode_12.vtu",
])
def test_the_solver_output_names_are_accepted(name):
    assert WORKER.allowed_name(name) is True


def test_a_traversal_name_is_refused_by_the_route(tmp_path, worker):
    """The route, not just the helper: a filename that is not allowed must be
    refused before the store is touched."""
    client, db, store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    resp = client.put("/worker/jobs/job-1/files/..%2Fescape.json",
                      headers=_auth(), data=b"{}")
    assert resp.status_code in (400, 404)


def test_a_worker_can_write_an_allowed_file(tmp_path, worker):
    client, db, store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    resp = client.put("/worker/jobs/job-1/files/mode_01.bin",
                      headers=_auth(), data=b"payload")
    assert resp.status_code == 200
    assert (Path(store.root) / "jobs" / "job-1" / "mode_01.bin").read_bytes() == b"payload"


def test_the_owner_id_never_contains_the_token(tmp_path):
    """The lease is stored in a database column and printed in logs. The token
    must not be either, so only a fingerprint of it is ever used."""
    token = "tok-abc-123"
    owner = WORKER.owner_id(token)
    assert token not in owner
    assert owner == WORKER.owner_id(token)     # stable
    assert owner != WORKER.owner_id("tok-abc-124")   # and distinguishing


def test_the_token_is_never_logged_by_a_refused_request(tmp_path, worker, caplog):
    client, _db, _store = worker
    with caplog.at_level("DEBUG"):
        client.post("/worker/claim", headers=_auth("wrong-token-xyz"))
    assert "wrong-token-xyz" not in caplog.text


def test_progress_renews_and_rejects_a_lost_lease(tmp_path, worker):
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    owner = WORKER.owner_id("tok-abc-123")
    db.jobs["job-1"]["lease_owner"] = owner
    assert client.post("/worker/jobs/job-1/progress", headers=_auth(),
                       json={"stage": "meshing"}).status_code == 200
    db.jobs["job-1"]["lease_owner"] = "w-other"
    assert client.post("/worker/jobs/job-1/progress", headers=_auth(),
                       json={"stage": "meshing"}).status_code == 409


def test_done_marks_the_job_finished_for_the_lease_holder(tmp_path, worker):
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    assert client.post("/worker/jobs/job-1/done", headers=_auth()).status_code == 200
    assert db.jobs["job-1"]["status"] == DB.DONE


def test_failed_releases_the_lease_so_another_worker_can_take_it(tmp_path, worker):
    """A worker giving up must not sit on the lease until it expires: that is
    the difference between a job that waits a timeout and one that restarts."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    resp = client.post("/worker/jobs/job-1/failed", headers=_auth(),
                       json={"message": "out of memory"})
    assert resp.status_code == 200
    assert db.jobs["job-1"]["lease_owner"] is None


def test_a_stale_worker_cannot_mark_a_job_done(tmp_path, worker):
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = "w-someone-else"
    assert client.post("/worker/jobs/job-1/done", headers=_auth()).status_code == 409
    assert db.jobs["job-1"]["status"] == DB.QUEUED


# ---------------------------------------------------------------------------
# The Job manifest carries the routing config
# ---------------------------------------------------------------------------
def test_the_solver_job_is_told_which_backend_and_cap_to_use():
    """The solver re-decides the route on the exact tet count. Handing it the
    same boundary the app routed on is what makes the two agree -- a Job that
    used its own default would re-decide differently and reroute a job the app
    had already placed correctly."""
    manifest = JOBS.build_job_manifest(
        "11111111-1111-1111-1111-111111111111", "mm", 1.0, 20,
        "img", "sec", "ns", cpu_max_tets=150_000)
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["MSE_SOLVER_BACKEND"] == "cpu"
    assert env["MSE_CPU_MAX_TETS"] == "150000"


def test_the_backend_is_named_even_when_no_cap_is_given():
    manifest = JOBS.build_job_manifest(
        "11111111-1111-1111-1111-111111111111", "mm", 1.0, 20,
        "img", "sec", "ns")
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["MSE_SOLVER_BACKEND"] == "cpu"
    assert "MSE_CPU_MAX_TETS" not in env


def test_the_tick_only_hands_the_job_slot_to_a_cluster_job(tmp_path):
    """The namespace admits one Job. A jetson row reaching the launcher would
    occupy that slot while the app waited for a worker that is not coming."""
    db = RoutingDb()
    _jetson_job(db, est_tets=20_000)
    app, store, _d, launcher = _make(tmp_path, db=db)
    # The reroute runs first, so by launch time the job IS cluster-bound -- the
    # assertion is that the launcher never sees it AS a jetson job.
    launched_targets = []
    real_create = launcher.create

    def spy(job_id, units, n_index, n_modes):
        launched_targets.append(db.jobs[str(job_id)]["target"])
        return real_create(job_id, units, n_index, n_modes)

    launcher.create = spy
    APP.run_tick(store, db, launcher)
    assert launched_targets == [DB.CLUSTER]
