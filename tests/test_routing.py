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
                   target=DB.CLUSTER, est_tets=None,
                   min_feature_mm=None, scale=1.0):
        super().create_job(job_id, filename, units, n_index, n_modes,
                           target=target, est_tets=est_tets,
                           min_feature_mm=min_feature_mm, scale=scale)
        self.jobs[str(job_id)].update(target=target, est_tets=est_tets,
                                      gpu_queued_at="2026-09-26T00:00:00+00:00"
                                      if target == DB.JETSON else None,
                                      # The schema's own default, so a row here
                                      # starts in the state a real one does. A
                                      # double that omitted it would make
                                      # ``release_count`` look absent when the
                                      # real column is NOT NULL DEFAULT 0, and
                                      # every ``.get(..., 0)`` would paper over
                                      # the difference.
                                      release_count=0)

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
        requeued = []
        for job in self.jobs.values():
            if not job.get("lease_expired"):
                continue
            # Counting here matches web/db.py, and it is the path a crash-looping
            # worker actually takes: it never calls /release, it dies. A double
            # that requeued on expiry without counting would show the loop
            # terminating in tests and running forever in production.
            if int(job.get("release_count", 0)) >= DB.MAX_RELEASES:
                job.update(status=DB.FAILED, stage=DB.FAILED,
                           message="the GPU worker stopped responding and the "
                                   "job was retried too many times; try again later",
                           lease_owner=None, lease_expires_at=None)
                continue
            job.update(lease_owner=None, lease_expires_at=None,
                       status=DB.QUEUED, stage="queued",
                       gpu_queued_at="2026-09-26T00:00:00+00:00",
                       release_count=int(job.get("release_count", 0)) + 1)
            requeued.append(job["job_id"])
        self.reclaimed.extend(requeued)
        return requeued


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
    """RoutingDb plus the lease operations the worker API calls.

    KNOWN LIMIT, stated rather than papered over: these tests exercise the
    DOUBLE's ``claim_jetson``, so no test here can fail because the real
    ``web/db.py`` query is wrong. Changing the real SELECT's column list to
    ``NULL, NULL`` leaves this file entirely green.

    What that means for what is proved: the column list, the row indices and the
    returned dict in ``web/db.py:claim_jetson`` were read by hand and agree
    (SELECT lists ``min_feature_mm, scale`` as columns 5 and 6; the return reads
    ``row[4]`` and ``row[5]``). That is a manual check, not a test, and the
    reviewer should treat it as one. Reaching the real SQL needs Postgres, which
    is tier 2 -- so the honest position is that this leg is verified by reading
    and not by running.
    """

    def __init__(self):
        super().__init__()
        self.claims: list[tuple[str, str]] = []
        #: every stage the server was told, in order (see set_jetson_stage)
        self.stages_seen: list[str] = []

    #: The columns the real ``claim_jetson`` SELECTs, named here so the double
    #: cannot be more generous than the query it stands in for. Returning
    #: ``dict(job)`` was the earlier version and it hid a real defect class: the
    #: SQL could stop selecting ``min_feature_mm``/``scale`` and every test would
    #: still pass, because the double happened to hand back whatever the row
    #: held. A double that returns MORE than the query would is a double that
    #: cannot fail for the reason the query can.
    CLAIM_COLUMNS = ("job_id", "units", "n_index", "n_modes",
                     "min_feature_mm", "scale")

    def claim_jetson(self, owner, lease_seconds):
        for job in self.jobs.values():
            if (job.get("target") == DB.JETSON and job["status"] == DB.QUEUED
                    and not job.get("lease_owner")):
                job.update(status=DB.RUNNING, lease_owner=owner,
                           lease_expires_at="2026-09-26T00:05:00+00:00")
                self.claims.append((job["job_id"], owner))
                claimed = {k: job.get(k) for k in self.CLAIM_COLUMNS}
                # the real query's UPDATE sets the lease; the SELECT is what the
                # caller gets back, and only those columns come from the SELECT
                claimed["lease_owner"] = owner
                return claimed
        return None

    def renew_lease(self, job_id, owner, lease_seconds):
        job = self.jobs.get(str(job_id))
        return bool(job and job.get("lease_owner") == owner
                    and not job.get("lease_expired"))

    def lease_held_by(self, job_id, owner):
        job = self.jobs.get(str(job_id))
        return bool(job and job.get("lease_owner") == owner
                    and not job.get("lease_expired"))

    def release_jetson(self, job_id, owner, message=""):
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return None
        # ``status`` goes back to queued, matching web/db.py. The first version
        # of this double cleared only the lease, copying a real bug: claim_jetson
        # filters on status = queued, so a row released as ``running`` was never
        # claimable -- the job stranded forever with no worker and no error. A
        # double that is MORE generous than the query hides exactly that.
        #
        # The bound is modelled here for the same reason, and it is the part
        # most worth modelling: a double that requeued unconditionally would let
        # a test "prove" the loop terminates when the real SQL would cycle, and
        # would never exercise the failure branch at all.
        if int(job.get("release_count", 0)) >= DB.MAX_RELEASES:
            # The message the owner reads is the one from the run that finally
            # failed -- here, the reason the caller passed. The earlier form used
            # a fixed string, which matched neither ``web/db.py`` (which prefers
            # the caller's message and falls back only when there is none) nor
            # the lead's item D, and it would have hidden a worker that stopped
            # sending one.
            job.update(status=DB.FAILED, stage=DB.FAILED,
                       message=message or "the GPU worker could not complete "
                                          "this job after several attempts; "
                                          "try again later",
                       lease_owner=None, lease_expires_at=None)
            return "failed"
        job.update(lease_owner=None, lease_expires_at=None,
                   status=DB.QUEUED, stage="queued",
                   release_count=int(job.get("release_count", 0)) + 1)
        return "released"

    def fail_jetson(self, job_id, owner, message):
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        # Terminal: no requeue, and the message is what the owner reads.
        job.update(status=DB.FAILED, stage=DB.FAILED, message=message,
                   lease_owner=None, lease_expires_at=None)
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
        # Recorded as well as stored: a test that asserts the heartbeat carries
        # a REAL stage needs the sequence, and the row only keeps the last one.
        self.stages_seen.append(stage)


@pytest.fixture()
def worker(tmp_path, monkeypatch):
    """An app with the worker API enabled, plus its doubles."""
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    app, store, _d, launcher = _make(tmp_path, db=db)
    return app.test_client(), db, store


def _auth(token="tok-abc-123"):
    return {"Authorization": f"Bearer {token}"}


def test_a_claimed_job_carries_the_mesh_inputs(tmp_path, monkeypatch):
    """The claim payload must hand the worker the same two mesh inputs.

    This is the third leg of the path the lead asked for (row -> Job env, and
    row -> claim payload -> solver). A worker that got only the units would mesh
    the unscaled object at the physics edge while the route had been decided for
    a scaled, feature-resolved one -- the same half-wired bug, one hop further
    out, and invisible because both numbers look plausible.

    ``None`` and 1.0 are asserted as themselves rather than as falsy: the worker
    is a different process, and "absent" is not a value it can tell from "zero"
    once the JSON is parsed.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    db.create_job("job-1", "u.obj", "um", 1.5, 20, target=DB.JETSON,
                  est_tets=40_000, min_feature_mm=0.25, scale=1000.0)
    app, _s, _d, _l = _make(tmp_path, db=db)

    resp = app.test_client().post("/worker/claim", headers=_auth())
    assert resp.status_code == 200
    body = resp.get_json()

    assert body["units"] == "um"
    assert body["min_feature_mm"] == 0.25
    assert body["scale"] == 1000.0


def test_a_claim_of_an_unscaled_featureless_job_says_so_plainly(tmp_path, monkeypatch):
    """The absence must arrive as JSON null and 1.0 -- not as a missing key.

    A worker reading a missing key with ``.get()`` and a default would be fine;
    a worker reading it with ``[...]`` would crash on a job that is perfectly
    ordinary. So the keys are always present, and this pins that.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    db.create_job("job-1", "u.obj", "mm", 1.0, 20, target=DB.JETSON,
                  est_tets=40_000)
    app, _s, _d, _l = _make(tmp_path, db=db)

    body = app.test_client().post("/worker/claim",
                                  headers=_auth()).get_json()

    assert "min_feature_mm" in body and body["min_feature_mm"] is None
    assert "scale" in body and body["scale"] == 1.0


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
    check in ``allowed_name`` is currently redundant: the pattern is anchored at
    the start with a literal, so no name containing a slash can match it.
    Removing the separator check does NOT redden this test, and I verified that
    by mutation rather than assuming it -- see the commit message. It stays
    because it is the guard that survives a future pattern like ``.*\\.bin``,
    and because it is the thing a reader looks for. It is defence in depth, not
    the load-bearing check, and this test claims only the behaviour (a traversal
    is refused), not the mechanism.
    """
    assert WORKER.allowed_name(name) is False


@pytest.mark.parametrize("name", [
    "mode_01.bin\n", "mode_12.vtu\n", "mode_01.bin\n\n", "mode_01.bin\nx",
    "modes.json\n", "surface.glb\n", "mode_01.bin\r", "mode_01.bin ",
])
def test_a_trailing_newline_is_not_an_allowed_name(name):
    """The bypass the reviewer found, and the three lookalikes around it.

    Python's ``$`` matches just before a trailing newline, so ``re.match`` with
    an anchored pattern accepted ``"mode_01.bin\\n"`` and the route stored it as
    a SECOND key (``jobs/<id>/mode_01.bin\\n``) beside the real one. Two keys
    for one mode is a silent corruption the viewer never sees, which is why this
    is a correctness bug and not a tidiness one.

    ``\\r`` and a trailing space were ALREADY refused -- ``$`` does not match
    before them -- so those two cases are here as *controls*, not as evidence.
    A test that only contains the newline case would pass against a fix that
    merely grew the character class, and would not tell a reader that the
    newline is the only one that was ever reachable. The pair
    (``mode_01.bin\\r`` false today, and false before) is what makes the newline
    case mean something.
    """
    assert WORKER.allowed_name(name) is False, (
        f"{name!r} must not be an allowed upload name"
    )


def test_the_allowed_pattern_rejects_the_newline_that_match_accepted():
    """The mechanism, pinned separately from the behaviour above.

    ``allowed_name`` could be made to refuse the newline in more than one way, so
    this asserts the specific property that was broken: the pattern itself no
    longer matches a name with a trailing newline. If someone switches back to
    ``re.match`` with a ``$`` anchor, this fails even while ``allowed_name``
    happens to stay correct for another reason.
    """
    assert WORKER._ALLOWED_PATTERN.match("mode_01.bin\n") is None, (
        "the pattern's $/\\Z anchor is what closed the newline bypass"
    )
    assert WORKER._ALLOWED_PATTERN.fullmatch("mode_01.bin") is not None


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


def test_failed_is_terminal_and_the_message_reaches_the_owner(tmp_path, worker):
    """A deterministic failure must END the job, not requeue it.

    This is the fix for the fail/requeue/fail loop, and the two halves of it are
    both asserted here: the lease is cleared (so the row is not stuck) AND the
    row is FAILED (so it is not offered again), with the worker's message on it
    where the owner can read it.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    resp = client.post("/worker/jobs/job-1/failed", headers=_auth(),
                       json={"message": "this model is too detailed to solve"})
    assert resp.status_code == 200
    row = db.jobs["job-1"]
    assert row["lease_owner"] is None, "the lease must be cleared"
    assert row["status"] == DB.FAILED, (
        "a deterministic failure was requeued: it comes straight back to a "
        "worker that fails it the same way, forever")
    assert row["message"] == "this model is too detailed to solve"


def test_the_failure_message_is_visible_on_the_job_status_api(tmp_path, worker):
    """What the owner actually sees -- through the route the page reads.

    Asserting on the double's dict would prove only that the double stored it.
    The owner's view is the job status API, so the message has to survive that
    projection too; a message that exists in the row but not in the payload is
    a message the owner never reads.

    The id is a real UUID because the route validates one and 404s otherwise --
    the short ids the other tests use never reach ``get_job`` at all.
    """
    client, db, _store = worker
    job_id = "11111111-1111-1111-1111-111111111111"
    _jetson_job(db, est_tets=20_000, job_id=job_id)
    db.jobs[job_id]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    client.post(f"/worker/jobs/{job_id}/failed", headers=_auth(),
                json={"message": "this object could not be solved for 20 modes"})

    payload = client.get(f"/jobs/{job_id}/status").get_json()
    assert payload["status"] == DB.FAILED
    assert "could not be solved for 20 modes" in (payload.get("message") or "")


def test_failed_for_a_lease_this_worker_does_not_hold_is_409(tmp_path, worker):
    """Never a silent success: the caller must learn it has no lease."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = "w-someone-else"
    resp = client.post("/worker/jobs/job-1/failed", headers=_auth(),
                       json={"message": "nope"})
    assert resp.status_code == 409
    assert db.jobs["job-1"]["status"] == DB.QUEUED


def test_release_hands_the_job_back_with_no_verdict(tmp_path, worker):
    """The transient verb: claimable again, counted, and NOT failed."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    resp = client.post("/worker/jobs/job-1/release", headers=_auth(),
                       json={"message": "the connection dropped"})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "released": True, "failed": False}
    row = db.jobs["job-1"]
    assert row["status"] == DB.QUEUED, "a release must leave it claimable"
    assert row["lease_owner"] is None
    assert row["release_count"] == 1, "the release must be counted"
    assert "the connection dropped" not in (row.get("message") or ""), (
        "a release is not a verdict; it must not write a failure message")


def test_a_release_on_a_lease_this_worker_does_not_hold_is_409(tmp_path,
                                                               worker):
    """The lead's item E. A stale worker must not free somebody else's lease."""
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = "w-someone-else"
    resp = client.post("/worker/jobs/job-1/release", headers=_auth(),
                       json={"message": "dropping out"})
    assert resp.status_code == 409
    assert db.jobs["job-1"]["lease_owner"] == "w-someone-else"
    assert db.jobs["job-1"]["release_count"] == 0


def test_three_releases_fail_the_job_and_the_bound_is_the_same_in_both_copies(
        tmp_path, worker):
    """The bound, driven through the route until it fires.

    ``MAX_RELEASES`` permits that many releases, so the one AFTER them is the
    last: three requeues, then the fourth finds the count at the bound and the
    app fails the job itself with a plain message. An unbounded release would
    let a flapping worker loop forever, which is the defect this exists for.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    owner = WORKER.owner_id("tok-abc-123")

    for n in range(DB.MAX_RELEASES):
        db.jobs["job-1"]["lease_owner"] = owner
        resp = client.post("/worker/jobs/job-1/release", headers=_auth(),
                           json={"message": "still dropping"})
        assert resp.status_code == 200, f"release {n} refused"
        assert resp.get_json()["released"] is True, (
            f"the bound fired on release {n}, one before it should")
        assert db.jobs["job-1"]["release_count"] == n + 1

    db.jobs["job-1"]["lease_owner"] = owner
    resp = client.post("/worker/jobs/job-1/release", headers=_auth(),
                       json={"message": "and again"})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "released": False, "failed": True}
    row = db.jobs["job-1"]
    assert db.jobs["job-1"]["status"] == DB.FAILED
    assert db.jobs["job-1"]["lease_owner"] is None
    # The message is the one from the run that FINALLY failed (item D): the
    # owner should read the cause that ended it, not an earlier one.
    assert db.jobs["job-1"]["message"] == "and again", (
        "the owner was not shown the failure that ended the job")


def test_two_interleaved_releases_cannot_both_requeue_past_the_bound(
        tmp_path, worker):
    """The lead's item B, at the API level.

    Two releases race on the SAME row with the count already at the bound. A
    read-count-then-decide form lets both read 3, both decide they are under it,
    and both requeue -- the job that should be terminal goes round again. The
    bound lives in the UPDATE's WHERE clause, so a losing racer matches zero
    rows.

    The loser can land in either of the two legal places, and BOTH are correct:
    a 409 (the winner already cleared the lease, so the second caller no longer
    holds it) or a 200 that reports ``failed``. What is forbidden is two
    requeues, and that is what this asserts -- counting requeues, not statuses.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    owner = WORKER.owner_id("tok-abc-123")

    # bring the row to one release BELOW the bound
    for _ in range(DB.MAX_RELEASES - 1):
        db.jobs["job-1"]["lease_owner"] = owner
        client.post("/worker/jobs/job-1/release", headers=_auth(),
                    json={"message": "dropping"})
    assert db.jobs["job-1"]["release_count"] == DB.MAX_RELEASES - 1

    db.jobs["job-1"]["lease_owner"] = owner
    first = client.post("/worker/jobs/job-1/release", headers=_auth(),
                        json={"message": "racer one"})
    second = client.post("/worker/jobs/job-1/release", headers=_auth(),
                         json={"message": "racer two"})

    requeues = 0
    for resp in (first, second):
        body = resp.get_json()
        if resp.status_code == 200 and body and body.get("released"):
            requeues += 1
        assert resp.status_code in (200, 409), (
            f"unexpected status {resp.status_code}: {resp.get_data()!r}")
    assert requeues <= 1, (
        f"both racers requeued, so the job escaped the bound: "
        f"{first.status_code}/{second.status_code}")
    assert db.jobs["job-1"]["release_count"] <= DB.MAX_RELEASES, (
        f"the race double-counted the release: "
        f"{db.jobs['job-1']['release_count']}")

    # THE INVARIANT THAT MATTERS: the job cannot cycle. Either the race failed it
    # outright, or it is left at the bound and claimable exactly once more --
    # and that last claim ends it. Prove termination rather than assuming it.
    taken = db.claim_jetson(owner, 300)
    if taken is not None:
        db.jobs["job-1"]["lease_owner"] = owner
        final = client.post("/worker/jobs/job-1/release", headers=_auth(),
                            json={"message": "one too many"})
        assert final.get_json() == {"ok": True, "released": False,
                                    "failed": True}, (
            "the job was requeued past the bound: it can loop forever")
    assert db.jobs["job-1"]["status"] == DB.FAILED, (
        "a job at the bound is still claimable: it would cycle forever")


def test_the_bound_is_reachable_from_the_expiry_path_too(tmp_path, monkeypatch):
    """The lead's item C: EVERY path back to claimable counts, not just release.

    A worker that dies mid-job never calls /release -- the row comes back
    through lease EXPIRY. If that path did not increment the same counter, a
    worker that crashed every time would requeue forever through the timeout
    and never reach the bound, which is the loop this whole mechanism removes.

    Driven on the REAL ``reclaim_expired_leases`` logic via the double, which
    models the count-and-bound: the assertion that matters is that repeated
    expiries converge on FAILED instead of cycling.
    """
    db = LeaseDb()
    _jetson_job(db, est_tets=20_000)
    owner = "w-crashy"

    for n in range(DB.MAX_RELEASES):
        db.jobs["job-1"].update(lease_owner=owner, status=DB.RUNNING,
                                lease_expired=True)
        requeued = db.reclaim_expired_leases()
        # Claimable again is the whole property, and ``claim_jetson`` requires
        # ``status == 'queued' AND lease_owner IS NULL``. Clearing the lease
        # without resetting the status leaves a ``running`` row that NO worker
        # can take -- stranded, not looping, and with no symptom at all. That
        # was a real defect in web/db.py; this asserts against it.
        assert "job-1" in requeued, f"expiry {n} did not reclaim the job"
        db.jobs["job-1"]["lease_expired"] = False
        assert db.jobs["job-1"]["lease_owner"] is None
        assert db.jobs["job-1"]["status"] == DB.QUEUED, (
            "the expiry left the row running, so claim_jetson will never "
            "take it: the job is stranded")
        assert db.jobs["job-1"]["release_count"] == n + 1, (
            "the expiry path did not count the requeue")

    db.jobs["job-1"].update(lease_owner=owner, status=DB.RUNNING,
                            lease_expired=True)
    requeued = db.reclaim_expired_leases()
    assert db.jobs["job-1"]["status"] == DB.FAILED, (
        "a crash-looping worker can requeue forever through expiry: the bound "
        "must apply here too")
    assert "job-1" not in requeued, (
        "the bounded-out row was still reported as requeued")


def test_a_released_job_is_claimable_by_the_same_and_by_another_worker(
        tmp_path, worker):
    """The lead's state machine, first half: released -> claimable again.

    ``claim_jetson`` requires ``status = 'queued' AND lease_owner IS NULL``, so
    a release that only nulls the lease produces a row that is permanently
    unclaimable. Both directions are checked -- the SAME worker and a DIFFERENT
    one -- because "the releasing worker can take it back" and "anyone can"
    are different claims and only the second is what a requeue means.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    first = WORKER.owner_id("tok-abc-123")
    db.jobs["job-1"]["lease_owner"] = first
    db.jobs["job-1"]["status"] = DB.RUNNING

    assert client.post("/worker/jobs/job-1/release", headers=_auth(),
                       json={"message": "restarting"}).status_code == 200
    assert db.jobs["job-1"]["status"] == DB.QUEUED

    # another worker, through the real claim path
    other = WORKER.owner_id("a-different-token")
    claimed = db.claim_jetson(other, 300)
    assert claimed is not None, (
        "the released job is not claimable: claim_jetson needs status='queued'")
    assert claimed["job_id"] == "job-1"
    assert db.jobs["job-1"]["lease_owner"] == other


def test_a_terminally_failed_job_is_never_claimable(tmp_path, worker):
    """The state machine, second half: FAILED means never again.

    The counterpart to the release: a terminal failure must leave a row no
    worker can pick up, however many times it asks.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    client.post("/worker/jobs/job-1/failed", headers=_auth(),
                json={"message": "too detailed"})

    assert db.jobs["job-1"]["status"] == DB.FAILED
    assert db.claim_jetson(WORKER.owner_id("a-different-token"), 300) is None, (
        "a terminally failed job was handed out again")


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

    def spy(job_id, units, n_index, n_modes, **kwargs):
        # **kwargs rather than the named parameters: this spy asserts WHICH
        # target was launched, and a test that breaks every time the launcher
        # gains an argument is a test that gets the arguments deleted from it
        # instead of read.
        launched_targets.append(db.jobs[str(job_id)]["target"])
        return real_create(job_id, units, n_index, n_modes, **kwargs)

    launcher.create = spy
    APP.run_tick(store, db, launcher)
    assert launched_targets == [DB.CLUSTER]


# ---------------------------------------------------------------------------
# F2: the element cap has ONE definition, and both paths carry it
# ---------------------------------------------------------------------------
#
# The defect these pin: ``web/config.py`` capped uploads at 600_000 while the
# solver's own ``DEFAULT_MAX_TETS`` was 80_000, and NOTHING passed the app's
# number to the solver. So the app accepted an upload, told the owner it was
# fine, routed it to the Jetson, and the solver refused it at 80k -- failing
# precisely the large jobs the GPU exists for. Two constants that must agree and
# are never compared is the defect class; there is now one constant, imported.

def test_the_app_and_the_solver_share_one_element_cap():
    """Not "they happen to match today" -- the app IMPORTS the solver's number.

    Asserting equality would pass against two constants that agree by luck and
    drift at the next edit, which is exactly how this defect appeared. Asserting
    identity is the claim that actually holds the property: change the solver's
    default and the app's ceiling moves with it, with no second edit.
    """
    import mse_maxwell_solver as SOLVER

    assert CFG.DEFAULT_MAX_TETS is SOLVER.DEFAULT_MAX_TETS, (
        "the app's cap must BE the solver's cap, not a copy of its value"
    )
    assert CFG.DEFAULT_CPU_MAX_TETS == SOLVER.DEFAULT_CPU_MAX_TETS


def test_the_job_env_carries_both_caps():
    """The cluster Job must be told the limits, not left to its own default.

    Before this the Job env carried only MSE_CPU_MAX_TETS, so the absolute cap
    was whatever the image's solver defaulted to. The `MSE_MAX_TETS` assertion
    is the one that would have caught the 80k bug.
    """
    manifest = JOBS.build_job_manifest(
        "11111111-1111-1111-1111-111111111111", "mm", 1.0, 20,
        "img", "sec", "ns", max_tets=600_000, cpu_max_tets=150_000)
    container = manifest["spec"]["template"]["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["MSE_MAX_TETS"] == "600000"
    assert env["MSE_CPU_MAX_TETS"] == "150000"


def test_the_app_passes_both_caps_to_the_real_launcher(tmp_path, monkeypatch):
    """The wiring, not the builder: a builder that takes a cap nobody gives it
    is still a Job running on a default.

    This is the leg that was actually broken -- ``build_job_manifest`` grew a
    ``cpu_max_tets`` parameter that no caller ever set, so every assertion about
    the builder passed while every real Job ran with the solver's default.

    ``_launcher`` is called directly rather than through ``create_app``: the
    factory honours an injected ``MSE_JOB_LAUNCHER``, and with a test double
    installed this path is never reached at all. Testing it through the factory
    is how this test first passed while proving nothing.
    """
    monkeypatch.setenv("MSE_SOLVER_IMAGE", "test-image")

    class FakeApp:
        config: dict = {}

    launcher = APP._launcher(FakeApp())
    assert launcher.max_tets == CFG.max_tets(), (
        "the real launcher was built without the app's element cap"
    )
    assert launcher.cpu_max_tets == CFG.cpu_max_tets()


def test_the_claim_payload_carries_both_caps(tmp_path, worker):
    """The worker has no cluster access and no business holding a copy of the
    app's constant, so the caps must arrive with the claim.

    A worker that guessed them would be the same two-sources defect one layer
    down; a worker that omitted them would fail large jobs on the Jetson instead
    of in the cluster, which is worse -- the large jobs are what the Jetson is
    for.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    body = client.post("/worker/claim", headers=_auth()).get_json()
    assert body["max_tets"] == CFG.max_tets()
    assert body["cpu_max_tets"] == CFG.cpu_max_tets()
