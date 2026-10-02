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
import json
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
def test_the_token_is_one_definition_not_two_that_agree():
    """The solver writes it and the app looks for it -- ONE string, one source.

    This used to be an equality between two retyped literals, with this test as
    the thing keeping them equal. That is the weaker arrangement: it holds only
    while somebody runs it, and the copy in ``web/jobs.py`` could drift the
    moment nobody did. Both sides now import ``mse_common.REROUTE_TOKEN``, which
    the web image ships, so there is one string in the process and nothing to
    drift. ``is`` rather than ``==``: two equal literals would still be two
    definitions, and this test exists to rule that out, not to re-check equality.
    """
    import mse_common
    import mse_maxwell_solver as SOLVER

    assert SOLVER.REROUTE_TOKEN is mse_common.REROUTE_TOKEN
    assert JOBS.REROUTE_TOKEN is mse_common.REROUTE_TOKEN


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
                   min_feature_m=None, scale=1.0,
                   kind=DB.CAVITY, pcs=None, tds=None):
        super().create_job(job_id, filename, units, n_index, n_modes,
                           target=target, est_tets=est_tets,
                           min_feature_m=min_feature_m, scale=scale,
                           kind=kind, pcs=pcs, tds=tds)
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
                                      release_count=0,
                                      # Modelled so a pc row keeps its kind
                                      # through the routing double. The base
                                      # FakeDb stores them now; these are the
                                      # schema's own defaults made explicit.
                                      kind=kind, pcs=pcs)

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
            # A LEASE ONLY SPEAKS FOR A RUNNING ROW. This filter mirrors
            # web/db.py, and it is not decoration: without it this double
            # requeues a terminal row, which is the bug found live at 20:15Z
            # (a FAILED job whose old mark_failed left the lease set was
            # "claimable again" and got solved). A double that judged the lease
            # alone would keep reporting that bug fixed.
            if job.get("status") != DB.RUNNING:
                continue
            # Counting here matches web/db.py, and it is the path a crash-looping
            # worker actually takes: it never calls /release, it dies. A double
            # that requeued on expiry without counting would show the loop
            # terminating in tests and running forever in production.
            if int(job.get("release_count", 0)) >= DB.MAX_RELEASES:
                job.update(status=DB.FAILED, stage=DB.FAILED,
                           message="the GPU worker stopped responding and the "
                                   f"job was retried up to {DB.MAX_RELEASES} "
                                   "times; try again later",
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
# A broken reclaim must leave a trace (not just "return []")
#
# The tick's answer is [] both when nothing expired and when the statement that
# reclaims is broken. That is the could-not-ask shape: the two are identical on
# the only surface a monitor reads, and this is the ONLY path that bounds a
# crash-looping GPU worker, so a silently broken reclaim makes the bound
# unreachable exactly when it is needed. The remedy is a health signal, and the
# tests below are what stop it being decoration.
# ---------------------------------------------------------------------------
class _BrokenReclaim(RoutingDb):
    """Reclaims raise, to stand in for a statement that fails every tick."""

    def __init__(self, exc):
        super().__init__()
        self._exc = exc

    def reclaim_expired_leases(self):
        raise self._exc


def test_a_transport_failure_reclaiming_is_logged_and_not_degraded(tmp_path):
    """A database that is briefly unreachable is not a degraded scheduler.

    Transport errors are the transient case: the next tick retries and the
    cluster fallback still runs. Raising a health alarm for a blip would train
    whoever reads it to ignore the alarm.
    """
    import psycopg2

    db = _BrokenReclaim(psycopg2.OperationalError("server closed the connection"))
    app, store, _d, launcher = _make(tmp_path, db=db)
    APP.SCHEDULER_DEGRADED.clear()

    messages = []
    result = APP.run_tick(store, db, launcher, log=messages.append)

    assert result["reclaimed"] == []
    assert APP.SCHEDULER_DEGRADED == [], (
        "a transport blip must not mark the scheduler degraded"
    )
    assert any("could not reach the database" in m for m in messages), (
        "the tick must still say what it could not do"
    )


def test_a_broken_statement_reclaiming_degrades_the_scheduler(tmp_path):
    """A statement error will fail every tick, so it must be visible.

    This is the 3/3 defect itself: ``ANY(%s)`` over ``list[str]`` against a
    ``uuid`` column raises ``ProgrammingError``, which the tick swallowed into
    ``[]``. Only a health signal distinguishes it from "nothing expired".
    """
    import psycopg2

    db = _BrokenReclaim(psycopg2.ProgrammingError(
        "operator does not exist: uuid = text"))
    app, store, _d, launcher = _make(tmp_path, db=db)
    APP.SCHEDULER_DEGRADED.clear()

    messages = []
    APP.run_tick(store, db, launcher, log=messages.append)

    assert APP.SCHEDULER_DEGRADED, "a broken statement must set the flag"
    assert "ProgrammingError" in APP.SCHEDULER_DEGRADED[0]
    assert any("ERROR" in m for m in messages), "and must be logged at ERROR"

    # and /healthz reports it, which is the whole point of the flag
    app.test_client().get("/")
    body = app.test_client().get("/healthz").get_json()
    assert body["degraded"] is True
    assert "ProgrammingError" in body["degraded_reason"]

    APP.SCHEDULER_DEGRADED.clear()


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
    (SELECT lists ``min_feature_m, scale`` as columns 5 and 6; the return reads
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
        #: mode rows by job, populated by finish_jetson_with_modes. Backed here
        #: rather than left to the base because a leg that asserts the rows were
        #: written must read them from somewhere that the WRITE populated.
        self.modes: dict[str, list] = {}

    #: The columns the real ``claim_jetson`` SELECTs, named here so the double
    #: cannot be more generous than the query it stands in for. Returning
    #: ``dict(job)`` was the earlier version and it hid a real defect class: the
    #: SQL could stop selecting ``min_feature_m``/``scale`` and every test would
    #: still pass, because the double happened to hand back whatever the row
    #: held. A double that returns MORE than the query would is a double that
    #: cannot fail for the reason the query can.
    CLAIM_COLUMNS = ("job_id", "units", "n_index", "n_modes",
                     "min_feature_m", "scale", "materials", "kind", "pcs", "tds")

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

    def defer_busy(self, job_id, owner, message=""):
        # Modelled on web/db.py, including the two branches a double is most
        # tempted to skip: a non-td row is refused (None), and the wait is
        # bounded. ``busy_minutes`` stands in for now() - busy_since; tests set
        # it to put a row past the bound. release_count is never touched.
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner or job.get("kind") != DB.TD:
            return None
        if job.get("busy_minutes", 0) >= DB.BUSY_WAIT_MINUTES:
            job.update(status=DB.FAILED, stage=DB.FAILED,
                       message=DB.BUSY_GAVE_UP_MESSAGE,
                       lease_owner=None, lease_expires_at=None)
            return "failed"
        job.setdefault("busy_minutes", 0)
        job.update(lease_owner=None, lease_expires_at=None, status=DB.QUEUED,
                   stage=DB.BUSY_STAGE, message=DB._busy_message(message))
        return "deferred"

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
        # Both messages come from ``web/db.py``'s own helpers rather than being
        # restated here. The form this double used to have -- ``message or
        # <fixed string>`` on the over-bound branch, and no message at all on the
        # retry branch -- was the DEFECT that item 2b fixed in the real store,
        # faithfully copied. A double that models the bug instead of the fix
        # makes every test written against it report the fixed code as correct
        # for the wrong reason, and hides the regression when it comes back.
        if int(job.get("release_count", 0)) >= DB.MAX_RELEASES:
            job.update(status=DB.FAILED, stage=DB.FAILED,
                       message=DB._terminal_message(
                           message, int(job.get("release_count", 0))),
                       lease_owner=None, lease_expires_at=None)
            return "failed"
        attempt = int(job.get("release_count", 0)) + 1
        job.update(lease_owner=None, lease_expires_at=None,
                   status=DB.QUEUED, stage="queued",
                   message=DB._retry_message(message, DB.MAX_RELEASES,
                                             attempt=attempt),
                   release_count=attempt)
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

    def finish_jetson_with_modes(self, job_id, owner, rows):
        """Mark done and record the rows; ``False`` if the lease went.

        Both writes or neither, matching the real store -- a double that marked
        the job done on a lost lease would make the route's 409 branch
        unreachable and hide exactly the case the real transaction exists for.
        """
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(status=DB.DONE, stage=DB.DONE, lease_owner=None)
        self.modes[str(job_id)] = [
            {"index": int(i), "k0_per_m": float(k0), "freq_hz": float(f),
             "vacuum_wavelength_m": float(w)}
            for (i, k0, f, w) in rows
        ]
        return True

    def finish_jetson_with_bands(self, job_id, owner, request, bands):
        """Mark a pc job done and merge the bands into pcs; ``False`` if the
        lease went.

        The MERGE is modelled, not a plain assignment: the real statement is
        ``pcs = coalesce(pcs, '{}') || %s``, so the request that was solved
        survives. A double that overwrote ``pcs`` outright would let a test
        assert the bands landed while the real store dropped the request --
        the exact shape of a double more generous than the query it stands in
        for, which this file has a KNOWN LIMIT about already.
        """
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(status=DB.DONE, stage=DB.DONE, lease_owner=None)
        merged = dict(job.get("pcs") or {})
        merged["bands"] = bands
        job["pcs"] = merged
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


def _rows(n, **overrides):
    """``n`` mode rows as a ``/done`` body carries them.

    Indices are 1..n and every number is finite, so this is the VALID shape --
    the legs that test refusal mutate a copy of it. Defaults to the ``n_modes``
    ``_jetson_job`` creates, because the route checks the count against the JOB,
    not against anything the caller says.
    """
    rows = [
        {"index": i + 1, "k0_per_m": 1.0e6 + i, "freq_hz": 1.0e14 + i,
         "vacuum_wavelength_m": 1.0e-6 - i * 1.0e-9}
        for i in range(n)
    ]
    for row in rows:
        row.update(overrides)
    return rows


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
                  est_tets=40_000, min_feature_m=0.25, scale=1000.0)
    app, _s, _d, _l = _make(tmp_path, db=db)

    resp = app.test_client().post("/worker/claim", headers=_auth())
    assert resp.status_code == 200
    body = resp.get_json()

    assert body["units"] == "um"
    assert body["min_feature_m"] == 0.25
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

    assert "min_feature_m" in body and body["min_feature_m"] is None
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
    "trace.bin", "spectrum.json", "frame_01.bin", "frame_60.bin",
])
def test_the_solver_output_names_are_accepted(name):
    assert WORKER.allowed_name(name) is True


@pytest.mark.parametrize("name", [
    "frame_1.bin", "frame_001.bin", "frame_01.vtu", "frame_ab.bin",
    "frame_01.bin\n", "x/frame_01.bin", "trace.json", "spectrum.bin",
])
def test_near_miss_td_names_are_refused(name):
    assert WORKER.allowed_name(name) is False


def test_the_worker_and_the_app_agree_on_every_td_result_name():
    """The worker's enumerator and the app's allow-list are two lists; a name
    only one of them accepts is a file lost while the job reports done."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "worker"))
    import mse_worker as W
    import td_solve as T
    names = [T.TRACE_NAME, T.SPECTRUM_NAME] + [
        T.FRAME_NAME.format(n) for n in (1, 9, 10, T._solver()[0].MAX_FRAMES)]
    for name in names:
        worker_ok = name in W._UPLOAD_EXACT or W._is_frame_file(name)
        assert worker_ok and WORKER.allowed_name(name), name
    for bad in ("frame_1.bin", "frame_001.bin", "frame_01.vtu", "frame_ab.bin"):
        assert not W._is_frame_file(bad) and not WORKER.allowed_name(bad), bad


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
    assert client.post("/worker/jobs/job-1/done", headers=_auth(),
                       json={"modes": _rows(20)}).status_code == 200
    assert db.jobs["job-1"]["status"] == DB.DONE


def test_done_writes_the_mode_rows_it_was_given(tmp_path, worker):
    """DONE and the rows land together -- the point of the paired write.

    Asserting only the status would pass on the deploy-3 shape: a finished job
    whose mode list is empty, which the job page renders as a job with no modes
    rather than as a job that never reported any.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    rows = _rows(20)
    assert client.post("/worker/jobs/job-1/done", headers=_auth(),
                       json={"modes": rows}).status_code == 200
    written = db.modes["job-1"]
    assert len(written) == 20, f"expected 20 mode rows, got {written!r}"
    assert [m["index"] for m in written] == list(range(1, 21))
    assert written[0]["freq_hz"] == rows[0]["freq_hz"]
    assert db.jobs["job-1"]["status"] == DB.DONE


@pytest.mark.parametrize("body, why", [
    ({}, "no modes key at all"),
    ({"modes": []}, "an empty list"),
    ({"modes": _rows(19)}, "too few rows"),
    ({"modes": _rows(21)}, "too many rows"),
    ({"modes": "not a list"}, "the wrong type"),
    ({"modes": [None] * 20}, "entries that are not objects"),
])
def test_done_refuses_a_body_it_cannot_trust(tmp_path, worker, body, why):
    """A done WITHOUT usable rows must not produce DONE.

    The positive control is in the same leg: the neighbouring test posts a good
    body and gets 200, so this is not a route that refuses everything.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    response = client.post("/worker/jobs/job-1/done", headers=_auth(), json=body)
    assert response.status_code == 400, f"{why}: got {response.status_code}"
    assert db.jobs["job-1"]["status"] != DB.DONE, (
        f"{why}: the job was marked DONE anyway")
    assert db.modes.get("job-1") in (None, []), (
        f"{why}: rows were written from a body that was refused")


def test_done_refuses_a_nan_frequency(tmp_path, worker):
    """NaN arrives as a real float through JSON, so a type check misses it.

    ``json.loads`` accepts a bare ``NaN`` literal and returns a float for it;
    ``isinstance(nan, float)`` is True. v1 shipped NaN in a payload once, so the
    guard is ``math.isfinite`` and this is the leg that proves it.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    rows = _rows(20)
    # NOT via json= -- Flask would serialise the NaN back out as a literal and
    # the point is to hand the route the bytes a hostile or buggy worker would.
    payload = json.dumps({"modes": rows}).replace('"freq_hz": 1', '"freq_hz": NaN', 1)
    assert "NaN" in payload, "the NaN never made it into the payload"
    response = client.post("/worker/jobs/job-1/done", headers=_auth(),
                           data=payload, content_type="application/json")
    assert response.status_code == 400, response.get_data(as_text=True)
    assert db.jobs["job-1"]["status"] != DB.DONE
    assert db.modes.get("job-1") in (None, [])


def test_done_refuses_a_gap_or_a_duplicate_index(tmp_path, worker):
    """n rows in 1..n is the condition -- counting rows is not enough.

    Each mutation keeps the COUNT right (20 rows) and breaks only the index
    set, so a route that checked ``len(rows) == n_modes`` and nothing else would
    pass all of these. The duplicate, the gap and the out-of-range index are
    three different ways to put a mode list in the app that the job page cannot
    render honestly.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    # (position, new index, what is wrong with it)
    for position, index, why in ((1, 1, "a duplicate of row 0"),
                                 (19, 21, "out of range for 20 modes"),
                                 (0, 0, "zero is not a 1-based index")):
        rows = _rows(20)
        rows[position]["index"] = index
        response = client.post("/worker/jobs/job-1/done", headers=_auth(),
                               json={"modes": rows})
        assert response.status_code == 400, why
        assert db.jobs["job-1"]["status"] != DB.DONE, why
        assert db.modes.get("job-1") in (None, []), why


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
    # A release IS allowed to write a message now, and should: the owner sees
    # "retrying (attempt 1 of 3): the connection dropped" while they wait, which
    # is the honest thing to show. What it must not write is a TERMINAL verdict
    # -- no "failed", no bound sentence, because the retries are not over. The
    # earlier assertion here ("no message at all") was correct for a row that
    # had no message column being written, and would now forbid the retry text
    # the owner needs.
    assert row["message"] == DB._retry_message("the connection dropped",
                                               DB.MAX_RELEASES, attempt=1), (
        "a release must record the retry and its attempt number, and nothing "
        "that reads as a verdict")
    assert "could not complete this job after" not in row["message"], (
        "a release wrote the bound sentence; the owner would be told the job "
        "was over while it is still queued")


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
    # owner should read the cause that ended it, not an earlier one. It also
    # carries the bound sentence -- the owner has to be told the retries are
    # over, or they will wait for an attempt that is not coming. The expected
    # text comes from the production helper, not a copy retyped here.
    assert db.jobs["job-1"]["message"] == DB._terminal_message(
        "and again", DB.MAX_RELEASES), (
        "the owner was not shown the failure that ended the job, or was not "
        "told the retries were over")


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


def test_a_terminal_row_with_a_stale_lease_is_not_requeued(tmp_path):
    """The 20:15Z resurrection, driven on the lease-path double directly.

    The bug found live on the third deploy: ``reclaim_expired_leases`` selected
    on the expired lease ALONE, so a FAILED row that the OLD ``mark_failed``
    left leased was requeued to ``queued`` and a worker solved a job the owner
    had been told was over ("its GPU worker's lease expired; the job is
    claimable again").

    SCOPE, and it is the same caveat the web-routes sibling carries: this drives
    the DOUBLE's logic, so it pins that the double models the filter, not that
    the real SQL has it. Real Postgres pins the SQL -- ``pg_race_check.py`` D3
    is the leg that goes red when the real ``AND status = running`` is removed.
    What this leg buys is that the double cannot quietly go back to modelling
    the bug, which is how the unit suite came to be unable to see it.
    """
    db = LeaseDb()
    _jetson_job(db, est_tets=20_000)

    for status in (DB.FAILED, DB.DONE):
        db.jobs["job-1"].update(status=status, stage=status, lease_owner="w-dead",
                                lease_expired=True)
        requeued = db.reclaim_expired_leases()
        assert db.jobs["job-1"]["status"] == status, (
            f"a {status} row was requeued by its stale lease; the lease only "
            f"speaks for a RUNNING row")
        assert "job-1" not in requeued, (
            f"a {status} row was reported as reclaimed")


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
    # Valid rows, so the 409 is about the LEASE and not about the body. Sending
    # an empty body here would pass for the wrong reason once the route validates.
    response = client.post("/worker/jobs/job-1/done", headers=_auth(),
                           json={"modes": _rows(20)})
    assert response.status_code == 409, response.get_data(as_text=True)
    assert db.jobs["job-1"]["status"] == DB.QUEUED


def test_a_lost_lease_does_not_delete_the_rows(tmp_path, worker):
    """The DELETE and the lease check must be ONE unit, or a stale worker wipes
    a job it does not own.

    The worker that lost the lease is the one with the rows it computed, and it
    is the one whose ``/done`` must change nothing. If the DELETE ran before the
    lease was judged, the job would come back to the queue with the modes of the
    attempt that failed deleted and no replacement, and whoever claimed it next
    would be finishing a job whose table had been emptied underneath it.
    """
    client, db, _store = worker
    _jetson_job(db, est_tets=20_000)
    db.jobs["job-1"]["lease_owner"] = "w-someone-else"
    db.modes["job-1"] = [{"index": 1, "k0_per_m": 1.0, "freq_hz": 2.0,
                          "vacuum_wavelength_m": 3.0}]
    response = client.post("/worker/jobs/job-1/done", headers=_auth(),
                           json={"modes": _rows(20)})
    assert response.status_code == 409
    assert db.modes["job-1"] == [{"index": 1, "k0_per_m": 1.0, "freq_hz": 2.0,
                                  "vacuum_wavelength_m": 3.0}], (
        "a refused /done deleted the rows the job already had")
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


# ---------------------------------------------------------------------------
# The multi-part estimate -- the routing-critical piece.
# ---------------------------------------------------------------------------
#
# The solver's own mesher module, imported by name here rather than reached
# through ``MESH``: ``web/meshing.py`` re-exports the estimate and the edge
# helper, but ``tetrahedralize`` and ``effective_edge_m`` in the length space
# the SOLVER uses live in the image's module. Importing it is what makes these
# tests speak about the mesh the solver builds rather than about the app's view
# of it.
import mse_common as COMMON  # noqa: E402
import mse_meshing as MESHSOLVER  # noqa: E402
import numpy as np  # noqa: E402
#
# These are tier-1: they mesh a 6 mm box with a 2 mm core, which is ~20k tets
# and takes a second or two. The route decision is made from this number, so a
# test that only checked "the estimate went up" would be the shape this rig
# keeps finding -- a check that cannot fail for the reason it exists.

def _nested_parts(units="mm"):
    """A 6 mm wall box with a 2 mm core, as a validated PartSet."""
    import trimesh

    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    return PARTS.load_parts(obj, units=units), PARTS.merged_surface(
        PARTS.load_parts(obj, units=units))


def test_the_wall_region_volume_excludes_its_children():
    """Decision 6, on the object that decides the route.

    A nested part's every point is inside its parent, so the parent's ENCLOSED
    volume counts it twice -- once as parent, once as child. The element count
    goes like V/h^3, so that inflation is what pushes a Jetson-sized job onto
    the cluster.

    Measured on this fixture: the merged PLC reports 224 mm^3 where the object
    fills 216 (6^3), because the 2 mm core is inside the 6 mm box.

    The mutation is "fill = enclosed": it must redden HERE and on the
    estimate test below.
    """
    parts, merged = _nested_parts()
    wall_fill_m3 = parts.filled_volume_m3(0, "mm")
    core_fill_m3 = parts.filled_volume_m3(1, "mm")

    assert core_fill_m3 == pytest.approx(8e-9, rel=1e-9), (
        "the core has nothing nested in it, so its fill IS its enclosed volume"
    )
    assert wall_fill_m3 == pytest.approx(208e-9, rel=1e-9), (
        f"the wall fills 6^3 - 2^3 = 208 mm3; got {wall_fill_m3 * 1e9:.4g}"
    )
    # And the DOUBLE COUNT the enclosed volume would give, named so the figure
    # in the comment above is measured rather than remembered.
    enclosed_m3 = abs(float(merged.volume)) * MESHSOLVER.unit_scale("mm") ** 3
    assert enclosed_m3 == pytest.approx(224e-9, rel=1e-9), (
        "the merged PLC's own volume should be 216 + 8 = 224 mm3 -- the core "
        "counted twice, which is exactly what filled_volume_m3 subtracts"
    )
    assert wall_fill_m3 < enclosed_m3, (
        "the wall's fill must be LESS than the merged enclosure; equal means "
        "the children were not subtracted"
    )


def test_the_multipart_estimate_is_pinned_to_the_mesh_the_solver_builds():
    """The estimate and the mesher must agree, in one length space.

    This is the row's own trap. The app's estimate ROUTES the job -- cluster
    (slow, cheap) vs Jetson (fast, scarce) -- and the mesher builds the mesh.
    If the two derive the per-region edge or the region volume independently,
    the app decides the route for a job nobody runs: a job that needs 19k
    elements is estimated at 3k and sent to the Jetson to die.

    So the assertion is not "the estimate grew". It is that the estimate and the
    count TetGen actually returns agree within the fitted coefficient's own
    slack -- and the multi-part estimate is asserted to be EXACTLY the one-part
    formula at h_min (here 8x the one-part count at h, because the core's index
    halves the edge), which is what makes this a test of the multi-part branch
    rather than of arithmetic both paths share.
    """
    parts, merged = _nested_parts()
    materials = {"wall": 1.0, "core": 2.0}

    estimate = MESH.estimate_tets(merged, 3, "mm", parts=parts,
                                  materials=materials)
    assert estimate > 0, "the multi-part estimate returned 0 (could not ask)"

    _points, tets, region_of_tet = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=3, n_index=1.0, max_tets=200_000, verbose=False,
        materials=materials, parts=parts,
    )
    actual = int(tets.shape[0])
    assert actual > 1000, "the fixture must be big enough for a ratio to mean"
    assert len(np.unique(region_of_tet)) == 2, (
        "the mesher must build two regions, or this is not the multi-part path"
    )

    ratio = estimate / actual
    # The slack is the coefficient's own: EST_TETS_C is calibrated against real
    # tetrahedralizations to within a small factor, and the whole point is that
    # the TWO PATHS share it. A ratio near 1.5 is the fitted factor; a ratio of
    # 0.15 is not a slack, it is the wrong formula.
    assert 0.5 <= ratio <= 3.0, (
        f"the estimate ({estimate:,}) and the mesh ({actual:,}) disagree by "
        f"{ratio:.2f}x -- the app would route this job on a number the mesher "
        f"does not confirm"
    )

    # THE DISCRIMINATOR. Under the uniform rule the multi-part estimate IS the
    # one-part formula evaluated at h_min, so the discriminating claim is about
    # the EDGE, not about two formulas existing: this object has a core at n=2 in
    # a wall at n=1, so h_min = h/2 and the estimate must be the one-part count
    # at that finer edge -- 2^3 = 8x the one-part count at h, same volume, same
    # coefficient. A branch that forgot the index (leaving h_min = h) gives 1x; a
    # branch that squared the edge instead of cubing gives 2x or 4x.
    #
    # This is deliberately NOT `one_part != estimate` as a bare inequality: that
    # passes for any accidental difference. It is the exact factor, which names
    # the mechanism.
    one_part_at_h = MESH.estimate_tets(merged, 3, "mm")
    assert one_part_at_h != estimate, (
        f"the one-part and multi-part estimates are both {estimate:,}, so this "
        f"test cannot tell whether the multi-part branch ran at all"
    )
    h = COMMON.effective_edge_m(merged, n_modes=3, units="mm")
    plan = COMMON.region_plan(parts, materials, h, "mm")
    h_min = plan[0]["edge_units"]
    v_cavity = sum(info["filled_volume_m3"] for info in plan.values())
    expected = int(MESH.EST_TETS_C * v_cavity / (h_min ** 3))
    assert estimate == expected, (
        f"the multi-part estimate ({estimate:,}) is not the one-part formula at "
        f"h_min ({expected:,}); the two paths must share the expression and the "
        f"calibration constant, or the route and the mesh disagree"
    )

    # THE VOLUME IS THE FILLED SUM, NOT THE MERGED SURFACE'S ENCLOSED VOLUME.
    # The merged PLC of a nested pair has its inner shell wound relative to the
    # CORE, so trimesh's oriented volume ADDS it: measured 2.24e-7 for a wall of
    # 216 mm^3 holding a core of 8 mm^3, where the cavity is 216 mm^3 and the
    # filled sum is 2.16e-7. Using the merged figure over-counts the core by
    # 2x (it appears once in the wall's enclosed volume and once again as its
    # own region) -- the double-count the filled volume exists to remove.
    # `enclosure_volume_m3` is the one-part path's input and stays that; it is
    # the wrong input HERE, and the gap is large enough to move a route.
    merged_enclosed = COMMON.enclosure_volume_m3(merged, "mm")
    assert merged_enclosed > v_cavity, (
        f"the merged enclosed volume ({merged_enclosed:g}) is not above the filled "
        f"sum ({v_cavity:g}); this test's premise about nested windings has changed"
    )
    assert merged_enclosed == pytest.approx(224.0 * 1e-9, rel=1e-9), (
        f"the merged enclosed volume is {merged_enclosed:g} m^3, not 224 mm^3 -- "
        f"the inner shell's winding (and so the double-count) has changed"
    )
    assert v_cavity == pytest.approx(216.0 * 1e-9, rel=1e-9), (
        f"the filled sum is {v_cavity:g} m^3, not 216 mm^3 -- 6^3 with the 2^3 "
        f"core removed"
    )
    octave = (h / h_min) ** 3
    assert octave == pytest.approx(8.0, rel=1e-9), (
        f"h/h_min cubed is {octave:g}, not 8 -- the fixture no longer has the "
        f"n=2-in-n=1 shape this test is built on"
    )
    # THE 8x, MEASURED AGAINST THE EXACT ONE-PART EXPRESSION, NOT ITS ROUNDED
    # COUNT. `estimate_tets` returns `int(c*V/h^3)`, so comparing against
    # `one_part_at_h * 8 * (vc/me)` would multiply an ALREADY-TRUNCATED integer:
    # the one-part count is int(2971.4508) = 2971, and 2971*8*(216/224) = 22919.1
    # against the true 22922.6 -- a 3.5-element gap invented entirely by where the
    # int() sits. The claim is about the FORMULA (the multi-part path is the
    # one-part formula at h_min on the filled volume), so both sides are compared
    # before truncation and the code's own int() is asserted separately below.
    exact_one_part = (MESH.EST_TETS_C * merged_enclosed / (h ** 3))
    exact_multi = MESH.EST_TETS_C * v_cavity / (h_min ** 3)
    assert exact_multi == pytest.approx(exact_one_part * 8.0 * (v_cavity / merged_enclosed),
                                        rel=1e-12), (
        f"the multi-part formula ({exact_multi:.4f}) is not the one-part formula "
        f"({exact_one_part:.4f}) rescaled by the two real changes -- the edge "
        f"({h/h_min:g}x finer, so 8x the count) and the volume ({v_cavity/merged_enclosed:.6f}x, "
        f"because the merged surface double-counts the nested core). That identity "
        f"IS the uniform rule."
    )
    # ...and the code's own truncation is the one it should be.
    assert estimate == int(exact_multi), (
        f"the estimate ({estimate:,}) is not int({exact_multi:.4f}) = "
        f"{int(exact_multi):,}; the branch is not computing the one-part formula "
        f"at h_min on the filled volume"
    )


def test_every_region_is_meshed_at_h_min_the_finest_region_edge():
    """The uniform rule: ONE edge for the whole cavity, the finest region's.

    THIS TEST HAS ASSERTED THREE DIFFERENT MODELS, and the history is the point.
    It first required ``effective_edge_m`` to return ``min(h / n_region)`` --
    the finest region's edge applied as the GLOBAL -- which meshed the WALL at
    the core's resolution and refused a GPU-sized upload as cluster-sized. Then
    it required the opposite: the global stays the WALL's edge and each region
    is fined RELATIVE to it. That model is right about the physics and cannot be
    implemented here: TetGen's per-region volume, ``add_region``'s ``max_vol``,
    was measured INERT on a nested PLC -- a nine-order sweep left the region's
    own element count byte-identical.

    So the model is now UNIFORM: every region is meshed at ``h_min = min_r(h *
    n_wall / n_r)``, floored at ``h``. ``effective_edge_m`` still returns the
    WEYL edge (index-independent, as its own docstring says); ``h_min`` lives in
    ``region_plan``, which the mesher AND the estimate both call.

    Three claims, each with its mutation named:
      * ``effective_edge_m`` does not move with either index. Mutation: make
        ``target_edge_length`` fold ``n_index`` in.
      * a denser core moves the plan's single edge DOWN, by exactly
        ``n_wall/n_core``. Mutation: collapse the ratio to 1.
      * a LESS dense core does not move it UP. Mutation: drop the ``min(h, ...)``
        floor, so a vacuum core coarsens the wall.
    """
    parts, merged = _nested_parts()

    vacuum = MESHSOLVER.effective_edge_m(merged, n_modes=3, units="mm",
                                         parts=parts,
                                         materials={"wall": 1.0, "core": 1.0})
    # ``target_edge_length`` states that n_index deliberately does not enter --
    # the geometric k is independent of the filling medium, and the in-medium
    # wavelength is handled downstream. So neither index moves this.
    for mats in ({"wall": 3.0, "core": 1.0}, {"wall": 1.0, "core": 4.0}):
        got = MESHSOLVER.effective_edge_m(
            merged, n_modes=3, units="mm", parts=parts, materials=mats
        )
        assert got == pytest.approx(vacuum, rel=1e-9), (
            f"materials {mats} moved the Weyl edge to {got:.6g} from "
            f"{vacuum:.6g}; it is index-independent by design"
        )

    base = vacuum / MESHSOLVER.unit_scale("mm")

    # Equal indices: the plan's one edge IS the Weyl edge.
    eq = COMMON.region_plan(parts, {"wall": 1.0, "core": 1.0}, base, "mm")
    assert {round(i["edge_units"], 12) for i in eq.values()} == {round(base, 12)}, (
        f"equal indices must give one edge equal to h, got "
        f"{[i['edge_units'] for i in eq.values()]}"
    )

    # A DENSER core pulls the single edge down by exactly n_wall/n_core.
    dense = COMMON.region_plan(parts, {"wall": 1.0, "core": 4.0}, base, "mm")
    edges = {i["edge_units"] for i in dense.values()}
    assert len(edges) == 1, (
        f"the plan must report ONE edge for every region, got {edges}"
    )
    assert next(iter(edges)) == pytest.approx(base / 4.0, rel=1e-9), (
        f"a core of n=4 must mesh the cavity at h*n_wall/n_core = "
        f"{base / 4.0:.6g}, got {next(iter(edges)):.6g}"
    )

    # A LESS dense core must NOT coarsen the wall: h_min is a floor.
    thin = COMMON.region_plan(parts, {"wall": 1.0, "core": 0.25}, base, "mm")
    thin_edges = {i["edge_units"] for i in thin.values()}
    assert len(thin_edges) == 1
    assert next(iter(thin_edges)) == pytest.approx(base, rel=1e-9), (
        f"a core of n=0.25 must not coarsen the cavity past h={base:.6g}; got "
        f"{next(iter(thin_edges)):.6g} -- h_min is a resolution floor, not a "
        f"function of the densest region alone"
    )


def _region_median_edges(points_m, tets, region_of_tet):
    """Per-region (median edge length in METRES, tet count) from a mesh.

    ``tetrahedralize`` returns its nodes in METRES. Measured two ways on a 6 mm
    box: the returned point extent is 0.006, which equals the box extent (6) x
    ``unit_scale("mm")`` = 0.006 m; and ``achieved_edge_length(points, tets)``
    gives 1.1687e-3 against a 1.169 mm verbose line, while
    ``achieved_edge_length(points * unit_scale, tets)`` gives 1.1687e-6.

    THIS HELPER ASSERTED BOTH WRONG ANSWERS BEFORE IT ASSERTED THIS ONE, which
    is why the unit is spelled out here rather than implied. Version 1 took the
    points as already metres and multiplied by ``unit_scale`` again; version 2
    took them as surface units and did not convert, then compared a metre median
    against a millimetre bound. Each made the resolution assertion below pass
    vacuously (a 1000x-small edge is under every bound), which is the
    could-not-ask shape, so the comparison is now also pinned from below.
    """
    from itertools import combinations

    pairs = np.array(list(combinations(range(4), 2)))
    out = {}
    for rid in np.unique(region_of_tet):
        sel = np.nonzero(region_of_tet == rid)[0]
        V = points_m[tets[sel]]
        e = np.linalg.norm(V[:, pairs[:, 0], :] - V[:, pairs[:, 1], :], axis=2).ravel()
        out[int(rid)] = (float(np.median(e)), int(sel.size))
    return out


def test_the_matrix_holds_with_real_tetgen_and_no_steiner_cap():
    """The lead's acceptance test 1: n_wall x n_core, REAL TetGen, 9 cells.

    Two things made an earlier version of this measurement lie, and both are
    removed here:

      * a ``steinerleft`` cap that bounded EVERY run identically, so nine very
        different meshes all returned the same count. There is no such cap
        here, and the per-region counts are printed so a capped run is visible
        in the log rather than inferred.
      * reading only the GLOBAL tet total, which the wall region dominates, so
        a change confined to the core was invisible. The assertion below is on
        the CORE'S OWN median edge.

    The claims, under the uniform rule:

    1. THE RESOLUTION REQUIREMENT. Each region r must be meshed no coarser than
       its own required edge ``h * n_wall / n_r``. Since ``h_min`` is
       ``min(h, h*n_wall/n_max)`` and ``n_r <= n_max``, the tightest case is
       ``r = argmax n`` -- so proving the tightest case proves them all, and
       the test asserts it on the DENSEST region's measured median edge, which
       is what actually failed under the old model.
    2. THE ESTIMATE AGREES. ``est / actual`` sits in the same band the one-part
       calibration is fitted to, because under (i') the two are the SAME
       expression at the same edge on the same volume. A band of [0.5, 3.0]
       would be too loose to catch a wrong formula; the one-part fit is
       conservative (~1.0-2.0 measured), so the band here is the same shape.

    Cost: the feature floor is set to 0.9 mm so the whole matrix is a few
    thousand to ~140k tets -- tier 1, seconds, no cluster.
    """
    parts, merged = _nested_parts()
    min_feature_m = 9.0e-4  # metres; floors the edge so the matrix stays tiny

    us = MESHSOLVER.unit_scale("mm")  # metres per surface unit
    h_base = MESHSOLVER.effective_edge_m(
        merged, n_modes=3, units="mm", min_feature_m=min_feature_m
    )
    # ``h_base`` is METRES and so are the returned points (see
    # ``_region_median_edges``), so there is ONE space here and no conversion.
    rows = []
    for n_wall in (1.0, 2.0, 3.0):
        for n_core in (1.0, 2.0, 3.5):
            materials = {"wall": n_wall, "core": n_core}
            plan = COMMON.region_plan(parts, materials, h_base, "mm")
            h_min = plan[0]["edge_units"]
            assert all(
                info["edge_units"] == h_min for info in plan.values()
            ), "the uniform rule means ONE edge for every region"

            points, tets, region_of_tet = MESHSOLVER.tetrahedralize(
                merged, "mm", n_modes=3, n_index=1.0, max_tets=500_000,
                min_feature_m=min_feature_m, verbose=False,
                materials=materials, parts=parts,
            )
            actual = int(tets.shape[0])
            # ``points`` are in the surface's own units (mm); no rescaling.
            med = _region_median_edges(points, tets, region_of_tet)
            # Region ids follow part order: 0 = wall (the enclosure), 1 = core.
            assert set(med) == {0, 1}, (
                f"n_wall={n_wall} n_core={n_core}: expected 2 regions, got "
                f"{sorted(med)}"
            )

            est = MESH.estimate_tets(
                merged, 3, "mm", parts=parts, materials=materials,
                min_feature_m=min_feature_m,
            )
            rows.append((n_wall, n_core, {
                "h_min_m": h_min, "n_tets": actual, "est": est,
                "wall": med[0], "core": med[1],
            }))

            # 1. THE RESOLUTION REQUIREMENT, on the DENSEST region (the core
            #    here, since n_core >= n_wall in every cell that has one). Its
            #    required edge is h * n_wall / n_core; its achieved median edge
            #    must be no coarser than that, times the mesh's own overshoot.
            #
            #    BOTH SIDES ARE IN SURFACE UNITS. A previous version compared a
            #    metre-space bound against a number it had already divided by
            #    1000, so every cell passed at ~1e-6 against ~0.9 -- a vacuous
            #    green. The lower bound below proves the bound can fail.
            h_core_required_m = h_base * n_wall / n_core
            achieved_core = med[1][0]
            assert achieved_core <= h_core_required_m * MESHSOLVER.MAX_EDGE_OVERSHOOT, (
                f"n_wall={n_wall} n_core={n_core}: the core is meshed at "
                f"{achieved_core * 1e3:.4f} mm but needs "
                f"{h_core_required_m * 1e3:.4f} mm (h_min="
                f"{h_min * 1e3:.4f} mm); the densest region is under-resolved, "
                f"which is the exact failure (i) would have shipped"
            )
            # POSITIVE CONTROL: the median edge must also be no finer than a
            # small fraction of the requirement. Without this half, a bound
            # written ``achieved <= required`` is satisfied by ANY fine mesh --
            # including one built at a 1000x-too-small edge, which is exactly
            # the unit bug this test already shipped once. Requiring the edge
            # to be within [required/10, required*overshoot] pins it from BOTH
            # sides, so a mesh at the wrong SCALE fails here even though it
            # would pass the upper bound alone.
            assert achieved_core >= h_core_required_m / 10.0, (
                f"n_wall={n_wall} n_core={n_core}: the core's median edge "
                f"{achieved_core * 1e3:.5f} mm is more than 10x FINER than "
                f"required ({h_core_required_m * 1e3:.5f} mm); that is not a mesh "
                f"this code asks for, it is a scale error, and the upper bound "
                f"alone would accept it"
            )
            assert med[1][1] > 0, (
                f"n_wall={n_wall} n_core={n_core}: the core got no tets at all, "
                f"so every other number here is about the wall only"
            )

            # 2. THE ESTIMATE. Same band the one-part calibration is fitted to.
            ratio = est / actual
            assert 0.5 <= ratio <= 3.0, (
                f"n_wall={n_wall} n_core={n_core}: est {est:,} vs actual "
                f"{actual:,} is {ratio:.3f}x, outside the calibrated band"
            )

    # PER-REGION COUNTS, PRINTED. A capped or collapsed run shows up here as
    # rows whose counts are identical across very different h_min -- which is
    # the signature of the instrument error this test exists to exclude.
    # ``med`` is in metres; printing at 1e3 gives mm, and the label says so.
    report = "\n".join(
        f"  n_wall={nw:>3} n_core={nc:>3}  h_min={r['h_min_m'] * 1e3:7.4f}mm  "
        f"n={r['n_tets']:>7,}  est={r['est']:>7,}  "
        f"wall(med {r['wall'][0] * 1e3:7.4f}mm, {r['wall'][1]:>6,} tets)  "
        f"core(med {r['core'][0] * 1e3:7.4f}mm, {r['core'][1]:>6,} tets)"
        for nw, nc, r in rows
    )
    print("\nMATRIX (real TetGen, no steinerleft cap):\n" + report)

    # A collapse detector, not decoration: the rows must not all share a count.
    counts = {r["n_tets"] for _nw, _nc, r in rows}
    assert len(counts) >= 3, (
        f"only {len(counts)} distinct tet counts across nine cells -- a cap or a "
        f"collapsed edge is bounding every run the same way:\n{report}"
    )


def test_equal_indices_give_the_same_count_as_the_one_part_mesh():
    """The lead's acceptance test 4. With one index everywhere the cavity is
    ONE material, so the multi-part mesh and the one-part mesh must agree.

    This is the test that catches ``eps_by_tet`` applied twice, or the region
    branch passing ``-A`` with a switch set that changes TetGen's element
    count. It is asserted on the COUNT rather than on bytes because the
    multi-part path must add the region attribute and the one-part path must
    not (``regionattrib=True`` changes TetGen's switches, so byte-identity
    across the two branches is not achievable -- only the one-part path against
    ITSELF is byte-identical, which the one-part regression test covers).

    Equal indices also mean ``h_min = h`` exactly, so the two meshes are asked
    for the SAME edge on the SAME PLC -- if the counts differ, the region
    branch changed something other than the tags.
    """
    parts, merged = _nested_parts()
    materials = {"wall": 1.0, "core": 1.0}
    min_feature_m = 9.0e-4

    h_base = MESHSOLVER.effective_edge_m(
        merged, n_modes=3, units="mm", min_feature_m=min_feature_m
    )
    plan = COMMON.region_plan(parts, materials, h_base, "mm")
    assert plan[0]["edge_units"] == pytest.approx(h_base, rel=1e-12), (
        "equal indices must leave h_min at h, or this test is not comparing "
        "like with like"
    )

    _p_multi, tets_multi, region_of_tet = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=3, n_index=1.0, max_tets=500_000,
        min_feature_m=min_feature_m, verbose=False,
        materials=materials, parts=parts,
    )
    _p_one, tets_one = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=3, n_index=1.0, max_tets=500_000,
        min_feature_m=min_feature_m, verbose=False,
    )[:2]

    assert len(np.unique(region_of_tet)) == 2, (
        "the region branch must still tag both parts even when their indices "
        "are equal; otherwise the mesh is right and the materials are lost"
    )
    assert tets_multi.shape[0] == tets_one.shape[0], (
        f"equal indices gave {tets_multi.shape[0]:,} tets through the region "
        f"branch and {tets_one.shape[0]:,} through the one-part branch; with "
        f"h_min = h and one material these are the same mesh"
    )


def test_the_dense_core_upload_routes_on_the_real_mesh_not_the_estimate(tmp_path):
    """The lead's acceptance test 2: {wall 2.0, core 3.5} through the REAL upload.

    The estimate is what routes, so this test does not compare the estimate to
    itself. It takes the job the app actually created -- its ``est_tets`` and its
    ``target`` -- and puts the DECISION next to the count a real TetGen returns
    for the same upload:

      * if the app sent it to the Jetson, a real mesh must fit the GPU's range
        (below the absolute cap), or the route sent it somewhere it cannot run;
      * if the app refused it, the real mesh must genuinely exceed the cap --
        the refusal has to be about a mesh that exists, not about an estimate
        nobody confirmed.

    The pair is chosen so h_min = h * 2/3.5 moves the estimate across the 15k
    GPU threshold: a job that the OLD model (wall edge only) would have called
    small. The discriminator is that the app's ``target`` is the GPU, which only
    happens because the core's index lowered h_min.
    """
    import trimesh

    wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
    core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    data = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")

    app, store, jobs_db, launcher = _make(tmp_path)
    payload = {
        "meshfile": (io.BytesIO(data), "nested.obj"),
        "units": "mm",
        "n_index": "1.0",
        "n_modes": "10",
        "part_index.wall": "2.0",
        "part_index.core": "3.5",
    }
    resp = app.test_client().post("/", data=payload,
                                  content_type="multipart/form-data")
    assert resp.status_code == 302, (
        "the upload must be accepted; a redirect means the form validation "
        "passed, which is itself part of what this test checks"
    )
    assert jobs_db.jobs, (
        "the upload created no job, so it was refused -- the flash said why; "
        "this test needs an ACCEPTED upload to compare a route against a mesh"
    )
    (job,) = jobs_db.jobs.values()

    # The upload REALLY carried the materials: a job with no materials table
    # would route through the one-part branch and this test would prove nothing.
    assert job.get("materials") == {"wall": 2.0, "core": 3.5}, (
        f"the job carried materials {job.get('materials')!r}, not the two "
        f"indices the form posted -- the multi-part branch did not run"
    )

    # The real mesh, at the same settings the upload used.
    parts, merged = _nested_parts()
    points, tets, region_of_tet = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=10, n_index=1.0, max_tets=CFG.max_tets(),
        verbose=False, materials={"wall": 2.0, "core": 3.5}, parts=parts,
    )
    real = int(tets.shape[0])
    assert len(np.unique(region_of_tet)) == 2

    if job["target"] == DB.JETSON:
        # ROUTED TO THE GPU: the real mesh must fit the GPU's range.
        assert real <= CFG.max_tets(), (
            f"the app sent this to the Jetson on est={job['est_tets']:,} but the "
            f"real mesh is {real:,} elements, over the {CFG.max_tets():,} cap "
            f"the GPU enforces -- the route sent it where it cannot run"
        )
        assert job["est_tets"] >= CFG.gpu_min_tets(), (
            "a Jetson target below the GPU threshold is not a route this code "
            "can produce, so the assertion above is about nothing"
        )
    elif job["target"] == DB.CLUSTER:
        # ROUTED TO THE CPU: only correct if the real mesh genuinely fits there,
        # because a dense core sent to the cluster dies at the memory limit.
        assert real <= CFG.cpu_max_tets(), (
            f"the app sent this to the cluster on est={job['est_tets']:,} but "
            f"the real mesh is {real:,} elements, over the CPU's "
            f"{CFG.cpu_max_tets():,} -- this is the dense-core-to-the-CPU "
            f"failure the multi-part estimate exists to prevent"
        )
    else:  # pragma: no cover - refusal path is asserted separately below
        raise AssertionError(f"unexpected target {job['target']!r}")

    # THE DISCRIMINATOR, on the decision itself: the core's n=3.5 is what moved
    # this across the GPU threshold. With the WALL edge alone (the old model)
    # the estimate would be the one-part number at h, which is below the
    # threshold -- so a target of JETSON is only reachable via h_min.
    one_part_est = MESH.estimate_tets(merged, 10, "mm")
    assert job["est_tets"] > one_part_est, (
        f"the multi-part estimate ({job['est_tets']:,}) is not above the "
        f"one-part one ({one_part_est:,}); the core's index did not move the "
        f"edge, so this upload would have routed the same under the old model "
        f"and the test proves nothing about the new one"
    )


def test_a_dense_core_upload_too_big_is_refused_on_the_real_mesh(tmp_path):
    """The other half of test 2: a refusal must be about a mesh that exists.

    The refusal threshold is TWICE the cap, so a refused upload must produce a
    real mesh that genuinely exceeds the cap. This test drives the REAL route
    with a coefficient raised to force the refusal, then builds the real mesh
    and shows it is over -- the refusal is not an estimate refusing itself.
    """
    import os

    import trimesh

    # The coefficient goes through the ENVIRONMENT, because that is where
    # ``CFG.est_tets_c()`` reads it: an ``app.config`` key is silently ignored
    # and the upload is accepted, which is how the first version of this test
    # failed -- a green-looking refusal test that refused nothing.
    os.environ["MSE_EST_TETS_C"] = "100000"
    try:
        wall = trimesh.creation.box(extents=(6.0, 6.0, 6.0))
        core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
        data = trimesh.Scene({"wall": wall, "core": core}).export(
            file_type="obj").encode("utf-8")
        app, store, jobs_db, launcher = _make(tmp_path)
        resp = app.test_client().post("/", data={
            "meshfile": (io.BytesIO(data), "nested.obj"),
            "units": "mm",
            "n_index": "1.0",
            "n_modes": "10",
            "part_index.wall": "2.0",
            "part_index.core": "3.5",
        }, content_type="multipart/form-data")
        assert resp.status_code == 302
        assert not jobs_db.jobs, (
            "a refused upload must create no job at all -- otherwise it is "
            "queued for a machine that cannot run it"
        )
    finally:
        os.environ.pop("MSE_EST_TETS_C", None)

    # AND THE REAL MESH, at the same settings, is genuinely large. The point is
    # not that it exceeds the (raised-coefficient) threshold -- that is
    # arithmetic -- but that a real TetraGen mesh of this upload exists and has
    # a definite size, so the refusal is about a real object.
    parts, merged = _nested_parts()
    _p, tets, _r = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=3, n_index=1.0, max_tets=2_000_000,
        verbose=False, materials={"wall": 2.0, "core": 3.5}, parts=parts,
    )
    assert int(tets.shape[0]) > 0, (
        "the refused upload must still be a meshable object; a refusal on "
        "something TetGen cannot mesh at all would be a different message"
    )


def test_meshing_at_the_walls_edge_leaves_the_dense_core_unresolved(monkeypatch):
    """The lead's acceptance test 3: the MUTATION.

    Mesh the whole cavity at the WALL's h instead of h_min -- i.e. drop the
    ``n_wall / n_max`` factor from ``region_edges`` -- and test 1's
    core-resolution assertion must go red, by name.

    This is the mutation that separates (i) from (i'): (i) meshed at the wall's
    edge and is exactly this. If the mutation does NOT redden the matrix's
    assertion, then that assertion is not testing the uniform rule and the whole
    of test 1 is decoration.

    The mutation is applied to ``region_edges`` itself, so it moves the EDGE the
    mesher is handed -- not a copy in the test.
    """
    import mse_common as COMMON

    parts, merged = _nested_parts()
    min_feature_m = 9.0e-4
    us = MESHSOLVER.unit_scale("mm")
    h_base = MESHSOLVER.effective_edge_m(
        merged, n_modes=3, units="mm", min_feature_m=min_feature_m
    )

    # THE ANCHOR: prove the mutation is applied before believing the red.
    original = COMMON.region_edges

    def wall_edge_only(h, n_wall, materials, parts_):
        # The mutation: every region at the wall's edge, no n_wall/n_max factor.
        return {int(r): float(h) for r in
                parts_.region_materials(materials)}

    # Apply it where region_plan reads it -- mse_common's own module global.
    monkeypatch.setattr(COMMON, "region_edges", wall_edge_only)
    assert COMMON.region_edges is wall_edge_only, (
        "the mutation did not take: region_plan would still call the original"
    )

    n_wall, n_core = 2.0, 3.5
    materials = {"wall": n_wall, "core": n_core}

    # Under the mutation the planned edge is h, not h*2/3.5.
    mutated_h_min = COMMON.region_plan(parts, materials, h_base, "mm")[0]["edge_units"]
    assert mutated_h_min == pytest.approx(h_base, rel=1e-12), (
        f"the mutation must leave the edge at the wall's h={h_base:.4f}, got "
        f"{mutated_h_min:.4f}"
    )
    true_h_min = h_base * n_wall / n_core
    assert mutated_h_min > true_h_min * 1.5, (
        "the mutated edge is not coarser than the true h_min, so the mutation "
        "is not (i) and this test is not testing the discrimination between "
        "them"
    )

    points, tets, region_of_tet = MESHSOLVER.tetrahedralize(
        merged, "mm", n_modes=3, n_index=1.0, max_tets=500_000,
        min_feature_m=min_feature_m, verbose=False,
        materials=materials, parts=parts,
    )
    med = _region_median_edges(points, tets, region_of_tet)
    achieved_core = med[1][0]
    h_core_required_m = h_base * n_wall / n_core

    # THE ASSERTION THAT MUST REDDEN -- the same predicate as test 1, on the
    # same quantity. It is written as an explicit ``not`` so the failure names
    # the mutation rather than reading as a generic assertion error.
    assert not (achieved_core <= h_core_required_m * MESHSOLVER.MAX_EDGE_OVERSHOOT), (
        f"the mutation did NOT redden the core-resolution assertion: with the "
        f"edge clamped to the wall's h={mutated_h_min * 1e3:.4f} mm, the core's "
        f"median edge is {achieved_core * 1e3:.4f} mm, which still satisfies the "
        f"required {h_core_required_m * 1e3:.4f} mm -- so test 1's assertion "
        f"does not discriminate the uniform rule from meshing at the wall"
    )


def _nested_parts_in(units, size_units):
    """The SAME physical object as ``_nested_parts``, written in ``units``.

    ``size_units`` is the object's size expressed in ``units``, so that every
    call describes one object: a 6x6x6 mm box is 0.6x0.6x0.6 cm and
    0.006x0.006x0.006 m. That is the whole point -- the estimate must not care
    which spelling it is handed.
    """
    import trimesh

    import mse_parts as PARTS

    wall = trimesh.creation.box(extents=(size_units, size_units, size_units))
    core = trimesh.creation.box(
        extents=(size_units / 3.0, size_units / 3.0, size_units / 3.0))
    obj = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")
    parts = PARTS.load_parts(obj, units=units)
    return parts, PARTS.merged_surface(parts)


def test_the_multipart_estimate_is_invariant_to_the_upload_units():
    """GATE (a): ONE physical object, six spellings, one estimate.

    This is the defect this test exists for. ``region_plan`` called
    ``PartSet.filled_volume_m3(part_index)`` without ``units``, so every
    multi-part upload's volume was converted as if written in millimetres while
    ``h_min`` beside it stayed true metres. The estimate was then wrong by the
    cube of the unit ratio -- measured through the real upload on one physical
    object: mm 51,187 (GPU), in 3 (cluster, 17,000x UNDER), um 5.1e13 and
    nm 5.1e19 (refused).

    Note the TWO directions of failure, which is why this needs a sweep and not
    one check: for a LARGER unit (m, in) the estimate collapses and a job the
    GPU should run is routed to the cluster, where it dies at the memory limit;
    for a SMALLER unit (um, nm) it explodes and the upload is REFUSED. The old
    test suite was green through all of it because every fixture was in mm --
    the default WAS the bug, so nothing that used the default could see it.

    The one-part path is checked beside it: it threaded units already, so it is
    the control that proves invariance is achievable on this fixture rather
    than being a property the assertion cannot have.
    """
    mm_size = 6.0
    cases = {
        "m": mm_size * 1e-3,
        "cm": mm_size * 1e-2,
        "mm": mm_size,
        "um": mm_size * 1e3,
        "nm": mm_size * 1e6,
        "in": mm_size / 25.4,
    }

    multi = {}
    one_part = {}
    for units, size in cases.items():
        parts, merged = _nested_parts_in(units, size)
        materials = {"wall": 1.0, "core": 2.0}
        edge = COMMON.effective_edge_m(merged, n_modes=10, units=units,
                                       parts=parts, materials=materials)
        multi[units] = COMMON.estimate_tets_by_region(
            parts, materials, edge, merged, units, MESH.EST_TETS_C)
        one_part[units] = MESH.estimate_tets(
            merged, n_modes=10, units=units, c=MESH.EST_TETS_C)

    base = multi["mm"]
    for units, value in multi.items():
        assert value == pytest.approx(base, rel=1e-9), (
            f"the MULTI-part estimate for one physical object depends on the "
            f"unit it was written in: {units} gives {value:,} where mm gives "
            f"{base:,} (ratio {value / max(base, 1):.4g}). The volume and the "
            f"edge are in different length spaces somewhere on this path."
        )

    base_one = one_part["mm"]
    for units, value in one_part.items():
        assert value == pytest.approx(base_one, rel=1e-9), (
            f"the ONE-part estimate is not invariant either: {units} gives "
            f"{value:,} vs mm {base_one:,}. This path threaded units already, "
            f"so a failure here means the regression is not the one this test "
            f"was written for."
        )

    # The estimate must also be non-degenerate: an invariance test is satisfied
    # by a constant, and a constant of 0 would route every job to the GPU.
    assert base > 1_000, (
        f"the multi-part estimate is {base:,}, which is too small to be a real "
        f"count -- a sweep that returns a constant would pass the invariance "
        f"assertions above for the wrong reason."
    )


def test_the_multipart_estimate_mutation_reddens_the_invariance_test():
    """GATE (c): drop ``units`` again and the invariance test must REDDEN.

    A mutation must be confirmed to APPLY before a green suite means anything,
    and it must redden the RIGHT test by name. Here the mutation restores the
    exact defect -- ``filled_volume_m3`` called without its unit -- so the
    assertion is not "a test failed" but "the test that exists for this failed".
    """
    import mse_parts as PARTS

    original = PARTS.PartSet.filled_volume_m3

    def drop_units(self, part_index, units):
        return original(self, part_index, "mm")

    PARTS.PartSet.filled_volume_m3 = drop_units
    try:
        with pytest.raises(AssertionError) as excinfo:
            test_the_multipart_estimate_is_invariant_to_the_upload_units()
    finally:
        PARTS.PartSet.filled_volume_m3 = original

    assert "depends on the unit it was written in" in str(excinfo.value), (
        f"the mutation reddened something, but not the unit-invariance "
        f"assertion: {excinfo.value}"
    )
    assert "in" in str(excinfo.value) or "nm" in str(excinfo.value), (
        f"the mutation reddened, but not on a NON-mm unit -- which would mean "
        f"the fixture is not actually exercising the default: {excinfo.value}"
    )


def test_a_nanometre_multipart_upload_is_not_refused_by_a_unit_bug(tmp_path):
    """GATE (b): the REAL route, a non-mm multi-part upload.

    The defaulted unit in ``filled_volume_m3`` made a small unit's volume
    explode: the same physical object measured as if in mm reads a volume
    1e9x (nm) or 1e18x (nm^3) too large, so the estimate crossed the refusal
    threshold and the owner was told their model was too detailed -- for a
    model that meshes to a few thousand elements.

    The upload is the SAME PHYSICAL OBJECT as the mm case, written in nm: a
    6 mm box is 6,000,000 nm across. What makes this a real test rather than a
    restatement is that the assertion is on the ROUTE the app took, and the
    mesh is built from the same bytes the app saw.

    Named for the nanometre case because that is the failure the owner would
    have hit -- "too detailed" for a tiny object -- but the sweep in the
    invariance test is what covers every unit.
    """
    import trimesh

    # 6 mm expressed in nm.
    size_nm = 6.0e6
    wall = trimesh.creation.box(extents=(size_nm, size_nm, size_nm))
    core = trimesh.creation.box(extents=(size_nm / 3.0, size_nm / 3.0,
                                         size_nm / 3.0))
    data = trimesh.Scene({"wall": wall, "core": core}).export(
        file_type="obj").encode("utf-8")

    app, store, jobs_db, launcher = _make(tmp_path)
    payload = {
        "meshfile": (io.BytesIO(data), "nested_nm.obj"),
        "units": "nm",
        "n_index": "1.0",
        "n_modes": "10",
        "part_index.wall": "1.0",
        "part_index.core": "2.0",
    }
    resp = app.test_client().post("/", data=payload,
                                  content_type="multipart/form-data")
    assert resp.status_code == 302, (
        "the upload must be ACCEPTED -- it is the same physical object as the "
        "mm case, which routes fine. A refusal here is the unit bug: the "
        "volume was converted as if the upload were in mm."
    )
    assert jobs_db.jobs, (
        "the nm upload created no job, so it was refused. The flash message "
        "would say 'too detailed'; the real mesh is a few thousand elements."
    )
    (job,) = jobs_db.jobs.values()
    assert job.get("units") == "nm", (
        f"the job's units are {job.get('units')!r}, not nm -- the route is not "
        f"testing what this test says it is"
    )

    # And the estimate must be the SAME as the mm case's, because it is the
    # same object. Computed here from the app's own stored value rather than
    # re-derived, so a drift between the two paths is visible.
    parts_mm, merged_mm = _nested_parts()
    edge_mm = COMMON.effective_edge_m(
        merged_mm, n_modes=10, units="mm",
        parts=parts_mm, materials={"wall": 1.0, "core": 2.0})
    est_mm = COMMON.estimate_tets_by_region(
        parts_mm, {"wall": 1.0, "core": 2.0}, edge_mm, merged_mm, "mm",
        MESH.EST_TETS_C)

    assert job["est_tets"] == pytest.approx(est_mm, rel=1e-6), (
        f"the nm upload estimated {job['est_tets']:,} where the identical mm "
        f"object estimated {est_mm:,} -- the estimate is unit-dependent, which "
        f"is the defect this test is for."
    )


def test_the_claim_payload_names_the_kind_and_carries_the_pc_request(tmp_path, monkeypatch):
    """The worker's only way to learn WHICH solve to run.

    The worker cannot read the job row -- it has no database access and no
    cluster access. `target` told it which machine to poll; `kind` is the only
    place it can learn whether to run the 3-D cavity solver or pc.py. A payload
    that dropped the key, or sent it as a constant, would make every
    photonic-crystal job silently run the wrong program -- and the two programs
    read completely different rows for their inputs, so the failure would be a
    band diagram that is not one rather than an error.

    Mutation that must turn this red: replace ``job.get("kind")`` in
    web/worker_api.py's claim payload with the literal ``"cavity"``.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    pcs = {"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5, "n_bg": 1.0,
           "n_bands": 12, "divisions": 16, "polarisation": "TM"}
    db.create_job("job-pc", None, "mm", 1.0, 12, target=DB.JETSON,
                  kind=DB.PC, pcs=pcs)
    app, _s, _d, _l = _make(tmp_path, db=db)

    body = app.test_client().post("/worker/claim",
                                  headers=_auth()).get_json()

    assert body["kind"] == DB.PC
    assert body["pcs"] == pcs, "the request must arrive as an object, not a string"


def test_the_claim_payload_says_cavity_for_a_cavity_job(tmp_path, monkeypatch):
    """The key is ALWAYS present, and a cavity job says so explicitly.

    Not a missing key defaulted by the worker: a worker that read absence as
    ``cavity`` would run the 3-D solver the day a proxy or an older app stopped
    sending the field, and the wrong answer would look like a right one.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    db.create_job("job-cav", "box.obj", "mm", 1.0, 20, target=DB.JETSON)
    app, _s, _d, _l = _make(tmp_path, db=db)

    body = app.test_client().post("/worker/claim",
                                  headers=_auth()).get_json()

    assert "kind" in body and body["kind"] == DB.CAVITY
    assert "pcs" in body and body["pcs"] is None
    assert "tds" in body and body["tds"] is None


def test_the_claim_payload_carries_the_td_kind_and_request(tmp_path, monkeypatch):
    """A td job reaches the worker with its kind and its request as an object.

    Mutation that must turn this red: drop the ``"tds"`` line from
    web/worker_api.py's claim payload.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    tds = {"boundary": "open", "source_kind": "point", "profile": "pulse",
           "frequency_hz": 4.0e14, "bandwidth": 0.5}
    db.create_job("job-td", "box.obj", "nm", 1.0, 0, target=DB.JETSON,
                  kind=DB.TD, tds=tds)
    app, _s, _d, _l = _make(tmp_path, db=db)

    body = app.test_client().post("/worker/claim",
                                  headers=_auth()).get_json()

    assert body["kind"] == DB.TD
    assert body["tds"] == tds, "the request must arrive as an object, not a string"


# ---------------------------------------------------------------------------
# The pc result path: a band diagram, not modes
# ---------------------------------------------------------------------------

#: The owner id the test token resolves to, read from the ROUTE's own
#: derivation rather than hand-rolled: a magic string that happened to match
#: would pass here and fail against a route that hashed differently.
_OWNER = WORKER.owner_id("tok-abc-123")

_PC_REQUEST = {"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5,
               "n_bg": 1.0, "n_bands": 3, "divisions": 16, "polarisation": "TM"}


def _pc_job(db, job_id="job-pc"):
    """A CLAIMED pc row with a real request in pcs.

    Status is RUNNING, not queued: the route only reaches the kind branch after
    ``_held`` sees a live lease, and a row that is queued carries no lease in
    the real store. Modelling it queued would let the tests pass against a row
    the route could never actually have been called on.
    """
    db.create_job(job_id, None, "mm", 1.0, 12, target=DB.JETSON,
                  kind=DB.PC, pcs=dict(_PC_REQUEST))
    db.jobs[job_id]["status"] = DB.RUNNING
    return db.jobs[job_id]


def _bands_body(n_k=3, n_bands=3, **over):
    """A valid ``/done`` body for a pc job.

    One k_point [kx, ky] and one band vector per k-point, every vector the same
    length -- the shape ``_band_payload`` requires and the legs below mutate.
    """
    body = {
        "labels": ["G", "X", "M", "G"][:n_k],
        "k_points": [[float(i) * 0.5, 0.0] for i in range(n_k)],
        "bands": [[float(i + b) for b in range(n_bands)] for i in range(n_k)],
    }
    body.update(over)
    return body


def test_a_pc_job_is_done_with_its_band_diagram(tmp_path, monkeypatch):
    """The pc branch of /done writes pcs, and does NOT touch mse.modes.

    The whole point of the kind branch: a pc job has no modes, and a route that
    tried to validate a modes list would 400 and never mark it done.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    job = _pc_job(db)
    job["lease_owner"] = _OWNER
    app, _s, _d, _l = _make(tmp_path, db=db)

    body = _bands_body()
    resp = app.test_client().post("/worker/jobs/job-pc/done",
                                  headers=_auth(), json={"bands": body})

    assert resp.status_code == 200, resp.data
    assert db.jobs["job-pc"]["status"] == DB.DONE
    assert db.jobs["job-pc"]["pcs"]["bands"] == body
    # The REQUEST survives the merge -- a pc row that lost its request could
    # not say what was solved.
    assert db.jobs["job-pc"]["pcs"]["lattice"] == "square"
    # No mode rows: the pc path never reaches finish_jetson_with_modes.
    assert str("job-pc") not in db.modes


def test_a_pc_done_without_bands_is_refused_and_leaves_the_job_running(
        tmp_path, monkeypatch):
    """A malformed diagram is a 400, and the job is NOT marked done.

    The deploy-3 shape, pc edition: a DONE with an empty diagram reads as "a
    finished job with no bands" rather than "a job that never reported them".
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    job = _pc_job(db)
    job["lease_owner"] = _OWNER
    app, _s, _d, _l = _make(tmp_path, db=db)

    resp = app.test_client().post("/worker/jobs/job-pc/done",
                                  headers=_auth(), json={})

    assert resp.status_code == 400, resp.data
    assert db.jobs["job-pc"]["status"] == DB.RUNNING
    assert "bands" not in (db.jobs["job-pc"].get("pcs") or {})


@pytest.mark.parametrize("bad,raw", [
    ({"labels": [], "k_points": [], "bands": []}, None),
    ({"labels": ["G"], "k_points": [[0.0, 0.0], [0.5, 0.0]], "bands": [[1.0]]}, None),
    ({"labels": ["G", "X"], "k_points": [[0.0, 0.0]], "bands": [[1.0], [2.0]]}, None),
    ({"labels": ["G"], "k_points": [[0.0]], "bands": [[1.0]]}, None),
    ({"labels": [1], "k_points": [[0.0, 0.0]], "bands": [[1.0]]}, None),
    ({"labels": ["G"], "k_points": [[0.0, 0.0]], "bands": [[]]}, None),
    # NaN/Infinity cannot go through ``json=``: Flask's encoder REFUSES them
    # before the route sees them, so the test would prove the encoder, not the
    # guard. They arrive as a real float through a raw body, exactly as
    # ``test_done_refuses_a_nan_frequency`` sends them -- a bare literal that
    # ``json.loads`` accepts and hands back as a float.
    ({"labels": ["G"], "k_points": [[0.0, 0.0]], "bands": [[1.0]]},
     ('"bands": [[1.0]]', '"bands": [[1.0, NaN]]')),
    ({"labels": ["G"], "k_points": [[0.0, 0.0]], "bands": [[1.0]]},
     ('"k_points": [[0.0, 0.0]]', '"k_points": [[0.0, Infinity]]')),
])
def test_a_ragged_or_non_finite_diagram_is_refused(bad, raw, tmp_path,
                                                   monkeypatch):
    """Every part of the shape is checked: lengths agree and numbers are real.

    A diagram is only meaningful if its three lists agree and its numbers are
    finite. Each case here is valid element-by-element and wrong as a whole,
    which is exactly what a per-value type check would wave through.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    job = _pc_job(db)
    job["lease_owner"] = _OWNER
    app, _s, _d, _l = _make(tmp_path, db=db)

    if raw is None:
        resp = app.test_client().post("/worker/jobs/job-pc/done",
                                      headers=_auth(), json={"bands": bad})
    else:
        payload = json.dumps({"bands": bad}).replace(*raw)
        assert raw[1].split(": ")[1] in payload, "the literal never made it in"
        resp = app.test_client().post("/worker/jobs/job-pc/done",
                                      headers=_auth(), data=payload,
                                      content_type="application/json")

    assert resp.status_code == 400, resp.data
    assert db.jobs["job-pc"]["status"] == DB.RUNNING


def test_a_cavity_job_still_takes_the_modes_branch(tmp_path, monkeypatch):
    """A bands body on a CAVITY job is refused; kind decides, not the body.

    The branch is on the claimed ROW's kind, so a cavity job cannot be finished
    with a diagram, and a pc job cannot be finished with mode rows -- the two
    result shapes are not interchangeable by payload.
    """
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    _jetson_job(db, est_tets=1000)
    db.jobs["job-1"].update(lease_owner=_OWNER, status=DB.RUNNING)
    app, _s, _d, _l = _make(tmp_path, db=db)

    # A bands body against a cavity job: the pc branch is skipped, _mode_rows
    # sees no 'modes' key and refuses.
    resp = app.test_client().post("/worker/jobs/job-1/done",
                                  headers=_auth(),
                                  json={"bands": _bands_body()})
    assert resp.status_code == 400, resp.data
    assert db.jobs["job-1"]["status"] == DB.RUNNING

    # And the cavity body still works, unchanged.
    ok = app.test_client().post("/worker/jobs/job-1/done",
                                headers=_auth(), json={"modes": _rows(20)})
    assert ok.status_code == 200, ok.data
    assert db.jobs["job-1"]["status"] == DB.DONE


def test_a_pc_done_on_a_lost_lease_is_a_409(tmp_path, monkeypatch):
    """The merge and the DONE are one unit: a lost lease writes neither."""
    monkeypatch.setenv("MSE_WORKER_TOKEN", "tok-abc-123")
    db = LeaseDb()
    _pc_job(db)
    db.jobs["job-pc"]["lease_owner"] = "someone-else"
    app, _s, _d, _l = _make(tmp_path, db=db)

    resp = app.test_client().post("/worker/jobs/job-pc/done",
                                  headers=_auth(),
                                  json={"bands": _bands_body()})

    assert resp.status_code == 409, resp.data
    assert "bands" not in (db.jobs["job-pc"].get("pcs") or {})


# ---------------------------------------------------------------------------
# A td job never leaves the Jetson (FDTD 4.2)
#
# The cluster runs only the cavity solver, so every path that can change a
# row's ``target`` is driven here with a td row: the unclaimed-GPU fallback
# (the tick's abandon pass), the reaper, lease expiry and release. The fake's
# ``reroute_to_cluster`` deliberately has NO kind guard (the real SQL does), so
# the abandon test reaches the app's own decision. tests/td_routing_check.py
# proves the same properties against real Postgres.
# ---------------------------------------------------------------------------

_TDS = {"boundary": "open", "source_kind": "point", "profile": "pulse"}


def _td_job(db, job_id="job-td"):
    db.create_job(job_id, "u.obj", "um", 1.0, 0, target=DB.JETSON,
                  est_tets=1_000, kind=DB.TD, tds=dict(_TDS))
    return db.jobs[job_id]


def test_an_unclaimed_td_job_stays_on_the_gpu_queue(tmp_path):
    """Red on the pc-only guard: the td row is moved and logged as moved."""
    db = RoutingDb()
    _td_job(db)
    logs: list[str] = []

    moved = APP._reroute_abandoned_gpu_jobs(db, FakeLauncher(), log=logs.append)

    assert moved == 0
    assert db.jobs["job-td"]["target"] == DB.JETSON
    assert db.reroutes == []
    assert any("'td' job" in line and "stays queued for the GPU" in line
               for line in logs), logs


def test_the_abandon_pass_still_moves_a_cavity_row_beside_a_td_row(tmp_path):
    """The whitelist must not freeze the cavity fallback it sits in."""
    db = RoutingDb()
    _td_job(db)
    _jetson_job(db, est_tets=20_000, job_id="job-cav")

    moved = APP._reroute_abandoned_gpu_jobs(db, FakeLauncher(), log=lambda _m: None)

    assert moved == 1
    assert db.jobs["job-cav"]["target"] == DB.CLUSTER
    assert db.jobs["job-td"]["target"] == DB.JETSON


def test_an_expired_td_lease_requeues_it_on_the_gpu(tmp_path):
    db = RoutingDb()
    job = _td_job(db)
    job.update(lease_owner="w-dead", lease_expired=True, status=DB.RUNNING)

    # The reclaim step alone: the whole tick would also run the launch pass,
    # whose FakeDb.next_queued ignores target and kind (the real one takes
    # cluster cavity rows only), so it would "launch" the td row in the double.
    reclaimed = APP._reclaim_expired_leases(db, log=lambda _m: None)

    assert reclaimed == ["job-td"]
    assert db.jobs["job-td"]["target"] == DB.JETSON
    assert db.jobs["job-td"]["status"] == DB.QUEUED


def test_the_reaper_does_not_fail_a_running_td_job(tmp_path):
    """A td row has no k8s Job by construction; its lease owns its liveness."""
    db = RoutingDb()
    job = _td_job(db)
    job.update(status=DB.RUNNING, lease_owner="w-1", k8s_job_name=None)

    failed = APP._fail_dead_jobs(FakeLauncher(), db, log=lambda _m: None)

    assert failed == 0
    assert db.jobs["job-td"]["status"] == DB.RUNNING
    assert db.jobs["job-td"]["target"] == DB.JETSON


def test_the_real_store_refuses_a_td_job_off_the_jetson():
    """Refused before any connection, so no later path has to rescue the row."""
    store = DB.JobStore({"host": "invalid.invalid", "dbname": "x"})
    with pytest.raises(ValueError, match="Jetson only"):
        store.create_job("job-td", "u.obj", "um", 1.0, 0, target=DB.CLUSTER,
                         kind=DB.TD, tds=dict(_TDS))


def _busy_td_job(db, job_id="td-1"):
    db.create_job(job_id, "u.obj", "um", 1.0, 0, target=DB.JETSON,
                  est_tets=1_000, kind=DB.TD,
                  tds={"boundary": "open", "source_kind": "point"})
    db.jobs[job_id]["lease_owner"] = WORKER.owner_id("tok-abc-123")
    return db.jobs[job_id]


def test_busy_requeues_a_td_job_without_spending_a_release(tmp_path, worker):
    """4.2: a Jetson short of GPU memory is no verdict on the job."""
    client, db, _store = worker
    _busy_td_job(db)
    resp = client.post("/worker/jobs/td-1/busy", headers=_auth(),
                       json={"message": "3.1 GB free, 6.0 GB needed"})
    assert resp.status_code == 200
    assert resp.get_json() == {"ok": True, "deferred": True, "failed": False}
    row = db.jobs["td-1"]
    assert row["status"] == DB.QUEUED and row["lease_owner"] is None
    assert row["release_count"] == 0, "a busy deferral must not spend a release"
    assert row["stage"] == DB.BUSY_STAGE
    assert "6.0 GB needed" in row["message"]
    assert f"{DB.BUSY_WAIT_MINUTES} minutes" in row["message"]


def test_busy_on_a_lease_this_worker_does_not_hold_is_409(tmp_path, worker):
    client, db, _store = worker
    _busy_td_job(db)["lease_owner"] = "w-someone-else"
    resp = client.post("/worker/jobs/td-1/busy", headers=_auth(), json={})
    assert resp.status_code == 409
    assert db.jobs["td-1"]["lease_owner"] == "w-someone-else"


def test_busy_past_the_wait_fails_the_job_readably(tmp_path, worker):
    client, db, _store = worker
    _busy_td_job(db)["busy_minutes"] = DB.BUSY_WAIT_MINUTES
    resp = client.post("/worker/jobs/td-1/busy", headers=_auth(), json={})
    assert resp.get_json() == {"ok": True, "deferred": False, "failed": True}
    row = db.jobs["td-1"]
    assert row["status"] == DB.FAILED
    assert row["message"] == DB.BUSY_GAVE_UP_MESSAGE
    assert "60 minutes" in row["message"], "the owner is told how long it waited"
