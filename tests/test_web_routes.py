"""The web app's routes and its one-at-a-time scheduler.

The app takes its store, database and Job launcher by injection, so all of this
runs with no cluster, no MinIO and no Postgres. The point of these tests is the
scheduling contract, which is easy to get subtly wrong and expensive to debug in
the cluster: exactly one Job at a time, a 403 quota is a wait rather than a
failure, and a Job is deleted once its result is recorded.
"""

from __future__ import annotations

import io
import json
import re
import sys
import types
import uuid
from pathlib import Path

import numpy as np
import pytest
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "web"))

import app as APP  # noqa: E402
import config as CFG  # noqa: E402
import db as DB  # noqa: E402
import fields as FLD  # noqa: E402
import jobs as JOBS  # noqa: E402
import outline as OUTLINE  # noqa: E402
import storage as STORE  # noqa: E402


# ---------------------------------------------------------------------------
# Doubles for the three collaborators
# ---------------------------------------------------------------------------
class MemoryStore(STORE.LocalStore):
    """A LocalStore with the bucket semantics the app assumes."""

    def __init__(self, root):
        super().__init__(root)

    def keys_under(self, job_id: str) -> list[str]:
        """The names stored under ``jobs/<job_id>/``, for tests that assert on
        WHICH files arrived rather than only that the job finished."""
        base = Path(self.root) / "jobs" / job_id
        if not base.is_dir():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_file())


class FakeDb:
    """The parts of JobStore the routes and the tick actually use."""

    def __init__(self):
        self.jobs: dict[str, dict] = {}
        self.modes: dict[str, list] = {}
        self.failures: list[tuple] = []
        self.launched: list[tuple] = []

    # writes
    def create_job(self, job_id, filename, units, n_index, n_modes,
                   target=DB.CLUSTER, est_tets=None,
                   min_feature_m=None, min_feature_value=None,
                   min_feature_unit=None, scale=1.0, materials=None,
                   kind=DB.CAVITY, pcs=None, tds=None):
        # All THREE feature columns, because the real store has all three: the
        # number the owner typed, the unit they chose, and the canonical metres
        # the solver gets. A double that kept only the metres would pass a test
        # in which the app wrote a value the database silently drops -- and the
        # two typed fields are what the job page shows back to the owner.
        #
        # ``materials`` is stored as the object it was given, or None. It is NOT
        # defaulted to {}: the real store writes SQL NULL for None and the two
        # mean different things downstream (no table vs an empty table), so a
        # double that collapsed them would let a route test pass while the
        # solver received an empty materials table it was never sent.
        self.jobs[str(job_id)] = {
            "job_id": str(job_id), "created_at": "2026-09-26T00:00:00+00:00",
            "original_filename": filename, "units": units, "n_index": n_index,
            "n_modes": n_modes, "status": DB.QUEUED, "stage": DB.QUEUED,
            "message": None, "k8s_job_name": None, "peak_rss_mb": None,
            "finished_at": None,
            "target": target, "est_tets": est_tets,
            "min_feature_m": min_feature_m,
            "min_feature_value": min_feature_value,
            "min_feature_unit": min_feature_unit,
            "scale": scale,
            "materials": materials,
            # ``kind`` and ``pcs`` are modelled because the real store has
            # them and the claim payload now carries them. Left out, this
            # double would raise TypeError the moment a route test enqueued a
            # pc job -- the same class of hole as the missing ``import json``
            # that 500'd multi-part uploads against 591 green tests.
            "kind": kind,
            "pcs": pcs,
            # The time-domain request, modelled for the reason pcs is.
            "tds": tds,
        }

    def mark_running(self, job_id, name):
        self.jobs[str(job_id)].update(status=DB.RUNNING, stage="meshing",
                                      k8s_job_name=name)
        self.launched.append((str(job_id), name))

    def set_k8s_job_name(self, job_id, name):
        self.jobs[str(job_id)]["k8s_job_name"] = name

    def mark_failed(self, job_id, message):
        # The REAL store clears the lease here (see DB.JobStore.mark_failed) so a
        # worker holding the row learns it is over from its next heartbeat. The
        # double has to model that, or a test asserting the worker is stopped
        # would pass against a double that never stopped it.
        self.jobs[str(job_id)].update(status=DB.FAILED, stage=DB.FAILED,
                                      message=message, lease_owner=None,
                                      lease_expires_at=None)
        self.failures.append((str(job_id), message))

    # -- the Jetson lease, modelled on the real store ----------------------
    #
    # These exist so a test can put a row in the state a real GPU worker leaves
    # it in -- CLAIMED, running, holding a lease, with NO k8s Job. That state is
    # the one the second dispatch deploy died on and the one nothing in this
    # suite could previously produce, because a FakeDb with no claim path can
    # only ever hold cluster rows.
    def claim_jetson(self, owner, lease_seconds):
        """Claim the oldest claimable jetson row, as the real store does."""
        for job in self.jobs.values():
            if (job["target"] == DB.JETSON and job["status"] == DB.QUEUED
                    and not job.get("lease_owner")):
                job["lease_owner"] = owner
                job["lease_expires_at"] = "2099-01-01T00:00:00+00:00"
                job["status"] = DB.RUNNING
                job["stage"] = "claimed"
                return dict(job)
        return None

    def renew_lease(self, job_id, owner, lease_seconds):
        """Renew iff ``owner`` holds it -- False is the 409 the worker stops on."""
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job["lease_expires_at"] = "2099-01-01T00:00:00+00:00"
        return True

    def lease_held_by(self, job_id, owner):
        job = self.jobs.get(str(job_id))
        return bool(job and job.get("lease_owner") == owner)

    # The three TERMINAL worker verbs, so this double matches the store the
    # harness and the routing tests drive.
    #
    # SCOPE, because it is easy to over-read: they were missing here, but the
    # route BODIES were not untested -- ``tests/test_worker.py`` drives the same
    # blueprint on a real server against ``LeaseDb``, which has all three. What
    # was missing is that anything using THIS double (the tick tests here, and
    # ``scripts/jetson_worker_e2e.py``) hit an AttributeError the app turned into
    # a 500. That is a REAL finding about the harness -- a 500 with no reason is
    # exactly what made deploy 3 undiagnosable -- but it is not a hole in the
    # worker-route coverage, and saying it was would be the same overclaim as
    # reading a green suite as evidence a path works.
    #
    # All three return the real store's own bool/str, because the ROUTES branch
    # on those values (``release_jetson`` -> None becomes a 409, "failed"
    # becomes a terminal response) and a double that returned something else
    # would test a branch the real store cannot take.
    def finish_jetson(self, job_id, owner):
        """Mark done iff ``owner`` still holds it; the real store's bool."""
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(status=DB.DONE, stage=DB.DONE, finished_at="2026-09-26T00:00:00Z",
                   message=None, lease_owner=None, lease_expires_at=None)
        return True

    def finish_jetson_with_modes(self, job_id, owner, rows):
        """Mark done AND record the rows, as the real store does in one transaction.

        The rows are recorded HERE rather than by the route, because the whole
        point of the paired write is that a caller cannot get DONE without
        modes. A double that marked the job done and dropped the rows would let
        a route that forgot to pass them still go green -- the fake would be the
        only thing that noticed, and it would not notice.
        """
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(status=DB.DONE, stage=DB.DONE, finished_at="2026-09-26T00:00:00Z",
                   message=None, lease_owner=None, lease_expires_at=None)
        self.modes[str(job_id)] = [
            {"index": int(i), "k0_per_m": float(k0), "freq_hz": float(f),
             "vacuum_wavelength_m": float(w)}
            for (i, k0, f, w) in rows
        ]
        return True

    def fail_jetson(self, job_id, owner, message):
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return False
        job.update(status=DB.FAILED, stage=DB.FAILED, message=message,
                   finished_at="2026-09-26T00:00:00Z",
                   lease_owner=None, lease_expires_at=None)
        self.failures.append((str(job_id), message))
        return True

    def release_jetson(self, job_id, owner, message=""):
        """``"released"``, ``"failed"`` at the bound, or ``None`` if not held.

        The BOUND is modelled, not just the release: a double that always
        returned "released" would make the route's terminal branch unreachable
        and the crash-loop bound untestable at this level.
        """
        job = self.jobs.get(str(job_id))
        if not job or job.get("lease_owner") != owner:
            return None
        if int(job.get("release_count", 0)) >= DB.MAX_RELEASES:
            job.update(status=DB.FAILED, stage=DB.FAILED, message=message,
                       finished_at="2026-09-26T00:00:00Z",
                       lease_owner=None, lease_expires_at=None)
            return "failed"
        job.update(status=DB.QUEUED, stage=DB.QUEUED, message=message or None,
                   lease_owner=None, lease_expires_at=None,
                   release_count=int(job.get("release_count", 0)) + 1)
        return "released"

    def clear_terminal_leases(self):
        """The startup repair: terminal rows stop looking leased. Returns a count."""
        cleared = 0
        for job in self.jobs.values():
            if job.get("status") in DB.FINISHED and job.get("lease_owner"):
                job["lease_owner"] = None
                job["lease_expires_at"] = None
                cleared += 1
        return cleared

    # reads
    def get_job(self, job_id):
        return self.jobs.get(str(job_id))

    def list_jobs(self, limit=25):
        return list(self.jobs.values())[:limit]

    def list_modes(self, job_id):
        return self.modes.get(str(job_id), [])

    def next_queued(self):
        for job in self.jobs.values():
            if job["status"] == DB.QUEUED:
                return job
        return None

    def queue_position(self, job_id):
        job = self.jobs.get(str(job_id))
        if not job or job["status"] != DB.QUEUED:
            return None
        order = [j["job_id"] for j in self.jobs.values()
                 if j["status"] == DB.QUEUED]
        return order.index(str(job_id)) + 1

    def count_running(self):
        return sum(1 for j in self.jobs.values() if j["status"] == DB.RUNNING)

    def reclaim_expired_leases(self):
        """Expire leases past their deadline, as the real store does.

        The ``lease_expired`` flag is this double's stand-in for
        ``lease_expires_at < now()``.

        **The ``status != running`` filter mirrors web/db.py's, and it is the
        point of this method existing at all.** An earlier version returned
        ``[]`` unconditionally, which is a double that cannot disagree with the
        code: it made the 20:15Z resurrection unreachable in the unit suite, so
        no test here could have caught it and none can now. With the filter
        modelled, reverting the real SQL alone still leaves the suite green --
        the real predicate is pinned on Postgres by ``pg_race_check.py`` D3 --
        but a tick or double that requeues a TERMINAL row shows up here.

        The rest of the healthy case is unchanged and still worth keeping: the
        tick calls this on every pass, and "the double has no such method" is an
        AttributeError, which the tick's middle ``except`` correctly reads as a
        BROKEN STATEMENT -- degrading the scheduler over an omission in a test
        double. A double that does not model a method the real store has would
        make ``/healthz`` report degraded for a reason that cannot happen in
        production, so the double has to answer it.
        """
        requeued = []
        for job in self.jobs.values():
            if not job.get("lease_expired"):
                continue
            if job.get("status") != DB.RUNNING:
                continue
            if int(job.get("release_count", 0)) >= DB.MAX_RELEASES:
                job.update(status=DB.FAILED, stage=DB.FAILED,
                           message="the GPU worker stopped responding and the "
                                   f"job was retried up to {DB.MAX_RELEASES} "
                                   "times; try again later",
                           lease_owner=None, lease_expires_at=None)
                continue
            job.update(lease_owner=None, lease_expires_at=None,
                       status=DB.QUEUED, stage=DB.QUEUED,
                       release_count=int(job.get("release_count", 0)) + 1)
            requeued.append(job["job_id"])
        return requeued

    # set a job up as finished, with a mode table and field files on disk
    def finish(self, job_id, store, n_modes=2, achieved_median_edge_m=None):
        grid = {"nx": 4, "ny": 3, "nz": 2, "origin": [0.0, 0.0, 0.0],
                "spacing": [1.0, 1.0, 1.0]}
        self.jobs[str(job_id)].update(status=DB.DONE, stage=DB.DONE,
                                      finished_at="2026-09-26T00:10:00+00:00")
        rows = []
        for index in range(1, n_modes + 1):
            rows.append({"index": index, "k0_per_m": 1.0e3 * index,
                         "freq_hz": 1.0e11 * index,
                         "vacuum_wavelength_m": 3.0e-3 / index})
            field = np.ones((2, 3, 4, 3), dtype="<f4") * index
            store.write(STORE.job_key(job_id, f"mode_{index:02d}.bin"),
                        field.tobytes())
        self.modes[str(job_id)] = rows
        manifest = {"units": "mm", "n_index": 1.0, "n_tets": 6,
                    "n_edges": 12, "grid": grid, "modes": rows}
        # Only written when the solver measured one, mirroring the real
        # ``write_results``: a manifest that always carried the key would hide
        # the "no measurement yet" branch behind a null.
        if achieved_median_edge_m is not None:
            manifest["achieved_median_edge_m"] = achieved_median_edge_m
        store.write(STORE.job_key(job_id, "modes.json"),
                    json.dumps(manifest).encode())
        store.write(STORE.job_key(job_id, "surface.glb"), b"glTF")
        return rows


class FakeLauncher:
    """A launcher whose cluster state is a set of names.

    ``failed`` is the set of Job names the cluster reports as FAILED, which is
    how the tick tells a crashed solve (OOM/deadline) from a live one.

    ``cannot_ask`` models a 403 on the list calls. It raises
    ``JOBS.CannotAskCluster`` from both, because that is what the real launcher
    does -- and it must NOT return an empty result, because the whole point of
    the fix is that "we were not allowed to look" and "there is nothing there"
    are different answers. A double that returned ``[]``/``{}`` here would model
    the bug rather than the contract.

    ``swallowed_403`` is the mode the 403 tests need to run against, and it is
    not ``cannot_ask``: it is the *pre-fix* launcher, which answered a 403 with
    an empty result. The two are different doubles on purpose. ``cannot_ask``
    raises, so it models the contract every launcher must honour and can only
    show that the callers handle "we could not ask" -- a launcher that raises is
    already fixed whatever the callers do. ``swallowed_403`` models the actual
    defect, so a test claiming to pin it fails when it comes back.
    """

    def __init__(self, active=None, quota_full=False, fail_with=None,
                 failed=None, cannot_ask=False, swallowed_403=False,
                 raise_on=None):
        self.active = list(active or [])
        self.failed = dict(failed or {})
        self.quota_full = quota_full
        self.fail_with = fail_with
        self.cannot_ask = cannot_ask
        self.swallowed_403 = swallowed_403
        self.raise_on = raise_on
        self.created: list[dict] = []
        self.deleted: list[str] = []
        self.dry_run = False

    def active_job_names(self):
        if self.cannot_ask or self.raise_on == "active_job_names":
            raise JOBS.CannotAskCluster("not allowed to list solver jobs")
        if self.swallowed_403:
            return []                      # the e5b2e3e behaviour: 403 read as empty
        return list(self.active)

    def job_states(self):
        """Only the names still present can be reporting a state.

        This class doubles the *launcher*, not the Kubernetes client, so it
        answers ``job_states`` outright and never exercises ``_failed_reason``
        or the pod list. That is why it did not catch the ``status.failed``
        TypeError and is not expected to: the double that hid that bug is
        ``_FakeApi`` in tests/test_web_app.py, which stands in for the client and
        used to hand back pods from a field the real ``V1JobStatus`` does not
        have. Tests for the real client shapes belong there, against real
        ``kubernetes.client`` objects -- see
        ``test_a_real_job_status_whose_failure_is_only_a_count_does_not_raise``.
        """
        if self.cannot_ask or self.raise_on == "job_states":
            raise JOBS.CannotAskCluster("not allowed to list solver jobs")
        if self.swallowed_403:
            return {}                      # the e5b2e3e behaviour
        return {n: r for n, r in self.failed.items() if n in self.active}

    def create(self, job_id, units, n_index, n_modes,
               min_feature_m=None, scale=1.0, kind=None):
        if self.quota_full:
            raise JOBS.QuotaExceeded("exceeded quota: count/jobs.batch")
        if self.fail_with is not None:
            raise self.fail_with
        name = JOBS.job_name(job_id)
        self.created.append({"job_id": str(job_id), "units": units,
                             "n_index": n_index, "n_modes": n_modes,
                             "min_feature_m": min_feature_m, "scale": scale,
                             # The kind the tick read off the row and handed
                             # the launcher. Recorded so a test can assert the
                             # real launcher is FED the kind -- the refusal in
                             # Launcher.create is dead code if run_tick never
                             # passes it, and that is a coverage gap a fake
                             # without this parameter would hide.
                             "kind": kind,
                             # recorded so a test can assert the OPTIONAL flag is
                             # absent, not merely falsy -- the real manifest omits
                             # --min-feature rather than passing a zero, and that
                             # distinction is what keeps 'auto' meaning 'no opinion'
                             "args": list(JOBS.build_job_manifest(
                                 job_id, units, n_index, n_modes, "img", "sec",
                                 "ns", min_feature_m=min_feature_m,
                                 scale=scale)["spec"]["template"]["spec"]
                                 ["containers"][0]["args"]),
                             "name": name})
        self.active.append(name)
        return name

    def delete(self, name):
        self.deleted.append(name)
        if name in self.active:
            self.active.remove(name)


def make_app(tmp_path, launcher=None, db=None):
    """Build an app wired to in-memory doubles."""
    store = MemoryStore(tmp_path)
    jobs_db = db or FakeDb()
    launcher = launcher or FakeLauncher()
    app, _store, _db, _launcher = APP.create_app({
        "TESTING": True, "MSE_LOCAL_STORE": str(tmp_path),
        "MSE_DB_MOCK": jobs_db, "MSE_JOB_LAUNCHER": launcher,
    })
    return app, store, jobs_db, launcher


def closed_box_bytes():
    return trimesh.creation.box(extents=[10, 7, 5]).export(file_type="obj").encode()


def upload(client, filename="box.obj", data=None, units="mm", n_index="1.0",
           **extra):
    """POST the upload form.

    ``**extra`` carries the optional controls (``detail``, ``scale``) and is the
    reason this helper does not have to grow a parameter every time the form
    does: the fields are form values, and a test that names the one it cares
    about reads better than a positional list nobody can remember.
    """
    fields = {
        "meshfile": (io_bytes(data if data is not None else closed_box_bytes()),
                     filename),
        "units": units, "n_index": n_index,
    }
    fields.update(extra)
    return client.post("/", data=fields, content_type="multipart/form-data")


def upload_to_the_cluster(client, monkeypatch, **extra):
    """Upload a row that the routing decision sends to the CLUSTER.

    WHY THIS EXISTS. ``closed_box_bytes()`` estimates at ~19.8k tets, which is
    over ``MSE_GPU_MIN_TETS`` (15k), so the plain ``upload()`` helper routes to
    the **Jetson** -- by design. Tests about cluster-Job behaviour (a
    ``DeadlineExceeded`` condition, an OOMKilled container, a Job lost with its
    node) were therefore running their assertions against a *jetson* row. They
    passed only because ``FakeDb.next_queued`` ignored the target and handed the
    jetson row to the cluster launcher, a state the real app cannot produce.

    That is the unfaithful-double half of the dispatch rollback: with a jetson
    row carrying a ``k8s_job_name``, the reaper's ``target == 'jetson'`` skip
    never fired in these tests. Raising the GPU threshold makes the routing
    decision pick the cluster for the same fixture, which is what the test
    actually means to exercise.
    """
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    return upload(client, **extra)


def io_bytes(data):
    import io

    return io.BytesIO(data)


# ---------------------------------------------------------------------------
def test_a_transient_list_error_then_a_good_list_behaves_normally(tmp_path, monkeypatch):
    """One failed list must not cost the solve its life.

    A blip -- a timeout, a 403 that clears, a rolling API server -- makes one
    tick unable to see the cluster. The row must survive that tick untouched, and
    the next good list must reconcile it normally, because the Job really is
    still there. This is the reviewer's case in the shape the owner would meet
    it: nothing failed, nothing was deleted, and the solve ran to completion.

    The failure is injected by flipping ``cannot_ask`` between the two ticks.
    That means this test passes on ``e5b2e3e`` too, and it is worth saying why
    rather than finding out later: the swallow makes the *first* tick fail the
    row, and this test only inspects the row's survival, which is a difference
    it cannot see -- both a fixed and a broken launcher leave the second tick
    looking the same. It is a regression test for the reconcile-not-crash path,
    not a discriminator for the 403 defect. The launcher's own half -- the raise
    that stops the empty answer being produced -- is covered behaviourally in
    ``test_web_app.py``, which reaches the real 403 branch through a fake
    ``_client``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert job["status"] == DB.RUNNING

    launcher.cannot_ask = True
    first = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert first["failed"] == 0, "a blip must not fail a live solve"
    assert first["launched"] is None, "and must not launch into an unknown slot"
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert name not in launcher.deleted

    # the blip clears; the Job is still there and healthy
    launcher.cannot_ask = False
    second = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert second["failed"] == 0, "the Job is present and not failed"
    assert second["deleted"] == []
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert launcher.active == [name]


def test_one_call_raising_while_the_other_answers_still_fails_nothing(tmp_path, monkeypatch):
    """The guarantee is "no answer", not the placeholder, and not both calls failing.

    These two list calls are made separately, and only one of them has to fail
    for the tick to be unable to answer. The dangerous shape is the *split* one:
    ``active_job_names`` raises -- so the launch gate shuts on the placeholder --
    while ``job_states`` answers ``{}``, an empty dict that says "every Job is
    healthy, none has failed". Crossed against the placeholder that is not an
    empty list, ``_fail_dead_jobs`` still reaches its ``else`` branch and fails
    the live row.

    So the protection is not the placeholder and it is not "both calls raise
    together". It is that a call which cannot answer raises, and the tick treats
    a raise as a reason to stop. That holds per-call, which is what these two
    tests check -- one for each direction, since either call can be the one that
    fails first.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert job["status"] == DB.RUNNING

    # active_job_names cannot answer; job_states answers "nothing has failed"
    launcher.raise_on = "active_job_names"
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0, (
        "job_states answering {} must not be read as 'no Job here failed'"
    )
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert db.jobs[job["job_id"]]["message"] is None, (
        "a live solve must not be failed off a split answer"
    )
    assert name not in launcher.deleted


def test_a_split_answer_the_other_way_round_also_fails_nothing(tmp_path):
    """The mirror: ``job_states`` raises while ``active_job_names`` answers empty.

    This is the closer call of the two. An empty name list is a legitimately
    empty cluster, so the launch gate is entitled to open -- and it does, in
    ``run_tick``, before anyone knows that ``job_states`` is about to raise.
    What must not happen is a running row being failed on that tick: the raise
    from ``job_states`` arrives at ``_fail_dead_jobs``' handler and the tick
    fails nothing.

    Note what this test does *not* assert: that nothing was launched. Launching
    on a genuinely empty list is correct behaviour, and this launcher's empty
    list is indistinguishable from one. Only the failure decision is in scope,
    and that is the decision the reviewer's defect got wrong.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]

    launcher.raise_on = "job_states"
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert db.jobs[job["job_id"]]["message"] is None
    assert name not in launcher.deleted


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def test_healthz(tmp_path):
    """Liveness is ``status``; ``degraded`` is the scheduler's separate health.

    The two are deliberately different fields: a failing reclaim does not make
    the process down (pages still serve, the cluster fallback still runs), but
    it must not be invisible either -- the tick's return value is ``[]`` whether
    nothing expired or the statement is broken, so ``/healthz`` has to carry it.
    """
    app, *_ = make_app(tmp_path)
    assert app.test_client().get("/healthz").get_json() == {
        "status": "ok", "degraded": False,
    }


def test_index_lists_nothing_initially(tmp_path):
    app, *_ = make_app(tmp_path)
    body = app.test_client().get("/").get_data(as_text=True)
    assert "No objects yet" in body
    assert "Compute modes" in body


def test_an_upload_is_queued_launched_and_redirected(tmp_path):
    app, store, db, launcher = make_app(tmp_path)
    response = upload(app.test_client())
    assert response.status_code == 302
    assert "/jobs/" in response.headers["Location"]

    (job,) = db.jobs.values()
    assert job["status"] == DB.RUNNING
    assert job["original_filename"] == "box.obj"
    assert job["n_modes"] == 20
    # the file really landed in the bucket under the contract's key
    assert store.read(STORE.job_key(job["job_id"], "input.obj")) == closed_box_bytes()
    assert len(launcher.created) == 1


def test_the_upload_form_options_reach_the_job(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client(), units="cm", n_index="1.5")
    (job,) = db.jobs.values()
    assert (job["units"], job["n_index"]) == ("cm", 1.5)
    assert launcher.created[0]["units"] == "cm"
    assert launcher.created[0]["n_index"] == 1.5


def test_the_detail_option_reaches_the_solver_not_just_the_form(tmp_path):
    """The field the whole row exists for: it must not stop at the row.

    The feature was accepted and validated and then dropped, so the mesh was
    built at the physics edge while the route was computed as if it were not.
    This asserts the full path the lead asked for -- the row AND the Job the
    solver gets handed -- because passing one and not the other is exactly the
    half-wired state that produced the bug.

    The units are the point of the round: ``0.2`` typed with the default unit
    (mm) is stored as ``2e-4`` METRES for the solver, and the typed pair is
    kept beside it so the page can say it back in the owner's own words.
    """
    app, _store, db, launcher = make_app(tmp_path)
    # 0.2 mm on the test's box is below the physics edge but still inside the
    # cap. Worth knowing: a feature fine enough to blow the cap is REFUSED at
    # upload rather than queued, which is the conservative-estimate behaviour
    # doing its job -- 0.05 mm on this box estimates 39.2M elements.
    upload(app.test_client(), detail="0.2", feature_unit="mm")
    (job,) = db.jobs.values()

    assert job["min_feature_m"] == pytest.approx(2e-4)
    assert job["min_feature_value"] == 0.2
    assert job["min_feature_unit"] == "mm"
    assert launcher.created[0]["min_feature_m"] == pytest.approx(2e-4)
    assert "--min-feature" in launcher.created[0]["args"]


def test_the_job_page_shows_the_feature_in_the_unit_the_owner_typed(tmp_path):
    """The page is where the owner checks what they asked for.

    The feature's unit is its own, so the page must print the TYPED unit and
    never the object's -- a page that re-spelled 25 nm as "2.5e-05 mm" would be
    arithmetically true and useless to read. This is also the render check the
    rig insists on: an HTTP 200 is not evidence that an element is present, so
    the response body is parsed for the element and its text.
    """
    for value, unit, want_m, want_text in (
        ("0.2", "mm", 2e-4, "0.2 mm"),
        ("25", "nm", 2.5e-8, "25 nm"),
        ("0.025", "um", 2.5e-8, "0.025 um"),
    ):
        app, _store, db, _launcher = make_app(tmp_path / f"{value}{unit}")
        client = app.test_client()
        nm_box = trimesh.creation.box(extents=[200.0, 120.0, 80.0])
        upload(client, data=nm_box.export(file_type="obj").encode(),
               units="nm", detail=value, feature_unit=unit)
        (job,) = db.jobs.values()

        assert job["min_feature_m"] == pytest.approx(want_m), (
            f"{value} {unit} stored the wrong length; and note 25 nm and "
            f"0.025 um are the SAME physical length, which is why they share an "
            f"expectation"
        )
        assert job["min_feature_unit"] == unit

        body = client.get(f"/jobs/{job['job_id']}").get_data(as_text=True)
        found = re.search(r'<span id="resolution">(.*?)</span>', body, re.S)
        assert found is not None, "the resolution element is absent from the page"
        assert " ".join(found.group(1).split()) == \
            f"requested mesh resolution {want_text}"


def test_the_job_page_reports_the_achieved_edge_once_the_run_is_finished(tmp_path):
    """The achieved median edge reaches the owner, in their own unit.

    This drives the REAL path the lead named: the solver's ``modes.json``, read
    back through ``web/fields.load_modes_json`` by the job route, rendered into
    the page. The caption is computed by ``_resolution_caption``, which
    re-expresses a metre measurement in the unit the owner TYPED -- and which
    used to reference a bare ``MESH`` that does not exist in ``app.py``. That
    was a NameError on the ``achieved_m`` branch, so every finished job with a
    feature set would have 500'd. An HTTP 200 would not have caught it: the
    branch only runs when a job is DONE *and* its manifest carries a
    measurement, so the test must lay down both.

    The second half is the honest-statement requirement: when the mesh landed
    above the request, say so, and say whether it is inside the 1.6x allowance
    the solver enforces. A bare ratio the reader has to judge reads as a defect.
    """
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    box = trimesh.creation.box(extents=[200.0, 120.0, 80.0])
    upload(client, data=box.export(file_type="obj").encode(),
           units="nm", detail="5", feature_unit="nm")
    (job,) = db.jobs.values()

    # 1. BEFORE the run: no measurement exists, and the page says so rather
    #    than inventing a number.
    body = client.get(f"/jobs/{job['job_id']}").get_data(as_text=True)
    assert "appears here once the run finishes" in " ".join(body.split())
    assert "median element edge" not in body

    # 2. The solver finishes and the worker writes the measurement into the
    #    manifest, exactly as write_results does.
    achieved_m = 6.1e-9          # 6.1 nm, i.e. above the 5 nm requested
    db.finish(job["job_id"], store, achieved_median_edge_m=achieved_m)

    response = client.get(f"/jobs/{job['job_id']}")
    assert response.status_code == 200, (
        "the finished job page raised -- check _resolution_caption's unit "
        "lookup, which must use the module-level unit_scale_m"
    )
    detail = re.search(r'<p class="muted small" id="resolution-detail">(.*?)</p>',
                       response.get_data(as_text=True), re.S)
    assert detail is not None, "the resolution detail is absent from the page"
    text = " ".join(detail.group(1).split())

    # The achieved edge is in the OWNER'S unit (nm), not the object's and not
    # metres: 6.1e-9 m must read as "6.1 nm".
    assert "6.1 nm" in text, text
    assert "1.22" in text, f"the over-request ratio is missing or wrong: {text}"
    assert "1.6" in text, f"the allowance must be stated: {text}"

    # 3. And the measurement came through the REAL reader, so it is the value
    #    the solver wrote, not a number this test's store manufactured.
    manifest = FLD.load_modes_json(
        store.read(STORE.job_key(job["job_id"], "modes.json")))
    assert manifest["achieved_median_edge_m"] == achieved_m


def test_the_achieved_caption_honours_the_feature_unit_not_the_objects(tmp_path):
    """The same physical measurement, three spellings, three true sentences.

    The feature carries its OWN unit, so re-expressing the achieved median in
    that unit is the only correct reading -- and the conversion goes through the
    same table the form converted the input with. 0.27 mm, 270 um and 2.7e5 nm
    are one length; a caption that printed metres would be true and unreadable,
    and one that printed the object's units would describe a different object.
    """
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    box = trimesh.creation.box(extents=[0.01, 0.007, 0.005])
    upload(client, data=box.export(file_type="obj").encode(), units="m",
           detail="0.2", feature_unit="mm")
    (job,) = db.jobs.values()

    db.finish(job["job_id"], store, achieved_median_edge_m=2.7e-4)
    body = client.get(f"/jobs/{job['job_id']}").get_data(as_text=True)
    assert "0.27 mm" in body, f"achieved edge not in the typed unit: {body[:400]}"


def test_auto_leaves_the_feature_unset_rather_than_zero(tmp_path):
    """'auto' must be None, not 0.0 -- 0 is an edge length, and a wrong one.

    A stored 0 would reach the solver as a declared feature of zero, which the
    solver's own guard would then have to rescue. The absence has to stay an
    absence; this is the could-not-ask shape at the form boundary.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client(), detail="auto")
    (job,) = db.jobs.values()

    assert job["min_feature_m"] is None
    assert launcher.created[0]["min_feature_m"] is None
    # and the job the solver is handed must not carry the flag at all
    assert "--min-feature" not in launcher.created[0].get("args", [])


def test_the_scale_reaches_the_row_and_the_job(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client(), scale="1000")
    (job,) = db.jobs.values()

    assert job["scale"] == 1000.0
    assert launcher.created[0]["scale"] == 1000.0


def test_an_absent_scale_is_one_not_none(tmp_path):
    """Blank must mean 1.0, so nothing downstream ever multiplies by None.

    And the flag must be OMITTED, not passed as 1.0: an unscaled job's Job
    spec has to stay byte-identical to what it was before the scale control
    existed, because that spec is what the pre-existing tests were written
    against and a gratuitous extra argument is the kind of change that quietly
    invalidates their expectations.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()

    assert job["scale"] == 1.0
    assert launcher.created[0]["scale"] == 1.0
    assert "--scale" not in launcher.created[0]["args"], (
        "an unscaled job must not carry --scale; the omission is the contract"
    )
    assert "--min-feature" not in launcher.created[0]["args"]


def test_a_refused_scale_says_what_the_limit_is(tmp_path):
    """The message has to be actionable, so it carries the actual bounds."""
    app, _store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    client.post("/", data={
        "meshfile": (io_bytes(closed_box_bytes()), "box.obj"),
        "units": "mm", "n_index": "1.0", "scale": "1e12",
    }, content_type="multipart/form-data")
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    assert any("between" in m for m in flashes), flashes
    assert any("1000000000" in m or "1e+09" in m for m in flashes), flashes
    assert not db.jobs


# ---------------------------------------------------------------------------
# Rejected uploads
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("filename,data,units,n_index,needle", [
    ("model.stl", None, "mm", "1.0", ".obj"),
    ("box.obj", b"", "mm", "1.0", "empty"),
    # trimesh does not raise on junk text; it yields a mesh with no faces
    ("box.obj", b"not a mesh", "mm", "1.0", "no triangles"),
    ("box.obj", None, "furlongs", "1.0", "units"),
    ("box.obj", None, "mm", "0.5", "1.0 or greater"),
    ("box.obj", None, "mm", "not-a-number", "must be a number"),
])
def test_a_bad_upload_is_refused_without_creating_anything(
        tmp_path, filename, data, units, n_index, needle):
    app, store, db, launcher = make_app(tmp_path)
    client = app.test_client()
    response = upload(client, filename=filename, data=data,
                      units=units, n_index=n_index)
    assert response.status_code == 302          # back to the form
    assert db.jobs == {}                        # no row
    assert launcher.created == []               # no Job
    # follow the redirect, the way the browser does: the reason is delivered as
    # a flashed message on the next render, not in this response
    body = client.get(response.headers["Location"]).get_data(as_text=True)
    assert needle.lower() in body.lower()


def test_a_non_watertight_upload_is_refused(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    box = trimesh.creation.box(extents=[10, 7, 5])
    open_mesh = trimesh.Trimesh(vertices=np.array(box.vertices),
                                faces=np.array(box.faces)[:-1], process=False)
    client = app.test_client()
    response = upload(client, data=open_mesh.export(file_type="obj").encode())
    assert response.status_code == 302
    assert db.jobs == {}
    assert launcher.created == []
    body = client.get(response.headers["Location"]).get_data(as_text=True)
    assert "closed surface" in body


# ---------------------------------------------------------------------------
# The status JSON the page polls
# ---------------------------------------------------------------------------
def test_status_json_for_a_running_job(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    payload = app.test_client().get(f"/jobs/{job['job_id']}/status").get_json()
    assert payload["status"] == DB.RUNNING
    assert payload["stage"] == "meshing"
    assert "modes" not in payload          # nothing to show yet


def test_status_json_for_a_finished_job_includes_the_modes(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    rows = db.finish(job["job_id"], store)
    payload = app.test_client().get(f"/jobs/{job['job_id']}/status").get_json()
    assert payload["status"] == DB.DONE
    assert [m["index"] for m in payload["modes"]] == [r["index"] for r in rows]
    assert payload["manifest_url"].endswith("/files/modes.json")
    assert payload["surface_url"].endswith("/files/surface.glb")


def test_status_json_for_a_failed_job_carries_the_message(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.mark_failed(job["job_id"], "This object is too large or detailed to "
                                  "resolve 20 modes.")
    payload = app.test_client().get(f"/jobs/{job['job_id']}/status").get_json()
    assert payload["status"] == DB.FAILED
    assert "too large" in payload["message"]


def test_status_json_reports_the_queue_position(tmp_path):
    app, store, db, launcher = make_app(tmp_path)
    launcher.active = ["something-else"]
    launcher.quota_full = False
    upload(app.test_client())
    first = next(iter(db.jobs.values()))
    # a second upload while the first is queued must show position 2
    first.update(status=DB.QUEUED, stage=DB.QUEUED)
    upload(app.test_client())
    second = [j for j in db.jobs.values() if j["job_id"] != first["job_id"]][0]
    payload = app.test_client().get(f"/jobs/{second['job_id']}/status").get_json()
    assert payload["queue_position"] == 2


def test_an_unknown_job_is_a_404(tmp_path):
    app, *_ = make_app(tmp_path)
    client = app.test_client()
    assert client.get(f"/jobs/{uuid.uuid4()}").status_code == 404
    assert client.get(f"/jobs/{uuid.uuid4()}/status").status_code == 404
    assert client.get("/jobs/not-a-uuid").status_code == 404


# ---------------------------------------------------------------------------
# Proxied result files
# ---------------------------------------------------------------------------
def test_result_files_are_proxied_when_the_job_is_done(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    client = app.test_client()

    manifest = client.get(f"/jobs/{job['job_id']}/files/modes.json")
    assert manifest.status_code == 200
    assert manifest.get_json()["n_tets"] == 6

    binary = client.get(f"/jobs/{job['job_id']}/files/mode_01.bin")
    assert binary.status_code == 200
    assert binary.mimetype == "application/octet-stream"
    # 2*3*4 samples * 3 components * 4 bytes
    assert len(binary.data) == 2 * 3 * 4 * 3 * 4


def test_result_files_are_not_served_before_the_job_finishes(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    response = app.test_client().get(
        f"/jobs/{job['job_id']}/files/surface.glb")
    assert response.status_code == 404


@pytest.mark.parametrize("name", ["../secret", "input.obj", "mode_1.bin",
                                  "mode_9999.bin", "run.sh", "modes.json.bak"])
def test_only_solver_output_filenames_are_proxied(tmp_path, name):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    response = app.test_client().get(f"/jobs/{job['job_id']}/files/{name}")
    assert response.status_code == 404, name


def test_a_missing_file_is_a_404_not_a_500(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    path = Path(store.root) / STORE.job_key(job["job_id"], "surface.glb")
    path.unlink()
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/surface.glb").status_code == 404


# ---------------------------------------------------------------------------
# The .vtu download
# ---------------------------------------------------------------------------
# The viewer renders the field itself, so the only reason to fetch a .vtu is to
# carry it away and open it in ParaView. That makes this route's job different
# from the proxy's: the bytes have to arrive intact *and* be saved rather than
# displayed, under a name a person can find again on a downloads shelf.
def _plant_vtu(store, job_id, name="mode_01.vtu", data=b"<VTKFile>...</VTKFile>"):
    store.write(STORE.job_key(job_id, name), data)


def test_a_vtu_is_offered_as_a_download_under_a_name_the_upload_earned(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    upload(client, filename="my lens.obj")
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    _plant_vtu(store, job["job_id"])

    response = client.get(f"/jobs/{job['job_id']}/files/mode_01.vtu")

    assert response.status_code == 200
    assert response.data == b"<VTKFile>...</VTKFile>"
    # as_attachment is the whole point: without it the browser renders the XML.
    assert "attachment" in response.headers["Content-Disposition"]
    disposition = response.headers["Content-Disposition"]
    assert "my-lens-mode01.vtu" in disposition, disposition
    # Not the solver's name: twenty `mode_01.vtu` from twenty jobs are
    # indistinguishable once they are all sitting in a downloads folder.
    assert "mode_01.vtu" not in disposition.replace("my-lens-mode01.vtu", "")


@pytest.mark.parametrize("filename", [
    "lens.obj", "a/b/c.obj", "….obj", "lens..obj",
    "  spaced name .obj", "-leading.obj", "no-extension-here.obj",
])
def test_the_download_name_is_a_plain_filename_whatever_was_uploaded(
        tmp_path, filename):
    """A name from the upload reaches Content-Disposition, so it may not be one.

    A path separator in the filename header makes some browsers write a file
    somewhere other than the downloads folder, and the escaping rules for the
    header are inconsistent enough that the safe move is to leave nothing in the
    name that needs escaping. Whatever comes in, what goes out is one path
    segment of ``[A-Za-z0-9._-]``.
    """
    app, store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    upload(client, filename=filename)
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    _plant_vtu(store, job["job_id"])

    from app import _vtu_download_name
    name = _vtu_download_name(job, "mode_01.vtu")

    assert "/" not in name and "\\" not in name, name
    assert name.endswith("-mode01.vtu"), name
    assert name == Path(name).name, f"{name!r} is not a single path segment"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", name), name


def test_a_vtu_the_job_does_not_have_is_a_404(tmp_path):
    """A missing .vtu is a missing file, not an empty download.

    ``send_file`` with no data would be a 0-byte attachment that opens as a
    corrupt mesh in ParaView -- a failure the user has to debug themselves.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)

    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/mode_01.vtu").status_code == 404


def test_a_vtu_is_not_downloadable_before_the_solver_wrote_it(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    _plant_vtu(store, job["job_id"])

    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/mode_01.vtu").status_code == 404


def test_an_unreachable_store_does_not_hand_back_a_partial_vtu(tmp_path):
    """The store failing must not read as a file that simply is not there.

    A 404 here would tell the user the solver never wrote the mode; the truth is
    that we could not look. This route already distinguishes them, so the test
    exists to keep it that way.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)
    _plant_vtu(store, job["job_id"])

    def unreachable(key):
        raise STORE.StorageError("the store is unreachable")

    # The route closes over the store object built at create_app time, so the
    # one it will actually call is the one in the app's extensions.
    app.extensions["mse"]["store"].read = unreachable
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/files/mode_01.vtu").status_code == 502


@pytest.mark.parametrize("name", [
    "mode_01.vtu.bak", "../mode_01.vtu", "mode_1.vtu", "mode_001.vtu",
    "mode_01.VTU", "sub/mode_01.vtu", "mode_01.vtu/../modes.json",
])
def test_a_vtu_has_to_be_a_vtu_the_solver_writes(tmp_path, name):
    """The whitelist is what makes traversal impossible, so prove it holds.

    Nothing shaped like a path can satisfy the pattern at all, which is why the
    guard is a full match rather than a sanitizer: there is no escaping step to
    get wrong.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store)

    response = app.test_client().get(f"/jobs/{job['job_id']}/files/{name}")
    assert response.status_code == 404, name


# ---------------------------------------------------------------------------
# The viewer's data endpoint
# ---------------------------------------------------------------------------
def test_the_index_names_every_mode_and_carries_no_field_data(tmp_path):
    """The index is the list, not the payload.

    This is the contract the lazy load exists for: one small response the phone
    can fetch immediately, with the multi-megabyte arrays behind a second,
    per-mode request. If a bulk ``slices`` ever comes back here, the payload has
    silently regressed to the 11 MB shape it was split to avoid.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    payload = app.test_client().get(
        f"/jobs/{job['job_id']}/index.json").get_json()
    assert [m["index"] for m in payload["modes"]] == [1, 2, 3]
    for mode in payload["modes"]:
        assert "slices" not in mode and "cloud" not in mode
        assert mode["available"] is True
        assert mode["freq_hz"] is not None
    assert payload["units"] == "mm"
    assert payload["grid"]["nx"] > 0


def test_one_mode_payload_holds_that_mode_and_only_that_mode(tmp_path):
    """One mode's whole vector field, as float16, and no other mode's.

    The viewer turns this back into the mode's *scalar* field itself, so the
    only thing the route has to get right is that the bytes are one mode's, in
    the grid's shape, and that they are the mode that was asked for. A payload
    that carried the wrong mode would draw a perfectly plausible picture of the
    wrong thing.
    """
    import gzip

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    grid = {"nx": 4, "ny": 3, "nz": 2}
    count = grid["nx"] * grid["ny"] * grid["nz"] * 3

    response = app.test_client().get(f"/jobs/{job['job_id']}/field/2.bin")
    assert response.status_code == 200
    assert response.headers["X-Field-Dtype"] == "float16-le"
    # Two bytes a sample, exactly: a longer body would mean the decoder kept a
    # dtype it was not asked for, and the texture upload would be misaligned.
    assert response.headers["Content-Encoding"] == "gzip"
    raw = gzip.decompress(response.data)
    assert len(raw) == count * 2

    # And it is *that* mode. The fixture gives each mode a different field, so
    # the comparison is against the solver's own bytes for mode 2.
    source = np.frombuffer(
        store.read(STORE.job_key(job["job_id"], "mode_02.bin")), dtype="<f4")
    half = np.frombuffer(raw, dtype="<f2")
    mine = app.test_client().get(f"/jobs/{job['job_id']}/field/2.bin")
    again = np.frombuffer(gzip.decompress(mine.data), dtype="<f2")
    assert np.array_equal(half, again)
    finite = ~np.isnan(source)
    assert np.allclose(half[finite].astype(np.float64), source[finite],
                       rtol=1e-3, atol=1e-5)


def test_a_mode_payload_for_a_mode_the_job_lacks_is_404(tmp_path):
    """A mode number outside the solver's table, and a non-number, both 404."""
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    client = app.test_client()
    for bad in ("4", "0"):
        assert client.get(
            f"/jobs/{job['job_id']}/field/{bad}.bin").status_code == 404


def test_a_mode_whose_field_file_is_missing_answers_404_not_500(tmp_path):
    """The mode is in the table but its field is gone: 404, never a crash.

    Worth pinning because the two routes disagree on purpose. The *index* says
    ``available: false`` and still returns 200 -- the list is still useful. The
    per-mode route has nothing to send, so it is a 404. A regression here would
    surface as a 500 in the viewer's fetch, which the browser shows as a generic
    failure with no hint that only one mode is affected.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    (Path(store.root) / STORE.job_key(job["job_id"], "mode_02.bin")).unlink()

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()
    flags = {m["index"]: m["available"] for m in index["modes"]}
    assert flags == {1: True, 2: False, 3: True}
    assert client.get(
        f"/jobs/{job['job_id']}/field/2.bin").status_code == 404
    assert client.get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 200


def test_a_truncated_field_file_is_unavailable_rather_than_half_drawn(tmp_path):
    """Size is checked, so a partial write cannot decode into a wrong shape."""
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    blob = Path(store.root) / STORE.job_key(job["job_id"], "mode_01.bin")
    blob.write_bytes(blob.read_bytes()[:100])

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()
    assert {m["index"]: m["available"] for m in index["modes"]}[1] is False
    assert client.get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 404


def test_the_index_and_the_fetch_agree_on_which_modes_exist(tmp_path):
    """``available: true`` must mean the fetch will actually serve the mode.

    The reviewer found the shape of this bug: the index resolved a mode's
    filename one way and the fetch route resolved it another, so the list could
    promise a mode the fetcher 404ed -- a could-not-ask wearing the other mask,
    discovered only after a doomed request. Today there is one resolver
    (``FIELD_NAME``); this test exists to catch a *second* one appearing.

    Its most important fixture is the **named** file. The original bug was only
    reachable when ``modes.json`` named a field file, and the ``db.finish``
    helper never writes that key -- so a test built on the default manifest
    would pass against a re-introduced second resolver, exactly as the first
    version of this test did. A mode is renamed here and the manifest is made to
    name it, so any code path that honours ``field_file`` is exercised rather
    than skipped.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=4)

    prefix = STORE.job_key(job["job_id"], "")
    key = lambda name: STORE.job_key(job["job_id"], name)   # noqa: E731

    # mode 1: moved to a name the convention would not guess, and declared.
    store.write(key("mode_01_moved.bin"), store.read(key("mode_01.bin")))
    (Path(store.root) / key("mode_01.bin")).unlink()

    # mode 2: gone entirely, still declared.
    (Path(store.root) / key("mode_02.bin")).unlink()

    # mode 3: present under the convention but the wrong size.
    blob = Path(store.root) / key("mode_03.bin")
    blob.write_bytes(blob.read_bytes()[:64])

    # mode 4: untouched.

    manifest = json.loads(store.read(key("modes.json")))
    declared = {1: "mode_01_moved.bin", 2: "mode_02.bin"}
    for mode in manifest["modes"]:
        if int(mode["index"]) in declared:
            mode["field_file"] = declared[int(mode["index"])]
    store.write(key("modes.json"), json.dumps(manifest).encode())
    assert prefix  # the key layout is the job's, not a bare name

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()

    for mode in index["modes"]:
        served = client.get(
            f"/jobs/{job['job_id']}/field/{mode['index']}.bin")
        if mode["available"]:
            assert served.status_code == 200, (
                f"the index promised mode {mode['index']} and the fetch refused "
                f"it ({served.status_code}) -- the two routes disagree about "
                "which file holds this mode"
            )
        else:
            assert served.status_code == 404, (
                f"the index hid mode {mode['index']} but the fetch served it: "
                "the list is hiding something the server would have given"
            )

    # mode 1 is present in the store, but under a name only the manifest knows,
    # so the single resolver does not find it and the index says so. That is the
    # point: with one resolver, `available` is answerable and correct. A second
    # resolver that honours `field_file` would report it available, the fetch
    # would still 404, and the loop above fails -- which is how this catches the
    # reviewer's bug coming back.
    assert {m["index"]: m["available"] for m in index["modes"]} == {
        1: False, 2: False, 3: False, 4: True}


# ---------------------------------------------------------------------------
# The volume plot's field endpoint
# ---------------------------------------------------------------------------
def test_the_field_route_serves_float16_that_decodes_back_to_the_mode(tmp_path):
    """The bytes the browser textures are the mode, in half precision.

    Round-trips through the real decoder rather than trusting the header: what
    matters is that a texture built from this payload shows the field the solver
    wrote, at half precision, with ``(nz, ny, nx, 3)`` samples in the order the
    grid expects.
    """
    import gzip
    import struct

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    response = app.test_client().get(
        f"/jobs/{job['job_id']}/field/2.bin")
    assert response.status_code == 200
    assert response.headers["Content-Encoding"] == "gzip"
    assert response.headers["X-Field-Dtype"] == "float16-le"

    raw = gzip.decompress(response.data)
    original = np.frombuffer(
        store.read(STORE.job_key(job["job_id"], "mode_02.bin")), dtype="<f4")
    half = np.frombuffer(raw, dtype="<f2")
    assert half.size == original.size
    # Every finite sample agrees to half precision; the count of NaN matches.
    finite = ~np.isnan(original)
    assert int(np.isnan(half).sum()) == int((~finite).sum())
    assert np.allclose(half[finite].astype("f4"), original[finite], atol=1e-3)


def test_the_field_route_keeps_nan_transparent_where_the_solver_put_it(tmp_path):
    """NaN is the object's outside; the plot shows it by *not* showing it.

    If the encoder wrote a number here the browser would paint coloured fog
    around the object, which still looks like a plausible picture and so would
    not be noticed. The route must carry NaN through unchanged.
    """
    import gzip

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    # The default fixture writes a finite field, so it could not exercise this
    # at all -- the assertion below would have been vacuous. Write a blob that
    # is NaN over part of its samples, in the shape the grid requires, then
    # check the encoder preserved the NaN mask exactly.
    key = STORE.job_key(job["job_id"], "mode_01.bin")
    source = np.frombuffer(store.read(key), dtype="<f4").copy()
    source[: source.size // 3] = np.nan
    store.write(key, source.tobytes())

    raw = gzip.decompress(app.test_client().get(
        f"/jobs/{job['job_id']}/field/1.bin").data)
    half = np.frombuffer(raw, dtype="<f2")
    assert np.isnan(half).sum() > 0, "the fixture must have NaN to test with"
    assert np.array_equal(np.isnan(half), np.isnan(source))


def test_both_field_routes_agree_on_which_modes_exist(tmp_path):
    """The index and the binary fetch must resolve the same file.

    The reviewer's original bug was two routes resolving one filename
    differently, so a mode the list promised 404ed on fetch. The field route
    resolves ``FIELD_NAME`` too, and this pins that they stay in step: a mode
    the index marks available must serve bytes, and one it hides must not.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=4)
    (Path(store.root) / STORE.job_key(job["job_id"], "mode_02.bin")).unlink()

    client = app.test_client()
    index = client.get(f"/jobs/{job['job_id']}/index.json").get_json()
    for mode in index["modes"]:
        served = client.get(f"/jobs/{job['job_id']}/field/{mode['index']}.bin")
        if mode["available"]:
            assert served.status_code == 200, mode["index"]
        else:
            assert served.status_code == 404, mode["index"]


def test_a_mode_the_job_lacks_is_404_on_the_field_route(tmp_path):
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    client = app.test_client()
    for bad in ("4", "0"):
        assert client.get(
            f"/jobs/{job['job_id']}/field/{bad}.bin").status_code == 404


def test_an_unreachable_store_is_a_502_on_the_field_route_too(tmp_path):
    """Same invariant as the index: a store we cannot ask is not "no mode".

    A 404 here would tell the viewer this mode is missing and let it move on
    quietly; the truth is that we could not find out. That distinction is what
    ``available: false`` is only ever allowed to mean from a *successful* check.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    def boom(*_a, **_k):
        raise STORE.StorageError("the store is unreachable")

    app.extensions["mse"]["store"].read = boom
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 502


def test_an_unreachable_store_is_an_error_not_a_page_of_missing_modes(tmp_path):
    """A store we cannot ask must not answer "all modes are gone".

    ``available: false`` is a claim about the data. ``StorageError`` means the
    claim could not be checked at all -- S3 down, credentials lapsed, a network
    partition -- and rendering it as a 200 full of false flags tells the owner
    something false: every mode greyed out, and no error anywhere to explain it.

    This is the signature defect in its third costume (403 into an empty list,
    not-allowed-to-look into nothing-there, and now storage-unreachable into
    file-missing). The store already draws the line -- ``None`` is "looked and
    it is not there", an exception is "could not look" -- so the test asserts the
    route keeps it, and the asymmetry that proves it: the very same store serves
    a mode the index refused to describe.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    client = app.test_client()
    # A positive control first: with a working store the index is a 200 and the
    # modes are genuinely available. Without this, a route that always 502s
    # would pass the assertion below for the wrong reason.
    healthy = client.get(f"/jobs/{job['job_id']}/index.json")
    assert healthy.status_code == 200
    assert {m["available"] for m in healthy.get_json()["modes"]} == {True}

    def unreachable(key):
        raise STORE.StorageError(f"could not check {key} in storage")

    # The app builds its own store from ``MSE_LOCAL_STORE``; ``make_app``'s
    # store is a second handle on the same directory, used to lay down fixtures.
    # Patching the app's handle is the one the routes call.
    app.extensions["mse"]["store"].size = unreachable
    response = client.get(f"/jobs/{job['job_id']}/index.json")

    assert response.status_code == 502, (
        "an uncheckable store must be an error the viewer can report, not a "
        "200 claiming every mode is missing"
    )
    # And the fetch route, on the same broken store, still serves the mode --
    # which is the proof that the index's "all false" would have been a lie
    # about the data rather than a fact about it.
    assert client.get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 200


def test_the_index_route_is_the_only_json_the_viewer_needs(tmp_path):
    """The index still has to be strict JSON, and it still has to be small.

    The NaN lesson lives on in a different place now. The field payload is
    float16 bytes rather than JSON, so ``json.dumps`` can no longer leak a bare
    ``NaN`` token into it -- but the *index* is still JSON, and the grid it
    carries still comes from ``modes.json``, which the solver wrote. So this
    parses the index the way the browser does, with the hook that only fires on
    a non-finite token, and separately checks that no mode entry has grown a
    field-sized array back into it.
    """
    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=3)

    response = app.test_client().get(f"/jobs/{job['job_id']}/index.json")
    assert response.status_code == 200

    def reject(token):
        raise AssertionError(
            f"the index wrote the bare token {token!r}; JSON.parse would reject "
            "the whole payload and the viewer would show an error instead")

    payload = json.loads(response.get_data(as_text=True), parse_constant=reject)
    assert [m["index"] for m in payload["modes"]] == [1, 2, 3]
    for mode in payload["modes"]:
        assert "slices" not in mode and "cloud" not in mode and "field" not in mode
    # The whole point of the split: this response must stay small enough to
    # fetch on page load. One mode's float16 field is ~110 kB for this grid, so
    # a plain ceiling well under that catches a bulk array coming back.
    assert len(response.data) < 4000, (
        f"the index is {len(response.data)} bytes; a field-sized array has "
        "probably come back into it, which is the 11 MB shape it was split from")


def test_a_real_field_response_is_written_for_the_node_test(tmp_path):
    """Write the bytes a live route returns, for the node test to parse.

    The old fixture was JSON and the node test used ``JSON.parse``; the payload
    is float16 bytes now, so the instrument changes with it. What must not
    change is the principle: the fixture is produced by a *live route*, not
    hand-written, so ``test_viewfield.mjs`` is checking the real thing rather
    than a double that models what we think we send.
    """
    import gzip

    app, store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    db.finish(job["job_id"], store, n_modes=1)

    key = STORE.job_key(job["job_id"], "mode_01.bin")
    grid = {"nx": 4, "ny": 3, "nz": 2}
    field = np.full((grid["nz"], grid["ny"], grid["nx"], 3), np.nan, dtype="<f4")
    field[0, 0, 0] = [1.0, -0.5, 0.25]
    field[1, 2, 3] = [0.125, 0.0, -1.0]
    store.write(key, field.tobytes())

    response = app.test_client().get(f"/jobs/{job['job_id']}/field/1.bin")
    assert response.status_code == 200

    fixtures = Path(__file__).resolve().parent / "js" / "fixtures"
    fixtures.mkdir(parents=True, exist_ok=True)
    (fixtures / "mode_field.bin").write_bytes(gzip.decompress(response.data))
    # The grid the node test needs to decode it, alongside the bytes. Written
    # from the live index rather than from the literal above, so a change to how
    # the server reports a grid cannot pass unnoticed.
    index = app.test_client().get(f"/jobs/{job['job_id']}/index.json").get_json()
    (fixtures / "mode_grid.json").write_text(json.dumps(index["grid"]))
    # Remove the old JSON fixture: leaving it would let a stale artifact be read
    # by a test that no longer describes the API.
    stale = fixtures / "mode_payload.json"
    if stale.exists():
        stale.unlink()


def test_the_json_provider_refuses_to_write_a_non_finite_number(tmp_path):
    """The tripwire, tested directly rather than only through a route.

    A future field that forgets ``_round`` should fail loudly in tests. Without
    ``allow_nan=False`` the failure is silent, reaches the browser, and surfaces
    as a message about the network -- which is what happened to the owner.
    """
    app, _store, _db, _launcher = make_app(tmp_path)
    with app.test_request_context():
        from flask import jsonify
        # jsonify serialises eagerly, so the ValueError lands here rather than
        # at response time -- either way it is a 500 in the request, not a
        # payload the browser cannot parse.
        with pytest.raises(ValueError):
            jsonify({"bad": float("nan")})


def test_the_index_is_not_served_before_the_job_finishes(tmp_path):
    app, _store, db, _launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/index.json").status_code == 404
    assert app.test_client().get(
        f"/jobs/{job['job_id']}/field/1.bin").status_code == 404


# ---------------------------------------------------------------------------
# Static assets must revalidate, or a deploy serves the old viewer
# ---------------------------------------------------------------------------
def test_static_assets_must_revalidate_rather_than_expire_on_a_timer(tmp_path):
    """A cached viewer.js must never outlive the API it was written against.

    The endpoints changed shape: ``/index.json`` used to carry every mode's
    field and is now an index. A browser holding the old ``viewer.js`` would ask
    for the new shape, find no ``slices`` key, and render blank slices -- with no
    error anywhere, which is what makes it worth a test rather than a comment.

    This passes today because Flask's ``send_from_directory`` sends
    ``Cache-Control: no-cache`` plus an ETag: the browser caches, but must
    revalidate, and revalidation fails the moment the file changes.
    ``no-cache`` is not ``no-store`` -- it is exactly the behaviour we want.

    What must NOT appear is an expiry without revalidation. ``max-age`` (or
    ``S-F``-style freshness) lets a browser reuse the file with no question
    asked, for as long as it says. That is the regression this test exists for,
    and it is the only test here that a version query-string on the script tag
    would not satisfy -- deliberately, because a query-string is a workaround for
    a missing revalidation guarantee, not the guarantee itself.

    The behavioural half mutates a **copy** of ``web/static``, not the tracked
    files, and points ``app.static_folder`` at it. Mutating the real one would
    work, but a kill between the write and the cleanup, or a concurrent run,
    would leave the source modified -- the same shape as the revert-tool damage
    that needed a lock. Working on a copy removes the window rather than
    tidying up after it.
    """
    import shutil

    app, _store, _db, _launcher = make_app(tmp_path)
    client = app.test_client()

    for asset in ("viewer.js", "modecache.js", "style.css"):
        response = client.get(f"/static/{asset}")
        assert response.status_code == 200, asset
        cache_control = (response.headers.get("Cache-Control") or "").lower()
        assert "max-age" not in cache_control, (
            f"/static/{asset} sends Cache-Control {cache_control!r}, which lets a "
            "browser reuse it without revalidating -- a deploy would then keep "
            "serving the old viewer against the new API"
        )
        # Revalidation is only meaningful if there is something to compare.
        assert response.headers.get("ETag") or \
            response.headers.get("Last-Modified"), asset

    # From here on the app serves a copy, so this test cannot touch the checkout.
    staged = tmp_path / "static-copy"
    shutil.copytree(app.static_folder, staged)
    real_viewer = Path(app.static_folder) / "viewer.js"
    original = real_viewer.read_bytes()
    app.static_folder = str(staged)

    # And the guarantee must be real, not just declarable: change the file and
    # the stale copy must be refused. Without this the assertions above could
    # pass on a server that ignores validators entirely.
    first = client.get("/static/viewer.js")
    unchanged = client.get("/static/viewer.js", headers={
        "If-None-Match": first.headers["ETag"]})
    assert unchanged.status_code == 304, "an unchanged asset should be reused"

    (staged / "viewer.js").write_text(
        (staged / "viewer.js").read_text() + "\n// deployed change\n")
    revalidated = client.get("/static/viewer.js", headers={
        "If-None-Match": first.headers["ETag"]})
    assert revalidated.status_code == 200, (
        "a changed asset was served as 304: the browser would keep the old "
        "viewer with no way to notice"
    )
    assert real_viewer.read_bytes() == original, (
        "this test modified the tracked web/static/viewer.js; it must only ever "
        "mutate its own copy"
    )


# ---------------------------------------------------------------------------
# The scheduler: exactly one Job, a quota 403 is a wait, and a finished Job is
# deleted so the next upload can start.
# ---------------------------------------------------------------------------
def test_a_second_upload_waits_while_one_is_running(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    first = next(iter(db.jobs.values()))
    assert first["status"] == DB.RUNNING

    upload(app.test_client())
    second = [j for j in db.jobs.values() if j["job_id"] != first["job_id"]][0]
    assert second["status"] == DB.QUEUED
    assert len(launcher.created) == 1          # only one Job exists


def test_a_quota_403_waits_rather_than_failing(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    launcher.quota_full = True
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert job["status"] == DB.QUEUED          # still waiting
    assert db.failures == []                   # NOT a failure


def test_the_tick_launches_the_next_queued_job_once_the_slot_frees(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    first = next(iter(db.jobs.values()))
    upload(app.test_client())
    second = [j for j in db.jobs.values() if j["job_id"] != first["job_id"]][0]
    assert second["status"] == DB.QUEUED

    # the first Job finishes. The app -- NOT the test -- must delete it: the
    # quota counts finished Jobs, so a Job left behind holds the only slot and
    # the queue deadlocks after the first upload.
    db.jobs[first["job_id"]].update(status=DB.DONE, stage=DB.DONE)
    APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert first["k8s_job_name"] in launcher.deleted
    assert db.jobs[second["job_id"]]["status"] == DB.RUNNING
    assert len(launcher.created) == 2


def test_a_finished_job_is_deleted_by_the_tick_without_being_asked(tmp_path):
    """The tick frees the slot itself; nothing else calls launcher.delete.

    Regression: delete() had no caller at all, so a finished Job held the
    namespace's single Job slot forever and every later upload sat queued.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert launcher.active == [name]

    db.jobs[job["job_id"]].update(status=DB.DONE, stage=DB.DONE)
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert launcher.deleted == [name]
    assert result["deleted"] == [name]
    assert launcher.active == [], "the slot must actually be free afterwards"


def test_a_failed_job_is_deleted_too(tmp_path):
    """A failed solve is recorded in the DB; its Job must not hold the slot."""
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]

    db.jobs[job["job_id"]].update(status=DB.FAILED, stage=DB.FAILED,
                                  message="boom")
    APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert launcher.deleted == [name]


def test_a_queued_or_running_job_is_never_deleted(tmp_path, monkeypatch):
    """Only terminal rows may be reaped -- deleting a live Job loses the solve."""
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (running,) = [j for j in db.jobs.values() if j["status"] == DB.RUNNING]

    APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert running["k8s_job_name"] not in launcher.deleted
    assert launcher.active == [running["k8s_job_name"]]


def test_a_job_reporting_deadline_exceeded_fails_the_row_with_that_reason(
        tmp_path, monkeypatch):
    """A Job killed by activeDeadlineSeconds must name the deadline, not "gone".

    The owner needs to know whether the solver ran out of time or out of memory
    -- "try fewer modes" is useless advice for the first if it was the second.
    Kubernetes puts the reason on the Job's Failed condition and the tick maps
    it through ``JOBS.failure_message``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]

    launcher.failed[name] = "DeadlineExceeded"
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 1
    assert db.jobs[job["job_id"]]["status"] == DB.FAILED
    assert db.jobs[job["job_id"]]["message"] == \
        JOBS.FAILURE_MESSAGES["DeadlineExceeded"]


def test_a_job_reporting_oomkilled_fails_the_row_with_that_reason(tmp_path,
                                                               monkeypatch):
    """The original incident's cause: the container was OOM-killed.

    This is the case that actually happened -- the writing stage hit its 14 Gi
    limit -- so the message the owner sees must say memory, not "unexpectedly".
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()
    launcher.failed[job["k8s_job_name"]] = "OOMKilled"

    APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert db.jobs[job["job_id"]]["message"] == \
        JOBS.FAILURE_MESSAGES["OOMKilled"]


def test_a_job_failed_for_an_unknown_reason_does_not_leak_the_kubernetes_token(
        tmp_path, monkeypatch):
    """An unmapped reason must fall back, never reach the owner verbatim.

    Kubernetes reasons are token-like strings ('BackoffLimitExceeded'); putting
    one in front of the owner is worse than saying nothing specific.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()
    launcher.failed[job["k8s_job_name"]] = "BackoffLimitExceeded"

    APP.run_tick(None, db, launcher, log=lambda *a: None)

    message = db.jobs[job["job_id"]]["message"]
    assert message == JOBS.FAILURE_DEFAULT
    assert "BackoffLimitExceeded" not in message


def test_a_live_job_is_never_failed_or_deleted(tmp_path, monkeypatch):
    """The case that must NOT be touched: Job present, no Failed condition.

    This is the other half of the fix and the more dangerous half to get wrong.
    The reconcile step crosses the database against the cluster, so a bug that
    treats "absent from job_states" as "dead" would fail every healthy solve the
    moment it started writing -- the owner would see failures on every upload.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert launcher.active == [name]

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING
    assert db.jobs[job["job_id"]]["message"] is None
    assert name not in launcher.deleted, "a live solve must keep its slot"
    assert db.failures == []


def test_a_claimed_jetson_job_survives_a_tick(tmp_path):
    """THE INTEGRATION TEST: a real tick must not kill a real Jetson claim.

    This is the test whose absence let the second dispatch deploy through five
    green gates. Every existing test drove ``_fail_dead_jobs`` against a
    ``FakeDb`` that could only ever hold CLUSTER rows -- a row with a
    ``k8s_job_name`` the fake launcher reported. A Jetson job is not launched by
    the app at all: it is a systemd unit the owner runs, so it has NO
    ``k8s_job_name`` and is absent from ``active_job_names`` by construction.

    ``_fail_dead_jobs`` read that absence as death, so the first tick after a
    claim failed every GPU job in the cluster:

        [mse-web] worker w-b55c... claimed bdaa3fbd...
        [mse-web] job bdaa3fbd... failed: its Job <unnamed> is gone

    The test drives the REAL tick (``run_tick`` -> ``_fail_dead_jobs``) against a
    REAL claim path (``claim_jetson``), with a launcher reporting an EMPTY
    cluster -- which is not a fault here, it is what a Jetson-only fleet always
    looks like. A Jetson row's liveness is its LEASE; the cluster has nothing to
    say about it.

    Mutation that must turn this red: delete the ``target == DB.JETSON`` skip in
    ``_fail_dead_jobs``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    # A row queued for the GPU the way the routing decision leaves it, then
    # CLAIMED the way a real worker does. Neither step leaves a k8s Job behind,
    # which is the whole point.
    _make_jetson_row(db)
    claimed = db.claim_jetson("w-integration", 300.0)
    assert claimed is not None, "the claim path must produce a row to test"
    assert claimed["k8s_job_name"] is None, (
        "a Jetson job genuinely has no k8s Job -- that is the premise")
    assert launcher.active == [], (
        "and therefore the cluster reports nothing for it")

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    row = db.jobs[str(claimed["job_id"])]
    assert row["status"] == DB.RUNNING, (
        "a tick failed a claimed Jetson job -- its Job <unnamed> is gone is what "
        "the reaper said, and it must not say that about a lease-owned row"
    )
    assert row["lease_owner"] == "w-integration", "the lease is its liveness"
    assert db.failures == [], f"nothing may be failed: {db.failures}"
    assert result["failed"] == 0


def test_a_failed_jetson_row_is_not_cleared_by_a_later_tick(tmp_path):
    """The other direction, so the skip is a SKIP and not a blind spot.

    A guard that excludes Jetson rows must not also exclude them from every
    OTHER mechanism. This pins that the reaper's exclusion is narrow: it stops
    this function from judging them, and nothing more.

    SCOPE, stated because it is easy to over-read: this runs against ``FakeDb``,
    so it checks the TICK's behaviour (it does not re-fail an already-terminal
    row) and NOT the real store's SQL. ``FakeDb.mark_failed`` clears the lease
    to mirror the real store, but a double agreeing with the code it doubles
    proves nothing about the code -- verified by mutation: reverting the real
    ``mark_failed`` leaves this test GREEN. The real lease-clear is pinned on
    real Postgres by ``pg_race_check.py`` D1, which is the leg that goes red.
    """

    app, _store, db, launcher = make_app(tmp_path)
    row = _make_jetson_row(db)
    db.claim_jetson("w-integration", 300.0)

    # The APP fails it (a launch error, a reaper verdict elsewhere, an operator).
    db.mark_failed(row["job_id"], "the GPU worker was stopped by an operator")

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    after = db.jobs[str(row["job_id"])]
    assert after["status"] == DB.FAILED
    assert after["lease_owner"] is None, (
        "mark_failed must clear the lease, or the worker keeps renewing against "
        "a dead row and never learns to stop"
    )
    assert result["failed"] == 0, "it was already terminal; the tick did not re-fail it"


def test_a_terminal_row_with_a_stale_lease_is_not_resurrected(tmp_path):
    """The 20:15Z bug, at the level a unit test CAN see it: reclaim's filter.

    Found live on the third deploy: the reaper failed a Jetson job while the OLD
    ``mark_failed`` left its lease set, then ``reclaim_expired_leases`` found the
    expired lease, requeued the TERMINAL row to ``queued``, and a worker claimed
    and solved a job the owner had been told was FAILED. The web log read "its
    GPU worker's lease expired; the job is claimable again".

    SCOPE, stated because it is easy to over-read, and the same caveat as the
    sibling test above: this drives ``FakeDb``, so it pins the TICK's and the
    double's behaviour and NOT the real SQL. The double now carries the
    ``status != RUNNING`` filter to mirror the store, so reverting the real
    SQL's filter alone leaves this GREEN -- the real filter is pinned on real
    Postgres by ``pg_race_check.py`` D3, which is the leg that goes red.

    What it DOES pin, and what makes it worth keeping: the state is reachable at
    all through the tick, and "already terminal" is not re-litigated by the next
    pass. A tick that requeued it would show here.
    """
    app, _store, db, launcher = make_app(tmp_path)
    row = _make_jetson_row(db)
    db.claim_jetson("w-integration", 300.0)

    # The legacy shape, written by hand because the fixed code no longer makes
    # it: terminal, still holding the lease, deadline in the past.
    row.update(status=DB.FAILED, stage=DB.FAILED, lease_expired=True)

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert db.jobs[str(row["job_id"])]["status"] == DB.FAILED, (
        "a TERMINAL row must keep its outcome; a reclaim that judges the lease "
        "without checking the status resurrects it and a worker solves a job "
        "the owner was told was over"
    )
    assert result.get("reclaimed", []) == [], (
        "nothing may be reported as reclaimed while the row stayed terminal"
    )


def _make_jetson_row(db):
    """Put a row in ``db`` queued for the GPU, as the routing decision does."""
    import uuid as _uuid
    job_id = str(_uuid.uuid4())
    db.create_job(job_id, "box.obj", "mm", 1.0, 20, target=DB.JETSON)
    return db.jobs[job_id]


def test_a_403_on_the_list_call_does_not_fail_a_live_solve(tmp_path):
    """The reviewer's finding, reproduced: an unreadable cluster is not an empty one.

    ``K8sJobLauncher`` used to swallow a 403 from both list calls into ``[]`` and
    ``{}``. That made "we were not allowed to look" indistinguishable from "there
    is nothing there", so ``_fail_dead_jobs``'s ``except``-to-0 never ran and a
    live solve was failed. Because ``mark_failed`` is terminal, a later tick then
    deleted that still-running Job via ``_finished_job_names`` -- discarding real
    work on the strength of a permission error, which is the exact outcome the
    docstring promised to avoid.

    What this test pins is the *caller's* half: given an unreadable cluster, the
    tick fails nothing and reaps nothing. It drives ``swallowed_403`` for that,
    because the swallow is the input the caller has to survive. It does not fail
    on ``e5b2e3e``, and cannot: the swallow is exactly what makes the tick answer
    an empty cluster, whose correct handling is not to wait -- so the tick's own
    behaviour is the same either way. The launcher's half of the fix, the raise
    that keeps "could not ask" from ever becoming the empty answer, is covered
    behaviourally in ``test_web_app.py``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    name = job["k8s_job_name"]
    assert job["status"] == DB.RUNNING

    launcher.cannot_ask = True

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 0
    assert db.jobs[job["job_id"]]["status"] == DB.RUNNING, (
        "a 403 on the list call must not fail a solve that is still running"
    )
    assert db.jobs[job["job_id"]]["message"] is None
    assert name not in launcher.deleted, (
        "and it must not be reaped as terminal on a later tick"
    )


def test_a_403_does_not_let_the_tick_launch_against_the_quota(tmp_path):
    """The other half: an unreadable cluster is not a licence to start a Job.

    ``active_job_names`` returning ``[]`` on 403 also told the tick the namespace
    was empty, so it would launch into a quota it could not read. The gate must
    stay shut when the answer is "unknown" -- the namespace admits one Job, so a
    second start is at best a 403 from the quota and at worst a double solve.

    Like its sibling above, this pins the caller and uses ``cannot_ask``: the
    launcher's own raise, which keeps "could not ask" from arriving as the empty
    answer in the first place, is covered behaviourally in ``test_web_app.py``.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload(app.test_client())
    (job,) = db.jobs.values()
    # a healthy Job is occupying the slot, but the app cannot see that
    launcher.cannot_ask = True

    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["launched"] is None
    assert result["waited"] is True
    assert len(launcher.created) == 1, (
        "the tick must not start a second solve against a quota it cannot read"
    )


def test_a_running_row_whose_job_vanished_is_failed_not_requeued(tmp_path,
                                                                 monkeypatch):
    """The deploy-blocker-3 fix: a dead solve is FAILED, never silently retried.

    This test replaces one that asserted the old behaviour (re-queue), which was
    the bug: a row whose Job had vanished was put back in the queue, so a solve
    that had already died -- an OOM, say -- would be re-run forever and the owner
    would watch a spinner that never ended. The row must end `failed`, with a
    message the owner can read.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    (job,) = db.jobs.values()

    # the DB says running, but the cluster has nothing (lost with its node)
    launcher.active.clear()
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert result["failed"] == 1
    assert db.jobs[job["job_id"]]["status"] == DB.FAILED
    assert db.jobs[job["job_id"]]["message"] == JOBS.FAILURE_DEFAULT
    assert len(launcher.created) == 1, "the dead job must not be relaunched"


def test_the_tick_does_nothing_when_the_slot_is_occupied(tmp_path, monkeypatch):
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert len(launcher.created) == 1


def test_a_dead_job_frees_the_slot_for_the_next_upload(tmp_path, monkeypatch):
    """Failing the dead row must not wedge the queue behind it.

    The old behaviour re-queued the dead job, which kept the single Job slot
    busy with a solve that could never succeed. Failing it is only correct if
    the *next* upload then runs -- otherwise the fix just swaps one permanent
    spinner for a permanently stuck queue.
    """
    app, _store, db, launcher = make_app(tmp_path)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    upload_to_the_cluster(app.test_client(), monkeypatch)
    first, second = list(db.jobs.values())
    assert first["status"] == DB.RUNNING

    # the running Job vanished from the cluster without recording a result
    launcher.active.clear()
    APP.run_tick(None, db, launcher, log=lambda *a: None)

    assert db.jobs[first["job_id"]]["status"] == DB.FAILED
    assert db.jobs[second["job_id"]]["status"] == DB.RUNNING, (
        "the queued upload must be picked up once the dead job is failed"
    )


def test_a_launch_failure_is_recorded_against_the_job(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    launcher.fail_with = JOBS.JobLaunchError(
        "The service account is not allowed to start solver jobs.")
    upload(app.test_client())
    (job,) = db.jobs.values()
    assert job["status"] == DB.FAILED
    assert "not allowed" in job["message"]
    assert len(db.failures) == 1


def test_nothing_is_launched_when_nothing_is_queued(tmp_path):
    app, _store, db, launcher = make_app(tmp_path)
    result = APP.run_tick(None, db, launcher, log=lambda *a: None)
    assert result["launched"] is None
    assert launcher.created == []


# ---------------------------------------------------------------------------
# Multi-material uploads: the /parts endpoint, its rails, and the job page.
# ---------------------------------------------------------------------------

def nested_obj_bytes(names=("wall", "core"), sizes=((6.0, 6.0, 6.0),
                                                   (2.0, 2.0, 2.0))):
    """A wall box with a smaller core inside, as a two-group .obj."""
    import trimesh

    geoms = {n: trimesh.creation.box(extents=s)
             for n, s in zip(names, sizes)}
    return trimesh.Scene(geoms).export(file_type="obj").encode("utf-8")


def test_parts_endpoint_returns_the_names_the_solver_will_use(tmp_path):
    """``POST /parts`` parses with the SOLVER'S parser, in file order.

    The names this returns are the ones the owner assigns an index to, and the
    ones the solver will re-derive from the same bytes. A second parser (a
    client-side scan, or a duplicate server-side one) is a second source of
    truth for exactly the thing that decides the physics, and the failure it
    produces -- an index on the wrong region with every count agreeing -- is
    the one this feature was designed to avoid. So the assertion is not just
    "it returned names"; it is that they match ``load_parts`` on the same
    bytes, which is the call the solve itself makes.
    """
    app, _store, _db, _launcher = make_app(tmp_path)
    data = nested_obj_bytes()

    response = app.test_client().post("/parts", data={
        "meshfile": (io_bytes(data), "nested.obj"), "units": "mm",
    }, content_type="multipart/form-data")

    # Parse the payload rather than trusting the status: a 200 carrying an
    # error body reads as success to a check that only looks at the code.
    assert response.status_code == 200, response.get_data(as_text=True)[:400]
    body = response.get_json()
    assert "error" not in body, body
    assert body["parts"] == ["wall", "core"], (
        f"the endpoint returned {body['parts']}; the solver's own parser on "
        f"the same bytes says {APP.load_parts(data, 'mm').names}"
    )
    # And it really is the solver's parser, not an agreement by luck: a file
    # with different names must come back with THOSE names.
    other = nested_obj_bytes(names=("shell", "rod"))
    response = app.test_client().post("/parts", data={
        "meshfile": (io_bytes(other), "other.obj"), "units": "mm",
    }, content_type="multipart/form-data")
    assert response.get_json()["parts"] == ["shell", "rod"]


def test_parts_endpoint_names_the_part_it_refuses(tmp_path):
    """A refusal must say WHICH part, or the owner cannot fix it.

    An open part is the refusal an owner is most likely to hit (a mesh exported
    without capping a hole). "invalid file" would be the could-not-ask shape:
    the message cannot be acted on, so the owner retries the same file.
    """
    app, _store, _db, _launcher = make_app(tmp_path)
    import trimesh

    open_core = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    open_core.update_faces(np.arange(len(open_core.faces) - 1))  # drop a face
    data = trimesh.Scene({
        "wall": trimesh.creation.box(extents=(6.0, 6.0, 6.0)),
        "core": open_core,
    }).export(file_type="obj").encode("utf-8")

    response = app.test_client().post("/parts", data={
        "meshfile": (io_bytes(data), "open.obj"), "units": "mm",
    }, content_type="multipart/form-data")

    assert response.status_code == 400
    message = response.get_json()["error"]
    assert "core" in message, (
        f"the refusal does not name the offending part: {message!r}"
    )


def test_parts_endpoint_rejects_a_missing_file_and_bad_units(tmp_path):
    """The two non-geometry refusals, at the same status as the geometric ones."""
    app, _store, _db, _launcher = make_app(tmp_path)
    client = app.test_client()

    response = client.post("/parts", data={"units": "mm"},
                           content_type="multipart/form-data")
    assert response.status_code == 400 and "error" in response.get_json()

    response = client.post("/parts", data={
        "meshfile": (io_bytes(nested_obj_bytes()), "n.obj"), "units": "furlong",
    }, content_type="multipart/form-data")
    assert response.status_code == 400 and "error" in response.get_json()


def test_an_upload_with_materials_stores_the_table_and_routes_on_it(tmp_path):
    """The final POST carries the table, and the ROW keeps it.

    The route is decided from the estimate, and the estimate for a multi-part
    upload is summed over regions. This test pins the two ends: the table the
    owner posted is what the row holds (verbatim -- not defaulted, not
    re-keyed), and the row was routed off the multi-part estimate rather than
    the one-part one, which on this fixture under-predicts by 6.5x.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    data = nested_obj_bytes()

    response = upload(app.test_client(), filename="nested.obj", data=data,
                      **{"part_index.wall": "1.0", "part_index.core": "2.0"})
    assert response.status_code == 302, response.get_data(as_text=True)[:400]

    (job,) = db.jobs.values()
    assert job["materials"] == {"wall": 1.0, "core": 2.0}, (
        f"the row holds {job['materials']!r}; the owner posted "
        f"{{'wall': 1.0, 'core': 2.0}}"
    )
    # The one-part estimate on the same bytes, which is what a route decided
    # before the materials branch would have used.
    import meshing as MESHALIAS
    one_part = MESHALIAS.estimate_tets(
        MESHALIAS.load_surface(data), 10, "mm", margin=CFG.est_tets_margin())
    assert job["est_tets"] != one_part, (
        f"the row's estimate ({job['est_tets']:,}) equals the ONE-PART estimate "
        f"on the same file; the multi-part branch did not run and this job was "
        f"routed on a number 6.5x too small"
    )


def test_a_material_outside_the_rail_is_refused_by_name(tmp_path):
    """``1 <= n <= 10``, refused at upload and naming the part.

    The rail exists because this is a repeated field: an extra digit in one row
    is an easy typo, and a typo that reached the mesher refines its region by
    that factor -- cubing its element count -- while every count downstream
    still looks plausible.

    The message is read off the page the redirect lands on, not off a session
    object: that is where the owner sees it, and a flash that never renders is
    a refusal nobody reads.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    data = nested_obj_bytes()
    for bad in ("0.5", "11", "nan", "inf", "0"):
        client = app.test_client()
        response = upload(client, data=data,
                          **{"part_index.wall": "1.0", "part_index.core": bad})
        assert response.status_code == 302, bad
        assert not db.jobs, f"n={bad!r} was accepted"
        # The refusal names the part, so the owner knows which field to fix.
        page = client.get("/").get_data(as_text=True)
        assert "core" in page, (
            f"n={bad!r} was refused without naming the part; the page said: "
            f"{page[:600]!r}"
        )


def test_a_table_that_does_not_match_the_file_is_refused(tmp_path):
    """Missing and unknown part names, both refused rather than defaulted.

    A part with no index would otherwise sit at whatever the default is --
    vacuum -- and the owner would get frequencies for a cavity that is partly
    empty with nothing saying so. An index naming a part that is not in the
    file is the same mismatch from the other side, and it means the form and
    the bytes disagree about the upload.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    data = nested_obj_bytes()

    # one part missing from the table
    response = upload(app.test_client(), data=data, **{"part_index.wall": "1.0"})
    assert response.status_code == 302 and not db.jobs

    # a name that is not in the file
    response = upload(app.test_client(), data=data,
                      **{"part_index.wall": "1.0",
                         "part_index.core": "2.0",
                         "part_index.ghost": "1.5"})
    assert response.status_code == 302 and not db.jobs


def test_the_job_page_renders_the_material_table_it_stored(tmp_path):
    """The page shows the STORED table, and it renders.

    Two things are asserted, and neither is the status code. First, that the
    table reached the HTML with the part names and indices in it -- a template
    that never rendered the block would still return 200, which is this rig's
    ``HTTP 200 is not rendered`` lesson. Second, that the page does NOT
    re-derive the table from the upload: the rendered table is built from
    ``job.materials``, so a job whose stored table differs from its file shows
    the stored one.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    data = nested_obj_bytes()
    upload(app.test_client(), data=data,
           **{"part_index.wall": "1.0", "part_index.core": "2.0"})
    (job_id,) = db.jobs.keys()

    html = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)
    assert 'id="materials"' in html, (
        "the material table block did not render; a 200 says nothing about "
        "whether the template produced it"
    )
    assert "wall" in html and "core" in html
    # The values, as the page formats them (a '%g' format, so 2.0 -> "2").
    assert ">2<" in html.replace(" ", "").replace("\n", "") or "2" in html

    # Now plant a table the FILE does not have. If the page re-parsed the
    # upload it would still show wall/core at 1/2; reading the stored JSON it
    # must show the planted one.
    db.jobs[job_id]["materials"] = {"planted": 7.0}
    html = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)
    assert "planted" in html, (
        "the page did not show the STORED materials table; it is re-deriving "
        "the table from the upload, which can differ from what the solve used"
    )
    assert "wall" not in html.split('id="materials"')[1].split("</div>")[0]


def test_a_td_job_page_renders_a_metal_part_without_a_500(tmp_path):
    """The page renders every material a td job's table can hold.

    A td job's stored table is a refractive index, the string ``pec``, or a
    dispersive metal's NAME -- the owner's Au / 1.5 / Au film is exactly this
    shape. The cavity-era table formatted each value with ``'%g'``, which raised
    ``TypeError: must be real number, not str`` on the first metal: an
    unhandled 500 on the deployed page (the owner's job 180cb65f). This app is
    built with ``TESTING`` on, so the red surfaces here as that TypeError
    escaping the test client rather than as a 500 status; same defect, one flag
    apart (see ``test_a_pc_radius_too_big_for_its_cell_is_refused_not_a_500``).

    Rendered through the REAL template and store, not a fake: the defect is in
    the template's formatting of a stored value, so a test that never renders
    ``job.html`` against a td row cannot fail on it.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    job_id = str(uuid.uuid4())
    db.create_job(job_id, "film.obj (time domain)", "um", 1.0, 0,
                  target=DB.JETSON, kind=DB.TD,
                  materials={"substrate": "Au", "film": 1.5, "top_lid": "pec"},
                  tds=dict(_TDS))

    body = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)

    table = re.search(r'<table class="materials">(.*?)</table>', body, re.S)
    assert table is not None, (
        "the materials table did not render on a td job page; a 200 alone says "
        "nothing about whether the block was produced"
    )
    # Per row, so a page that printed the three values anywhere would not pass:
    # the question is which material sits against which part.
    rows = dict(re.findall(r"<tr><td>(.*?)</td><td>(.*?)</td></tr>",
                           table.group(1)))
    assert rows == {"substrate": "Au", "film": "1.5",
                    "top_lid": "perfect conductor (pec)"}, rows
    # The cavity wording is false for a td run: the index column is not an
    # index for a metal, and the "meshed to its own resolution" note describes
    # a cavity's graded mesh, not this. Header and note both name the kind.
    assert "Refractive index" not in table.group(1)


def test_a_coated_part_shows_its_coating_on_the_job_page(tmp_path):
    """A coating is a material the form accepts, so the page shows it.

    Coatings are NOT in ``job.materials`` -- they live in the stored ``tds``
    request (a part's base material and its coating are different things, and
    only the base material goes to the solver's layout table). So this is its
    own assertion: the page must find them in the request, and a page that
    rendered only the materials table would silently show a coated part as
    uncoated.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    job_id = str(uuid.uuid4())
    # The request carries the TD form's own length unit beside the thicknesses
    # it was applied to, as the form stores it -- the page cannot recover it
    # from the metres. ``units`` here is the OBJECT's upload unit, a different
    # field of the same form; a hand-built row where the two coincide is
    # exactly the fixture that hid the mislabelling, so the page's
    # unit-follows-the-form behaviour is proved through the real form in
    # tests/test_td_form.py instead.
    tds = dict(_TDS, coatings={"film": ["Au", 20e-9]})
    # All-float base materials on purpose: the red this test must show is the
    # ABSENT coatings block, not the metal-formatting TypeError its sibling
    # covers, so the materials table has to render cleanly here.
    db.create_job(job_id, "film.obj (time domain)", "um", 1.0, 0,
                  target=DB.JETSON, kind=DB.TD,
                  materials={"substrate": 1.5, "film": 1.0}, tds=tds)

    body = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)

    block = re.search(r'id="coatings"(.*?)</table>', body, re.S)
    assert block is not None, "the coating of a coated part is absent from the page"
    assert "film" in block.group(1) and "Au" in block.group(1)
    # 20 nm, in the unit the job was typed in (um) and in plain words.
    assert "0.02" in block.group(1)


def test_a_coating_request_without_its_unit_is_shown_in_metres(tmp_path):
    """A request written before the form's unit was stored has only metres.

    The page must not fall back to ``job["units"]``: that is the OBJECT's
    upload unit, so a fallback there would print a confidently wrong number for
    exactly the rows the fallback exists for. With nothing to divide by, the
    metre value stands as it is -- and this job's object unit (um) is
    deliberately the one the old code divided by, so a page that still reaches
    for it prints 0.01 here and fails.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    job_id = str(uuid.uuid4())
    tds = {k: v for k, v in _TDS.items() if k != "td_unit"}
    tds["coatings"] = {"film": ["Au", 1e-8]}
    db.create_job(job_id, "film.obj (time domain)", "um", 1.0, 0,
                  target=DB.JETSON, kind=DB.TD,
                  materials={"substrate": 1.5}, tds=tds)

    body = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)
    block = re.search(r'id="coatings"(.*?)</table>', body, re.S)
    assert block is not None, "the coating is absent from the page"
    cell = re.search(r"<tr><td>film</td><td>.*?</td>\s*<td>(.*?)</td></tr>",
                     block.group(1))
    assert cell is not None, block.group(1)
    assert cell.group(1).strip() == "1e-08 m", cell.group(1)


# ---------------------------------------------------------------------------
# The photonic-crystal upload: a second form with no file
# ---------------------------------------------------------------------------
def pc_form(**over):
    """The pc arm's form fields, as the template renders them."""
    fields = {
        "mode": "pc", "lattice": "square", "r_over_a": "0.2",
        "n_rod": "3.5", "n_bg": "1.0", "n_bands": "4",
        "divisions": "16", "polarisation": "TM",
    }
    fields.update(over)
    return fields


def test_a_pc_form_queues_a_pc_job_with_its_request_stored(tmp_path,
                                                           monkeypatch):
    """The happy path: the row is a kind='pc' job whose pcs holds the request.

    Every field is asserted against the ROW, not against the response code: a
    queue that accepted the POST and wrote the wrong request is the failure this
    is here for, and a 302 would hide it perfectly.
    """
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    app, _store, db, _launcher = make_app(tmp_path)
    resp = app.test_client().post("/", data=pc_form(),
                                  content_type="multipart/form-data")
    assert resp.status_code == 302, resp.data

    job = next(iter(db.jobs.values()))
    assert job["kind"] == DB.PC
    assert job["pcs"] == {"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5,
                          "n_bg": 1.0, "n_bands": 4, "divisions": 16,
                          "polarisation": "TM"}
    # The request is the SEVEN fields -- not the derived node count, which is
    # the web side's estimate and lives on the row's est_tets instead.
    assert "nodes" not in job["pcs"] and "k_points" not in job["pcs"]
    # est_tets is the node count the cap was checked against, recorded for the
    # same reason a cavity row records its element estimate.
    assert job["est_tets"] > 0
    # A pc job runs on the JETSON, and this assertion used to say the opposite.
    # It read "A pc job runs on the cluster: it has no GPU path" -- which was
    # true of the *estimate* (a pc solve is small) and false of the *solver*:
    # the only pc solver is the Jetson worker, which claims target='jetson'. The
    # row was created target='cluster', the cluster launcher built a CAVITY Job
    # for it, and the band diagram was never produced. The route now creates the
    # row for the machine that can actually solve it.
    assert job["target"] == DB.JETSON


def test_a_pc_form_with_a_file_is_refused_not_solved(tmp_path, monkeypatch):
    """A pc form carrying a file is a mismatch, and it is refused by name.

    The danger is the SILENT version: a handler that ignores the file would
    queue a band diagram while the owner believes they uploaded something. The
    assertion is on the row count -- a 302 would look identical either way.
    """
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    app, _store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    fields = pc_form()
    fields["meshfile"] = (io_bytes(closed_box_bytes()), "box.obj")
    resp = client.post("/", data=fields, content_type="multipart/form-data")
    assert resp.status_code == 302, resp.data
    assert db.jobs == {}, "a pc form with a file was queued anyway"
    # The flash lives in the SAME client's session; a fresh test_client() is a
    # different cookie jar and would see an empty page no matter what was said.
    # Read it from the session directly, the idiom the scale-refusal test uses.
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    assert any("no file to upload" in m for m in flashes), flashes


def test_a_pc_form_over_the_node_cap_is_refused_with_the_limit_named(
        tmp_path, monkeypatch):
    """The cap refusal happens BEFORE a row is written.

    A row that is certain to fail is worse than a refusal: it costs a worker a
    claim and the owner has to read a failure to learn what a form could have
    said. So the assertion is `db.jobs == {}` -- not merely that the page
    complained.
    """
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    app, _store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    resp = client.post("/", data=pc_form(divisions="70"),
                       content_type="multipart/form-data")
    assert resp.status_code == 302
    assert db.jobs == {}, "an over-cap request was queued anyway"
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    # BOTH numbers: the count it computed and the limit it hit. A refusal that
    # named only "too big" would leave the owner guessing which lever to pull.
    assert any("over this app's limit" in m for m in flashes), flashes
    assert any("nodes" in m for m in flashes), flashes


def test_an_unknown_lattice_is_refused_by_name(tmp_path, monkeypatch):
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    app, _store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    resp = client.post("/", data=pc_form(lattice="hexagonal"),
                       content_type="multipart/form-data")
    assert resp.status_code == 302
    assert db.jobs == {}
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    assert any("hexagonal" in m and "Unknown lattice" in m for m in flashes), \
        flashes


def test_a_pc_radius_too_big_for_its_cell_is_refused_not_a_500(
        tmp_path, monkeypatch):
    """A refusal raised by the GEOMETRY still reaches the owner as a refusal.

    The rod has to fit inside the cell, and that bound is lattice-dependent:
    r/a = 0.45 is ordinary for the square cell and impossible for the triangular
    one, whose inradius is sin(60)/2 = 0.4330. The form does not restate the
    bound -- it would be a second copy to drift from the geometry's -- so the
    geometry raises its own ``PCError``, and ``_handle_pc_upload`` catches only
    ``PCRequestError``. Without the translation in ``plan_request`` this POST
    never reached the owner as a refusal: it was an unhandled exception, HTTP
    500 on the deployed app.

    Red surfaces here as the ``PCError`` escaping the test client rather than as
    a 500 status, because this app is built with ``TESTING`` on and Flask
    re-raises instead of rendering under that flag. Same defect, one flag apart;
    the deployed app is where it is a 500.

    Both halves are asserted, since either alone is a different bug: the status
    (a 500 says the refusal never reached the owner) and the words (a refusal
    naming the wrong box sends them to the wrong field). ``db.jobs == {}`` is the
    third -- a refusal that still wrote the row would cost a worker a claim.
    """
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    app, _store, db, _launcher = make_app(tmp_path)
    client = app.test_client()
    resp = client.post("/",
                       data=pc_form(lattice="triangular", r_over_a="0.45"),
                       content_type="multipart/form-data")
    assert resp.status_code == 302, resp.data
    assert db.jobs == {}, "an impossible request was queued anyway"
    with client.session_transaction() as session:
        flashes = [str(m) for m in session.get("_flashes", [])]
    assert any("triangular cell" in m for m in flashes), flashes
    # ...and it is ON THE PAGE, not only in the session: base.html renders the
    # flashes, so this GET is what the owner actually reads. Reading _flashes
    # above does not consume them, so the same client still has them here.
    page = client.get("/").get_data(as_text=True)
    assert "triangular cell" in page


def test_the_cavity_form_still_uploads_with_no_mode_field(tmp_path,
                                                          monkeypatch):
    """A form from before the selector existed still takes the cavity path.

    The dispatch is on `mode == 'pc'`, so an ABSENT mode must mean cavity --
    and the assertion is that a real upload lands, not merely that no error was
    returned. This is the negative control for the whole selector: if the
    dispatch defaulted the other way, every one of these would break and a test
    that only checked "not refused" would still pass.
    """
    monkeypatch.setenv("MSE_GPU_MIN_TETS", str(10 ** 9))
    app, _store, db, _launcher = make_app(tmp_path)
    resp = upload(app.test_client())
    assert resp.status_code == 302, resp.data
    job = next(iter(db.jobs.values()))
    assert job["kind"] == DB.CAVITY
    assert job["pcs"] is None


def test_the_job_page_renders_a_pc_summary_and_not_the_cavity_header(
        tmp_path, monkeypatch):
    """A pc job's page must not print "(none) modes" over a band diagram.

    The two summaries are mutually exclusive in the template, and this asserts
    BOTH halves: the pc summary is present AND the cavity line is absent. A test
    that only checked for the pc text would pass on a page that printed both.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    # A REAL uuid: the job route validates the id before looking it up, so a
    # readable literal like "job-pc" 404s before the template is reached -- and
    # a 404 body would make the "expressions absent" half of this test pass for
    # the wrong reason.
    job_id = str(uuid.uuid4())
    db.create_job(job_id, "square lattice", "mm", 1.0, 16,
                  target=DB.CLUSTER, kind=DB.PC,
                  pcs={"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5,
                       "n_bg": 1.0, "n_bands": 4, "divisions": 16,
                       "polarisation": "TM"})
    html = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)
    assert 'id="pc-summary"' in html
    assert "modes requested" not in html, (
        "the cavity header rendered for a pc job; two summaries on one page "
        "means the reader cannot tell which the numbers belong to"
    )
    # The band panel exists in the DOM (hidden until the status poll says the
    # job is done) and the 3-D viewer's controls do not.
    assert 'id="bands-card"' in html
    assert "TM polarisation" in html


def test_a_finished_pc_page_renders_when_the_store_holds_no_job_files(tmp_path):
    """A finished photonics page must be the page, not a 404.

    ``job_page`` reads the achieved mesh edge out of the solver's
    ``modes.json`` -- a CAVITY job's manifest. A pc job uploads no mesh and
    writes no result files, so no manifest exists for it, and
    ``_done_manifest`` answers a missing one with 404. That made every finished
    photonics job's own page a "not found" the moment it finished -- the one
    page the feature exists to show. It was found in a browser against a real
    job, not by this suite: the pc page test above builds a QUEUED row, so the
    ``status == DONE`` branch never ran and 591 green tests said nothing.

    The store is empty on purpose and the emptiness is asserted, because that
    IS the state a real pc job is in. A test that laid down a manifest first
    would be testing a job that cannot exist.
    """
    app, store, db, _launcher = make_app(tmp_path)
    job_id = str(uuid.uuid4())
    db.create_job(job_id, "square lattice", "mm", 1.0, 16,
                  target=DB.CLUSTER, kind=DB.PC,
                  pcs={"lattice": "square", "r_over_a": 0.2, "n_rod": 3.5,
                       "n_bg": 1.0, "n_bands": 4, "divisions": 16,
                       "polarisation": "TM"})
    # A finished pc job's whole result is the diagram on the row -- the worker
    # writes it through ``finish_jetson_with_bands``, into ``pcs``, not into a
    # file. So mark it done the way the worker does and put the bands there.
    db.jobs[job_id].update(status=DB.DONE, stage=DB.DONE,
                           finished_at="2026-10-01T15:00:00+00:00")
    db.jobs[job_id]["pcs"]["bands"] = {
        "labels": ["G", "X", "M"],
        "k_points": [[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]],
        "bands": [[0.0, 0.44], [0.25, 0.421], [0.281, 0.5]],
    }
    assert store.keys_under(job_id) == [], (
        "the premise is an empty job directory: a pc job writes no result "
        "files, which is exactly why reading modes.json from the page 404s"
    )

    response = app.test_client().get(f"/jobs/{job_id}")
    assert response.status_code == 200, (
        f"a finished photonics page answered {response.status_code}: the page "
        "is asking a meshed-only question of a job that has no mesh"
    )
    # An HTTP 200 is not evidence that an element is present, so parse the body
    # for the two the band card is made of.
    html = response.get_data(as_text=True)
    assert 'id="bands-card"' in html
    assert 'id="bands-svg"' in html


def test_the_job_page_for_a_cavity_job_has_no_band_panel(tmp_path):
    """The other direction: a cavity job must not grow the pc branch.

    Asserted as an absence of the pc-only markers, which is a weaker statement
    than the pc test's mutual exclusion and is stated as such -- the cavity page
    has always carried its own summary and this is a regression guard on it.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    job_id = str(uuid.uuid4())
    db.create_job(job_id, "box.obj", "mm", 1.0, 10, target=DB.CLUSTER)
    html = app.test_client().get(f"/jobs/{job_id}").get_data(as_text=True)
    assert 'id="pc-summary"' not in html
    assert "modes requested" in html


def test_the_job_list_says_which_kind_each_row_is(tmp_path):
    """A pc row and a cavity row are told apart in the list, without opening them.

    Both kinds land in the same table with no uploaded filename. If the list
    did not name the kind, two rows could read identically and one of them
    would be a band diagram -- the reader would have to open every row to
    learn which. The assertion is per-row: a page that said "crystal"
    somewhere would pass a substring check while labelling the wrong row.
    """
    app, _store, db, _launcher = make_app(tmp_path)
    pc_id, cav_id = str(uuid.uuid4()), str(uuid.uuid4())
    db.create_job(pc_id, "square lattice, r/a=0.2", "mm", 1.0, 16,
                  target=DB.CLUSTER, kind=DB.PC, pcs={"lattice": "square"})
    db.create_job(cav_id, "box.obj", "mm", 1.0, 10, target=DB.CLUSTER)
    td_id = str(uuid.uuid4())
    db.create_job(td_id, "scene.obj (time domain)", "um", 1.0, 10,
                  target=DB.CLUSTER, kind=DB.TD)

    html = app.test_client().get("/").get_data(as_text=True)
    # Split the table into rows so each is checked against its OWN label. Each
    # row ends at its </tr>: the last one would otherwise run on through the
    # rest of the page, script comments included.
    rows = {r.split('href="/jobs/')[1].split('"')[0]: r.split("</tr>")[0]
            for r in html.split("<tr>") if 'href="/jobs/' in r}
    assert pc_id in rows and cav_id in rows and td_id in rows, \
        "a row is missing from the list"
    assert "crystal" in rows[pc_id], rows[pc_id]
    assert "crystal" not in rows[cav_id], rows[cav_id]
    assert "object" in rows[cav_id], rows[cav_id]
    assert "time-domain" in rows[td_id] and "object" not in rows[td_id], rows[td_id]


# ---------------------------------------------------------------------------
# td results: status keys and the files/ route
# ---------------------------------------------------------------------------
_TDS = {"boundary": "open", "source_kind": "point", "profile": "pulse",
        "frequency_hz": 4.0e14, "bandwidth": 0.5, "td_unit": "um"}


def _done_td_job(db, units="nm", scale=1.0):
    job_id = str(uuid.uuid4())
    db.create_job(job_id, "box.obj", units, 1.0, 0, target=DB.JETSON,
                  kind=DB.TD, tds=dict(_TDS), scale=scale)
    db.jobs[job_id].update(status=DB.DONE, stage=DB.DONE,
                           finished_at="2026-10-02T09:00:00+00:00")
    return job_id


def test_a_done_td_status_points_at_its_own_results_not_modes(tmp_path):
    """A finished td job's status names the spectrum, the trace and the object
    outline, and none of the cavity keys: ``modes: []`` would read as "found no
    modes", and the manifest/surface URLs name files a td job never writes.

    Mutation that must turn this red: delete the ``DB.TD`` branch in job_status.
    """
    app, _store, db, _l = make_app(tmp_path)
    job_id = _done_td_job(db)

    body = app.test_client().get(f"/jobs/{job_id}/status").get_json()

    assert body["kind"] == DB.TD
    assert body["spectrum_url"] == f"/jobs/{job_id}/files/spectrum.json"
    assert body["trace_url"] == f"/jobs/{job_id}/files/trace.bin"
    assert body["outline_url"] == f"/jobs/{job_id}/outline.glb"
    for key in ("modes", "manifest_url", "surface_url", "bands"):
        assert key not in body, f"a td status must not carry {key!r}"


#: The td movie's object outline. A box that is neither cubic nor centred on
#: the origin, in its own units: the two ways a frame bug hides. Its absolute
#: corner is (5, 3.5, 2) mm, so an outline translated to the mesh's own origin
#: (as the cavity's surface is built) or to the grid box's centre lands
#: somewhere else, and unequal extents tell an axis mix-up from a unit one.
_OUTLINE_BOX_CENTRE_MM = (7.0, 5.0, 3.0)
_OUTLINE_BOX_EXTENTS_MM = (4.0, 3.0, 2.0)


def _outline_box_bytes():
    box = trimesh.creation.box(extents=list(_OUTLINE_BOX_EXTENTS_MM))
    box.apply_translation(_OUTLINE_BOX_CENTRE_MM)
    return box.export(file_type="obj").encode()


def test_the_outline_route_puts_the_object_where_the_field_is(tmp_path):
    """The outline is the upload in the manifest's own frame: absolute metres,
    times the job's unit scale and free scale, plus nothing.

    The transform has one term because that is what the worker's own algebra
    gives -- ``fdtd_layout``'s nodes are absolute metres measured from the
    parts' bounding-box corner, and ``td_solve.py`` stores
    ``origin_m + node``, so ``frames.positions_m`` is the upload's coordinates
    times ``unit_scale(units) x scale`` with no offset. This test pins that:
    ``mm`` and ``scale=2`` move the box by 2e-3, and the box's own offset is
    preserved rather than replaced.

    Mutation that must turn this red: drop the ``length_scale`` multiply in
    web/outline.py (1000x too big), use the mesh's own origin as the frame (the
    corner goes to 0), centre it on the grid box, or feed ``scale`` alone
    without the unit.
    """
    app, store, db, _l = make_app(tmp_path)
    job_id = _done_td_job(db, units="mm", scale=2.0)
    store.write(STORE.job_key(job_id, "input.obj"), _outline_box_bytes())

    res = app.test_client().get(f"/jobs/{job_id}/outline.glb")

    assert res.status_code == 200
    assert res.mimetype == "model/gltf-binary"
    scene = trimesh.load(io.BytesIO(res.data), file_type="glb")
    factor = 1e-3 * 2.0
    want = np.array([[5.0, 3.5, 2.0], [9.0, 6.5, 4.0]]) * factor
    got = np.asarray(scene.bounds, dtype=float)
    # glTF stores float32, so a coordinate is exact only to about 1e-7
    # relative. 1e-6 leaves that room while staying 1000x finer than the
    # smallest bug this can catch (a missed unit is a factor of 1000).
    assert got == pytest.approx(want, rel=1e-6), got.tolist()


def _two_part_obj_bytes():
    """Two disjoint boxes as an .obj, in groups of their own names."""
    plate = trimesh.creation.box(extents=[4.0, 3.0, 0.5])
    plate.apply_translation([2.0, 1.5, 0.25])
    pin = trimesh.creation.box(extents=[1.0, 1.0, 3.0])
    pin.apply_translation([3.5, 1.0, 2.0])
    return (trimesh.Scene({"plate": plate, "pin": pin})
            .export(file_type="obj").encode())


def test_the_outline_draws_every_part_of_a_multi_part_upload(tmp_path):
    """Both parts come back, not just one.

    The cavity view kept only the last mesh of a multi-part surface, so every
    other part drew with no outline at all. A td upload may be several parts by
    design (the owner's film is offset plates), so the outline must carry them
    all -- and each where the upload put it, not stacked at one place.

    Mutation that must turn this red: keep only the largest part (2 -> 1).

    SCOPE, because it is easy to over-read: this fixture's group names are
    distinct, and ``mse_parts.parse_obj_groups`` refuses a repeated group name
    outright, so a name COLLISION never reaches ``_scene_geometries`` through
    this route. It is guarded anyway (the scene dict is keyed by that name, and
    the parser's refusal is not what this function's contract says), and the
    guard has its own direct test below rather than a mutation claimed here.
    """
    app, store, db, _l = make_app(tmp_path)
    job_id = _done_td_job(db, units="mm", scale=1.0)
    store.write(STORE.job_key(job_id, "input.obj"), _two_part_obj_bytes())

    res = app.test_client().get(f"/jobs/{job_id}/outline.glb")

    assert res.status_code == 200
    scene = trimesh.load(io.BytesIO(res.data), file_type="glb")
    assert len(scene.geometry) == 2, sorted(scene.geometry)
    # Each part is on its own side of the gap between them: a part stacked at
    # the other's place would still count two geometries but show one shape.
    tops = sorted(float(g.bounds[1][2]) for g in scene.geometry.values())
    assert tops[0] == pytest.approx(0.5e-3, rel=1e-6)   # the plate's top
    assert tops[1] == pytest.approx(3.5e-3, rel=1e-6)   # the pin's top


def test_two_parts_of_the_same_name_both_survive_into_the_scene():
    """A repeated part name does not lose a part.

    Driven directly, because the route above cannot reach this: the parser
    refuses a file that names one group twice. But ``_scene_geometries`` keys a
    DICT by the part name, so the moment anything upstream relaxes that refusal
    -- or a caller passes names from anywhere else -- the second part would
    land on the first and draw as one shape, silently. The contract is checked
    where it lives: two parts in, two geometries out, and the second's key is
    still recognisably tied to the name it came from.

    Mutation that must turn this red: drop the ``if key in out`` uniqueness in
    web/outline.py.
    """
    plate = trimesh.creation.box(extents=[4.0, 3.0, 0.5])
    other = trimesh.creation.box(extents=[1.0, 1.0, 3.0])
    geometry = OUTLINE._scene_geometries(["plate", "plate"], [plate, other])
    assert len(geometry) == 2, sorted(geometry)
    assert list(geometry)[0] == "plate", sorted(geometry)
    # The two mesh objects are the ones handed in, not one of them twice.
    assert {id(g) for g in geometry.values()} == {id(plate), id(other)}


def test_the_outline_route_answers_404_rather_than_500_when_it_cannot_draw(tmp_path):
    """Three ways to have no outline -- no stored upload, an upload that is not
    a mesh, and a job that has not finished -- are all an answer, not a fault.
    The page names the reason instead of drawing nothing.

    The last line is the control this test needs: the same route with a valid
    upload stored answers 200, so the three 404s are the checks above failing
    and not a route that never answers anything.
    """
    app, store, db, _l = make_app(tmp_path)
    client = app.test_client()

    no_upload = _done_td_job(db)
    assert client.get(f"/jobs/{no_upload}/outline.glb").status_code == 404

    junk = _done_td_job(db)
    store.write(STORE.job_key(junk, "input.obj"), b"not an obj at all\n")
    assert client.get(f"/jobs/{junk}/outline.glb").status_code == 404

    running = str(uuid.uuid4())
    db.create_job(running, "box.obj", "mm", 1.0, 0, kind=DB.TD, tds=dict(_TDS))
    store.write(STORE.job_key(running, "input.obj"), _outline_box_bytes())
    assert client.get(f"/jobs/{running}/outline.glb").status_code == 404

    good = _done_td_job(db, units="mm", scale=2.0)
    store.write(STORE.job_key(good, "input.obj"), _outline_box_bytes())
    assert client.get(f"/jobs/{good}/outline.glb").status_code == 200


def test_the_files_route_serves_every_td_result(tmp_path):
    """Each td file comes back byte for byte through files/, with JSON as JSON.

    The frame names include a three-digit one: a 120-frame movie's frames 100+
    have to come back, not 404 (they were dropped while the job reported done).

    Mutation that must turn this red: drop the td alternation from
    _safe_result_name (every name then 404s), or narrow its frame alternation
    back to two digits (frame_100.bin 404s).
    """
    app, store, db, _l = make_app(tmp_path)
    job_id = _done_td_job(db)
    files = {"trace.bin": b"\x00\x00\x80?" * 3,
             "spectrum.json": b'{"version": 1}',
             "frame_01.bin": b"\x01\x02\x03\x04",
             "frame_60.bin": b"\x05\x06\x07\x08",
             "frame_100.bin": b"\x09\x0a\x0b\x0c",
             "frame_120.bin": b"\x0d\x0e\x0f\x10"}
    for name, data in files.items():
        store.write(STORE.job_key(job_id, name), data)
    client = app.test_client()

    for name, data in files.items():
        response = client.get(f"/jobs/{job_id}/files/{name}")
        assert response.status_code == 200, name
        assert response.data == data, name
    assert client.get(f"/jobs/{job_id}/files/spectrum.json"
                      ).mimetype == "application/json"


def test_td_results_404_until_done_missing_or_misnamed(tmp_path):
    """The 404 rules: a running job's files, a file not written, and names the
    solver never writes (wrong digit count, wrong extension, traversal)."""
    app, store, db, _l = make_app(tmp_path)
    job_id = _done_td_job(db)
    store.write(STORE.job_key(job_id, "frame_01.bin"), b"\x00" * 4)
    client = app.test_client()

    assert client.get(f"/jobs/{job_id}/files/frame_02.bin").status_code == 404
    for bad in ("frame_1.bin", "frame_0001.bin", "frame_01.vtu",
                "trace.json", "spectrum.bin", "../frame_01.bin"):
        assert client.get(f"/jobs/{job_id}/files/{bad}").status_code == 404, bad

    db.jobs[job_id].update(status=DB.RUNNING, stage=DB.RUNNING)
    assert client.get(f"/jobs/{job_id}/files/frame_01.bin").status_code == 404


def test_the_app_serves_every_name_the_td_solve_writes(tmp_path):
    """The read side agrees with the writer: the names come from td_solve's own
    constants, not retyped, at the largest frame number it can write."""
    import importlib
    worker = str(Path(__file__).resolve().parents[1] / "worker")
    sys.path.insert(0, worker)
    try:
        td = importlib.import_module("td_solve")
        fdtd_max = importlib.import_module("fdtd").MAX_MOVIE_FRAMES
    finally:
        sys.path.remove(worker)
    for name in (td.TRACE_NAME, td.SPECTRUM_NAME, td.FRAME_NAME.format(1),
                 td.FRAME_NAME.format(fdtd_max)):
        assert APP._safe_result_name(name), name
